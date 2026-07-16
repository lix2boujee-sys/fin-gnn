"""MonoT5 evidence reranker.

Paper Table II: MonoT5 reranks a fixed BGE-M3 top-50 candidate pool using
a T5 model fine-tuned for relevance scoring.  The model scores each
(query, passage) pair with a "true"/"false" token probability and the
final score is ``logit(true) - logit(false)``.

Model
-----
Default: ``castorini/monot5-base-msmarco`` (public on HuggingFace).
Also supports local paths (e.g. ``cache/models/monot5-base-msmarco``).

Score caching
-------------
Because MonoT5 inference over 5703 × 50 pairs is expensive, every
query-candidate score is persisted to a score cache directory.  The cache
is keyed by model, candidate pool, corpus, and prompt template so it
cannot be accidentally reused for a different configuration.

Reference
---------
Nogueira et al., "Document Ranking with a Pretrained Sequence-to-Sequence
Model", Findings of EMNLP 2020.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from feg_rag.data.chunker import Chunk


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

MONO_T5_PROMPT_TEMPLATE = "Query: {query} Document: {passage} Relevant:"
MONO_T5_TRUE_TOKEN = "true"
MONO_T5_FALSE_TOKEN = "false"
SCORE_DEFINITION = "logit_true_minus_false"
CACHE_META_FILENAME = "cache_meta.json"
SCORES_PICKLE_FILENAME = "scores.pkl"


# ═════════════════════════════════════════════════════════════════════════════
# Score cache helpers
# ═════════════════════════════════════════════════════════════════════════════

def _compute_file_hash(path: Path) -> str:
    """SHA-256 hex digest of a file (first 16 chars)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _build_cache_meta(
    *,
    method: str,
    model_name_or_path: str,
    candidate_pool_name: str,
    candidate_results_jsonl: str,
    corpus_cache: str,
    top_n: int,
    max_length: int,
    prompt_template: str,
    score_definition: str,
) -> Dict:
    """Build the metadata dict stored alongside cached scores."""
    meta: Dict = {
        "method": method,
        "model_name_or_path": model_name_or_path,
        "candidate_pool_name": candidate_pool_name,
        "candidate_results_jsonl": candidate_results_jsonl,
        "corpus_cache": corpus_cache,
        "top_n": top_n,
        "max_length": max_length,
        "prompt_template": prompt_template,
        "score_definition": score_definition,
        "created_at": None,  # filled on first write
    }
    # Optionally add file hashes if paths exist
    candidate_path = Path(candidate_results_jsonl)
    corpus_path = Path(corpus_cache)
    if candidate_path.is_file():
        meta["candidate_results_jsonl_hash"] = _compute_file_hash(candidate_path)
    if corpus_path.is_file():
        meta["corpus_cache_hash"] = _compute_file_hash(corpus_path)
    return meta


def _meta_matches(cached: Dict, expected: Dict) -> bool:
    """Return True if *cached* metadata matches *expected*.

    Only compares keys present in *expected* so that newer fields added
    later don't break older caches unnecessarily.  Hashes are compared
    only when both sides have them.
    """
    for key, expected_val in expected.items():
        if key in ("created_at",):
            continue
        cached_val = cached.get(key)
        if key.endswith("_hash") and (cached_val is None or expected_val is None):
            continue  # skip hash check if either side lacks it
        if cached_val != expected_val:
            return False
    return True


def _load_score_cache(cache_dir: Path, expected_meta: Dict) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Load scores from *cache_dir* if meta matches.

    Returns:
        (scores, is_valid) where *scores* is ``{question_id: {chunk_id: score}}``
        and *is_valid* indicates whether the meta matched.
    """
    meta_path = cache_dir / CACHE_META_FILENAME
    scores_path = cache_dir / SCORES_PICKLE_FILENAME
    if not meta_path.exists() or not scores_path.exists():
        return {}, False
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            cached_meta = json.load(fh)
    except Exception:
        return {}, False
    if not _meta_matches(cached_meta, expected_meta):
        return {}, False
    try:
        with scores_path.open("rb") as fh:
            scores = pickle.load(fh)
    except Exception:
        return {}, False
    if not isinstance(scores, dict):
        return {}, False
    return scores, True


def _save_score_cache(cache_dir: Path, meta: Dict, scores: Dict[str, Dict[str, float]]) -> None:
    """Atomically write score cache to *cache_dir*."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["created_at"] = datetime.now(timezone.utc).isoformat()

    meta_tmp = cache_dir / f"{CACHE_META_FILENAME}.tmp"
    scores_tmp = cache_dir / f"{SCORES_PICKLE_FILENAME}.tmp"

    with meta_tmp.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    meta_tmp.replace(cache_dir / CACHE_META_FILENAME)

    with scores_tmp.open("wb") as fh:
        pickle.dump(scores, fh, protocol=pickle.HIGHEST_PROTOCOL)
    scores_tmp.replace(cache_dir / SCORES_PICKLE_FILENAME)


# ═════════════════════════════════════════════════════════════════════════════
# MonoT5 Reranker
# ═════════════════════════════════════════════════════════════════════════════

class MonoT5Reranker:
    """Rerank candidate chunks with MonoT5 pointwise relevance scoring.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model id or local path.  Default:
        ``castorini/monot5-base-msmarco``.
    batch_size:
        Batch size for T5 inference.
    max_length:
        Maximum token length for the T5 encoder/decoder.
    device:
        Torch device string (``"cuda"``, ``"cpu"``, etc.).
    use_fp16:
        Whether to load the model in half precision.
    """

    def __init__(
        self,
        model_name_or_path: str = "castorini/monot5-base-msmarco",
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        use_fp16: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or "cpu"
        self.use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None
        self._true_id: Optional[int] = None
        self._false_id: Optional[int] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        passages: List[str],
    ) -> List[float]:
        """Compute MonoT5 relevance scores for a batch of passages.

        Each passage is scored independently against *query* using the
        ``"Query: {q} Document: {d} Relevant:"`` prompt format.

        Returns a list of floats (higher = more relevant).
        """
        if not passages:
            return []
        self._ensure_loaded()
        texts = [
            MONO_T5_PROMPT_TEMPLATE.format(query=query, passage=p)
            for p in passages
        ]
        return self._score_batch(texts)

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Tuple[Chunk, float]],
        top_k: int = 50,
    ) -> List[Tuple[Chunk, float]]:
        """Score and rerank candidate chunks.

        Args:
            query: The question text.
            candidate_chunks: List of (Chunk, retrieval_score).
            top_k: Number of top results to return.

        Returns:
            Reranked list of (Chunk, monot5_score).
        """
        if not candidate_chunks:
            return []
        passages = [chunk.text for chunk, _ in candidate_chunks]
        scores = self.score(query, passages)
        scored = [
            (chunk, float(s))
            for (chunk, _), s in zip(candidate_chunks, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the T5 model and tokenizer."""
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MonoT5 requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from exc

        print(f"  [MonoT5] Loading model: {self.model_name_or_path}")
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True,
        )
        load_kwargs = {}
        if self.use_fp16:
            load_kwargs["torch_dtype"] = torch.float16
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path, **load_kwargs,
        )
        self._model.to(self.device)
        self._model.eval()

        # Resolve true / false token ids
        true_token = MONO_T5_TRUE_TOKEN
        false_token = MONO_T5_FALSE_TOKEN
        self._true_id = self._resolve_token_id(true_token, "true")
        self._false_id = self._resolve_token_id(false_token, "false")

        print(f"  [MonoT5] Model loaded in {time.time() - t0:.1f}s "
              f"(device={self.device}, fp16={self.use_fp16})")
        print(f"  [MonoT5] true token id={self._true_id}, "
              f"false token id={self._false_id}")
        self._loaded = True

    def _resolve_token_id(self, token_str: str, label: str) -> int:
        """Resolve a string to a single token id, raising if ambiguous."""
        ids = self._tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"MonoT5: expected '{label}'='{token_str}' to be a single token, "
                f"got {len(ids)} tokens (ids={ids}). "
                f"Check tokenizer of '{self.model_name_or_path}'."
            )
        return ids[0]

    # ------------------------------------------------------------------
    # Internal: scoring
    # ------------------------------------------------------------------

    def _score_batch(self, texts: List[str]) -> List[float]:
        """Run T5 forward on *texts* and return relevance scores."""
        import torch

        scores: List[float] = []
        model = self._model
        tokenizer = self._tokenizer
        true_id = self._true_id
        false_id = self._false_id

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                # We need decoder_input_ids to force the first token prediction.
                # Use the model's internal decoder_start_token_id.
                decoder_start = torch.full(
                    (enc["input_ids"].shape[0], 1),
                    model.config.decoder_start_token_id or 0,
                    dtype=torch.long,
                    device=self.device,
                )
                outputs = model(
                    **enc,
                    decoder_input_ids=decoder_start,
                )

            # logits: (batch, 1, vocab_size) — the first generated token
            logits = outputs.logits[:, 0, :]  # (batch, vocab_size)
            true_logit = logits[:, true_id]
            false_logit = logits[:, false_id]
            batch_scores = (true_logit - false_logit).cpu().tolist()
            scores.extend(batch_scores)

        return scores


# ═════════════════════════════════════════════════════════════════════════════
# High-level runner (called from experiment scripts)
# ═════════════════════════════════════════════════════════════════════════════

def run_mono_t5_reranking(
    samples: List[Dict],
    chunk_by_id: Dict[str, Chunk],
    candidate_pool: Dict[str, List[str]],  # question -> list of chunk_ids
    gold_map: Dict[str, List[str]],
    *,
    model_name_or_path: str = "castorini/monot5-base-msmarco",
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    use_fp16: bool = False,
    top_n: int = 50,
    output_k: int = 10,
    score_cache_dir: Optional[Path] = None,
    rebuild_score_cache: bool = False,
    resume_rerank: bool = False,
    candidate_pool_name: str = "BGE-M3-Dense",
    candidate_results_jsonl: str = "",
    corpus_cache: str = "",
    allow_fallback: bool = False,
    checkpoint_every: int = 100,
    partial_output_path: Optional[Path] = None,
) -> List[Dict]:
    """Run MonoT5 reranking over a fixed candidate pool with score caching.

    This is the entry point called by the Table 1 / Table II experiment
    script.  It:

    1. Loads or builds a per-(query, candidate) score cache.
    2. Scores missing pairs with MonoT5.
    3. Reranks each query's candidates by MonoT5 score.
    4. Writes partial JSONL results every *checkpoint_every* queries.
    5. Returns the full result list suitable for metric computation.

    Parameters
    ----------
    samples:
        List of dataset samples (each must have ``"id"`` and ``"question"``).
    chunk_by_id:
        Mapping from chunk_id to Chunk.
    candidate_pool:
        Mapping from question text to pre-computed candidate chunk_ids
        (typically BGE-M3 top-50).
    gold_map:
        Mapping from sample id to list of gold evidence chunk_ids.
    model_name_or_path:
        Model identifier for MonoT5.
    score_cache_dir:
        Directory for persistent query-candidate score cache.
    rebuild_score_cache:
        If True, ignore existing cache and recompute all scores.
    resume_rerank:
        If True, load existing partial scores and only compute missing ones.
    allow_fallback:
        If True, fall back to original candidate order on error.
    checkpoint_every:
        Write partial JSONL every N queries (0 to disable).
    partial_output_path:
        Path for partial JSONL output.

    Returns:
        List of result dicts with ``question_id``, ``question``,
        ``gold_evidence_ids``, ``retrieved_chunk_ids``, ``method``.
    """
    # ---- Resolve candidate pool into ordered lists ----
    # candidate_pool keys are questions; build lookup by question
    candidate_by_qid: Dict[str, List[str]] = {}
    qid_to_question: Dict[str, str] = {}
    for s in samples:
        qid = s["id"]
        question = s["question"]
        qid_to_question[qid] = question
        # Look up by question text in the candidate pool
        cand_ids = candidate_pool.get(question, [])[:top_n]
        candidate_by_qid[qid] = cand_ids

    # ---- Score cache ----
    scores: Dict[str, Dict[str, float]] = {}
    cache_meta = _build_cache_meta(
        method="mono_t5",
        model_name_or_path=model_name_or_path,
        candidate_pool_name=candidate_pool_name,
        candidate_results_jsonl=candidate_results_jsonl,
        corpus_cache=corpus_cache,
        top_n=top_n,
        max_length=max_length,
        prompt_template=MONO_T5_PROMPT_TEMPLATE,
        score_definition=SCORE_DEFINITION,
    )

    cache_status = "disabled"
    if score_cache_dir is not None:
        score_cache_dir = Path(score_cache_dir)
        if rebuild_score_cache:
            print(f"  [MonoT5] Rebuilding score cache: {score_cache_dir}")
            cache_status = "rebuilt"
        else:
            loaded_scores, is_valid = _load_score_cache(score_cache_dir, cache_meta)
            if is_valid:
                scores = loaded_scores
                cache_status = "loaded"
                print(f"  [MonoT5] Score cache loaded: {score_cache_dir} "
                      f"({_count_cached_scores(scores)} scores)")
            elif resume_rerank:
                # Load whatever partial scores exist (even if meta doesn't perfectly match)
                partial = _load_scores_any(score_cache_dir)
                if partial:
                    scores = partial
                    cache_status = "partial-resume"
                    print(f"  [MonoT5] Partial score cache resumed: {score_cache_dir} "
                          f"({_count_cached_scores(scores)} scores)")
                else:
                    cache_status = "new"
                    print(f"  [MonoT5] No reusable cache; building new: {score_cache_dir}")
            else:
                cache_status = "new"
                if not is_valid and score_cache_dir.exists():
                    print(f"  [MonoT5] Cache meta mismatch; building new: {score_cache_dir}")
                else:
                    print(f"  [MonoT5] No cache found; building new: {score_cache_dir}")

    print(f"  [MonoT5] Score cache status: {cache_status}")
    if scores:
        print(f"  [MonoT5] Cached scores: {_count_cached_scores(scores)}")

    # ---- Determine missing scores ----
    missing_pairs: List[Tuple[str, str, str]] = []  # (qid, question, chunk_id)
    for s in samples:
        qid = s["id"]
        question = s["question"]
        cand_ids = candidate_by_qid.get(qid, [])
        q_scores = scores.get(qid, {})
        for cid in cand_ids:
            if cid not in q_scores:
                missing_pairs.append((qid, question, cid))

    print(f"  [MonoT5] Missing scores to compute: {len(missing_pairs)}")

    # ---- Compute missing scores with MonoT5 ----
    if missing_pairs:
        reranker = MonoT5Reranker(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            use_fp16=use_fp16,
        )

        t_start = time.time()
        done = 0
        for start in range(0, len(missing_pairs), batch_size):
            pair_batch = missing_pairs[start:start + batch_size]
            valid_pairs: List[Tuple[str, str, str, str]] = []
            for qid, question, cid in pair_batch:
                chunk = chunk_by_id.get(cid)
                if chunk is None:
                    continue
                valid_pairs.append((qid, question, cid, chunk.text))

            if not valid_pairs:
                done += len(pair_batch)
                continue

            try:
                # MonoT5 conditions on the query, so batch only candidates
                # from the same question. If a mixed batch appears, split it.
                questions = {p[1] for p in valid_pairs}
                if len(questions) == 1:
                    question = valid_pairs[0][1]
                    batch_scores = reranker.score(
                        question,
                        [passage for _, _, _, passage in valid_pairs],
                    )
                    for (qid, _, cid, _), score in zip(valid_pairs, batch_scores):
                        scores.setdefault(qid, {})[cid] = float(score)
                else:
                    for qid, question, cid, passage in valid_pairs:
                        batch_scores = reranker.score(question, [passage])
                        scores.setdefault(qid, {})[cid] = float(batch_scores[0])
            except Exception:
                if not allow_fallback:
                    raise
                for qid, _, cid, _ in valid_pairs:
                    scores.setdefault(qid, {})[cid] = 0.0

            # Progress
            done += len(pair_batch)
            if checkpoint_every > 0 and (done % checkpoint_every == 0 or done == len(missing_pairs)):
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                eta = (len(missing_pairs) - done) / max(rate, 1e-6)
                print(f"  [MonoT5] scored {done}/{len(missing_pairs)} pairs "
                      f"({done / max(len(missing_pairs), 1):.1%}) "
                      f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

                # Save score cache incrementally
                if score_cache_dir is not None:
                    _save_score_cache(score_cache_dir, cache_meta, scores)

        print(f"  [MonoT5] Scoring done in {time.time() - t_start:.1f}s")

    # Final save of score cache
    if score_cache_dir is not None and missing_pairs:
        _save_score_cache(score_cache_dir, cache_meta, scores)
        print(f"  [MonoT5] Score cache saved: {score_cache_dir}")

    # ---- Rerank each query ----
    results: List[Dict] = []
    for s in samples:
        qid = s["id"]
        question = s["question"]
        gold_ids = gold_map.get(qid, [])
        cand_ids = candidate_by_qid.get(qid, [])
        q_scores = scores.get(qid, {})

        # Score each candidate
        scored: List[Tuple[str, float]] = []
        for cid in cand_ids:
            s_val = q_scores.get(cid, 0.0)
            scored.append((cid, s_val))

        # Sort descending by MonoT5 score
        scored.sort(key=lambda x: x[1], reverse=True)
        retrieved_ids = [cid for cid, _ in scored[:output_k]]

        r = {
            "question_id": qid,
            "question": question,
            "gold_answer": s.get("answer", ""),
            "gold_evidence_ids": gold_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "method": "mono_t5",
        }
        results.append(r)

    # ---- Write partial JSONL ----
    if partial_output_path is not None:
        _write_jsonl(partial_output_path, results)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _count_cached_scores(scores: Dict[str, Dict[str, float]]) -> int:
    """Return total number of cached query-candidate pairs."""
    return sum(len(v) for v in scores.values())


def _load_scores_any(cache_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load scores pickle even if meta doesn't match (for resume)."""
    scores_path = cache_dir / SCORES_PICKLE_FILENAME
    if not scores_path.exists():
        return {}
    try:
        with scores_path.open("rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    """Write result rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

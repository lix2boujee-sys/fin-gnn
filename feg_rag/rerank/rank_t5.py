"""RankT5 evidence reranker.

Paper Table II: RankT5 reranks a fixed BGE-M3 top-50 candidate pool using
a T5-based listwise or pairwise reranker.  Like MonoT5, scores are cached
per (query, chunk) pair.

.. important::
    RankT5 does **not** ship with a default public checkpoint.  You
    **must** supply ``--rank_t5_model`` pointing to a local path or a
    HuggingFace model id that you have access to.  If the model cannot
    be downloaded or the path does not exist, the code fails with a clear
    error — it will never silently fall back to another model.

Architecture
------------
RankT5 uses the same pointwise scoring approach as MonoT5 internally
(encode ``"Query: ... Document: ..."`` → score), but is kept as a
separate class so that a listwise-calibrated checkpoint can be plugged in
without changing the experiment infrastructure.

Reference
---------
Zhuang et al., "RankT5: Fine-Tuning T5 for Text Ranking with Ranking
Losses", SIGIR 2023.
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

RANK_T5_PROMPT_TEMPLATE = "Query: {query} Document: {passage} Relevant:"
RANK_T5_TRUE_TOKEN = "true"
RANK_T5_FALSE_TOKEN = "false"
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
            chunk = fh.read(1 << 20)
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
        "created_at": None,
    }
    candidate_path = Path(candidate_results_jsonl)
    corpus_path = Path(corpus_cache)
    if candidate_path.is_file():
        meta["candidate_results_jsonl_hash"] = _compute_file_hash(candidate_path)
    if corpus_path.is_file():
        meta["corpus_cache_hash"] = _compute_file_hash(corpus_path)
    return meta


def _meta_matches(cached: Dict, expected: Dict) -> bool:
    for key, expected_val in expected.items():
        if key in ("created_at",):
            continue
        cached_val = cached.get(key)
        if key.endswith("_hash") and (cached_val is None or expected_val is None):
            continue
        if cached_val != expected_val:
            return False
    return True


def _load_score_cache(cache_dir: Path, expected_meta: Dict) -> Tuple[Dict[str, Dict[str, float]], bool]:
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


def _load_scores_any(cache_dir: Path) -> Dict[str, Dict[str, float]]:
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


def _count_cached_scores(scores: Dict[str, Dict[str, float]]) -> int:
    return sum(len(v) for v in scores.values())


# ═════════════════════════════════════════════════════════════════════════════
# RankT5 Reranker
# ═════════════════════════════════════════════════════════════════════════════

class RankT5Reranker:
    """Rerank candidate chunks with a T5-based ranker (RankT5 family).

    .. note::
        There is **no default** model.  You must supply *model_name_or_path*.
        If the model cannot be loaded (missing path, gated repo, network
        error), the constructor raises immediately — no silent fallback.

    Parameters
    ----------
    model_name_or_path:
        **Required.**  HuggingFace model id or local directory.
    batch_size:
        Batch size for T5 inference.
    max_length:
        Maximum token length for the T5 encoder/decoder.
    device:
        Torch device string.
    use_fp16:
        Whether to load the model in half precision.
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        use_fp16: bool = False,
    ):
        if not model_name_or_path:
            raise ValueError(
                "RankT5 requires --rank_t5_model (model_name_or_path). "
                "There is no default public RankT5 checkpoint. "
                "Please provide a local path or a HuggingFace model id."
            )
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
        """Compute RankT5 relevance scores for a batch of passages.

        Uses the pointwise ``"Query: ... Document: ... Relevant:"`` format.
        """
        if not passages:
            return []
        self._ensure_loaded()
        texts = [
            RANK_T5_PROMPT_TEMPLATE.format(query=query, passage=p)
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
            Reranked list of (Chunk, rank_t5_score).
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
                "RankT5 requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from exc

        # Verify the path/model exists before attempting download
        model_path = Path(self.model_name_or_path)
        if model_path.exists():
            print(f"  [RankT5] Loading model from local path: {self.model_name_or_path}")
        else:
            print(f"  [RankT5] Loading model from HuggingFace: {self.model_name_or_path}")
            print(f"  [RankT5] NOTE: This model may be gated. Ensure you have access.")

        t0 = time.time()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, use_fast=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"RankT5: Failed to load tokenizer from '{self.model_name_or_path}'. "
                f"Error: {exc}"
            ) from exc

        try:
            load_kwargs = {}
            if self.use_fp16:
                load_kwargs["torch_dtype"] = torch.float16
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path, **load_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"RankT5: Failed to load model from '{self.model_name_or_path}'. "
                f"If this is a gated repo, ensure you have accepted the terms "
                f"and run `huggingface-cli login`. Error: {exc}"
            ) from exc

        self._model.to(self.device)
        self._model.eval()

        # Resolve true / false token ids
        self._true_id = self._resolve_token_id(RANK_T5_TRUE_TOKEN, "true")
        self._false_id = self._resolve_token_id(RANK_T5_FALSE_TOKEN, "false")

        print(f"  [RankT5] Model loaded in {time.time() - t0:.1f}s "
              f"(device={self.device}, fp16={self.use_fp16})")
        print(f"  [RankT5] true token id={self._true_id}, "
              f"false token id={self._false_id}")
        self._loaded = True

    def _resolve_token_id(self, token_str: str, label: str) -> int:
        ids = self._tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"RankT5: expected '{label}'='{token_str}' to be a single token, "
                f"got {len(ids)} tokens (ids={ids}). "
                f"Check tokenizer of '{self.model_name_or_path}'."
            )
        return ids[0]

    # ------------------------------------------------------------------
    # Internal: scoring
    # ------------------------------------------------------------------

    def _score_batch(self, texts: List[str]) -> List[float]:
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

            logits = outputs.logits[:, 0, :]
            true_logit = logits[:, true_id]
            false_logit = logits[:, false_id]
            batch_scores = (true_logit - false_logit).cpu().tolist()
            scores.extend(batch_scores)

        return scores


# ═════════════════════════════════════════════════════════════════════════════
# High-level runner
# ═════════════════════════════════════════════════════════════════════════════

def run_rank_t5_reranking(
    samples: List[Dict],
    chunk_by_id: Dict[str, Chunk],
    candidate_pool: Dict[str, List[str]],
    gold_map: Dict[str, List[str]],
    *,
    model_name_or_path: str,
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
    """Run RankT5 reranking over a fixed candidate pool with score caching.

    See :func:`feg_rag.rerank.mono_t5.run_mono_t5_reranking` for the full
    parameter description — the two runners share the same signature and
    logic, differing only in the model class and metadata tags.
    """
    if not model_name_or_path:
        raise ValueError(
            "RankT5 requires --rank_t5_model. "
            "There is no default public RankT5 checkpoint."
        )

    # ---- Resolve candidate pool ----
    candidate_by_qid: Dict[str, List[str]] = {}
    for s in samples:
        qid = s["id"]
        question = s["question"]
        cand_ids = candidate_pool.get(question, [])[:top_n]
        candidate_by_qid[qid] = cand_ids

    # ---- Score cache ----
    scores: Dict[str, Dict[str, float]] = {}
    cache_meta = _build_cache_meta(
        method="rank_t5",
        model_name_or_path=model_name_or_path,
        candidate_pool_name=candidate_pool_name,
        candidate_results_jsonl=candidate_results_jsonl,
        corpus_cache=corpus_cache,
        top_n=top_n,
        max_length=max_length,
        prompt_template=RANK_T5_PROMPT_TEMPLATE,
        score_definition=SCORE_DEFINITION,
    )

    cache_status = "disabled"
    if score_cache_dir is not None:
        score_cache_dir = Path(score_cache_dir)
        if rebuild_score_cache:
            print(f"  [RankT5] Rebuilding score cache: {score_cache_dir}")
            cache_status = "rebuilt"
        else:
            loaded_scores, is_valid = _load_score_cache(score_cache_dir, cache_meta)
            if is_valid:
                scores = loaded_scores
                cache_status = "loaded"
                print(f"  [RankT5] Score cache loaded: {score_cache_dir} "
                      f"({_count_cached_scores(scores)} scores)")
            elif resume_rerank:
                partial = _load_scores_any(score_cache_dir)
                if partial:
                    scores = partial
                    cache_status = "partial-resume"
                    print(f"  [RankT5] Partial score cache resumed: {score_cache_dir} "
                          f"({_count_cached_scores(scores)} scores)")
                else:
                    cache_status = "new"
                    print(f"  [RankT5] No reusable cache; building new: {score_cache_dir}")
            else:
                cache_status = "new"
                if not is_valid and score_cache_dir.exists():
                    print(f"  [RankT5] Cache meta mismatch; building new: {score_cache_dir}")
                else:
                    print(f"  [RankT5] No cache found; building new: {score_cache_dir}")

    print(f"  [RankT5] Score cache status: {cache_status}")
    if scores:
        print(f"  [RankT5] Cached scores: {_count_cached_scores(scores)}")

    # ---- Missing scores ----
    missing_pairs: List[Tuple[str, str, str]] = []
    for s in samples:
        qid = s["id"]
        question = s["question"]
        cand_ids = candidate_by_qid.get(qid, [])
        q_scores = scores.get(qid, {})
        for cid in cand_ids:
            if cid not in q_scores:
                missing_pairs.append((qid, question, cid))

    print(f"  [RankT5] Missing scores to compute: {len(missing_pairs)}")

    # ---- Compute missing scores ----
    if missing_pairs:
        reranker = RankT5Reranker(
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

            done += len(pair_batch)
            if checkpoint_every > 0 and (done % checkpoint_every == 0 or done == len(missing_pairs)):
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                eta = (len(missing_pairs) - done) / max(rate, 1e-6)
                print(f"  [RankT5] scored {done}/{len(missing_pairs)} pairs "
                      f"({done / max(len(missing_pairs), 1):.1%}) "
                      f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

                if score_cache_dir is not None:
                    _save_score_cache(score_cache_dir, cache_meta, scores)

        print(f"  [RankT5] Scoring done in {time.time() - t_start:.1f}s")

    if score_cache_dir is not None and missing_pairs:
        _save_score_cache(score_cache_dir, cache_meta, scores)
        print(f"  [RankT5] Score cache saved: {score_cache_dir}")

    # ---- Rerank ----
    results: List[Dict] = []
    for s in samples:
        qid = s["id"]
        question = s["question"]
        gold_ids = gold_map.get(qid, [])
        cand_ids = candidate_by_qid.get(qid, [])
        q_scores = scores.get(qid, {})

        scored = sorted(
            [(cid, q_scores.get(cid, 0.0)) for cid in cand_ids],
            key=lambda x: x[1],
            reverse=True,
        )
        retrieved_ids = [cid for cid, _ in scored[:output_k]]

        r = {
            "question_id": qid,
            "question": question,
            "gold_answer": s.get("answer", ""),
            "gold_evidence_ids": gold_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "method": "rank_t5",
        }
        results.append(r)

    if partial_output_path is not None:
        partial_output_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_output_path.open("w", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return results

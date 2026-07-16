"""ListT5 listwise evidence reranker.

This runner reranks a fixed BGE-M3 candidate pool with ListT5. It keeps a
persistent decision cache because ListT5 compares candidate groups instead of
assigning independent pointwise scores.

References:
  - https://github.com/soyoung97/ListT5
  - https://huggingface.co/Soyoung97/ListT5-base
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from feg_rag.data.chunker import Chunk


LIST_T5_PROMPT_TEMPLATE = "Query: {query}, Index: {index}, Context: {passage}"
CACHE_META_FILENAME = "cache_meta.json"
DECISIONS_PICKLE_FILENAME = "decisions.pkl"


def _compute_file_hash(path: Path) -> str:
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
    model_name_or_path: str,
    candidate_pool_name: str,
    candidate_results_jsonl: str,
    corpus_cache: str,
    top_n: int,
    max_length: int,
    listwise_k: int,
    out_k: int,
) -> Dict:
    meta: Dict = {
        "method": "list_t5",
        "model_name_or_path": model_name_or_path,
        "candidate_pool_name": candidate_pool_name,
        "candidate_results_jsonl": candidate_results_jsonl,
        "corpus_cache": corpus_cache,
        "top_n": top_n,
        "max_length": max_length,
        "listwise_k": listwise_k,
        "out_k": out_k,
        "prompt_template": LIST_T5_PROMPT_TEMPLATE,
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
        if key == "created_at":
            continue
        cached_val = cached.get(key)
        if key.endswith("_hash") and (cached_val is None or expected_val is None):
            continue
        if cached_val != expected_val:
            return False
    return True


def _load_decision_cache(cache_dir: Path, expected_meta: Dict) -> Tuple[Dict, bool]:
    meta_path = cache_dir / CACHE_META_FILENAME
    decisions_path = cache_dir / DECISIONS_PICKLE_FILENAME
    if not meta_path.exists() or not decisions_path.exists():
        return {}, False
    try:
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not _meta_matches(cached_meta, expected_meta):
            return {}, False
        with decisions_path.open("rb") as fh:
            decisions = pickle.load(fh)
        return decisions if isinstance(decisions, dict) else {}, True
    except Exception:
        return {}, False


def _load_decisions_any(cache_dir: Path) -> Dict:
    decisions_path = cache_dir / DECISIONS_PICKLE_FILENAME
    if not decisions_path.exists():
        return {}
    try:
        with decisions_path.open("rb") as fh:
            data = pickle.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_decision_cache(cache_dir: Path, meta: Dict, decisions: Dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta_tmp = cache_dir / f"{CACHE_META_FILENAME}.tmp"
    decisions_tmp = cache_dir / f"{DECISIONS_PICKLE_FILENAME}.tmp"
    meta_tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_tmp.replace(cache_dir / CACHE_META_FILENAME)
    with decisions_tmp.open("wb") as fh:
        pickle.dump(decisions, fh, protocol=pickle.HIGHEST_PROTOCOL)
    decisions_tmp.replace(cache_dir / DECISIONS_PICKLE_FILENAME)


class FiDT5Mixin:
    """Small local FiD wrapper matching the official ListT5 inference shape."""

    @staticmethod
    def build_class():
        import torch
        import transformers
        from torch import nn

        class CheckpointWrapper(torch.nn.Module):
            """Official ListT5 encoder block wrapper.

            The released checkpoint stores encoder block weights under
            ``encoder.encoder.block.N.module.*``. Keeping this wrapper in the
            architecture lets those weights load without being treated as
            missing/unexpected.
            """

            def __init__(self, module, use_checkpoint: bool = False):
                super().__init__()
                self.module = module
                self.use_checkpoint = use_checkpoint

            def forward(self, *args, **kwargs):
                if self.use_checkpoint and self.training:
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}

                    def custom_forward(*inputs):
                        output = self.module(*inputs, **kwargs)
                        empty = torch.tensor(
                            [],
                            dtype=torch.float,
                            device=output[0].device,
                            requires_grad=True,
                        )
                        return tuple(x if x is not None else empty for x in output)

                    output = torch.utils.checkpoint.checkpoint(
                        custom_forward,
                        *args,
                    )
                    return tuple(x if x.size() != 0 else None for x in output)

                return self.module(*args, **kwargs)

        def apply_checkpoint_wrapper(t5stack, use_checkpoint: bool = False):
            t5stack.block = nn.ModuleList([
                CheckpointWrapper(mod, use_checkpoint)
                for mod in t5stack.block
            ])

        class EncoderWrapper(torch.nn.Module):
            def __init__(self, encoder, use_checkpoint: bool = False):
                super().__init__()
                self.encoder = encoder
                self.main_input_name = encoder.main_input_name
                self.embed_tokens = encoder.embed_tokens
                self.n_passages = 1
                apply_checkpoint_wrapper(self.encoder, use_checkpoint)

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                bsz, total_length = input_ids.shape
                passage_length = total_length // self.n_passages
                input_ids = input_ids.view(bsz * self.n_passages, passage_length)
                attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                last_hidden_state = outputs[0].view(bsz, self.n_passages * passage_length, -1)
                if kwargs.get("return_dict"):
                    outputs.last_hidden_state = last_hidden_state
                    return outputs
                return (last_hidden_state,) + outputs[1:]

        class FiDT5(transformers.T5ForConditionalGeneration):
            def __init__(self, config):
                super().__init__(config)
                self.wrap_encoder()

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                if input_ids is not None and input_ids.dim() == 3:
                    self.encoder.n_passages = input_ids.size(1)
                    input_ids = input_ids.view(input_ids.size(0), -1)
                if attention_mask is not None and attention_mask.dim() == 3:
                    attention_mask = attention_mask.view(attention_mask.size(0), -1)
                return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

            def generate(self, input_ids, attention_mask, max_length, **kwargs):
                self.encoder.n_passages = input_ids.size(1)
                return super().generate(
                    input_ids=input_ids.reshape(input_ids.size(0), -1),
                    attention_mask=attention_mask.reshape(attention_mask.size(0), -1),
                    max_length=max_length,
                    **kwargs,
                )

            def wrap_encoder(self, use_checkpoint: bool = False):
                self.encoder = EncoderWrapper(
                    self.encoder,
                    use_checkpoint=use_checkpoint,
                )

            def unwrap_encoder(self):
                self.encoder = self.encoder.encoder
                self.encoder.block = nn.ModuleList([
                    mod.module for mod in self.encoder.block
                ])

        return FiDT5


class ListT5Reranker:
    def __init__(
        self,
        model_name_or_path: str = "Soyoung97/ListT5-base",
        batch_size: int = 8,
        max_length: int = 128,
        listwise_k: int = 5,
        out_k: int = 2,
        device: Optional[str] = None,
        use_fp16: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.listwise_k = listwise_k
        self.out_k = out_k
        self.device = device or "cpu"
        self.use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import T5Tokenizer

        FiDT5 = FiDT5Mixin.build_class()
        print(f"  [ListT5] Loading model: {self.model_name_or_path}")
        t0 = time.time()
        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path, legacy=False)
        load_kwargs = {}
        if self.use_fp16:
            load_kwargs["torch_dtype"] = torch.float16
        self._model = FiDT5.from_pretrained(self.model_name_or_path, **load_kwargs)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        print(f"  [ListT5] Model loaded in {time.time() - t0:.1f}s "
              f"(device={self.device}, fp16={self.use_fp16})")

    def _make_texts(self, query: str, passages: List[str]) -> List[str]:
        return [
            LIST_T5_PROMPT_TEMPLATE.format(query=query, index=i + 1, passage=passage)
            for i, passage in enumerate(passages)
        ]

    def choose(self, query: str, passages: List[str], k: int = 1) -> List[int]:
        self._ensure_loaded()
        if not passages:
            return []
        if len(passages) == 1:
            return [0]

        # ListT5 expects exactly listwise_k passages. Repeat existing passages
        # as dummies for short groups, then drop duplicate choices.
        group = list(passages)
        while len(group) < self.listwise_k:
            group.append(passages[len(group) % len(passages)])
        group = group[:self.listwise_k]

        texts = self._make_texts(query, group)
        raw = self._tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_tensors = {
            "input_ids": raw["input_ids"].unsqueeze(0).to(self.device),
            "attention_mask": raw["attention_mask"].unsqueeze(0).to(self.device),
        }
        with __import__("torch").no_grad():
            output = self._model.generate(
                **input_tensors,
                max_length=self.listwise_k + 2,
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = self._tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
        picked: List[int] = []
        for token in decoded.split():
            try:
                idx = int(token) - 1
            except ValueError:
                continue
            if 0 <= idx < len(passages) and idx not in picked:
                picked.append(idx)
        if not picked:
            picked = [0]
        return picked[-k:]

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 10, decisions: Optional[Dict] = None, qid: str = "") -> List[Chunk]:
        remaining = list(range(len(chunks)))
        chosen: List[int] = []
        decisions = decisions if decisions is not None else {}

        while remaining and len(chosen) < top_k:
            winners: List[int] = []
            for start in range(0, len(remaining), self.listwise_k):
                group_idx = remaining[start:start + self.listwise_k]
                cache_key = f"{qid}:{','.join(map(str, group_idx))}"
                cached = decisions.get(cache_key)
                if cached is None:
                    passages = [chunks[i].text for i in group_idx]
                    local_choices = self.choose(query, passages, k=min(self.out_k, len(group_idx)))
                    cached = [group_idx[i] for i in local_choices if i < len(group_idx)]
                    decisions[cache_key] = cached
                winners.extend(cached)

            # If several group winners remain, run another tournament round.
            while len(winners) > 1:
                new_winners: List[int] = []
                for start in range(0, len(winners), self.listwise_k):
                    group_idx = winners[start:start + self.listwise_k]
                    cache_key = f"{qid}:{','.join(map(str, group_idx))}"
                    cached = decisions.get(cache_key)
                    if cached is None:
                        passages = [chunks[i].text for i in group_idx]
                        local_choices = self.choose(query, passages, k=1)
                        cached = [group_idx[i] for i in local_choices if i < len(group_idx)]
                        decisions[cache_key] = cached
                    new_winners.extend(cached[:1])
                winners = new_winners

            best = winners[0] if winners else remaining[0]
            if best not in chosen:
                chosen.append(best)
            remaining = [i for i in remaining if i != best]

        full_order = chosen + [i for i in range(len(chunks)) if i not in chosen]
        return [chunks[i] for i in full_order[:top_k]]


def run_list_t5_reranking(
    samples: List[Dict],
    chunk_by_id: Dict[str, Chunk],
    candidate_pool: Dict[str, List[str]],
    gold_map: Dict[str, List[str]],
    *,
    model_name_or_path: str = "Soyoung97/ListT5-base",
    batch_size: int = 8,
    max_length: int = 128,
    listwise_k: int = 5,
    out_k: int = 2,
    device: str = "cuda",
    use_fp16: bool = False,
    top_n: int = 50,
    output_k: int = 10,
    decision_cache_dir: Optional[Path] = None,
    rebuild_decision_cache: bool = False,
    resume_rerank: bool = False,
    candidate_pool_name: str = "BGE-M3-Dense",
    candidate_results_jsonl: str = "",
    corpus_cache: str = "",
    allow_fallback: bool = False,
    checkpoint_every: int = 100,
    partial_output_path: Optional[Path] = None,
) -> List[Dict]:
    candidate_by_qid = {
        s["id"]: candidate_pool.get(s["question"], [])[:top_n]
        for s in samples
    }

    cache_meta = _build_cache_meta(
        model_name_or_path=model_name_or_path,
        candidate_pool_name=candidate_pool_name,
        candidate_results_jsonl=candidate_results_jsonl,
        corpus_cache=corpus_cache,
        top_n=top_n,
        max_length=max_length,
        listwise_k=listwise_k,
        out_k=out_k,
    )

    decisions: Dict = {}
    cache_status = "disabled"
    if decision_cache_dir is not None:
        decision_cache_dir = Path(decision_cache_dir)
        if rebuild_decision_cache:
            cache_status = "rebuilt"
            print(f"  [ListT5] Rebuilding decision cache: {decision_cache_dir}")
        else:
            loaded, valid = _load_decision_cache(decision_cache_dir, cache_meta)
            if valid:
                decisions = loaded
                cache_status = "loaded"
            elif resume_rerank:
                decisions = _load_decisions_any(decision_cache_dir)
                cache_status = "partial-resume" if decisions else "new"
            else:
                cache_status = "new"
        print(f"  [ListT5] Decision cache status: {cache_status} ({len(decisions)} decisions)")

    reranker = ListT5Reranker(
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        max_length=max_length,
        listwise_k=listwise_k,
        out_k=out_k,
        device=device,
        use_fp16=use_fp16,
    )

    results: List[Dict] = []
    t0 = time.time()
    for idx, s in enumerate(samples, 1):
        qid = s["id"]
        question = s["question"]
        cand_ids = candidate_by_qid.get(qid, [])
        chunks = [chunk_by_id[cid] for cid in cand_ids if cid in chunk_by_id]
        try:
            reranked_chunks = reranker.rerank(
                question, chunks, top_k=output_k, decisions=decisions, qid=qid
            )
            retrieved_ids = [c.chunk_id for c in reranked_chunks]
        except Exception:
            if not allow_fallback:
                raise
            retrieved_ids = [c.chunk_id for c in chunks[:output_k]]

        results.append({
            "question_id": qid,
            "question": question,
            "gold_answer": s.get("answer", ""),
            "gold_evidence_ids": gold_map.get(qid, []),
            "retrieved_chunk_ids": retrieved_ids,
            "method": "list_t5",
        })

        if checkpoint_every > 0 and (idx % checkpoint_every == 0 or idx == len(samples)):
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-6)
            eta = (len(samples) - idx) / max(rate, 1e-6)
            print(f"  [ListT5] reranked {idx}/{len(samples)} "
                  f"({idx / max(len(samples), 1):.1%}) elapsed={elapsed:.1f}s eta={eta:.1f}s",
                  flush=True)
            if decision_cache_dir is not None:
                _save_decision_cache(decision_cache_dir, cache_meta, decisions)
            if partial_output_path is not None:
                _write_jsonl(partial_output_path, results)

    if decision_cache_dir is not None:
        _save_decision_cache(decision_cache_dir, cache_meta, decisions)
        print(f"  [ListT5] Decision cache saved: {decision_cache_dir}")
    if partial_output_path is not None:
        _write_jsonl(partial_output_path, results)
    return results


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

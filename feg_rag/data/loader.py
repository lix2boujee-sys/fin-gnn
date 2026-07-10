"""Dataset loaders for FinDER, TAT-QA, and FinQA.

Each loader returns a uniform list[dict] with keys:
    id, question, answer, evidence_texts, metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(name: str, data_dir: str | Path, **kwargs) -> List[Dict]:
    """Load a dataset by name.

    Args:
        name: One of ``finder``, ``tatqa``, ``finqa``.
        data_dir: Directory containing raw dataset files.
        **kwargs: Passed through to the specific loader.

    Returns:
        List of uniform samples.
    """
    data_dir = Path(data_dir)
    loaders = {
        "finder": load_finder,
        "tatqa": load_tatqa,
        "finqa": load_finqa,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders)}.")
    return loaders[name](data_dir, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# FinDER
# ═════════════════════════════════════════════════════════════════════════════

def load_finder(data_dir: Path, split: Optional[str] = None) -> List[Dict]:
    """Load FinDER from parquet files.

    FinDER columns: _id, text (question), answer, references (evidence), type,
    category, reasoning.

    Returns uniform samples with:
        id, question, answer, evidence_texts, metadata
    """
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    samples: List[Dict] = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            refs = row["references"]
            # Handle numpy arrays, lists, or scalar strings from parquet
            if hasattr(refs, "tolist"):
                refs = refs.tolist()
            if not isinstance(refs, list):
                refs = [str(refs)]
            else:
                refs = [str(r) for r in refs]
            samples.append(
                {
                    "id": row["_id"],
                    "question": row["text"],
                    "answer": str(row["answer"]),
                    "evidence_texts": refs,
                    "metadata": {
                        "source": "finder",
                        "category": row.get("category", ""),
                        "type": row.get("type", ""),
                        "reasoning": bool(row.get("reasoning", False)),
                    },
                }
            )
    return samples


# ═════════════════════════════════════════════════════════════════════════════
# TAT-QA  (stub — replace with real loader when dataset is available)
# ═════════════════════════════════════════════════════════════════════════════

def load_tatqa(data_dir: Path, split: str = "train") -> List[Dict]:
    """Load TAT-QA from its raw JSON files.

    Placeholder — download TAT-QA first, then implement the actual parsing.
    """
    # TAT-QA provides train/dev/test JSON files with keys:
    #   question, answer, table, paragraphs, ...
    # See https://nextplusplus.github.io/TAT-QA/
    json_path = data_dir / f"tatqa_dataset_{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"TAT-QA file not found: {json_path}. "
            "Download from https://nextplusplus.github.io/TAT-QA/"
        )

    with open(json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    samples: List[Dict] = []
    for item in raw:
        samples.append(
            {
                "id": item.get("uid", ""),
                "question": item["question"],
                "answer": str(item.get("answer", "")),
                "evidence_texts": _extract_tatqa_evidence(item),
                "metadata": {
                    "source": "tatqa",
                    "scale": item.get("scale", ""),
                },
            }
        )
    return samples


def _extract_tatqa_evidence(item: dict) -> List[str]:
    """Extract gold evidence texts from a TAT-QA item."""
    texts: List[str] = []
    for para in item.get("paragraphs", []):
        texts.append(para["text"])
    if "table" in item:
        texts.append(json.dumps(item["table"]))  # keep table structure
    return texts


# ═════════════════════════════════════════════════════════════════════════════
# FinQA  (stub — replace with real loader when dataset is available)
# ═════════════════════════════════════════════════════════════════════════════

def load_finqa(data_dir: Path, split: str = "train") -> List[Dict]:
    """Load FinQA from its raw JSON files.

    Placeholder — download FinQA first, then implement the actual parsing.
    """
    json_path = data_dir / f"finqa_{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"FinQA file not found: {json_path}. "
            "Download from https://finqasite.github.io/"
        )

    with open(json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    samples: List[Dict] = []
    for item in raw:
        samples.append(
            {
                "id": item.get("id", ""),
                "question": item["qa"]["question"],
                "answer": str(item["qa"].get("answer", "")),
                "evidence_texts": item.get("pre_text", [])
                + item.get("post_text", [])
                + [json.dumps(t) for t in item.get("table", [])],
                "metadata": {
                    "source": "finqa",
                },
            }
        )
    return samples

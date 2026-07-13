"""LLM prompt templates for reranking and generation experiments.

Provides structured prompts for:
  - LLM Reranker (Experiment 2): Rank candidate evidence by relevance.
  - Answer Generator (Experiment 3): Generate financial answers from evidence.
"""

from __future__ import annotations

from typing import List

from feg_rag.data.chunker import Chunk


# ═════════════════════════════════════════════════════════════════════════════
# LLM Reranker prompt (Experiment 2)
# ═════════════════════════════════════════════════════════════════════════════

_RERANKER_SYSTEM = (
    "You are a financial evidence ranker. Your job is to rank candidate evidence "
    "passages by how likely they contain information that directly supports "
    "answering the given financial question.\n\n"
    "Rules:\n"
    "1. Rank the candidates from most relevant (rank 1) to least relevant.\n"
    "2. Consider financial metric names, years, companies, and filing types "
    "mentioned in the question.\n"
    "3. Do NOT generate an answer to the question — only rank the evidence.\n"
    "4. Output valid JSON only, with no extra text outside the JSON object."
)

_RERANKER_USER = """## Question
{question}

## Candidate Evidence Passages
{candidates}

## Task
Rank the candidates from most relevant (rank 1) to least relevant.
Output a JSON object with this exact structure:
```json
{{
  "ranked_candidate_ids": ["candidate_id_1", "candidate_id_2", ...],
  "rationale": "brief explanation of your ranking logic"
}}
```

Include ALL candidate IDs in the ranked list. Do not skip any."""


def build_reranker_messages(
    question: str,
    candidates: List[Chunk],
    max_candidate_text_chars: int = 500,
) -> list:
    """Build messages for the LLM reranker.

    Args:
        question: The financial question.
        candidates: Candidate evidence chunks with IDs.
        max_candidate_text_chars: Truncate each candidate's text to this length.

    Returns:
        List of {"role": ..., "content": ...} dicts for the chat API.
    """
    candidate_lines: List[str] = []
    for i, c in enumerate(candidates):
        text = c.text[:max_candidate_text_chars]
        if len(c.text) > max_candidate_text_chars:
            text += "..."
        meta_parts = []
        if c.company:
            meta_parts.append(f"company={c.company}")
        if c.filing_year:
            meta_parts.append(f"year={c.filing_year}")
        if c.filing_type:
            meta_parts.append(f"filing={c.filing_type}")
        meta = ", ".join(meta_parts)
        candidate_lines.append(
            f"[{c.chunk_id}] ({meta})\n{text}\n"
        )

    candidates_text = "\n".join(candidate_lines)
    user_content = _RERANKER_USER.format(
        question=question, candidates=candidates_text
    )

    return [
        {"role": "system", "content": _RERANKER_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Answer Generator prompt (Experiment 3)
# ═════════════════════════════════════════════════════════════════════════════

_GENERATOR_SYSTEM = (
    "You are a financial analyst answering questions based ONLY on the "
    "provided evidence passages. Follow these rules strictly:\n\n"
    "1. Use ONLY the information in the provided evidence passages.\n"
    "2. Do NOT use outside knowledge or training data.\n"
    "3. Answer concisely and directly.\n"
    "4. Preserve numbers and units exactly as they appear in the evidence.\n"
    "5. If the evidence is insufficient to answer the question, respond with "
    "exactly: INSUFFICIENT_EVIDENCE\n"
    "6. Cite the evidence IDs you used to construct your answer.\n"
    "7. Output valid JSON only, with no extra text outside the JSON object."
)

_GENERATOR_USER = """## Question
{question}

## Evidence Passages
{evidence}

## Task
Answer the question using ONLY the evidence passages above.
Output a JSON object with this exact structure:
```json
{{
  "answer": "your concise answer here",
  "evidence_ids_used": ["id1", "id2"],
  "confidence": "high|medium|low"
}}
```"""


def build_generator_messages(
    question: str,
    evidence_chunks: List[Chunk],
    max_evidence_text_chars: int = 800,
    include_metadata: bool = True,
) -> list:
    """Build messages for the answer generator.

    Args:
        question: The financial question.
        evidence_chunks: Top-k evidence chunks (typically k=5).
        max_evidence_text_chars: Truncate each chunk text to this length.
        include_metadata: Whether to include company/year/filing metadata.

    Returns:
        List of {"role": ..., "content": ...} dicts.
    """
    evidence_lines: List[str] = []
    for i, c in enumerate(evidence_chunks):
        text = c.text[:max_evidence_text_chars]
        if len(c.text) > max_evidence_text_chars:
            text += "..."

        if include_metadata:
            meta_parts = []
            if c.company:
                meta_parts.append(f"company={c.company}")
            if c.filing_year:
                meta_parts.append(f"year={c.filing_year}")
            if c.filing_type:
                meta_parts.append(f"filing={c.filing_type}")
            if c.section:
                meta_parts.append(f"section={c.section}")
            meta = ", ".join(meta_parts)
            evidence_lines.append(f"[{c.chunk_id}] ({meta})\n{text}\n")
        else:
            evidence_lines.append(f"[{c.chunk_id}]\n{text}\n")

    evidence_text = "\n".join(evidence_lines)
    user_content = _GENERATOR_USER.format(
        question=question, evidence=evidence_text
    )

    return [
        {"role": "system", "content": _GENERATOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

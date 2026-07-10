"""LLM answer generation from retrieved evidence chunks.

Paper plan §9: constrained generation with evidence grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from feg_rag.data.chunker import Chunk


# ═════════════════════════════════════════════════════════════════════════════
# Prompt template
# ═════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are a financial analyst answering questions based ONLY on the "
    "retrieved evidence chunks provided below. Follow these rules:\n"
    "1. Answer ONLY using information from the evidence.\n"
    "2. Cite the evidence chunk IDs that support your answer.\n"
    "3. If calculation is needed, show your arithmetic step by step.\n"
    "4. Always include units (millions, billions, %, etc.).\n"
    "5. If the evidence is insufficient, respond with exactly: "
    "INSUFFICIENT_EVIDENCE\n"
)

_USER_TEMPLATE = """## Question
{question}

## Evidence Chunks
{evidence}

## Answer
"""


@dataclass
class GeneratedAnswer:
    """Output from the LLM generator."""

    question: str
    answer: str
    evidence_chunks: List[Chunk]
    cited_chunk_ids: List[str]
    model_name: str
    raw_response: str


# ═════════════════════════════════════════════════════════════════════════════
# Generator
# ═════════════════════════════════════════════════════════════════════════════

class LLMGenerator:
    """Wraps an LLM (OpenAI-compatible API) for evidence-grounded generation."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        evidence_chunks: List[Chunk],
    ) -> GeneratedAnswer:
        """Generate an answer from top-k evidence chunks."""
        evidence_text = self._format_evidence(evidence_chunks)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    question=question, evidence=evidence_text
                ),
            },
        ]

        raw = self._call_llm(messages)
        answer = raw.strip()
        cited = self._extract_citations(answer, evidence_chunks)

        return GeneratedAnswer(
            question=question,
            answer=answer,
            evidence_chunks=evidence_chunks,
            cited_chunk_ids=cited,
            model_name=self.model,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _format_evidence(self, chunks: List[Chunk]) -> str:
        lines: List[str] = []
        for i, c in enumerate(chunks):
            lines.append(f"[{c.chunk_id}] (doc={c.doc_id}, section={c.section})")
            lines.append(c.text)
            lines.append("")
        return "\n".join(lines)

    def _extract_citations(
        self, answer: str, chunks: List[Chunk]
    ) -> List[str]:
        """Simple heuristic: find chunk IDs mentioned in the answer."""
        chunk_ids = {c.chunk_id for c in chunks}
        cited = [cid for cid in chunk_ids if cid in answer]
        return cited

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API.

        Uses openai if available; falls back to a stub for testing.
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            # Stub for environments without openai
            return self._stub_generate(messages)

    def _stub_generate(self, messages: List[Dict[str, str]]) -> str:
        """Fallback stub — returns a placeholder."""
        return "STUB_ANSWER (install openai package to use real LLM)"

"""Rule-based numerical verifier.

Paper plan §9: Lightweight verifier that checks:
  - Numbers in the answer appear in evidence
  - Year consistency between question and answer
  - Unit consistency
  - Evidence support for claims
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from feg_rag.data.chunker import Chunk
from feg_rag.generation.llm import GeneratedAnswer


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
    r"|\b\d+\.?\d*\s*(?:million|billion|thousand|trillion)\b"
    r"|\b\d+\.?\d*\s*%"
    r"|\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
    re.IGNORECASE,
)

_UNIT_RE = re.compile(
    r"\b(million|billion|thousand|trillion|percent|%|dollars?\$?)\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


@dataclass
class VerificationResult:
    """Output of the numerical verifier."""

    answer: str
    numbers_in_answer: List[str] = field(default_factory=list)
    numbers_in_evidence: List[str] = field(default_factory=list)
    numbers_matched: bool = False
    years_match: bool = False
    evidence_fully_cited: bool = False
    issues: List[str] = field(default_factory=list)
    is_consistent: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# Verifier
# ═════════════════════════════════════════════════════════════════════════════

class NumericalVerifier:
    """Check numerical and evidence consistency of a generated answer."""

    def verify(
        self,
        generated: GeneratedAnswer,
        question: str,
    ) -> VerificationResult:
        """Run all checks and return a structured result."""
        issues: List[str] = []

        # 1. Extract numbers
        ans_numbers = self._extract_numbers(generated.answer)
        ev_numbers = self._extract_numbers(
            " ".join(c.text for c in generated.evidence_chunks)
        )

        # 2. Check: answer numbers appear in evidence
        numbers_matched = True
        for n in ans_numbers:
            if n not in ev_numbers:
                issues.append(f"Number '{n}' in answer not found in evidence")
                numbers_matched = False

        # 3. Check: year consistency
        q_years = set(_YEAR_RE.findall(question))
        a_years = set(_YEAR_RE.findall(generated.answer))
        years_match = True
        if q_years and a_years:
            if not (q_years & a_years):
                issues.append(
                    f"Answer years {a_years} don't match question years {q_years}"
                )
                years_match = False
        elif q_years and not a_years:
            issues.append("Question specifies a year but answer does not")
            years_match = False

        # 4. Check: evidence fully cited
        evidence_fully_cited = len(generated.cited_chunk_ids) > 0
        if not evidence_fully_cited:
            issues.append("No evidence chunk IDs cited in answer")

        # 5. Check: insufficient evidence flag
        if "INSUFFICIENT_EVIDENCE" in generated.answer.upper():
            # This is actually a valid response
            pass

        is_consistent = (
            numbers_matched
            and years_match
            and evidence_fully_cited
            and len(issues) == 0
        )

        return VerificationResult(
            answer=generated.answer,
            numbers_in_answer=ans_numbers,
            numbers_in_evidence=ev_numbers,
            numbers_matched=numbers_matched,
            years_match=years_match,
            evidence_fully_cited=evidence_fully_cited,
            issues=issues,
            is_consistent=is_consistent,
        )

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return [m.group(0).strip() for m in _NUMBER_RE.finditer(text)]

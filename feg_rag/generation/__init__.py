"""Answer generation, LLM client, prompts, and numerical verification."""

from feg_rag.generation.llm import LLMGenerator
from feg_rag.generation.verifier import NumericalVerifier

# Lazy imports for heavy / optional dependencies
_LAZY = {
    "OpenRouterClient": "feg_rag.generation.openrouter_client",
    "TokenUsage": "feg_rag.generation.openrouter_client",
    "ChatResponse": "feg_rag.generation.openrouter_client",
    "build_reranker_messages": "feg_rag.generation.llm_prompts",
    "build_generator_messages": "feg_rag.generation.llm_prompts",
    "parse_json_response": "feg_rag.generation.llm_response_parser",
    "parse_reranker_response": "feg_rag.generation.llm_response_parser",
    "parse_generator_response": "feg_rag.generation.llm_response_parser",
    "TokenCostTracker": "feg_rag.generation.token_cost",
    "CostSummary": "feg_rag.generation.token_cost",
    "LLMCache": "feg_rag.generation.llm_cache",
    "AnswerEvaluator": "feg_rag.generation.answer_evaluator",
    "AggregateEvalResult": "feg_rag.generation.answer_evaluator",
}


def __getattr__(name: str):
    try:
        return _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

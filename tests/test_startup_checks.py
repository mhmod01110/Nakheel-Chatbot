from nakheel.config import Settings
from nakheel.core.generation.llm_client import LLMClient
from nakheel.core.ingestion import embedder as embedder_module
from nakheel.core.ingestion.embedder import DenseEmbedder
from nakheel.core.retrieval import reranker as reranker_module
from nakheel.core.retrieval.reranker import RerankerService


def test_dense_embedder_startup_check_accepts_fallback_backend(monkeypatch):
    monkeypatch.setattr(embedder_module, "BGEM3FlagModel", None)
    embedder = DenseEmbedder(Settings())

    check = embedder.startup_check()

    assert check["ok"] is True
    assert check["detail"] == "using deterministic fallback embedding backend"


def test_reranker_startup_check_accepts_fallback_backend(monkeypatch):
    monkeypatch.setattr(reranker_module, "FlagReranker", None)
    reranker = RerankerService(Settings())

    check = reranker.startup_check()

    assert check["ok"] is True
    assert check["detail"] == "using heuristic fallback reranker"


def test_llm_startup_check_accepts_missing_api_key():
    client = LLMClient(Settings(OPENAI_API_KEY=None))

    check = client.startup_check()

    assert check["ok"] is True
    assert check["detail"] == "OPENAI_API_KEY is not configured; using fallback responses"

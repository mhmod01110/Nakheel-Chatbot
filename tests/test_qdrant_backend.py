from types import SimpleNamespace

from nakheel.db.qdrant import QdrantDatabase


def test_normalize_llama_index_result_prefers_node_metadata():
    result = SimpleNamespace(
        ids=["fallback-id"],
        nodes=[SimpleNamespace(node_id="node-1", metadata={"chunk_id": "chk-1", "section_title": "Intro"})],
        similarities=[0.87],
    )

    normalized = QdrantDatabase._normalize_llama_index_result(result)

    assert normalized[0].id == "chk-1"
    assert normalized[0].payload["chunk_id"] == "chk-1"
    assert normalized[0].payload["section_title"] == "Intro"
    assert normalized[0].score == 0.87


def test_normalize_llama_index_result_falls_back_to_ids():
    result = SimpleNamespace(ids=["chk-2"], nodes=[], similarities=[0.55])

    normalized = QdrantDatabase._normalize_llama_index_result(result)

    assert normalized[0].id == "chk-2"
    assert normalized[0].payload == {"chunk_id": "chk-2"}
    assert normalized[0].score == 0.55

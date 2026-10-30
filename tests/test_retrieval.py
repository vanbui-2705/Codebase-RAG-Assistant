"""
Unit tests cho Tầng Retrieval (Giai đoạn 1):
Kiểm thử HybridEmbedder (Dense + Sparse), QdrantStore (Named Vectors, In-Memory Upsert & Hybrid Search)
và CrossEncoderReranker (Top-K Chấm điểm lại).
"""
from src.domain.schemas import CodeNode, compute_code_node_id
from src.retrieval.embedder import HybridEmbedder
from src.retrieval.vector_store import QdrantStore
from src.retrieval.reranker import CrossEncoderReranker


def test_hybrid_embedder_dimensions():
    """Kiểm tra HybridEmbedder sinh Dense 1024-d và Sparse dict token:weight."""
    embedder = HybridEmbedder()
    texts = ["def process_payment(amount: float) -> bool:", "Payment timeout in stripe gateway"]

    dense_vecs = embedder.embed_dense(texts)
    assert len(dense_vecs) == 2
    assert len(dense_vecs[0]) == 1024
    assert len(dense_vecs[1]) == 1024

    sparse_vecs = embedder.embed_sparse(texts)
    assert len(sparse_vecs) == 2
    assert isinstance(sparse_vecs[0], dict)
    assert len(sparse_vecs[0]) > 0
    assert all(isinstance(k, int) and isinstance(v, float) for k, v in sparse_vecs[0].items())


def test_qdrant_store_in_memory_upsert_and_hybrid_search():
    """Kiểm tra QdrantStore tạo collection, upsert nodes và hybrid search trong bộ nhớ."""
    embedder = HybridEmbedder()
    store = QdrantStore(location=":memory:", embedder=embedder)

    code1 = "def authenticate_user(token: str) -> bool:\n    return verify_jwt(token)"
    code2 = "def process_refund(order_id: str) -> bool:\n    return refund_api(order_id)"

    node1 = CodeNode(
        node_id=compute_code_node_id("auth.py", "authenticate_user", code1),
        repo_name="demo_repo",
        file_path="src/auth.py",
        language="python",
        symbol_type="function",
        symbol_name="authenticate_user",
        start_line=1,
        end_line=2,
        imports=["import jwt"],
        raw_code=code1,
        context_header=f"# File: src/auth.py\n{code1}",
    )
    node2 = CodeNode(
        node_id=compute_code_node_id("payment.py", "process_refund", code2),
        repo_name="demo_repo",
        file_path="src/payment.py",
        language="python",
        symbol_type="function",
        symbol_name="process_refund",
        start_line=10,
        end_line=15,
        imports=["import stripe"],
        raw_code=code2,
        context_header=f"# File: src/payment.py\n{code2}",
    )

    store.upsert_nodes("code_chunks", [node1, node2])

    # Thực hiện Hybrid Search với từ khóa hoàn tiền
    results = store.hybrid_search("code_chunks", query="hàm xử lý hoàn tiền order_id", top_k=2)
    assert len(results) > 0
    assert any("refund" in r.content.lower() for r in results)

    # Kiểm tra tính năng xóa theo file_path (Incremental Indexing)
    store.delete_by_file_paths("code_chunks", ["src/payment.py"])
    after_delete = store.hybrid_search("code_chunks", query="refund", top_k=2)
    assert not any(r.file_path == "src/payment.py" for r in after_delete)


def test_cross_encoder_reranker():
    """Kiểm tra CrossEncoderReranker chấm điểm và lọc top_k kết quả."""
    reranker = CrossEncoderReranker()
    from src.domain.schemas import RetrievedContext

    contexts = [
        RetrievedContext(
            node_id="1",
            source_type="code",
            file_path="src/user.py",
            content="def get_user_profile(): return user",
            score=0.1,
            symbol_name="get_user_profile",
        ),
        RetrievedContext(
            node_id="2",
            source_type="code",
            file_path="src/payment.py",
            content="def process_payment_timeout(): raise TimeoutError('Payment gateway timeout')",
            score=0.2,
            symbol_name="process_payment_timeout",
        ),
        RetrievedContext(
            node_id="3",
            source_type="code",
            file_path="src/logger.py",
            content="def log_event(event): print(event)",
            score=0.05,
            symbol_name="log_event",
        ),
    ]

    reranked = reranker.rerank(query="xử lý lỗi payment gateway timeout", contexts=contexts, top_k=2)

    assert len(reranked) == 2
    # Kết quả liên quan nhất phải xếp đầu tiên
    assert reranked[0].symbol_name == "process_payment_timeout"
    assert reranked[0].score >= reranked[1].score

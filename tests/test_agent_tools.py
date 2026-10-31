"""
Unit tests cho src/agent/tools.py:
Kiểm thử toàn diện 5 công cụ tra cứu của Agent:
1. parse_stack_trace (Python, Java, Go, JS/TS, Edge cases)
2. search_codebase_tool (Hybrid Search + Reranker)
3. get_callers_and_callees_tool (AST Analysis cho Call Graph)
4. search_git_history_tool (Local Git & Vector store fallback)
5. search_runbooks_tool (Runbook & Docs retrieval)
"""
from unittest.mock import MagicMock
import pytest

from src.agent.tools import (
    AgentTools,
    parse_stack_trace,
    search_codebase_tool,
    get_callers_and_callees_tool,
    search_git_history_tool,
    search_runbooks_tool,
)
from src.domain.schemas import CodeNode, IncidentDocNode, RetrievedContext, compute_code_node_id
from src.retrieval.embedder import HybridEmbedder
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import QdrantStore


# =============================================================================
# TEST CÔNG CỤ 1: parse_stack_trace
# =============================================================================
def test_parse_stack_trace_python():
    """Kiểm thử bóc tách stack trace của Python."""
    py_log = """
Traceback (most recent call last):
  File "src/api/routes.py", line 45, in handle_payment
    return service.process_refund(order_id)
  File "src/payment/service.py", line 112, in process_refund
    raise TimeoutError("Stripe gateway timed out after 30s")
TimeoutError: Stripe gateway timed out after 30s
    """
    res = parse_stack_trace(py_log)

    assert res["exception_type"] == "TimeoutError"
    assert "Stripe gateway timed out" in res["exception_message"]
    assert len(res["frames"]) == 2

    # Frame cuối cùng
    last_frame = res["last_frame"]
    assert last_frame is not None
    assert last_frame["file_path"] == "src/payment/service.py"
    assert last_frame["line_number"] == 112
    assert last_frame["function_name"] == "process_refund"
    assert "TimeoutError" in last_frame["code_context"]

    # Các trường root
    assert res["file_path"] == "src/payment/service.py"
    assert res["line_number"] == 112
    assert res["function_name"] == "process_refund"
    assert "TimeoutError" in res["error_signature"]


def test_parse_stack_trace_java():
    """Kiểm thử bóc tách stack trace của Java."""
    java_log = """
Exception in thread "main" java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.service.PaymentService.processPayment(PaymentService.java:120)
    at com.example.controller.OrderController.handleOrder(OrderController.java:54)
    """
    res = parse_stack_trace(java_log)

    assert res["exception_type"] == "NullPointerException"
    assert "Cannot invoke method" in res["exception_message"]
    assert len(res["frames"]) == 2
    assert res["frames"][0]["file_path"] == "PaymentService.java"
    assert res["frames"][0]["line_number"] == 120
    assert res["frames"][0]["function_name"] == "PaymentService.processPayment"


def test_parse_stack_trace_go():
    """Kiểm thử bóc tách panic và stack trace của Go."""
    go_log = """
panic: runtime error: invalid memory address or nil pointer dereference
[signal SIGSEGV: code=0x1 addr=0x0 pc=0x45a2b1]

goroutine 1 [running]:
main.processOrder(0xc0000a4000)
\t/app/src/order/order.go:88 +0x39
main.main()
\t/app/src/main.go:24 +0x5a
    """
    res = parse_stack_trace(go_log)

    assert res["exception_type"] == "Panic"
    assert "invalid memory address" in res["exception_message"]
    assert len(res["frames"]) == 2
    assert res["frames"][0]["file_path"] == "/app/src/order/order.go"
    assert res["frames"][0]["line_number"] == 88


def test_parse_stack_trace_javascript():
    """Kiểm thử bóc tách stack trace của Node.js / JavaScript."""
    js_log = """
TypeError: Cannot read properties of undefined (reading 'amount')
    at processRefund (/app/src/refund.js:34:12)
    at handleRequest (/app/src/server.js:80:5)
    """
    res = parse_stack_trace(js_log)

    assert res["exception_type"] == "TypeError"
    assert "Cannot read properties" in res["exception_message"]
    assert len(res["frames"]) == 2
    assert res["frames"][0]["file_path"] == "/app/src/refund.js"
    assert res["frames"][0]["line_number"] == 34
    assert res["frames"][0]["function_name"] == "processRefund"


def test_parse_stack_trace_empty_or_invalid():
    """Kiểm thử stack trace rỗng hoặc không hợp lệ không làm crash hệ thống."""
    res_empty = parse_stack_trace("")
    assert res_empty["frames"] == []
    assert res_empty["last_frame"] is None

    res_random = parse_stack_trace("Đây là một đoạn log bình thường không có lỗi gì cả.")
    assert res_random["frames"] == []
    assert res_random["last_frame"] is None


# =============================================================================
# TEST CÔNG CỤ 2: search_codebase_tool
# =============================================================================
def test_search_codebase_tool_in_memory():
    """Kiểm thử search_codebase_tool kết hợp Qdrant in-memory và CrossEncoderReranker."""
    embedder = HybridEmbedder()
    store = QdrantStore(location=":memory:", embedder=embedder)
    reranker = CrossEncoderReranker()

    raw1 = "def authenticate_user(token: str) -> bool:\n    return verify_jwt(token)"
    raw2 = "def process_refund(order_id: str) -> bool:\n    return stripe_refund(order_id)"

    node1 = CodeNode(
        node_id=compute_code_node_id("auth.py", "authenticate_user", raw1),
        repo_name="test_repo",
        file_path="src/auth.py",
        language="python",
        symbol_type="function",
        symbol_name="authenticate_user",
        start_line=1,
        end_line=2,
        imports=["import jwt"],
        raw_code=raw1,
        context_header=f"# File: src/auth.py\n{raw1}",
    )
    node2 = CodeNode(
        node_id=compute_code_node_id("payment.py", "process_refund", raw2),
        repo_name="test_repo",
        file_path="src/payment.py",
        language="python",
        symbol_type="function",
        symbol_name="process_refund",
        start_line=10,
        end_line=20,
        imports=["import stripe"],
        raw_code=raw2,
        context_header=f"# File: src/payment.py\n{raw2}",
    )
    store.upsert_nodes("code_chunks", [node1, node2])

    tools = AgentTools(vector_store=store, reranker=reranker)

    # Tìm kiếm theo từ khóa nghiệp vụ
    results = tools.search_codebase_tool(query="hoàn tiền cho đơn hàng order_id", top_k=1)
    assert len(results) == 1
    assert results[0].symbol_name == "process_refund"
    assert "stripe_refund" in results[0].content

    # Tìm kiếm kèm bộ lọc file_path
    filtered = tools.search_codebase_tool(query="jwt", file_path="src/auth.py", top_k=2)
    assert len(filtered) == 1
    assert filtered[0].symbol_name == "authenticate_user"


# =============================================================================
# TEST CÔNG CỤ 3: get_callers_and_callees_tool
# =============================================================================
def test_get_callers_and_callees_tool():
    """Kiểm thử trích xuất call graph (callers & callees) từ AST."""
    embedder = HybridEmbedder()
    store = QdrantStore(location=":memory:", embedder=embedder)

    code_caller = """
def checkout(order_id: str, token: str):
    user = authenticate_user(token)
    success = process_refund(order_id)
    send_notification(user)
    return success
"""
    code_target = """
def process_refund(order_id: str):
    gateway = get_payment_gateway()
    result = gateway.charge_back(order_id)
    audit_log(order_id, result)
    return result
"""
    node_caller = CodeNode(
        node_id=compute_code_node_id("checkout.py", "checkout", code_caller),
        repo_name="test_repo",
        file_path="src/checkout.py",
        language="python",
        symbol_type="function",
        symbol_name="checkout",
        start_line=1,
        end_line=10,
        raw_code=code_caller,
        context_header=code_caller,
    )
    node_target = CodeNode(
        node_id=compute_code_node_id("payment.py", "process_refund", code_target),
        repo_name="test_repo",
        file_path="src/payment.py",
        language="python",
        symbol_type="function",
        symbol_name="process_refund",
        start_line=1,
        end_line=10,
        raw_code=code_target,
        context_header=code_target,
    )
    store.upsert_nodes("code_chunks", [node_caller, node_target])

    tools = AgentTools(vector_store=store)

    graph = tools.get_callers_and_callees_tool(symbol_name="process_refund")

    # Callees của process_refund phải bao gồm get_payment_gateway, charge_back, audit_log
    assert "get_payment_gateway" in graph["callees"]
    assert "charge_back" in graph["callees"]
    assert "audit_log" in graph["callees"]

    # Callers của process_refund phải bao gồm checkout
    assert "checkout" in graph["callers"]


# =============================================================================
# TEST CÔNG CỤ 4: search_git_history_tool
# =============================================================================
def test_search_git_history_tool_local_and_fallback():
    """Kiểm thử search_git_history_tool từ Git cục bộ và fallback Vector Store."""
    tools = AgentTools(repo_path=".")
    # Kiểm tra truy vấn từ repo hiện tại
    commits = tools.search_git_history_tool(limit=5)
    assert isinstance(commits, list)
    if commits:
        assert isinstance(commits[0], IncidentDocNode)
        assert commits[0].source_type == "commit"
        assert commits[0].commit_sha is not None

    # Kiểm tra fallback qua vector store
    embedder = HybridEmbedder()
    store = QdrantStore(location=":memory:", embedder=embedder)
    mock_commit_node = IncidentDocNode(
        doc_id="commit123",
        source_type="commit",
        title="Commit: Fix payment timeout bug",
        content="Commit SHA: abc123def\nFix payment gateway timeout",
        related_files=["src/payment.py"],
        commit_sha="abc123def",
    )
    store.upsert_nodes("incident_and_docs", [mock_commit_node])

    # Giả lập repo không có Git bằng đường dẫn ảo
    tools_fallback = AgentTools(vector_store=store, repo_path="/invalid/non_existent_dir")
    fb_commits = tools_fallback.search_git_history_tool(file_path="src/payment.py", limit=2)
    assert len(fb_commits) > 0
    assert fb_commits[0].source_type == "commit"
    assert "Fix payment" in fb_commits[0].title


# =============================================================================
# TEST CÔNG CỤ 5: search_runbooks_tool
# =============================================================================
def test_search_runbooks_tool():
    """Kiểm thử search_runbooks_tool tìm kiếm tài liệu ứng cứu Runbook."""
    embedder = HybridEmbedder()
    store = QdrantStore(location=":memory:", embedder=embedder)
    reranker = CrossEncoderReranker()

    rb_node = IncidentDocNode(
        doc_id="rb_timeout",
        source_type="runbook",
        title="Runbook: Khắc phục sự cố Payment Timeout",
        content="Khi gặp lỗi PaymentTimeoutError hoặc HTTP 504: Cần restart stripe-proxy pod và kiểm tra circuit breaker.",
        related_files=["src/payment.py"],
    )
    doc_node = IncidentDocNode(
        doc_id="doc_general",
        source_type="markdown_doc",
        title="Kiến trúc hệ thống thanh toán",
        content="Tổng quan kiến trúc module payment và kết nối cổng thanh toán.",
        related_files=["src/payment.py"],
    )
    store.upsert_nodes("incident_and_docs", [rb_node, doc_node])

    tools = AgentTools(vector_store=store, reranker=reranker)

    runbooks = tools.search_runbooks_tool(error_signature="PaymentTimeoutError", top_k=2)
    assert len(runbooks) > 0
    assert "Payment Timeout" in runbooks[0].content or "Payment Timeout" in runbooks[0].metadata.get("title", "")

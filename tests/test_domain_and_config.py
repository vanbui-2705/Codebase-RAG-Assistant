"""
Unit tests cho Giai đoạn 0: Kiểm thử Tầng Domain và Cấu hình Settings.
Đảm bảo Clean Architecture, tính toàn vẹn của Schemas và cấu hình Pydantic.
"""
import sys
from src.config.settings import Settings, get_settings
from src.domain.schemas import (
    CodeNode,
    IncidentDocNode,
    RetrievedContext,
    TaskType,
    compute_code_node_id,
)
from src.domain.interfaces import BaseCodeParser, BaseVectorStore, BaseReranker


def test_settings_singleton_and_defaults():
    """Kiểm tra việc load cấu hình Settings và hoạt động của singleton."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2
    assert settings1.gemini_model == "gemini-2.5-flash"
    assert settings1.qdrant_port == 6333
    assert settings1.embedding_model_name == "BAAI/bge-m3"
    assert settings1.reranker_model_name == "BAAI/bge-reranker-v2-m3"


def test_code_node_schema_and_id_hash():
    """Kiểm tra tính hợp lệ của schema CodeNode và hàm băm SHA256 id."""
    file_path = "src/services/payment.py"
    symbol_name = "PaymentService.process_refund"
    raw_code = "def process_refund(order_id: str) -> bool:\n    return True"

    node_id = compute_code_node_id(file_path, symbol_name, raw_code)
    assert len(node_id) == 64  # SHA256 length

    node = CodeNode(
        node_id=node_id,
        repo_name="my-service-repo",
        file_path=file_path,
        language="python",
        symbol_type="method",
        symbol_name=symbol_name,
        start_line=10,
        end_line=25,
        imports=["import stripe", "from decimal import Decimal"],
        raw_code=raw_code,
        context_header=f"# File: {file_path}\n# Imports: stripe\n{raw_code}",
        last_modified_commit="a1b2c3d4",
    )

    assert node.node_id == node_id
    assert node.symbol_name == symbol_name
    assert len(node.imports) == 2
    assert node.last_modified_commit == "a1b2c3d4"


def test_incident_doc_node_schema():
    """Kiểm tra schema IncidentDocNode cho Runbook và Pull Request."""
    doc = IncidentDocNode(
        doc_id="doc-uuid-12345",
        source_type="runbook",
        title="Hướng dẫn xử lý lỗi Payment Timeout",
        content="Khi gặp lỗi timeout, thực hiện rollback service payment...",
        related_files=["src/services/payment.py", "config/payment.yaml"],
        author="sre-team",
    )

    assert doc.doc_id == "doc-uuid-12345"
    assert doc.source_type == "runbook"
    assert len(doc.related_files) == 2


def test_retrieved_context_schema():
    """Kiểm tra schema RetrievedContext kết quả truy xuất."""
    ctx = RetrievedContext(
        node_id="node-123",
        source_type="code",
        file_path="src/auth.py",
        content="def verify_token(): pass",
        score=0.89,
        start_line=1,
        end_line=5,
        symbol_name="verify_token",
        metadata={"repo": "main-repo"},
    )

    assert ctx.score == 0.89
    assert ctx.symbol_name == "verify_token"
    assert ctx.metadata["repo"] == "main-repo"


def test_clean_architecture_domain_independence():
    """
    Quy tắc Clean Architecture:
    Kiểm tra các module trong src/domain/ tuyệt đối KHÔNG import từ
    src.ingestion, src.retrieval, src.agent hoặc src.api.
    """
    domain_modules = [m for m in sys.modules.keys() if m.startswith("src.domain")]
    forbidden_prefixes = ("src.ingestion", "src.retrieval", "src.agent", "src.api")

    for mod_name in domain_modules:
        mod = sys.modules[mod_name]
        if hasattr(mod, "__file__") and mod.__file__:
            with open(mod.__file__, "r", encoding="utf-8") as f:
                content = f.read()
                for prefix in forbidden_prefixes:
                    assert f"from {prefix}" not in content, f"Vi phạm Clean Architecture: {mod_name} import {prefix}"
                    assert f"import {prefix}" not in content, f"Vi phạm Clean Architecture: {mod_name} import {prefix}"

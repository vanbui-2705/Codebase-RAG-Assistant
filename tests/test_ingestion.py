"""
Unit tests cho Tầng Ingestion (Giai đoạn 1):
Kiểm thử TreeSitterCodeParser (AST Parsing, Header Injection, Line Boundaries)
và RunbookParser (Markdown Heading Chunking, Code Block Preservation).
"""
from pathlib import Path
import tempfile
from src.ingestion.code_parser import TreeSitterCodeParser
from src.ingestion.doc_parser import RunbookParser


def test_tree_sitter_python_parsing_and_header():
    """Kiểm tra TreeSitterCodeParser bóc tách chính xác class, method, function và tiêm header."""
    sample_code = '''import os
from decimal import Decimal

class PaymentService:
    """Dịch vụ xử lý thanh toán và hoàn tiền."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def process_refund(self, order_id: str, amount: Decimal) -> bool:
        if amount <= 0:
            return False
        return True

def calculate_fee(amount: float) -> float:
    return amount * 0.02
'''

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(sample_code)
        temp_file = f.name

    try:
        parser = TreeSitterCodeParser(repo_name="test_repo")
        nodes = parser.parse_file(temp_file, language="python")

        # Kỳ vọng tìm thấy: class PaymentService, method __init__, method process_refund, function calculate_fee
        symbol_names = [node.symbol_name for node in nodes]
        assert "PaymentService" in symbol_names
        assert "PaymentService.process_refund" in symbol_names
        assert "PaymentService.__init__" in symbol_names
        assert "calculate_fee" in symbol_names

        # Kiểm tra method process_refund
        refund_node = next(n for n in nodes if n.symbol_name == "PaymentService.process_refund")
        assert refund_node.symbol_type == "method"
        assert refund_node.start_line >= 10
        assert refund_node.end_line > refund_node.start_line
        assert any("Decimal" in imp for imp in refund_node.imports)

        # Kiểm tra Contextual Header
        assert "# File:" in refund_node.context_header
        assert "PaymentService.process_refund" in refund_node.context_header
        assert "def process_refund" in refund_node.context_header
    finally:
        Path(temp_file).unlink(missing_ok=True)


def test_tree_sitter_empty_or_nonexistent_file():
    """Kiểm tra parser xử lý an toàn khi gặp file rỗng hoặc không tồn tại."""
    parser = TreeSitterCodeParser()

    assert parser.parse_file("nonexistent_path_xyz.py", "python") == []

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("   \n\n  ")
        empty_file = f.name

    try:
        assert parser.parse_file(empty_file, "python") == []
    finally:
        Path(empty_file).unlink(missing_ok=True)


def test_runbook_parser_headings_and_codeblocks():
    """Kiểm tra RunbookParser tách theo Heading và bảo toàn khối code blocks."""
    sample_markdown = """# Hướng dẫn Xử lý Sự cố Thanh toán

Tài liệu này mô tả quy trình khi gặp lỗi kết nối tới Payment Gateway.

## Dấu hiệu Lỗi Timeout
Khi hệ thống ghi nhận ngoại lệ `PaymentTimeoutException` trong file `src/services/payment.py`:
- Mã lỗi HTTP 504.
- Ảnh hưởng trực tiếp tới đơn hàng.

```bash
# Lệnh kiểm tra log khẩn cấp
kubectl logs -l app=payment-service --tail=100
curl -X GET http://payment.internal/health
```

## Quy trình Khắc phục
1. Kiểm tra circuit breaker tại `config/gateway.yaml`.
2. Khởi động lại worker nếu nghẽn queue.
"""

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(sample_markdown)
        temp_file = f.name

    try:
        parser = RunbookParser()
        docs = parser.parse_markdown(temp_file)

        assert len(docs) >= 2
        titles = [d.title for d in docs]
        assert any("Dấu Hiệu Lỗi Timeout" in t or "Dấu hiệu Lỗi Timeout" in t for t in titles)
        assert any("Quy Trình Khắc Phục" in t or "Quy trình Khắc phục" in t for t in titles)

        # Kiểm tra bảo toàn code block
        timeout_doc = next(d for d in docs if "Timeout" in d.title)
        assert "```bash" in timeout_doc.content
        assert "kubectl logs" in timeout_doc.content
        assert "src/services/payment.py" in timeout_doc.related_files
    finally:
        Path(temp_file).unlink(missing_ok=True)

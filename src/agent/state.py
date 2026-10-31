"""
Module: src.agent.state
Chức năng: Định nghĩa cấu trúc AgentGraphState cho đồ thị trạng thái LangGraph.
Lưu trữ toàn bộ trạng thái xuyên suốt quá trình phân tích, truy xuất đa nguồn,
đánh giá độ đầy đủ, mở rộng phụ thuộc, sinh phản hồi và kiểm định trích dẫn.
Tuân thủ Clean Architecture: Chỉ phụ thuộc vào tầng src.domain.
"""

from typing import TypedDict

from src.domain.schemas import RetrievedContext


class AgentGraphState(TypedDict):
    """
    Trạng thái luồng xử lý trong LangGraph StateGraph của On-call Agent.

    Attributes:
        raw_input: Câu hỏi của người dùng hoặc nội dung Error Log / Stack Trace ban đầu.
        task_type: Loại tác vụ ('codebase_qa' hoặc 'incident_rca').
        error_signature: Tên ngoại lệ / chữ ký lỗi nếu là sự cố on-call, hoặc None.
        extracted_symbols: Danh sách tên hàm, class hoặc đường dẫn file bóc tách được.
        code_contexts: Danh sách ngữ cảnh mã nguồn đã được truy xuất và xếp hạng.
        incident_contexts: Danh sách ngữ cảnh Runbook, Git commit / PR liên quan.
        iteration_count: Số lần lặp mở rộng phụ thuộc Call Graph (tối đa 3 lần).
        is_sufficient: Cờ đánh dấu ngữ cảnh hiện tại đã đủ để trả lời hay chưa.
        citation_valid: Cờ đánh dấu tất cả trích dẫn mã nguồn có hợp lệ hay không.
        citation_error: Chi tiết lỗi trích dẫn nếu phát hiện số dòng ngoài phạm vi thực tế.
        final_response: Câu trả lời cuối cùng hoặc báo cáo RCA hoàn chỉnh.
    """

    raw_input: str
    task_type: str  # 'codebase_qa' | 'incident_rca'
    error_signature: str | None
    extracted_symbols: list[str]
    code_contexts: list[RetrievedContext]
    incident_contexts: list[RetrievedContext]
    iteration_count: int
    is_sufficient: bool
    citation_valid: bool
    citation_error: str | None
    final_response: str

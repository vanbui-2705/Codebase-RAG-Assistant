"""
Module: src.domain.schemas
Chức năng: Định nghĩa các Data Models dùng chung trong toàn bộ hệ thống (Domain Schemas).
Tuân thủ Clean Architecture: Không phụ thuộc vào bất kỳ tầng bên ngoài nào.
"""
from enum import Enum
import hashlib
from typing import Any
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Phân loại ý định tác vụ của người dùng."""
    CODEBASE_QA = "codebase_qa"
    INCIDENT_RCA = "incident_rca"


def compute_code_node_id(file_path: str, symbol_name: str, raw_code: str) -> str:
    """Tạo mã định danh băm SHA256 duy nhất cho một đoạn mã nguồn."""
    content_hash = hashlib.sha256(raw_code.encode("utf-8")).hexdigest()[:16]
    raw_key = f"{file_path}::{symbol_name}::{content_hash}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class CodeNode(BaseModel):
    """
    Biểu diễn một đơn vị cú pháp mã nguồn (Hàm, Class, Method, Module)
    sau khi được phân tích qua cây cú pháp AST (Tree-sitter).
    Lưu trong Qdrant collection `code_chunks`.
    """
    node_id: str = Field(description="SHA256 băm từ file_path::symbol_name::raw_code")
    repo_name: str = Field(description="Tên repository")
    file_path: str = Field(description="Đường dẫn tương đối từ repo root")
    language: str = Field(description="Ngôn ngữ lập trình: python, typescript, go, java")
    symbol_type: str = Field(description="Loại thực thể: function, class, method, module")
    symbol_name: str = Field(description="Tên định danh (VD: PaymentService.process_refund)")
    start_line: int = Field(description="Dòng bắt đầu trong file (1-indexed)")
    end_line: int = Field(description="Dòng kết thúc trong file (1-indexed)")
    imports: list[str] = Field(default_factory=list, description="Danh sách thư viện/module import")
    raw_code: str = Field(description="Nội dung mã nguồn gốc kèm docstring")
    context_header: str = Field(description="Đoạn header ngữ cảnh (file, imports, class) phục vụ embedding")
    last_modified_commit: str | None = Field(default=None, description="SHA commit sửa đổi gần nhất")


class IncidentDocNode(BaseModel):
    """
    Biểu diễn một đơn vị tài liệu kỹ thuật, Runbook, Pull Request hoặc Commit.
    Lưu trong Qdrant collection `incident_and_docs`.
    """
    doc_id: str = Field(description="UUID hoặc SHA băm định danh tài liệu")
    source_type: str = Field(
        description="Nguồn dữ liệu: runbook, markdown_doc, pull_request, commit, github_issue"
    )
    title: str = Field(description="Tiêu đề tài liệu hoặc tiêu đề commit/PR")
    content: str = Field(description="Nội dung chi tiết")
    related_files: list[str] = Field(default_factory=list, description="Các file liên quan")
    commit_sha: str | None = Field(default=None, description="Mã SHA commit (nếu là commit)")
    author: str | None = Field(default=None, description="Tác giả")
    timestamp: str | None = Field(default=None, description="Thời gian định dạng ISO 8601")
    url: str | None = Field(default=None, description="Đường dẫn tới PR, Issue hoặc Wiki")


class RetrievedContext(BaseModel):
    """
    Biểu diễn một mẩu ngữ cảnh được truy xuất và xếp hạng bởi tầng Retrieval.
    Sẵn sàng đưa vào đồ thị điều phối Agent và LLM.
    """
    node_id: str = Field(description="Mã định danh của node gốc (CodeNode hoặc IncidentDocNode)")
    source_type: str = Field(description="Loại nguồn: code, runbook, commit, pull_request")
    file_path: str = Field(description="Đường dẫn file liên quan")
    content: str = Field(description="Nội dung mã hoặc tài liệu")
    score: float = Field(default=0.0, description="Điểm tương đồng sau khi Rerank hoặc RRF")
    start_line: int | None = Field(default=None, description="Dòng bắt đầu nếu là code")
    end_line: int | None = Field(default=None, description="Dòng kết thúc nếu là code")
    symbol_name: str | None = Field(default=None, description="Tên định danh hàm/class nếu là code")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata bổ sung")

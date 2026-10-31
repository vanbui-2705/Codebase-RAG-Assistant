"""
Module: src.api.routes
Chức năng: Khai báo 4 REST API Endpoints cốt lõi của hệ thống On-call & Codebase RAG:
1. POST /api/v1/query: Hỏi đáp kiến trúc codebase, trả về câu trả lời kèm trích dẫn dòng.
2. POST /api/v1/incident/rca: Tiếp nhận Error log / Stack trace và trả về báo cáo RCA 4 phần.
3. POST /api/v1/ingest/full: Kích hoạt Full Indexing cho toàn bộ kho mã nguồn và tài liệu runbook.
4. POST /api/v1/ingest/webhook: Nhận GitHub Webhook payload để kích hoạt Incremental Sync.

Tuân thủ Clean Architecture: Phụ thuộc domain, config và nhận service qua FastAPI Depends.
"""

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from src.agent.state import AgentGraphState
from src.domain.schemas import RetrievedContext
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["On-Call & Codebase RAG"])

CITATION_REGEX = re.compile(r"\[([^\[\]\n]+:L\d+-L\d+)\]")


# =============================================================================
# PYDANTIC SCHEMAS CHO REQUEST / RESPONSE
# =============================================================================


class QueryRequest(BaseModel):
    """Schema yêu cầu hỏi đáp kiến trúc và luồng xử lý mã nguồn."""

    query: str = Field(..., description="Nội dung câu hỏi kiến trúc hoặc logic nghiệp vụ")
    top_k: int = Field(default=5, ge=1, le=50, description="Số lượng ngữ cảnh tối đa cần truy xuất")


class QueryResponse(BaseModel):
    """Schema phản hồi kết quả hỏi đáp kèm trích dẫn và ngữ cảnh."""

    answer: str = Field(..., description="Câu trả lời từ Agent kèm trích dẫn chính xác")
    citations: list[str] = Field(
        default_factory=list,
        description="Danh sách trích dẫn mã nguồn trích xuất được dạng [file_path:L_start-L_end]",
    )
    retrieved_contexts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Danh sách chi tiết các mẩu ngữ cảnh đã dùng để sinh câu trả lời",
    )


class IncidentRCARequest(BaseModel):
    """Schema yêu cầu phân tích sự cố và chẩn đoán nguyên nhân gốc rễ (RCA)."""

    error_log: str = Field(..., description="Log lỗi, exception hoặc stack trace sự cố on-call")
    repo_path: str | None = Field(
        default=None,
        description="Đường dẫn repository liên quan (nếu có, mặc định dùng repo đã cấu hình)",
    )


class IncidentRCAResponse(BaseModel):
    """Schema phản hồi báo cáo RCA 4 phần tiêu chuẩn."""

    rca_report: str = Field(..., description="Nội dung báo cáo RCA chi tiết theo cấu trúc 4 phần")
    error_signature: str | None = Field(
        default=None,
        description="Chữ ký lỗi hoặc tên ngoại lệ bóc tách được (ví dụ: ZeroDivisionError)",
    )
    culprit_commits: list[str] = Field(
        default_factory=list,
        description="Danh sách các commit nghi vấn gây ra lỗi trong lịch sử Git",
    )
    runbook_recommendations: list[str] = Field(
        default_factory=list,
        description="Danh sách tài liệu Runbook hoặc hướng dẫn khắc phục liên quan",
    )


class IngestFullRequest(BaseModel):
    """Schema kích hoạt Full Indexing cho toàn bộ repository."""

    repo_path: str = Field(..., description="Đường dẫn tới repository mã nguồn cần đánh chỉ mục")
    docs_path: str | None = Field(
        default=None,
        description="Đường dẫn thư mục chứa runbook/tài liệu markdown (nếu nằm ngoài repo)",
    )
    repo_full_name: str | None = Field(
        default=None,
        description="Tên GitHub repo định dạng 'owner/repo' nếu muốn đồng bộ Pull Requests",
    )


class IngestFullResponse(BaseModel):
    """Schema phản hồi kết quả Full Indexing."""

    status: str = Field(..., description="Trạng thái thực thi ('success' hoặc 'failed')")
    indexed_summary: dict[str, int] = Field(
        ...,
        description="Thống kê số lượng node mã nguồn và tài liệu đã được đánh chỉ mục",
    )


class WebhookPayload(BaseModel):
    """
    Schema tiếp nhận payload GitHub Webhook cho các sự kiện push hoặc pull_request.
    Cho phép các trường mở rộng từ GitHub Webhook.
    """

    model_config = ConfigDict(extra="allow")

    ref: str | None = Field(default=None, description="Git ref (ví dụ: refs/heads/main)")
    before: str | None = Field(default=None, description="SHA commit trước khi push")
    after: str | None = Field(default=None, description="SHA commit sau khi push")
    repository: dict[str, Any] | None = Field(
        default=None, description="Thông tin repository gửi webhook"
    )
    pull_request: dict[str, Any] | None = Field(
        default=None, description="Chi tiết Pull Request nếu là PR event"
    )
    action: str | None = Field(
        default=None, description="Hành động sự kiện (ví dụ: opened, closed, synchronize)"
    )


class WebhookResponse(BaseModel):
    """Schema phản hồi kết quả đồng bộ Incremental Sync qua Webhook."""

    status: str = Field(..., description="Trạng thái đồng bộ ('success', 'skipped' hoặc 'failed')")
    sync_summary: dict[str, int] = Field(
        ...,
        description="Thống kê số file xóa và số node mới được nạp lại",
    )


# =============================================================================
# DEPENDENCY INJECTION HELPERS
# =============================================================================


def get_workflow_runner(request: Request) -> Any:
    """Lấy runner hoặc compiled graph của LangGraph Workflow từ app state."""
    workflow = getattr(request.app.state, "workflow", None)
    if workflow is None:
        workflow = getattr(request.app.state, "compiled_workflow", None)

    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Agent Workflow chưa được khởi tạo trong ứng dụng.",
        )

    # Nếu truyền vào OnCallAgentWorkflow instance chưa compile, tự động biên dịch
    if hasattr(workflow, "compile_graph"):
        compiled = getattr(request.app.state, "compiled_workflow", None)
        if compiled is None:
            compiled = workflow.compile_graph()
            request.app.state.compiled_workflow = compiled
        return compiled

    return workflow


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    """Lấy IngestionPipeline instance từ app state."""
    pipeline = getattr(request.app.state, "ingestion_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion Pipeline chưa được cấu hình trong ứng dụng.",
        )
    return pipeline


def _extract_citations(text: str) -> list[str]:
    """Trích xuất tất cả trích dẫn dạng [file_path:L_start-L_end] từ văn bản trả lời."""
    matches = CITATION_REGEX.findall(text)
    # Loại bỏ trùng lặp giữ nguyên thứ tự xuất hiện
    seen: set[str] = set()
    unique_citations: list[str] = []
    for m in matches:
        formatted = f"[{m}]"
        if formatted not in seen:
            seen.add(formatted)
            unique_citations.append(formatted)
    return unique_citations


def _serialize_contexts(contexts: list[RetrievedContext] | list[Any]) -> list[dict[str, Any]]:
    """Chuyển đổi danh sách RetrievedContext sang dạng dict an toàn cho JSON serialization."""
    serialized: list[dict[str, Any]] = []
    for ctx in contexts:
        if isinstance(ctx, RetrievedContext):
            serialized.append(ctx.model_dump())
        elif hasattr(ctx, "model_dump"):
            serialized.append(ctx.model_dump())
        elif isinstance(ctx, dict):
            serialized.append(ctx)
        else:
            serialized.append({
                "node_id": str(getattr(ctx, "node_id", getattr(ctx, "doc_id", "unknown"))),
                "source_type": str(getattr(ctx, "source_type", "unknown")),
                "file_path": str(getattr(ctx, "file_path", "")),
                "content": str(getattr(ctx, "content", "")),
                "score": float(getattr(ctx, "score", 0.0)),
            })
    return serialized


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Hỏi đáp kiến trúc codebase",
    description="Nhận câu hỏi kiến trúc/code và trả về giải thích kèm trích dẫn dòng chính xác.",
)
async def query_codebase(
    payload: QueryRequest,
    workflow_runner: Any = Depends(get_workflow_runner),
) -> QueryResponse:
    """
    Endpoint xử lý hỏi đáp về mã nguồn với LangGraph Agent Workflow.
    Đảm bảo trích dẫn chính xác [file_path:L_start-L_end] và chống ảo giác.
    """
    logger.info(f"Tiếp nhận truy vấn kiến trúc: '{payload.query}' (top_k={payload.top_k})")

    initial_state: AgentGraphState = {
        "raw_input": payload.query,
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": [],
        "code_contexts": [],
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }

    try:
        final_state = workflow_runner.invoke(initial_state)
    except Exception as exc:
        logger.error(f"Lỗi khi thực thi LangGraph workflow trong /query: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quá trình xử lý truy vấn thất bại: {exc}",
        ) from exc

    answer = str(final_state.get("final_response", ""))
    citations = _extract_citations(answer)

    code_ctxs = final_state.get("code_contexts", [])
    incident_ctxs = final_state.get("incident_contexts", [])
    all_contexts = _serialize_contexts(code_ctxs + incident_ctxs)

    # Giới hạn số lượng context trả về theo payload.top_k nếu có
    if payload.top_k and len(all_contexts) > payload.top_k:
        all_contexts = all_contexts[: payload.top_k]

    return QueryResponse(
        answer=answer,
        citations=citations,
        retrieved_contexts=all_contexts,
    )


@router.post(
    "/incident/rca",
    response_model=IncidentRCAResponse,
    summary="Phân tích sự cố và chẩn đoán nguyên nhân gốc rễ (RCA)",
    description="Nhận stack trace/log lỗi, tự động bóc tách và tạo báo cáo RCA 4 phần chuẩn hóa.",
)
async def analyze_incident_rca(
    payload: IncidentRCARequest,
    workflow_runner: Any = Depends(get_workflow_runner),
) -> IncidentRCAResponse:
    """
    Endpoint phân tích sự cố On-call với báo cáo RCA 4 phần:
    1. Tóm tắt sự cố & Tác động
    2. Nguyên nhân gốc rễ (Root Cause)
    3. File và Commit nghi vấn (Culprit Commits)
    4. Khuyến nghị khắc phục & Runbook liên quan
    """
    logger.info("Tiếp nhận yêu cầu phân tích sự cố RCA từ stack trace/error log.")

    initial_state: AgentGraphState = {
        "raw_input": payload.error_log,
        "task_type": "incident_rca",
        "error_signature": None,
        "extracted_symbols": [],
        "code_contexts": [],
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }

    try:
        final_state = workflow_runner.invoke(initial_state)
    except Exception as exc:
        logger.error(
            f"Lỗi khi thực thi LangGraph workflow trong /incident/rca: {exc}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quá trình phân tích sự cố thất bại: {exc}",
        ) from exc

    rca_report = str(final_state.get("final_response", ""))
    error_sig_val = final_state.get("error_signature")
    error_signature = str(error_sig_val) if error_sig_val is not None else None

    # Bóc tách commit và runbook từ incident_contexts
    culprit_commits: list[str] = []
    runbook_recommendations: list[str] = []

    for ctx in final_state.get("incident_contexts", []):
        stype = getattr(ctx, "source_type", "")
        meta = getattr(ctx, "metadata", {}) or {}

        if stype == "commit":
            commit_sha = meta.get("commit_sha") or getattr(ctx, "node_id", "")
            title = meta.get("title", "")
            commit_entry = f"{commit_sha[:7]} - {title}".strip(" -") if commit_sha else title
            if commit_entry and commit_entry not in culprit_commits:
                culprit_commits.append(commit_entry)
        elif stype == "runbook":
            raw_title = meta.get("title") or getattr(ctx, "file_path", "")
            title = raw_title or getattr(ctx, "node_id", "")
            if title and title not in runbook_recommendations:
                runbook_recommendations.append(title)

    return IncidentRCAResponse(
        rca_report=rca_report,
        error_signature=error_signature,
        culprit_commits=culprit_commits,
        runbook_recommendations=runbook_recommendations,
    )


@router.post(
    "/ingest/full",
    response_model=IngestFullResponse,
    summary="Kích hoạt Full Indexing cho toàn bộ repository",
    description="Quét toàn bộ mã nguồn, runbook markdown và lịch sử git để nạp vào Qdrant.",
)
async def ingest_full_repository(
    payload: IngestFullRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestFullResponse:
    """Endpoint kích hoạt toàn diện quá trình phân tích và đánh chỉ mục toàn bộ repository."""
    logger.info(f"Bắt đầu Full Indexing: repo={payload.repo_path}, docs={payload.docs_path}")

    try:
        summary = pipeline.run_full_indexing(
            repo_path=payload.repo_path,
            docs_path=payload.docs_path,
            repo_full_name=payload.repo_full_name,
        )
        return IngestFullResponse(status="success", indexed_summary=summary)
    except FileNotFoundError as exc:
        logger.warning(f"Không tìm thấy đường dẫn repo trong /ingest/full: {exc}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(f"Lỗi trong quá trình Full Indexing: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full indexing thất bại: {exc}",
        ) from exc


@router.post(
    "/ingest/webhook",
    response_model=WebhookResponse,
    summary="GitHub Webhook đồng bộ thay đổi Incremental Sync",
    description="Nhận webhook push hoặc PR merged từ GitHub để cập nhật các file bị thay đổi.",
)
async def ingest_github_webhook(
    payload: WebhookPayload,
    request: Request,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> WebhookResponse:
    """
    Endpoint tiếp nhận Webhook từ GitHub:
    - Nếu là Pull Request: Chỉ kích hoạt đồng bộ khi PR được merge (action='closed' và merged=True).
    - Nếu là Push: Lấy commit before và after để tìm diff và chỉ re-index đúng file thay đổi.
    """
    logger.info(f"Nhận GitHub Webhook payload: ref={payload.ref}, action={payload.action}")

    # Xác định repo path từ settings hoặc app state
    default_repo = getattr(request.app.state, "default_repo_path", ".")

    base_commit: str | None = None
    head_commit: str | None = None

    # 1. Xử lý trường hợp Pull Request Webhook
    if payload.pull_request:
        pr_data = payload.pull_request
        is_merged = pr_data.get("merged", False)
        action = payload.action

        if action == "closed" and is_merged:
            base_commit = pr_data.get("base", {}).get("sha")
            head_commit = pr_data.get("head", {}).get("sha")
            logger.info(f"Phát hiện PR merged: {base_commit} -> {head_commit}")
        else:
            logger.info(f"PR webhook nhận action='{action}', merged={is_merged}. Bỏ qua đồng bộ.")
            return WebhookResponse(
                status="skipped",
                sync_summary={"deleted_files": 0, "reindexed_nodes": 0},
            )

    # 2. Xử lý trường hợp Push Event Webhook
    elif payload.after:
        head_commit = payload.after
        base_commit = payload.before

        # Nếu là commit khởi tạo hoặc branch mới
        if not base_commit or base_commit.startswith("0000000"):
            base_commit = f"{head_commit}~1"

    if not base_commit or not head_commit:
        logger.warning("Webhook payload không chứa đủ thông tin commit để tính git diff.")
        return WebhookResponse(
            status="skipped",
            sync_summary={"deleted_files": 0, "reindexed_nodes": 0},
        )

    try:
        summary = pipeline.run_incremental_sync(
            repo_path=default_repo,
            base_commit=base_commit,
            head_commit=head_commit,
        )
        return WebhookResponse(status="success", sync_summary=summary)
    except Exception as exc:
        logger.error(f"Lỗi khi thực hiện Incremental Sync từ Webhook: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Incremental sync thất bại: {exc}",
        ) from exc

"""
Module: src.api.app
Chức năng: Khởi tạo ứng dụng FastAPI với Lifespan Management, CORS, Request Logging,
Global Exception Handler và Health Check Endpoint.
Tuân thủ Clean Architecture: Quản lý vòng đời dịch vụ và Dependency Injection qua app.state.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agent.tools import AgentTools
from src.agent.workflow import OnCallAgentWorkflow
from src.api.routes import router as api_router
from src.config.settings import get_settings
from src.domain.interfaces import BaseVectorStore
from src.ingestion.code_parser import TreeSitterCodeParser
from src.ingestion.doc_parser import RunbookParser
from src.ingestion.git_connector import GitHistoryCollector
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.embedder import get_hybrid_embedder
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)


class FallbackLLM:
    """Fallback LLM giả lập phản hồi an toàn khi không có GEMINI_API_KEY hoặc mất mạng."""

    def invoke(self, prompt: str) -> str:
        """Sinh phản hồi mặc định dựa trên từ khóa trong prompt."""
        prompt_lower = prompt.lower()
        is_classify = "phân loại ý định" in prompt_lower or (
            "json" in prompt_lower and "task_type" in prompt_lower
        )
        if is_classify:
            return '{"task_type": "codebase_qa", "error_signature": null, "extracted_symbols": []}'
        if "đánh giá mức độ đầy đủ" in prompt_lower or "is_sufficient" in prompt_lower:
            return '{"is_sufficient": true, "reason": "Ngữ cảnh đầy đủ", "missing_aspects": []}'
        return (
            "Dựa trên ngữ cảnh codebase hiện tại, yêu cầu đã được xử lý thành công. "
            "Hệ thống vận hành ở chế độ dev/fallback."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời ứng dụng FastAPI (Lifespan Context Manager):
    - Khởi tạo QdrantStore, Embedder, Reranker, AgentTools, Workflow và IngestionPipeline.
    - Lưu trữ các dịch vụ vào app.state phục vụ Dependency Injection.
    - Dọn dẹp tài nguyên khi ứng dụng tắt.
    """
    logger.info("=== BẮT ĐẦU KHỞI TẠO HỆ THỐNG ON-CALL RAG API ===")
    settings = get_settings()

    # 1. Khởi tạo Vector Store nếu chưa được inject từ bên ngoài
    if not hasattr(app.state, "vector_store") or app.state.vector_store is None:
        try:
            embedder = getattr(app.state, "embedder", None) or get_hybrid_embedder()
            app.state.embedder = embedder
            app.state.vector_store = QdrantStore(embedder=embedder)
            logger.info("Đã kết nối QdrantStore thành công.")
        except Exception as exc:
            logger.warning(
                f"Không thể kết nối Qdrant Server ({exc}). "
                f"Chuyển sang Qdrant In-Memory mode cho local/dev."
            )
            app.state.vector_store = QdrantStore(location=":memory:")

    # 2. Khởi tạo Reranker
    if not hasattr(app.state, "reranker") or app.state.reranker is None:
        try:
            app.state.reranker = get_reranker()
        except Exception as exc:
            logger.warning(f"Lỗi khởi tạo Reranker ({exc}). Để AgentTools tự quản lý.")
            app.state.reranker = None

    # 3. Khởi tạo LLM
    if not hasattr(app.state, "llm") or app.state.llm is None:
        if settings.gemini_api_key and settings.gemini_api_key != "mock_or_dev_key":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                app.state.llm = ChatGoogleGenerativeAI(
                    model=settings.gemini_model,
                    google_api_key=settings.gemini_api_key,
                    temperature=0.0,
                )
                logger.info(f"Đã khởi tạo Gemini LLM model: {settings.gemini_model}")
            except Exception as exc:
                logger.warning(f"Lỗi kết nối Gemini API ({exc}). Sử dụng FallbackLLM.")
                app.state.llm = FallbackLLM()
        else:
            logger.info("Chưa cấu hình GEMINI_API_KEY hợp lệ. Sử dụng FallbackLLM cho dev mode.")
            app.state.llm = FallbackLLM()

    # 4. Khởi tạo Parsers & Connectors
    code_parser = getattr(app.state, "code_parser", None) or TreeSitterCodeParser()
    doc_parser = getattr(app.state, "doc_parser", None) or RunbookParser()
    git_collector = getattr(app.state, "git_collector", None) or GitHistoryCollector()

    app.state.code_parser = code_parser
    app.state.doc_parser = doc_parser
    app.state.git_collector = git_collector

    # 5. Khởi tạo AgentTools nếu chưa có
    if not hasattr(app.state, "agent_tools") or app.state.agent_tools is None:
        app.state.agent_tools = AgentTools(
            vector_store=app.state.vector_store,
            reranker=app.state.reranker,
            code_parser=code_parser,
            git_collector=git_collector,
        )

    # 6. Khởi tạo OnCallAgentWorkflow và compile graph nếu chưa có
    if not hasattr(app.state, "workflow") or app.state.workflow is None:
        workflow_instance = OnCallAgentWorkflow(
            tools=app.state.agent_tools,
            llm=app.state.llm,
        )
        app.state.workflow = workflow_instance
        app.state.compiled_workflow = workflow_instance.compile_graph()
        logger.info("Đã biên dịch đồ thị LangGraph OnCallAgentWorkflow thành công.")
    elif (
        hasattr(app.state.workflow, "compile_graph")
        and not hasattr(app.state, "compiled_workflow")
    ):
        app.state.compiled_workflow = app.state.workflow.compile_graph()

    # 7. Khởi tạo IngestionPipeline nếu chưa có
    if not hasattr(app.state, "ingestion_pipeline") or app.state.ingestion_pipeline is None:
        app.state.ingestion_pipeline = IngestionPipeline(
            code_parser=code_parser,
            doc_parser=doc_parser,
            git_connector=git_collector,
            vector_store=app.state.vector_store,
        )

    logger.info("=== HỆ THỐNG ON-CALL RAG API ĐÃ SẴN SÀNG TIẾP NHẬN YÊU CẦU ===")
    yield

    logger.info("=== ĐANG TẮT HỆ THỐNG VÀ DỌN DẸP TÀI NGUYÊN ===")


def create_app(
    vector_store: BaseVectorStore | None = None,
    llm: Any = None,
    agent_tools: Any = None,
    workflow: Any = None,
    pipeline: IngestionPipeline | None = None,
    default_repo_path: str = ".",
) -> FastAPI:
    """
    Hàm factory tạo và cấu hình ứng dụng FastAPI.
    Hỗ trợ Dependency Injection cho testing và production.

    Args:
        vector_store: Instance tùy chọn của Vector Store (ví dụ: in-memory mock cho test).
        llm: Instance tùy chọn của LLM Model.
        agent_tools: Instance tùy chọn của AgentTools.
        workflow: Instance tùy chọn của Agent Workflow hoặc Compiled Graph.
        pipeline: Instance tùy chọn của IngestionPipeline.
        default_repo_path: Thư mục repository mặc định cho Webhook sync.

    Returns:
        FastAPI: Đối tượng ứng dụng FastAPI đã được cấu hình đầy đủ.
    """
    app = FastAPI(
        title="On-call & Codebase RAG Assistant API",
        description=(
            "Hệ thống Agentic RAG chuyên biệt cho mã nguồn — hỗ trợ hỏi đáp kiến trúc codebase, "
            "chẩn đoán sự cố (RCA) từ stack trace, tra cứu lịch sử thiết kế và đồng bộ repository."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Đăng ký các instances được inject sẵn (nếu có) vào app.state
    app.state.vector_store = vector_store
    app.state.llm = llm
    app.state.agent_tools = agent_tools
    app.state.workflow = workflow
    if workflow is not None and not hasattr(workflow, "compile_graph"):
        app.state.compiled_workflow = workflow
    app.state.ingestion_pipeline = pipeline
    app.state.default_repo_path = default_repo_path

    # =========================================================================
    # MIDDLEWARES
    # =========================================================================

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Request Logging Middleware
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Incoming Request: {request.method} {request.url.path} from {client_ip}")

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            logger.info(
                f"Response: {request.method} {request.url.path} -> "
                f"Status {response.status_code} ({duration:.3f}s)"
            )
            return response
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                f"Request Error: {request.method} {request.url.path} -> "
                f"Exception: {exc} ({duration:.3f}s)",
                exc_info=True,
            )
            raise exc

    # =========================================================================
    # GLOBAL EXCEPTION HANDLER
    # =========================================================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(
            f"HTTPException [{exc.status_code}] tại {request.method} {request.url.path}: "
            f"{exc.detail}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "status_code": exc.status_code,
                "message": exc.detail,
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Lỗi chưa xử lý (Unhandled Exception) tại {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "status_code": 500,
                "message": "Đã xảy ra lỗi không mong muốn trên máy chủ.",
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    # =========================================================================
    # HEALTH CHECK ENDPOINT
    # =========================================================================

    @app.get(
        "/health",
        tags=["System"],
        summary="Kiểm tra trạng thái hoạt động của hệ thống",
        description="Trả về tình trạng API, phiên bản và trạng thái kết nối Vector Database.",
    )
    async def health_check(request: Request) -> dict[str, Any]:
        settings = get_settings()
        db_status = "connected"
        db_details = f"Qdrant kết nối tại {settings.qdrant_host}:{settings.qdrant_port}"

        try:
            vs = getattr(request.app.state, "vector_store", None)
            if vs is not None and hasattr(vs, "client"):
                # Thử ping Qdrant bằng lệnh lấy danh sách collections
                vs.client.get_collections()
            else:
                db_status = "uninitialized"
                db_details = "Vector store chưa được khởi tạo."
        except Exception as exc:
            db_status = "disconnected"
            db_details = f"Không thể kết nối Qdrant: {exc}"

        is_healthy = db_status in ("connected", "uninitialized")
        return {
            "status": "healthy" if is_healthy else "degraded",
            "version": "0.1.0",
            "service": "On-call & Codebase RAG Assistant",
            "database": {
                "status": db_status,
                "details": db_details,
            },
        }

    # Đăng ký REST API routes
    app.include_router(api_router)

    return app

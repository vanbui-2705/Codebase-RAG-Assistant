"""
Module: ui.chainlit_app
Chức năng: Giao diện Chat trực quan với Chainlit dành cho On-call & Codebase RAG Assistant:
1. Tiếp nhận câu hỏi kiến trúc hoặc Error Log / Stack trace (tin nhắn hoặc file đính kèm).
2. Tự động nhận diện định dạng log sự cố và chuyển đổi giữa chế độ QA và Incident RCA.
3. Trực quan hóa tiến trình suy luận (Thinking / Step Progression) của LangGraph qua cl.Step.
4. Highlight các trích dẫn dòng [file_path:L_start-L_end] và đính kèm ngữ cảnh mã nguồn/runbook.
Tuân thủ Clean Architecture: Tái sử dụng tầng Agent và Retrieval từ src.
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Any

import chainlit as cl

from src.agent.state import AgentGraphState
from src.agent.tools import AgentTools
from src.agent.workflow import OnCallAgentWorkflow
from src.config.settings import get_settings
from src.ingestion.code_parser import TreeSitterCodeParser
from src.ingestion.git_connector import GitHistoryCollector
from src.retrieval.chat_history_db import SQLiteChainlitDataLayer, get_chat_history_db
from src.retrieval.embedder import get_hybrid_embedder
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r"\[([^\[\]\n]+:L\d+-L\d+)\]")

STACK_TRACE_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r'File ".*?", line \d+', re.IGNORECASE),
    re.compile(r"(?:Exception|Error|panic):", re.IGNORECASE),
    re.compile(r"at [a-zA-Z0-9_$.]+\([a-zA-Z0-9_]+:\d+\)", re.IGNORECASE),
    re.compile(r"goroutine \d+ \[running\]:", re.IGNORECASE),
    re.compile(r"\tat [a-zA-Z0-9_$./]+\(.*?\.java:\d+\)", re.IGNORECASE),
]


class FallbackLLM:
    """Fallback LLM giả lập phản hồi cho giao diện Chainlit khi chưa có GEMINI_API_KEY."""

    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        is_classify = "phân loại ý định" in prompt_lower or (
            "json" in prompt_lower and "task_type" in prompt_lower
        )
        if is_classify:
            return '{"task_type": "codebase_qa", "error_signature": null, "extracted_symbols": []}'
        if "đánh giá mức độ đầy đủ" in prompt_lower or "is_sufficient" in prompt_lower:
            return '{"is_sufficient": true, "reason": "Ngữ cảnh đầy đủ", "missing_aspects": []}'
        return (
            "Dựa trên các đoạn mã nguồn và tài liệu đã truy xuất, hệ thống đã ghi nhận yêu cầu. "
            "(Chế độ demo / Fallback: Vui lòng cấu hình GEMINI_API_KEY trong file .env)."
        )


def _detect_is_stack_trace(text: str) -> bool:
    """Kiểm tra xem nội dung văn bản có chứa cấu trúc stack trace hoặc error log không."""
    for pattern in STACK_TRACE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _highlight_citations(text: str) -> str:
    """Tô đậm và định dạng lại các trích dẫn dòng [file:L1-L10] để hiển thị trực quan."""

    def _replace_citation(match: re.Match) -> str:
        citation_inner = match.group(1)
        return f"🔍 **`[{citation_inner}]`**"

    return CITATION_REGEX.sub(_replace_citation, text)


def _init_agent_workflow() -> Any:
    """Khởi tạo và biên dịch StateGraph cho ứng dụng Chainlit."""
    settings = get_settings()

    # 1. Vector Store & Embedder
    try:
        embedder = get_hybrid_embedder()
        vector_store = QdrantStore(embedder=embedder)
    except Exception as exc:
        logger.warning(f"Không thể kết nối Qdrant server ({exc}). Dùng Qdrant In-Memory mode.")
        vector_store = QdrantStore(location=":memory:")

    # 2. Reranker
    try:
        reranker = get_reranker()
    except Exception as exc:
        logger.warning(f"Lỗi khởi tạo Reranker ({exc}). Để AgentTools tự quản lý.")
        reranker = None

    # 3. LLM
    llm: Any = None
    if settings.gemini_api_key and settings.gemini_api_key != "mock_or_dev_key":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.gemini_api_key,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(f"Không thể kết nối Gemini API ({exc}). Sử dụng FallbackLLM.")
            llm = FallbackLLM()
    else:
        llm = FallbackLLM()

    # 4. Parsers & Agent Tools
    code_parser = TreeSitterCodeParser()
    git_collector = GitHistoryCollector()

    tools = AgentTools(
        vector_store=vector_store,
        reranker=reranker,
        code_parser=code_parser,
        git_collector=git_collector,
    )

    # 5. Biên dịch đồ thị LangGraph
    workflow_engine = OnCallAgentWorkflow(tools=tools, llm=llm)
    return workflow_engine.compile_graph()


@cl.data_layer
def custom_data_layer():
    """Đăng ký SQLite Data Layer để Chainlit hiển thị Sidebar lịch sử bên trái."""
    return SQLiteChainlitDataLayer()


@cl.header_auth_callback
async def header_auth_callback(headers: dict):
    """Tự động xác thực tài khoản cục bộ cho người dùng không cần gõ mật khẩu."""
    return cl.User(identifier="developer", metadata={"role": "admin"})


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Callback kích hoạt khi người dùng bấm vào một phiên trò chuyện ở Sidebar bên trái."""
    thread_id = thread.get("id")
    if not thread_id:
        return

    cl.user_session.set("thread_id", thread_id)
    meta = thread.get("metadata") or {}
    repo_name = meta.get("repo_name", "Codebase-RAG-Assistant")
    cl.user_session.set("repo_name", repo_name)

    # Nạp lại ngữ cảnh lịch sử hội thoại cho Workflow
    restored_history: list[dict[str, str]] = []
    for step in thread.get("steps", []):
        stype = step.get("type")
        content = step.get("output", "")
        if stype in ("user_message", "assistant_message") and content:
            role = "user" if stype == "user_message" else "assistant"
            restored_history.append({"role": role, "content": content})

    cl.user_session.set("chat_history", restored_history[-12:])
    logger.info(f"Đã khôi phục phiên trò chuyện từ Sidebar: {thread_id} ({len(restored_history)} tin nhắn)")


@cl.on_chat_start
async def on_chat_start():
    """Khởi tạo phiên trò chuyện Chainlit, nạp workflow và hiển thị hướng dẫn."""
    logger.info("Bắt đầu phiên làm việc mới trên Chainlit UI.")

    # 1. Khởi tạo Persistent Thread trong SQLite
    chat_db = get_chat_history_db()
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("repo_name", "Codebase-RAG-Assistant")

    # 2. Khởi tạo workflow và lưu vào session
    chunk_count = 0
    doc_count = 0
    try:
        compiled_workflow = _init_agent_workflow()
        cl.user_session.set("workflow", compiled_workflow)
        cl.user_session.set("chat_history", [])
        session_ready = True

        # Đếm số lượng chunks hiện có trong cơ sở dữ liệu
        settings = get_settings()
        from qdrant_client import QdrantClient
        raw_key = settings.qdrant_api_key.strip() if settings.qdrant_api_key and settings.qdrant_api_key.strip() else None
        qc = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, api_key=raw_key, https=False if raw_key is None else None)
        if qc.collection_exists("code_chunks"):
            chunk_count = qc.count("code_chunks").count
        if qc.collection_exists("incident_and_docs"):
            doc_count = qc.count("incident_and_docs").count
    except Exception as exc:
        logger.error(f"Lỗi khởi tạo Agent Workflow trong UI: {exc}", exc_info=True)
        session_ready = False

    status_tag = f"🟢 **Dữ liệu hiện có:** `{chunk_count}` code chunks, `{doc_count}` tài liệu/docs" if chunk_count > 0 else "🟡 **Chưa nạp dữ liệu:** Nạp mã nguồn bằng cách dán URL GitHub hoặc đính kèm file!"

    # 3. Lấy danh sách các phiên trò chuyện cũ từ SQLite
    recent_threads = chat_db.list_threads(limit=10)
    valid_threads = [
        th for th in recent_threads
        if th.get("message_count", 0) > 0 or th.get("title") not in ("Phiên trò chuyện mới", "Cuộc trò chuyện mới")
    ]
    history_actions: list[cl.Action] = []
    history_summary_lines: list[str] = []

    if valid_threads:
        history_summary_lines.append("\n### 📜 Cuộc trò chuyện gần đây:")
        for idx, th in enumerate(valid_threads[:4], 1):
            t_title = th.get("title", "Không có tiêu đề")
            t_repo = th.get("repo_name", "unknown")
            t_id = th.get("thread_id", "")
            t_cnt = th.get("message_count", 0)
            t_time = th.get("updated_at", "")[:16].replace("T", " ")
            history_summary_lines.append(f"- **{t_title}** *(Repo: `{t_repo}` | `{t_cnt}` tin nhắn | {t_time})*")
            history_actions.append(
                cl.Action(
                    name="resume_thread",
                    payload={"thread_id": t_id},
                    label=f"👉 Mở #{idx}: {t_title[:20]}",
                )
            )

    history_text = "\n".join(history_summary_lines) if history_summary_lines else ""

    welcome_message = (
        "## 🛠️ On-call & Codebase RAG Assistant\n\n"
        "Hệ thống **Agentic RAG** chuyên biệt cho mã nguồn & on-call đã sẵn sàng:\n\n"
        f"- {status_tag}\n\n"
        "### 📥 2 Cách nạp dữ liệu để RAG Chunking & Indexing:\n"
        "1. 🌐 **Dán thẳng URL GitHub**: Dán link repo (VD: `https://github.com/owner/repo`) vào khung chat, hệ thống tự động clone, parse AST và chunking.\n"
        "2. 📎 **Đính kèm File (Upload)**: Kéo thả file nén `.zip` của project hoặc các file mã nguồn rời (`.py`, `.ts`, `.go`, `.java`, `.md`...) vào nút kẹp ghim 📎 và bấm gửi để chunking.\n\n"
        "### 📜 Lịch sử các phiên trò chuyện (Menu bên trái):\n"
        "- 👈 **Sidebar bên trái:** Bấm trực tiếp vào các cuộc trò chuyện trên **Menu bên trái** để mở lại lịch sử và hỏi đáp tiếp.\n"
        "- ➕ **Nút '+ New Chat':** Nằm ở phía trên Menu bên trái để tạo cuộc trò chuyện mới bất kỳ lúc nào.\n"
        f"{history_text}\n\n"
        "*(Hãy dán link GitHub, đính kèm file hoặc đặt câu hỏi bên dưới để bắt đầu!)*"
    )

    if not session_ready:
        welcome_message += (
            "\n\n⚠️ *Cảnh báo: Không thể kết nối dịch vụ backend. Vui lòng kiểm tra Docker.*"
        )

    await cl.Message(
        content=welcome_message,
        actions=history_actions if history_actions else None,
    ).send()


async def _restore_thread_to_session(target_id: str) -> bool:
    """Khôi phục ngữ cảnh và lịch sử hội thoại từ SQLite vào user session hiện tại."""
    chat_db = get_chat_history_db()
    thread = chat_db.get_thread(target_id)
    if not thread:
        # Tìm thử theo prefix thread_id hoặc từ khóa title
        all_threads = chat_db.list_threads(limit=50)
        matched = [
            t for t in all_threads
            if t["thread_id"].startswith(target_id) or target_id.lower() in t["title"].lower()
        ]
        if matched:
            thread = matched[0]
            target_id = thread["thread_id"]

    if not thread:
        await cl.Message(content=f"⚠️ Không tìm thấy phiên trò chuyện tương ứng với `{target_id}` trong SQLite.").send()
        return False

    messages = chat_db.get_thread_messages(target_id)
    cl.user_session.set("thread_id", target_id)
    cl.user_session.set("repo_name", thread.get("repo_name", "Codebase-RAG-Assistant"))

    restored_history: list[dict[str, str]] = []
    display_lines: list[str] = [
        f"### 🔄 Đã khôi phục phiên: **{thread.get('title')}**\n"
        f"- 🆔 **Thread ID:** `{target_id}`\n"
        f"- 📁 **Repository:** `{thread.get('repo_name')}`\n"
        f"- 💬 **Lịch sử:** `{len(messages)}` tin nhắn\n\n---"
    ]

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        restored_history.append({"role": role, "content": content})
        prefix = "👤 **Bạn:**" if role == "user" else "🤖 **Assistant:**"
        snippet = content[:200] + ("..." if len(content) > 200 else "")
        display_lines.append(f"{prefix} {snippet}\n")

    cl.user_session.set("chat_history", restored_history[-12:])
    display_lines.append("\n*Ngữ cảnh cũ đã được nạp lại vào bộ nhớ! Bạn có thể đặt câu hỏi tiếp theo ngay bây giờ.*")

    await cl.Message(content="\n".join(display_lines)).send()
    return True


@cl.action_callback("resume_thread")
async def on_resume_thread_action(action: cl.Action):
    """Callback khi người dùng bấm nút mở lại phiên trò chuyện cũ từ SQLite."""
    target_id = action.payload.get("thread_id")
    if target_id:
        await _restore_thread_to_session(target_id)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Tiếp nhận tin nhắn người dùng hoặc file đính kèm:
    - Bóc tách nội dung từ text và file đính kèm.
    - Nhận diện sự cố vs hỏi đáp code.
    - Chạy LangGraph StateGraph và cập nhật từng bước suy luận qua cl.Step.
    - Định dạng câu trả lời và làm nổi bật trích dẫn.
    """
    compiled_workflow = cl.user_session.get("workflow")
    if compiled_workflow is None:
        compiled_workflow = _init_agent_workflow()
        cl.user_session.set("workflow", compiled_workflow)

    # 1. Đọc nội dung tin nhắn và các file đính kèm nếu có
    raw_text = message.content.strip()
    attached_contents: list[str] = []
    attached_file_paths: list[Path] = []

    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                try:
                    file_path = Path(element.path)
                    if file_path.is_file():
                        attached_file_paths.append(file_path)
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        attached_contents.append(f"=== ĐÍNH KÈM: {element.name} ===\n{content}")
                except Exception as exc:
                    logger.warning(f"Không thể đọc file đính kèm {element.name}: {exc}")

    if attached_contents:
        full_input = "\n\n".join(attached_contents)
        if raw_text:
            full_input = f"{raw_text}\n\n{full_input}"
    else:
        full_input = raw_text

    if not full_input and not attached_file_paths:
        await cl.Message(content="⚠️ Vui lòng nhập câu hỏi, dán link GitHub hoặc đính kèm file mã nguồn.").send()
        return

    # Lệnh nhanh: /clear để reset ngữ cảnh
    if full_input.strip().lower() == "/clear":
        cl.user_session.set("chat_history", [])
        await cl.Message(
            content="🧹 **Đã làm mới phiên trò chuyện!** Toàn bộ ngữ cảnh các câu hỏi trước đã được xóa sạch để bắt đầu chủ đề mới."
        ).send()
        return

    # Lệnh nhanh: /new để tạo phiên mới
    if full_input.strip().lower() in ("/new", "/newchat", "/reset"):
        chat_db = get_chat_history_db()
        new_thread_id = str(uuid.uuid4())
        repo_name = cl.user_session.get("repo_name", "Codebase-RAG-Assistant")
        chat_db.create_thread(
            thread_id=new_thread_id,
            repo_name=repo_name,
            title="Phiên trò chuyện mới",
        )
        cl.user_session.set("thread_id", new_thread_id)
        cl.user_session.set("chat_history", [])
        await cl.Message(
            content=(
                f"✨ **Đã tạo phiên trò chuyện mới trong SQLite!**\n\n"
                f"- 🆔 **Thread ID:** `{new_thread_id}`\n"
                f"- 📁 **Repository:** `{repo_name}`\n\n"
                f"*Bạn có thể bắt đầu đặt câu hỏi hoặc nạp mã nguồn ngay bây giờ.*"
            )
        ).send()
        return

    # Lệnh nhanh: /history để xem danh sách các phiên trò chuyện cũ
    if full_input.strip().lower() in ("/history", "/sessions", "/threads"):
        chat_db = get_chat_history_db()
        threads = chat_db.list_threads(limit=10)
        valid_threads = [
            th for th in threads
            if th.get("message_count", 0) > 0 or th.get("title") not in ("Phiên trò chuyện mới", "Cuộc trò chuyện mới")
        ]
        if not valid_threads:
            await cl.Message(content="📭 Chưa có phiên trò chuyện nào được lưu trong SQLite.").send()
            return

        current_thread_id = cl.user_session.get("thread_id")
        history_actions: list[cl.Action] = []
        msg_lines = ["### 📜 Danh sách các phiên trò chuyện đã lưu trong SQLite:\n"]
        for idx, th in enumerate(valid_threads, 1):
            t_id = th.get("thread_id", "")
            t_title = th.get("title", "Không có tiêu đề")
            t_repo = th.get("repo_name", "unknown")
            t_cnt = th.get("message_count", 0)
            t_time = th.get("updated_at", "")[:16].replace("T", " ")
            active_mark = " *(Đang mở)*" if t_id == current_thread_id else ""
            msg_lines.append(f"- **#{idx}: {t_title}**{active_mark}\n  *(Repo: `{t_repo}` | `{t_cnt}` tin nhắn | ID: `{t_id[:8]}...` | Cập nhật: {t_time})*")
            if t_id != current_thread_id:
                history_actions.append(
                    cl.Action(
                        name="resume_thread",
                        payload={"thread_id": t_id},
                        label=f"👉 Mở #{idx}: {t_title[:20]}",
                    )
                )
        msg_lines.append("\n*Bấm vào các nút bên dưới hoặc gõ `/resume <thread_id>` để mở lại phiên.*")
        await cl.Message(content="\n".join(msg_lines), actions=history_actions if history_actions else None).send()
        return

    # Lệnh nhanh: /resume <thread_id>
    if full_input.strip().lower().startswith("/resume"):
        parts = full_input.strip().split(maxsplit=1)
        if len(parts) > 1:
            await _restore_thread_to_session(parts[1].strip())
        else:
            await cl.Message(content="⚠️ Vui lòng cung cấp mã phiên: `/resume <thread_id>` hoặc dùng lệnh `/history` để chọn.").send()
        return

    # TRƯỜNG HỢP 1: TẢI FILE ĐÍNH KÈM ĐỂ CHUNKING (Upload file zip hoặc file mã nguồn để index)
    has_zip = any(fp.suffix.lower() == ".zip" for fp in attached_file_paths)
    is_upload_command = full_input.strip().lower().startswith("/upload") or has_zip or (
        attached_file_paths and any(keyword in raw_text.lower() for keyword in ["chunk", "index", "nạp", "nap", "học", "tải"])
    )

    if is_upload_command and attached_file_paths:
        indexing_step = cl.Step(name="🔄 Đang giải nén, quét AST và chunking các file tải lên...", type="run")
        await indexing_step.send()
        try:
            from src.ingestion.pipeline import IngestionPipeline
            from src.ingestion.doc_parser import RunbookParser
            embedder = get_hybrid_embedder()
            vector_store = QdrantStore(embedder=embedder)
            pipeline = IngestionPipeline(
                code_parser=TreeSitterCodeParser(),
                vector_store=vector_store,
                doc_parser=RunbookParser(),
                git_collector=GitHistoryCollector(),
            )
            uploaded_repo = cl.user_session.get("repo_name", "uploaded_project")
            summary = pipeline.index_uploaded_files(
                files=attached_file_paths,
                repo_name=uploaded_repo,
            )
            # Tạo phiên làm việc mới riêng cho project vừa tải lên
            new_thread_id = str(uuid.uuid4())
            cl.user_session.set("thread_id", new_thread_id)
            cl.user_session.set("chat_history", [])
            chat_db = get_chat_history_db()
            chat_db.create_thread(
                thread_id=new_thread_id,
                repo_name=uploaded_repo,
                title=f"Hỏi đáp {uploaded_repo}",
            )
            await indexing_step.remove()
            await cl.Message(
                content=(
                    f"✅ **Đã phân tích AST và chunking thành công các file đính kèm!**\n\n"
                    f"- 📁 **Mã nguồn (Code chunks):** `{summary.get('code_nodes', 0)}` chunks\n"
                    f"- 📚 **Tài liệu (Docs/Runbooks):** `{summary.get('doc_nodes', 0)}` bản ghi\n"
                    f"- 💾 **Phiên SQLite:** Đã tạo thread riêng `{new_thread_id[:8]}...` cho `{uploaded_repo}`\n\n"
                    f"*Dữ liệu đã được nạp vào Vector DB Qdrant. Bây giờ bạn có thể đặt câu hỏi về các file này!*"
                )
            ).send()
            return
        except Exception as exc:
            await indexing_step.remove()
            await cl.Message(content=f"❌ **Lỗi khi chunking file tải lên:** {exc}").send()
            return

    # TRƯỜNG HỢP 2: ĐƯA ĐƯỜNG DẪN REPO GITHUB (Dán link GitHub trực tiếp hoặc dùng /index <github_url>)
    github_match = re.search(r"https?://github\.com/([\w\.-]+)/([\w\.-]+?)(?:\.git)?(?:/)?$", full_input.strip())
    if not github_match and full_input.strip().startswith(("/index ", "/clone ")):
        parts = full_input.strip().split(maxsplit=1)
        if len(parts) > 1:
            github_match = re.search(r"https?://github\.com/([\w\.-]+)/([\w\.-]+)", parts[1].strip())

    if github_match:
        repo_owner = github_match.group(1)
        repo_slug = github_match.group(2)
        repo_full_name = f"{repo_owner}/{repo_slug}"
        github_url = f"https://github.com/{repo_full_name}"

        indexing_step = cl.Step(name=f"📥 Đang clone repository từ GitHub `{repo_full_name}` và chunking AST...", type="run")
        await indexing_step.send()

        try:
            from src.ingestion.pipeline import IngestionPipeline
            from src.ingestion.doc_parser import RunbookParser
            embedder = get_hybrid_embedder()
            vector_store = QdrantStore(embedder=embedder)
            pipeline = IngestionPipeline(
                code_parser=TreeSitterCodeParser(),
                vector_store=vector_store,
                doc_parser=RunbookParser(),
                git_connector=GitHistoryCollector(),
            )
            res = pipeline.index_from_github_url(repo_url=github_url)
            cl.user_session.set("repo_name", repo_full_name)
            cl.user_session.set("chat_history", [])  # Reset phiên mới cho repo vừa clone

            # Tạo thread mới trong SQLite gắn với repo vừa nạp
            new_thread_id = str(uuid.uuid4())
            cl.user_session.set("thread_id", new_thread_id)
            chat_db = get_chat_history_db()
            chat_db.create_thread(
                thread_id=new_thread_id,
                repo_name=repo_full_name,
                title=f"Phân tích {repo_full_name}",
            )

            await indexing_step.remove()
            await cl.Message(
                content=(
                    f"🎉 **Đã clone và đánh chỉ mục thành công GitHub Repository!**\n\n"
                    f"- 🔗 **Repository:** [{repo_full_name}]({github_url})\n"
                    f"- 📁 **Mã nguồn (Code chunks):** `{res.get('code_nodes', 0)}` chunks\n"
                    f"- 📚 **Lịch sử Git & PRs:** `{res.get('doc_nodes', 0)}` bản ghi\n"
                    f"- 💾 **Phiên SQLite:** Đã khởi tạo thread riêng `{new_thread_id[:8]}...`\n\n"
                    f"*Đã thiết lập phiên làm việc riêng cho repo `{repo_full_name}`. Hãy đặt câu hỏi bất kỳ về mã nguồn này!*"
                )
            ).send()
            return
        except Exception as exc:
            await indexing_step.remove()
            await cl.Message(content=f"❌ **Lỗi khi clone & index repo từ GitHub:** {exc}").send()
            return

    # Lệnh nhanh: /index <thư_mục_local>
    if full_input.strip().lower().startswith("/index"):
        parts = full_input.strip().split(maxsplit=1)
        target_path = parts[1].strip() if len(parts) > 1 else "/app"
        if target_path in (".", "./"):
            target_path = "/app"

        indexing_step = cl.Step(name=f"🔄 Đang quét AST và nạp chỉ mục cho thư mục: `{target_path}`...", type="run")
        await indexing_step.send()

        try:
            from src.ingestion.pipeline import IngestionPipeline
            from src.ingestion.doc_parser import RunbookParser
            embedder = get_hybrid_embedder()
            vector_store = QdrantStore(embedder=embedder)
            pipeline = IngestionPipeline(
                code_parser=TreeSitterCodeParser(),
                vector_store=vector_store,
                doc_parser=RunbookParser(),
                git_connector=GitHistoryCollector(),
            )
            summary = pipeline.run_full_indexing(
                repo_path=target_path,
                docs_path=f"{target_path}/Docs" if Path(f"{target_path}/Docs").is_dir() else None,
            )
            await indexing_step.remove()
            await cl.Message(
                content=(
                    f"✅ **Đã nạp thành công mã nguồn vào Vector Database!**\n\n"
                    f"- 📁 **Mã nguồn (Code chunks):** `{summary.get('code_nodes', 0)}` chunks\n"
                    f"- 📚 **Tài liệu (Docs/Runbooks):** `{summary.get('doc_nodes', 0)}` bản ghi\n\n"
                    f"*Bây giờ bạn có thể đặt câu hỏi về kiến trúc hoặc chẩn đoán lỗi trong repo này!*"
                )
            ).send()
        except Exception as exc:
            await indexing_step.remove()
            await cl.Message(content=f"❌ **Lỗi khi nạp chỉ mục:** {exc}").send()
        return

    # 2. Quản lý Thread ID và lưu trữ câu hỏi người dùng vào SQLite
    chat_db = get_chat_history_db()
    thread_id = cl.user_session.get("thread_id")
    repo_name = cl.user_session.get("repo_name", "Codebase-RAG-Assistant")
    if not thread_id:
        thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", thread_id)

    # Kiểm tra hoặc tạo mới thread trong SQLite
    thread_info = chat_db.get_thread(thread_id)
    clean_title = " ".join(full_input.split())[:45]
    if not thread_info:
        chat_db.create_thread(
            thread_id=thread_id,
            repo_name=repo_name,
            title=clean_title or "Phiên trò chuyện",
        )
    elif thread_info.get("title") in ("Phiên trò chuyện mới", "Cuộc trò chuyện mới", "New Chat", ""):
        chat_db.update_thread_title(thread_id, clean_title)

    # Lưu tin nhắn người dùng vào SQLite
    chat_db.add_message(thread_id=thread_id, role="user", content=full_input)

    # 3. Tự động nhận diện sự cố (Incident RCA) hay Hỏi đáp kiến trúc (Codebase QA)
    is_stack_trace = _detect_is_stack_trace(full_input)
    task_mode = "incident_rca" if is_stack_trace else "codebase_qa"

    status_badge = (
        "🚨 **Chế độ Chẩn đoán Sự cố (Incident RCA)**"
        if is_stack_trace
        else "🔍 **Chế độ Hỏi đáp Kiến trúc Codebase (Codebase Q&A)**"
    )

    # Lấy lịch sử hội thoại của session hiện tại (Multi-turn memory)
    chat_history: list[dict[str, str]] = cl.user_session.get("chat_history", [])

    # 4. Chuẩn bị state ban đầu cho đồ thị LangGraph
    initial_state: AgentGraphState = {
        "raw_input": full_input,
        "task_type": task_mode,
        "error_signature": None,
        "extracted_symbols": [],
        "code_contexts": [],
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
        "repo_name": repo_name,
        "chat_history": list(chat_history),
    }

    current_state: dict[str, Any] = dict(initial_state)

    # 5. Hiển thị thanh tiến trình chạy ngầm duy nhất (Status Indicator)
    status_step = cl.Step(name="🔍 Đang phân loại ý định & bóc tách cú pháp...", type="run")
    await status_step.send()

    thought_logs: list[str] = []

    try:
        # Chạy stream qua từng Node trong StateGraph
        for step_event in compiled_workflow.stream(initial_state):
            for node_name, node_update in step_event.items():
                current_state.update(node_update)

                if node_name == "classify_and_parse":
                    ttype = current_state.get("task_type", "codebase_qa")
                    sig = current_state.get("error_signature")
                    thought_logs.append(
                        f"**1. Phân loại:** `{ttype}` | Lỗi: `{sig or 'None'}`"
                    )
                    status_step.name = "📁 Đang truy xuất mã nguồn, lịch sử Git & Runbooks..."
                    await status_step.update()

                elif node_name == "multi_source_retrieval":
                    c_cnt = len(current_state.get("code_contexts", []))
                    i_cnt = len(current_state.get("incident_contexts", []))
                    thought_logs.append(
                        f"**2. Truy xuất:** {c_cnt} chunks code, {i_cnt} docs/commits"
                    )
                    status_step.name = "⚖️ Đang đánh giá mức độ đầy đủ của ngữ cảnh..."
                    await status_step.update()

                elif node_name == "grade_context":
                    is_suff = current_state.get("is_sufficient", True)
                    iter_num = current_state.get("iteration_count", 0)
                    thought_logs.append(
                        f"**3. Đánh giá:** {'Đủ' if is_suff else 'Thiếu'} (vòng {iter_num}/3)"
                    )
                    if not is_suff and iter_num < 3:
                        status_step.name = f"🔄 Đang mở rộng phụ thuộc Call Graph (vòng {iter_num + 1}/3)..."
                        await status_step.update()

                elif node_name == "expand_dependencies":
                    c_cnt = len(current_state.get("code_contexts", []))
                    iter_num = current_state.get("iteration_count", 0)
                    thought_logs.append(
                        f"**4. Mở rộng:** Call Graph mở rộng lên {c_cnt} chunks"
                    )
                    status_step.name = f"⚖️ Đang kiểm định lại ngữ cảnh (vòng {iter_num}/3)..."
                    await status_step.update()

                elif node_name == "generate_answer":
                    thought_logs.append("**5. Sinh phản hồi:** Hoàn tất sinh câu trả lời")
                    status_step.name = "✍️ Đang kiểm tra đối chiếu trích dẫn mã nguồn..."
                    await status_step.update()

                elif node_name == "validate_citations":
                    valid = current_state.get("citation_valid", True)
                    thought_logs.append(
                        f"**6. Trích dẫn:** {'Hợp lệ' if valid else 'Đã hiệu chỉnh'}"
                    )

    except Exception as exc:
        logger.error(f"Lỗi trong quá trình chạy LangGraph stream: {exc}", exc_info=True)
        try:
            current_state = compiled_workflow.invoke(initial_state)
        except Exception as inner_exc:
            await status_step.remove()
            await cl.Message(content=f"❌ **Đã xảy ra lỗi khi thực thi xử lý:** {inner_exc}").send()
            return

    # 5. Định dạng phản hồi cuối cùng
    final_raw_response = current_state.get("final_response", "")
    if not final_raw_response:
        final_raw_response = "Không tìm thấy nội dung phản hồi phù hợp trong hệ thống."

    # Highlight trích dẫn [file:L1-L10]
    highlighted_response = _highlight_citations(final_raw_response)

    # 6. Trích xuất danh sách context để hiển thị nguồn tham khảo
    code_contexts = current_state.get("code_contexts", [])
    incident_contexts = current_state.get("incident_contexts", [])

    elements: list[cl.Element] = []

    # Tạo các file snippet tham khảo có thể mở rộng trên giao diện
    for i, ctx in enumerate(code_contexts[:5], 1):
        file_path = getattr(ctx, "file_path", f"chunk_{i}")
        s_line = getattr(ctx, "start_line", 1) or 1
        e_line = getattr(ctx, "end_line", s_line) or s_line
        content = getattr(ctx, "content", "")
        elements.append(
            cl.Text(
                name=f"[{file_path}:L{s_line}-L{e_line}]",
                content=content,
                display="side",
            )
        )

    for i, ctx in enumerate(incident_contexts[:3], 1):
        stype = getattr(ctx, "source_type", "doc")
        file_path = getattr(ctx, "file_path", f"doc_{i}")
        meta = getattr(ctx, "metadata", {}) or {}
        title = meta.get("title", file_path)
        content = getattr(ctx, "content", "")
        elements.append(
            cl.Text(
                name=f"[{stype.upper()}] {title}",
                content=content,
                display="side",
            )
        )

    thinking_details = ""
    if thought_logs:
        log_items = "\n".join(f"- {log}" for log in thought_logs)
        thinking_details = f"\n\n<details>\n<summary>🧠 <i>Chi tiết tiến trình suy luận (Thinking)</i></summary>\n\n{log_items}\n\n</details>"

    response_text = f"{status_badge}\n\n---\n\n{highlighted_response}{thinking_details}"

    # Xóa thanh tiến trình chạy ngầm và gửi phản hồi chính thức
    try:
        await status_step.remove()
    except Exception:
        pass

    # Cập nhật lịch sử hội thoại vào session để ghi nhớ ngữ cảnh các lượt hỏi tiếp theo (Multi-turn memory)
    updated_history = cl.user_session.get("chat_history", [])
    updated_history.append({"role": "user", "content": full_input})
    updated_history.append({"role": "assistant", "content": final_raw_response})
    # Giữ tối đa 12 lượt gần nhất để tránh tràn context
    cl.user_session.set("chat_history", updated_history[-12:])

    # Lưu tin nhắn Assistant và citations vào SQLite
    citations_to_save: list[str] = []
    for m in CITATION_REGEX.finditer(final_raw_response):
        citations_to_save.append(m.group(1))
    for ctx in code_contexts:
        fp = getattr(ctx, "file_path", "")
        sl = getattr(ctx, "start_line", 1) or 1
        el = getattr(ctx, "end_line", sl) or sl
        if fp:
            citations_to_save.append(f"{fp}:L{sl}-L{el}")
    unique_citations = list(dict.fromkeys(citations_to_save))

    chat_db.add_message(
        thread_id=thread_id,
        role="assistant",
        content=final_raw_response,
        citations=unique_citations[:10],
    )

    await cl.Message(
        content=response_text,
        elements=elements if elements else None,
    ).send()

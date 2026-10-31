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
from pathlib import Path
from typing import Any

import chainlit as cl

from src.agent.state import AgentGraphState
from src.agent.tools import AgentTools
from src.agent.workflow import OnCallAgentWorkflow
from src.config.settings import get_settings
from src.ingestion.code_parser import TreeSitterCodeParser
from src.ingestion.git_connector import GitHistoryCollector
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


@cl.on_chat_start
async def on_chat_start():
    """Khởi tạo phiên trò chuyện Chainlit, nạp workflow và hiển thị hướng dẫn."""
    logger.info("Bắt đầu phiên làm việc mới trên Chainlit UI.")

    # Khởi tạo workflow và lưu vào session
    try:
        compiled_workflow = _init_agent_workflow()
        cl.user_session.set("workflow", compiled_workflow)
        session_ready = True
    except Exception as exc:
        logger.error(f"Lỗi khởi tạo Agent Workflow trong UI: {exc}", exc_info=True)
        session_ready = False

    welcome_message = (
        "## 🛠️ On-call & Codebase RAG Assistant\n\n"
        "Chào mừng bạn! Hệ thống **Agentic RAG** chuyên biệt cho mã nguồn đã sẵn sàng:\n\n"
        "- 🔍 **Hỏi đáp kiến trúc**: Giải thích flow xử lý kèm trích dẫn `[file:L_start-L_end]`.\n"
        "- 🚨 **Chẩn đoán sự cố (RCA)**: Bóc tách lỗi stack trace và lập báo cáo RCA 4 phần.\n"
        "- 📎 **Tải file đính kèm**: Kéo thả trực tiếp file `.log`, `.txt` vào khung chat.\n"
        "- 🛡️ **Anti-Hallucination**: Kiểm định số dòng thực tế trước khi hiển thị.\n\n"
        "*(Hãy nhập câu hỏi kiến trúc hoặc đính kèm log lỗi bên dưới để bắt đầu!)*"
    )

    if not session_ready:
        welcome_message += (
            "\n\n⚠️ *Cảnh báo: Không thể kết nối dịch vụ backend. Vui lòng kiểm tra Docker.*"
        )

    await cl.Message(content=welcome_message).send()


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

    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                try:
                    file_path = Path(element.path)
                    if file_path.is_file():
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

    if not full_input:
        await cl.Message(content="⚠️ Vui lòng nhập câu hỏi hoặc đính kèm file log sự cố.").send()
        return

    # 2. Tự động nhận diện sự cố (Incident RCA) hay Hỏi đáp kiến trúc (Codebase QA)
    is_stack_trace = _detect_is_stack_trace(full_input)
    task_mode = "incident_rca" if is_stack_trace else "codebase_qa"

    status_badge = (
        "🚨 **Chế độ Chẩn đoán Sự cố (Incident RCA)**"
        if is_stack_trace
        else "🔍 **Chế độ Hỏi đáp Kiến trúc Codebase (Codebase Q&A)**"
    )

    # 3. Chuẩn bị state ban đầu cho đồ thị LangGraph
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
    }

    current_state: dict[str, Any] = dict(initial_state)

    # 4. Trực quan hóa tiến trình suy luận từng bước (Step Progression) của LangGraph
    try:
        # Chạy stream qua từng Node trong StateGraph
        for step_event in compiled_workflow.stream(initial_state):
            for node_name, node_update in step_event.items():
                current_state.update(node_update)

                if node_name == "classify_and_parse":
                    async with cl.Step(
                        name="1. Phân loại Ý định & Bóc tách Cú pháp Log",
                        type="tool",
                    ) as step:
                        ttype = current_state.get("task_type", "codebase_qa")
                        sig = current_state.get("error_signature")
                        syms = current_state.get("extracted_symbols", [])
                        sym_str = ", ".join(f"`{s}`" for s in syms) if syms else "Không có"
                        step.output = (
                            f"- **Phân loại tác vụ:** `{ttype}`\n"
                            f"- **Chữ ký ngoại lệ / Lỗi:** `{sig or 'Không có'}`\n"
                            f"- **Thực thể phát hiện:** {sym_str}"
                        )

                elif node_name == "multi_source_retrieval":
                    async with cl.Step(
                        name="2. Truy xuất Đa nguồn Song song (Code, Git, Runbook)",
                        type="retrieval",
                    ) as step:
                        c_cnt = len(current_state.get("code_contexts", []))
                        i_cnt = len(current_state.get("incident_contexts", []))
                        step.output = (
                            f"- 📁 **Mã nguồn (Code Chunks):** {c_cnt} đoạn mã phù hợp\n"
                            f"- 📚 **Lịch sử Git & Runbooks:** {i_cnt} bản ghi liên quan"
                        )

                elif node_name == "grade_context":
                    async with cl.Step(
                        name="3. Đánh giá Mức độ Đầy đủ của Ngữ cảnh",
                        type="llm",
                    ) as step:
                        is_suff = current_state.get("is_sufficient", True)
                        iter_num = current_state.get("iteration_count", 0)
                        suff_badge = (
                            "✅ Ngữ cảnh đã đầy đủ" if is_suff else "⚠️ Ngữ cảnh còn thiếu"
                        )
                        step.output = (
                            f"- **Độ đầy đủ:** {suff_badge}\n"
                            f"- **Số vòng lặp Call Graph:** {iter_num}/3"
                        )

                elif node_name == "expand_dependencies":
                    async with cl.Step(
                        name="4. Mở rộng Phụ thuộc qua Call Graph (Callers / Callees)",
                        type="tool",
                    ) as step:
                        c_cnt = len(current_state.get("code_contexts", []))
                        iter_num = current_state.get("iteration_count", 0)
                        step.output = (
                            f"- Đã duyệt Call Graph bổ sung các hàm liên quan.\n"
                            f"- **Tổng số ngữ cảnh mã nguồn hiện tại:** {c_cnt} (vòng {iter_num})"
                        )

                elif node_name == "generate_answer":
                    async with cl.Step(
                        name="5. Sinh Phản hồi & Báo cáo kèm Trích dẫn Dòng",
                        type="llm",
                    ) as step:
                        step.output = "LLM đã hoàn thành việc sinh nội dung kèm trích dẫn."

                elif node_name == "validate_citations":
                    async with cl.Step(
                        name="6. Kiểm định Trích dẫn Chống Ảo giác",
                        type="tool",
                    ) as step:
                        valid = current_state.get("citation_valid", True)
                        err = current_state.get("citation_error")
                        if valid:
                            step.output = (
                                "✅ Mọi trích dẫn `[file_path:L_start-L_end]` đều hợp lệ."
                            )
                        else:
                            step.output = f"⚠️ Phát hiện số dòng ngoài phạm vi, đã sửa chữa:\n{err}"

    except Exception as exc:
        logger.error(f"Lỗi trong quá trình chạy LangGraph stream: {exc}", exc_info=True)
        # Fallback gọi invoke thông thường nếu stream gặp trục trặc
        try:
            current_state = compiled_workflow.invoke(initial_state)
        except Exception as inner_exc:
            await cl.Message(
                content=f"❌ **Đã xảy ra lỗi khi thực thi xử lý:** {inner_exc}"
            ).send()
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

    response_text = f"{status_badge}\n\n---\n\n{highlighted_response}"

    await cl.Message(
        content=response_text,
        elements=elements if elements else None,
    ).send()

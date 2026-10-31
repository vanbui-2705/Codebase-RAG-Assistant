"""
Module: src.agent.workflow
Chức năng: Xây dựng và biên dịch đồ thị điều phối tác tử (LangGraph StateGraph).
Điều phối quy trình phân loại ý định, truy xuất đa nguồn song song, đánh giá độ đầy đủ,
mở rộng Call Graph, sinh câu trả lời kèm trích dẫn bắt buộc và kiểm định chống ảo giác.
Tuân thủ Clean Architecture: Nhận AgentTools và LLM qua Dependency Injection.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, runtime_checkable

from langgraph.graph import END, START, StateGraph

from src.agent.prompts import (
    CITATION_REPROMPT_TEMPLATE,
    CLASSIFIER_PROMPT,
    GRADER_PROMPT,
    SYSTEM_PROMPT,
)
from src.agent.state import AgentGraphState
from src.domain.schemas import IncidentDocNode, RetrievedContext

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentTools(Protocol):
    """
    Giao diện công cụ tra cứu cho Agent (Agent Tools Protocol).
    Tuân thủ Dependency Inversion (Clean Architecture).
    """

    def search_codebase_tool(
        self,
        query: str,
        file_path: str | None = None,
        symbol_name: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """Tìm kiếm Hybrid trên collection code_chunks."""
        ...

    def search_git_history_tool(
        self,
        file_path: str | None = None,
        limit: int = 5,
    ) -> list[IncidentDocNode] | list[RetrievedContext]:
        """Tìm kiếm lịch sử commit và PR."""
        ...

    def search_runbooks_tool(
        self,
        error_signature: str,
        top_k: int = 3,
    ) -> list[RetrievedContext]:
        """Tìm kiếm tài liệu runbook trên collection incident_and_docs."""
        ...

    def get_callers_and_callees_tool(
        self,
        symbol_name: str,
        file_path: str | None = None,
    ) -> dict[str, list[str]] | list[RetrievedContext]:
        """Tra cứu đồ thị gọi hàm (Call Graph) để tìm callers và callees."""
        ...

    def parse_stack_trace(
        self,
        error_log: str,
    ) -> dict[str, Any]:
        """Bóc tách cấu trúc stack trace thành các frame và ngoại lệ."""
        ...


class OnCallAgentWorkflow:
    """
    Điều phối luồng trạng thái (StateGraph) của On-call & Codebase RAG Assistant.
    Bao gồm 6 nodes xử lý tuần tự/rẽ nhánh và 2 conditional edges kiểm soát chất lượng.
    """

    def __init__(
        self,
        tools: AgentTools,
        llm: Any,
        max_iterations: int = 3,
        max_citation_retries: int = 2,
    ) -> None:
        """
        Khởi tạo workflow với Dependency Injection.

        Args:
            tools: Đối tượng cung cấp các công cụ tra cứu mã nguồn, git và runbook.
            llm: Instance mô hình ngôn ngữ (Google Gemini hoặc LangChain wrapper).
            max_iterations: Số lần tối đa cho phép mở rộng Call Graph (mặc định 3).
            max_citation_retries: Số lần tối đa cho phép LLM sinh lại khi trích dẫn sai dải dòng.
        """
        self.tools = tools
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_citation_retries = max_citation_retries
        self._citation_retries = 0

    # =========================================================================
    # CÁC NODE XỬ LÝ (GRAPH NODES)
    # =========================================================================

    def classify_and_parse_node(self, state: AgentGraphState) -> dict[str, Any]:
        """
        Node 1: Phân loại ý định người dùng (codebase_qa vs incident_rca)
        kết hợp bóc tách Regex log/stack trace nếu có.
        """
        raw_input = state.get("raw_input", "").strip()
        logger.info(
            f"Node 1 [classify_and_parse]: Bắt đầu phân loại input (độ dài: {len(raw_input)} ký tự)"
        )

        # 1. Bóc tách qua công cụ tools.parse_stack_trace nếu có
        extracted_sig: str | None = None
        extracted_symbols: list[str] = []
        is_incident = False

        if hasattr(self.tools, "parse_stack_trace"):
            try:
                parsed_trace = self.tools.parse_stack_trace(raw_input)
                if isinstance(parsed_trace, dict):
                    sig = parsed_trace.get("error_signature")
                    frames = parsed_trace.get("frames", [])
                    if frames or (
                        sig and sig != "RuntimeError: Lỗi không xác định" and sig != "UnknownError"
                    ):
                        is_incident = True
                        extracted_sig = sig
                        if parsed_trace.get("file_path"):
                            extracted_symbols.append(str(parsed_trace["file_path"]))
                        if parsed_trace.get("function_name") and parsed_trace[
                            "function_name"
                        ] not in ("<module>", "<unknown>"):
                            extracted_symbols.append(str(parsed_trace["function_name"]))
                        for f in frames:
                            if isinstance(f, dict):
                                if f.get("file_path"):
                                    extracted_symbols.append(str(f["file_path"]))
                                if f.get("function_name") and f["function_name"] not in (
                                    "<module>",
                                    "<unknown>",
                                ):
                                    extracted_symbols.append(str(f["function_name"]))
            except Exception as e:
                logger.warning(f"Lỗi khi gọi tools.parse_stack_trace: {e}")

        # 2. Bổ sung kiểm tra Regex nội bộ nếu tools chưa xác định được
        if not is_incident:
            reg_incident, reg_sig, reg_symbols = self._regex_parse_log(raw_input)
            if reg_incident:
                is_incident = True
                extracted_sig = reg_sig or extracted_sig
                extracted_symbols.extend(reg_symbols)

        # 3. Nếu Regex/Tools xác định đây là sự cố có Stack trace
        if is_incident:
            unique_symbols = list(dict.fromkeys(s.strip() for s in extracted_symbols if s.strip()))
            logger.info(
                f"Phát hiện sự cố On-call: signature='{extracted_sig}', symbols={unique_symbols}"
            )
            return {
                "task_type": "incident_rca",
                "error_signature": extracted_sig,
                "extracted_symbols": unique_symbols,
                "iteration_count": 0,
                "is_sufficient": False,
                "citation_valid": True,
                "citation_error": None,
            }

        # 4. Ngược lại, gọi LLM Classifier để phân loại ngữ nghĩa câu hỏi
        prompt = (
            f"{CLASSIFIER_PROMPT}\n\n"
            f'Văn bản cần phân loại:\n"""\n{raw_input}\n"""\n\n'
            f"Trả về kết quả định dạng JSON:"
        )

        try:
            llm_output = self._invoke_llm(prompt)
            parsed_json = self._extract_json(llm_output)

            task_type = "codebase_qa"
            error_signature = None
            symbols_from_llm: list[str] = []

            if parsed_json and isinstance(parsed_json, dict):
                task_type = parsed_json.get("task_type", "codebase_qa")
                if task_type not in ("incident_rca", "codebase_qa"):
                    task_type = "codebase_qa"
                error_signature = parsed_json.get("error_signature") or None
                raw_syms = parsed_json.get("extracted_symbols", [])
                if isinstance(raw_syms, list):
                    symbols_from_llm = [str(s).strip() for s in raw_syms if str(s).strip()]

            combined_symbols = list(dict.fromkeys(extracted_symbols + symbols_from_llm))
            logger.info(
                f"LLM Classifier xác định: task_type='{task_type}', "
                f"symbols={combined_symbols}, error_sig='{error_signature}'"
            )

            return {
                "task_type": task_type,
                "error_signature": error_signature or extracted_sig,
                "extracted_symbols": combined_symbols,
                "iteration_count": 0,
                "is_sufficient": False,
                "citation_valid": True,
                "citation_error": None,
            }
        except Exception as e:
            logger.warning(f"Lỗi khi gọi LLM Classifier ({e}). Fallback về codebase_qa.")
            return {
                "task_type": "codebase_qa",
                "error_signature": None,
                "extracted_symbols": list(dict.fromkeys(extracted_symbols)),
                "iteration_count": 0,
                "is_sufficient": False,
                "citation_valid": True,
                "citation_error": None,
            }

    def multi_source_retrieval_node(self, state: AgentGraphState) -> dict[str, Any]:
        """
        Node 2: Truy xuất đa nguồn song song (Codebase chunks, Git history, và Runbooks).
        """
        logger.info(
            "Node 2 [multi_source_retrieval]: Đang truy xuất song song Code, Git và Runbooks..."
        )

        code_contexts: list[RetrievedContext] = []
        git_contexts: list[RetrievedContext] = []
        runbook_contexts: list[RetrievedContext] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            code_future = executor.submit(self._retrieve_code, state)
            git_future = executor.submit(self._retrieve_git, state)
            runbook_future = executor.submit(self._retrieve_runbooks, state)

            try:
                code_contexts = code_future.result()
            except Exception as e:
                logger.error(f"Lỗi truy xuất Code chunks: {e}")

            try:
                git_contexts = git_future.result()
            except Exception as e:
                logger.error(f"Lỗi truy xuất Git history: {e}")

            try:
                runbook_contexts = runbook_future.result()
            except Exception as e:
                logger.error(f"Lỗi truy xuất Runbooks: {e}")

        # Gộp các tài liệu liên quan đến sự cố (Git + Runbooks)
        incident_contexts = git_contexts + runbook_contexts

        logger.info(
            f"Hoàn thành truy xuất: {len(code_contexts)} code chunks, "
            f"{len(incident_contexts)} incident/doc contexts."
        )

        return {
            "code_contexts": code_contexts,
            "incident_contexts": incident_contexts,
        }

    def grade_context_node(self, state: AgentGraphState) -> dict[str, Any]:
        """
        Node 3: Đánh giá xem ngữ cảnh mã nguồn đã đủ để giải thích luồng hoặc nguyên nhân chưa.
        Nếu thiếu các hàm liên quan trong Call Graph, đề xuất bổ sung symbol.
        """
        code_contexts = state.get("code_contexts", [])
        raw_input = state.get("raw_input", "")
        error_signature = state.get("error_signature")
        existing_symbols = list(state.get("extracted_symbols", []))

        logger.info(
            f"Node 3 [grade_context]: Đang chấm điểm độ đầy đủ ngữ cảnh "
            f"({len(code_contexts)} chunks, {len(existing_symbols)} symbols)..."
        )

        # Nếu không có code context nào được tìm thấy
        if not code_contexts:
            logger.info("Ngữ cảnh mã nguồn trống -> Cần mở rộng hoặc fallback.")
            return {
                "is_sufficient": False,
                "extracted_symbols": existing_symbols,
            }

        # Tạo bản tóm tắt các chunk hiện có cho Grader LLM
        context_summary_lines = []
        for ctx in code_contexts:
            lines = (
                f"L{ctx.start_line}-L{ctx.end_line}"
                if ctx.start_line is not None
                else "unknown lines"
            )
            sym = f" ({ctx.symbol_name})" if ctx.symbol_name else ""
            context_summary_lines.append(f"- [{ctx.file_path}:{lines}]{sym}")

        context_summary = "\n".join(context_summary_lines)

        prompt = (
            f"{GRADER_PROMPT}\n\n"
            f"CÂU HỎI / SỰ CỐ GỐC:\n{raw_input}\n"
            f"ERROR SIGNATURE: {error_signature or 'N/A'}\n\n"
            f"DANH SÁCH MÃ NGUỒN HIỆN CÓ TRONG CONTEXT:\n"
            f"{context_summary}\n\n"
            f"Trả về kết quả dưới dạng JSON:"
        )

        try:
            llm_output = self._invoke_llm(prompt)
            parsed_json = self._extract_json(llm_output)

            is_sufficient = True
            missing_symbols: list[str] = []

            if parsed_json and isinstance(parsed_json, dict):
                is_sufficient = bool(parsed_json.get("is_sufficient", True))
                raw_missing = parsed_json.get("missing_symbols", [])
                if isinstance(raw_missing, list):
                    missing_symbols = [str(s).strip() for s in raw_missing if str(s).strip()]

            new_symbols = list(dict.fromkeys(existing_symbols + missing_symbols))
            logger.info(
                f"Kết quả Grader: is_sufficient={is_sufficient}, missing_symbols={missing_symbols}"
            )

            return {
                "is_sufficient": is_sufficient,
                "extracted_symbols": new_symbols,
            }
        except Exception as e:
            logger.warning(f"Lỗi khi gọi Grader LLM ({e}). Mặc định coi ngữ cảnh là đủ.")
            return {
                "is_sufficient": True,
                "extracted_symbols": existing_symbols,
            }

    def expand_dependencies_node(self, state: AgentGraphState) -> dict[str, Any]:
        """
        Node 4: Kéo thêm các hàm/class phụ thuộc trong Call Graph nếu thiếu (tăng iteration_count).
        """
        iteration_count = state.get("iteration_count", 0) + 1
        current_code_contexts = list(state.get("code_contexts", []))
        symbols = state.get("extracted_symbols", [])

        logger.info(
            f"Node 4 [expand_dependencies]: Lần lặp {iteration_count}/{self.max_iterations}. "
            f"Đang mở rộng Call Graph cho {len(symbols)} symbols..."
        )

        existing_node_ids = {ctx.node_id for ctx in current_code_contexts}
        new_contexts: list[RetrievedContext] = []

        for symbol in symbols:
            try:
                # 1. Gọi get_callers_and_callees_tool nếu có
                if hasattr(self.tools, "get_callers_and_callees_tool"):
                    cg_result = self.tools.get_callers_and_callees_tool(symbol)
                    if isinstance(cg_result, dict):
                        # Trích xuất danh sách callers và callees
                        related = cg_result.get("callers", []) + cg_result.get("callees", [])
                        for rel_sym in related[:3]:
                            extra = self.tools.search_codebase_tool(
                                query=rel_sym, symbol_name=rel_sym, top_k=2
                            )
                            for c in extra:
                                if c.node_id not in existing_node_ids:
                                    existing_node_ids.add(c.node_id)
                                    new_contexts.append(c)
                    elif isinstance(cg_result, list):
                        for c in cg_result:
                            if (
                                isinstance(c, RetrievedContext)
                                and c.node_id not in existing_node_ids
                            ):
                                existing_node_ids.add(c.node_id)
                                new_contexts.append(c)
                # 2. Hoặc tìm kiếm trực tiếp symbol bằng search_codebase_tool
                elif hasattr(self.tools, "search_codebase_tool"):
                    extra = self.tools.search_codebase_tool(
                        query=symbol, symbol_name=symbol, top_k=2
                    )
                    for c in extra:
                        if c.node_id not in existing_node_ids:
                            existing_node_ids.add(c.node_id)
                            new_contexts.append(c)
            except Exception as e:
                logger.warning(f"Lỗi mở rộng phụ thuộc cho symbol '{symbol}': {e}")

        logger.info(f"Đã mở rộng thêm {len(new_contexts)} chunks mới từ Call Graph.")

        return {
            "code_contexts": current_code_contexts + new_contexts,
            "iteration_count": iteration_count,
        }

    def generate_answer_node(self, state: AgentGraphState) -> dict[str, Any]:
        """
        Node 5: Gọi LLM sinh báo cáo RCA hoặc giải thích QA kèm trích dẫn bắt buộc.
        Nếu trước đó có lỗi trích dẫn (citation_error), đưa thêm cảnh báo để LLM tự sửa.
        """
        task_type = state.get("task_type", "codebase_qa")
        raw_input = state.get("raw_input", "")
        code_contexts = state.get("code_contexts", [])
        incident_contexts = state.get("incident_contexts", [])
        citation_error = state.get("citation_error")

        logger.info(
            f"Node 5 [generate_answer]: Bắt đầu sinh câu trả lời cho task '{task_type}' "
            f"({len(code_contexts)} code chunks, {len(incident_contexts)} docs)..."
        )

        # Định dạng ngữ cảnh đưa vào LLM
        formatted_context = self._format_contexts(code_contexts, incident_contexts)

        # Xây dựng prompt
        prompt_parts = [
            SYSTEM_PROMPT,
            "\n========================",
            "NGỮ CẢNH HỆ THỐNG ĐƯỢC CUNG CẤP (CONTEXT)",
            "========================",
            formatted_context,
            "\n========================",
            f"YÊU CẦU CỦA NGƯỜI DÙNG (TASK = {task_type.upper()}):",
            "========================",
            raw_input,
        ]

        # Nếu có cảnh báo trích dẫn sai dải dòng từ lần sinh trước
        if citation_error:
            logger.info(
                "Phát hiện citation_error từ lần trước, bổ sung CITATION_REPROMPT_TEMPLATE."
            )
            reprompt = CITATION_REPROMPT_TEMPLATE.format(invalid_citations=citation_error)
            prompt_parts.extend(["\n========================", reprompt])

        full_prompt = "\n".join(prompt_parts)

        try:
            response_text = self._invoke_llm(full_prompt)
        except Exception as e:
            logger.error(f"Lỗi khi gọi LLM sinh câu trả lời: {e}")
            response_text = (
                f"Đã xảy ra lỗi trong quá trình sinh câu trả lời từ mô hình ngôn ngữ. Chi tiết: {e}"
            )

        return {
            "final_response": response_text,
            "citation_error": None,
        }

    def validate_citations_node(self, state: AgentGraphState) -> dict[str, Any]:
        """
        Node 6: Regex kiểm tra toàn bộ trích dẫn mã nguồn [file_path:Lstart-Lend]
        trong final_response xem có tồn tại và nằm trong dải dòng của context không.
        """
        final_response = state.get("final_response", "")
        code_contexts = state.get("code_contexts", [])

        logger.info("Node 6 [validate_citations]: Đang kiểm định trích dẫn chống ảo giác...")

        # Nếu không có code context và LLM trả lời "Không tìm thấy trong codebase hiện tại"
        if not code_contexts and "Không tìm thấy trong codebase hiện tại" in final_response:
            logger.info("Không có code context và LLM báo đúng theo guardrail -> Trích dẫn hợp lệ.")
            return {
                "citation_valid": True,
                "citation_error": None,
            }

        # Regex tìm các citation dạng [file_path:Lstart-Lend] hoặc [file_path:Lstart]
        citation_pattern = re.compile(
            r"\[([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+):L(\d+)(?:-L?(\d+))?\]"
        )
        matches = citation_pattern.findall(final_response)

        invalid_citations: list[str] = []

        for match in matches:
            cited_file, start_str, end_str = match
            try:
                c_start = int(start_str)
                c_end = int(end_str) if end_str else c_start
            except ValueError:
                invalid_citations.append(
                    f"[{cited_file}:{start_str}-{end_str}] (số dòng không hợp lệ)"
                )
                continue

            if c_end < c_start:
                invalid_citations.append(
                    f"[{cited_file}:L{c_start}-L{c_end}] (dòng kết thúc nhỏ hơn dòng bắt đầu)"
                )
                continue

            # Kiểm tra xem có chunk nào trong code_contexts chứa dải dòng này không
            is_covered = False
            for ctx in code_contexts:
                if self._matches_path(cited_file, ctx.file_path):
                    if ctx.start_line is not None and ctx.end_line is not None:
                        if ctx.start_line <= c_start and c_end <= ctx.end_line:
                            is_covered = True
                            break

            if not is_covered:
                invalid_citations.append(
                    f"[{cited_file}:L{c_start}-L{c_end}] (vượt ngoài dải dòng thực tế được cấp)"
                )

        if invalid_citations:
            error_msg = "\n".join(f"- {c}" for c in invalid_citations)
            logger.warning(f"Phát hiện trích dẫn sai dải dòng:\n{error_msg}")
            return {
                "citation_valid": False,
                "citation_error": error_msg,
            }

        logger.info(f"Kiểm định hoàn tất: {len(matches)} trích dẫn đều hợp lệ.")
        return {
            "citation_valid": True,
            "citation_error": None,
        }

    # =========================================================================
    # CONDITIONAL EDGES
    # =========================================================================

    def _should_expand_or_generate(self, state: AgentGraphState) -> str:
        """
        Conditional Edge 1 (sau grade_context):
        Nếu thiếu ngữ cảnh và iteration_count < 3 -> sang 'expand_dependencies',
        ngược lại -> 'generate_answer'.
        """
        is_sufficient = state.get("is_sufficient", True)
        iteration_count = state.get("iteration_count", 0)

        if not is_sufficient and iteration_count < self.max_iterations:
            logger.info(
                f"Conditional Edge 1: Ngữ cảnh chưa đủ "
                f"(lần lặp {iteration_count}/{self.max_iterations}). "
                f"Rẽ nhánh sang 'expand_dependencies'."
            )
            return "expand_dependencies"

        logger.info(
            "Conditional Edge 1: Ngữ cảnh đã đủ hoặc đạt giới hạn iterations. "
            "Rẽ nhánh sang 'generate_answer'."
        )
        return "generate_answer"

    def _check_citation_validity(self, state: AgentGraphState) -> str:
        """
        Conditional Edge 2 (sau validate_citations):
        Nếu citation hợp lệ -> END,
        nếu bịa số dòng -> quay lại 'generate_answer' kèm citation_error.
        """
        citation_valid = state.get("citation_valid", True)

        if citation_valid:
            logger.info("Conditional Edge 2: Trích dẫn hợp lệ. Kết thúc đồ thị (END).")
            self._citation_retries = 0
            return END

        if self._citation_retries >= self.max_citation_retries:
            logger.warning(
                f"Conditional Edge 2: Đã đạt giới hạn số lần sửa trích dẫn "
                f"({self.max_citation_retries}). Dừng vòng lặp chuyển sang END."
            )
            self._citation_retries = 0
            return END

        self._citation_retries += 1
        logger.warning(
            f"Conditional Edge 2: Trích dẫn không hợp lệ. Lần sửa thứ {self._citation_retries}/"
            f"{self.max_citation_retries}. Quay lại 'generate_answer'."
        )
        return "generate_answer"

    # =========================================================================
    # PHƯƠNG THỨC BIÊN DỊCH ĐỒ THỊ (COMPILE GRAPH)
    # =========================================================================

    def compile_graph(self):
        """
        Kết nối các Node và Cạnh điều hướng thành StateGraph hoàn chỉnh và biên dịch.

        Returns:
            CompiledStateGraph: Đồ thị trạng thái LangGraph đã sẵn sàng thực thi.
        """
        workflow = StateGraph(AgentGraphState)

        # 1. Khai báo 6 Nodes
        workflow.add_node("classify_and_parse", self.classify_and_parse_node)
        workflow.add_node("multi_source_retrieval", self.multi_source_retrieval_node)
        workflow.add_node("grade_context", self.grade_context_node)
        workflow.add_node("expand_dependencies", self.expand_dependencies_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("validate_citations", self.validate_citations_node)

        # 2. Khai báo các Cạnh cố định
        workflow.add_edge(START, "classify_and_parse")
        workflow.add_edge("classify_and_parse", "multi_source_retrieval")
        workflow.add_edge("multi_source_retrieval", "grade_context")
        workflow.add_edge("expand_dependencies", "grade_context")
        workflow.add_edge("generate_answer", "validate_citations")

        # 3. Khai báo Conditional Edges
        # Edge 1: sau grade_context
        workflow.add_conditional_edges(
            "grade_context",
            self._should_expand_or_generate,
            {
                "expand_dependencies": "expand_dependencies",
                "generate_answer": "generate_answer",
            },
        )

        # Edge 2: sau validate_citations
        workflow.add_conditional_edges(
            "validate_citations",
            self._check_citation_validity,
            {
                "generate_answer": "generate_answer",
                END: END,
            },
        )

        return workflow.compile()

    # =========================================================================
    # HÀM BỔ TRỢ NỘI BỘ (INTERNAL HELPERS)
    # =========================================================================

    def _invoke_llm(self, prompt: str) -> str:
        """Thực thi gọi mô hình ngôn ngữ với hỗ trợ cả invoke lẫn callable wrapper."""
        if hasattr(self.llm, "invoke"):
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        elif callable(self.llm):
            response = self.llm(prompt)
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        raise TypeError(
            f"Mô hình LLM được truyền vào không hỗ trợ invoke hoặc call: {type(self.llm)}"
        )

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Trích xuất và parse JSON từ chuỗi văn bản (kể cả khi nằm trong Markdown code block)."""
        clean_text = text.strip()
        # Tìm khối ```json ... ```
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if json_match:
            clean_text = json_match.group(1).strip()

        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Tìm chuỗi { ... } lớn nhất nếu json.loads trực tiếp thất bại
        brace_match = re.search(r"(\{[\s\S]*\})", clean_text)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        return None

    def _regex_parse_log(self, text: str) -> tuple[bool, str | None, list[str]]:
        """
        Bóc tách nhanh log/stack trace qua biểu thức chính quy Regex.
        Returns: (is_incident, error_signature, extracted_symbols)
        """
        symbols: list[str] = []
        error_sig: str | None = None

        # 1. Kiểm tra dấu hiệu Stack trace
        has_trace_header = bool(
            re.search(r"Traceback \(most recent call last\):", text, re.IGNORECASE)
            or re.search(r"Exception in thread", text, re.IGNORECASE)
            or re.search(r"goroutine \d+ \[running\]:", text)
        )

        # 2. Tìm ngoại lệ / Error
        exc_match = re.search(
            r"([A-Za-z0-9_]+(?:Error|Exception|Failure|Fault|Panic)):?\s*([^\r\n]*)",
            text,
        )
        if exc_match:
            err_type = exc_match.group(1).strip()
            err_msg = exc_match.group(2).strip()
            error_sig = f"{err_type}: {err_msg}" if err_msg else err_type

        # 3. Tìm frames: File ".../file.py", line 123, in func_name
        py_frames = re.findall(r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+([A-Za-z0-9_]+)', text)
        for fpath, _, fn in py_frames:
            symbols.append(fpath)
            symbols.append(fn)

        # 4. Tìm frames: at package.Class.method(File.java:123)
        java_frames = re.findall(r"at\s+([A-Za-z0-9_\.\$]+)\(([^:]+):(\d+)\)", text)
        for full_sym, fpath, _ in java_frames:
            symbols.append(full_sym)
            symbols.append(fpath)

        # 5. Tìm dạng path/file.py:123
        path_frames = re.findall(r"([a-zA-Z0-9_\-\./\\]+\.(?:py|ts|js|go|java)):(\d+)", text)
        for fpath, _ in path_frames:
            symbols.append(fpath)

        is_incident = has_trace_header or (error_sig is not None and len(symbols) > 0)
        unique_symbols = list(dict.fromkeys(s.strip() for s in symbols if s.strip()))

        return is_incident, error_sig, unique_symbols

    def _retrieve_code(self, state: AgentGraphState) -> list[RetrievedContext]:
        """Gọi công cụ tìm kiếm Codebase chunks."""
        query = state.get("raw_input", "")
        symbols = state.get("extracted_symbols", [])
        primary_file = next((s for s in symbols if "." in s and "/" in s), None)
        primary_symbol = next((s for s in symbols if "." not in s or "/" not in s), None)

        if hasattr(self.tools, "search_codebase_tool"):
            return self.tools.search_codebase_tool(
                query=query,
                file_path=primary_file,
                symbol_name=primary_symbol,
            )
        return []

    def _retrieve_git(self, state: AgentGraphState) -> list[RetrievedContext]:
        """Gọi công cụ tìm kiếm Git history (Commits và PRs)."""
        symbols = state.get("extracted_symbols", [])
        primary_file = next(
            (
                s
                for s in symbols
                if any(s.endswith(ext) for ext in (".py", ".ts", ".js", ".go", ".java"))
            ),
            None,
        )

        raw_commits: list[Any] = []
        if hasattr(self.tools, "search_git_history_tool"):
            raw_commits = self.tools.search_git_history_tool(file_path=primary_file, limit=5)

        git_contexts: list[RetrievedContext] = []
        for item in raw_commits:
            if isinstance(item, RetrievedContext):
                git_contexts.append(item)
            elif isinstance(item, IncidentDocNode):
                git_contexts.append(
                    RetrievedContext(
                        node_id=item.doc_id,
                        source_type=item.source_type,
                        file_path=item.related_files[0] if item.related_files else "",
                        content=item.content,
                        score=1.0,
                        metadata={
                            "title": item.title,
                            "commit_sha": item.commit_sha,
                            "author": item.author,
                            "timestamp": item.timestamp,
                            "url": item.url,
                            "related_files": item.related_files,
                        },
                    )
                )

        return git_contexts

    def _retrieve_runbooks(self, state: AgentGraphState) -> list[RetrievedContext]:
        """Gọi công cụ tìm kiếm Runbooks."""
        query = state.get("error_signature") or state.get("raw_input", "")

        if hasattr(self.tools, "search_runbooks_tool"):
            return self.tools.search_runbooks_tool(error_signature=query)
        return []

    def _format_contexts(
        self,
        code_contexts: list[RetrievedContext],
        incident_contexts: list[RetrievedContext],
    ) -> str:
        """Định dạng các context thành văn bản sạch cho LLM kèm thông tin số dòng để trích dẫn."""
        if not code_contexts and not incident_contexts:
            return "Không tìm thấy đoạn mã hoặc tài liệu liên quan trong hệ thống."

        sections: list[str] = []

        if code_contexts:
            sections.append("### 1. CODE CHUNKS (MÃ NGUỒN)")
            for i, ctx in enumerate(code_contexts, 1):
                start = ctx.start_line if ctx.start_line is not None else 1
                end = ctx.end_line if ctx.end_line is not None else start
                sym = f" | Symbol: {ctx.symbol_name}" if ctx.symbol_name else ""
                header = f"[{ctx.file_path}:L{start}-L{end}]{sym}"
                sections.append(f"--- Code Chunk #{i}: {header} ---\n{ctx.content}\n")

        if incident_contexts:
            sections.append("### 2. INCIDENT & RUNBOOK CONTEXTS (TÀI LIỆU & LỊCH SỬ GIT)")
            for i, ctx in enumerate(incident_contexts, 1):
                stype = ctx.source_type.upper()
                doc_title = ctx.metadata.get("title", ctx.node_id)
                header = f"[{stype}: {ctx.file_path}] (Tiêu đề: {doc_title})"
                sections.append(f"--- Doc #{i}: {header} ---\n{ctx.content}\n")

        return "\n".join(sections)

    def _matches_path(self, cited_path: str, actual_path: str) -> bool:
        """So khớp đường dẫn file giữa trích dẫn và context thực tế (bỏ qua slash OS)."""
        clean_cited = cited_path.replace("\\", "/").strip("/").lower()
        clean_actual = actual_path.replace("\\", "/").strip("/").lower()

        if clean_cited == clean_actual:
            return True
        if clean_actual.endswith(clean_cited):
            return True
        if clean_cited.endswith(clean_actual):
            return True
        return False

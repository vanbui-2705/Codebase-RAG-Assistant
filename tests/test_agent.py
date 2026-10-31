"""
Unit tests cho Tầng Agent (Giai đoạn 3):
Kiểm thử AgentGraphState, OnCallAgentWorkflow (6 nodes, 2 conditional edges)
và quá trình biên dịch, thực thi đồ thị StateGraph hoàn chỉnh.
"""

from typing import Any
from unittest.mock import MagicMock

from langgraph.graph import END

from src.agent.state import AgentGraphState
from src.agent.workflow import OnCallAgentWorkflow
from src.domain.schemas import IncidentDocNode, RetrievedContext


class MockAgentTools:
    """Mock AgentTools phục vụ unit test cho workflow."""

    def __init__(self):
        self.code_results = [
            RetrievedContext(
                node_id="chunk_auth_1",
                source_type="code",
                file_path="src/auth/jwt.py",
                content=(
                    "def verify_jwt(token: str) -> bool:\n"
                    "    return token.startswith('bearer_')"
                ),
                score=0.95,
                start_line=10,
                end_line=20,
                symbol_name="verify_jwt",
            )
        ]
        self.git_results = [
            IncidentDocNode(
                doc_id="commit_sha_123",
                source_type="commit",
                title="Commit: Update JWT validation",
                content="Commit SHA: a1b2c3d\nAuthor: dev@vinai.ai\nUpdate token verification",
                related_files=["src/auth/jwt.py"],
                commit_sha="a1b2c3d",
                author="dev@vinai.ai",
                timestamp="2026-10-30T10:00:00Z",
            )
        ]
        self.runbook_results = [
            RetrievedContext(
                node_id="runbook_auth_err",
                source_type="runbook",
                file_path="docs/runbooks/jwt_auth_failure.md",
                content=(
                    "# Runbook xử lý lỗi xác thực JWT\n"
                    "Kiểm tra JWT_SECRET và khởi động lại auth-service."
                ),
                score=0.88,
                metadata={"title": "JWT Auth Failure Runbook"},
            )
        ]

    def search_codebase_tool(
        self,
        query: str,
        file_path: str | None = None,
        symbol_name: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        return self.code_results

    def search_git_history_tool(
        self, file_path: str | None = None, limit: int = 5
    ) -> list[IncidentDocNode]:
        return self.git_results

    def search_runbooks_tool(self, error_signature: str, top_k: int = 3) -> list[RetrievedContext]:
        return self.runbook_results

    def get_callers_and_callees_tool(
        self, symbol_name: str, file_path: str | None = None
    ) -> dict[str, list[str]]:
        return {"callers": ["authenticate_request"], "callees": ["decode_base64"]}

    def parse_stack_trace(self, error_log: str) -> dict[str, Any]:
        if "ZeroDivisionError" in error_log:
            return {
                "exception_type": "ZeroDivisionError",
                "exception_message": "division by zero",
                "error_signature": "ZeroDivisionError: division by zero",
                "file_path": "src/math/calc.py",
                "line_number": 42,
                "function_name": "divide",
                "frames": [
                    {"file_path": "src/math/calc.py", "line_number": 42, "function_name": "divide"}
                ],
            }
        return {
            "exception_type": "UnknownError",
            "exception_message": "",
            "error_signature": "",
            "file_path": None,
            "line_number": None,
            "function_name": None,
            "frames": [],
        }


class MockLLM:
    """Mock LLM hỗ trợ invoke trả về phản hồi theo kịch bản."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.last_prompt = ""

    def invoke(self, prompt: Any) -> MagicMock:
        self.call_count += 1
        self.last_prompt = str(prompt)
        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            content = self.responses[idx]
        else:
            content = "Default mock answer [src/auth/jwt.py:L10-L20]"
        mock_resp = MagicMock()
        mock_resp.content = content
        return mock_resp


def test_agent_graph_state_creation():
    """Kiểm tra khởi tạo AgentGraphState đúng kiểu dữ liệu."""
    state: AgentGraphState = {
        "raw_input": "Hàm verify_jwt hoạt động thế nào?",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": ["verify_jwt"],
        "code_contexts": [],
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }
    assert state["task_type"] == "codebase_qa"
    assert state["iteration_count"] == 0
    assert len(state["extracted_symbols"]) == 1


def test_classify_and_parse_node_stack_trace():
    """Kiểm tra classify_and_parse_node phát hiện stack trace qua regex & tools."""
    tools = MockAgentTools()
    llm = MockLLM()
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    log_input = """
    Traceback (most recent call last):
      File "src/math/calc.py", line 42, in divide
        return a / b
    ZeroDivisionError: division by zero
    """
    state: AgentGraphState = {
        "raw_input": log_input,
        "task_type": "",
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

    result = workflow.classify_and_parse_node(state)
    assert result["task_type"] == "incident_rca"
    assert "ZeroDivisionError" in (result["error_signature"] or "")
    assert "src/math/calc.py" in result["extracted_symbols"]
    assert "divide" in result["extracted_symbols"]


def test_classify_and_parse_node_codebase_qa():
    """Kiểm tra classify_and_parse_node dùng LLM phân loại câu hỏi thông thường."""
    tools = MockAgentTools()
    llm = MockLLM(
        responses=[
            (
                '```json\n{"task_type": "codebase_qa", "extracted_symbols": ["verify_jwt"], '
                '"error_signature": null}\n```'
            )
        ]
    )
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "Cho tôi biết luồng xác thực verify_jwt diễn ra ở đâu?",
        "task_type": "",
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

    result = workflow.classify_and_parse_node(state)
    assert result["task_type"] == "codebase_qa"
    assert result["error_signature"] is None
    assert "verify_jwt" in result["extracted_symbols"]


def test_multi_source_retrieval_node():
    """Kiểm tra multi_source_retrieval_node lấy đồng thời Code, Git và Runbooks."""
    tools = MockAgentTools()
    llm = MockLLM()
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "verify_jwt failed",
        "task_type": "incident_rca",
        "error_signature": "AuthError",
        "extracted_symbols": ["src/auth/jwt.py", "verify_jwt"],
        "code_contexts": [],
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }

    result = workflow.multi_source_retrieval_node(state)
    assert len(result["code_contexts"]) == 1
    assert result["code_contexts"][0].file_path == "src/auth/jwt.py"
    # incident_contexts chứa cả Git và Runbook
    assert len(result["incident_contexts"]) == 2


def test_grade_context_node_sufficient():
    """Kiểm tra grade_context_node khi LLM đánh giá ngữ cảnh đã đủ."""
    tools = MockAgentTools()
    llm = MockLLM(
        responses=[
            '{"is_sufficient": true, "missing_symbols": [], "reason": "Đã có đủ hàm verify_jwt"}'
        ]
    )
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "verify_jwt",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": ["verify_jwt"],
        "code_contexts": tools.code_results,
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }

    result = workflow.grade_context_node(state)
    assert result["is_sufficient"] is True


def test_grade_context_node_insufficient():
    """Kiểm tra grade_context_node khi LLM phát hiện thiếu symbol trong Call Graph."""
    tools = MockAgentTools()
    llm = MockLLM(
        responses=[
            (
                '{"is_sufficient": false, "missing_symbols": ["decode_base64"], '
                '"reason": "Thiếu hàm decode"}'
            )
        ]
    )
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "verify_jwt",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": ["verify_jwt"],
        "code_contexts": tools.code_results,
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": True,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }

    result = workflow.grade_context_node(state)
    assert result["is_sufficient"] is False
    assert "decode_base64" in result["extracted_symbols"]


def test_expand_dependencies_node():
    """Kiểm tra expand_dependencies_node mở rộng Call Graph và tăng iteration_count."""
    tools = MockAgentTools()
    llm = MockLLM()
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "verify_jwt",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": ["verify_jwt"],
        "code_contexts": tools.code_results,
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }

    result = workflow.expand_dependencies_node(state)
    assert result["iteration_count"] == 1
    assert len(result["code_contexts"]) >= 1


def test_generate_answer_node():
    """Kiểm tra generate_answer_node sinh phản hồi kèm prompt citation."""
    tools = MockAgentTools()
    llm = MockLLM(
        responses=[
            "Hàm verify_jwt xác thực token bắt đầu bằng bearer_ tại [src/auth/jwt.py:L10-L20]."
        ]
    )
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "Giải thích verify_jwt",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": ["verify_jwt"],
        "code_contexts": tools.code_results,
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": True,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }

    result = workflow.generate_answer_node(state)
    assert "[src/auth/jwt.py:L10-L20]" in result["final_response"]
    assert result["citation_error"] is None


def test_validate_citations_node_valid():
    """Kiểm tra validate_citations_node chấp thuận khi dải dòng nằm trong chunk."""
    tools = MockAgentTools()
    llm = MockLLM()
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "test",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": [],
        "code_contexts": tools.code_results,  # L10-L20
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": True,
        "citation_valid": False,
        "citation_error": None,
        "final_response": "Xem định nghĩa tại [src/auth/jwt.py:L12-L18].",
    }

    result = workflow.validate_citations_node(state)
    assert result["citation_valid"] is True
    assert result["citation_error"] is None


def test_validate_citations_node_out_of_bounds():
    """Kiểm tra validate_citations_node từ chối khi trích dẫn vượt dải dòng context."""
    tools = MockAgentTools()
    llm = MockLLM()
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "test",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": [],
        "code_contexts": tools.code_results,  # L10-L20
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": True,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "Đoạn code bịa số dòng [src/auth/jwt.py:L100-L150].",
    }

    result = workflow.validate_citations_node(state)
    assert result["citation_valid"] is False
    assert result["citation_error"] is not None
    assert "L100-L150" in result["citation_error"]


def test_validate_citations_node_not_found_guardrail():
    """Kiểm tra validate_citations_node khi LLM trả lời 'Không tìm thấy trong codebase'."""
    tools = MockAgentTools()
    llm = MockLLM()
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)

    state: AgentGraphState = {
        "raw_input": "test",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": [],
        "code_contexts": [],
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": False,
        "citation_error": None,
        "final_response": "Không tìm thấy trong codebase hiện tại.",
    }

    result = workflow.validate_citations_node(state)
    assert result["citation_valid"] is True


def test_conditional_edges_logic():
    """Kiểm tra logic 2 conditional edges."""
    tools = MockAgentTools()
    llm = MockLLM()
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm, max_iterations=3)

    # Edge 1: Insufficient and iter < 3 -> expand_dependencies
    state_expand: AgentGraphState = {
        "raw_input": "",
        "task_type": "codebase_qa",
        "error_signature": None,
        "extracted_symbols": [],
        "code_contexts": [],
        "incident_contexts": [],
        "iteration_count": 1,
        "is_sufficient": False,
        "citation_valid": True,
        "citation_error": None,
        "final_response": "",
    }
    assert workflow._should_expand_or_generate(state_expand) == "expand_dependencies"

    # Edge 1: Insufficient but iter >= 3 -> generate_answer
    state_max_iter: AgentGraphState = {
        **state_expand,
        "iteration_count": 3,
    }
    assert workflow._should_expand_or_generate(state_max_iter) == "generate_answer"

    # Edge 2: Valid citation -> END
    state_valid_cite: AgentGraphState = {
        **state_expand,
        "citation_valid": True,
    }
    assert workflow._check_citation_validity(state_valid_cite) == END

    # Edge 2: Invalid citation -> generate_answer (lần 1)
    state_invalid_cite: AgentGraphState = {
        **state_expand,
        "citation_valid": False,
    }
    assert workflow._check_citation_validity(state_invalid_cite) == "generate_answer"


def test_compile_and_run_full_graph():
    """Kiểm tra biên dịch StateGraph và thực thi trọn vẹn luồng từ START đến END."""
    tools = MockAgentTools()
    llm = MockLLM(
        responses=[
            (
                '{"task_type": "codebase_qa", "extracted_symbols": ["verify_jwt"], '
                '"error_signature": null}'
            ),
            '{"is_sufficient": true, "missing_symbols": []}',
            "Hàm verify_jwt xác thực token tại [src/auth/jwt.py:L10-L15].",
        ]
    )
    workflow = OnCallAgentWorkflow(tools=tools, llm=llm)
    app = workflow.compile_graph()

    initial_state: AgentGraphState = {
        "raw_input": "Hàm verify_jwt hoạt động thế nào?",
        "task_type": "",
        "error_signature": None,
        "extracted_symbols": [],
        "code_contexts": [],
        "incident_contexts": [],
        "iteration_count": 0,
        "is_sufficient": False,
        "citation_valid": False,
        "citation_error": None,
        "final_response": "",
    }

    final_state = app.invoke(initial_state)

    assert final_state["task_type"] == "codebase_qa"
    assert len(final_state["code_contexts"]) > 0
    assert final_state["is_sufficient"] is True
    assert final_state["citation_valid"] is True
    assert "[src/auth/jwt.py:L10-L15]" in final_state["final_response"]

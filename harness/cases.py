"""
Module: harness.cases
Chức năng: Cung cấp bộ dữ liệu chuẩn (Golden Dataset) gồm 40 ca kiểm thử thực tế
phân chia theo 3 nhóm cốt lõi:
1. codebase_qa (18 ca): Hỏi đáp kiến trúc, luồng xử lý, hybrid retrieval, AST parsing, v.v.
2. incident_rca (16 ca): Báo cáo sự cố từ stack trace thực tế, chẩn đoán commit/PR thủ phạm.
3. rejection (6 ca): Kiểm thử phòng thủ chống ảo giác (Out-of-Domain technologies).
Tuân thủ Clean Architecture, Python 3.11+, không có placeholder.
"""

from harness.schema import EvaluationCase


def get_golden_dataset() -> list[EvaluationCase]:
    """
    Trả về danh sách 40 ca kiểm thử chuẩn hóa hoàn chỉnh dùng cho Benchmark Harness.

    Returns:
        Danh sách các đối tượng EvaluationCase với đầy đủ metadata và ground truth.
    """
    cases: list[EvaluationCase] = [
        # =====================================================================
        # NHÓM 1: CODEBASE_QA (18 ca kiểm thử kiến trúc, thuật toán, call flow)
        # =====================================================================
        EvaluationCase(
            case_id="QA-001",
            input_text=(
                "Kiến trúc hệ thống RAG này sử dụng những tầng nào "
                "và tuân thủ quy tắc phụ thuộc gì?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/domain/schemas.py", "src/domain/interfaces.py", "AGENTS.md"],
            culprit_commit=None,
            expected_symbols=[
                "Clean Architecture",
                "CodeNode",
                "IncidentDocNode",
                "RetrievedContext",
            ],
            is_rejection=False,
            description="Kiểm tra kiến trúc Clean Architecture 4 tầng và Dependency Rule",
            category="architecture",
        ),
        EvaluationCase(
            case_id="QA-002",
            input_text=(
                "Cơ chế Hybrid Search kết hợp Dense và Sparse vector trong "
                "QdrantStore hoạt động như thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/retrieval/vector_store.py", "src/retrieval/embedder.py"],
            culprit_commit=None,
            expected_symbols=["QdrantStore", "hybrid_search", "HybridEmbedder", "Named Vectors"],
            is_rejection=False,
            description="Kiểm tra tìm kiếm Dense + Sparse với Reciprocal Rank Fusion",
            category="retrieval",
        ),
        EvaluationCase(
            case_id="QA-003",
            input_text=(
                "Làm thế nào TreeSitterCodeParser phân tách các khối code function và class từ AST?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/ingestion/code_parser.py", "src/domain/interfaces.py"],
            culprit_commit=None,
            expected_symbols=["TreeSitterCodeParser", "parse_file", "CodeNode", "tree_sitter"],
            is_rejection=False,
            description="Kiểm tra cơ chế AST-based code chunking hỗ trợ đa ngôn ngữ",
            category="ingestion",
        ),
        EvaluationCase(
            case_id="QA-004",
            input_text=(
                "CrossEncoderReranker sử dụng mô hình nào và "
                "cách xếp hạng lại danh sách context ra sao?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/retrieval/reranker.py", "src/domain/interfaces.py"],
            culprit_commit=None,
            expected_symbols=[
                "CrossEncoderReranker",
                "rerank",
                "bge-reranker-v2-m3",
                "RetrievedContext",
            ],
            is_rejection=False,
            description="Kiểm tra quy trình re-ranking danh sách ứng viên bằng Cross-Encoder",
            category="reranker",
        ),
        EvaluationCase(
            case_id="QA-005",
            input_text=(
                "Lớp OnCallAgentWorkflow quản lý các node và conditional edges "
                "trong LangGraph như thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/agent/workflow.py", "src/agent/state.py"],
            culprit_commit=None,
            expected_symbols=[
                "OnCallAgentWorkflow",
                "AgentGraphState",
                "build_graph",
                "validate_citations_node",
            ],
            is_rejection=False,
            description="Kiểm định cấu trúc đồ thị trạng thái Agentic LangGraph",
            category="workflow",
        ),
        EvaluationCase(
            case_id="QA-006",
            input_text=(
                "GitHistoryCollector bóc tách thông tin commit message "
                "và file thay đổi như thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/ingestion/git_connector.py", "src/domain/schemas.py"],
            culprit_commit=None,
            expected_symbols=[
                "GitHistoryCollector",
                "collect_commits",
                "IncidentDocNode",
                "GitPython",
            ],
            is_rejection=False,
            description="Kiểm tra khả năng đồng bộ lịch sử Git và xác định file thay đổi",
            category="git",
        ),
        EvaluationCase(
            case_id="QA-007",
            input_text=(
                "RunbookParser xử lý các tài liệu sự cố Markdown "
                "và trích xuất danh sách related_files như thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/ingestion/doc_parser.py", "src/domain/schemas.py"],
            culprit_commit=None,
            expected_symbols=[
                "RunbookParser",
                "parse_directory",
                "IncidentDocNode",
                "related_files",
            ],
            is_rejection=False,
            description="Kiểm tra cơ chế phân tích Markdown Runbook sự cố",
            category="runbook",
        ),
        EvaluationCase(
            case_id="QA-008",
            input_text=(
                "Node validate_citations_node kiểm tra trích dẫn "
                "[file:Lstart-Lend] chống ảo giác ra sao?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/agent/workflow.py", "src/agent/prompts.py"],
            culprit_commit=None,
            expected_symbols=[
                "validate_citations_node",
                "citation_valid",
                "citation_pattern",
                "CITATION_REPROMPT_TEMPLATE",
            ],
            is_rejection=False,
            description="Kiểm tra thuật toán kiểm định dải dòng trích dẫn của LLM",
            category="guardrail",
        ),
        EvaluationCase(
            case_id="QA-009",
            input_text=(
                "Settings trong src/config/settings.py đọc các biến môi trường nào từ file .env?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/config/settings.py", ".env.example"],
            culprit_commit=None,
            expected_symbols=["Settings", "QDRANT_HOST", "GEMINI_API_KEY", "EMBEDDING_MODEL_NAME"],
            is_rejection=False,
            description="Kiểm tra cấu hình nạp biến môi trường với Pydantic BaseSettings",
            category="config",
        ),
        EvaluationCase(
            case_id="QA-010",
            input_text=(
                "Các công cụ nào được định nghĩa trong AgentTools "
                "để phục vụ Agent tra cứu thông tin?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/agent/tools.py", "src/domain/interfaces.py"],
            culprit_commit=None,
            expected_symbols=[
                "search_codebase_tool",
                "search_git_history_tool",
                "search_runbooks_tool",
                "parse_stack_trace",
            ],
            is_rejection=False,
            description="Kiểm tra bộ công cụ tra cứu tích hợp của tác tử On-call",
            category="tools",
        ),
        EvaluationCase(
            case_id="QA-011",
            input_text=(
                "Hàm IngestionPipeline thực hiện ingestion tăng dần "
                "(incremental) đối với commit mới thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/ingestion/pipeline.py", "src/ingestion/git_connector.py"],
            culprit_commit=None,
            expected_symbols=[
                "IngestionPipeline",
                "run_incremental",
                "diff_commits",
                "upsert_nodes",
            ],
            is_rejection=False,
            description="Kiểm tra logic ingest tăng dần tránh nạp lại toàn bộ repo",
            category="pipeline",
        ),
        EvaluationCase(
            case_id="QA-012",
            input_text=(
                "Node grade_context_node xác định độ đầy đủ của ngữ cảnh dựa trên tiêu chí gì?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/agent/workflow.py", "src/agent/prompts.py"],
            culprit_commit=None,
            expected_symbols=[
                "grade_context_node",
                "is_sufficient",
                "GRADER_PROMPT",
                "_should_expand_or_generate",
            ],
            is_rejection=False,
            description="Kiểm tra thuật toán đánh giá độ đầy đủ ngữ cảnh",
            category="grader",
        ),
        EvaluationCase(
            case_id="QA-013",
            input_text="Cách tạo node_id băm duy nhất cho CodeNode trong compute_code_node_id?",
            expected_task_type="codebase_qa",
            golden_files=["src/domain/schemas.py"],
            culprit_commit=None,
            expected_symbols=[
                "compute_code_node_id",
                "hashlib.sha256",
                "content_hash",
                "file_path",
            ],
            is_rejection=False,
            description="Kiểm tra thuật toán băm SHA256 tạo khóa định danh node code",
            category="domain",
        ),
        EvaluationCase(
            case_id="QA-014",
            input_text=(
                "Cơ chế mở rộng phụ thuộc Call Graph trong "
                "expand_dependencies_node hoạt động như thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/agent/workflow.py", "src/agent/tools.py"],
            culprit_commit=None,
            expected_symbols=[
                "expand_dependencies_node",
                "get_callers_and_callees_tool",
                "iteration_count",
                "max_iterations",
            ],
            is_rejection=False,
            description="Kiểm tra quy trình mở rộng gọi hàm ngược xuôi khi ngữ cảnh chưa đủ",
            category="call_graph",
        ),
        EvaluationCase(
            case_id="QA-015",
            input_text="Định dạng dữ liệu của AgentGraphState bao gồm những trường trạng thái nào?",
            expected_task_type="codebase_qa",
            golden_files=["src/agent/state.py", "src/domain/schemas.py"],
            culprit_commit=None,
            expected_symbols=[
                "AgentGraphState",
                "raw_input",
                "code_contexts",
                "incident_contexts",
                "final_response",
            ],
            is_rejection=False,
            description="Kiểm tra cấu trúc TypedDict của đồ thị LangGraph",
            category="state",
        ),
        EvaluationCase(
            case_id="QA-016",
            input_text=(
                "Làm thế nào để khởi tạo collection Qdrant với các Named Vectors dense và sparse?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/retrieval/vector_store.py"],
            culprit_commit=None,
            expected_symbols=[
                "create_collection",
                "VectorParams",
                "Distance.COSINE",
                "SparseVectorParams",
            ],
            is_rejection=False,
            description="Kiểm tra khởi tạo schema Qdrant hỗ trợ Hybrid Search",
            category="vector_store",
        ),
        EvaluationCase(
            case_id="QA-017",
            input_text=(
                "Prompt CITATION_REPROMPT_TEMPLATE hướng dẫn LLM sửa "
                "lỗi trích dẫn dòng code như thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/agent/prompts.py", "src/agent/workflow.py"],
            culprit_commit=None,
            expected_symbols=[
                "CITATION_REPROMPT_TEMPLATE",
                "invalid_citations",
                "generate_answer_node",
            ],
            is_rejection=False,
            description="Kiểm tra cơ chế tái sinh có định hướng khi trích dẫn vi phạm dải dòng",
            category="prompts",
        ),
        EvaluationCase(
            case_id="QA-018",
            input_text=(
                "API endpoint nào phụ trách xử lý truy vấn hỏi đáp và "
                "phân tích sự cố on-call trong hệ thống?"
            ),
            expected_task_type="codebase_qa",
            golden_files=["src/api/routes.py", "src/domain/schemas.py"],
            culprit_commit=None,
            expected_symbols=["/query", "/incident/analyze", "TaskType", "FastAPI"],
            is_rejection=False,
            description="Kiểm tra các REST endpoints cung cấp dịch vụ On-call Assistant",
            category="api",
        ),
        # =====================================================================
        # NHÓM 2: INCIDENT_RCA (16 ca kiểm thử chẩn đoán sự cố, stack trace & commit)
        # =====================================================================
        EvaluationCase(
            case_id="RCA-001",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/retrieval/vector_store.py", line 142, in hybrid_search\n'
                "    results = self.client.query_points(\n"
                "qdrant_client.http.exceptions.ResponseHandlingException: "
                "Connection refused to http://localhost:6333"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/retrieval/vector_store.py", "src/config/settings.py"],
            culprit_commit="c4b81a2",
            expected_symbols=[
                "QdrantStore.hybrid_search",
                "ResponseHandlingException",
                "QdrantClient",
            ],
            is_rejection=False,
            description="Sự cố mất kết nối tới Qdrant Vector DB do sai port hoặc chưa chạy",
            category="connection_error",
        ),
        EvaluationCase(
            case_id="RCA-002",
            input_text=(
                "ERROR:root:Payment processing failed unexpectedly.\n"
                "Traceback (most recent call last):\n"
                '  File "services/payment_service.py", line 88, in process_refund\n'
                '    currency = invoice["currency"].upper()\n'
                "AttributeError: 'NoneType' object has no attribute 'upper'\n"
                "Commit culprit có thể là 7f2d9e1 vừa merge hôm qua."
            ),
            expected_task_type="incident_rca",
            golden_files=["services/payment_service.py", "src/ingestion/git_connector.py"],
            culprit_commit="7f2d9e1",
            expected_symbols=["process_refund", "AttributeError", "invoice"],
            is_rejection=False,
            description="Lỗi NullPointer / AttributeError khi invoice không chứa trường currency",
            category="null_pointer",
        ),
        EvaluationCase(
            case_id="RCA-003",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/retrieval/reranker.py", line 64, in rerank\n'
                "    scores = self.model.predict(pairs, batch_size=256)\n"
                "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.40 GiB."
            ),
            expected_task_type="incident_rca",
            golden_files=["src/retrieval/reranker.py"],
            culprit_commit="8e31fc0",
            expected_symbols=["CrossEncoderReranker.rerank", "OutOfMemoryError", "batch_size"],
            is_rejection=False,
            description="OOM trên GPU do commit 8e31fc0 tăng batch_size rerank lên quá lớn (256)",
            category="oom",
        ),
        EvaluationCase(
            case_id="RCA-004",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/agent/workflow.py", line 187, in _extract_json\n'
                "    return json.loads(cleaned_text)\n"
                "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n"
                "Gặp phải khi LLM trả lời chuỗi rỗng sau commit #142."
            ),
            expected_task_type="incident_rca",
            golden_files=["src/agent/workflow.py", "src/agent/prompts.py"],
            culprit_commit="#142",
            expected_symbols=["_extract_json", "JSONDecodeError", "json.loads"],
            is_rejection=False,
            description="Lỗi phân tích JSON khi đầu ra mô hình không chứa khối JSON hợp lệ",
            category="json_decode",
        ),
        EvaluationCase(
            case_id="RCA-005",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/ingestion/code_parser.py", line 75, in parse_file\n'
                "    tree = parser.parse(bytes(code, 'utf-8'))\n"
                "tree_sitter.TreeSitterError: Syntax grammar for language 'rust' is not loaded."
            ),
            expected_task_type="incident_rca",
            golden_files=["src/ingestion/code_parser.py"],
            culprit_commit="3a1c89f",
            expected_symbols=["TreeSitterCodeParser.parse_file", "TreeSitterError", "tree_sitter"],
            is_rejection=False,
            description="Lỗi thiếu grammar thư viện khi parse ngôn ngữ chưa đăng ký",
            category="parser_error",
        ),
        EvaluationCase(
            case_id="RCA-006",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/agent/tools.py", line 120, in search_git_history_tool\n'
                "    commits = list(self.repo.iter_commits(paths=file_path, max_count=limit))\n"
                "git.exc.GitCommandError: 'git rev-list --max-count=5 HEAD -- f' returned exit 128"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/agent/tools.py", "src/ingestion/git_connector.py"],
            culprit_commit="b1d4e65",
            expected_symbols=["search_git_history_tool", "GitCommandError", "iter_commits"],
            is_rejection=False,
            description="Lỗi thực thi lệnh Git trên repository khi đường dẫn không tồn tại",
            category="git_error",
        ),
        EvaluationCase(
            case_id="RCA-007",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/ingestion/pipeline.py", line 150, in run_full_ingestion\n'
                "    dense_emb, sparse_emb = self.embedder.embed(chunk.context_header)\n"
                "httpx.ConnectTimeout: Timed out waiting for connection\n"
                "Nghi vấn do commit d90fa11 đổi timeout từ 60s xuống 1s."
            ),
            expected_task_type="incident_rca",
            golden_files=["src/ingestion/pipeline.py", "src/retrieval/embedder.py"],
            culprit_commit="d90fa11",
            expected_symbols=["IngestionPipeline.run_full_ingestion", "ConnectTimeout", "embed"],
            is_rejection=False,
            description="ConnectTimeout khi embedding chunk do cấu hình timeout bất hợp lý",
            category="timeout",
        ),
        EvaluationCase(
            case_id="RCA-008",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/agent/workflow.py", line 450, in generate_answer_node\n'
                "    response_text = self._invoke_llm(full_prompt)\n"
                "google.api_core.exceptions.ResourceExhausted: 429 Resource has been exhausted."
            ),
            expected_task_type="incident_rca",
            golden_files=["src/agent/workflow.py", "src/config/settings.py"],
            culprit_commit=None,
            expected_symbols=["generate_answer_node", "_invoke_llm", "ResourceExhausted"],
            is_rejection=False,
            description="Lỗi cạn kiệt Quota API Gemini (HTTP 429 ResourceExhausted)",
            category="rate_limit",
        ),
        EvaluationCase(
            case_id="RCA-009",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/domain/schemas.py", line 37, in __init__\n'
                "pydantic_core._pydantic_core.ValidationError: 1 validation error for CodeNode\n"
                "start_line\n"
                "  Input should be a valid integer [type=int_parsing, input_value='abc']"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/domain/schemas.py", "src/ingestion/code_parser.py"],
            culprit_commit="5e2a77c",
            expected_symbols=["CodeNode", "ValidationError", "start_line"],
            is_rejection=False,
            description="Lỗi Pydantic ValidationError do truyền sai kiểu chuỗi vào start_line",
            category="validation_error",
        ),
        EvaluationCase(
            case_id="RCA-010",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/agent/workflow.py", line 247, in multi_source_retrieval_node\n'
                "    code_contexts = code_future.result(timeout=5)\n"
                "concurrent.futures.TimeoutError: Future timed out after 5.0 seconds\n"
                "Commit gây ra là 2c3d4e5."
            ),
            expected_task_type="incident_rca",
            golden_files=["src/agent/workflow.py"],
            culprit_commit="2c3d4e5",
            expected_symbols=["multi_source_retrieval_node", "ThreadPoolExecutor", "TimeoutError"],
            is_rejection=False,
            description="Đa luồng bị tắc nghẽn dẫn tới TimeoutError trong node retrieval",
            category="concurrency",
        ),
        EvaluationCase(
            case_id="RCA-011",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/ingestion/doc_parser.py", line 45, in parse_file\n'
                "    with open(file_path, 'r', encoding='utf-8') as f:\n"
                "FileNotFoundError: [Errno 2] No such file: 'docs/runbooks/db_outage.md'"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/ingestion/doc_parser.py"],
            culprit_commit="fa12bc4",
            expected_symbols=["RunbookParser.parse_file", "FileNotFoundError"],
            is_rejection=False,
            description="Lỗi không tìm thấy file Runbook do đổi tên thư mục trong commit fa12bc4",
            category="file_error",
        ),
        EvaluationCase(
            case_id="RCA-012",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/retrieval/embedder.py", line 58, in embed\n'
                "    model_out = self.model.encode(text, return_dense=True)\n"
                "RuntimeError: Expected all tensors to be on the same device: cuda:0 and cpu!"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/retrieval/embedder.py"],
            culprit_commit="9b8a7c6",
            expected_symbols=["HybridEmbedder.embed", "RuntimeError", "encode"],
            is_rejection=False,
            description="Lỗi xung đột thiết bị (device mismatch CUDA và CPU) sau commit 9b8a7c6",
            category="device_error",
        ),
        EvaluationCase(
            case_id="RCA-013",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/agent/workflow.py", line 510, in validate_citations_node\n'
                "    if ctx.start_line <= c_start and c_end <= ctx.end_line:\n"
                "TypeError: '<=' not supported between instances of 'NoneType' and 'int'"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/agent/workflow.py", "src/domain/schemas.py"],
            culprit_commit="1122334",
            expected_symbols=["validate_citations_node", "TypeError", "start_line"],
            is_rejection=False,
            description="Lỗi so sánh khi start_line của RetrievedContext là None gây TypeError",
            category="type_error",
        ),
        EvaluationCase(
            case_id="RCA-014",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/retrieval/vector_store.py", line 185, in upsert_code_nodes\n'
                "    self.client.upsert(collection_name=self.code_collection, points=points)\n"
                "grpc._channel._InactiveRpcError: StatusCode.DEADLINE_EXCEEDED"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/retrieval/vector_store.py"],
            culprit_commit="#205",
            expected_symbols=["QdrantStore.upsert_code_nodes", "DEADLINE_EXCEEDED", "upsert"],
            is_rejection=False,
            description="Hết hạn thời gian (DEADLINE_EXCEEDED) khi upsert batch vào Qdrant",
            category="grpc_error",
        ),
        EvaluationCase(
            case_id="RCA-015",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/agent/workflow.py", line 372, in expand_dependencies_node\n'
                "    for rel_sym in related[:3]:\n"
                "RecursionError: maximum recursion depth exceeded in Call Graph"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/agent/workflow.py", "src/agent/tools.py"],
            culprit_commit="6677889",
            expected_symbols=[
                "expand_dependencies_node",
                "RecursionError",
                "get_callers_and_callees_tool",
            ],
            is_rejection=False,
            description="Vòng lặp đệ quy vô tận khi duyệt Call Graph có chu trình phụ thuộc",
            category="recursion_error",
        ),
        EvaluationCase(
            case_id="RCA-016",
            input_text=(
                "Traceback (most recent call last):\n"
                '  File "src/api/routes.py", line 55, in query_assistant\n'
                "    result = await workflow.ainvoke(initial_state)\n"
                "ConnectionResetError: [WinError 10054] Connection was closed by remote host"
            ),
            expected_task_type="incident_rca",
            golden_files=["src/api/routes.py", "src/agent/workflow.py"],
            culprit_commit=None,
            expected_symbols=["query_assistant", "ConnectionResetError", "workflow.ainvoke"],
            is_rejection=False,
            description="Lỗi ngắt kết nối socket đột ngột từ client trong lúc stream kết quả API",
            category="network_error",
        ),
        # =====================================================================
        # NHÓM 3: REJECTION (6 ca kiểm thử phòng thủ ngoài miền - Out-of-Domain)
        # =====================================================================
        EvaluationCase(
            case_id="REJ-001",
            input_text=(
                "Cách tạo một migration thêm cột trong Ruby on Rails "
                "ActiveRecord bằng file db/migrate ra sao?"
            ),
            expected_task_type="codebase_qa",
            golden_files=[],
            culprit_commit=None,
            expected_symbols=[],
            is_rejection=True,
            description="Công nghệ Ruby on Rails không có trong repo, kỳ vọng từ chối",
            category="rejection_rails",
        ),
        EvaluationCase(
            case_id="REJ-002",
            input_text="Làm sao để emit Transfer event trong smart contract Solidity chuẩn ERC-20?",
            expected_task_type="codebase_qa",
            golden_files=[],
            culprit_commit=None,
            expected_symbols=[],
            is_rejection=True,
            description="Công nghệ Solidity smart contracts ngoài phạm vi codebase Python RAG",
            category="rejection_solidity",
        ),
        EvaluationCase(
            case_id="REJ-003",
            input_text=(
                "Cấu trúc của một pallet consensus trong Rust Substrate "
                "blockchain framework hoạt động như thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=[],
            culprit_commit=None,
            expected_symbols=[],
            is_rejection=True,
            description="Rust Substrate blockchain không tồn tại trong hệ thống, bắt buộc từ chối",
            category="rejection_rust_substrate",
        ),
        EvaluationCase(
            case_id="REJ-004",
            input_text=(
                "Cách thiết lập quan hệ hasMany và belongsTo trong "
                "Laravel Eloquent ORM của PHP như thế nào?"
            ),
            expected_task_type="codebase_qa",
            golden_files=[],
            culprit_commit=None,
            expected_symbols=[],
            is_rejection=True,
            description="PHP Laravel Eloquent ORM không có trong dự án",
            category="rejection_php_laravel",
        ),
        EvaluationCase(
            case_id="REJ-005",
            input_text=(
                "Trình tự gọi viewDidLoad, viewWillAppear và viewDidLayoutSubviews "
                "trong UIViewController của Swift iOS là gì?"
            ),
            expected_task_type="codebase_qa",
            golden_files=[],
            culprit_commit=None,
            expected_symbols=[],
            is_rejection=True,
            description="Swift UIKit UIViewController lifecycle ngoài miền codebase",
            category="rejection_swift_ios",
        ),
        EvaluationCase(
            case_id="REJ-006",
            input_text=(
                "Cách cấu hình DbContext OnModelCreating trong "
                "C# .NET Entity Framework Core với Fluent API?"
            ),
            expected_task_type="codebase_qa",
            golden_files=[],
            culprit_commit=None,
            expected_symbols=[],
            is_rejection=True,
            description="C# ASP.NET Core Entity Framework hoàn toàn ngoài phạm vi hệ thống",
            category="rejection_csharp_dotnet",
        ),
    ]

    return cases

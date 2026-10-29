# AGENTS.md — Nguyên tắc Chung cho AI Agents

> File này định nghĩa các quy tắc, ngữ cảnh dự án và conventions mà **mọi AI Agent**
> (Claude, Gemini, Copilot, Cursor, v.v.) **phải tuân thủ** khi làm việc trên codebase này.

---

## 1. Tổng quan Dự án

- **Tên:** On-call & Codebase RAG Assistant
- **Mục đích:** Hệ thống Agentic RAG chuyên biệt cho mã nguồn — hỗ trợ hỏi đáp kiến trúc codebase, chẩn đoán sự cố (RCA) từ stack trace, tra cứu lịch sử thiết kế, và phân tích tác động thay đổi.
- **Ngôn ngữ chính:** Python 3.11+
- **Kiến trúc:** Clean Architecture 4 tầng (Ingestion → Retrieval → Agent → API)

---

## 2. Tech Stack Bắt buộc

| Thành phần    | Công nghệ                        | Ghi chú                      |
| :------------ | :------------------------------- | :--------------------------- |
| Code Parser   | `tree-sitter` + `py-tree-sitter` | AST-based code splitting     |
| Embedding     | `BAAI/bge-m3` (FlagEmbedding)    | Dense + Sparse trong 1 model |
| Vector DB     | `Qdrant` (Docker local)          | Named Vectors, built-in RRF  |
| Reranker      | `BAAI/bge-reranker-v2-m3`        | Cross-Encoder                |
| LLM           | `Google Gemini 2.5 Flash`        | Qua `langchain-google-genai` |
| Orchestration | `LangGraph`                      | Stateful graph workflow      |
| Backend       | `FastAPI` + `Uvicorn`            | Async API                    |
| UI            | `Chainlit`                       | Chat interface               |
| Evaluation    | `Ragas` + `Langfuse`             | RAG metrics + tracing        |
| Git           | `GitPython` + `httpx`            | Local repo + GitHub API      |

**KHÔNG được thay thế** bất kỳ thành phần nào ở trên trừ khi được user yêu cầu rõ ràng.

---

## 3. Cấu trúc Thư mục

```text
oncall-codebase-rag/
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── .env                          # KHÔNG commit file này
├── AGENTS.md                     # File này
├── CLAUDE.md                     # Config riêng cho Claude
├── Docs/                         # Tài liệu thiết kế (PRD, skeleton)
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Pydantic Settings, load từ .env
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── schemas.py            # CodeNode, IncidentDocNode, RetrievedContext
│   │   └── interfaces.py         # ABC: BaseCodeParser, BaseVectorStore, BaseReranker
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── code_parser.py        # TreeSitterCodeParser
│   │   ├── git_connector.py      # GitHistoryCollector
│   │   ├── doc_parser.py         # RunbookParser
│   │   └── pipeline.py           # IngestionPipeline (full + incremental)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py           # HybridEmbedder (bge-m3)
│   │   ├── vector_store.py       # QdrantStore
│   │   └── reranker.py           # CrossEncoderReranker
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py              # AgentGraphState (TypedDict)
│   │   ├── tools.py              # 5 agent tools
│   │   └── workflow.py           # OnCallAgentWorkflow (LangGraph)
│   └── api/
│       ├── __init__.py
│       ├── routes.py             # 4 endpoints
│       └── app.py                # FastAPI app factory
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_agent.py
│   └── golden_dataset/
└── ui/
    └── chainlit_app.py
```

**Quy tắc:**

- KHÔNG tạo file ngoài cấu trúc trên trừ khi cần thiết và phải giải thích.
- Mỗi module (`ingestion/`, `retrieval/`, `agent/`, `api/`) phải có `__init__.py`.
- File test đặt trong `tests/`, tên bắt đầu bằng `test_`.

---

## 4. Coding Conventions

### 4.1. Ngôn ngữ & Style

- **Python 3.11+** — sử dụng type hints bắt buộc cho mọi function signature.
- **Type hints:** Dùng `list[str]` thay vì `List[str]`, `dict[str, Any]` thay vì `Dict[str, Any]` (PEP 585).
- **Docstrings:** Viết bằng **tiếng Việt** (theo tài liệu PRD). Mỗi class và public method phải có docstring.
- **Comments:** Tiếng Việt cho comments giải thích logic nghiệp vụ. Tiếng Anh cho comments kỹ thuật inline ngắn.
- **Formatter:** Black (line-length=100).
- **Linter:** Ruff.
- **Import order:** stdlib → third-party → local (ruff tự sắp xếp).

### 4.2. Naming Conventions

| Loại            | Convention         | Ví dụ                                 |
| :-------------- | :----------------- | :------------------------------------ |
| File            | `snake_case.py`    | `code_parser.py`, `vector_store.py`   |
| Class           | `PascalCase`       | `TreeSitterCodeParser`, `QdrantStore` |
| Function/Method | `snake_case`       | `parse_file()`, `hybrid_search()`     |
| Constant        | `UPPER_SNAKE_CASE` | `SKIP_DIRS`, `EXTENSION_MAP`          |
| Private         | Prefix `_`         | `_scan_repo()`, `_to_qdrant_sparse()` |
| Type alias      | `PascalCase`       | `ParsedStackFrame`                    |

### 4.3. Error Handling

- **KHÔNG dùng bare `except:`** — luôn chỉ rõ exception type.
- **File parse lỗi:** Log warning + skip file, KHÔNG crash pipeline.
- **API errors:** Return proper HTTP status code + error message JSON.
- **LLM call fails:** Retry 2 lần với exponential backoff, sau đó raise.
- **Qdrant connection fails:** Raise rõ ràng với message hướng dẫn kiểm tra Docker.

```python
# ✅ Đúng
try:
    nodes = self.code_parser.parse_file(file_path, language)
except tree_sitter.TreeSitterError as e:
    logger.warning(f"Không parse được file {file_path}: {e}")
    return []

# ❌ Sai
try:
    nodes = self.code_parser.parse_file(file_path, language)
except:
    pass
```

### 4.4. Logging

- Dùng `logging` stdlib, KHÔNG dùng `print()` cho production code.
- Logger name = module name: `logger = logging.getLogger(__name__)`
- Levels:
  - `DEBUG`: Chi tiết embedding/search internals
  - `INFO`: Pipeline progress, số lượng chunks, thời gian xử lý
  - `WARNING`: File parse lỗi, API rate limit
  - `ERROR`: Qdrant connection fail, LLM timeout

---

## 5. Domain Models — Schema Bắt buộc

### CodeNode (lưu trong collection `code_chunks`)

| Field                  | Type         | Bắt buộc | Mô tả                                                  |
| :--------------------- | :----------- | :------: | :----------------------------------------------------- |
| `node_id`              | `str` (UUID) |    ✅    | SHA256 hash của `file_path::symbol_name::content_hash` |
| `repo_name`            | `str`        |    ✅    | Tên repository                                         |
| `file_path`            | `str`        |    ✅    | Đường dẫn tương đối từ repo root                       |
| `language`             | `str`        |    ✅    | `python`, `javascript`, `typescript`, `go`, `java`     |
| `symbol_type`          | `str`        |    ✅    | `function`, `class`, `method`, `module`                |
| `symbol_name`          | `str`        |    ✅    | VD: `PaymentService.process_refund`                    |
| `start_line`           | `int`        |    ✅    | Dòng bắt đầu (1-indexed)                               |
| `end_line`             | `int`        |    ✅    | Dòng kết thúc (1-indexed)                              |
| `imports`              | `list[str]`  |    ❌    | Danh sách imports trong file                           |
| `raw_code`             | `str`        |    ✅    | Nội dung code gốc                                      |
| `context_header`       | `str`        |    ✅    | Header + raw_code (dùng cho embedding)                 |
| `last_modified_commit` | `str`        |    ❌    | SHA commit sửa gần nhất                                |

### IncidentDocNode (lưu trong collection `incident_and_docs`)

| Field           | Type         | Bắt buộc | Mô tả                                                               |
| :-------------- | :----------- | :------: | :------------------------------------------------------------------ |
| `doc_id`        | `str` (UUID) |    ✅    | UUID5 từ URL hoặc SHA                                               |
| `source_type`   | `str`        |    ✅    | `runbook`, `markdown_doc`, `pull_request`, `commit`, `github_issue` |
| `title`         | `str`        |    ✅    | Tiêu đề                                                             |
| `content`       | `str`        |    ✅    | Nội dung đầy đủ                                                     |
| `related_files` | `list[str]`  |    ❌    | File paths liên quan                                                |
| `commit_sha`    | `str`        |    ❌    | SHA commit (nếu là commit)                                          |
| `author`        | `str`        |    ❌    | Tác giả                                                             |
| `timestamp`     | `str`        |    ❌    | ISO 8601 datetime                                                   |
| `url`           | `str`        |    ❌    | Link tới PR/Issue/Wiki                                              |

---

## 6. Quy tắc Kiến trúc

### 6.1. Dependency Rule (Clean Architecture)

```
domain/ ← KHÔNG phụ thuộc vào bất kỳ tầng nào khác
ingestion/ → domain/
retrieval/ → domain/
agent/ → domain/, retrieval/
api/ → agent/, ingestion/
```

- `domain/` chỉ chứa schemas và interfaces (ABC).
- `ingestion/` và `retrieval/` implement interfaces từ `domain/`.
- `agent/` sử dụng `retrieval/` qua dependency injection.
- `api/` là entry point, wire mọi thứ lại.

### 6.2. Dependency Injection

- KHÔNG hardcode class implementations trong constructor.
- Dùng interfaces (ABC) trong type hints.
- Wire dependencies tại `api/app.py` hoặc factory function.

```python
# ✅ Đúng — dùng interface
class IngestionPipeline:
    def __init__(self, code_parser: BaseCodeParser, vector_store: BaseVectorStore):
        self.code_parser = code_parser
        self.vector_store = vector_store

# ❌ Sai — hardcode implementation
class IngestionPipeline:
    def __init__(self):
        self.code_parser = TreeSitterCodeParser()
        self.vector_store = QdrantStore("localhost", 6333)
```

### 6.3. Configuration

- Mọi config (API keys, host, port, model names) đều đọc từ `.env` qua `src/config/settings.py`.
- KHÔNG hardcode credentials, URLs, hoặc model names trong code.
- KHÔNG commit `.env` — chỉ commit `.env.example`.

---

## 7. Quy tắc cho LLM Prompts

Khi viết prompts cho LLM (trong `agent/workflow.py`):

1. **System prompt phải bắt buộc citation:** Mọi trích dẫn code phải kèm `[file_path:L_start-L_end]`.
2. **Hallucination guardrail:** Nếu không tìm thấy → trả lời "Không tìm thấy trong codebase hiện tại", KHÔNG tự sinh code giả định.
3. **Structured output:** Dùng JSON format cho classify node, evaluation node.
4. **Iteration limit:** `evaluate_context_sufficiency_node` tối đa 3 lần expand.
5. **Temperature:** Dùng `temperature=0` cho classify/evaluate nodes, `temperature=0.3` cho generate node.

---

## 8. Quy tắc Testing

- Mọi module phải có unit test tương ứng.
- Test file đặt trong `tests/`, prefix `test_`.
- Dùng `pytest` + `pytest-asyncio`.
- Integration tests cần Docker Qdrant chạy sẵn — đánh dấu `@pytest.mark.integration`.
- Mock LLM calls trong unit tests (KHÔNG gọi API thật trong CI).
- Golden dataset (30-50 cặp Q&A) đặt trong `tests/golden_dataset/`.

---

## 9. Git Conventions

- **Branch:** `feature/<phase>-<module>` (VD: `feature/phase1-code-parser`)
- **Commit message:** Tiếng Việt hoặc Anh, format: `[Phase X] <mô tả ngắn>`
  - VD: `[Phase 1] Implement TreeSitterCodeParser with Python support`
  - VD: `[Phase 2] Add QdrantStore hybrid search with RRF fusion`
- **KHÔNG commit:** `.env`, `__pycache__/`, `*.pyc`, `.ruff_cache/`, model weights

---

## 10. Quy tắc An toàn

- KHÔNG log hoặc print API keys, tokens.
- KHÔNG lưu credentials trong code, comments, hoặc docstrings.
- GitHub token chỉ đọc từ environment variable.
- Qdrant chạy local Docker, KHÔNG expose ra public network.

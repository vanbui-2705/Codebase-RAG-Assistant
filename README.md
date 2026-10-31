# 🚨 On-call & Codebase RAG Assistant

> **Hệ thống Agentic RAG chuyên biệt cho mã nguồn** — Hỗ trợ hỏi đáp kiến trúc codebase,
> chẩn đoán sự cố (RCA) từ stack trace, tra cứu lịch sử thiết kế, và phân tích tác động thay đổi.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Use Cases](#-use-cases)
- [Kiến trúc Hệ thống](#-kiến-trúc-hệ-thống)
- [Tech Stack](#-tech-stack)
- [Cài đặt & Chạy](#-cài-đặt--chạy)
- [Cấu trúc Thư mục](#-cấu-trúc-thư-mục)
- [API Endpoints](#-api-endpoints)
- [Phát triển](#-phát-triển)
- [Testing](#-testing)
- [Đánh giá Chất lượng](#-đánh-giá-chất-lượng)
- [Tài liệu Tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Tổng quan

Trong các đội ngũ phát triển phần mềm, kỹ sư mới hoặc kỹ sư trực hệ thống (On-call) thường gặp 3 rào cản lớn:

| Vấn đề | Mô tả |
|:-------|:------|
| **Phân mảnh tri thức** | Mã nguồn trên Git, tài liệu trên Confluence/Markdown, lịch sử bug trên GitHub Issues/PRs |
| **Chi phí đọc hiểu lớn** | Tìm kiếm Regex/Grep chỉ tìm tên hàm, không hiểu ngữ cảnh luồng gọi hàm hay lý do lịch sử |
| **Áp lực thời gian sự cố** | Khi lỗi Production, cần nhanh chóng xác định commit nào gây ảnh hưởng, hàm nào lỗi, runbook nào áp dụng |

**Giải pháp:** Hệ thống kết hợp **AST-aware Code Parsing** (Tree-sitter) + **Hybrid Search** (Dense + Sparse) + **Agentic Workflow** (LangGraph) để trả lời câu hỏi kèm trích dẫn chính xác dòng code.

---

## 🔍 Use Cases

### UC-01: Hỏi đáp Kiến trúc & Luồng nghiệp vụ
```
Input:  "Luồng xác thực JWT hoạt động qua những hàm nào?"
Output: Giải thích từng bước kèm trích dẫn [file_path:L_start-L_end]
```

### UC-02: Chẩn đoán Sự cố từ Stack Trace (RCA)
```
Input:  NullPointerException stack trace
Output: Hàm gây lỗi + PR/commit gần nhất + Runbook + hướng khắc phục
```

### UC-03: Tra cứu Lịch sử Thiết kế
```
Input:  "Tại sao hàm calculateTax lại có 3 tham số optional?"
Output: Tổng hợp từ Git Blame, PR description, Issue discussion
```

### UC-04: Phân tích Tác động Thay đổi
```
Input:  "Nếu sửa hàm PaymentService.process_refund, ảnh hưởng gì?"
Output: Danh sách callers/callees, test cases cần cập nhật
```

---

## 🏗 Kiến trúc Hệ thống

```
┌──────────────────────────────────────────────────────────────┐
│                    Tầng 4: API & UI                          │
│              FastAPI endpoints + Chainlit Chat                │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│              Tầng 3: Agentic Retrieval                       │
│         LangGraph Workflow (Classify → Retrieve →            │
│          Evaluate → Expand → Generate)                       │
│         5 Agent Tools + Cross-Encoder Reranking              │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│            Tầng 2: Indexing & Storage                        │
│      Qdrant Vector DB (Named Vectors: Dense + Sparse)        │
│      Hybrid Search + RRF Fusion + Payload Indexes            │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│        Tầng 1: Ingestion & AST Chunking                      │
│    Tree-sitter Parser + Git Connector + Runbook Parser        │
│    Full Indexing & Incremental Sync (git diff)                │
└──────────────────────────────────────────────────────────────┘
```

**Agent Workflow:**

```
[Input] → Classify & Parse → Retrieve Context → Evaluate Sufficiency
                                                        │
                                          ┌─────────────┴──────────────┐
                                          │                            │
                                     SUFFICIENT                  INSUFFICIENT
                                          │                     (max 3 loops)
                                          ▼                            │
                                   Generate Answer ◄── Expand Query ◄──┘
                                          │
                                      [Output]
```

---

## ⚙ Tech Stack

| Thành phần | Công nghệ | Vai trò |
|:-----------|:----------|:--------|
| **Code Parser** | `tree-sitter` + `py-tree-sitter` | Tách code theo ranh giới AST (function/class/method) |
| **Embedding** | `BAAI/bge-m3` (FlagEmbedding) | Dense (1024-dim) + Sparse vectors trong 1 model |
| **Vector DB** | `Qdrant` (Docker) | Named Vectors, built-in RRF Fusion, Payload Indexes |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Cross-Encoder reranking top-20 → top-5 |
| **LLM** | `Google Gemini 2.5 Flash` | Suy luận, phân loại, sinh câu trả lời |
| **Orchestration** | `LangGraph` | Stateful graph workflow với conditional edges |
| **Backend** | `FastAPI` + `Uvicorn` | Async REST API |
| **UI** | `Chainlit` | Chat interface với code highlighting |
| **Evaluation** | `Ragas` + `Langfuse` | RAG quality metrics + LLM tracing |
| **Git** | `GitPython` + `httpx` | Local repo analysis + GitHub REST API |

---

## 🚀 Cài đặt & Chạy

### Yêu cầu Hệ thống

- **Python** 3.11+
- **Docker** & Docker Compose (cho Qdrant)
- **RAM** ≥ 8GB (cho embedding model `bge-m3`)
- **GPU** (khuyến nghị, không bắt buộc) — tăng tốc embedding & reranking

### Bước 1: Clone & Cài đặt Dependencies

```bash
git clone https://github.com/your-org/oncall-codebase-rag.git
cd oncall-codebase-rag

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài đặt project + dev dependencies
pip install -e ".[dev]"
```

### Bước 2: Khởi động Qdrant

```bash
docker compose up -d
```

Kiểm tra: Truy cập [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Bước 3: Cấu hình Environment

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
# Bắt buộc — Lấy API key tại https://aistudio.google.com/apikey
GEMINI_API_KEY=your-gemini-api-key-here

# Optional — Để fetch GitHub PRs/Issues
GITHUB_TOKEN=ghp_your_github_token

# Mặc định, không cần sửa
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL_NAME=BAAI/bge-m3
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
```

### Bước 4: Index một Repository

```bash
# Full indexing (lần đầu)
curl -X POST http://localhost:8000/api/v1/ingest/full \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/your/repo", "docs_path": "/path/to/your/docs"}'
```

### Bước 5: Chạy ứng dụng

```bash
# Backend API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Chat UI (terminal khác)
chainlit run ui/chainlit_app.py --port 8080
```

Truy cập:
- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Chat UI:** [http://localhost:8080](http://localhost:8080)

---

## 📂 Cấu trúc Thư mục

```
oncall-codebase-rag/
│
├── pyproject.toml                      # Dependencies & project config
├── docker-compose.yml                  # Qdrant Vector Database
├── .env.example                        # Template biến môi trường
├── AGENTS.md                           # Nguyên tắc chung cho AI Agents
├── CLAUDE.md                           # Config riêng cho Claude AI
│
├── Docs/                               # Tài liệu thiết kế
│   ├── PRD_SYSTEM_ARCHITECTURE.md      # PRD & kiến trúc hệ thống
│   ├── oncall_rag_codebase_structure.md # Skeleton code & function signatures
│   └── SYSTEM_PROMPTS_AND_RULES.md     # LLM prompts & guardrails
│
├── src/                                # Mã nguồn chính
│   ├── config/
│   │   └── settings.py                 # Pydantic Settings (load từ .env)
│   ├── domain/
│   │   ├── schemas.py                  # CodeNode, IncidentDocNode, RetrievedContext
│   │   └── interfaces.py              # ABC: BaseCodeParser, BaseVectorStore, BaseReranker
│   ├── ingestion/
│   │   ├── code_parser.py             # TreeSitterCodeParser (AST splitting)
│   │   ├── git_connector.py           # GitHistoryCollector (commits, PRs, diffs)
│   │   ├── doc_parser.py              # RunbookParser (Markdown splitting)
│   │   └── pipeline.py               # IngestionPipeline (full + incremental)
│   ├── retrieval/
│   │   ├── embedder.py                # HybridEmbedder (bge-m3: dense + sparse)
│   │   ├── vector_store.py            # QdrantStore (hybrid search + RRF)
│   │   └── reranker.py               # CrossEncoderReranker (bge-reranker-v2-m3)
│   ├── agent/
│   │   ├── state.py                   # AgentGraphState (TypedDict)
│   │   ├── tools.py                   # 5 agent tools (search, git, runbook...)
│   │   ├── prompts.py                 # System prompts & LLM templates
│   │   └── workflow.py               # OnCallAgentWorkflow (LangGraph)
│   └── api/
│       ├── routes.py                  # 4 REST endpoints
│       └── app.py                     # FastAPI app factory
│
├── tests/                              # Tests
│   ├── test_domain_and_config.py      # Domain schemas & settings
│   ├── test_ingestion.py              # AST parser, doc parser
│   ├── test_pipeline_and_git.py       # Pipeline & git connector
│   ├── test_retrieval.py              # Embedder, vector store, reranker
│   └── golden_dataset/                # Evaluation dataset (30-50 Q&A pairs)
│
└── ui/
    └── chainlit_app.py                # Chainlit chat interface
```

---

## 🌐 API Endpoints

| Endpoint | Method | Mô tả |
|:---------|:------:|:------|
| `/api/v1/ingest/full` | `POST` | Đánh chỉ mục toàn bộ repo + tài liệu lần đầu |
| `/api/v1/ingest/webhook` | `POST` | Nhận GitHub Webhook để incremental sync |
| `/api/v1/query` | `POST` | Hỏi đáp kiến trúc / luồng nghiệp vụ codebase |
| `/api/v1/incident/rca` | `POST` | Chẩn đoán sự cố từ stack trace (RCA report) |

### Ví dụ Request

**Query codebase:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Hàm nào xử lý thanh toán trong module payment?",
    "filters": {"language": "python"}
  }'
```

**Gửi stack trace để RCA:**
```bash
curl -X POST http://localhost:8000/api/v1/incident/rca \
  -H "Content-Type: application/json" \
  -d '{
    "stack_trace": "Traceback (most recent call last):\n  File \"src/services/payment.py\", line 45, in process_refund\n    ...\nTimeoutError: Connection timed out",
    "context": "Lỗi xảy ra trên production từ 14:00 hôm nay"
  }'
```

---

## 🛠 Phát triển

### Coding Conventions

- **Python 3.11+** với type hints bắt buộc
- **Docstrings** bằng tiếng Việt
- **Formatter:** Black (`line-length=100`)
- **Linter:** Ruff
- Xem chi tiết tại [AGENTS.md](AGENTS.md)

### Chạy Linting

```bash
# Format code
ruff format src/ tests/

# Lint check
ruff check src/ tests/

# Type check
mypy src/
```

### Git Workflow

```bash
# Branch naming
git checkout -b feature/phase1-code-parser

# Commit message format
git commit -m "[Phase 1] Implement TreeSitterCodeParser with Python support"
```

---

## 🧪 Testing

```bash
# Chạy toàn bộ tests
pytest

# Chạy tests cụ thể
pytest tests/test_ingestion.py -v
pytest tests/test_retrieval.py -v

# Chạy integration tests (cần Docker Qdrant)
pytest -m integration

# Chạy với coverage
pytest --cov=src --cov-report=html
```

### Test Structure

| File | Kiểm thử |
|:-----|:---------|
| `test_domain_and_config.py` | Schemas validation, Settings loading |
| `test_ingestion.py` | AST parsing, doc parsing, language detection |
| `test_pipeline_and_git.py` | Full/incremental indexing, git connector |
| `test_retrieval.py` | Embedding, Qdrant upsert/search, reranking |

---

## 📊 Đánh giá Chất lượng

### Ragas Metrics

| Metric | Mục tiêu | Mô tả |
|:-------|:--------:|:------|
| **Context Precision** | ≥ 0.7 | Contexts truy xuất có liên quan đến câu hỏi? |
| **Context Recall** | ≥ 0.7 | Lấy đủ context cần thiết chưa? |
| **Faithfulness** | ≥ 0.8 | Câu trả lời trung thực với context không? |
| **Answer Relevancy** | ≥ 0.7 | Câu trả lời liên quan đến câu hỏi không? |

### Langfuse Tracing

Theo dõi từng bước Agent tại [http://localhost:3000](http://localhost:3000):
- Latency mỗi node (classify, retrieve, evaluate, generate)
- Token usage per request
- Retrieved contexts quality

---

## 📚 Tài liệu Tham khảo

| Tài liệu | Đường dẫn | Mô tả |
|:----------|:----------|:------|
| PRD & Kiến trúc | [Docs/PRD_SYSTEM_ARCHITECTURE.md](Docs/PRD_SYSTEM_ARCHITECTURE.md) | Bài toán, use cases, kiến trúc 4 tầng |
| Skeleton Code | [Docs/oncall_rag_codebase_structure.md](Docs/oncall_rag_codebase_structure.md) | Function signatures, interfaces |
| System Prompts | [Docs/SYSTEM_PROMPTS_AND_RULES.md](Docs/SYSTEM_PROMPTS_AND_RULES.md) | LLM prompts, citation rules |
| Agent Guidelines | [AGENTS.md](AGENTS.md) | Coding conventions, schema bắt buộc |
| Claude Config | [CLAUDE.md](CLAUDE.md) | Patterns, anti-patterns, checklists |

---

## 📄 License

MIT License — Xem [LICENSE](LICENSE) để biết chi tiết.


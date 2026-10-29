# Bộ Khung Mã Nguồn (Project Skeleton): On-call & Codebase RAG Assistant

Tài liệu này định nghĩa **cấu trúc thư mục thuần túy (Clean Architecture)**, **giao diện (Interfaces)** và **chữ ký hàm (Function Signatures)** của từng module. Toàn bộ dữ liệu giả lập và logic gán cứng đã được loại bỏ để bạn bắt tay vào code trực tiếp trên bộ khung chuẩn.

## 1. Cấu trúc Thư mục (Directory Structure)

```text
oncall-codebase-rag/
├── pyproject.toml                     # Khai báo dependencies và cấu hình project
├── docker-compose.yml                 # Khởi tạo hạ tầng Vector Database (Qdrant)
├── .env.example                       # Danh sách biến môi trường cần thiết
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                # Quản lý cấu hình hệ thống từ biến môi trường
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── schemas.py                 # Data Models dùng chung (CodeNode, IncidentDocNode, RetrievedContext)
│   │   └── interfaces.py              # Abstract Base Classes (BaseCodeParser, BaseVectorStore, BaseReranker)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── code_parser.py             # Tách mã nguồn theo AST và gắn Contextual Header
│   │   ├── git_connector.py           # Đồng bộ Git Commits, Pull Requests và xác định git diff
│   │   ├── doc_parser.py              # Bóc tách tài liệu kỹ thuật & Runbook (.md)
│   │   └── pipeline.py                # Điều phối Full Indexing và Incremental Indexing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py                # Sinh Dense Vector Embeddings và Sparse Embeddings (BM25)
│   │   ├── vector_store.py            # Thực thi lưu trữ, xóa theo file_path và Hybrid Search trên Qdrant
│   │   └── reranker.py                # Chấm điểm lại kết quả truy xuất bằng Cross-Encoder
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py                   # Định nghĩa cấu trúc AgentGraphState cho LangGraph
│   │   ├── tools.py                   # Các công cụ tra cứu (Code, Call Graph, Git History, Runbook)
│   │   └── workflow.py                # Các Node điều phối trong LangGraph (Classify, Retrieve, Grade, Generate)
│   └── api/
│       ├── __init__.py
│       ├── routes.py                  # Các HTTP Endpoints (/ingest/full, /ingest/webhook, /query, /incident/rca)
│       └── app.py                     # Khởi tạo ứng dụng FastAPI
└── tests/
    ├── __init__.py
    ├── test_ingestion.py              # Unit tests cho tầng Ingestion
    ├── test_retrieval.py              # Unit tests cho tầng Retrieval
    └── test_agent.py                  # Unit tests cho tầng Agent
```

## 2. Đặc tả Chức năng & Chữ ký Hàm theo Từng Tầng

### 2.1. Tầng Domain (`src/domain/`)

| File | Class / Model | Các hàm & Thuộc tính cần cài đặt | Chức năng chính |
| :--- | :--- | :--- | :--- |
| `schemas.py` | `CodeNode` | `node_id`, `file_path`, `symbol_type`, `symbol_name`, `start_line`, `end_line`, `imports`, `raw_code` | Chuẩn hóa dữ liệu đầu ra của một hàm/class sau khi cắt qua AST. |
| `schemas.py` | `IncidentDocNode` | `doc_id`, `source_type`, `title`, `content`, `related_files`, `commit_sha`, `author`, `timestamp` | Chuẩn hóa dữ liệu của một mục Runbook, Pull Request hoặc Commit. |
| `interfaces.py` | `BaseCodeParser` | `parse_file(file_path: str, language: str) -> List[CodeNode]` | Giao diện chung cho bộ phân tích cú pháp mã nguồn. |
| `interfaces.py` | `BaseVectorStore` | `upsert_nodes(...)`, `delete_by_file_paths(...)`, `hybrid_search(...)` | Giao diện chung cho Vector DB (đảm bảo dễ thay đổi giữa Qdrant, Milvus, pgvector). |
| `interfaces.py` | `BaseReranker` | `rerank(query: str, contexts: List[RetrievedContext], top_k: int)` | Giao diện chung cho mô hình xếp hạng lại kết quả. |

### 2.2. Tầng Thu thập & Tiền xử lý (`src/ingestion/`)

| File | Class | Các phương thức (Methods) | Chức năng cần hiện thực |
| :--- | :--- | :--- | :--- |
| `code_parser.py` | `TreeSitterCodeParser` | `parse_file(file_path, language)`<br>`inject_context_header(node)` | Dùng Tree-sitter tách hàm/class và ghép thêm đường dẫn file + imports vào đầu mỗi đoạn code. |
| `git_connector.py` | `GitHistoryCollector` | `fetch_recent_commits(repo_path, limit)`<br>`fetch_pull_requests(repo_full_name, state)`<br>`get_changed_files_between_commits(repo_path, old_sha, new_sha)` | Lấy lịch sử commit/PR và tìm danh sách file bị thay đổi giữa 2 commit để tránh phải index lại toàn bộ repo. |
| `doc_parser.py` | `RunbookParser` | `parse_markdown(file_path)` | Cắt tài liệu hướng dẫn xử lý sự cố theo cấu trúc tiêu đề Markdown (`#`, `##`). |
| `pipeline.py` | `IngestionPipeline` | `run_full_indexing(repo_path, docs_path)`<br>`run_incremental_sync(repo_path, base_commit, head_commit)` | Kết nối các bộ parser với Vector Store để chạy nạp dữ liệu toàn phần hoặc cập nhật từng phần. |

### 2.3. Tầng Lưu trữ & Truy xuất (`src/retrieval/`)

| File | Class | Các phương thức (Methods) | Chức năng cần hiện thực |
| :--- | :--- | :--- | :--- |
| `embedder.py` | `HybridEmbedder` | `embed_dense(texts)`<br>`embed_sparse(texts)` | Tạo đồng thời Dense Vector (ngữ nghĩa) và Sparse Vector (từ khóa tên hàm/mã lỗi). |
| `vector_store.py` | `QdrantStore` | `ensure_collections()`<br>`upsert_nodes(collection, nodes)`<br>`delete_by_file_paths(collection, file_paths)`<br>`hybrid_search(collection, query, filters, top_k)` | Thiết lập Qdrant, đánh chỉ mục cho trường `file_path` và thực thi truy vấn kết hợp RRF (Reciprocal Rank Fusion). |
| `reranker.py` | `CrossEncoderReranker` | `rerank(query, contexts, top_k)` | Chấm điểm chéo giữa câu hỏi và các đoạn code/runbook lấy ra để lọc bỏ nhiễu. |

### 2.4. Tầng Điều phối Agent (`src/agent/`)

| File | Class / Module | Các phương thức / Hàm | Chức năng cần hiện thực |
| :--- | :--- | :--- | :--- |
| `state.py` | `AgentGraphState` | `raw_input`, `task_type`, `parsed_stack_trace`, `code_contexts`, `incident_contexts`, `iteration_count`, `final_response` | Lưu trữ trạng thái xuyên suốt các bước suy luận của đồ thị LangGraph. |
| `tools.py` | Agent Tools | `parse_stack_trace(error_log)`<br>`search_codebase_tool(query, file_path, symbol_name)`<br>`get_callers_and_callees_tool(symbol_name)`<br>`search_git_history_tool(file_path, limit)`<br>`search_runbooks_tool(error_signature)` | Các hàm công cụ độc lập để Agent tra cứu chéo giữa mã nguồn, lịch sử sửa đổi và tài liệu vận hành. |
| `workflow.py` | `OnCallAgentWorkflow` | `classify_and_parse_node(state)`<br>`retrieve_context_node(state)`<br>`evaluate_context_sufficiency_node(state)`<br>`expand_query_node(state)`<br>`generate_answer_node(state)`<br>`compile_graph()` | Định nghĩa các Node và luồng rẽ nhánh (nếu thiếu hàm phụ thuộc thì tự động quay lại bước `expand_query_node` trước khi sinh câu trả lời). |

### 2.5. Tầng Giao tiếp API (`src/api/`)

| File | Endpoint / Hàm | Phương thức HTTP | Chức năng cần hiện thực |
| :--- | :--- | :--- | :--- |
| `routes.py` | `/api/v1/ingest/full` | `POST` | Kích hoạt đánh chỉ mục toàn bộ kho mã nguồn và tài liệu lần đầu. |
| `routes.py` | `/api/v1/ingest/webhook` | `POST` | Nhận payload từ GitHub Webhook khi có PR được merge để cập nhật lại đúng các file vừa sửa. |
| `routes.py` | `/api/v1/query` | `POST` | Nhận câu hỏi về kiến trúc/luồng nghiệp vụ codebase và trả về giải thích kèm trích dẫn dòng. |
| `routes.py` | `/api/v1/incident/rca` | `POST` | Nhận Stack Trace / Error Log và trả về báo cáo chẩn đoán nguyên nhân gốc rễ (RCA). |

## 3. Khung Giao diện & Chữ ký Hàm Chi tiết (Skeleton Code)

### 3.1. `src/domain/interfaces.py`
```python
"""
Module: src/domain/interfaces.py
Chức năng: Định nghĩa các Interface (Abstract Base Classes) để tách biệt tầng nghiệp vụ và thư viện bên thứ ba.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.domain.schemas import CodeNode, IncidentDocNode, RetrievedContext


class BaseCodeParser(ABC):
    """Interface cho bộ bóc tách mã nguồn theo cây cú pháp (AST)."""

    @abstractmethod
    def parse_file(self, file_path: str, language: str) -> List[CodeNode]:
        """Đọc 1 file code và tách thành danh sách các CodeNode theo hàm/class."""
        raise NotImplementedError


class BaseVectorStore(ABC):
    """Interface cho cơ sở dữ liệu Vector (Qdrant / Milvus / pgvector)."""

    @abstractmethod
    def upsert_nodes(self, collection: str, nodes: List[Any]) -> None:
        """Lưu hoặc cập nhật danh sách node vào collection."""
        raise NotImplementedError

    @abstractmethod
    def delete_by_file_paths(self, collection: str, file_paths: List[str]) -> None:
        """Xóa các chunk cũ của những file vừa bị sửa/xóa khi chạy Incremental Indexing."""
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        collection: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[RetrievedContext]:
        """Thực hiện tìm kiếm kết hợp (Dense Vector + Sparse BM25) kèm bộ lọc Metadata."""
        raise NotImplementedError


class BaseReranker(ABC):
    """Interface cho bộ chấm điểm lại thứ tự ưu tiên (Cross-Encoder Reranker)."""

    @abstractmethod
    def rerank(
        self, query: str, contexts: List[RetrievedContext], top_k: int = 5
    ) -> List[RetrievedContext]:
        """Chấm điểm lại danh sách kết quả truy xuất và trả về top_k kết quả sát nhất."""
        raise NotImplementedError
```

### 3.2. `src/ingestion/pipeline.py`
```python
"""
Module: src/ingestion/pipeline.py
Chức năng: Điều phối luồng nạp dữ liệu (Full Indexing lần đầu & Incremental Indexing khi có commit mới).
"""
from src.domain.interfaces import BaseCodeParser, BaseVectorStore
from src.ingestion.git_connector import GitHistoryCollector
from src.ingestion.doc_parser import RunbookParser


class IngestionPipeline:
    """Quản lý toàn bộ quy trình quét dữ liệu, bóc tách chunk và đẩy vào Vector Store."""

    def __init__(
        self,
        code_parser: BaseCodeParser,
        doc_parser: RunbookParser,
        git_collector: GitHistoryCollector,
        vector_store: BaseVectorStore,
    ) -> None:
        self.code_parser = code_parser
        self.doc_parser = doc_parser
        self.git_collector = git_collector
        self.vector_store = vector_store

    def run_full_indexing(self, repo_path: str, docs_path: str) -> None:
        """Quét toàn bộ kho mã nguồn, tài liệu Runbook và lịch sử Git để đánh chỉ mục mới."""
        raise NotImplementedError

    def run_incremental_sync(self, repo_path: str, base_commit: str, head_commit: str) -> None:
        """Chỉ xóa và đánh chỉ mục lại các file có thay đổi giữa base_commit và head_commit."""
        raise NotImplementedError
```

### 3.3. `src/agent/workflow.py`
```python
"""
Module: src/agent/workflow.py
Chức năng: Xây dựng và biên dịch đồ thị điều phối tác tử (LangGraph StateGraph).
"""
from src.agent.state import AgentGraphState


class OnCallAgentWorkflow:
    """Định nghĩa các Node xử lý và cạnh điều hướng (Conditional Edges) của hệ thống Agentic RAG."""

    def classify_and_parse_node(self, state: AgentGraphState) -> AgentGraphState:
        """Node 1: Phân loại yêu cầu (Hỏi đáp Code hay Sự cố On-call) và bóc tách Stack Trace nếu có."""
        raise NotImplementedError

    def retrieve_context_node(self, state: AgentGraphState) -> AgentGraphState:
        """Node 2: Gọi song song các công cụ tra cứu Codebase, Git PR History và Runbooks."""
        raise NotImplementedError

    def evaluate_context_sufficiency_node(self, state: AgentGraphState) -> str:
        """
        Node 3 (Conditional Edge): Kiểm tra xem ngữ cảnh đã đủ để kết luận chưa.
        Trả về 'expand_query' nếu cần lấy thêm hàm phụ thuộc, hoặc 'generate' nếu đã đủ.
        """
        raise NotImplementedError

    def expand_query_node(self, state: AgentGraphState) -> AgentGraphState:
        """Node 4: Sinh truy vấn bổ sung để lấy các hàm/class liên quan còn thiếu trong Call Graph."""
        raise NotImplementedError

    def generate_answer_node(self, state: AgentGraphState) -> AgentGraphState:
        """Node 5: Tổng hợp câu trả lời hoặc báo cáo RCA kèm trích dẫn chính xác [file_path:L_start-L_end]."""
        raise NotImplementedError

    def compile_graph(self):
        """Kết nối các Node thành StateGraph hoàn chỉnh và trả về đối tượng thực thi."""
        raise NotImplementedError
```

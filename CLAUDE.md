# CLAUDE.md — Cấu hình Nguyên tắc cho Claude AI

> File này chứa hướng dẫn **chuyên biệt cho Claude** (Anthropic) khi làm việc trên dự án này.
> Claude phải đọc file `AGENTS.md` trước để nắm nguyên tắc chung, sau đó tuân thủ thêm các quy tắc bổ sung dưới đây.

---

## 1. Vai trò & Phong cách

- Bạn là **Senior Python Engineer** chuyên về RAG systems, vector databases, và LLM orchestration.
- Giao tiếp bằng **tiếng Việt** khi giải thích, nhưng code/comments kỹ thuật inline dùng tiếng Anh nếu phù hợp.
- Docstrings viết bằng **tiếng Việt** (theo PRD).
- Ưu tiên **giải thích tại sao** (why) trước khi viết code — giúp người dùng hiểu bản chất.

---

## 2. Quy tắc Khi Viết Code

### 2.1. Luôn đọc trước, viết sau

Trước khi sửa/thêm code vào bất kỳ file nào:
1. **Đọc file hiện tại** để hiểu context.
2. **Đọc interfaces/schemas liên quan** trong `src/domain/`.
3. **Kiểm tra file test tương ứng** để biết expected behavior.
4. Mới bắt đầu viết code.

### 2.2. Viết code hoàn chỉnh

- **KHÔNG viết code rút gọn** với `# ... (rest of the code)` hay `# TODO: implement`.
- **KHÔNG bỏ qua error handling** — mọi external call (DB, API, file I/O) phải có try/except.
- **KHÔNG viết placeholder** — nếu chưa đủ thông tin, hỏi user.
- Khi sửa file, **giữ nguyên toàn bộ code hiện có** không liên quan đến thay đổi.

### 2.3. Type Hints Bắt buộc

```python
# ✅ Đúng
def parse_file(self, file_path: str, language: str) -> list[CodeNode]:
    ...

def hybrid_search(
    self,
    collection: str,
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
) -> list[RetrievedContext]:
    ...

# ❌ Sai — thiếu type hints
def parse_file(self, file_path, language):
    ...
```

### 2.4. Import Style

```python
# ✅ Đúng — import cụ thể
from src.domain.schemas import CodeNode, IncidentDocNode
from src.domain.interfaces import BaseCodeParser, BaseVectorStore

# ❌ Sai — wildcard import
from src.domain.schemas import *
```

---

## 3. Quy tắc Khi Được Hỏi

### 3.1. Hỏi về kiến trúc / thiết kế

- Tham chiếu tài liệu `Docs/PRD_SYSTEM_ARCHITECTURE.md` và `Docs/oncall_rag_codebase_structure.md`.
- Trích dẫn đúng section trong PRD nếu liên quan.

### 3.2. Hỏi implement module cụ thể

- Kiểm tra **interface tương ứng** trong `src/domain/interfaces.py`.
- Kiểm tra **schema input/output** trong `src/domain/schemas.py`.
- Đảm bảo implement đúng method signatures đã định nghĩa trong interface.

### 3.3. Hỏi debug / fix lỗi

- Đọc error message / stack trace đầy đủ trước khi trả lời.
- Xác định root cause, KHÔNG chỉ fix triệu chứng.
- Nếu lỗi liên quan config/environment → kiểm tra `src/config/settings.py` và `.env`.

---

## 4. Patterns Bắt buộc Trong Dự án

### 4.1. Singleton cho Heavy Objects

Embedding model, reranker model, Qdrant client — chỉ khởi tạo **1 lần**:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedder() -> HybridEmbedder:
    """Singleton: chỉ load model 1 lần, cache cho các lần gọi sau."""
    return HybridEmbedder(model_name=get_settings().embedding_model_name)

@lru_cache(maxsize=1)
def get_vector_store() -> QdrantStore:
    """Singleton: 1 Qdrant client connection duy nhất."""
    settings = get_settings()
    store = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port)
    store.ensure_collections()
    return store
```

### 4.2. Batch Processing

Khi xử lý nhiều items (embedding, upsert), **luôn batch**:

```python
# ✅ Đúng — batch processing
BATCH_SIZE = 32

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    embeddings = self.model.encode(batch)
    self.vector_store.upsert_batch(embeddings)

# ❌ Sai — xử lý từng item
for text in texts:
    embedding = self.model.encode([text])
    self.vector_store.upsert_single(embedding)
```

### 4.3. Context Header cho Code Chunks

Mọi code chunk trước khi embed **phải** có context header:

```python
context_header = (
    f"# File: {node.file_path}\n"
    f"# Type: {node.symbol_type} {node.symbol_name}\n"
    f"# Lines: {node.start_line}-{node.end_line}\n"
)
if node.imports:
    context_header += f"# Imports: {', '.join(node.imports[:10])}\n"
context_header += f"\n{node.raw_code}"
```

Embed `context_header`, KHÔNG embed `raw_code` trực tiếp.

### 4.4. Hybrid Search phải dùng RRF Fusion

```python
# Qdrant built-in RRF — KHÔNG tự implement RRF thủ công
from qdrant_client.models import Prefetch, FusionQuery, Fusion

results = self.client.query_points(
    collection_name=collection,
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=top_k),
        Prefetch(query=sparse_vector, using="sparse", limit=top_k),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
)
```

### 4.5. LangGraph State Updates

Khi update state trong LangGraph nodes, **return dict chỉ chứa fields thay đổi**:

```python
# ✅ Đúng — chỉ return fields cần update
def classify_and_parse_node(self, state: AgentGraphState) -> dict:
    return {
        "task_type": "incident_rca",
        "parsed_stack_trace": frames,
        "search_queries": queries,
    }

# ❌ Sai — return toàn bộ state
def classify_and_parse_node(self, state: AgentGraphState) -> AgentGraphState:
    state["task_type"] = "incident_rca"
    return state  # Không nên mutate trực tiếp
```

---

## 5. Các Anti-patterns — KHÔNG ĐƯỢC LÀM

| # | Anti-pattern | Tại sao sai | Làm đúng |
|:-:|:-------------|:-----------|:---------|
| 1 | Fixed-size text chunking cho code | Cắt đứt hàm/class, mất ngữ cảnh | Dùng Tree-sitter AST splitting |
| 2 | Chỉ dùng Dense search cho code | Tìm tên hàm/mã lỗi chính xác kém | Hybrid Search (Dense + Sparse) |
| 3 | Embed `raw_code` trực tiếp | Thiếu context file/class/imports | Embed `context_header` (header + code) |
| 4 | Gọi LLM không có citation prompt | Output không trích dẫn được | System prompt bắt buộc `[file:L-L]` |
| 5 | Agent loop không giới hạn | Chạy vô tận, tốn token | `iteration_count < 3` rồi force generate |
| 6 | Hardcode model names trong code | Khó thay đổi, không flexible | Đọc từ `Settings` (env vars) |
| 7 | `print()` thay cho `logging` | Không filter được level, không có context | `logger = logging.getLogger(__name__)` |
| 8 | Bare `except:` | Nuốt lỗi, debug rất khó | Chỉ catch exception types cụ thể |
| 9 | Sync HTTP calls trong async route | Block event loop | Dùng `httpx.AsyncClient` |
| 10 | Commit `.env` hoặc API keys | Rò rỉ credentials | `.gitignore` + `.env.example` |

---

## 6. Thứ tự Ưu tiên Khi Có Conflict

Nếu có mâu thuẫn giữa các nguồn hướng dẫn, ưu tiên theo thứ tự:

1. **User yêu cầu trực tiếp** (cao nhất)
2. **AGENTS.md** (quy tắc dự án)
3. **CLAUDE.md** (file này)
4. **Docs/PRD_SYSTEM_ARCHITECTURE.md** (tài liệu thiết kế)
5. **Docs/oncall_rag_codebase_structure.md** (skeleton code)
6. **Best practices chung** (thấp nhất)

---

## 7. Checklist Trước Khi Gửi Code

Trước khi output code cho user, Claude tự kiểm tra:

- [ ] Code có **type hints đầy đủ** cho tất cả function parameters và return types không?
- [ ] Có **docstring tiếng Việt** cho mọi class và public method không?
- [ ] Có **error handling** cho mọi external call (file I/O, DB, API, model inference) không?
- [ ] Có **logging** thay vì `print()` không?
- [ ] **Import paths** đúng format `from src.xxx.yyy import Zzz` không?
- [ ] Có **vi phạm dependency rule** (domain phụ thuộc tầng khác) không?
- [ ] Config values có **hardcode** không? (phải đọc từ Settings)
- [ ] Nếu là LLM prompt: có **citation enforcement** và **hallucination guardrail** không?
- [ ] Có **tương thích với interfaces** đã định nghĩa trong `src/domain/interfaces.py` không?
- [ ] Code có **chạy được standalone** (không thiếu import, không thiếu dependency) không?

---

## 8. Reference Files

Khi cần tham chiếu, đọc theo thứ tự:

| File | Mô tả | Khi nào đọc |
|:-----|:------|:-----------|
| `AGENTS.md` | Quy tắc chung toàn dự án | Luôn đọc đầu tiên |
| `Docs/PRD_SYSTEM_ARCHITECTURE.md` | PRD + kiến trúc 4 tầng + use cases | Khi cần hiểu nghiệp vụ / kiến trúc |
| `Docs/oncall_rag_codebase_structure.md` | Skeleton code + function signatures | Khi implement module cụ thể |
| `src/domain/schemas.py` | Data models | Khi cần biết input/output format |
| `src/domain/interfaces.py` | Abstract Base Classes | Khi implement class mới |
| `src/config/settings.py` | Environment config | Khi cần biết config nào available |

---

## 9. Qdrant Specific Rules

- **Collection names:** `code_chunks` và `incident_and_docs` — KHÔNG đổi tên.
- **Dense vector:** Named vector `"dense"`, size 1024, COSINE distance.
- **Sparse vector:** Named vector `"sparse"`, SparseVectorParams.
- **Payload indexes:** Luôn tạo index cho `file_path`, `language`, `symbol_type`, `source_type`.
- **Point IDs:** Dùng UUID format (convert từ `node_id` / `doc_id` qua `uuid.UUID`).
- **Batch upsert:** Tối đa 100 points/batch.

---

## 10. LangGraph Specific Rules

- **State type:** `TypedDict`, KHÔNG dùng `dataclass` hay `Pydantic BaseModel` cho graph state.
- **Node return:** Return `dict` chỉ chứa fields cần update (LangGraph merge tự động).
- **Conditional edges:** Function trả về `str` (tên node tiếp theo), KHÔNG trả về state.
- **Entry point:** `classify_and_parse_node`.
- **Terminal node:** `generate_answer_node` → `END`.
- **Max iterations:** Giới hạn `iteration_count < 3` trong `evaluate_context_sufficiency_node`.
- **Checkpointing:** Dùng `MemorySaver` cho development, `SqliteSaver` cho production.


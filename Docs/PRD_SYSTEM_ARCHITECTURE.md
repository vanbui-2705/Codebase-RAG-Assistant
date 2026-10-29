Tài liệu Thiết kế Hệ thống (PRD & System Architecture)
Dự án: Trợ lý Codebase & Xử lý Sự cố (On-call Dev Assistant)

1. Tổng quan Dự án & Bài toán (Problem Statement)
   Trong các đội ngũ phát triển phần mềm, kỹ sư mới (Onboarding) hoặc kỹ sư trực hệ thống (On-call Engineer) thường gặp 3 rào cản lớn:

Phân mảnh tri thức: Mã nguồn nằm trên Git, tài liệu kiến trúc và Runbook nằm trên Confluence/Markdown, lịch sử thảo luận bug nằm trong GitHub Issues/Pull Requests (PRs) hoặc Jira.
Chi phí đọc hiểu Codebase lớn: Tìm kiếm bằng từ khóa (Regex/Grep) chỉ tìm được tên hàm chính xác mà không hiểu ngữ cảnh luồng gọi hàm (Call Graph) hoặc lý do lịch sử vì sao đoạn code đó được viết như vậy.
Áp lực thời gian khi xử lý sự cố (Incident Response): Khi có lỗi Production (Stack trace / Error log), kỹ sư trực cần nhanh chóng xác định commit/PR nào gần đây gây ảnh hưởng, hàm nào đang phát sinh ngoại lệ và quy trình xử lý (Runbook) tương ứng.

Mục tiêu hệ thống: Xây dựng hệ thống Agentic & AST-aware RAG kết hợp tra cứu mã nguồn theo cấu trúc cú pháp, lịch sử Git/PR và tài liệu kỹ thuật để giải đáp kiến trúc codebase và hỗ trợ chẩn đoán nguyên nhân gốc rễ (Root Cause Analysis - RCA) khi xảy ra sự cố. 2. Các Use Cases Cốt lõi (Core Use Cases)
Mã UC
Tên Use Case
Đầu vào (Input)
Đầu ra kỳ vọng (Output)
UC-01
Hỏi đáp Kiến trúc & Luồng nghiệp vụ (Codebase Q&A)
Câu hỏi tự nhiên (VD: "Luồng xác thực JWT và kiểm tra quyền thanh toán hoạt động qua những hàm nào?")
Giải thích luồng xử lý từng bước, trích dẫn chính xác đường dẫn file (file_path), số dòng (start_line - end_line) và đoạn code liên quan.
UC-02
Chẩn đoán Sự cố từ Stack Trace / Log (On-call RCA)
Error log / Stack trace (VD: NullPointerException hoặc TimeoutError tại module X)
Xác định hàm gây lỗi, kiểm tra các PR/Commit thay đổi file đó gần đây, đối chiếu Runbook và đề xuất hướng khắc phục (Hotfix / Rollback).
UC-03
Tra cứu Lịch sử Thiết kế (Why-is-it-written-this-way)
Câu hỏi về lý do thiết kế hoặc đoạn code khó hiểu
Tổng hợp thông tin từ Git Blame, mô tả Pull Request (PR Description) và thảo luận trong Issue liên quan để giải thích bối cảnh ra quyết định.
UC-04
Kiểm tra Tác động Thay đổi (Impact Analysis)
Tên hàm, class hoặc API endpoint dự định sửa đổi
Liệt kê các module/file phụ thuộc (Callers/Callees) và các test cases cần cập nhật.

3. Kiến trúc Hệ thống Chi tiết (System Architecture)
   Hệ thống được chia thành 4 tầng xử lý chính:
   3.1. Tầng Thu thập & Tiền xử lý Dữ liệu (Multi-source Ingestion & AST Chunking)
   Khác với văn bản thông thường, mã nguồn nếu cắt theo số ký tự cố định (Fixed-size chunking) sẽ làm đứt gãy hàm/class và mất ngữ cảnh. Hệ thống áp dụng:

Phân tách mã nguồn theo cây cú pháp trừu tượng (AST-based Code Splitting): Sử dụng Tree-sitter (thông qua thư viện py-tree-sitter hoặc module LlamaIndex CodeSplitter) để tách code theo đúng ranh giới Class, Function, Method và Interface.
Trích xuất Ngữ cảnh Phụ thuộc (Contextual Header Injection): Trước khi tạo vector embedding cho một hàm, tự động gắn thêm phần đầu (header) chứa: Đường dẫn file, tên Class cha, danh sách imports và chuỗi Docstring/Comment của hàm đó.
Thu thập Tài liệu & Lịch sử Git:
Markdown / Runbooks: Tách theo cấu trúc Heading (H1, H2, H3), giữ nguyên các khối lệnh (code blocks).
Git Commits / Pull Requests / Issues: Thu thập tiêu đề, nội dung mô tả, danh sách file thay đổi (changed_files) và nhãn (labels).
3.2. Tầng Lưu trữ & Đánh chỉ mục (Indexing & Storage Layer)
Sử dụng cơ sở dữ liệu vector Qdrant kết hợp cơ chế Payload Indexing để hỗ trợ cả tìm kiếm ngữ nghĩa (Dense Vector) và tìm kiếm từ khóa chính xác (Sparse Vector / BM25):

Tại sao cần Hybrid Search cho Code?: Khi tìm tên biến, mã lỗi (VD: ERR_CONNECTION_REFUSED), tên hàm (calculateTaxV2), tìm kiếm từ khóa (BM25/Sparse) chính xác hơn nhiều so với Dense Embedding; trong khi Dense Embedding mạnh về câu hỏi khái niệm ("Hàm nào xử lý hoàn tiền?").
Đồ thị phụ thuộc nhẹ (Lightweight Code Graph): Lưu trữ quan hệ imports, calls và inherits vào Metadata hoặc Graph Store để khi truy xuất 1 hàm, hệ thống có thể lấy kèm hàm được gọi (Callee) ở bước mở rộng ngữ cảnh.
3.3. Tầng Truy xuất & Điều phối Tác tử (Agentic Retrieval & Orchestration)
Sử dụng LangGraph để xây dựng luồng điều phối trạng thái (Stateful Agent Workflow) thay vì chỉ dùng một bước truy xuất tuyến tính:

Query Router & Classifier: Phân loại yêu cầu của người dùng thuộc nhóm Hỏi đáp Code, Chẩn đoán Lỗi (Incident) hay Tra cứu Lịch sử PR.
Query Decomposition & Rewriting: Tách câu hỏi phức tạp hoặc phân tích Stack Trace thành danh sách các file/hàm nghi vấn.
Multi-step Retrieval & Tool Calling:
Công cụ search_codebase: Tìm kiếm Hybrid (Dense + Sparse) kết hợp lọc theo language, file_path hoặc module.
Công cụ get_symbol_definition: Lấy toàn bộ nội dung hàm/class khi biết tên định danh.
Công cụ search_git_history: Tìm các PR/Commit gần nhất tác động đến file đang gặp lỗi.
Công cụ search_runbooks: Tra cứu tài liệu hướng dẫn xử lý sự cố.
Cross-Encoder Reranking: Chấm điểm lại top 20 kết quả từ Hybrid Search xuống top 5 đoạn ngữ cảnh sát nhất trước khi đưa vào LLM.
Self-Reflection / Grading Node: Kiểm tra xem các đoạn code lấy ra đã đủ để trả lời chưa; nếu hàm A gọi hàm B ở file khác mà chưa có nội dung hàm B, Agent tự động gọi thêm bước truy xuất hàm B.
3.4. Tầng Sinh câu trả lời & Trích dẫn (Generation & Guardrails)
Strict Citation Enforcement: Bắt buộc LLM trả lời kèm đường dẫn file và dòng code ([src/auth/jwt.py:L45-L82]).
Hallucination Guardrail: Nếu không tìm thấy hàm hoặc thông tin trong repo, hệ thống phải thông báo rõ không tìm thấy trong codebase hiện tại thay vì tự sinh code giả định. 4. Thiết kế Data Schema & Metadata (Payload Schema)
Mỗi điểm dữ liệu (Point) lưu trong Qdrant được chuẩn hóa theo cấu trúc Metadata sau:
4.1. Collection code_chunks
chunk_id (UUID): Định danh duy nhất của đoạn code (băm từ file_path + symbol_name + git_blob_sha).
repo_name (Keyword): Tên repository.
file_path (Keyword): Đường dẫn tương đối (VD: app/services/payment.py).
language (Keyword): Ngôn ngữ lập trình (python, typescript, go, java).
symbol_type (Keyword): Loại thực thể (function, class, method, module).
symbol_name (Text/Keyword): Tên hàm hoặc class (VD: PaymentService.process_refund).
start_line & end_line (Integer): Dòng bắt đầu và kết thúc.
imports (Keyword Array): Danh sách các module/hàm được import.
last_modified_commit (Keyword): Mã SHA của commit sửa đổi gần nhất.
raw_code (Text): Nội dung code gốc kèm docstring.
4.2. Collection incident_and_docs
doc_id (UUID): Định danh tài liệu/PR/Issue.
source_type (Keyword): runbook, markdown_doc, pull_request, github_issue.
title (Text): Tiêu đề tài liệu hoặc PR.
related_files (Keyword Array): Danh sách đường dẫn file liên quan trong PR/Runbook.
created_at (Datetime): Thời gian tạo/cập nhật để ưu tiên tài liệu mới nhất.
url (Keyword): Đường dẫn trực tiếp tới PR/Issue/Wiki. 5. Đề xuất Công nghệ (Tech Stack Options)
Tùy theo điều kiện hạ tầng và mục tiêu dự án, có thể chọn 1 trong 2 cấu hình:

Thành phần
Phương án A: Tối ưu Chất lượng & Tốc độ (Cloud API)
Phương án B: Bảo mật & Chạy Local / Open-source (0đ API)
Parser & Chunking
Tree-sitter + LlamaIndex CodeSplitter
Tree-sitter + LlamaIndex CodeSplitter
Embedding Model
VoyageCode (voyage-code-2 / voyage-code-3) hoặc text-embedding-3-large
BAAI/bge-m3 hoặc jina-embeddings-v2-base-code (chạy qua Ollama / SentenceTransformers)
Vector Database
Qdrant (Docker hoặc Qdrant Cloud Free Tier)
Qdrant chạy trên Docker local
Reranker
Cohere Rerank v3
BAAI/bge-reranker-v2-m3
Orchestration
LangGraph
LangGraph
LLM Suy luận
Gemini 2.5 Pro / Claude 3.5 Sonnet / GPT-4o
Qwen-2.5-Coder (7B/14B/32B) hoặc DeepSeek-Coder chạy qua Ollama / vLLM
Backend & UI
FastAPI + Streamlit / Chainlit (hoặc tích hợp Slack/Discord Bot)
FastAPI + Chainlit
Đánh giá & Giám sát
Ragas + Langfuse
Ragas + Langfuse (Self-hosted)

6. Lộ trình Triển khai 4 Giai đoạn (Implementation Roadmap)
   Giai đoạn 1 — MVP Codebase Ingestion & Hybrid Search (Tuần 1):
   Xây dựng script quét GitHub repository (bỏ qua .git, node_modules, venv, file build).
   Tích hợp Tree-sitter để tách hàm/class và đẩy vào Qdrant với Hybrid Search (Dense + BM25).
   Xây dựng CLI hoặc API cơ bản trả lời câu hỏi về code có trích dẫn dòng.
   Giai đoạn 2 — Tích hợp Đa nguồn: Git History, PRs & Runbooks (Tuần 2):
   Viết connector lấy lịch sử Commit, Pull Requests, Issues và các file Markdown/Runbook.
   Liên kết metadata file_path giữa các commit/PR với các chunk mã nguồn.
   Giai đoạn 3 — Agentic Workflow với LangGraph (Tuần 3):
   Thiết kế đồ thị tác tử trên LangGraph: phân tích Stack Trace, tự động tra cứu file lỗi $\rightarrow$ tìm commit sửa file gần nhất $\rightarrow$ tra cứu Runbook $\rightarrow$ tổng hợp báo cáo RCA.
   Thêm cơ chế cập nhật Incremental Indexing (chỉ đánh chỉ mục lại các file thay đổi qua git diff hoặc GitHub Webhook).
   Giai đoạn 4 — Đánh giá Chất lượng (Evaluation) & Giao diện (Tuần 4):
   Tạo bộ dữ liệu kiểm thử (Golden Dataset: 30–50 cặp câu hỏi/stack trace và file/hàm đáp án chuẩn).
   Đo lường hệ thống bằng bộ chỉ số của Ragas (Context Precision, Context Recall, Faithfulness, Answer Relevancy) kết hợp theo dõi trace trên Langfuse.
   Đóng gói Docker Compose và giao diện Chat (Chainlit / Web UI).

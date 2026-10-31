# ==============================================================================
# ON-CALL & CODEBASE RAG ASSISTANT — MULTI-STAGE DOCKERFILE
# ==============================================================================
FROM python:3.12-slim AS base

# Cài đặt các công cụ hệ thống cần thiết để build C-extensions (tree-sitter, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy cấu hình package để tận dụng Docker layer caching
COPY pyproject.toml README.md ./

# Cài đặt pip packages chính
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    "tree-sitter>=0.21.0" \
    "tree-sitter-python" \
    "tree-sitter-javascript" \
    "tree-sitter-typescript" \
    "tree-sitter-go" \
    "tree-sitter-java" \
    "qdrant-client>=1.8.0" \
    "langgraph>=0.0.30" \
    "langchain-google-genai>=1.0.0" \
    "google-generativeai>=0.4.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.28.0" \
    "chainlit>=1.0.0" \
    "pydantic>=2.6.0" \
    "pydantic-settings>=2.2.0" \
    "gitpython>=3.1.40" \
    "httpx>=0.27.0"

# Copy toàn bộ mã nguồn vào container
COPY src/ /app/src/
COPY ui/ /app/ui/
COPY harness/ /app/harness/
COPY Docs/ /app/Docs/

# Thiết lập biến môi trường Python
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Lệnh mặc định khởi chạy FastAPI
CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

"""
Unit tests cho Tầng Giao tiếp API (Giai đoạn 5):
Kiểm thử 4 REST endpoints và health check bằng FastAPI TestClient:
1. GET /health
2. POST /api/v1/query
3. POST /api/v1/incident/rca
4. POST /api/v1/ingest/full
5. POST /api/v1/ingest/webhook
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.domain.schemas import RetrievedContext


@pytest.fixture
def mock_workflow():
    """Mock Agent Workflow Runner cho API tests."""
    runner = MagicMock()
    runner.compile_graph.return_value = runner

    def default_invoke(state):
        task_type = state.get("task_type", "codebase_qa")
        if task_type == "codebase_qa":
            return {
                "final_response": (
                    "Hàm verify_jwt xác thực JWT được định nghĩa tại [src/auth/jwt.py:L10-L20]. "
                    "Hàm này kiểm tra tiền tố bearer_ của token."
                ),
                "task_type": "codebase_qa",
                "error_signature": None,
                "code_contexts": [
                    RetrievedContext(
                        node_id="chunk_jwt_1",
                        source_type="code",
                        file_path="src/auth/jwt.py",
                        content=(
                            "def verify_jwt(token: str) -> bool:\n"
                            "    return token.startswith('bearer_')"
                        ),
                        score=0.92,
                        start_line=10,
                        end_line=20,
                        symbol_name="verify_jwt",
                    )
                ],
                "incident_contexts": [],
            }
        else:
            return {
                "final_response": (
                    "### 1. Tóm tắt sự cố\nLỗi chia cho 0 tại process_payment.\n"
                    "### 2. Nguyên nhân gốc rễ\nBiến discount_rate không được validate "
                    "[src/billing/calc.py:L40-L50].\n"
                    "### 3. Commit nghi vấn\nCommit a1b2c3d sửa đổi cách tính chiết khấu.\n"
                    "### 4. Khuyến nghị\nTham khảo Runbook ZeroDivision."
                ),
                "task_type": "incident_rca",
                "error_signature": "ZeroDivisionError",
                "code_contexts": [
                    RetrievedContext(
                        node_id="chunk_calc_1",
                        source_type="code",
                        file_path="src/billing/calc.py",
                        content="def calculate_discount(price, rate):\n    return price / rate",
                        score=0.95,
                        start_line=40,
                        end_line=50,
                        symbol_name="calculate_discount",
                    )
                ],
                "incident_contexts": [
                    RetrievedContext(
                        node_id="commit_a1b2c3d",
                        source_type="commit",
                        file_path="src/billing/calc.py",
                        content="Commit: Update discount formula",
                        score=0.88,
                        metadata={
                            "commit_sha": "a1b2c3d4e5f6",
                            "title": "Update discount formula",
                        },
                    ),
                    RetrievedContext(
                        node_id="runbook_zero_div",
                        source_type="runbook",
                        file_path="docs/runbooks/zero_division.md",
                        content="# Runbook Zero Division",
                        score=0.85,
                        metadata={"title": "Zero Division Fix Guide"},
                    ),
                ],
            }

    runner.invoke.side_effect = default_invoke
    return runner


@pytest.fixture
def mock_pipeline():
    """Mock IngestionPipeline cho API tests."""
    pipeline = MagicMock()
    pipeline.run_full_indexing.return_value = {
        "code_nodes": 45,
        "doc_nodes": 12,
    }
    pipeline.run_incremental_sync.return_value = {
        "deleted_files": 2,
        "reindexed_nodes": 8,
    }
    return pipeline


@pytest.fixture
def mock_vector_store():
    """Mock Vector Store."""
    store = MagicMock()
    store.client.get_collections.return_value = ["code_chunks", "incident_and_docs"]
    return store


@pytest.fixture
def client(mock_workflow, mock_pipeline, mock_vector_store):
    """Khởi tạo FastAPI TestClient với mocked dependencies."""
    app = create_app(
        vector_store=mock_vector_store,
        workflow=mock_workflow,
        pipeline=mock_pipeline,
        default_repo_path="./test_repo",
    )
    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# TESTS FOR /health ENDPOINT
# =============================================================================


def test_health_check_endpoint(client):
    """Kiểm tra endpoint GET /health trả về trạng thái hoạt động và DB connection."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert data["database"]["status"] == "connected"


# =============================================================================
# TESTS FOR /api/v1/query ENDPOINT
# =============================================================================


def test_query_codebase_success(client, mock_workflow):
    """Kiểm tra endpoint POST /api/v1/query trả về câu trả lời kèm trích dẫn dòng."""
    payload = {
        "query": "Làm thế nào để xác thực token JWT trong hệ thống?",
        "top_k": 5,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "verify_jwt" in data["answer"]
    assert len(data["citations"]) > 0
    assert "[src/auth/jwt.py:L10-L20]" in data["citations"]
    assert len(data["retrieved_contexts"]) == 1
    assert data["retrieved_contexts"][0]["file_path"] == "src/auth/jwt.py"
    mock_workflow.invoke.assert_called_once()


def test_query_codebase_validation_error(client):
    """Kiểm tra validation khi thiếu trường query bắt buộc."""
    response = client.post("/api/v1/query", json={})
    assert response.status_code == 422


# =============================================================================
# TESTS FOR /api/v1/incident/rca ENDPOINT
# =============================================================================


def test_incident_rca_success(client, mock_workflow):
    """Kiểm tra endpoint POST /api/v1/incident/rca trả về báo cáo RCA 4 phần chuẩn hóa."""
    payload = {
        "error_log": (
            "Traceback (most recent call last):\n"
            '  File "src/billing/calc.py", line 45, in calculate_discount\n'
            "    return price / rate\n"
            "ZeroDivisionError: division by zero"
        ),
        "repo_path": "/var/repo/billing",
    }
    response = client.post("/api/v1/incident/rca", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "ZeroDivisionError" in data["error_signature"]
    assert "Tóm tắt sự cố" in data["rca_report"]
    assert "Nguyên nhân gốc rễ" in data["rca_report"]
    assert len(data["culprit_commits"]) > 0
    assert "a1b2c3d" in data["culprit_commits"][0]
    assert len(data["runbook_recommendations"]) > 0
    assert "Zero Division Fix Guide" in data["runbook_recommendations"][0]


# =============================================================================
# TESTS FOR /api/v1/ingest/full ENDPOINT
# =============================================================================


def test_ingest_full_success(client, mock_pipeline):
    """Kiểm tra endpoint POST /api/v1/ingest/full kích hoạt Full Indexing thành công."""
    payload = {
        "repo_path": "./src",
        "docs_path": "./docs",
        "repo_full_name": "vinai/oncall-assistant",
    }
    response = client.post("/api/v1/ingest/full", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["indexed_summary"]["code_nodes"] == 45
    assert data["indexed_summary"]["doc_nodes"] == 12
    mock_pipeline.run_full_indexing.assert_called_once_with(
        repo_path="./src",
        docs_path="./docs",
        repo_full_name="vinai/oncall-assistant",
    )


def test_ingest_full_repo_not_found(client, mock_pipeline):
    """Kiểm tra xử lý lỗi khi thư mục repository không tồn tại (404)."""
    mock_pipeline.run_full_indexing.side_effect = FileNotFoundError("Thư mục repo không tồn tại")
    payload = {"repo_path": "./non_existent_directory"}
    response = client.post("/api/v1/ingest/full", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "không tồn tại" in data["message"]


# =============================================================================
# TESTS FOR /api/v1/ingest/webhook ENDPOINT
# =============================================================================


def test_webhook_push_event_success(client, mock_pipeline):
    """Kiểm tra webhook đồng bộ Incremental Sync khi nhận push event."""
    payload = {
        "ref": "refs/heads/main",
        "before": "commit_111111",
        "after": "commit_222222",
    }
    response = client.post("/api/v1/ingest/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["sync_summary"]["deleted_files"] == 2
    assert data["sync_summary"]["reindexed_nodes"] == 8
    mock_pipeline.run_incremental_sync.assert_called_once_with(
        repo_path="./test_repo",
        base_commit="commit_111111",
        head_commit="commit_222222",
    )


def test_webhook_pull_request_merged_success(client, mock_pipeline):
    """Kiểm tra webhook đồng bộ Incremental Sync khi nhận Pull Request merged."""
    payload = {
        "action": "closed",
        "pull_request": {
            "merged": True,
            "base": {"sha": "base_sha_abc"},
            "head": {"sha": "head_sha_xyz"},
        },
    }
    response = client.post("/api/v1/ingest/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["sync_summary"]["reindexed_nodes"] == 8
    mock_pipeline.run_incremental_sync.assert_called_once_with(
        repo_path="./test_repo",
        base_commit="base_sha_abc",
        head_commit="head_sha_xyz",
    )


def test_webhook_pull_request_unmerged_skipped(client, mock_pipeline):
    """Kiểm tra webhook bỏ qua đồng bộ khi PR chỉ được đóng mà không merge."""
    payload = {
        "action": "closed",
        "pull_request": {
            "merged": False,
            "base": {"sha": "base_sha_abc"},
            "head": {"sha": "head_sha_xyz"},
        },
    }
    response = client.post("/api/v1/ingest/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "skipped"
    assert data["sync_summary"]["reindexed_nodes"] == 0
    mock_pipeline.run_incremental_sync.assert_not_called()


def test_webhook_missing_commit_info_skipped(client, mock_pipeline):
    """Kiểm tra webhook bỏ qua đồng bộ khi payload thiếu thông tin commit."""
    payload = {"ref": "refs/tags/v1.0.0"}
    response = client.post("/api/v1/ingest/webhook", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "skipped"
    mock_pipeline.run_incremental_sync.assert_not_called()

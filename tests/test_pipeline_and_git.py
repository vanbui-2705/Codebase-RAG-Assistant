"""
Unit tests cho Giai đoạn 2: Kiểm thử GitHistoryCollector và IngestionPipeline.
Kiểm thử trích xuất Git commits, tính toán Git diff giữa 2 commit,
Full Indexing pipeline và Incremental Sync với Qdrant in-memory store.
"""
from pathlib import Path
import tempfile
import git

from src.ingestion.code_parser import TreeSitterCodeParser
from src.ingestion.doc_parser import RunbookParser
from src.ingestion.git_connector import GitHistoryCollector
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.embedder import HybridEmbedder
from src.retrieval.vector_store import QdrantStore


def test_git_history_collector_commits_and_diff():
    """Kiểm tra GitHistoryCollector trích xuất commit và tính diff giữa 2 commit."""
    collector = GitHistoryCollector()

    # Kiểm tra đọc commit từ chính repository hiện tại
    current_repo_path = str(Path(".").resolve())
    commits = collector.fetch_recent_commits(current_repo_path, limit=5)

    assert len(commits) >= 2
    assert all(c.source_type == "commit" for c in commits)
    assert commits[0].commit_sha is not None
    assert commits[0].author is not None

    # Lấy 2 commit gần nhất để kiểm tra tính năng diff
    head_sha = commits[0].commit_sha
    base_sha = commits[1].commit_sha

    files_to_delete, files_to_index = collector.get_changed_files_between_commits(
        current_repo_path, base_sha, head_sha
    )

    # Giữa 2 commit Phase 0 và Phase 1 phải có danh sách file thay đổi
    assert isinstance(files_to_delete, list)
    assert isinstance(files_to_index, list)
    assert len(files_to_index) > 0


def test_git_history_collector_fallback_pr():
    """Kiểm tra tính năng lấy PR trả về danh sách rỗng an toàn khi không có token / repo giả lập."""
    collector = GitHistoryCollector(github_token="fake_test_token_123")
    prs = collector.fetch_pull_requests("nonexistent-org/nonexistent-repo-xyz", limit=5)
    assert isinstance(prs, list)
    assert len(prs) == 0


def test_ingestion_pipeline_full_and_incremental_sync():
    """
    Tạo một Git repository tạm thời, chạy Full Indexing, sau đó chỉnh sửa file
    và chạy Incremental Sync để kiểm tra tính năng đồng bộ vi sai trên Qdrant.
    """
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir)
    repo = None

    try:
        # 1. Khởi tạo Git repo tạm
        repo = git.Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test Committer").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Tạo file code 1 và file markdown
        file1 = tmp_path / "auth.py"
        file1.write_text("def authenticate(): return True\n", encoding="utf-8")

        runbook = tmp_path / "runbook.md"
        runbook.write_text("# Sổ tay sự cố\nNội dung hướng dẫn...\n", encoding="utf-8")

        repo.index.add(["auth.py", "runbook.md"])
        commit1 = repo.index.commit("Initial commit: auth and runbook").hexsha

        # 2. Khởi tạo Pipeline với Qdrant in-memory
        embedder = HybridEmbedder()
        vector_store = QdrantStore(location=":memory:", embedder=embedder)
        pipeline = IngestionPipeline(
            code_parser=TreeSitterCodeParser(repo_name="temp_repo"),
            doc_parser=RunbookParser(),
            git_connector=GitHistoryCollector(),
            vector_store=vector_store,
        )

        # 3. Chạy Full Indexing
        full_res = pipeline.run_full_indexing(str(tmp_path))
        assert full_res["code_nodes"] >= 1
        assert full_res["doc_nodes"] >= 2  # runbook + commit

        # Kiểm tra tìm kiếm được hàm authenticate
        search1 = vector_store.hybrid_search("code_chunks", query="authenticate", top_k=2)
        assert len(search1) > 0
        assert search1[0].symbol_name == "authenticate"

        # 4. Sửa file auth.py thành hàm mới (thay thế authenticate bằng process_payment)
        file1.write_text("def process_payment(): return 'paid'\n", encoding="utf-8")
        repo.index.add(["auth.py"])
        commit2 = repo.index.commit("Update auth.py to process_payment").hexsha

        # 5. Chạy Incremental Sync
        inc_res = pipeline.run_incremental_sync(str(tmp_path), commit1, commit2)
        assert inc_res["deleted_files"] == 1
        assert inc_res["reindexed_nodes"] >= 1

        # 6. Kiểm tra lại Vector DB: Hàm cũ không còn, hàm mới xuất hiện
        search_old = vector_store.hybrid_search("code_chunks", query="authenticate", top_k=2)
        assert not any(r.symbol_name == "authenticate" for r in search_old)

        search_new = vector_store.hybrid_search("code_chunks", query="process_payment", top_k=2)
        assert len(search_new) > 0
        assert any(r.symbol_name == "process_payment" for r in search_new)
    finally:
        # Đóng repo object để giải phóng file lock trên Windows trước khi cleanup
        if repo is not None:
            repo.close()
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

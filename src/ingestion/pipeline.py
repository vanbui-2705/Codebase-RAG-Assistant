"""
Module: src.ingestion.pipeline
Chức năng: Điều phối luồng Ingestion và Indexing toàn diện:
- Full Indexing: Quét toàn bộ repository, lọc rác, parse AST code, parse docs, kéo git commits/PRs và nạp vào Qdrant.
- Incremental Sync: Dựa vào Git Diff (hoặc Webhook) để chỉ xóa các chunk cũ và re-index đúng các file bị thay đổi.
Tuân thủ Clean Architecture: Sử dụng Dependency Injection nhận các interfaces từ domain.
"""
import logging
from pathlib import Path
from typing import Any

from src.domain.interfaces import BaseCodeParser, BaseVectorStore
from src.domain.schemas import CodeNode, IncidentDocNode
from src.ingestion.doc_parser import RunbookParser
from src.ingestion.git_connector import GitHistoryCollector

logger = logging.getLogger(__name__)

# Danh sách các thư mục cần bỏ qua khi quét repository
SKIP_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "qdrant_storage",
    "tests",
    "harness",
    ".idea",
    ".vscode",
}

# Danh sách các file lock hoặc file sinh tự động cần bỏ qua
SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}

# Bản đồ ánh xạ đuôi file sang ngôn ngữ lập trình cho Tree-sitter
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
}

MAX_FILE_SIZE_BYTES = 1024 * 1024  # 1 MB


class IngestionPipeline:
    """
    Pipeline điều phối việc bóc tách dữ liệu và nạp vào Vector Database Qdrant.
    Hỗ trợ cả chế độ Full Indexing và Incremental Sync.
    """

    def __init__(
        self,
        code_parser: BaseCodeParser,
        doc_parser: RunbookParser,
        git_connector: GitHistoryCollector,
        vector_store: BaseVectorStore,
    ):
        self.code_parser = code_parser
        self.doc_parser = doc_parser
        self.git_connector = git_connector
        self.vector_store = vector_store

    def run_full_indexing(
        self,
        repo_path: str,
        docs_path: str | None = None,
        repo_full_name: str | None = None,
    ) -> dict[str, int]:
        """
        Đánh chỉ mục toàn bộ repository mã nguồn, tài liệu Runbook và lịch sử Git.

        Args:
            repo_path: Đường dẫn tới thư mục gốc mã nguồn.
            docs_path: Đường dẫn tới thư mục chứa tài liệu Markdown/Runbook (mặc định tìm trong repo).
            repo_full_name: Tên repository trên GitHub ('owner/repo') nếu muốn kéo PR.

        Returns:
            Thống kê số lượng node đã đánh chỉ mục: {"code_nodes": X, "doc_nodes": Y}
        """
        logger.info(f"=== BẮT ĐẦU FULL INDEXING CHO REPO: {repo_path} ===")
        repo_root = Path(repo_path).resolve()
        if not repo_root.is_dir():
            raise FileNotFoundError(f"Thư mục repository không tồn tại: {repo_path}")

        # 1. Quét và phân tích mã nguồn AST
        all_code_files = self._scan_code_files(repo_root)
        logger.info(f"Tìm thấy {len(all_code_files)} files mã nguồn hợp lệ cần index.")

        all_code_nodes: list[CodeNode] = []
        for file_p in all_code_files:
            rel_path = str(file_p.relative_to(repo_root)).replace("\\", "/")
            lang = EXTENSION_LANGUAGE_MAP.get(file_p.suffix.lower(), "python")
            nodes = self.code_parser.parse_file(str(file_p), language=lang)
            # Cập nhật đường dẫn tương đối chuẩn cho từng node
            for n in nodes:
                n.file_path = rel_path
            all_code_nodes.extend(nodes)

        # Upsert CodeNodes vào Qdrant collection 'code_chunks'
        if all_code_nodes:
            logger.info(f"Đang nạp {len(all_code_nodes)} code chunks vào collection 'code_chunks'...")
            self.vector_store.upsert_nodes("code_chunks", all_code_nodes)

        # 2. Quét và phân tích tài liệu Runbook / Markdown
        target_docs_dir = Path(docs_path).resolve() if docs_path else repo_root
        all_doc_files = self._scan_markdown_files(target_docs_dir)
        logger.info(f"Tìm thấy {len(all_doc_files)} files markdown/runbook.")

        all_doc_nodes: list[IncidentDocNode] = []
        for doc_p in all_doc_files:
            doc_nodes = self.doc_parser.parse_markdown(str(doc_p))
            all_doc_nodes.extend(doc_nodes)

        # 3. Thu thập Git Commits từ local repository
        git_commits = self.git_connector.fetch_recent_commits(str(repo_root), limit=50)
        all_doc_nodes.extend(git_commits)

        # 4. Thu thập Pull Requests từ GitHub API (nếu được cấu hình)
        if repo_full_name:
            pr_nodes = self.git_connector.fetch_pull_requests(repo_full_name, limit=30)
            all_doc_nodes.extend(pr_nodes)

        # Upsert IncidentDocNodes vào Qdrant collection 'incident_and_docs'
        if all_doc_nodes:
            logger.info(f"Đang nạp {len(all_doc_nodes)} doc & git nodes vào collection 'incident_and_docs'...")
            self.vector_store.upsert_nodes("incident_and_docs", all_doc_nodes)

        summary = {
            "code_nodes": len(all_code_nodes),
            "doc_nodes": len(all_doc_nodes),
        }
        logger.info(f"=== FULL INDEXING HOÀN TẤT: {summary} ===")
        return summary

    def run_incremental_sync(
        self,
        repo_path: str,
        base_commit: str,
        head_commit: str,
    ) -> dict[str, int]:
        """
        Cập nhật vi sai (Incremental Indexing) giữa 2 commit:
        - Xóa toàn bộ chunk cũ của những file bị sửa đổi hoặc bị xóa.
        - Chỉ parse lại AST/Docs và nạp chunk mới cho các file được thêm/sửa.

        Args:
            repo_path: Đường dẫn tới repository.
            base_commit: Commit SHA gốc.
            head_commit: Commit SHA mới.

        Returns:
            Thống kê kết quả: {"deleted_files": X, "reindexed_nodes": Y}
        """
        logger.info(f"=== BẮT ĐẦU INCREMENTAL SYNC ({base_commit[:7]} -> {head_commit[:7]}) ===")
        repo_root = Path(repo_path).resolve()

        # 1. Tính toán danh sách file thay đổi từ Git Diff
        files_to_delete, files_to_index = self.git_connector.get_changed_files_between_commits(
            str(repo_root), base_commit, head_commit
        )

        # Chuẩn hóa đường dẫn dạng forward slash
        normalized_delete_paths = [p.replace("\\", "/") for p in files_to_delete]

        # 2. Xóa các chunk cũ khỏi cả 2 collection trên Qdrant
        if normalized_delete_paths:
            self.vector_store.delete_by_file_paths("code_chunks", normalized_delete_paths)
            self.vector_store.delete_by_file_paths("incident_and_docs", normalized_delete_paths)

        # 3. Phân tích và index lại chỉ các file mới/sửa còn tồn tại trên đĩa
        new_code_nodes: list[CodeNode] = []
        new_doc_nodes: list[IncidentDocNode] = []

        for rel_path in files_to_index:
            norm_rel = rel_path.replace("\\", "/")
            full_file_path = repo_root / norm_rel

            if not full_file_path.is_file():
                continue

            suffix = full_file_path.suffix.lower()
            if suffix in EXTENSION_LANGUAGE_MAP:
                lang = EXTENSION_LANGUAGE_MAP[suffix]
                nodes = self.code_parser.parse_file(str(full_file_path), language=lang)
                for n in nodes:
                    n.file_path = norm_rel
                    n.last_modified_commit = head_commit
                new_code_nodes.extend(nodes)
            elif suffix == ".md":
                doc_nodes = self.doc_parser.parse_markdown(str(full_file_path))
                new_doc_nodes.extend(doc_nodes)

        # 4. Nạp lại các chunks mới
        if new_code_nodes:
            self.vector_store.upsert_nodes("code_chunks", new_code_nodes)
        if new_doc_nodes:
            self.vector_store.upsert_nodes("incident_and_docs", new_doc_nodes)

        summary = {
            "deleted_files": len(normalized_delete_paths),
            "reindexed_nodes": len(new_code_nodes) + len(new_doc_nodes),
        }
        logger.info(f"=== INCREMENTAL SYNC HOÀN TẤT: {summary} ===")
        return summary

    def _scan_code_files(self, directory: Path) -> list[Path]:
        """Duyệt đệ quy tìm tất cả các file mã nguồn hợp lệ, loại trừ các thư mục rác."""
        valid_files: list[Path] = []
        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            # Bỏ qua nếu bất kỳ thư mục cha nào nằm trong SKIP_DIRS
            if any(part in SKIP_DIRS for part in path.parts):
                continue

            # Bỏ qua các file lock
            if path.name in SKIP_FILES:
                continue

            # Bỏ qua file quá lớn
            try:
                if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue

            # Kiểm tra đuôi file có thuộc danh sách hỗ trợ không
            if path.suffix.lower() in EXTENSION_LANGUAGE_MAP:
                valid_files.append(path)

        return valid_files

    def _scan_markdown_files(self, directory: Path) -> list[Path]:
        """Duyệt đệ quy tìm tất cả các file markdown/runbook."""
        valid_docs: list[Path] = []
        for path in directory.rglob("*.md"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            valid_docs.append(path)
        return valid_docs

    def index_from_github_url(
        self,
        repo_url: str,
        docs_path: str | None = None,
        target_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Tự động clone repository từ GitHub URL và kích hoạt Full Indexing:
        - Hỗ trợ cả public lẫn private repository (tự chèn GitHub Token nếu có).
        - Parse AST mã nguồn bằng Tree-sitter, trích xuất lịch sử commits & PRs.
        - Nạp toàn bộ Named Vectors vào Qdrant.

        Args:
            repo_url: Đường dẫn GitHub (VD: 'https://github.com/owner/repo').
            docs_path: Thư mục docs riêng biệt (nếu có).
            target_dir: Thư mục clone (mặc định trong temp_repos).

        Returns:
            Thống kê kết quả: {"status": "success", "repo_full_name": ..., "cloned_path": ..., "code_nodes": X, "doc_nodes": Y}
        """
        import re
        logger.info(f"=== BẮT ĐẦU INDEX TỪ GITHUB URL: {repo_url} ===")
        # Bóc tách repo_full_name
        match = re.search(r"github\.com[/:]([\w\.-]+)/([\w\.-]+?)(?:\.git)?$", repo_url.strip())
        repo_full_name = f"{match.group(1)}/{match.group(2)}" if match else None

        cloned_path = self.git_connector.clone_repository(repo_url, target_dir=target_dir)
        summary = self.run_full_indexing(
            repo_path=cloned_path,
            docs_path=docs_path,
            repo_full_name=repo_full_name,
        )

        return {
            "status": "success",
            "repo_url": repo_url,
            "repo_full_name": repo_full_name or "unknown/repo",
            "cloned_path": cloned_path,
            "code_nodes": summary.get("code_nodes", 0),
            "doc_nodes": summary.get("doc_nodes", 0),
        }

    def index_uploaded_files(
        self,
        files: list[str | Path],
        repo_name: str = "uploaded_project",
    ) -> dict[str, int]:
        """
        Đánh chỉ mục cho danh sách các file được người dùng tải lên (kể cả file nén .zip).
        - Nếu gặp file .zip: tự động giải nén và quét toàn bộ thư mục bên trong.
        - Nếu gặp các file mã nguồn rời (.py, .ts, .js, .go, .java): phân tích AST và chunking.
        - Nếu gặp file .md: phân tích cú pháp runbook/tài liệu.

        Args:
            files: Danh sách đường dẫn các file đã lưu tạm trên máy chủ.
            repo_name: Tên định danh cho tập file tải lên.

        Returns:
            Thống kê: {"code_nodes": X, "doc_nodes": Y}
        """
        import zipfile
        import tempfile
        import shutil

        logger.info(f"=== BẮT ĐẦU INDEX TẬP FILE TẢI LÊN ({len(files)} files) ===")
        code_nodes: list[CodeNode] = []
        doc_nodes: list[IncidentDocNode] = []

        for f_item in files:
            p = Path(f_item).resolve()
            if not p.is_file():
                continue

            # Xử lý trường hợp file ZIP nén toàn bộ project
            if p.suffix.lower() == ".zip":
                temp_extract_dir = Path(tempfile.gettempdir()) / f"extracted_{p.stem}"
                temp_extract_dir.mkdir(parents=True, exist_ok=True)
                try:
                    logger.info(f"Phát hiện file ZIP: {p.name}. Đang giải nén vào {temp_extract_dir}...")
                    with zipfile.ZipFile(str(p), "r") as zip_ref:
                        zip_ref.extractall(str(temp_extract_dir))
                    zip_summary = self.run_full_indexing(str(temp_extract_dir), repo_full_name=repo_name)
                    return zip_summary
                finally:
                    pass

            # Xử lý file mã nguồn đơn lẻ
            suffix = p.suffix.lower()
            if suffix in EXTENSION_LANGUAGE_MAP:
                lang = EXTENSION_LANGUAGE_MAP[suffix]
                nodes = self.code_parser.parse_file(str(p), language=lang)
                for n in nodes:
                    n.file_path = p.name
                    n.repo_name = repo_name
                code_nodes.extend(nodes)
            elif suffix == ".md":
                docs = self.doc_parser.parse_markdown(str(p))
                doc_nodes.extend(docs)

        if code_nodes:
            logger.info(f"Nạp {len(code_nodes)} code chunks từ file tải lên vào Qdrant...")
            self.vector_store.upsert_nodes("code_chunks", code_nodes)
        if doc_nodes:
            logger.info(f"Nạp {len(doc_nodes)} doc nodes từ file tải lên vào Qdrant...")
            self.vector_store.upsert_nodes("incident_and_docs", doc_nodes)

        return {
            "code_nodes": len(code_nodes),
            "doc_nodes": len(doc_nodes),
        }

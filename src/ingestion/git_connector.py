"""
Module: src.ingestion.git_connector
Chức năng: Thu thập lịch sử Git Commits, tính toán Git Diff phục vụ Incremental Sync,
và tích hợp GitHub REST API để trích xuất Pull Requests / Discussions.
Tuân thủ Clean Architecture: Chỉ phụ thuộc vào src.domain và src.config.
"""
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
from typing import Any
import git
import httpx

from src.config.settings import get_settings
from src.domain.schemas import IncidentDocNode

logger = logging.getLogger(__name__)


class GitHistoryCollector:
    """
    Bộ thu thập và phân tích dữ liệu lịch sử Git cục bộ và GitHub API.
    Hỗ trợ bóc tách Commits, Pull Requests và xác định danh sách file thay đổi giữa 2 commit.
    """

    def __init__(self, github_token: str | None = None):
        settings = get_settings()
        self.github_token = github_token or settings.github_token
        self.github_base_url = "https://api.github.com"

    def fetch_recent_commits(self, repo_path: str, limit: int = 50) -> list[IncidentDocNode]:
        """
        Lấy danh sách các commit gần đây nhất từ local Git repository.

        Args:
            repo_path: Đường dẫn tới thư mục gốc của repository.
            limit: Số lượng commit tối đa cần lấy.

        Returns:
            Danh sách các IncidentDocNode có source_type="commit".
        """
        nodes: list[IncidentDocNode] = []
        try:
            repo = git.Repo(repo_path)
            if repo.bare:
                logger.warning(f"Repository tại {repo_path} là bare repository.")
                return []

            commits = list(repo.iter_commits(max_count=limit))
            for c in commits:
                # Lấy danh sách file bị thay đổi trong commit
                changed_files: list[str] = []
                if c.parents:
                    diffs = c.parents[0].diff(c)
                    for d in diffs:
                        if d.b_path:
                            changed_files.append(d.b_path)
                        elif d.a_path:
                            changed_files.append(d.a_path)
                else:
                    # Initial commit
                    changed_files = list(c.stats.files.keys())

                timestamp_iso = datetime.fromtimestamp(c.authored_date, tz=timezone.utc).isoformat()
                doc_id = hashlib.sha256(f"commit::{c.hexsha}".encode("utf-8")).hexdigest()

                message_lines = c.message.strip().splitlines() if c.message else [""]
                title = f"Commit: {message_lines[0][:100]}"
                content = (
                    f"Commit SHA: {c.hexsha}\n"
                    f"Tác giả: {c.author.name} <{c.author.email}>\n"
                    f"Thời gian: {timestamp_iso}\n"
                    f"Nội dung thay đổi:\n{c.message}\n"
                    f"Danh sách file ảnh hưởng: {', '.join(changed_files)}"
                )

                nodes.append(
                    IncidentDocNode(
                        doc_id=doc_id,
                        source_type="commit",
                        title=title,
                        content=content,
                        related_files=changed_files,
                        commit_sha=c.hexsha,
                        author=c.author.name,
                        timestamp=timestamp_iso,
                        url=None,
                    )
                )

            logger.info(f"Đã thu thập {len(nodes)} commits từ {repo_path}.")
        except Exception as e:
            logger.warning(f"Không thể đọc Git commits từ {repo_path}: {e}")

        return nodes

    def get_changed_files_between_commits(
        self, repo_path: str, base_sha: str, head_sha: str
    ) -> tuple[list[str], list[str]]:
        """
        So sánh Git Diff giữa 2 commit để xác định danh sách file cần xóa và cần đánh chỉ mục lại.

        Args:
            repo_path: Đường dẫn tới repository.
            base_sha: Commit SHA gốc (cũ).
            head_sha: Commit SHA đích (mới).

        Returns:
            Tuple: (files_to_delete, files_to_index)
            - files_to_delete: Các file bị sửa đổi (M) hoặc bị xóa (D).
            - files_to_index: Các file mới thêm (A) hoặc bị sửa đổi (M).
        """
        files_to_delete: set[str] = set()
        files_to_index: set[str] = set()

        try:
            repo = git.Repo(repo_path)
            base_commit = repo.commit(base_sha)
            head_commit = repo.commit(head_sha)

            diff_index = base_commit.diff(head_commit)

            for diff in diff_index:
                change_type = diff.change_type  # 'A', 'D', 'M', 'R'
                old_path = diff.a_path
                new_path = diff.b_path

                if change_type == "M":
                    # File bị chỉnh sửa: Cần xóa chunk cũ và index lại chunk mới
                    if old_path:
                        files_to_delete.add(old_path)
                    if new_path:
                        files_to_index.add(new_path)
                elif change_type == "D":
                    # File bị xóa: Chỉ cần xóa chunk cũ
                    if old_path:
                        files_to_delete.add(old_path)
                elif change_type == "A":
                    # File mới thêm: Chỉ cần index chunk mới
                    if new_path:
                        files_to_index.add(new_path)
                elif change_type == "R":
                    # File bị đổi tên: Xóa chunk ở đường dẫn cũ, nạp chunk ở đường dẫn mới
                    if old_path:
                        files_to_delete.add(old_path)
                    if new_path:
                        files_to_index.add(new_path)

            logger.info(
                f"Git diff ({base_sha[:7]}..{head_sha[:7]}): "
                f"{len(files_to_delete)} files cần xóa chunk, {len(files_to_index)} files cần index mới."
            )
        except Exception as e:
            logger.error(f"Lỗi khi so sánh Git diff giữa {base_sha} và {head_sha}: {e}")

        return sorted(list(files_to_delete)), sorted(list(files_to_index))

    def fetch_pull_requests(
        self, repo_full_name: str, state: str = "all", limit: int = 30
    ) -> list[IncidentDocNode]:
        """
        Gọi GitHub REST API để lấy danh sách Pull Requests và các file liên quan.
        Có cơ chế Graceful Fallback khi không có token hoặc bị giới hạn rate limit.

        Args:
            repo_full_name: Tên repository trên GitHub dạng 'owner/repo'.
            state: Trạng thái PR ('open', 'closed', 'all').
            limit: Số lượng PR tối đa cần lấy.

        Returns:
            Danh sách các IncidentDocNode có source_type="pull_request".
        """
        nodes: list[IncidentDocNode] = []
        if not repo_full_name or "/" not in repo_full_name:
            logger.debug(f"Bỏ qua lấy GitHub PR: repo_full_name không hợp lệ ({repo_full_name}).")
            return []

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Oncall-Codebase-RAG-Assistant",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        url = f"{self.github_base_url}/repos/{repo_full_name}/pulls"
        params = {"state": state, "per_page": min(limit, 100)}

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=headers, params=params)

                if response.status_code == 401 or response.status_code == 403:
                    logger.warning(
                        f"GitHub API bị giới hạn quyền hoặc rate limit (Status {response.status_code}). "
                        f"Bỏ qua bước lấy PR và tiếp tục với Git local."
                    )
                    return []

                if response.status_code != 200:
                    logger.warning(
                        f"Không thể lấy Pull Requests từ GitHub ({url}): "
                        f"HTTP {response.status_code} - {response.text[:200]}"
                    )
                    return []

                pr_list = response.json()
                for pr in pr_list[:limit]:
                    pr_number = pr.get("number")
                    pr_title = pr.get("title", "")
                    pr_body = pr.get("body", "") or "Không có mô tả chi tiết."
                    pr_url = pr.get("html_url")
                    author = pr.get("user", {}).get("login", "unknown")
                    created_at = pr.get("created_at")
                    merge_commit_sha = pr.get("merge_commit_sha")

                    # Lấy danh sách file thay đổi của PR nếu khả dụng
                    changed_files: list[str] = []
                    files_url = pr.get("url", "") + "/files"
                    if files_url and self.github_token:
                        try:
                            f_resp = client.get(files_url, headers=headers)
                            if f_resp.status_code == 200:
                                changed_files = [f.get("filename") for f in f_resp.json() if f.get("filename")]
                        except Exception:
                            pass

                    doc_id = hashlib.sha256(f"pr::{repo_full_name}::{pr_number}".encode("utf-8")).hexdigest()
                    content = (
                        f"Pull Request #{pr_number}: {pr_title}\n"
                        f"Người tạo: {author}\n"
                        f"Trạng thái: {pr.get('state')}\n"
                        f"Thời gian: {created_at}\n"
                        f"Mô tả / Thảo luận:\n{pr_body}\n"
                        f"Các file thay đổi: {', '.join(changed_files) if changed_files else 'Chưa đồng bộ'}"
                    )

                    nodes.append(
                        IncidentDocNode(
                            doc_id=doc_id,
                            source_type="pull_request",
                            title=f"PR #{pr_number}: {pr_title}",
                            content=content,
                            related_files=changed_files,
                            commit_sha=merge_commit_sha,
                            author=author,
                            timestamp=created_at,
                            url=pr_url,
                        )
                    )

            logger.info(f"Đã thu thập thành công {len(nodes)} Pull Requests từ {repo_full_name}.")
        except Exception as e:
            logger.warning(f"Lỗi kết nối GitHub API khi lấy PRs ({repo_full_name}): {e}")

        return nodes

    def clone_repository(self, repo_url: str, target_dir: str | None = None) -> str:
        """
        Clone một kho mã nguồn từ GitHub URL về máy cục bộ.
        Tự động chèn GitHub Token nếu có để hỗ trợ các private repository.

        Args:
            repo_url: URL GitHub (ví dụ: 'https://github.com/owner/repo' hoặc 'https://github.com/owner/repo.git').
            target_dir: Thư mục đích để lưu trữ mã nguồn (nếu None sẽ tự sinh trong temp_repos).

        Returns:
            Đường dẫn tuyệt đối tới thư mục repo đã clone.
        """
        import re
        import tempfile
        import shutil

        clean_url = repo_url.strip()
        # Chuẩn hóa tên repo
        match = re.search(r"github\.com[/:]([\w\.-]+)/([\w\.-]+?)(?:\.git)?$", clean_url)
        repo_name = f"{match.group(1)}_{match.group(2)}" if match else "cloned_repo"

        if not target_dir:
            base_dir = Path("/tmp/cloned_repos" if Path("/tmp").exists() else tempfile.gettempdir()) / "oncall_rag_repos"
            base_dir.mkdir(parents=True, exist_ok=True)
            target_path = base_dir / repo_name
        else:
            target_path = Path(target_dir).resolve()

        auth_url = clean_url
        if self.github_token and clean_url.startswith("https://"):
            # Chèn token vào https://token@github.com/...
            auth_url = clean_url.replace("https://", f"https://{self.github_token}@")

        if target_path.exists() and (target_path / ".git").is_dir():
            try:
                logger.info(f"Repository đã tồn tại tại {target_path}. Đang tiến hành git pull...")
                repo = git.Repo(str(target_path))
                origin = repo.remotes.origin
                origin.pull()
                repo.close()
                return str(target_path)
            except Exception as exc:
                logger.warning(f"Không thể git pull tại {target_path} ({exc}). Đang dọn dẹp và clone lại...")
                shutil.rmtree(target_path, ignore_errors=True)

        logger.info(f"Bắt đầu clone repository từ {clean_url} vào {target_path}...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        repo = git.Repo.clone_from(auth_url, str(target_path), depth=50)
        repo.close()
        logger.info(f"Clone hoàn tất thành công: {target_path}")
        return str(target_path)

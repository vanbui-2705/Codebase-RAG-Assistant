"""
Module: src.retrieval.chat_history_db
Chức năng: Quản lý lưu trữ bền vững (Persistent Storage) cho các phiên trò chuyện
và lịch sử hội thoại nhiều lượt (Multi-turn conversational memory) bằng SQLite cục bộ.
- Lưu trữ theo từng Repository và từng Session/Thread.
- Hỗ trợ lưu trữ tin nhắn người dùng, câu trả lời của Agent và trích dẫn mã nguồn.
- Tự động lưu và tải lại context cũ kể cả sau khi khởi động lại server hoặc refresh trình duyệt.
Tuân thủ Clean Architecture: Đặt tại tầng retrieval, tái sử dụng cho cả API và UI.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from chainlit.data.base import BaseDataLayer
    from chainlit.step import StepDict
    from chainlit.types import (
        Feedback,
        PageInfo,
        PaginatedResponse,
        Pagination,
        ThreadDict,
        ThreadFilter,
    )
    from chainlit.user import PersistedUser, User

    CHAINLIT_AVAILABLE = True
except ImportError:
    BaseDataLayer = object  # type: ignore
    CHAINLIT_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/chat_history.db")


class ChatHistoryDB:
    """
    Trình quản lý cơ sở dữ liệu SQLite cho các phiên hội thoại và ngữ cảnh trò chuyện.
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            self.db_path = DEFAULT_DB_PATH
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Tạo kết nối tới SQLite với hàng trả về dạng dict (sqlite3.Row)."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Khởi tạo cấu trúc các bảng nếu chưa tồn tại."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 1. Bảng lưu trữ phiên trò chuyện (Threads / Sessions)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_threads (
                        thread_id TEXT PRIMARY KEY,
                        repo_name TEXT NOT NULL,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                # 2. Bảng lưu trữ từng tin nhắn trong phiên (Messages)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        citations TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (thread_id) REFERENCES chat_threads (thread_id) ON DELETE CASCADE
                    )
                    """
                )
                # Tạo index tối ưu truy vấn
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON chat_messages(thread_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_threads_repo ON chat_threads(repo_name)"
                )
                conn.commit()
                logger.info(f"Khởi tạo bảng SQLite Chat History thành công tại: {self.db_path}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo SQLite database {self.db_path}: {e}")
            raise

    def create_thread(
        self,
        repo_name: str = "Codebase-RAG-Assistant",
        title: str = "Cuộc trò chuyện mới",
        thread_id: str | None = None,
    ) -> str:
        """
        Tạo một phiên hội thoại mới.

        Args:
            repo_name: Tên repository gắn liền với phiên chat.
            title: Tiêu đề tóm tắt của phiên chat.
            thread_id: Mã UUID của phiên (nếu không truyền sẽ tự sinh).

        Returns:
            thread_id đã tạo.
        """
        t_id = thread_id or str(uuid.uuid4())
        now_str = datetime.now(timezone.utc).isoformat()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chat_threads (thread_id, repo_name, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (t_id, repo_name, title, now_str, now_str),
                )
                conn.commit()
                logger.info(f"Đã tạo phiên hội thoại mới: id={t_id}, repo={repo_name}, title='{title}'")
                return t_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo thread trong SQLite: {e}")
            return t_id

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Lấy thông tin một phiên trò chuyện theo ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM chat_threads WHERE thread_id = ?", (thread_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy thread {thread_id}: {e}")
            return None

    def list_threads(
        self, repo_name: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        """
        Liệt kê danh sách các phiên trò chuyện gần nhất.

        Args:
            repo_name: Lọc theo tên repository (nếu có).
            limit: Số lượng phiên tối đa cần lấy.

        Returns:
            Danh sách các dict thông tin phiên trò chuyện sắp xếp giảm dần theo thời gian cập nhật.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if repo_name:
                    cursor.execute(
                        """
                        SELECT t.thread_id, t.repo_name, t.title, t.created_at, t.updated_at, COUNT(m.id) AS message_count
                        FROM chat_threads t
                        LEFT JOIN chat_messages m ON t.thread_id = m.thread_id
                        WHERE t.repo_name = ?
                        GROUP BY t.thread_id, t.repo_name, t.title, t.created_at, t.updated_at
                        ORDER BY t.updated_at DESC
                        LIMIT ?
                        """,
                        (repo_name, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT t.thread_id, t.repo_name, t.title, t.created_at, t.updated_at, COUNT(m.id) AS message_count
                        FROM chat_threads t
                        LEFT JOIN chat_messages m ON t.thread_id = m.thread_id
                        GROUP BY t.thread_id, t.repo_name, t.title, t.created_at, t.updated_at
                        ORDER BY t.updated_at DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách threads: {e}")
            return []

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        citations: list[str] | None = None,
    ) -> int:
        """
        Lưu một tin nhắn mới vào phiên hội thoại.

        Args:
            thread_id: ID phiên hội thoại.
            role: Vai trò ('user' hoặc 'assistant').
            content: Nội dung tin nhắn.
            citations: Danh sách trích dẫn dải dòng (nếu có).

        Returns:
            ID của dòng tin nhắn vừa được tạo.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations, ensure_ascii=False) if citations else None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO chat_messages (thread_id, role, content, citations, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (thread_id, role, content, citations_json, now_str),
                )
                msg_id = cursor.lastrowid

                # Cập nhật thời gian updated_at của thread
                cursor.execute(
                    "UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?",
                    (now_str, thread_id),
                )
                conn.commit()
                return msg_id or 0
        except Exception as e:
            logger.error(f"Lỗi khi lưu tin nhắn vào SQLite (thread {thread_id}): {e}")
            return 0

    def get_thread_messages(self, thread_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Lấy toàn bộ lịch sử tin nhắn của một phiên trò chuyện theo thứ tự thời gian.

        Args:
            thread_id: ID phiên hội thoại.
            limit: Giới hạn số lượng tin nhắn gần nhất.

        Returns:
            Danh sách các dict {"role": ..., "content": ..., "citations": ..., "created_at": ...}
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT role, content, citations, created_at FROM chat_messages
                    WHERE thread_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (thread_id, limit),
                )
                rows = cursor.fetchall()
                results: list[dict[str, Any]] = []
                for r in rows:
                    item = dict(r)
                    if item.get("citations"):
                        try:
                            item["citations"] = json.loads(item["citations"])
                        except Exception:
                            item["citations"] = []
                    else:
                        item["citations"] = []
                    results.append(item)
                return results
        except Exception as e:
            logger.error(f"Lỗi khi lấy tin nhắn của thread {thread_id}: {e}")
            return []

    def update_thread_title(self, thread_id: str, title: str) -> None:
        """Cập nhật tiêu đề hiển thị cho phiên trò chuyện."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE chat_threads SET title = ?, updated_at = ? WHERE thread_id = ?",
                    (title, datetime.now(timezone.utc).isoformat(), thread_id),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Lỗi cập nhật title cho thread {thread_id}: {e}")

    def delete_thread(self, thread_id: str) -> bool:
        """Xóa hoàn toàn một phiên trò chuyện và toàn bộ tin nhắn liên quan."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
                cursor.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))
                conn.commit()
                logger.info(f"Đã xóa phiên trò chuyện: {thread_id}")
                return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa thread {thread_id}: {e}")
            return False


# Singleton instance để tái sử dụng toàn cục
_chat_db_instance: ChatHistoryDB | None = None


def get_chat_history_db(db_path: str | Path | None = None) -> ChatHistoryDB:
    """Hàm factory lấy singleton instance của ChatHistoryDB."""
    global _chat_db_instance
    if _chat_db_instance is None or db_path is not None:
        _chat_db_instance = ChatHistoryDB(db_path=db_path)
    return _chat_db_instance


class SQLiteChainlitDataLayer(BaseDataLayer):
    """
    Data Layer chính thức kết nối SQLite với giao diện Chainlit.
    Tự động hiển thị toàn bộ Thread History lên Menu Sidebar bên trái:
    - Bấm vào thread để khôi phục toàn bộ lịch sử chat.
    - Bấm nút '+ New Chat' trên Sidebar để tạo thread mới.
    - Hỗ trợ đổi tên và xóa thread trực tiếp từ Sidebar.
    """

    def __init__(self, db: ChatHistoryDB | None = None):
        self.db = db or get_chat_history_db()

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        return PersistedUser(
            id=identifier,
            identifier=identifier,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        return PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return feedback.id or str(uuid.uuid4())

    async def create_element(self, element: Any) -> None:
        pass

    async def get_element(self, thread_id: str, element_id: str) -> Any:
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None) -> None:
        pass

    async def create_step(self, step_dict: StepDict) -> None:
        thread_id = step_dict.get("threadId")
        step_type = step_dict.get("type")
        output = step_dict.get("output", "")
        if thread_id and output and step_type in ("user_message", "assistant_message"):
            role = "user" if step_type == "user_message" else "assistant"
            meta = step_dict.get("metadata") or {}
            citations = meta.get("citations")
            self.db.add_message(thread_id=thread_id, role=role, content=output, citations=citations)

    async def update_step(self, step_dict: StepDict) -> None:
        pass

    async def delete_step(self, step_id: str) -> None:
        pass

    async def get_thread_author(self, thread_id: str) -> str:
        return "developer"

    async def delete_thread(self, thread_id: str) -> bool:
        return self.db.delete_thread(thread_id)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        threads = self.db.list_threads(limit=100)
        data: list[ThreadDict] = []
        for t in threads:
            title = t.get("title", "")
            # Bỏ qua các thread rỗng chưa có tin nhắn và vẫn mang tên mặc định
            if t.get("message_count", 0) == 0 and title in ("Phiên trò chuyện mới", "Cuộc trò chuyện mới"):
                continue
            if filters.search and filters.search.lower() not in title.lower():
                continue

            data.append(
                {
                    "id": t["thread_id"],
                    "createdAt": t["created_at"],
                    "name": title,
                    "userId": filters.userId or "developer",
                    "userIdentifier": filters.userId or "developer",
                    "tags": [t["repo_name"]] if t.get("repo_name") else [],
                    "metadata": {"repo_name": t.get("repo_name", "")},
                    "steps": [],
                    "elements": [],
                }
            )

        limit = pagination.first or 20
        sliced = data[:limit]
        has_next = len(data) > limit
        start_c = sliced[0]["id"] if sliced else None
        end_c = sliced[-1]["id"] if sliced else None

        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=start_c,
                endCursor=end_c,
            ),
            data=sliced,
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        t = self.db.get_thread(thread_id)
        if not t:
            return None

        messages = self.db.get_thread_messages(thread_id)
        steps: list[StepDict] = []
        for msg in messages:
            is_user = msg.get("role") == "user"
            steps.append(
                {
                    "id": str(msg.get("id", uuid.uuid4())),
                    "threadId": thread_id,
                    "name": "User" if is_user else "Assistant",
                    "type": "user_message" if is_user else "assistant_message",
                    "output": msg.get("content", ""),
                    "createdAt": msg.get("created_at"),
                    "streaming": False,
                    "waitForAnswer": False,
                    "isError": False,
                    "metadata": {"citations": msg.get("citations", [])},
                    "tags": [],
                    "input": "",
                    "generation": None,
                    "showInput": None,
                    "language": None,
                    "feedback": None,
                    "parentId": None,
                    "start": None,
                    "end": None,
                    "defaultOpen": None,
                    "autoCollapse": None,
                    "icon": None,
                    "command": None,
                    "modes": None,
                }
            )

        return {
            "id": t["thread_id"],
            "createdAt": t["created_at"],
            "name": t.get("title", "Cuộc trò chuyện"),
            "userId": "developer",
            "userIdentifier": "developer",
            "tags": [t.get("repo_name", "")],
            "metadata": {"repo_name": t.get("repo_name", "")},
            "steps": steps,
            "elements": [],
        }

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        if name:
            self.db.update_thread_title(thread_id, name)

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    async def get_favorite_steps(self, user_id: str) -> list[StepDict]:
        return []

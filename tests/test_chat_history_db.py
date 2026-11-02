"""
Unit tests cho module ChatHistoryDB (SQLite persistence).
"""
import pytest
from pathlib import Path
from src.retrieval.chat_history_db import ChatHistoryDB


@pytest.fixture
def temp_db(tmp_path: Path) -> ChatHistoryDB:
    db_file = tmp_path / "test_chat.db"
    return ChatHistoryDB(db_path=db_file)


def test_create_and_get_thread(temp_db: ChatHistoryDB):
    t_id = temp_db.create_thread(repo_name="test_repo", title="Kiểm thử phiên chat")
    assert t_id is not None

    thread_info = temp_db.get_thread(t_id)
    assert thread_info is not None
    assert thread_info["repo_name"] == "test_repo"
    assert thread_info["title"] == "Kiểm thử phiên chat"


def test_add_and_get_messages(temp_db: ChatHistoryDB):
    t_id = temp_db.create_thread(repo_name="my_repo", title="Hỏi đáp kiến trúc")
    temp_db.add_message(t_id, role="user", content="Kiến trúc gồm mấy tầng?")
    temp_db.add_message(
        t_id,
        role="assistant",
        content="Gồm 4 tầng.",
        citations=["src/api/app.py:L1-L10"],
    )

    messages = temp_db.get_thread_messages(t_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Kiến trúc gồm mấy tầng?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["citations"] == ["src/api/app.py:L1-L10"]


def test_list_and_delete_threads(temp_db: ChatHistoryDB):
    t1 = temp_db.create_thread(repo_name="repo_a", title="Phiên 1")
    t2 = temp_db.create_thread(repo_name="repo_b", title="Phiên 2")

    all_threads = temp_db.list_threads()
    assert len(all_threads) == 2

    repo_a_threads = temp_db.list_threads(repo_name="repo_a")
    assert len(repo_a_threads) == 1
    assert repo_a_threads[0]["thread_id"] == t1

    deleted = temp_db.delete_thread(t1)
    assert deleted is True
    assert temp_db.get_thread(t1) is None
    assert len(temp_db.list_threads()) == 1


@pytest.mark.asyncio
async def test_chainlit_data_layer_integration(temp_db: ChatHistoryDB):
    from src.retrieval.chat_history_db import SQLiteChainlitDataLayer
    from chainlit.types import Pagination, ThreadFilter
    from chainlit.user import User

    layer = SQLiteChainlitDataLayer(db=temp_db)
    user = await layer.get_user("developer")
    assert user is not None
    assert user.identifier == "developer"

    # Tạo thread và message
    t_id = temp_db.create_thread(repo_name="my_test_repo", title="Cuộc trò chuyện thử nghiệm")
    temp_db.add_message(t_id, role="user", content="Test question")
    temp_db.add_message(t_id, role="assistant", content="Test answer")

    # Kiểm tra list_threads
    res = await layer.list_threads(pagination=Pagination(first=10), filters=ThreadFilter(userId="developer"))
    assert len(res.data) == 1
    assert res.data[0]["id"] == t_id
    assert res.data[0]["name"] == "Cuộc trò chuyện thử nghiệm"

    # Kiểm tra get_thread
    thread_dict = await layer.get_thread(t_id)
    assert thread_dict is not None
    assert len(thread_dict["steps"]) == 2
    assert thread_dict["steps"][0]["output"] == "Test question"
    assert thread_dict["steps"][1]["output"] == "Test answer"

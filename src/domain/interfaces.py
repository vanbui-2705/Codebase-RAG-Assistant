"""
Module: src.domain.interfaces
Chức năng: Định nghĩa các Interface (Abstract Base Classes) để tách biệt tầng nghiệp vụ và thư viện bên thứ ba.
Tuân thủ nguyên tắc Dependency Inversion (Clean Architecture).
"""
from abc import ABC, abstractmethod
from typing import Any
from src.domain.schemas import CodeNode, RetrievedContext


class BaseCodeParser(ABC):
    """Giao diện chung cho bộ phân tích cú pháp mã nguồn theo cây cú pháp (AST)."""

    @abstractmethod
    def parse_file(self, file_path: str, language: str) -> list[CodeNode]:
        """
        Đọc và phân tích 1 file code, tách thành danh sách các CodeNode theo hàm/class.

        Args:
            file_path: Đường dẫn tới file mã nguồn.
            language: Ngôn ngữ lập trình (vd: python, typescript, go).

        Returns:
            Danh sách các CodeNode đã được gắn Contextual Header.
        """
        raise NotImplementedError


class BaseVectorStore(ABC):
    """Giao diện chung cho cơ sở dữ liệu Vector (Qdrant, Milvus, pgvector...)."""

    @abstractmethod
    def upsert_nodes(self, collection: str, nodes: list[Any]) -> None:
        """
        Lưu hoặc cập nhật danh sách node (CodeNode hoặc IncidentDocNode) vào collection.

        Args:
            collection: Tên collection lưu trữ.
            nodes: Danh sách các thực thể dữ liệu cần nạp.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_by_file_paths(self, collection: str, file_paths: list[str]) -> None:
        """
        Xóa toàn bộ các chunk cũ của những file bị thay đổi phục vụ Incremental Indexing.

        Args:
            collection: Tên collection cần xóa.
            file_paths: Danh sách đường dẫn file cần xóa chunk.
        """
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        collection: str,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[RetrievedContext]:
        """
        Thực hiện tìm kiếm lai (Hybrid Search: Dense Vector + Sparse BM25) kèm bộ lọc Payload.

        Args:
            collection: Tên collection cần tìm kiếm.
            query: Câu truy vấn văn bản của người dùng hoặc signature lỗi.
            filters: Các điều kiện lọc metadata (file_path, language, symbol_type...).
            top_k: Số lượng kết quả tối đa lấy ra.

        Returns:
            Danh sách các RetrievedContext được sắp xếp theo độ tương đồng.
        """
        raise NotImplementedError


class BaseReranker(ABC):
    """Giao diện chung cho mô hình chấm điểm và xếp hạng lại kết quả (Cross-Encoder Reranker)."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        contexts: list[RetrievedContext],
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """
        Chấm điểm lại danh sách kết quả truy xuất để lọc nhiễu và trả về top_k kết quả sát nhất.

        Args:
            query: Câu truy vấn gốc.
            contexts: Danh sách ngữ cảnh lấy ra từ bước Hybrid Search.
            top_k: Số lượng ngữ cảnh cô đọng cần giữ lại cho LLM.

        Returns:
            Danh sách ngữ cảnh đã được sắp xếp lại theo điểm relevance.
        """
        raise NotImplementedError

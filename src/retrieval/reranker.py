"""
Module: src.retrieval.reranker
Chức năng: Chấm điểm lại kết quả truy xuất bằng mô hình Cross-Encoder (BAAI/bge-reranker-v2-m3),
giúp chắt lọc Top-5 đoạn ngữ cảnh chính xác nhất từ Top-20 ứng viên sau bước Hybrid Search.
Tuân thủ Clean Architecture: Phụ thuộc vào src.domain và src.config.
"""
from functools import lru_cache
import logging
from typing import Any

from src.config.settings import get_settings
from src.domain.interfaces import BaseReranker
from src.domain.schemas import RetrievedContext

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Hiện thực BaseReranker sử dụng mô hình Cross-Encoder BAAI/bge-reranker-v2-m3.
    Đưa đồng thời cặp (query, context) qua self-attention để tính toán mức độ liên quan.
    """

    def __init__(self, model_name: str | None = None, use_gpu: bool = False):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_name
        self._reranker: Any = None
        self._is_fallback = False
        self._init_model(use_gpu)

    def _init_model(self, use_gpu: bool) -> None:
        """Khởi tạo FlagReranker hoặc chuyển sang chế độ fallback an toàn cho môi trường test."""
        try:
            from FlagEmbedding import FlagReranker

            logger.info(f"Đang tải mô hình Cross-Encoder Reranker '{self.model_name}'...")
            self._reranker = FlagReranker(self.model_name, use_fp16=use_gpu)
            logger.info(f"Tải mô hình Reranker '{self.model_name}' thành công.")
        except Exception as e:
            logger.warning(
                f"Không thể khởi tạo FlagReranker ({e}). "
                f"Kích hoạt chế độ CrossEncoderReranker Fallback (phục vụ unit test và offline mode)."
            )
            self._is_fallback = True

    def rerank(
        self,
        query: str,
        contexts: list[RetrievedContext],
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """
        Chấm điểm lại danh sách ngữ cảnh và trả về top_k kết quả có điểm relevance cao nhất.

        Args:
            query: Câu hỏi hoặc mã lỗi cần tìm.
            contexts: Danh sách ngữ cảnh ban đầu (thường là Top-20 từ Hybrid Search).
            top_k: Số lượng ngữ cảnh tốt nhất cần giữ lại cho LLM (mặc định 5).

        Returns:
            Danh sách top_k RetrievedContext đã được gán điểm mới và sắp xếp giảm dần.
        """
        if not contexts:
            return []

        if len(contexts) <= top_k and not self._reranker and self._is_fallback:
            return contexts

        pairs = [[query, ctx.content] for ctx in contexts]

        if not self._is_fallback and self._reranker is not None:
            try:
                scores = self._reranker.compute_score(pairs)
                if isinstance(scores, (float, int)):
                    scores = [scores]
                for i, score in enumerate(scores):
                    contexts[i].score = float(score)
            except Exception as e:
                logger.warning(f"Lỗi khi thực thi Cross-Encoder compute_score: {e}, dùng fallback.")
                self._compute_fallback_scores(query, contexts)
        else:
            self._compute_fallback_scores(query, contexts)

        # Sắp xếp giảm dần theo điểm score và lấy top_k
        ranked_contexts = sorted(contexts, key=lambda c: c.score, reverse=True)
        return ranked_contexts[:top_k]

    def _compute_fallback_scores(self, query: str, contexts: list[RetrievedContext]) -> None:
        """Tính điểm tương tác giữa query và content bằng thuật toán overlap mở rộng khi chạy offline/test."""
        query_terms = set(query.lower().split())

        for ctx in contexts:
            content_lower = ctx.content.lower()
            overlap_count = sum(1 for term in query_terms if term in content_lower)

            # Ưu tiên nếu tên symbol xuất hiện trực tiếp trong query
            symbol_bonus = 0.0
            if ctx.symbol_name and ctx.symbol_name.lower() in query.lower():
                symbol_bonus = 2.0

            # Ưu tiên nếu file_path xuất hiện trong query
            file_bonus = 0.0
            if ctx.file_path and ctx.file_path.lower() in query.lower():
                file_bonus = 1.5

            base_score = float(overlap_count) / max(1, len(query_terms))
            ctx.score = base_score + symbol_bonus + file_bonus


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Singleton: Chỉ khởi tạo mô hình Cross-Encoder Reranker đúng 1 lần trong suốt ứng dụng."""
    return CrossEncoderReranker()

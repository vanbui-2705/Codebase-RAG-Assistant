"""
Module: src.retrieval.reranker
Chức năng: Chấm điểm lại kết quả truy xuất bằng mô hình Cross-Encoder (BAAI/bge-reranker-v2-m3),
giúp chắt lọc Top-5 đoạn ngữ cảnh chính xác nhất từ Top-20 ứng viên sau bước Hybrid Search.
Tuân thủ Clean Architecture: Phụ thuộc vào src.domain và src.config.
"""
from functools import lru_cache
import logging
import re
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

        # Sắp xếp giảm dần theo điểm score
        ranked_contexts = sorted(contexts, key=lambda c: c.score, reverse=True)

        # Loại bỏ các chunk trùng lặp (cùng file_path và dải dòng)
        deduped: list[RetrievedContext] = []
        seen_spans: set[tuple[str, int | None, int | None]] = set()
        for ctx in ranked_contexts:
            key = (ctx.file_path, ctx.start_line, ctx.end_line)
            if key not in seen_spans:
                seen_spans.add(key)
                deduped.append(ctx)

        return deduped[:top_k]

    def _compute_fallback_scores(self, query: str, contexts: list[RetrievedContext]) -> None:
        """
        Tính điểm tương tác giữa query và content bằng thuật toán overlap mở rộng khi chạy offline/test:
        - Lọc bỏ stop words tiếng Việt (các, ở, từ, nằm, nào, là, của...).
        - Mở rộng từ đồng nghĩa chuyên ngành kỹ thuật (chunking <-> split/parse, tài liệu <-> doc/runbook).
        - Ưu tiên matching chính xác theo symbol_name và file_path.
        - Hạ điểm các file kiểm thử (harness, tests) khi tìm kiếm logic nghiệp vụ.
        """
        vietnamese_stopwords = {
            "các", "ở", "từ", "nằm", "nào", "là", "cho", "của", "trong", "với", "và",
            "những", "cái", "gì", "sao", "đâu", "được", "ra", "vào", "lại", "này", "thì",
            "đã", "sẽ", "đang", "bị", "bởi", "làm", "rồi", "khi", "có", "không", "một",
            "như", "để", "về", "thế", "đó", "ngay", "hãy", "hết", "file", "hỏi", "biết"
        }

        synonym_map = {
            "chunking": ["chunk", "chunking", "split", "parse", "parser", "tách", "phân tách", "bóc tách"],
            "chunk": ["chunk", "chunking", "split", "parse", "parser", "tách", "phân tách"],
            "tài": ["doc", "runbook", "markdown", "tài liệu", "document", "docs"],
            "liệu": ["doc", "runbook", "markdown", "tài liệu", "document", "docs"],
            "kiến": ["node", "docnode", "kiến thức", "knowledge"],
            "thức": ["node", "docnode", "kiến thức", "knowledge"],
            "parse": ["parse", "parser", "parsing", "ast", "tree_sitter", "bóc tách"],
            "ingest": ["ingest", "ingestion", "pipeline", "indexing", "nạp"],
            "retrieval": ["retrieval", "search", "hybrid", "truy xuất", "tìm kiếm"],
            "embed": ["embed", "embedding", "vector", "dense", "sparse", "nhúng"],
        }

        raw_terms = re.findall(r"\w+", query.lower())
        meaningful_terms = [t for t in raw_terms if t not in vietnamese_stopwords and len(t) > 1]
        if not meaningful_terms:
            meaningful_terms = raw_terms

        is_asking_function = any(k in query.lower() for k in ("hàm", "function", "method", "phương thức"))

        for ctx in contexts:
            content_lower = ctx.content.lower()
            fp_lower = (ctx.file_path or "").lower()
            sym_lower = (ctx.symbol_name or "").lower()
            stype = getattr(ctx, "source_type", "")

            score = 0.0
            for term in meaningful_terms:
                expansions = synonym_map.get(term, [term])
                matched_content = any(exp in content_lower for exp in expansions)
                matched_sym = any(exp in sym_lower for exp in expansions)
                matched_fp = any(exp in fp_lower for exp in expansions)

                if matched_content:
                    score += 1.5
                if matched_sym:
                    score += 4.5
                if matched_fp:
                    score += 3.5

            if is_asking_function and stype in ("function", "method"):
                score += 3.0

            # Hạ điểm các file kiểm thử và harness để ưu tiên code triển khai thực tế trong src/
            if fp_lower.startswith(("harness/", "tests/")):
                score *= 0.2

            ctx.score = score


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Singleton: Chỉ khởi tạo mô hình Cross-Encoder Reranker đúng 1 lần trong suốt ứng dụng."""
    return CrossEncoderReranker()

"""
Module: src.retrieval.embedder
Chức năng: Sinh đồng thời Dense Vectors (ngữ nghĩa 1024-d) và Sparse Vectors (BM25/Lexical)
sử dụng mô hình BAAI/bge-m3. Hỗ trợ batching, singleton pattern và fallback offline cho test.
Tuân thủ Clean Architecture: Phụ thuộc vào src.config và src.domain.
"""
from functools import lru_cache
import hashlib
import logging
import math
import re
from typing import Any

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class HybridEmbedder:
    """
    Bộ sinh Hybrid Embedding kết hợp Dense Vector và Sparse Vector.
    Mặc định sử dụng BAAI/bge-m3 qua thư viện FlagEmbedding.
    Có cơ chế tự động Fallback sang Lexical Tokenizer cho môi trường test/offline.
    """

    def __init__(self, model_name: str | None = None, use_gpu: bool = False):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.dense_dim = 1024
        self._model: Any = None
        self._is_fallback = False
        self._init_model(use_gpu)

    def _init_model(self, use_gpu: bool) -> None:
        """Khởi tạo mô hình BAAI/bge-m3 hoặc chuyển sang chế độ fallback nếu không có weights."""
        try:
            from FlagEmbedding import BGEM3FlagModel

            device = "cuda" if use_gpu else "cpu"
            logger.info(f"Đang tải mô hình Hybrid Embedding '{self.model_name}' trên thiết bị: {device}...")
            self._model = BGEM3FlagModel(self.model_name, use_fp16=use_gpu, device=device)
            logger.info(f"Tải mô hình '{self.model_name}' thành công.")
        except Exception as e:
            logger.warning(
                f"Không thể khởi tạo BGEM3FlagModel ({e}). "
                f"Kích hoạt chế độ Hybrid Embedder Fallback (phục vụ unit test và offline mode)."
            )
            self._is_fallback = True

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """
        Sinh danh sách Dense Vectors (1024 chiều) từ danh sách văn bản.

        Args:
            texts: Danh sách văn bản đầu vào.

        Returns:
            Danh sách các vector số thực 1024 chiều đã chuẩn hóa L2.
        """
        if not texts:
            return []

        if not self._is_fallback and self._model is not None:
            try:
                output = self._model.encode(
                    texts,
                    return_dense=True,
                    return_sparse=False,
                    batch_size=32,
                )
                dense_vecs = output["dense_vecs"]
                return [v.tolist() if hasattr(v, "tolist") else list(v) for v in dense_vecs]
            except Exception as e:
                logger.warning(f"Lỗi khi gọi model bge-m3 dense encode: {e}, dùng fallback")

        # Fallback generator: sinh deterministic 1024-dim dense vector dựa trên text hash
        return [self._fallback_dense_vector(t) for t in texts]

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """
        Sinh danh sách Sparse Vectors (token_id: weight) tương thích với Qdrant Sparse Index.

        Args:
            texts: Danh sách văn bản đầu vào.

        Returns:
            Danh sách dict ánh xạ {token_id: weight}.
        """
        if not texts:
            return []

        if not self._is_fallback and self._model is not None:
            try:
                output = self._model.encode(
                    texts,
                    return_dense=False,
                    return_sparse=True,
                    batch_size=32,
                )
                lexical_weights = output["lexical_weights"]
                sparse_list: list[dict[int, float]] = []
                for doc_weights in lexical_weights:
                    # Chuyển đổi string token id sang integer cho Qdrant
                    converted = {}
                    for k, v in doc_weights.items():
                        try:
                            token_id = int(k)
                        except ValueError:
                            token_id = abs(hash(k)) % (2**31 - 1)
                        converted[token_id] = float(v)
                    sparse_list.append(converted)
                return sparse_list
            except Exception as e:
                logger.warning(f"Lỗi khi gọi model bge-m3 sparse encode: {e}, dùng fallback")

        # Fallback generator: tính toán TF-IDF / BM25 lexical weights
        return [self._fallback_sparse_vector(t) for t in texts]

    def embed_query(self, query: str) -> tuple[list[float], dict[int, float]]:
        """
        Hàm tiện ích sinh cả Dense và Sparse Vector cho một câu truy vấn.

        Returns:
            Tuple gồm (dense_vector, sparse_vector_dict).
        """
        dense = self.embed_dense([query])[0]
        sparse = self.embed_sparse([query])[0]
        return dense, sparse

    def _fallback_dense_vector(self, text: str) -> list[float]:
        """Tạo dense vector 1024-d giả lập tất định (L2-normalized) khi chạy offline/test."""
        words = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dense_dim

        for word in words:
            # Băm từng từ vào các chiều vector
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dense_dim
            sign = 1.0 if (h // self.dense_dim) % 2 == 0 else -1.0
            vec[idx] += sign

        # Chuẩn hóa L2 norm
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def _fallback_sparse_vector(self, text: str) -> dict[int, float]:
        """Tạo sparse lexical weights (tần suất từ khóa) tương thích Qdrant Sparse Vector."""
        words = re.findall(r"\w+", text.lower())
        if not words:
            return {1: 1.0}

        tf: dict[int, float] = {}
        for word in words:
            token_id = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)
            tf[token_id] = tf.get(token_id, 0.0) + 1.0

        # Áp dụng log-scaling (tương tự BM25 term frequency)
        return {token_id: float(1.0 + math.log(count)) for token_id, count in tf.items()}


@lru_cache(maxsize=1)
def get_hybrid_embedder() -> HybridEmbedder:
    """Singleton: Chỉ khởi tạo mô hình Hybrid Embedding đúng 1 lần trong suốt ứng dụng."""
    return HybridEmbedder()

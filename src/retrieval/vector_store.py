"""
Module: src.retrieval.vector_store
Chức năng: Quản lý cơ sở dữ liệu Vector Qdrant với Named Vectors (Dense + Sparse),
thực thi lưu trữ, xóa theo file_path (Incremental Indexing) và Hybrid Search kết hợp RRF Fusion.
Tuân thủ Clean Architecture: Phụ thuộc vào src.domain và src.config.
"""
import logging
from typing import Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException

from src.config.settings import get_settings
from src.domain.interfaces import BaseVectorStore
from src.domain.schemas import CodeNode, IncidentDocNode, RetrievedContext
from src.retrieval.embedder import HybridEmbedder, get_hybrid_embedder

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    """
    Hiện thực BaseVectorStore cho cơ sở dữ liệu vector Qdrant.
    Hỗ trợ Named Vectors ('dense' và 'sparse') và thuật toán xếp hạng dung hợp RRF.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        embedder: HybridEmbedder | None = None,
        location: str | None = None,
    ):
        settings = get_settings()
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.api_key = api_key or settings.qdrant_api_key
        self.embedder = embedder or get_hybrid_embedder()

        # Kết nối tới Qdrant client (hỗ trợ cả :memory: cho testing)
        if location == ":memory:":
            self.client = QdrantClient(":memory:")
            logger.info("Khởi tạo Qdrant In-Memory mode cho kiểm thử.")
        else:
            try:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    timeout=10.0,
                )
                logger.info(f"Kết nối thành công tới Qdrant tại {self.host}:{self.port}")
            except Exception as e:
                logger.error(
                    f"Không thể kết nối tới Qdrant tại {self.host}:{self.port}. "
                    f"Vui lòng đảm bảo Docker container đang chạy bằng lệnh: docker-compose up -d"
                )
                raise ConnectionError(
                    f"Lỗi kết nối Qdrant: {e}. Vui lòng kiểm tra Docker container 'oncall_qdrant'."
                ) from e

        self.ensure_collections()

    def ensure_collections(self) -> None:
        """Đảm bảo các collection 'code_chunks' và 'incident_and_docs' tồn tại với Named Vectors."""
        collections = ["code_chunks", "incident_and_docs"]

        for coll in collections:
            try:
                if not self.client.collection_exists(coll):
                    logger.info(f"Đang tạo Qdrant collection '{coll}' với Named Vectors (dense + sparse)...")
                    self.client.create_collection(
                        collection_name=coll,
                        vectors_config={
                            "dense": rest.VectorParams(
                                size=self.embedder.dense_dim,
                                distance=rest.Distance.COSINE,
                            )
                        },
                        sparse_vectors_config={
                            "sparse": rest.SparseVectorParams(
                                index=rest.SparseIndexParams(on_disk=False)
                            )
                        },
                    )

                    # Tạo Payload Index để tăng tốc độ lọc metadata
                    self.client.create_payload_index(
                        collection_name=coll,
                        field_name="file_path",
                        field_schema=rest.PayloadSchemaType.KEYWORD,
                    )
                    if coll == "code_chunks":
                        self.client.create_payload_index(
                            collection_name=coll,
                            field_name="symbol_name",
                            field_schema=rest.PayloadSchemaType.KEYWORD,
                        )
                        self.client.create_payload_index(
                            collection_name=coll,
                            field_name="language",
                            field_schema=rest.PayloadSchemaType.KEYWORD,
                        )
                    else:
                        self.client.create_payload_index(
                            collection_name=coll,
                            field_name="source_type",
                            field_schema=rest.PayloadSchemaType.KEYWORD,
                        )
                    logger.info(f"Tạo collection '{coll}' và Payload Indexes hoàn tất.")
            except Exception as e:
                logger.warning(f"Lỗi khi kiểm tra hoặc tạo collection '{coll}': {e}")

    def upsert_nodes(self, collection: str, nodes: list[Any]) -> None:
        """
        Chuyển đổi danh sách CodeNode hoặc IncidentDocNode thành Qdrant Points và nạp vào DB.
        """
        if not nodes:
            return

        texts_to_embed = [
            node.context_header if hasattr(node, "context_header") and node.context_header else node.content
            for node in nodes
        ]

        # Sinh đồng thời Dense và Sparse Vectors
        dense_vecs = self.embedder.embed_dense(texts_to_embed)
        sparse_vecs = self.embedder.embed_sparse(texts_to_embed)

        points: list[rest.PointStruct] = []
        for i, node in enumerate(nodes):
            # Chuyển đổi sparse dict sang cấu trúc rest.SparseVector
            sparse_dict = sparse_vecs[i]
            sparse_vector = rest.SparseVector(
                indices=list(sparse_dict.keys()),
                values=list(sparse_dict.values()),
            )

            # Tạo point ID dạng UUID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, getattr(node, "node_id", getattr(node, "doc_id", str(i)))))

            # Thu thập payload
            payload: dict[str, Any] = node.model_dump() if hasattr(node, "model_dump") else dict(node)

            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vecs[i],
                        "sparse": sparse_vector,
                    },
                    payload=payload,
                )
            )

        # Batch upsert vào Qdrant
        self.client.upsert(collection_name=collection, points=points)
        logger.info(f"Đã upsert thành công {len(points)} points vào collection '{collection}'.")

    def delete_by_file_paths(self, collection: str, file_paths: list[str]) -> None:
        """
        Xóa toàn bộ các points của những file cụ thể phục vụ Incremental Indexing.
        """
        if not file_paths:
            return

        logger.info(f"Đang xóa các chunk cũ của {len(file_paths)} files trong collection '{collection}'...")
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=rest.FilterSelector(
                    filter=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="file_path",
                                match=rest.MatchAny(any=file_paths),
                            )
                        ]
                    )
                ),
            )
            logger.info("Xóa chunk theo file_paths thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi xóa chunk theo file_paths: {e}")

    def hybrid_search(
        self,
        collection: str,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[RetrievedContext]:
        """
        Thực hiện tìm kiếm lai (Hybrid Search: Dense + Sparse) kết hợp RRF Fusion trên Qdrant.
        """
        dense_vec, sparse_dict = self.embedder.embed_query(query)

        # Xây dựng bộ lọc Qdrant Filter từ dictionary
        qdrant_filter = self._build_qdrant_filter(filters)

        sparse_vector = rest.SparseVector(
            indices=list(sparse_dict.keys()),
            values=list(sparse_dict.values()),
        )

        try:
            # Sử dụng query_points với Prefetch và RRF Fusion
            response = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    rest.Prefetch(
                        query=dense_vec,
                        using="dense",
                        filter=qdrant_filter,
                        limit=top_k * 2,
                    ),
                    rest.Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        filter=qdrant_filter,
                        limit=top_k * 2,
                    ),
                ],
                query=rest.FusionQuery(fusion=rest.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            points = response.points
        except Exception as e:
            logger.warning(f"Lỗi khi thực hiện Qdrant RRF Fusion query: {e}. Thử fallback sang Dense search.")
            points = self.client.search(
                collection_name=collection,
                query_vector=("dense", dense_vec),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )

        contexts: list[RetrievedContext] = []
        for point in points:
            payload = point.payload or {}
            source_type = payload.get("symbol_type") or payload.get("source_type", "code")
            file_path = payload.get("file_path", "")
            content = payload.get("raw_code") or payload.get("content", "")
            start_line = payload.get("start_line")
            end_line = payload.get("end_line")
            symbol_name = payload.get("symbol_name")

            node_id = payload.get("node_id") or payload.get("doc_id") or str(point.id)

            contexts.append(
                RetrievedContext(
                    node_id=node_id,
                    source_type=source_type,
                    file_path=file_path,
                    content=content,
                    score=float(point.score) if hasattr(point, "score") and point.score is not None else 0.0,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=symbol_name,
                    metadata=payload,
                )
            )

        return contexts

    def _build_qdrant_filter(self, filters: dict[str, Any] | None) -> rest.Filter | None:
        """Chuyển đổi dictionary điều kiện lọc thành Qdrant Filter."""
        if not filters:
            return None

        must_conditions: list[rest.Condition] = []
        for key, val in filters.items():
            if isinstance(val, list):
                must_conditions.append(
                    rest.FieldCondition(key=key, match=rest.MatchAny(any=val))
                )
            else:
                must_conditions.append(
                    rest.FieldCondition(key=key, match=rest.MatchValue(value=val))
                )

        return rest.Filter(must=must_conditions) if must_conditions else None

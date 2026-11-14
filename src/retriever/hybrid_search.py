"""混合检索模块 — 融合向量检索和 BM25，支持加权融合"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from src.utils.config import get_config_section
from src.retriever.embedding import EmbeddingModel
from src.retriever.vector_store import VectorStore
from src.retriever.bm25_search import BM25Search


class HybridSearch:
    """混合检索引擎 — 支持 RRF 和加权融合"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = get_config_section("hybrid_search")

        self.enabled = config.get("enabled", True)
        self.fusion_method = config.get("fusion_method", "weighted")  # weighted or rrf
        self.vector_weight = config.get("vector_weight", 0.3)
        self.bm25_weight = config.get("bm25_weight", 0.7)
        self.rrf_k = config.get("rrf_k", 60)
        self.top_k_candidate = config.get("top_k_candidate", 20)
        self.top_k_final = config.get("top_k_final", 5)

        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(dim=self.embedding_model.get_dim())
        self.bm25_search = BM25Search()

    def load_indices(self) -> bool:
        vs_ok = self.vector_store.load()
        bm25_ok = self.bm25_search.load()
        if not vs_ok:
            logger.warning("向量索引加载失败")
        if not bm25_ok:
            logger.warning("BM25 索引加载失败")
        return vs_ok or bm25_ok

    def search(self, query: str) -> Dict[str, Any]:
        """执行混合搜索"""
        query_emb = self.embedding_model.encode_query(query)

        # 向量搜索
        distances, indices = self.vector_store.search(query_emb, self.top_k_candidate)
        vector_results = self.vector_store.get_results(indices, distances)
        for r in vector_results:
            r["search_type"] = "vector"

        # BM25 搜索
        bm25_results = self.bm25_search.search(query, self.top_k_candidate)

        # 融合
        if vector_results and bm25_results:
            if self.fusion_method == "rrf":
                fused = self._rrf_fusion(vector_results, bm25_results)
            else:
                fused = self._weighted_fusion(vector_results, bm25_results)
        elif vector_results:
            fused = vector_results[:self.top_k_final]
        else:
            fused = bm25_results[:self.top_k_final]

        return {
            "results": fused[:self.top_k_final],
            "query": {"original": query},
            "metadata": {
                "total_candidates": len(vector_results) + len(bm25_results),
                "search_type": "hybrid" if (vector_results and bm25_results) else "single",
                "fusion_method": self.fusion_method,
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
            },
        }

    def _weighted_fusion(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """加权分数融合 — BM25 权重大于向量，适合中文关键词"""
        doc_scores: Dict[str, Dict] = {}

        # 归一化向量分数
        v_scores = [r.get("score", 0) for r in vector_results]
        v_max = max(v_scores) if v_scores else 1
        v_min = min(v_scores) if v_scores else 0
        v_range = v_max - v_min if v_max != v_min else 1

        for r in vector_results:
            chunk_id = r.get("chunk_id", "?")
            norm_score = (r.get("score", 0) - v_min) / v_range
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {"result": r, "v_score": 0.0, "b_score": 0.0, "final_score": 0.0}
            doc_scores[chunk_id]["v_score"] = norm_score

        # 归一化 BM25 分数
        b_scores = [r.get("score", 0) for r in bm25_results]
        b_max = max(b_scores) if b_scores else 1
        b_min = min(b_scores) if b_scores else 0
        b_range = b_max - b_min if b_max != b_min else 1

        for r in bm25_results:
            chunk_id = r.get("chunk_id", "?")
            norm_score = (r.get("score", 0) - b_min) / b_range
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {"result": r, "v_score": 0.0, "b_score": 0.0, "final_score": 0.0}
            else:
                # 如果向量也找到了同一文档，保留向量结果的 metadata
                pass
            doc_scores[chunk_id]["b_score"] = norm_score

        # 加权求和
        for chunk_id, scores in doc_scores.items():
            scores["final_score"] = (
                self.vector_weight * scores["v_score"] +
                self.bm25_weight * scores["b_score"]
            )
            scores["result"]["score"] = scores["final_score"]
            scores["result"]["vector_score"] = scores["v_score"]
            scores["result"]["bm25_score"] = scores["b_score"]

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["final_score"], reverse=True)
        return [scores["result"] for _, scores in sorted_docs]

    def _rrf_fusion(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """RRF 融合"""
        doc_scores: Dict[str, Dict] = {}

        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.get("chunk_id", f"vec_{rank}")
            rrf_score = 1.0 / (self.rrf_k + rank)
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {"result": result, "rrf_score": 0.0}
            doc_scores[chunk_id]["rrf_score"] += rrf_score

        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.get("chunk_id", f"bm25_{rank}")
            rrf_score = 1.0 / (self.rrf_k + rank)
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {"result": result, "rrf_score": 0.0}
            doc_scores[chunk_id]["rrf_score"] += rrf_score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        final = []
        for doc_id, scores in sorted_docs:
            r = scores["result"].copy()
            r["score"] = scores["rrf_score"]
            final.append(r)
        return final

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "vector_store_size": self.vector_store.size,
            "bm25_size": self.bm25_search.size,
            "hybrid_enabled": self.enabled,
        }

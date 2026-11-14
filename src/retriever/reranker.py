"""重排序模块 - 基于 CrossEncoder 对检索结果进行精排"""

from typing import List, Dict, Any, Optional
from loguru import logger
from sentence_transformers import CrossEncoder

from src.utils.config import get_config_section


class Reranker:
    """CrossEncoder 重排序器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = get_config_section("reranker")

        self.model_name = config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_k_rerank = config.get("top_k_rerank", 10)
        self.device = config.get("device", "cpu")

        self.model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        """加载 CrossEncoder 模型"""
        logger.info(f"加载 Reranker 模型: {self.model_name}")
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker 模型加载成功")
        except Exception as e:
            logger.error(f"加载 Reranker 失败: {e}")
            logger.warning("重排序将不可用")

    def rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对检索结果重新排序"""
        if self.model is None:
            logger.warning("Reranker 不可用，返回原始结果")
            return results

        if not results:
            return results

        # 只重排 top_k 个候选
        candidates = results[:self.top_k_rerank]

        # 构建 (query, doc) 对
        pairs = [(query, r.get("content", "")) for r in candidates]

        # 打分
        scores = self.model.predict(pairs)

        # 更新分数
        for i, score in enumerate(scores):
            candidates[i]["original_score"] = candidates[i].get("score", 0)
            candidates[i]["score"] = float(score)
            candidates[i]["rerank_applied"] = True

        # 按新分数排序
        candidates.sort(key=lambda x: x["score"], reverse=True)

        logger.debug(f"重排序完成, 前3分数: {[f'{r.get('title','?')}:{r['score']:.4f}' for r in candidates[:3]]}")

        return candidates

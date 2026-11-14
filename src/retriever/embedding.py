"""Embedding 模块 — 使用 sentence-transformers，强制 HF 镜像"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.utils.config import get_config_section


class EmbeddingModel:
    """文本 Embedding 模型封装"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = get_config_section("embedding")

        self.model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.normalize = config.get("normalize", True)
        self.dim = config.get("dim", 384)

        # 强制使用 HF 镜像（国内网络必需）
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """加载模型"""
        logger.info(f"加载 Embedding 模型: {self.model_name}")

        load_kwargs = {
            "device": self.device,
            "trust_remote_code": True,
        }

        try:
            self.model = SentenceTransformer(self.model_name, **load_kwargs)
            test_emb = self.model.encode(["test"], show_progress_bar=False)
            actual_dim = test_emb.shape[1]
            if actual_dim != self.dim:
                logger.warning(f"模型实际维度 {actual_dim} 与配置 {self.dim} 不一致")
                self.dim = actual_dim
            logger.info(f"模型加载成功, 维度: {self.dim}")
        except Exception as e:
            logger.error(f"加载 {self.model_name} 失败: {e}")
            raise RuntimeError(f"无法加载 Embedding 模型: {e}") from e

    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """编码文本为向量"""
        if self.model is None:
            self._load_model()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
        )
        return np.array(embeddings).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], show_progress=False)[0]

    def get_dim(self) -> int:
        return self.dim

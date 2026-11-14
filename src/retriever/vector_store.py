"""向量数据库模块 - FAISS 索引管理"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from loguru import logger
from tqdm import tqdm

from src.utils.config import get_config_section


class VectorStore:
    """FAISS 向量存储，支持 IndexIVFPQ 近似搜索"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, dim: int = 2560):
        if config is None:
            config = get_config_section("vector_store")

        self.dim = dim
        self.index_type = config.get("index_type", "IVFPQ")
        self.nlist = config.get("nlist", 100)
        self.m = config.get("m", 64)
        self.nbits = config.get("nbits", 8)
        self.metric = config.get("metric", "cosine")

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_texts: List[str] = []
        self.is_trained = False

        self.store_dir = Path("vector_db")
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _create_index(self) -> faiss.Index:
        """创建 FAISS 索引"""
        if self.metric == "cosine":
            quantizer = faiss.IndexFlatIP(self.dim)  # IP 用于归一化向量的余弦相似度
        else:
            quantizer = faiss.IndexFlatL2(self.dim)

        nlist = min(self.nlist, 100)  # IVF 聚类中心数

        if self.index_type == "IVFPQ":
            index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, self.m, self.nbits)
        elif self.index_type == "IVFFlat":
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(self.dim)

        return index

    def _train_index(self, embeddings: np.ndarray) -> None:
        """训练 IVF 索引（需要训练数据）"""
        if not self.index:
            self.index = self._create_index()

        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            n_train = min(len(embeddings), max(self.nlist * 40, 1000))
            logger.info(f"训练 FAISS 索引, 训练样本数: {n_train}")
            self.index.train(embeddings[:n_train])
            self.is_trained = True

    def add_embeddings(self, embeddings: np.ndarray,
                       metadata: List[Dict[str, Any]],
                       texts: List[str]) -> None:
        """批量添加向量到索引"""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if self.index is None:
            self.index = self._create_index()

        # 训练 IVF 索引
        self._train_index(embeddings)

        # 添加向量
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        self.chunk_texts.extend(texts)

        logger.info(f"已添加 {len(embeddings)} 个向量, 索引总量: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最相似的 top_k 个向量"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("索引为空")
            return np.array([]), np.array([])

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        return distances[0], indices[0]

    def get_results(self, indices: np.ndarray, distances: np.ndarray) -> List[Dict[str, Any]]:
        """根据索引获取完整结果"""
        results = []
        for idx, dist in zip(indices, distances):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(dist)
            meta["content"] = self.chunk_texts[idx]
            results.append(meta)
        return results

    def save(self, name: str = "dst_faiss") -> None:
        """持久化索引和元数据"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("索引为空，跳过保存")
            return

        index_path = self.store_dir / f"{name}.index"
        meta_path = self.store_dir / f"{name}_metadata.json"
        texts_path = self.store_dir / f"{name}_texts.json"

        faiss.write_index(self.index, str(index_path))
        logger.info(f"索引已保存: {index_path} ({self.index.ntotal} 向量)")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_texts, f, ensure_ascii=False, indent=2)

    def load(self, name: str = "dst_faiss", dim: Optional[int] = None) -> bool:
        """加载持久化的索引和元数据"""
        if dim:
            self.dim = dim

        index_path = self.store_dir / f"{name}.index"
        meta_path = self.store_dir / f"{name}_metadata.json"
        texts_path = self.store_dir / f"{name}_texts.json"

        if not index_path.exists():
            logger.warning(f"索引文件不存在: {index_path}")
            return False

        self.index = faiss.read_index(str(index_path))
        logger.info(f"索引已加载: {index_path} ({self.index.ntotal} 向量)")

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        if texts_path.exists():
            with open(texts_path, "r", encoding="utf-8") as f:
                self.chunk_texts = json.load(f)

        return True

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def is_empty(self) -> bool:
        return self.size == 0

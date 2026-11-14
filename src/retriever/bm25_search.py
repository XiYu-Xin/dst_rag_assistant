"""BM25 关键词检索模块 - 基于 rank-bm25"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from loguru import logger


class BM25Tokenizer:
    """中文分词器（简易版，使用字符级 tokenization）"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """将文本分词（字符级 + 空白分割作为后备）"""
        # 简单策略：按空白分割 + 中文字符单独处理
        tokens = []
        for char in text:
            if char.isspace():
                continue
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
            elif char.isalnum():
                # 在末尾拼接待完整词
                if tokens and tokens[-1] and tokens[-1][-1].isalnum():
                    tokens[-1] += char
                else:
                    tokens.append(char)
            else:
                tokens.append(char)
        # 过滤太短的 token
        return [t for t in tokens if len(t) > 0]


class BM25Search:
    """BM25 关键词搜索引擎"""

    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []
        self.corpus: List[List[str]] = []
        self.bm25_path = Path("vector_db/dst_bm25.pkl")

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """构建 BM25 索引"""
        self.chunks = chunks
        self.corpus = [BM25Tokenizer.tokenize(c.get("content", "")) for c in chunks]
        self.index = BM25Okapi(self.corpus)
        logger.info(f"BM25 索引构建完成: {len(self.chunks)} 文档")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索"""
        if self.index is None:
            logger.warning("BM25 索引为空")
            return []

        tokenized = BM25Tokenizer.tokenize(query)
        scores = self.index.get_scores(tokenized)

        # 获取 top_k
        if top_k >= len(scores):
            top_indices = list(range(len(scores)))
        else:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            result = self.chunks[idx].copy()
            result["score"] = float(scores[idx])
            result["search_type"] = "bm25"
            results.append(result)

        return results

    def save(self) -> None:
        """保存 BM25 索引"""
        if self.index is None:
            return
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "corpus": self.corpus}, f)
        logger.info(f"BM25 索引已保存: {self.bm25_path}")

    def load(self) -> bool:
        """加载 BM25 索引"""
        if not self.bm25_path.exists():
            logger.warning(f"BM25 索引文件不存在: {self.bm25_path}")
            return False

        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.corpus = data["corpus"]
        self.index = BM25Okapi(self.corpus)
        logger.info(f"BM25 索引已加载: {len(self.chunks)} 文档")
        return True

    @property
    def size(self) -> int:
        return len(self.chunks)

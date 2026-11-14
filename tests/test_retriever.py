"""测试检索引擎模块"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever.embedding import EmbeddingModel
from src.retriever.vector_store import VectorStore
from src.retriever.bm25_search import BM25Search


def test_embedding():
    """测试 Embedding 模型"""
    model = EmbeddingModel()
    texts = ["蜘蛛是一种常见的敌对生物", "科学机器用于解锁科技"]
    embeddings = model.encode(texts)
    assert embeddings.shape == (2, model.get_dim())
    print(f"✓ Embedding 测试通过: {embeddings.shape}")


def test_vector_store():
    """测试向量存储"""
    store = VectorStore(dim=384)

    # 模拟数据
    embeddings = np.random.randn(10, 384).astype(np.float32)
    metadata = [{"chunk_id": f"test_{i}", "title": f"页面{i}"} for i in range(10)]
    texts = [f"测试内容 {i}" for i in range(10)]

    store.add_embeddings(embeddings, metadata, texts)
    assert store.size == 10

    # 搜索
    query_vec = np.random.randn(384).astype(np.float32)
    distances, indices = store.search(query_vec, top_k=3)
    assert len(indices) == 3
    print(f"✓ 向量存储测试通过: {store.size} 条")


def test_bm25():
    """测试 BM25 检索"""
    bm25 = BM25Search()
    chunks = [
        {"chunk_id": "1", "title": "蜘蛛", "content": "蜘蛛是一种敌对生物，会掉落怪物肉和蜘蛛丝。"},
        {"chunk_id": "2", "title": "科学机器", "content": "科学机器用于解锁科技配方，需要金块和木板制作。"},
        {"chunk_id": "3", "title": "冬季", "content": "冬天需要保暖，可以制作保暖石和冬帽。"},
    ]
    bm25.build_index(chunks)

    results = bm25.search("蜘蛛掉落物", top_k=2)
    assert len(results) > 0
    print(f"✓ BM25 测试通过: {len(results)} 结果, Top1: {results[0].get('title', '?')}")


if __name__ == "__main__":
    test_embedding()
    test_vector_store()
    test_bm25()
    print("\n所有测试通过!")

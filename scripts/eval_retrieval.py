"""检索效果评估脚本"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.retriever.hybrid_search import HybridSearch
from src.retriever.reranker import Reranker
from loguru import logger


# 测试查询集
TEST_QUERIES = [
    {
        "query": "蜘蛛有什么掉落物？",
        "expected_topics": ["蜘蛛", "掉落物", "怪物肉"],
    },
    {
        "query": "如何制作科学机器？",
        "expected_topics": ["科学机器", "制作", "科技"],
    },
    {
        "query": "冬天怎么保暖？",
        "expected_topics": ["冬天", "保暖", "体温"],
    },
    {
        "query": "威尔逊的技能是什么？",
        "expected_topics": ["威尔逊", "角色", "技能"],
    },
    {
        "query": "巨鹿怎么打？",
        "expected_topics": ["巨鹿", "Boss", "战斗"],
    },
]


def evaluate_retrieval():
    """评估检索效果"""
    setup_logger()

    hybrid_search = HybridSearch()
    if not hybrid_search.load_indices():
        logger.error("检索引擎加载失败，请先运行 build_index.py")
        return

    reranker = Reranker()

    logger.info("=" * 60)
    logger.info("开始检索效果评估")
    logger.info("=" * 60)

    total_time = 0
    total_queries = len(TEST_QUERIES)

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected = test["expected_topics"]

        start = time.time()
        search_result = hybrid_search.search(query)
        results = search_result.get("results", [])

        # 重排序
        if results:
            results = reranker.rerank(query, results)
        elapsed = time.time() - start
        total_time += elapsed

        # 简单评估：检查返回结果标题是否包含期望主题
        matched = 0
        result_topics = []
        for r in results[:5]:
            title = r.get("title", "")
            section = r.get("section_title", "")
            result_topics.append(f"{title}>{section}" if section else title)

            combined = (title + section).lower()
            for exp in expected:
                if exp.lower() in combined:
                    matched += 1
                    break

        recall = matched / len(expected) if expected else 0

        logger.info(f"\n查询 {i}/{total_queries}: {query}")
        logger.info(f"  耗时: {elapsed:.3f}s | 召回: {recall:.2%} | 结果数: {len(results)}")
        logger.info(f"  Top3: {result_topics[:3]}")

    avg_time = total_time / total_queries
    logger.info(f"\n{'=' * 60}")
    logger.info(f"评估完成")
    logger.info(f"  平均查询时间: {avg_time:.3f}s")
    logger.info(f"  总查询数: {total_queries}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    evaluate_retrieval()

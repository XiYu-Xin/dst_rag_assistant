"""一键构建知识库索引脚本

流程: 爬取 Wiki → 解析 HTML → 文本分块 → 向量化 → 保存索引
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger


def build_index(categories: list = None, skip_scrape: bool = False):
    """构建完整索引"""
    setup_logger()

    logger.info("=" * 60)
    logger.info("DST RAG Agent - 开始构建知识库索引")
    logger.info("=" * 60)

    # Step 1: 爬取 Wiki
    if not skip_scrape:
        from src.scraper.wiki_scraper import WikiScraper
        logger.info("[1/4] 爬取 Wiki 数据...")
        scraper = WikiScraper()
        pages = scraper.scrape(categories)
        logger.info(f"爬取完成: {len(pages)} 个页面")
    else:
        logger.info("[1/4] 跳过爬取 (--skip-scrape)")

    # Step 2: 解析 HTML
    from src.processor.html_parser import HTMLParser
    logger.info("[2/4] 解析 HTML 为结构化数据...")
    parser = HTMLParser()
    parser.parse_directory("data/raw", "data/parsed")
    logger.info(f"解析完成: {parser.stats}")

    # Step 3: 文本分块
    from src.processor.chunker import SemanticChunker
    logger.info("[3/4] 文本分块...")
    chunker = SemanticChunker()
    chunks = chunker.chunk_directory("data/parsed", "data/chunked")
    logger.info(f"分块完成: {len(chunks)} 个文本块")

    # Step 4: 向量化并保存
    from src.retriever.embedding import EmbeddingModel
    from src.retriever.vector_store import VectorStore
    from src.retriever.bm25_search import BM25Search
    import json

    logger.info("[4/4] 向量化并保存索引...")

    # 加载分块数据
    chunks_data = []
    chunks_texts = []
    chunks_meta = []
    chunks_file = Path("data/chunked/chunks.jsonl")
    if chunks_file.exists():
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line.strip())
                chunks_data.append(chunk)
                chunks_texts.append(chunk.get("content", ""))
                chunks_meta.append({
                    k: v for k, v in chunk.items() if k != "content"
                })
    else:
        logger.error(f"分块文件不存在: {chunks_file}")
        return

    # 向量化
    embedding_model = EmbeddingModel()
    logger.info(f"向量化 {len(chunks_texts)} 个文本块...")
    embeddings = embedding_model.encode(chunks_texts)

    # 保存 FAISS 索引
    vector_store = VectorStore(dim=embedding_model.get_dim())
    vector_store.add_embeddings(embeddings, chunks_meta, chunks_texts)
    vector_store.save()

    # 保存 BM25 索引
    bm25 = BM25Search()
    bm25.build_index(chunks_data)
    bm25.save()

    logger.info("=" * 60)
    logger.info("知识库索引构建完成!")
    logger.info(f"  - 页面数: {len(set(c.get('page_id', 0) for c in chunks_data))}")
    logger.info(f"  - 文本块: {len(chunks_data)}")
    logger.info(f"  - 向量维度: {embedding_model.get_dim()}")
    logger.info(f"  - 向量索引: {vector_store.size} 条")
    logger.info(f"  - BM25 索引: {bm25.size} 条")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DST RAG Agent 索引构建")
    parser.add_argument("--skip-scrape", action="store_true", help="跳过爬取步骤")
    parser.add_argument("--categories", nargs="+", help="指定爬取分类")
    args = parser.parse_args()

    build_index(categories=args.categories, skip_scrape=args.skip_scrape)


if __name__ == "__main__":
    main()

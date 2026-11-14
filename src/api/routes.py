"""API 路由定义 - 懒加载 + 轻量健康检查"""

import uuid
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import (
    ChatRequest, ChatResponse, SearchRequest, SearchResponse, HealthResponse
)

router = APIRouter(prefix="/api")

# 懒加载的全局引用
_hybrid_search = None
_reranker = None
_llm_client = None
sessions: dict = {}


def _get_hybrid_search():
    global _hybrid_search
    if _hybrid_search is None:
        from src.retriever.hybrid_search import HybridSearch
        _hybrid_search = HybridSearch()
        _hybrid_search.load_indices()
    return _hybrid_search


def _get_reranker():
    global _reranker
    if _reranker is None:
        from src.retriever.reranker import Reranker
        _reranker = Reranker()
    return _reranker


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from src.llm.deepseek_client import get_llm_client
        _llm_client = get_llm_client()
    return _llm_client


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 问答接口"""
    start_time = time.time()

    try:
        hybrid_search = _get_hybrid_search()
        reranker = _get_reranker()
        llm = _get_llm_client()

        # 1. 混合检索
        search_result = hybrid_search.search(request.query)
        results = search_result.get("results", [])

        # 2. 重排序
        if results:
            results = reranker.rerank(request.query, results)

        # 3. LLM 生成
        from src.llm.prompt_templates import (
            build_chat_messages, format_answer, NO_CONTEXT_RESPONSE, SYSTEM_PROMPT
        )
        if results:
            messages = build_chat_messages(request.query, results)
            answer = await llm.chat_async(messages, system_prompt=SYSTEM_PROMPT)
            formatted = format_answer(answer, results)
        else:
            formatted = NO_CONTEXT_RESPONSE.format(query=request.query)

        session_id = request.session_id or str(uuid.uuid4())[:8]
        elapsed = time.time() - start_time
        logger.info(f"查询完成 [{elapsed:.2f}s]: {request.query[:50]}...")

        return ChatResponse(
            answer=formatted,
            session_id=session_id,
            sources=[{
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "section_title": r.get("section_title", ""),
                "score": r.get("score", 0),
            } for r in results[:5]],
            search_metadata=search_result.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """知识搜索接口"""
    hybrid_search = _get_hybrid_search()
    reranker = _get_reranker()

    search_result = hybrid_search.search(request.query)
    results = search_result.get("results", [])

    if results:
        results = reranker.rerank(request.query, results)

    return SearchResponse(
        results=results[:request.top_k],
        total=len(results),
        query=request.query,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查 — 轻量级，不触发模型加载"""
    # 只检查索引文件是否存在，不加载模型
    from pathlib import Path
    faiss_exists = Path("vector_db/dst_faiss.index").exists()
    bm25_exists = Path("vector_db/dst_bm25.pkl").exists()

    if faiss_exists and bm25_exists:
        import os
        faiss_size = Path("vector_db/dst_faiss.index").stat().st_size
        return HealthResponse(
            status="ok",
            version="1.0.0",
            vector_store_size=faiss_size,
            bm25_size=int(bm25_exists),
        )
    else:
        return HealthResponse(
            status="no_index",
            version="1.0.0",
            vector_store_size=0,
            bm25_size=0,
        )

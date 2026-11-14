"""FastAPI 应用入口 — 启动时预热模型"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router as api_router
from src.utils.config import get_config_section


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 — 启动时预热模型"""
    from src.utils.logger import setup_logger
    setup_logger()
    logger.info("DST RAG Agent 启动中...")

    # 预热模型（避免首次请求超时）
    try:
        logger.info("预热 Embedding 模型...")
        from src.retriever.embedding import EmbeddingModel
        model = EmbeddingModel()
        _ = model.encode_query("预热测试")
        logger.info(f"Embedding 模型就绪 ({model.get_dim()}维)")

        logger.info("预热 Reranker 模型...")
        from src.retriever.reranker import Reranker
        _ = Reranker()
        logger.info("Reranker 模型就绪")

        logger.info("加载检索引擎...")
        from src.retriever.hybrid_search import HybridSearch
        hs = HybridSearch()
        if hs.load_indices():
            logger.info(f"检索引擎就绪 — {hs.stats}")
        else:
            logger.warning("检索引擎索引未找到，请先运行 build_index.py")
    except Exception as e:
        logger.error(f"模型预热失败: {e}")

    logger.info("DST RAG Agent 就绪!")
    yield
    logger.info("DST RAG Agent 已关闭")


def create_app() -> FastAPI:
    api_config = get_config_section("api")
    app = FastAPI(
        title="DST RAG Assistant",
        description="饥荒联机版 RAG 智能问答助手 API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.get("/")
    async def root():
        return {"message": "DST RAG Assistant API", "docs": "/docs"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    api_config = get_config_section("api")
    uvicorn.run(
        "src.api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("debug", False),
    )

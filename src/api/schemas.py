"""API 请求/响应数据模型"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    """对话请求"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20, description="检索结果数量")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    stream: bool = Field(default=False, description="是否流式输出")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str
    session_id: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    search_metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """搜索结果"""
    chunk_id: str
    title: str
    content: str
    score: float
    section_title: Optional[str] = None
    url: Optional[str] = None


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[Dict[str, Any]]
    total: int
    query: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    vector_store_size: int
    bm25_size: int

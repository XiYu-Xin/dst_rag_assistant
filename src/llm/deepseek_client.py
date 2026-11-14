"""DeepSeek API 客户端封装 - 兼容 OpenAI SDK"""

import os
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
from loguru import logger

from src.utils.config import get_config_section


class DeepSeekClient:
    """DeepSeek API 客户端"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = get_config_section("llm")

        self.model = os.getenv("DEEPSEEK_MODEL", config.get("model", "deepseek-chat"))
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", config.get("base_url", "https://api.deepseek.com/v1"))
        self.api_key = os.getenv("DEEPSEEK_API_KEY", config.get("api_key", ""))
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2048)

        if not self.api_key:
            logger.warning("DeepSeek API Key 未设置！请设置环境变量 DEEPSEEK_API_KEY")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(self, messages: List[Dict[str, str]],
             system_prompt: Optional[str] = None) -> str:
        """同步对话"""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"DeepSeek API 调用失败: {e}")
            return f"抱歉，AI 服务暂时不可用。错误: {str(e)}"

    async def chat_async(self, messages: List[Dict[str, str]],
                         system_prompt: Optional[str] = None) -> str:
        """异步对话"""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"DeepSeek API 异步调用失败: {e}")
            return f"抱歉，AI 服务暂时不可用。"

    async def chat_stream(self, messages: List[Dict[str, str]],
                          system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """流式对话"""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"DeepSeek 流式调用失败: {e}")
            yield f"\n\n[错误] AI 服务暂时不可用: {str(e)}"


# 全局单例
_llm_client: Optional[DeepSeekClient] = None


def get_llm_client() -> DeepSeekClient:
    """获取全局 LLM 客户端"""
    global _llm_client
    if _llm_client is None:
        _llm_client = DeepSeekClient()
    return _llm_client

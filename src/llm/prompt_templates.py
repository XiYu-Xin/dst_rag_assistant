"""Prompt 模板管理 - RAG 问答提示词"""

from typing import List, Dict, Any


# 系统提示词
SYSTEM_PROMPT = """你是《饥荒联机版》(Don't Starve Together) 的专家助手。
你的知识来源于游戏灰机 Wiki，请严格遵守以下规则：

1. **基于上下文回答**: 仅根据提供的 Wiki 上下文信息回答问题
2. **诚实**: 如果上下文信息不足以回答问题，请明确说明，不要编造或猜测
3. **引用来源**: 引用信息时注明来源页面标题
4. **结构清晰**: 使用分点、小标题等方式组织回答，便于阅读
5. **语言**: 使用中文回答，专业术语可附带英文原名
6. **游戏建议**: 涉及策略/建议时，提供多种选择并说明优劣

你是玩家的伙伴，请用友好、热情的语气交流！"""


def build_rag_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """构建 RAG 提示词"""
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title", "未知")
        section = ctx.get("section_title", "")
        content = ctx.get("content", "")
        source = f"【来源 {i}】{title}"
        if section:
            source += f" > {section}"
        context_parts.append(f"{source}\n{content}")

    context_text = "\n\n---\n\n".join(context_parts)

    prompt = f"""请根据以下 Wiki 信息回答用户问题。

## Wiki 参考信息
{context_text}

## 用户问题
{query}

## 请回答:"""

    return prompt


def build_chat_messages(query: str, contexts: List[Dict[str, Any]],
                        history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """构建完整对话消息"""
    messages = []

    # 历史对话
    if history:
        messages.extend(history)
    else:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # 当前问题 + 上下文
    user_message = build_rag_prompt(query, contexts)
    messages.append({"role": "user", "content": user_message})

    return messages


# 无相关知识时的回复
NO_CONTEXT_RESPONSE = """抱歉，我在当前的 Wiki 知识库中没有找到与「{query}」直接相关的信息。

建议:
- 尝试使用不同的关键词重新提问
- 检查问题中提到的游戏术语是否正确
- 该内容可能尚未收录到知识库中

你可以尝试问:
- 某个物品的配方和用途
- 某个生物的属性和打法
- 某个机制的详细说明
- 角色特性分析
"""


def format_answer(answer: str, sources: List[Dict[str, Any]] = None) -> str:
    """格式化最终回答"""
    if sources:
        source_lines = ["\n\n---\n📚 **参考来源**:"]
        seen = set()
        for s in sources:
            title = s.get("title", "未知")
            if title not in seen:
                seen.add(title)
                url = s.get("url", "")
                source_lines.append(f"- [{title}]({url})" if url else f"- {title}")
        answer += "\n".join(source_lines)

    return answer

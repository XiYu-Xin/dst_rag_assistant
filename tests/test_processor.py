"""测试数据处理模块"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processor.html_parser import HTMLParser
from src.processor.cleaner import TextCleaner
from src.processor.chunker import SemanticChunker


def test_cleaner():
    """测试文本清洗器"""
    cleaner = TextCleaner()

    # Wiki 标记测试
    text = "'''蜘蛛'''是一种[[生物]]，会掉落[[怪物肉]]和[[蜘蛛丝]]。详见[网址 蜘蛛攻略]。"
    cleaned = cleaner.clean(text)
    assert "蜘蛛" in cleaned
    assert "怪物肉" in cleaned
    assert "蜘蛛丝" in cleaned
    # 不应包含 Wiki 标记
    assert "'''" not in cleaned
    assert "[[" not in cleaned
    print(f"✓ 清洗测试通过: {cleaned}")


def test_chunker():
    """测试文本分块器"""
    chunker = SemanticChunker()

    # 模拟已解析数据
    parsed = {
        "page_id": 123,
        "title": "测试页面",
        "url": "https://example.com/wiki/Test",
        "categories": ["测试"],
        "sections": [
            {"level": 2, "title": "概述", "content": "这是一个测试页面的概述内容。" * 20},
            {"level": 2, "title": "属性", "content": "生命值: 100\n攻击力: 20\n掉落物: 测试物品" * 5},
        ],
        "infobox": {"生命值": "100", "攻击力": "20"},
        "full_text": "测试页面的完整文本。",
    }

    chunks = chunker.chunk_document(parsed)
    assert len(chunks) > 0, "应生成至少一个块"
    print(f"✓ 分块测试通过: {len(chunks)} 个块")

    for i, chunk in enumerate(chunks):
        print(f"  块 {i}: {chunk['chunk_id']} ({len(chunk['content'])} 字符)")


if __name__ == "__main__":
    test_cleaner()
    test_chunker()
    print("\n所有测试通过!")

"""文本清洗器 - 清理和标准化 Wiki 文本"""

import re
from typing import List, Optional


class TextCleaner:
    """文本清洗器"""

    # 常见的 Wiki 元文本模式
    PATTERNS: List = [
        (r"\[\[(?:File|Image):[^\]]+\]\]", ""),         # 文件引用
        (r"\{\{[^\}]+\}\}", ""),                           # 模板引用
        (r"<[^>]+>", ""),                                   # HTML 标签
        (r"&[a-z]+;", " "),                                 # HTML 实体
        (r"\[https?://[^\s\]]+\s?([^\]]*)\]", r"\1"),      # 外部链接
        (r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2"),           # 内部链接(有别名)
        (r"\[\[([^\]]+)\]\]", r"\1"),                       # 内部链接
        (r"'{2,5}", ""),                                     # Wiki 粗体/斜体标记
        (r"^\s*[#*:;]+\s*", "", re.MULTILINE),              # 列表标记
    ]

    def __init__(self, remove_refs: bool = True):
        self.remove_refs = remove_refs

    def clean(self, text: str) -> str:
        """完整清洗文本"""
        if not text:
            return ""

        for item in self.PATTERNS:
            if len(item) == 2:
                pattern, replacement = item
                flags = 0
            elif len(item) == 3:
                pattern, replacement, flags = item
            else:
                continue

            if isinstance(pattern, str):
                text = re.sub(pattern, replacement, text, flags=flags)
            else:
                text = re.sub(pattern[0], replacement, text, flags=pattern[2] if len(pattern) > 2 else 0)

        if self.remove_refs:
            text = re.sub(r"\[\d+\]", "", text)

        # 清理多余空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """批量清洗"""
        return [self.clean(t) for t in texts]

    @staticmethod
    def is_noise(text: str) -> bool:
        """判断是否为噪声文本"""
        if len(text) < 10:
            return True
        # 纯数字/符号/英文比例过高
        alpha_count = sum(1 for c in text if c.isascii() and c.isalpha())
        alpha_ratio = alpha_count / max(len(text), 1)
        if alpha_ratio > 0.9 and len(text) < 50:
            return True
        return False

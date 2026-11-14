"""文本分块器 - 支持语义分块和固定大小分块"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

from src.utils.config import get_config_section
from src.processor.cleaner import TextCleaner


class SemanticChunker:
    """语义分块器 - 按 Wiki 章节结构分块，保留语义完整性"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = get_config_section("chunking")

        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 64)
        self.separators = config.get("separators", ["\n\n", "\n", "。", "；", " "])
        self.min_chunk_size = config.get("min_chunk_size", 100)

        self.cleaner = TextCleaner()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def chunk_sections(self, page_id: int, title: str, url: str,
                       categories: List[str], sections: List[Dict],
                       infobox: Dict[str, str],
                       full_text: str) -> List[Dict[str, Any]]:
        """按章节分块"""
        chunks = []

        # 1. 标题块 (摘要)
        title_chunk = {
            "chunk_id": f"{page_id}_title",
            "page_id": page_id,
            "title": title,
            "url": url,
            "categories": categories,
            "section_title": "概述",
            "chunk_type": "summary",
            "content": f"条目: {title}\n" + "\n".join(f"{k}: {v}" for k, v in infobox.items()),
        }
        if len(title_chunk["content"]) >= self.min_chunk_size:
            chunks.append(title_chunk)

        # 2. 信息框块
        if infobox:
            info_lines = [f"【{title}】属性:"]
            for k, v in infobox.items():
                info_lines.append(f"  {k}: {v}")
            info_text = "\n".join(info_lines)
            if len(info_text) >= self.min_chunk_size:
                chunks.append({
                    "chunk_id": f"{page_id}_infobox",
                    "page_id": page_id,
                    "title": title,
                    "url": url,
                    "categories": categories,
                    "section_title": "属性信息",
                    "chunk_type": "infobox",
                    "content": info_text,
                })

        # 3. 章节块
        for section in sections:
            section_text = section.get("content", "").strip()
            if not section_text:
                continue

            # 先清洗再分块
            cleaned = self.cleaner.clean(section_text)
            if TextCleaner.is_noise(cleaned):
                continue

            section_title = section.get("title", "未命名")
            section_header = f"【{title}】{section_title}:\n"

            # 如果章节内容不长，直接作为一个块
            if len(cleaned) <= self.chunk_size:
                content = section_header + cleaned
                if len(content) >= self.min_chunk_size:
                    chunks.append({
                        "chunk_id": f"{page_id}_sec_{self._safe_id(section_title)}",
                        "page_id": page_id,
                        "title": title,
                        "url": url,
                        "categories": categories,
                        "section_title": section_title,
                        "chunk_type": "section",
                        "content": content,
                    })
            else:
                # 长章节递归分块
                sub_chunks = self.text_splitter.split_text(cleaned)
                for i, sub in enumerate(sub_chunks):
                    content = section_header + sub
                    if len(content) >= self.min_chunk_size:
                        chunks.append({
                            "chunk_id": f"{page_id}_sec_{self._safe_id(section_title)}_{i}",
                            "page_id": page_id,
                            "title": title,
                            "url": url,
                            "categories": categories,
                            "section_title": section_title,
                            "chunk_type": "section_part",
                            "content": content,
                        })

        return chunks

    def _safe_id(self, text: str) -> str:
        """生成安全的 ID 片段"""
        text = re.sub(r"[^\w\u4e00-\u9fff]", "_", text)
        return text[:30] if len(text) > 30 else text

    def chunk_document(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理单个已解析文档"""
        return self.chunk_sections(
            page_id=parsed_data.get("page_id", 0),
            title=parsed_data.get("title", ""),
            url=parsed_data.get("url", ""),
            categories=parsed_data.get("categories", []),
            sections=parsed_data.get("sections", []),
            infobox=parsed_data.get("infobox", {}),
            full_text=parsed_data.get("full_text", ""),
        )

    def chunk_directory(self, input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """批量分块处理"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob("*.json"))
        logger.info(f"找到 {len(files)} 个文件待分块")

        all_chunks = []
        output_file = output_path / "chunks.jsonl"

        with open(output_file, "w", encoding="utf-8") as f_out:
            for filepath in tqdm(files, desc="文本分块"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    chunks = self.chunk_document(data)
                    for chunk in chunks:
                        f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        all_chunks.append(chunk)
                except Exception as e:
                    logger.error(f"分块失败 {filepath}: {e}")

        logger.info(f"分块完成: {len(files)} 文件 → {len(all_chunks)} 块")
        return all_chunks


def main():
    chunker = SemanticChunker()
    chunks = chunker.chunk_directory("data/parsed", "data/chunked")
    print(f"共生成 {len(chunks)} 个文本块")


if __name__ == "__main__":
    main()

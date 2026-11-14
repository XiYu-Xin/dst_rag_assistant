"""HTML 解析器 - 将 Wiki HTML 转换为结构化 JSON"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup, Tag
from loguru import logger
from tqdm import tqdm


class HTMLParser:
    """HTML 解析器 - 提取结构化信息"""

    # 需要移除的无用标签
    REMOVE_TAGS = ["script", "style", "nav", "noscript", "iframe"]
    # 需要保留但提取文本的标签
    INLINE_TAGS = ["a", "b", "strong", "i", "em", "span", "code", "pre"]

    def __init__(self):
        self.stats = {"total": 0, "success": 0, "empty": 0}

    def parse_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """解析单个 Wiki 文件"""
        self.stats["total"] += 1

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"读取文件失败 {filepath}: {e}")
            return None

        html = raw_data.get("content_html", "")
        if not html:
            self.stats["empty"] += 1
            return None

        soup = BeautifulSoup(html, "lxml")

        # 提取各部分
        parsed = {
            "page_id": raw_data.get("page_id"),
            "title": raw_data.get("title", ""),
            "url": raw_data.get("url", ""),
            "categories": raw_data.get("categories", []),
            "sections": [],
            "infobox": {},
            "full_text": "",
        }

        # 提取信息框 (infobox)
        parsed["infobox"] = self._extract_infobox(soup)

        # 提取章节结构
        parsed["sections"] = self._extract_sections(soup)

        # 提取完整纯文本
        parsed["full_text"] = self._extract_full_text(soup)

        self.stats["success"] += 1
        return parsed

    def _extract_infobox(self, soup: BeautifulSoup) -> Dict[str, str]:
        """提取信息框 (右侧属性表格)"""
        infobox = {}
        infobox_elem = soup.find("table", class_="infobox")
        if not infobox_elem:
            infobox_elem = soup.find("table", class_=lambda c: c and "infobox" in c.lower() if c else False)

        if infobox_elem:
            for row in infobox_elem.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if key and value:
                        infobox[key] = value

        return infobox

    def _extract_sections(self, soup: BeautifulSoup) -> list:
        """提取章节结构 (h2/h3/h4 + 内容)"""
        sections = []
        content_div = soup.find("div", class_="mw-parser-output")
        if not content_div:
            content_div = soup

        current_section = None
        current_level = 0

        for elem in content_div.find_all(["h2", "h3", "h4", "p", "ul", "ol", "dl", "table", "div"]):
            if elem.name in ("h2", "h3", "h4"):
                # 保存上一节
                if current_section and current_section.get("content"):
                    sections.append(current_section)

                level = int(elem.name[1])
                title_elem = elem.find("span", class_="mw-headline")
                section_title = title_elem.get_text(strip=True) if title_elem else elem.get_text(strip=True)

                current_section = {
                    "level": level,
                    "title": section_title,
                    "content": "",
                }
                current_level = level
            elif current_section is not None:
                # 跳过内联的表格
                if elem.name == "table" and "infobox" in (elem.get("class", []) or []):
                    continue
                text = elem.get_text(separator=" ", strip=True)
                if text:
                    current_section["content"] += text + "\n"

        if current_section and current_section.get("content"):
            sections.append(current_section)

        return sections

    def _extract_full_text(self, soup: BeautifulSoup) -> str:
        """提取完整纯文本"""
        # 移除不需要的元素
        for tag in soup.find_all(["script", "style", "nav"]):
            tag.decompose()

        # 移除编辑链接
        for edit in soup.find_all("span", class_="mw-editsection"):
            edit.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # 清理多余空行
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def parse_directory(self, input_dir: str, output_dir: str) -> None:
        """批量解析目录下的所有文件"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob("*.json"))
        logger.info(f"找到 {len(files)} 个文件待解析")

        for filepath in tqdm(files, desc="解析 HTML"):
            parsed = self.parse_file(filepath)
            if parsed:
                out_file = output_path / filepath.name
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)

        logger.info(f"解析完成: {self.stats}")


def main():
    parser = HTMLParser()
    parser.parse_directory("data/raw", "data/parsed")


if __name__ == "__main__":
    main()

"""Wiki 爬虫模块 - 使用 MediaWiki API 获取饥荒联机版灰机 Wiki 数据"""

import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from src.utils.config import get_config_section


@dataclass
class WikiPage:
    """Wiki 页面数据结构"""
    page_id: int
    title: str
    url: str
    content_html: str
    content_text: str
    categories: List[str] = field(default_factory=list)
    namespace: int = 0
    revision_id: int = 0
    last_modified: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "title": self.title,
            "url": self.url,
            "content_html": self.content_html,
            "content_text": self.content_text,
            "categories": self.categories,
            "namespace": self.namespace,
            "revision_id": self.revision_id,
            "last_modified": self.last_modified,
        }


class WikiScraper:
    """灰机 Wiki 爬虫 - 基于 MediaWiki API"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = get_config_section("wiki")

        self.base_api = config.get("base_url", "https://dontstarve.huijiwiki.com/api.php")
        self.base_url = self.base_api.replace("/api.php", "")
        self.language = config.get("language", "zh")
        self.max_depth = config.get("max_depth", 3)
        self.request_delay = config.get("request_delay", 1.0)
        self.rate_limit = config.get("rate_limit", 5)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DST-RAG-Assistant/1.0 (https://github.com; educational project)",
            "Accept": "application/json",
        })

        self.raw_dir = Path("data/raw")
        self.parsed_dir = Path("data/parsed")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

        self.visited_pages: Set[int] = set()
        self._request_timestamps: List[float] = []

    def _rate_limit_check(self) -> None:
        """速率限制检查"""
        now = time.time()
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 1.0]
        if len(self._request_timestamps) >= self.rate_limit:
            sleep_time = 1.0 - (now - self._request_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_timestamps.append(time.time())
        time.sleep(self.request_delay)

    def _api_call(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """统一 API 调用"""
        self._rate_limit_check()
        default_params = {"format": "json", "formatversion": "2"}
        params = {**default_params, **params}
        try:
            resp = self.session.get(self.base_api, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.error(f"API 错误: {data['error']}")
                return None
            return data
        except requests.RequestException as e:
            logger.error(f"API 请求失败: {e}")
            return None

    def get_category_members(self, category: str, max_pages: int = 500) -> List[Dict[str, Any]]:
        """获取指定分类下的所有页面"""
        all_pages = []
        cmcontinue = None

        logger.info(f"获取分类: {category}")

        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": min(500, max_pages),
                "cmtype": "page",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            data = self._api_call(params)
            if not data:
                break

            members = data.get("query", {}).get("categorymembers", [])
            all_pages.extend(members)

            if "continue" in data:
                cmcontinue = data["continue"].get("cmcontinue")
            else:
                break

            if len(all_pages) >= max_pages:
                break

        logger.info(f"分类 '{category}' 获取到 {len(all_pages)} 个页面")
        return all_pages

    def get_page_content(self, page_id: int) -> Optional[WikiPage]:
        """获取单个页面的完整内容 (HTML + 纯文本)"""
        if page_id in self.visited_pages:
            return None
        self.visited_pages.add(page_id)

        # 获取解析后的 HTML
        params = {
            "action": "parse",
            "pageid": page_id,
            "prop": "text|categories|displaytitle",
        }
        data = self._api_call(params)
        if not data:
            return None

        parse = data.get("parse", {})
        html = parse.get("text", "")
        title = parse.get("title", "")

        # 提取分类
        categories = []
        for cat in parse.get("categories", []):
            cat_name = cat.get("*", "")
            if cat_name:
                categories.append(cat_name)

        # HTML → 纯文本
        soup = BeautifulSoup(html, "lxml")
        # 移除不需要的元素
        for tag in soup.find_all(["script", "style", "nav", "table.infobox"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

        # 清理多余空行
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        page = WikiPage(
            page_id=page_id,
            title=title,
            url=f"{self.base_url}/wiki/{title.replace(' ', '_')}",
            content_html=html,
            content_text=text,
            categories=categories,
            namespace=parse.get("pageid", 0),
            revision_id=parse.get("revid", 0),
            last_modified="",
        )

        return page

    def search_pages(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """搜索页面"""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
        }
        data = self._api_call(params)
        if not data:
            return []
        return data.get("query", {}).get("search", [])

    def scrape_by_categories(self, categories: Optional[List[str]] = None) -> List[WikiPage]:
        """按分类爬取所有页面"""
        if categories is None:
            categories = get_config_section("wiki").get("categories", ["生物", "物品", "食物"])

        all_pages: List[WikiPage] = []

        for category in categories:
            members = self.get_category_members(category)
            logger.info(f"开始爬取分类 '{category}' 的 {len(members)} 个页面")

            for member in tqdm(members, desc=f"类别: {category}"):
                page_id = member.get("pageid")
                title = member.get("title", "")

                # 跳过特殊页面
                if title.startswith(("File:", "Template:", "Module:", "Category:", "MediaWiki:")):
                    continue

                page = self.get_page_content(page_id)
                if page:
                    all_pages.append(page)
                    logger.debug(f"已爬取: {title}")

        return all_pages

    def save_raw(self, pages: List[WikiPage]) -> None:
        """保存原始数据为 JSON"""
        for page in tqdm(pages, desc="保存原始数据"):
            safe_title = hashlib.md5(page.title.encode()).hexdigest()[:12]
            filepath = self.raw_dir / f"{page.page_id}_{safe_title}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(page.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"已保存 {len(pages)} 个页面到 {self.raw_dir}")

    def scrape(self, categories: Optional[List[str]] = None) -> List[WikiPage]:
        """完整爬取流程"""
        logger.info("=" * 50)
        logger.info("开始爬取饥荒联机版 Wiki 数据")
        logger.info("=" * 50)

        pages = self.scrape_by_categories(categories)
        self.save_raw(pages)

        logger.info(f"爬取完成! 共获取 {len(pages)} 个页面")
        return pages


def main():
    """测试入口"""
    from src.utils.logger import setup_logger
    setup_logger()

    scraper = WikiScraper()
    pages = scraper.scrape()
    print(f"共爬取 {len(pages)} 个页面")


if __name__ == "__main__":
    main()

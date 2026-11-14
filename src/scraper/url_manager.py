"""URL 管理器 - 去重和优先级队列"""

from collections import deque
from typing import Deque, Set, Optional, Dict, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin


@dataclass(order=True)
class PrioritizedURL:
    """带优先级的 URL"""
    priority: int
    url: str = field(compare=False)
    title: str = field(compare=False, default="")
    depth: int = field(compare=False, default=0)
    category: str = field(compare=False, default="")


class URLManager:
    """URL 管理器 - 去重 + 优先级队列"""

    def __init__(self, base_url: str = "https://dontstarve.huijiwiki.com"):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.queue: Deque[PrioritizedURL] = deque()
        self.visited: Set[str] = set()
        self.in_queue: Set[str] = set()

    def normalize_url(self, url: str) -> str:
        """规范化 URL"""
        url = url.strip()
        if url.startswith("//"):
            url = "https:" + url
        elif url.startswith("/"):
            url = urljoin(self.base_url, url)
        # 移除锚点和尾部斜杠
        url = url.split("#")[0].rstrip("/")
        return url

    def is_valid(self, url: str) -> bool:
        """检查 URL 是否属于同一 wiki 域"""
        parsed = urlparse(url)
        return parsed.netloc == self.base_domain or not parsed.netloc

    def is_wiki_page(self, url: str) -> bool:
        """检查是否为 wiki 内容页"""
        parsed = urlparse(url)
        path = parsed.path

        # 排除特殊页面
        excluded = ["/api.php", "/wiki/Special:", "/wiki/File:", "/wiki/Template:",
                     "/wiki/Module:", "/wiki/Category:", "/wiki/MediaWiki:", "/wiki/User:"]
        if path.startswith("/api.php"):
            return False
        for prefix in excluded:
            if prefix in path:
                return False
        return "/wiki/" in path

    def add(self, url: str, title: str = "", depth: int = 0,
            category: str = "", priority: int = 0) -> bool:
        """添加 URL 到队列，成功返回 True"""
        url = self.normalize_url(url)

        if url in self.visited or url in self.in_queue:
            return False
        if not self.is_valid(url):
            return False

        item = PrioritizedURL(priority=priority, url=url, title=title,
                              depth=depth, category=category)
        self.queue.append(item)
        self.in_queue.add(url)
        return True

    def add_batch(self, urls: list, **kwargs) -> int:
        """批量添加 URL，返回成功添加数"""
        count = 0
        for url in urls:
            if isinstance(url, dict):
                if self.add(url.get("url", ""), url.get("title", ""), **kwargs):
                    count += 1
            elif isinstance(url, str):
                if self.add(url, **kwargs):
                    count += 1
        return count

    def get_next(self) -> Optional[PrioritizedURL]:
        """获取下一个 URL"""
        while self.queue:
            item = self.queue.popleft()
            if item.url not in self.visited:
                return item
        return None

    def mark_visited(self, url: str) -> None:
        """标记 URL 已访问"""
        url = self.normalize_url(url)
        self.visited.add(url)
        self.in_queue.discard(url)

    @property
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "queue_size": len(self.queue),
            "visited": len(self.visited),
            "in_queue": len(self.in_queue),
        }

    def is_exhausted(self) -> bool:
        return len(self.queue) == 0

    def clear(self) -> None:
        self.queue.clear()
        self.visited.clear()
        self.in_queue.clear()

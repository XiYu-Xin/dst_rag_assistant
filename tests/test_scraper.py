"""测试 Wiki 爬虫"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.wiki_scraper import WikiScraper
from src.scraper.url_manager import URLManager


def test_url_manager():
    """测试 URL 管理器"""
    manager = URLManager("https://dontstarve.huijiwiki.com")
    assert manager.is_valid("https://dontstarve.huijiwiki.com/wiki/蜘蛛")
    assert not manager.is_valid("https://google.com")
    assert manager.is_wiki_page("https://dontstarve.huijiwiki.com/wiki/蜘蛛")
    assert not manager.is_wiki_page("https://dontstarve.huijiwiki.com/wiki/Special:Random")
    print("✓ URLManager 测试通过")


def test_api_call():
    """测试 API 调用"""
    scraper = WikiScraper()
    # 测试搜索
    results = scraper.search_pages("蜘蛛", limit=5)
    assert len(results) > 0, "搜索结果不应为空"
    print(f"✓ 搜索测试通过: 找到 {len(results)} 个结果")

    # 测试分类获取
    members = scraper.get_category_members("生物", max_pages=10)
    print(f"✓ 分类获取测试通过: {len(members)} 个成员")


def test_page_content():
    """测试页面内容获取"""
    scraper = WikiScraper()
    # 先用搜索找到一个页面
    results = scraper.search_pages("蜘蛛", limit=1)
    if results:
        page_id = results[0]["pageid"]
        page = scraper.get_page_content(page_id)
        if page:
            assert page.title, "页面标题不应为空"
            assert page.content_text, "页面内容不应为空"
            print(f"✓ 页面内容测试通过: {page.title} ({len(page.content_text)} 字符)")


if __name__ == "__main__":
    test_url_manager()
    test_api_call()
    test_page_content()
    print("\n所有测试通过!")

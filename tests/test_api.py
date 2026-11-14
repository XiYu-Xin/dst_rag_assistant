"""测试 API 端点"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    """测试健康检查"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print(f"✓ 健康检查通过: {data}")


def test_root():
    """测试根路径"""
    response = client.get("/")
    assert response.status_code == 200
    print("✓ 根路径测试通过")


if __name__ == "__main__":
    test_health()
    test_root()
    print("\n所有测试通过!")

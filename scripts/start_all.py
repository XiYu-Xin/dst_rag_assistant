"""启动所有服务 - API + Streamlit 前端"""

import subprocess
import sys
import time
import os
from pathlib import Path

# 确保在项目根目录
os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path.cwd()))

from src.utils.config import get_config_section
from src.utils.logger import setup_logger, logger


def start_api(host: str = "0.0.0.0", port: int = 8000):
    """启动 FastAPI 服务"""
    logger.info(f"启动 API 服务: http://{host}:{port}")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", host,
        "--port", str(port),
    ])


def start_frontend(port: int = 8501):
    """启动 Streamlit 前端"""
    logger.info(f"启动前端: http://localhost:{port}")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", str(port),
        "--server.headless", "true",
    ])


def main():
    setup_logger()

    api_config = get_config_section("api")
    frontend_config = get_config_section("frontend")

    api_port = api_config.get("port", 8000)
    frontend_port = frontend_config.get("port", 8501)

    logger.info("=" * 50)
    logger.info("启动 DST RAG Assistant")
    logger.info("=" * 50)

    processes = []

    try:
        # 启动 API
        api_proc = start_api(port=api_port)
        processes.append(api_proc)
        time.sleep(2)

        # 启动前端
        frontend_proc = start_frontend(port=frontend_port)
        processes.append(frontend_proc)

        logger.info(f"API:    http://localhost:{api_port}")
        logger.info(f"API文档: http://localhost:{api_port}/docs")
        logger.info(f"前端:   http://localhost:{frontend_port}")
        logger.info("按 Ctrl+C 停止所有服务")

        # 等待所有进程
        for proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        logger.info("正在停止服务...")
        for proc in processes:
            proc.terminate()
            proc.wait()
        logger.info("所有服务已停止")


if __name__ == "__main__":
    main()

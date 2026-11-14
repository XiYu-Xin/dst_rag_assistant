"""日志配置模块 - 基于 loguru"""

import sys
from pathlib import Path
from loguru import logger
from .config import get_config_section


def setup_logger() -> None:
    """配置 loguru 日志"""
    config = get_config_section("logging")
    log_path = config.get("log_path", "logs/dst_rag.log")

    # 移除默认 handler
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stderr,
        level=config.get("level", "INFO"),
        format=config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"),
        colorize=True,
    )

    # 确保日志目录存在
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 文件输出 (带轮转)
    logger.add(
        str(log_file),
        level=config.get("level", "INFO"),
        format=config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"),
        rotation=config.get("rotation", "50 MB"),
        retention=config.get("retention", "7 days"),
        encoding="utf-8",
    )

    logger.info("日志系统初始化完成")


__all__ = ["setup_logger", "logger"]

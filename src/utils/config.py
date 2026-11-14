"""配置加载模块 - 从 YAML 和环境变量加载配置"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

_config_cache: Optional[Dict[str, Any]] = None


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件，支持缓存"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if config_path is None:
        config_path = get_project_root() / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 用环境变量覆盖配置
    _apply_env_overrides(config)
    _config_cache = config
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """用环境变量覆盖配置值"""
    env_mappings = {
        "DEEPSEEK_API_KEY": ("llm", "api_key"),
        "DEEPSEEK_BASE_URL": ("llm", "base_url"),
        "DEEPSEEK_MODEL": ("llm", "model"),
        "LOG_LEVEL": ("logging", "level"),
        "API_HOST": ("api", "host"),
        "API_PORT": ("api", "port"),
    }
    for env_key, config_path in env_mappings.items():
        value = os.getenv(env_key)
        if value:
            section, key = config_path
            if section in config:
                config[section][key] = value
            else:
                config[section] = {key: value}


def get_config_section(section: str) -> Dict[str, Any]:
    """获取指定配置段"""
    return load_config().get(section, {})


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """重新加载配置（清除缓存）"""
    global _config_cache
    _config_cache = None
    return load_config(config_path)

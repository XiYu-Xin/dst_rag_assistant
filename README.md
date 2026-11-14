# 饥荒联机版 RAG 智能问答助手

基于 RAG (检索增强生成) 技术的《饥荒联机版》Wiki 游戏助手。用户用自然语言提问，系统从灰机 Wiki 知识库中检索相关信息，结合 DeepSeek 大语言模型生成准确回答。

## 技术栈

- **爬虫**: requests + BeautifulSoup4 + MediaWiki API
- **数据处理**: Pandas + tiktoken
- **文本分块**: LangChain TextSplitter (语义分块)
- **Embedding**: all-MiniLM-L6-v2
- **向量数据库**: FAISS (IndexIVFPQ 近似搜索)
- **关键词检索**: rank-bm25
- **重排序**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **LLM**: DeepSeek-V4 Pro (API)
- **API 服务**: FastAPI + uvicorn
- **前端**: Streamlit
- **配置管理**: python-dotenv + YAML
- **日志**: loguru

## 项目结构

```
dst_rag_agent/
├── data/
│   ├── raw/                 # 爬取的原始HTML/JSON
│   ├── parsed/              # 解析后的结构化JSON
│   └── chunked/             # 切分后的文本块 (JSONL)
├── models/
│   ├── embedding/           # Embedding模型缓存
│   └── reranker/            # CrossEncoder模型缓存
├── vector_db/               # FAISS索引持久化
├── src/
│   ├── scraper/             # Wiki爬虫
│   ├── processor/           # 数据处理器
│   ├── retriever/           # 检索引擎
│   ├── llm/                 # LLM调用
│   ├── api/                 # FastAPI服务
│   └── utils/               # 工具模块
├── frontend/                # Streamlit界面
├── tests/                   # 单元测试
├── scripts/                 # 构建/评估脚本
├── config.yaml              # 主配置文件
├── .env.example             # 环境变量模板
├── requirements.txt         # Python依赖
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制环境变量模板
copy .env.example .env

# 编辑 .env 填入 DeepSeek API Key
# DEEPSEEK_API_KEY=sk-xxxxxxxx
```

### 3. 构建知识库

```bash
# 一键构建索引 (爬取 -> 解析 -> 分块 -> 向量化)
python scripts/build_index.py
```

### 4. 启动服务

```bash
# 启动 API + 前端
python scripts/start_all.py

# 或分别启动:
# API: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# 前端: streamlit run frontend/app.py
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/chat | 问答接口 |
| GET | /api/health | 健康检查 |
| POST | /api/search | 知识搜索 |

## 架构流程

```
用户提问 → 混合检索(向量+BM25) → 重排序 → LLM生成 → 流式输出
```

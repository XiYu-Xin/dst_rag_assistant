"""Streamlit 前端 - 饥荒联机版 RAG 助手"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests

from src.utils.config import get_config_section

st.set_page_config(
    page_title="饥荒联机版 RAG 助手",
    page_icon="🏕️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .main-header h1 { color: #FF6B35; font-size: 2.2rem; }
    .stChatMessage { padding: 1rem; border-radius: 10px; }
    .source-box { font-size: 0.8rem; padding: 0.5rem; background: #f0f0f0; border-radius: 5px; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---- Helper ----
@st.cache_resource(ttl=30)
def check_api_health(api_url: str):
    """检查 API 健康状态，带重试"""
    for attempt in range(3):
        try:
            resp = requests.get(f"{api_url}/api/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return None

# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("### ⚙️ 设置")
    api_url = st.text_input("API 地址", value="http://localhost:8000")
    top_k = st.slider("检索数量", min_value=1, max_value=10, value=5)

    st.divider()
    st.markdown("### 📊 状态")

    health = check_api_health(api_url)
    if health and health.get("status") in ("ok", "no_index"):
        st.success("🟢 API 在线")
        vs_size = health.get("vector_store_size", 0)
        bm25_ok = health.get("bm25_size", 0)
        if vs_size > 0:
            st.markdown(f"- 向量库: 已就绪")
            st.markdown(f"- BM25: {'已就绪' if bm25_ok else '未就绪'}")
        else:
            st.warning("⚠️ 索引未构建，请运行 build_index.py")
    else:
        st.error("🔴 API 离线 — 请先运行 `python scripts/start_all.py`")

    st.divider()
    st.markdown("### 💡 示例问题")
    examples = [
        "蜘蛛有哪些掉落物？",
        "如何制作科学机器？",
        "冬天应该如何生存？",
        "威尔逊有什么特殊能力？",
        "怎么打败巨鹿？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.example_query = ex

    st.divider()
    st.caption("🏕️ 饥荒联机版 RAG 助手 v1.0")

# ---- 主界面 ----
st.markdown('<div class="main-header"><h1>🏕️ 饥荒联机版 Wiki 助手</h1></div>',
            unsafe_allow_html=True)
st.caption("基于 AI 的 DST 智能问答助手 — 输入问题，获取准确的游戏攻略")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是饥荒联机版的 Wiki 助手 🏕️\n\n"
         "可以问我关于生物、物品、食物、科技、角色、生存技巧等任何游戏问题！"}
    ]
if "sources" not in st.session_state:
    st.session_state.sources = []

# 显示聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 显示来源
if st.session_state.sources:
    with st.expander("📚 参考来源", expanded=False):
        for src in st.session_state.sources:
            title = src.get("title", "未知")
            url = src.get("url", "")
            section = src.get("section_title", "")
            score = src.get("score", 0)
            label = f"{title} > {section}" if section else title
            if url:
                st.markdown(f"- [{label}]({url}) (相关度: {score:.3f})")
            else:
                st.markdown(f"- {label} (相关度: {score:.3f})")

# 用户输入
if "example_query" in st.session_state:
    prompt = st.session_state.example_query
    del st.session_state.example_query
else:
    prompt = st.chat_input("输入你的问题...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("正在检索知识库并生成回答..."):
                resp = requests.post(
                    f"{api_url}/api/chat",
                    json={"query": prompt, "top_k": top_k, "stream": False},
                    timeout=120,  # 给 LLM 足够时间
                )

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "无响应")
                sources = data.get("sources", [])
                message_placeholder.markdown(answer)
                full_response = answer
                st.session_state.sources = sources
            else:
                error_msg = f"⚠️ API 错误 ({resp.status_code}): {resp.text[:200]}"
                message_placeholder.error(error_msg)
                full_response = error_msg

        except requests.ConnectionError:
            error_msg = "⚠️ 无法连接到 API 服务。请确保已运行 `python scripts/start_all.py`"
            message_placeholder.error(error_msg)
            full_response = error_msg
        except requests.Timeout:
            error_msg = "⚠️ 请求超时。首次查询需要加载模型（约15-30秒），请重试。"
            message_placeholder.error(error_msg)
            full_response = error_msg
        except Exception as e:
            error_msg = f"⚠️ 请求失败: {str(e)}"
            message_placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 清除按钮
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🗑️ 清除对话"):
        st.session_state.messages = [
            {"role": "assistant", "content": "对话已清除，有什么可以帮你的？"}
        ]
        st.session_state.sources = []
        st.rerun()

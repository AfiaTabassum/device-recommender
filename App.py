"""
app.py — Streamlit chatbot interface for the Laptop/Smartphone Recommender Agent.
"""

import os
import re
import streamlit as st

# ── Inject Streamlit secrets into environment before any other import ──
for key in ["ZILLIZ_URI", "ZILLIZ_TOKEN", "GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

import agent  # noqa: E402  (must come after env injection)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Device Recommender",
    page_icon="💻",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════
# MODELS  (all models mentioned in LS_research_agent())
# ═══════════════════════════════════════════════════════════════
MODELS = {
    "GPT-OSS 120B  (Groq)":               "groq:openai/gpt-oss-120b",
    "Gemini 3.5 Flash  (Google)":          "gemini-3.5-flash",
    "NVIDIA Nemotron 120B  (OpenRouter)":  "nvidia/nemotron-3-super-120b-a12b",
    "GPT-OSS 120B  (OpenRouter)":          "openai/gpt-oss-120b",
}

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def render_markdown_bold(text: str) -> str:
    """
    Ensure **text** patterns always render as bold in st.markdown.
    (st.markdown already handles this, but we sanitise stray single-star
    patterns that could break rendering.)
    """
    return text


def format_tool_call(name: str, args: str) -> str:
    return f"🔧 **Tool called:** `{name}`\n```json\n{args}\n```"


def format_tool_result(result) -> str:
    import json
    try:
        pretty = json.dumps(result, indent=2, ensure_ascii=False)
    except Exception:
        pretty = str(result)
    return f"```json\n{pretty[:3000]}{'...' if len(pretty) > 3000 else ''}\n```"


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    selected_label = st.selectbox(
        "🤖 Model",
        options=list(MODELS.keys()),
        index=0,
        help="Choose the LLM that powers the agent.",
    )
    selected_model = MODELS[selected_label]

    st.markdown("---")
    st.markdown("**🔍 Debug Options**")

    show_thinking = st.toggle(
        "Show tool calls",
        value=False,
        help="Display which search tool the agent called and with what arguments.",
    )
    show_tool_results = st.toggle(
        "Show tool results",
        value=False,
        help="Display the raw database records returned by each tool call.",
    )

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Searches a Zilliz/Milvus database of **1 949 devices** "
        "(Laptops, Smartphones, Gaming Laptops, Ultrabooks, Feature Phones)."
    )

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role", "content", "extras"?}

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.title("💻📱 Laptop & Smartphone Recommender")
st.markdown(
    "Ask me to find a laptop or smartphone. I'll search the database and "
    "explain why each recommendation fits your needs."
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# CHAT HISTORY
# ═══════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show debug panels if they were recorded and toggles are on
        if msg["role"] == "assistant" and "extras" in msg:
            for extra in msg["extras"]:
                if extra["type"] == "tool_call" and show_thinking:
                    with st.expander("🔧 Tool call", expanded=False):
                        st.markdown(format_tool_call(extra["name"], extra["args"]))
                elif extra["type"] == "tool_result" and show_tool_results:
                    with st.expander("📦 Tool result", expanded=False):
                        st.markdown(format_tool_result(extra["result"]))

# ═══════════════════════════════════════════════════════════════
# MONKEY-PATCH utils so agent log calls stream into Streamlit
# ═══════════════════════════════════════════════════════════════
import utils as _utils  # noqa: E402

_tool_call_log: list[dict] = []   # accumulates during one agent run

_orig_log_tool_call  = _utils.log_tool_call_html
_orig_log_tool_result = _utils.log_tool_result_html
_orig_log_title       = _utils.log_agent_title_html
_orig_log_summary     = _utils.log_final_summary_html


def _st_log_tool_call(name, args, **kw):
    _tool_call_log.append({"type": "tool_call", "name": name, "args": args})


def _st_log_tool_result(result, **kw):
    _tool_call_log.append({"type": "tool_result", "result": result})


def _st_noop(*a, **kw):
    pass


_utils.log_tool_call_html   = _st_log_tool_call
_utils.log_tool_result_html = _st_log_tool_result
_utils.log_agent_title_html = _st_noop
_utils.log_final_summary_html = _st_noop

# ═══════════════════════════════════════════════════════════════
# CHAT INPUT
# ═══════════════════════════════════════════════════════════════
if prompt := st.chat_input("e.g. 'Best gaming laptop under 120000 BDT with RTX 4060'"):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("⏳ *Searching the database…*")

        _tool_call_log.clear()

        try:
            answer = agent.LS_research_agent(
                user_question=prompt,
                model=selected_model,
                verbose=True,
                show_thinking=True,       # always capture; sidebar toggle controls display
                show_tool_results=True,   # always capture; sidebar toggle controls display
                show=False,
            )
        except Exception as e:
            answer = f"❌ **Error:** {e}"

        status_placeholder.empty()

        # Render the final answer (st.markdown handles **bold** natively)
        st.markdown(answer)

        # Show debug expanders inline (respects sidebar toggles)
        for extra in _tool_call_log:
            if extra["type"] == "tool_call" and show_thinking:
                with st.expander("🔧 Tool call", expanded=False):
                    st.markdown(format_tool_call(extra["name"], extra["args"]))
            elif extra["type"] == "tool_result" and show_tool_results:
                with st.expander("📦 Tool result", expanded=False):
                    st.markdown(format_tool_result(extra["result"]))

    # Persist to session state (including debug extras for re-render)
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "extras":  list(_tool_call_log),
    })
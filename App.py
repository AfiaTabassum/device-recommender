"""
app.py — Tabbed workflow engine interface.
Eliminates all sidebar buttons/toggles; tool calls and tool results 
are persistently isolated and displayed inside the secondary tab.
"""

import os
import streamlit as st
import streamlit.components.v1 as components

# ── Inject Streamlit secrets into environment before any other import ──
for key in ["ZILLIZ_URI", "ZILLIZ_TOKEN", "GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# ── Patch aisuite bug safely ────────────────────────────────────────────────
try:
    import aisuite.client as _aisuite_client
    if not getattr(_aisuite_client, "MCP_AVAILABLE", True):
        _aisuite_client.is_mcp_config = lambda x: False
except Exception:
    pass

import agent  
import utils as _utils  

# ── Initialise Milvus now that credentials are in os.environ ──
import laptop_SP_search_tools as _lst  
if not _lst._milvus_ok:
    _lst._init_milvus()

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Hardware Procurement Engine",
    page_icon="💻",
    layout="wide",
)

MODELS = {
    "GPT-OSS 120B  (Groq)":               "groq:openai/gpt-oss-120b",
    "Gemini 3.5 Flash  (Google)":          "gemini-3.5-flash",
    "NVIDIA Nemotron 120B  (OpenRouter)":  "nvidia/nemotron-3-super-120b-a12b",
    "GPT-OSS 120B  (OpenRouter)":          "openai/gpt-oss-120b",
}

# ═══════════════════════════════════════════════════════════════
# STATE RETENTION PERSISTENCE
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_response" not in st.session_state:
    st.session_state.current_response = None
if "captured_logs" not in st.session_state:
    st.session_state.captured_logs = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR RE-INDEX UTILITY CONTROL
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Engine Architecture")
    st.markdown("---")

    selected_label = st.selectbox(
        "🤖 Processing Intelligence Model",
        options=list(MODELS.keys()),
        index=0,
    )
    selected_model = MODELS[selected_label]

    st.markdown("---")
    if st.button("🗑️ Clear Cache & Logs", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_response = None
        st.session_state.captured_logs = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# CRITICAL FIX: STATIC DELEGATED INTERCEPTION CHANNELS
# ═══════════════════════════════════════════════════════════════
class HTMLCaptureFrame:
    def __init__(self, raw_html_string):
        self.raw_html_string = raw_html_string

def stream_interceptor_sink(target_object):
    """Intercepts utility HTML payloads and routes them sequentially into global state storage."""
    if hasattr(target_object, 'raw_html_string'):
        st.session_state.captured_logs.append(target_object.raw_html_string)
    elif isinstance(target_object, str):
        st.session_state.captured_logs.append(target_object)

# Bind interception pipes directly before user execution loops fire
_utils.display = lambda wrapper_obj: stream_interceptor_sink(wrapper_obj)
_utils.HTML = lambda structural_str: HTMLCaptureFrame(structural_str)

# Map core function pointer names to our safe state wrappers
_utils.log_tool_call_html = lambda name, args: stream_interceptor_sink(
    f"<div style='border-left:4px solid #1976D2;padding:.8em;margin:1em 0;background-color:#e3f2fd;color:#0D47A1;font-family:sans-serif;'><strong>🔧 Tool Call:</strong> <code>{name}</code><pre style='background:#f4f4f4;padding:8px;border-radius:4px;'>{args}</pre></div>"
)
_utils.log_tool_result_html = lambda result: stream_interceptor_sink(
    f"<div style='border-left:4px solid #558B2F;padding:.8em;margin:1em 0;background-color:#f1f8e9;color:#33691E;'><strong>✅ Tool Result:</strong><pre style='white-space:pre-wrap;font-size:13px;color:#2E7D32;'>{result}</pre></div>"
)

# ═══════════════════════════════════════════════════════════════
# DASHBOARD INPUT INTERFACE
# ═══════════════════════════════════════════════════════════════
st.title("💻📱 Laptop & Smartphone Recommender")
st.markdown("Input your technical deployment specifications below to compile tailored hardware reports.")
st.markdown("---")

# Render historical messaging chains
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ═══════════════════════════════════════════════════════════════
# AGENT INTERACTIVE RUN BOUNDARY
# ═══════════════════════════════════════════════════════════════
if prompt := st.chat_input("e.g. 'Best gaming laptop under 120000 BDT with RTX 4060'"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.captured_logs = []
    st.session_state.current_response = None
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⏳ *Executing multi-turn agent pipeline and vector table search...*"):
            try:
                # Fire core agent logic loop directly
                answer = agent.LS_research_agent(
                    user_question=prompt,
                    model=selected_model,
                    verbose=True,
                    show_thinking=True,       
                    show_tool_results=True,   
                    show=False,
                )
                st.session_state.current_response = answer
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as execution_failure:
                err_text = f"❌ **Pipeline Execution Failure:** {execution_failure}"
                st.session_state.current_response = err_text
                st.session_state.messages.append({"role": "assistant", "content": err_text})
                
    st.rerun()

# ═══════════════════════════════════════════════════════════════
# PERSISTENT TABS DISPATCH MATRICES
# ═══════════════════════════════════════════════════════════════
if st.session_state.current_response is not None:
    st.markdown("---")
    
    # Structural layout separation matching your parameters
    home_tab, tools_tab = st.tabs(["🏠 Recommendation Home", "🛠️ Tool Call Records"])
    
    with home_tab:
        st.subheader("💡 System Evaluation Output")
        st.markdown(st.session_state.current_response)
        
    with tools_tab:
        st.subheader("📦 Database Query Execution Logs")
        
        if st.session_state.captured_logs:
            # Stream all logs sequentially without any button dependencies
            for html_payload in st.session_state.captured_logs:
                is_large_result = "Tool Result" in html_payload
                with st.container():
                    # Calculate tracking micro-frame box constraints safely
                    frame_height = 500 if is_large_result else 200
                    components.html(html_payload, height=frame_height, scrolling=True)
        else:
            st.info("No underlying vector tool transactions were captured for this instruction block.")

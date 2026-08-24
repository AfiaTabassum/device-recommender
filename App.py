"""
app.py — Tabbed workflow engine interface.
All visibility toggles are removed. Tool calls and results are always fully 
rendered inside the secondary tab using themed CSS variables.
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
# SIDEBAR CONTROL PANEL (No visibility toggles present)
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
# STATIC DELEGATED INTERCEPTION CHANNELS (THEMED)
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

THEME_JS_PATCH = """
<script>
    function matchParentTheme() {
        const parentStyles = window.parent.getComputedStyle(window.parent.document.body);
        const textColor = parentStyles.getPropertyValue('--text-color') || '#111111';
        const bgColor = parentStyles.getPropertyValue('--background-color') || '#ffffff';
        const secondaryBg = parentStyles.getPropertyValue('--secondary-background-color') || '#f4f4f4';

        document.body.style.color = textColor;
        document.body.style.backgroundColor = 'transparent'; 
        
        const callBox = document.querySelector('.theme-call-box');
        if (callBox) { callBox.style.backgroundColor = secondaryBg; callBox.style.color = textColor; }
        
        const resultBox = document.querySelector('.theme-result-box');
        if (resultBox) { resultBox.style.backgroundColor = secondaryBg; resultBox.style.color = textColor; }

        const preBlocks = document.querySelectorAll('pre, code');
        preBlocks.forEach(el => {
            el.style.background = bgColor;
            el.style.color = textColor;
            el.style.border = `1px solid ${secondaryBg}`;
        });
    }
    window.addEventListener('DOMContentLoaded', matchParentTheme);
    setInterval(matchParentTheme, 1000);
</script>
"""

# Force overwrite original utils functions to pipeline strings into active memory
_utils.display = lambda wrapper_obj: stream_interceptor_sink(wrapper_obj)
_utils.HTML = lambda structural_str: HTMLCaptureFrame(structural_str)

_utils.log_tool_call_html = lambda name, args: stream_interceptor_sink(
    f\"\"\"
    <div class="theme-call-box" style="border-left: 4px solid #1976D2; padding: .8em; margin: 1em 0; font-family: sans-serif; border-radius: 0 8px 8px 0;">
        <strong>🔧 Tool Call Initiated:</strong> <code style="padding: 2px 6px; border-radius: 4px;">{name}</code>
        <pre style="padding: 8px; border-radius: 6px; font-size: 13px; font-family: monospace; white-space: pre-wrap; overflow-x: auto;">{args}</pre>
    </div>
    {THEME_JS_PATCH}
    \"\"\"
)

_utils.log_tool_result_html = lambda result: stream_interceptor_sink(
    f\"\"\"
    <div class="theme-result-box" style="border-left: 4px solid #558B2F; padding: .8em; margin: 1em 0; font-family: sans-serif; border-radius: 0 8px 8px 0;">
        <strong>✅ Tool Execution Result:</strong>
        <pre style="padding: 8px; border-radius: 6px; font-size: 13px; font-family: monospace; white-space: pre-wrap; overflow-x: auto;">{result}</pre>
    </div>
    {THEME_JS_PATCH}
    \"\"\"
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
            for html_payload in st.session_state.captured_logs:
                is_large_result = "Tool Execution Result" in html_payload
                with st.container():
                    frame_height = 500 if is_large_result else 200
                    components.html(html_payload, height=frame_height, scrolling=True)
        else:
            st.info("No underlying vector tool transactions were captured for this instruction block.")

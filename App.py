"""
app.py — Completely isolated execution interface.
Removes all sidebar toggles. Tool records are forced into the secondary tab unconditionally.
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
# CRITICAL STATE BOUNDARIES (Fixes the Empty Tool-Call Infinite Loop)
# ═══════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_computed_answer" not in st.session_state:
    st.session_state.last_computed_answer = None
if "live_runtime_logs" not in st.session_state:
    st.session_state.live_runtime_logs = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR CONTROL PANEL (Cleaned, no toggle elements)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Engine Parameters")
    st.markdown("---")

    selected_label = st.selectbox(
        "🤖 Processing Model Selection",
        options=list(MODELS.keys()),
        index=0,
    )
    selected_model = MODELS[selected_label]

    st.markdown("---")
    if st.button("🗑️ Reset Engine States", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_computed_answer = None
        st.session_state.live_runtime_logs = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# DYNAMIC ACCENT THEME PATCH FOR OVERRIDDEN LOGS
# ═══════════════════════════════════════════════════════════════
class CapturedHTMLWrapper:
    def __init__(self, raw_html):
        self.raw_html = raw_html

def pipe_html_to_state_sink(html_payload):
    """Safely extracts HTML structures and appends them to clean storage frameworks."""
    if hasattr(html_payload, 'raw_html'):
        st.session_state.live_runtime_logs.append(html_payload.raw_html)
    elif isinstance(html_payload, str):
        st.session_state.live_runtime_logs.append(html_payload)

THEME_MUTATION_JS = """
<script>
    function parseDashboardTheme() {
        const dashboardBody = window.parent.getComputedStyle(window.parent.document.body);
        const fontColor = dashboardBody.getPropertyValue('--text-color') || '#111111';
        const canvasBg = dashboardBody.getPropertyValue('--background-color') || '#ffffff';
        const secondaryCardBg = dashboardBody.getPropertyValue('--secondary-background-color') || '#f8f9fa';

        document.body.style.color = fontColor;
        document.body.style.backgroundColor = 'transparent';
        
        const targetBlocks = document.querySelectorAll('.theme-card-node');
        targetBlocks.forEach(box => {
            box.style.backgroundColor = secondaryCardBg;
            box.style.color = fontColor;
        });

        const codeNodes = document.querySelectorAll('pre, code');
        codeNodes.forEach(node => {
            node.style.background = canvasBg;
            node.style.color = fontColor;
            node.style.border = `1px solid ${secondaryCardBg}`;
        });
    }
    window.addEventListener('DOMContentLoaded', parseDashboardTheme);
    setInterval(parseDashboardTheme, 1000);
</script>
"""

# Force mock IPython hooks natively prior to agent loop triggering
_utils.display = lambda payload_obj: pipe_html_to_state_sink(payload_obj)
_utils.HTML = lambda base_string: CapturedHTMLWrapper(base_string)

_utils.log_tool_call_html = lambda name, args: pipe_html_to_state_sink(
    f"""
    <div class="theme-card-node" style="border-left: 4px solid #1976D2; padding: .9em; margin: 1em 0; font-family: sans-serif; border-radius: 0 8px 8px 0;">
        <strong>🔧 Tool Call Dispatched:</strong> <code style="padding: 2px 6px; border-radius: 4px;">{name}</code>
        <pre style="padding: 8px; border-radius: 6px; font-size: 13px; font-family: monospace; white-space: pre-wrap; overflow-x: auto;">{args}</pre>
    </div>
    {THEME_MUTATION_JS}
    """
)

_utils.log_tool_result_html = lambda result: pipe_html_to_state_sink(
    f"""
    <div class="theme-card-node" style="border-left: 4px solid #558B2F; padding: .8em; margin: 1em 0; font-family: sans-serif; border-radius: 0 8px 8px 0;">
        <strong>✅ Vector Database Payload Returned:</strong>
        <pre style="padding: 8px; border-radius: 6px; font-size: 13px; font-family: monospace; white-space: pre-wrap; overflow-x: auto;">{result}</pre>
    </div>
    {THEME_MUTATION_JS}
    """
)

# ═══════════════════════════════════════════════════════════════
# DASHBOARD INTERFACE CONTAINER
# ═══════════════════════════════════════════════════════════════
st.title("💻📱 Laptop & Smartphone Recommender")
st.markdown("Provide your requirements details below to activate vector similarity matching runs.")
st.markdown("---")

# Display persistent messaging arrays
for chat_bubble in st.session_state.chat_history:
    with st.chat_message(chat_bubble["role"]):
        st.markdown(chat_bubble["content"])

# ═══════════════════════════════════════════════════════════════
# HARD BOUNDARY: CLEAN ISOLATED AGENT TRIGGER LOOP
# ═══════════════════════════════════════════════════════════════
if active_query := st.chat_input("e.g. 'Best gaming laptop under 120000 BDT with RTX 4060'"):
    
    # Store immediate user action
    st.session_state.chat_history.append({"role": "user", "content": active_query})
    
    # HARD RESET: Flush internal tracking states to guarantee a clean runtime frame
    st.session_state.live_runtime_logs = []
    st.session_state.last_computed_answer = None
    
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        with st.spinner("⏳ *Executing isolated multi-turn agent logic blocks...*"):
            try:
                # Force clean context parameter separation to pass to agent.py
                evaluated_payload = agent.LS_research_agent(
                    user_question=str(active_query),  # Cast explicitly to isolate string space
                    model=str(selected_model),
                    verbose=True,
                    show_thinking=True,       
                    show_tool_results=True,   
                    show=False,
                )
                st.session_state.last_computed_answer = evaluated_payload
                st.session_state.chat_history.append({"role": "assistant", "content": evaluated_payload})
            except Exception as processing_fault:
                fault_error_string = f"❌ **Pipeline Loop Exception:** {processing_fault}"
                st.session_state.last_computed_answer = fault_error_string
                st.session_state.chat_history.append({"role": "assistant", "content": fault_error_string})
                
    st.rerun()

# ═══════════════════════════════════════════════════════════════
# ALWAYS-ON STRUCTURAL VIEW TAB DISPATCH
# ═══════════════════════════════════════════════════════════════
if st.session_state.last_computed_answer is not None:
    st.markdown("---")
    
    # Split structural rendering tabs cleanly
    home_tab, tools_tab = st.tabs(["🏠 Recommendation Home", "🛠️ Tool Call Records"])
    
    with home_tab:
        st.subheader("💡 System Evaluation Output")
        st.markdown(st.session_state.last_computed_answer)
        
    with tools_tab:
        st.subheader("📦 Database Query Execution Logs")
        
        if st.session_state.live_runtime_logs:
            # Sequentially loop and print logs directly to the frame without conditional toggle limits
            for explicit_html in st.session_state.live_runtime_logs:
                is_database_row_payload = "Database Payload Returned" in explicit_html
                with st.container():
                    # Safely dynamically size micro-frame container constraints
                    calculated_height = 550 if is_database_row_payload else 220
                    components.html(explicit_html, height=calculated_height, scrolling=True)
        else:
            st.info("No tool call logs were captured during this transaction.")

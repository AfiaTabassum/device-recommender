"""
app.py — Fixed tabbed chatbot interface 
for the Laptop/Smartphone Recommender Agent.
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
    page_title="Device Recommender",
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
# SESSION STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_response" not in st.session_state:
    st.session_state.current_response = None
if "captured_logs" not in st.session_state:
    st.session_state.captured_logs = []

# ═══════════════════════════════════════════════════════════════
# SIDEBAR CONTROL PANEL
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Engine Control")
    st.markdown("---")

    selected_label = st.selectbox(
        "🤖 Processing Model",
        options=list(MODELS.keys()),
        index=0,
    )
    selected_model = MODELS[selected_label]

    st.markdown("---")
    st.markdown("**🔍 Debug View Filters**")

    # Linked toggles that directly show/hide specific log frames live
    show_thinking = st.toggle(
        "Show tool calls",
        value=True,
        help="Display which search tool the agent called and with what arguments.",
    )
    show_tool_results = st.toggle(
        "Show tool results",
        value=True,
        help="Display the raw database records returned by each tool call.",
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_response = None
        st.session_state.captured_logs = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# APPLICATION MONKEY-PATCH FOR UTILS DISPLAY INTERCEPTION
# ═══════════════════════════════════════════════════════════════
class HTMLCaptureObject:
    def __init__(self, data_str):
        self.data_str = data_str

def streamlit_html_interceptor(obj):
    """Intercepts IPython display calls and pipes HTML strings to session state."""
    if hasattr(obj, 'data_str'):
        st.session_state.captured_logs.append(obj.data_str)
    elif isinstance(obj, str):
        st.session_state.captured_logs.append(obj)

# Force overwrite original utils functions to pipeline strings into active memory
_utils.display = lambda x: streamlit_html_interceptor(x)
_utils.HTML = lambda x: HTMLCaptureObject(x)

# ═══════════════════════════════════════════════════════════════
# MAIN INTERFACE WORKSPACE
# ═══════════════════════════════════════════════════════════════
st.title("💻📱 Laptop & Smartphone Recommender Dashboard")
st.markdown("Submit your hardware specifications requirement below to generate recommendations.")
st.markdown("---")

# Render past chat history in chat boxes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ═══════════════════════════════════════════════════════════════
# CHAT INPUT PIPELINE EXECUTION BOUNDARY
# ═══════════════════════════════════════════════════════════════
if prompt := st.chat_input("e.g. 'Best gaming laptop under 120000 BDT with RTX 4060'"):
    
    # Save the user query message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Reset internal tracking variables for a clean run
    st.session_state.captured_logs = []
    st.session_state.current_response = None
    
    # Render user prompt immediately to prevent interface hanging
    with st.chat_message("user"):
        st.markdown(prompt)

    # Trigger Agentic Loop Execution
    with st.chat_message("assistant"):
        with st.spinner("⏳ *Searching the database and compiling recommendations...*"):
            try:
                # Core function execution
                answer = agent.LS_research_agent(
                    user_question=prompt,
                    model=selected_model,
                    verbose=True,
                    show_thinking=True,       # Force agent logging internally
                    show_tool_results=True,   # Force agent logging internally
                    show=False,
                )
                st.session_state.current_response = answer
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"❌ **Execution Error encountered during runtime:** {e}"
                st.session_state.current_response = error_msg
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

# ═══════════════════════════════════════════════════════════════
# OUTPUT TAB COMPONENT ROUTER
# ═══════════════════════════════════════════════════════════════
if st.session_state.current_response is not None:
    st.markdown("---")
    
    # Build structural presentation tabs
    home_tab, tools_tab = st.tabs(["🏠 Recommendation Home", "🛠️ Tool Call Records"])
    
    with home_tab:
        st.subheader("💡 Expert Recommendations Summary")
        st.markdown(st.session_state.current_response)
        
    with tools_tab:
        st.subheader("📦 Under-the-Hood Agent System Logs")
        
        has_visible_logs = False
        
        if st.session_state.captured_logs:
            for html_payload in st.session_state.captured_logs:
                # Isolate log types based on custom string layout classes
                is_call = "Tool Call" in html_payload
                is_result = "Tool Result" in html_payload
                
                # Check current sidebar toggle state filter matrix live
                if (is_call and show_thinking) or (is_result and show_tool_results):
                    has_visible_logs = True
                    with st.container():
                        # Auto-scale box framing heights based on payload sizing requirements
                        box_height = 450 if is_result else 180
                        components.html(html_payload, height=box_height, scrolling=True)
                        
        if not has_visible_logs:
            st.info("No tool call logs match your current sidebar display settings filter.")

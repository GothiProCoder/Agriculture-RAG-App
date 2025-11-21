import streamlit as st
import os
import time
import base64
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Import local modules
from src.config import Config
from src.ingestion import IngestionManager
from src.rag_engine import AgentManager

# ------------------
# CONFIGURATION
# ------------------
st.set_page_config(
    page_title="Jeevani - Biomass Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------
# CSS STYLING (Clean & Professional)
# ------------------
st.markdown("""
<style>
    /* Clean Theme */
    :root {
        --primary: #2d5016;
        --secondary: #4a7c2f;
        --bg-light: #f8f9fa;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(to right, #2d5016, #4a7c2f);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0;
    }
    
    /* Card Styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #eee;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Stat Styling */
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2d5016;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }

    /* Chat Messages */
    .user-msg {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #2d5016;
    }
    
    .bot-msg {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #666;
    }

</style>
""", unsafe_allow_html=True)

# ------------------
# SESSION STATE INIT
# ------------------
if "initialized" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.agent_manager = None
    st.session_state.df = None # Global DF
    st.session_state.ingestion_manager = IngestionManager()
    st.session_state.initialized = True
    
    # Try to load existing global index on startup
    try:
        df, engine = st.session_state.ingestion_manager.load_global_index()
        if df is not None and not df.empty:
            st.session_state.df = df
            # We need API key to init Agent, but we might not have it yet.
            # Store engine temporarily or wait for API key input.
            st.session_state.temp_engine = engine
    except Exception as e:
        print(f"Startup Load Error: {e}")

# ------------------
# HELPER FUNCTIONS
# ------------------
def initialize_agent(api_key):
    """Initializes the AgentManager with loaded data and API key."""
    if st.session_state.df is not None and 'temp_engine' in st.session_state:
        st.session_state.agent_manager = AgentManager(
            st.session_state.df, 
            st.session_state.temp_engine, 
            api_key
        )
        return True
    return False

def calculate_biomass_metrics(cattle_count): 
    """Calculate biomass potential from cattle count""" 
    daily_biomass_kg = cattle_count * 7 
    yearly_biomass_kg = daily_biomass_kg * 365 
    yearly_tons = yearly_biomass_kg / 1000 
    methane_kg = yearly_biomass_kg * 0.13
    methane_m3 = methane_kg / 0.72
    co2_equivalent_tons = (methane_kg * 25) / 1000
    
    return {
        'yearly_biomass_tons': yearly_tons,
        'methane_m3': methane_m3,
        'carbon_credits': co2_equivalent_tons
    }

# ------------------
# SIDEBAR
# ------------------
with st.sidebar:
    st.title("🌿 Jeevani")
    
    st.subheader("Configuration")
    api_key = st.text_input("Google API Key", type="password", value=Config.GOOGLE_API_KEY or "")
    
    if api_key:
        if st.session_state.agent_manager is None and st.session_state.df is not None:
            if initialize_agent(api_key):
                st.success("Agent Initialized")
    
    st.divider()
    
    # Status
    if st.session_state.df is not None:
        st.info(f"📚 **Knowledge Base Active**\n\nTotal Records: {len(st.session_state.df)}")
    else:
        st.warning("⚠️ Knowledge Base Empty")

# ------------------
# MAIN LAYOUT
# ------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>Jeevani Platform</h1>
    <p>Intelligent Biomass Analytics & Knowledge Base</p>
</div>
""", unsafe_allow_html=True)

# Tabs for Navigation
tab_home, tab_kb, tab_chat = st.tabs(["📊 Dashboard", "📂 Knowledge Base", "💬 Chat Assistant"])

# --- DASHBOARD TAB ---
with tab_home:
    if st.session_state.df is None:
        st.info("Please upload data in the 'Knowledge Base' tab to view analytics.")
    else:
        df = st.session_state.df
        
        # Metrics
        total_cattle = df['Cattle_Count'].sum() if 'Cattle_Count' in df.columns else 0
        metrics = calculate_biomass_metrics(total_cattle)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="info-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Gaushalas</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="info-card"><div class="metric-value">{total_cattle:,}</div><div class="metric-label">Total Cattle</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="info-card"><div class="metric-value">{metrics['yearly_biomass_tons']:.1f}k</div><div class="metric-label">Annual Biomass (Tons)</div></div>""", unsafe_allow_html=True)

        # Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if 'District' in df.columns:
                st.subheader("Cattle by District")
                dist_data = df.groupby('District')['Cattle_Count'].sum().sort_values(ascending=True)
                st.bar_chart(dist_data)
        
        with col_c2:
            if 'Status' in df.columns:
                st.subheader("Facility Status")
                status_counts = df['Status'].value_counts()
                # Simple pie chart using plotly
                fig = go.Figure(data=[go.Pie(labels=status_counts.index, values=status_counts.values)])
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
                st.plotly_chart(fig, use_container_width=True)


# --- KNOWLEDGE BASE TAB ---
with tab_kb:
    st.subheader("Global Knowledge Base Management")
    st.markdown("All uploaded files are merged into a single unified knowledge base.")
    
    col_up, col_list = st.columns([1, 2])
    
    with col_up:
        st.markdown("#### Upload New Report")
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded_file:
            if st.button("Process & Add to Knowledge Base"):
                with st.status("Processing...", expanded=True) as status:
                    try:
                        # 1. Save File
                        st.write("Parsing PDF...")
                        st.session_state.ingestion_manager.process_upload(uploaded_file.getvalue(), uploaded_file.name)
                        
                        # 2. Rebuild Global Index
                        st.write("Rebuilding Global Index (This merges all files)...")
                        df, engine = st.session_state.ingestion_manager.rebuild_global_index()
                        
                        # 3. Update Session
                        st.session_state.df = df
                        st.session_state.temp_engine = engine
                        # Re-init agent if key exists
                        if api_key:
                            initialize_agent(api_key)
                            
                        status.update(label="Success!", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:
                        status.update(label="Failed", state="error", expanded=False)
                        st.error(f"Error: {e}")

    with col_list:
        st.markdown("#### Managed Files")
        artifacts = st.session_state.ingestion_manager.get_all_artifacts()
        
        if not artifacts:
            st.info("No files in the knowledge base.")
        else:
            for art in artifacts:
                with st.expander(f"📄 {art['filename']} ({art.get('upload_date')})"):
                    st.write(f"**Rows Extracted:** {art.get('row_count')}")
                    st.write(f"**File Size:** {art.get('file_size')} bytes")
                    
                    # Actions
                    c1, c2 = st.columns(2)
                    if c1.button("Preview PDF", key=f"pdf_{art['file_hash']}"):
                        st.session_state['preview_pdf'] = art['file_hash']
                    
                    if c2.button("Delete", key=f"del_{art['file_hash']}"):
                        st.session_state.ingestion_manager.delete_artifact(art['file_hash'])
                        st.rerun()

    # Preview Section
    if 'preview_pdf' in st.session_state:
        st.divider()
        st.subheader("Document Preview")
        p_hash = st.session_state['preview_pdf']
        p_path = st.session_state.ingestion_manager.get_file_path(p_hash)
        
        if p_path:
            with open(p_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            if st.button("Close Preview"):
                del st.session_state['preview_pdf']
                st.rerun()
        else:
            st.error("File not found.")


# --- CHAT TAB ---
with tab_chat:
    st.subheader("Ask Jeevani")
    
    if st.session_state.agent_manager is None:
        st.info("Agent is not ready. Please ensure:\n1. You have entered a Google API Key in the sidebar.\n2. You have uploaded at least one document.")
    else:
        # Chat History
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f'<div class="user-msg">👤 <b>You:</b> {chat["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">🤖 <b>Jeevani:</b> {chat["content"]}</div>', unsafe_allow_html=True)
        
        # Input
        if prompt := st.chat_input("Ask about total biomass, specific districts, or cattle info..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

        # Response Generation (Triggered by rerun)
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with st.spinner("Thinking..."):
                try:
                    last_q = st.session_state.chat_history[-1]["content"]
                    response = st.session_state.agent_manager.query(last_q)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


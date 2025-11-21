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
# BIOMASS CALCULATION HELPERS (From Reference)
# ------------------
def calculate_biomass_from_cattle(cattle_count): 
    """Calculate biomass potential from cattle count (7kg per cattle per day)""" 
    daily_biomass_kg = cattle_count * 7 
    monthly_biomass_kg = daily_biomass_kg * 30 
    yearly_biomass_kg = daily_biomass_kg * 365 
     
    # Convert to tons for better readability 
    daily_tons = daily_biomass_kg / 1000 
    monthly_tons = monthly_biomass_kg / 1000 
    yearly_tons = yearly_biomass_kg / 1000 
     
    return { 
        'daily_kg': daily_biomass_kg, 
        'monthly_kg': monthly_biomass_kg, 
        'yearly_kg': yearly_biomass_kg, 
        'daily_tons': daily_tons, 
        'monthly_tons': monthly_tons, 
        'yearly_tons': yearly_tons 
    } 
 
def calculate_methane_from_biomass(biomass_kg): 
    """Calculate methane production from biomass (dung/manure)""" 
    methane_kg = biomass_kg * 0.13 
    methane_tons = methane_kg / 1000 
    methane_m3 = methane_kg / 0.72 
     
    return { 
        'methane_kg': methane_kg, 
        'methane_tons': methane_tons, 
        'methane_m3': methane_m3 
    } 
 
def calculate_bio_credits_from_methane(methane_kg): 
    """Calculate bio-credits/carbon credits from methane capture""" 
    co2_equivalent_kg = methane_kg * 25 
    co2_equivalent_tons = co2_equivalent_kg / 1000 
    carbon_credits = co2_equivalent_tons 
     
    return { 
        'co2_equivalent_kg': co2_equivalent_kg, 
        'co2_equivalent_tons': co2_equivalent_tons, 
        'carbon_credits': carbon_credits, 
    } 

# ------------------
# CSS STYLING (Jeevani Theme)
# ------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-green: #2d5016;
        --accent-green: #4a7c2f;
        --light-green: #8bc34a;
        --bg-dark: #1a1a1a;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Overall page background */
    .stApp {
        background: none !important;
        background-color: transparent !important;
    }
    
    /* Text styling improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #2d5016 !important;
    }
    
    p, div, span {
        color: #333333;
    }
    
    /* Success/Info/Warning boxes styling */
    .stSuccess {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important;
        border-left: 4px solid #4a7c2f !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stSuccess p, .stSuccess div {
        color: #1b3a0f !important;
        font-weight: 500;
    }

    .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        border-left: 4px solid #2196f3 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    .stWarning {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%) !important;
        border-left: 4px solid #ff9800 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Custom header styling */
    .main-header {
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(45, 80, 22, 0.15);
        background: linear-gradient(135deg, #f5f7f5 0%, #ffffff 100%);
        border: 1px solid #e0e0e0;
    }
    
    .main-title {
        color: #2d5016;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        color: #4a7c2f;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4a7c2f;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        height: 100%;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d5016;
        margin: 0;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Chat styling - User messages */
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important;
        border-left: 4px solid #4a7c2f;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(74, 124, 47, 0.15);
    }
    
    .stChatMessage[data-testid="user"] p, 
    .stChatMessage[data-testid="user"] div, 
    .stChatMessage[data-testid="user"] * {
        color: #1b3a0f !important;
    }

    /* Chat styling - Assistant messages */
    .stChatMessage[data-testid="assistant"] {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7f5 100%) !important;
        border-left: 4px solid #2d5016;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .stChatMessage[data-testid="assistant"] p, 
    .stChatMessage[data-testid="assistant"] div, 
    .stChatMessage[data-testid="assistant"] * {
        color: #2d5016 !important;
    }
    
    /* Chat input styling */
    [data-testid="stChatInput"] {
        background: white;
        border: 2px solid #4a7c2f;
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4a7c2f !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background: linear-gradient(135deg, #f8fff8 0%, #ffffff 100%) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #2d5016 !important;
        background: linear-gradient(135deg, #e8f5e9 0%, #f8fff8 100%) !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f7f5 0%, #ffffff 100%) !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4a7c2f 0%, #2d5016 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(74, 124, 47, 0.4);
    }

    /* Metric styling improvements */
    [data-testid="stMetricValue"] {
        color: #2d5016 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-size: 0.9rem !important;
    }

</style>
""", unsafe_allow_html=True)

# ------------------
# SESSION STATE INIT
# ------------------
if "initialized" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.documents = {} # To track metadata of uploaded files
    st.session_state.total_queries = 0
    st.session_state.initialized = True
    
    # Core Backend Objects
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = None
    if "df" not in st.session_state:
        st.session_state.df = None

# ------------------
# HELPER FUNCTIONS
# ------------------
def initialize_system(file_bytes, file_name, api_key):
    """Initializes the ingestion and agent manager."""
    try:
        ingestion = IngestionManager()
        df, retrieval_engine = ingestion.process_upload(file_bytes, file_name)
        
        agent_manager = AgentManager(df, retrieval_engine, api_key)
        
        st.session_state.df = df
        st.session_state.agent_manager = agent_manager
        st.session_state.documents = {file_name: {"rows": len(df), "date": datetime.now().strftime("%Y-%m-%d %H:%M")}}
        
        return True, f"Successfully loaded {len(df)} records."
    except Exception as e:
        return False, str(e)

# ------------------
# SIDEBAR & NAVIGATION
# ------------------
with st.sidebar:
    st.markdown("## 🌿 Navigation")
    
    tab_selection = st.radio(
        "Choose Section",
        ["Chat Interface", "Document Management", "Analytics Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.subheader("⚙️ Configuration")
    # API Key Handling
    try:
        default_key = Config.GOOGLE_API_KEY if Config.GOOGLE_API_KEY else ""
    except:
        default_key = ""
    
    api_key = st.text_input("Google API Key", value=default_key, type="password")
    
    if st.session_state.df is not None:
        st.success(f"✅ Knowledge Base Active\n{len(st.session_state.df)} Records Loaded")
    else:
        st.warning("⚠️ No Data Loaded")

    st.markdown("---")
    with st.expander("About Jeevani"):
        st.markdown("""
        **Jeevani**
        
        An AI-powered biomass intelligence platform that helps analyze
        district-wise biomass availability data for informed business decisions.
        
        **Features:**
        - 🧬 Biomass & Methane Analytics
        - 📊 Intelligent Query Responses
        - 🐄 Cattle Data Analysis
        """)

# ------------------
# MAIN HEADER
# ------------------
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Jeevani</h1>
    <p class="subtitle">Biomass Intelligence Platform by Biosanjeevani Sustainable Solutions Pvt. Ltd.</p>
</div>
""", unsafe_allow_html=True)

# ------------------
# TAB 1: CHAT INTERFACE
# ------------------
if tab_selection == "Chat Interface":
    
    if st.session_state.agent_manager is None:
        st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #f8fff8 0%, #ffffff 100%); 
                    border-radius: 15px; border: 2px dashed #4a7c2f; margin: 2rem 0;'>
            <h2 style='color: #2d5016; margin-bottom: 1rem;'>Welcome to Jeevani!</h2>
            <p style='color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>
                Your intelligent biomass analytics assistant is ready to help you analyze district-wise data.
            </p>
            <div style='background: white; padding: 1.5rem; border-radius: 10px; display: inline-block; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
                <p style='color: #4a7c2f; font-weight: 600; margin: 0;'>👉 Go to <strong>Document Management</strong> to upload gaushala reports.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("### ✨ What You Can Do")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4a7c2f; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05); height: 100%;'>
                <h4 style='color: #2d5016; margin-bottom: 0.5rem;'>🔍 Smart Queries</h4>
                <p style='color: #666; font-size: 0.9rem; margin: 0;'>
                    Ask questions about biomass availability, district data, and cattle populations.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4a7c2f; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05); height: 100%;'>
                <h4 style='color: #2d5016; margin-bottom: 0.5rem;'>Analytics</h4>
                <p style='color: #666; font-size: 0.9rem; margin: 0;'>
                    Get instant calculations for biomass potential and biofuel production.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4a7c2f; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05); height: 100%;'>
                <h4 style='color: #2d5016; margin-bottom: 0.5rem;'>💡 Insights</h4>
                <p style='color: #666; font-size: 0.9rem; margin: 0;'>
                    Receive data-driven insights for informed business decisions.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        # Chat Interface
        question = st.chat_input("Ask about biomass availability, districts, or cattle counts...")
        
        # Display History
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])
        
        if question:
            if not api_key:
                st.error("Please enter your Google API Key in the sidebar.")
            else:
                st.session_state.total_queries += 1
                
                # User Message
                with st.chat_message("user"):
                    st.write(question)
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Assistant Response
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Analyzing data..."):
                        try:
                            response = st.session_state.agent_manager.query(question)
                            
                            # Typewriter effect simulation
                            message_placeholder = st.empty()
                            full_response = ""
                            for chunk in response.split():
                                full_response += chunk + " "
                                time.sleep(0.02)
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

# ------------------
# TAB 2: DOCUMENT MANAGEMENT
# ------------------
elif tab_selection == "Document Management":
    st.subheader("📂 Document Library")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload New Data")
        uploaded_file = st.file_uploader(
            "Upload District Biomass Data (PDF)",
            type=["pdf"],
            help="Upload the official PDF report containing cattle/gaushala data."
        )
        
        if uploaded_file:
            if st.button("📥 Process & Add to Library", type="primary"):
                if not api_key:
                    st.error("Please enter Google API Key in sidebar first.")
                else:
                    with st.status("Processing document...", expanded=True) as status:
                        st.write("Parsing PDF and building knowledge base...")
                        success, msg = initialize_system(uploaded_file.getvalue(), uploaded_file.name, api_key)
                        
                        if success:
                            status.update(label="Success!", state="complete", expanded=False)
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            status.update(label="Failed", state="error", expanded=False)
                            st.error(msg)

    with col2:
        st.markdown("### Library Stats")
        if st.session_state.df is not None:
            st.metric("Total Records", len(st.session_state.df))
            st.metric("Districts Covered", st.session_state.df['District'].nunique() if 'District' in st.session_state.df.columns else 0)
            st.info("✅ Knowledge base active.")
        else:
            st.metric("Total Documents", 0)
            st.info("Waiting for upload...")

    # Display Current Documents
    st.markdown("---")
    st.markdown("### Current Documents")
    
    if st.session_state.documents:
        for doc_name, details in st.session_state.documents.items():
            st.markdown(f"""
            <div style='background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #4a7c2f; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>
                <h4 style='margin: 0; color: #2d5016;'>📄 {doc_name}</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                    <strong>Rows:</strong> {details['rows']} | <strong>Uploaded:</strong> {details['date']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.df is not None:
                with st.expander("Preview Data (First 5 Rows)"):
                    st.dataframe(st.session_state.df.head())
    else:
        st.info("📭 No documents in library. Upload some documents to get started!")

# ------------------
# TAB 3: ANALYTICS DASHBOARD
# ------------------
elif tab_selection == "Analytics Dashboard":
    st.subheader("Platform Analytics")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # -- CALCULATE CORE METRICS --
        total_cattle = df['Cattle_Count'].sum() if 'Cattle_Count' in df.columns else 0
        total_districts = df['District'].nunique() if 'District' in df.columns else 0
        
        # -- CALCULATE BIOMASS METRICS --
        biomass = calculate_biomass_from_cattle(total_cattle)
        methane = calculate_methane_from_biomass(biomass['yearly_kg'])
        credits = calculate_bio_credits_from_methane(methane['methane_kg'])
        
        # 1. PRIMARY METRICS ROW
        st.markdown("### 🏭 Operational Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""<div class="stat-card"><p class="stat-number">{len(df)}</p><p class="stat-label">Gaushalas</p></div>""", unsafe_allow_html=True)
        with col2:
             st.markdown(f"""<div class="stat-card"><p class="stat-number">{total_cattle:,}</p><p class="stat-label">Total Cattle</p></div>""", unsafe_allow_html=True)
        with col3:
             st.markdown(f"""<div class="stat-card"><p class="stat-number">{total_districts}</p><p class="stat-label">Districts</p></div>""", unsafe_allow_html=True)
        with col4:
             st.markdown(f"""<div class="stat-card"><p class="stat-number">{st.session_state.total_queries}</p><p class="stat-label">Queries</p></div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. ENVIRONMENTAL IMPACT ROW
        st.markdown("### 🌍 Environmental Impact & Potential")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #8bc34a;">
                <p class="stat-number">{biomass['daily_tons']:.1f}</p>
                <p class="stat-label">Daily Biomass (Tons)</p>
                <p style="font-size: 0.8rem; color: #888;">~{biomass['yearly_tons']:.0f} Tons / Year</p>
            </div>""", unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #ff9800;">
                <p class="stat-number">{methane['methane_m3']:,.0f}</p>
                <p class="stat-label">Methane Potential (m³/yr)</p>
                <p style="font-size: 0.8rem; color: #888;">From Anaerobic Digestion</p>
            </div>""", unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #2196f3;">
                <p class="stat-number">{credits['carbon_credits']:,.0f}</p>
                <p class="stat-label">Potential Carbon Credits</p>
                <p style="font-size: 0.8rem; color: #888;">CO2 Equivalent Offset</p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Cattle Distribution by District")
            if 'District' in df.columns and 'Cattle_Count' in df.columns:
                district_data = df.groupby('District')['Cattle_Count'].sum().sort_values(ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=district_data.values,
                    y=district_data.index,
                    orientation='h',
                    marker=dict(color='#4a7c2f')
                ))
                
                fig.update_layout(
                    title=None,
                    xaxis_title="Cattle Count",
                    yaxis_title="District",
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Required columns for chart not found.")
        
        with col2:
            st.markdown("### Status Breakdown")
            if 'Status' in df.columns:
                status_counts = df['Status'].value_counts()
                
                fig2 = go.Figure(data=[go.Pie(
                    labels=status_counts.index, 
                    values=status_counts.values,
                    hole=.4,
                    marker=dict(colors=['#2d5016', '#4a7c2f', '#8bc34a', '#c8e6c9'])
                )])
                
                fig2.update_layout(
                    title=None,
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Status column not found.")
                
    else:
        st.info("Analytics will appear once you upload data in the Document Management tab.")
        
        # Empty state placeholders
        col1, col2, col3, col4 = st.columns(4)
        for _ in range(4):
            with col1:
                st.markdown("""<div class="stat-card"><p class="stat-number">-</p><p class="stat-label">Waiting...</p></div>""", unsafe_allow_html=True)
            col1, col2, col3, col4 = col2, col3, col4, col1 # Rotate columns hack

# ------------------
# FOOTER
# ------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem 2rem 2rem 2rem;'>
    <p><strong>Jeevani</strong> - Powered by Biosanjeevani Sustainable Solutions Pvt. Ltd.</p>
    <p style='font-size: 0.9rem;'>Intelligent Biomass Analytics Platform</p>
    <p style='font-size: 0.85rem; margin-top: 0.5rem; color: #4a7c2f; font-weight: 600;'>Proudly made in India 🇮🇳</p>
</div>
""", unsafe_allow_html=True)

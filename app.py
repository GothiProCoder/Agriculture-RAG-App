import streamlit as st
import time
import os

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Agri-Insight AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS STYLING
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(left, #00C853, #69F0AE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #B0BEC5;
        text-align: center;
        margin-bottom: 30px;
    }
    .stChatMessage {
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC ---

def main():
    # -- SIDEBAR --
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Handling
        try:
            from src.config import Config
            default_key = Config.GOOGLE_API_KEY
        except:
            default_key = ""
        
        api_key = st.text_input("Google API Key", value=default_key, type="password")
        uploaded_file = st.file_uploader("Upload PDF Report", type=["pdf"])
        
        st.divider()
        if st.button("🧹 Reset System"):
            st.session_state.clear()
            st.rerun()

    # -- MAIN CONTENT --
    
    # Landing Page
    if not uploaded_file:
        st.markdown('<div class="main-title">Agri-Insight AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Advanced RAG Analytics for Agricultural Reports</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📂 **Persistent Storage**\nAutomatically saves and indexes uploaded reports.")
        with col2:
            st.success("🧠 **Hybrid Search**\nCombines SQL-like precision with semantic understanding.")
        with col3:
            st.warning("💬 **Analyst Bot**\nAsk about totals, specific shelters, or districts.")
        return

    # Check API Key
    if not api_key:
        st.error("Please enter your Google API Key in the sidebar to proceed.")
        st.stop()

    # -- SYSTEM INITIALIZATION (Only runs once per file load) --
    if "agent_manager" not in st.session_state:
        
        # Progress Container
        with st.status("🚀 Booting AI Engine...", expanded=True) as status:
            
            try:
                from src.ingestion import IngestionManager
                from src.rag_engine import AgentManager
                
                # 1. Ingestion (Handling Cache/Parsing)
                st.write("Checking Knowledge Base...")
                ingestion = IngestionManager()
                file_bytes = uploaded_file.getvalue()
                
                # This handles the check-disk-or-process logic
                df, retrieval_engine = ingestion.process_upload(file_bytes, uploaded_file.name)
                
                st.write(f"✅ Data Loaded: {len(df)} records found.")
                
                # 2. Agent Initialization
                st.write("Initializing Cognitive Agent...")
                agent_manager = AgentManager(df, retrieval_engine, api_key)
                
                # Store in Session State
                st.session_state.agent_manager = agent_manager
                
                status.update(label="System Online", state="complete", expanded=False)
                
            except Exception as e:
                st.error(f"Initialization Failed: {str(e)}")
                st.stop()

    # -- CHAT INTERFACE --
    st.markdown('<div class="main-title">Agri-Insight Chat</div>', unsafe_allow_html=True)

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "System ready. I can calculate totals, find specific contact numbers, or search by district."}]

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about the data..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("Thinking..."):
                    # Query the Agent
                    response = st.session_state.agent_manager.query(prompt)
                
                # Streaming Effect
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.03) # Slightly faster typing
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Processing Error: {e}")

if __name__ == "__main__":
    main()
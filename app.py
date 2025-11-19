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

# 2. CSS (CLEANED - No background overrides)
# We only style the specific titles, letting the config.toml handle the rest.
st.markdown("""
    <style>
    /* Hide standard Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Title Styling */
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
    
    /* Chat message tweaks */
    .stChatMessage {
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CACHING & LOGIC (Keep this exactly the same) ---

@st.cache_data(show_spinner=False)
def process_uploaded_file(file_bytes):
    import io
    from src.data_processor import parse_gaushala_pdf
    file_obj = io.BytesIO(file_bytes)
    df = parse_gaushala_pdf(file_obj)
    return df

@st.cache_resource(show_spinner=False)
def initialize_ai_engine(_df, api_key):
    from src.vector_store import RetrievalEngine
    from src.rag_engine import AgentManager
    retrieval_engine = RetrievalEngine(_df)
    agent_manager = AgentManager(_df, retrieval_engine, api_key)
    return agent_manager

def main():
    # -- SIDEBAR --
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        try:
            from src.config import Config
            default_key = Config.GOOGLE_API_KEY
        except:
            default_key = ""
        
        api_key = st.text_input("Google API Key", value=default_key, type="password")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        
        st.divider()
        if st.button("🧹 Reset App"):
            st.session_state.clear()
            st.rerun()

    # -- MAIN AREA --
    
    # LANDING PAGE
    if not uploaded_file:
        st.markdown('<div class="main-title">Agri-Insight AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Advanced RAG Analytics for Agricultural Reports</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📂 **Upload Data**\nSupports PDF Gaushala reports.")
        with col2:
            st.success("🧠 **AI Analysis**\nExtracts entities, contacts, and stats.")
        with col3:
            st.warning("💬 **Chat Bot**\nAsk natural language questions.")
        return

    # INITIALIZATION
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = None

    if st.session_state.agent_manager is None:
        if not api_key:
            st.error("Please enter your API Key in the sidebar.")
            st.stop()
            
        with st.status("🚀 Initializing System...", expanded=True) as status:
            st.write("Parsing PDF structure...")
            file_bytes = uploaded_file.getvalue()
            
            try:
                df = process_uploaded_file(file_bytes)
                st.write(f"✅ Loaded {len(df)} records.")
                
                st.write("Hydrating Vector Database...")
                st.session_state.agent_manager = initialize_ai_engine(df, api_key)
                
                status.update(label="System Online", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Critical Error: {e}")
                st.stop()

    # CHAT INTERFACE
    st.markdown('<div class="main-title">Agri-Insight Chat</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "System ready. Ask me about cattle counts, specific districts, or contact details."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Query the database..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Processing..."):
                try:
                    response = st.session_state.agent_manager.query(prompt)
                    # Typewriter effect
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error("Error processing request.")

if __name__ == "__main__":
    main()

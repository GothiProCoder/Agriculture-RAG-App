import streamlit as st
import time

# 1. PAGE CONFIGURATION (Must be the first command)
st.set_page_config(
    page_title="Agri-Insight AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS (For a Professional Look)
st.markdown("""
    <style>
    /* Hide Streamlit header and footer for a clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Title Style */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E7D32; /* Agricultural Green */
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: #f9f9f9;
        border-radius: 10px;
        border: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CACHED FUNCTIONS (The Performance Fix) ---

@st.cache_data(show_spinner=False)
def process_uploaded_file(file_bytes):
    """
    Extracts text from PDF bytes. Cached so it runs once per file.
    """
    import io
    from src.data_processor import parse_gaushala_pdf
    
    # Convert bytes back to file-like object for the parser
    file_obj = io.BytesIO(file_bytes)
    df = parse_gaushala_pdf(file_obj)
    return df

@st.cache_resource(show_spinner=False)
def initialize_ai_engine(_df, api_key):
    """
    Loads the Heavy AI Models (Embeddings, Vectors, LLM).
    Cached forever in memory until the app restarts.
    The underscore in _df tells Streamlit not to hash the dataframe (faster).
    """
    from src.vector_store import RetrievalEngine
    from src.rag_engine import AgentManager
    
    # 1. Build Vector Index
    retrieval_engine = RetrievalEngine(_df)
    
    # 2. Initialize Agents
    agent_manager = AgentManager(_df, retrieval_engine, api_key)
    return agent_manager

# --- 4. MAIN APPLICATION ---

def main():
    # -- Sidebar --
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=50)
        st.header("Configuration")
        
        # Lazy import config
        from src.config import Config
        default_key = Config.GOOGLE_API_KEY
        
        api_key = st.text_input("🔑 Google API Key", value=default_key if default_key else "", type="password")
        uploaded_file = st.file_uploader("📄 Upload Report (PDF)", type=["pdf"])
        
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # -- Main Area Logic --
    
    # A. Landing Page State (No File Uploaded)
    if not uploaded_file:
        st.markdown('<div class="main-title">🌾 Agri-Insight Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Upload your Gaushala Report to begin analyzing cattle data instantly.</div>', unsafe_allow_html=True)
        
        # Demo features grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📊 **Data Analytics**\nCount cattle, filter by status, and aggregate district data.")
        with col2:
            st.success("🔍 **Smart Search**\nFind specific people, phone numbers, and locations.")
        with col3:
            st.warning("🤖 **Hybrid RAG**\nCombines Keyword search with Semantic understanding.")
            
        if not api_key:
            st.error("⚠️ Please enter your Google API Key in the sidebar to start.")
        return

    # B. Processing State
    # We use session state to keep the agent alive without reloading
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = None

    # Process only if the agent isn't ready or file changed
    if st.session_state.agent_manager is None and api_key:
        with st.status("🚀 Booting up AI System...", expanded=True) as status:
            st.write("Parsing document structure...")
            file_bytes = uploaded_file.getvalue() # Read bytes once
            
            try:
                # Step 1: PDF Parsing
                df = process_uploaded_file(file_bytes)
                st.write(f"✅ Extracted {len(df)} records.")
                
                # Step 2: AI Loading
                st.write("Loading Embedding Models & Vector DB...")
                st.session_state.agent_manager = initialize_ai_engine(df, api_key)
                
                status.update(label="✅ System Online", state="complete", expanded=False)
            except Exception as e:
                st.error(f"System Failure: {str(e)}")
                st.stop()

    # C. Chat Interface
    st.markdown('<div class="main-title">🌾 Agri-Insight Assistant</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I have analyzed your document. You can ask me about cattle counts, contact details, or specific shelter statuses."}
        ]

    # Render Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle Input
    if prompt := st.chat_input("Ask a question..."):
        # 1. Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Analyzing..."):
                try:
                    # The heavy lifting is already done/cached, so this is fast
                    response = st.session_state.agent_manager.query(prompt)
                    
                    # Simulate typing effect (looks professional)
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    message_placeholder.error(f"Error: {str(e)}")
                    full_response = "Error processing request."

        # 3. Save Assistant Message
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import os
from src.config import Config
from src.data_processor import parse_gaushala_pdf
from src.vector_store import RetrievalEngine
from src.rag_engine import AgentManager

st.set_page_config(page_title="Agri-Data RAG Bot", layout="wide")

def main():
    st.title("🌾 Agricultural Data Assistant")
    st.markdown("Upload a PDF report to analyze cattle data, contact details, and registry information.")

    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key Handling
        env_key = Config.GOOGLE_API_KEY
        api_key = st.text_input("Google API Key", value=env_key if env_key else "", type="password")
        
        if not api_key:
            st.warning("Please provide a Google API Key to proceed.")
            return

        uploaded_file = st.file_uploader("Upload PDF Data", type=["pdf"])

    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = None

    # Data Processing Trigger
    if uploaded_file and st.session_state.agent_manager is None:
        with st.status("Processing Document...", expanded=True) as status:
            st.write("Parsing PDF tables...")
            try:
                df = parse_gaushala_pdf(uploaded_file)
                st.write(f"Extracted {len(df)} records.")
                
                st.write("Building Vector Index (FAISS + BM25)...")
                # Initialize Retrieval Engine
                retrieval_engine = RetrievalEngine(df)
                
                # Initialize Agents
                st.write("Initializing AI Agents...")
                st.session_state.agent_manager = AgentManager(df, retrieval_engine, api_key)
                
                status.update(label="System Ready!", state="complete", expanded=False)
                st.success("Document processed successfully.")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                return

    # Chat Interface
    if st.session_state.agent_manager:
        # Display History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User Input
        if prompt := st.chat_input("Ask about cattle counts, contacts, or specific shelters..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        response = st.session_state.agent_manager.query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    
    elif not uploaded_file:
        st.info("Please upload a PDF file from the sidebar to begin.")

if __name__ == "__main__":
    main()
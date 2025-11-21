import re
import os
import pickle
import pandas as pd

# --- MODERN IMPORTS ---
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings 
from sentence_transformers import CrossEncoder

from src.config import Config

class RetrievalEngine:
    def __init__(self, load_models_now=True):
        """
        Initialize models. 
        """
        self.documents = []
        self.faiss_retriever = None
        self.bm25_retriever = None
        self.embeddings = None
        self.cross_encoder = None
        self.vector_store = None
        
        if load_models_now:
            self._load_models()

    def _load_models(self):
        if self.embeddings is None:
            print("⏳ Loading Embedding Model...")
            self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        
        if self.cross_encoder is None:
            print("⏳ Loading Cross-Encoder...")
            self.cross_encoder = CrossEncoder(Config.CROSS_ENCODER_MODEL)

    def _normalize_regno(self, raw):
        if not raw: return None, None
        s = str(raw).strip()
        # Extract numbers from patterns like GSA-102, G.S.A 102, or just 102
        m = re.search(r'(?:GSA[\s\.\-]?0*)?(\d+)', s, re.IGNORECASE)
        if m:
            digits = int(m.group(1))
            norm = f"GSA-{digits}"
            return norm, digits
        return s, None

    def build_index(self, df: pd.DataFrame):
        """Builds the index from scratch using the dataframe."""
        self._load_models()
        print("🔨 Building SEMANTIC Narrative Index...")
        self.documents = []

        for index, row in df.iterrows():
            # 1. Normalize ID
            reg_raw = row.get('Registration_No', '')
            reg_norm, reg_digits = self._normalize_regno(reg_raw)
            
            # 2. Handle Missing Data Logic for Text Generation
            def clean(val, default="Unknown"):
                if pd.isna(val) or val == "" or str(val).lower() == "none":
                    return default
                return str(val).strip().title()

            name = clean(row.get('Gaushala_Name'), "Unnamed Gaushala")
            district = clean(row.get('District'), "Unknown District")
            village = clean(row.get('Village'), "Unknown Village")
            status = clean(row.get('Status'), "Active")
            count = row.get('Cattle_Count', 0)
            
            # 3. THE "RICH NARRATIVE" TEMPLATE (The Secret Sauce)
            # We construct a natural sentence. 
            # We explicitly mention "Cow Shelter" and "Gaushala" to link synonyms.
            # We emphasize the location hierarchy.
            
            text_content = (
                f"The {name} is a registered Gaushala (Cow Shelter) located in {village}, "
                f"which falls under the {district} district of Haryana. "
                f"It is currently {status} and manages a total of {count} cattle. "
                f"Official Registration Number: {reg_norm or reg_raw}. "
            )
            
            # Add Contact Info only if it exists (Reduces noise if empty)
            contact_p = row.get('Contact_Person')
            phone = row.get('Phone_Number')
            
            if pd.notna(contact_p) or pd.notna(phone):
                c_name = clean(contact_p, "the manager")
                c_num = clean(phone, "not listed")
                text_content += f" The primary contact person is {c_name}, reachable at phone number {c_num}."

            # 4. Metadata (Kept strict for filtering)
            metadata = {
                "row_id": int(index),
                "district": district,
                "registration_no_digits": reg_digits if reg_digits else -1,
                "full_info": text_content # Used for Reranker
            }
            
            self.documents.append(Document(page_content=text_content, metadata=metadata))

        if not self.documents:
            print("⚠️ No documents to index!")
            return

        # Build FAISS
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)

        # Build BM25
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 30

        # Helper to setup retrievers
        self._refresh_retrievers()
        print(f"✅ Index built with {len(self.documents)} documents.")

    def _refresh_retrievers(self):
        """Sets up the retriever interfaces."""
        if self.vector_store:
            # k=20 for initial fetch
            self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})

    def save_local(self, folder_path: str):
        """Saves FAISS index and Documents/BM25 data to disk."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if self.vector_store:
            self.vector_store.save_local(folder_path)
        
        # Save Documents (required for BM25 reconstruction)
        doc_path = os.path.join(folder_path, "documents.pkl")
        with open(doc_path, "wb") as f:
            pickle.dump(self.documents, f)
            
        print(f"💾 Index saved to {folder_path}")

    def load_local(self, folder_path: str):
        """Loads FAISS index and reconstructs BM25 from disk."""
        self._load_models()
        print(f"📂 Loading Index from {folder_path}...")
        
        # Load FAISS with dangerous deserialization allowed (safe for local trusted files)
        self.vector_store = FAISS.load_local(
            folder_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True 
        )
        
        # Load Documents and Rebuild BM25
        doc_path = os.path.join(folder_path, "documents.pkl")
        if os.path.exists(doc_path):
            with open(doc_path, "rb") as f:
                self.documents = pickle.load(f)
            
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 30
        else:
            print("⚠️ Warning: Documents file not found, BM25 will be empty.")

        self._refresh_retrievers()
        print("✅ Index Loaded Successfully.")

    def search(self, query: str) -> str:
        """Hybrid Search + Cross Encoder Rerank."""
        if not self.vector_store or not self.bm25_retriever:
            return "Error: Index not initialized."

        # 1. Exact ID Check (Regex Shortcut)
        # gsa_match = re.search(r'(?:GSA)[\s\.\-]?0*(\d+)', query, re.IGNORECASE)
        gsa_match = re.search(r'(?:GSA[\s\.\-]*)?0*(\d+)', query, re.IGNORECASE)
        
        if gsa_match:
            target_int = int(gsa_match.group(1))
            # Filter in memory for exact metadata match
            meta_hits = [d for d in self.documents 
                         if d.metadata.get("registration_no_digits") == target_int]
            if meta_hits:
                return "\n\n".join([f"🎯 EXACT METADATA MATCH:\n{d.page_content}" for d in meta_hits[:3]])

        # 2. Hybrid Retrieval (BM25 + FAISS)
        # Modern LangChain uses .invoke() instead of .get_relevant_documents()
        bm25_hits = self.bm25_retriever.invoke(query)
        
        faiss_hits = []
        if self.faiss_retriever:
            faiss_hits = self.faiss_retriever.invoke(query)

        # Merge and Deduplicate (preserve order)
        seen = set()
        candidate_docs = []
        for d in (bm25_hits + faiss_hits):
            key = d.page_content
            if key not in seen:
                seen.add(key)
                candidate_docs.append(d)

        if not candidate_docs:
            return "No info found."

        candidate_docs = candidate_docs[:20]
        
        # 3. Cross-Encoder Re-ranking
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.cross_encoder.predict(pairs)
        
        # Zip, Sort, and Filter
        scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Threshold filtering (adjust based on model performance)
        final_results = [doc for doc, score in scored_docs if score > -5.0]

        if not final_results:
            # Fallback to top 2 raw results if reranker hates everything
            final_results = candidate_docs[:2]

        # Return formatted string
        return "\n\n".join([f"[Result {i+1}]\n{d.page_content}" for i, d in enumerate(final_results[:5])])
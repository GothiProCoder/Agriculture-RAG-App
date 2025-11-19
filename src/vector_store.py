import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
from src.config import Config

class RetrievalEngine:
    """
    Manages the indexing and retrieval of document data using a hybrid approach
    (Dense + Sparse vectors) and Cross-Encoder re-ranking.
    """

    def __init__(self, df):
        self.df = df
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(Config.CROSS_ENCODER_MODEL)
        self.ensemble_retriever = self._build_index()

    def _build_index(self):
        """Constructs the granular multi-vector index."""
        granular_documents = []

        for index, row in self.df.iterrows():
            full_info = (
                f"Gaushala: {row['Gaushala_Name']}\n"
                f"ID: {row['Registration_No']}\n"
                f"Location: {row['District']}, {row['Village']}\n"
                f"Contact: {row['Contact_Person']} ({row['Phone_Number']})\n"
                f"Cattle: {row['Cattle_Count']}\n"
                f"Status: {row['Status']}"
            )

            meta = {
                "row_id": int(index),
                "contact_person": str(row['Contact_Person']),
                "registration_no": str(row['Registration_No']),
                "full_info": full_info 
            }

            # Vector A: Contact Specialist
            if row['Contact_Person'] != "Not Available":
                doc_a = Document(
                    page_content=f"Contact Person: {row['Contact_Person']}. Phone: {row['Phone_Number']}.",
                    metadata=meta
                )
                granular_documents.append(doc_a)

            # Vector B: ID Specialist
            doc_b = Document(
                page_content=f"Registration Number: {row['Registration_No']}.",
                metadata=meta
            )
            granular_documents.append(doc_b)

            # Vector C: Location/Name Specialist
            doc_c = Document(
                page_content=f"Gaushala Name: {row['Gaushala_Name']}. Located in {row['Village']}, District {row['District']}.",
                metadata=meta
            )
            granular_documents.append(doc_c)

        # Build Retrievers
        vector_store = FAISS.from_documents(granular_documents, self.embeddings)
        bm25_retriever = BM25Retriever.from_documents(granular_documents)
        bm25_retriever.k = Config.BM25_K

        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_store.as_retriever(search_kwargs={"k": Config.FAISS_K})],
            weights=[0.6, 0.4]
        )
        return ensemble

    def search(self, query: str) -> str:
        """
        Performs hybrid search with re-ranking.
        """
        # 1. Exact ID Detection
        gsa_match = re.search(r'(?:GSA)[\s\.\-]?0*(\d+)', query, re.IGNORECASE)
        if gsa_match:
            target_id_part = gsa_match.group(1)
            # Simple string match on extracted ID
            matches = self.df[self.df['Registration_No'].astype(str).str.contains(target_id_part, na=False)]
            if not matches.empty:
                row = matches.iloc[0]
                return f"🎯 EXACT MATCH:\nGaushala: {row['Gaushala_Name']}\nReg No: {row['Registration_No']}\nContact: {row['Contact_Person']} ({row['Phone_Number']})\nStatus: {row['Status']}"

        # 2. Ensemble Retrieval
        candidate_docs = self.ensemble_retriever.invoke(query)
        if not candidate_docs:
            return "No relevant information found."

        # 3. Re-Ranking
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.cross_encoder.predict(pairs)
        
        scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        
        final_results = []
        seen_ids = set()

        for doc, score in scored_docs:
            row_id = doc.metadata['row_id']
            if row_id in seen_ids: continue # Dedup based on original row
            
            if score > Config.RERANK_THRESHOLD:
                final_results.append(doc.metadata['full_info'])
                seen_ids.add(row_id)

            if len(final_results) >= 5:
                break

        if not final_results:
            # Fallback to top raw results if scores are low
            return candidate_docs[0].metadata['full_info']

        return "\n\n".join([f"[Result {i+1}]\n{res}" for i, res in enumerate(final_results)])
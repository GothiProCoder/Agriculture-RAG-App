import os
import hashlib
import io
import pandas as pd
from typing import Tuple

from src.data_processor import parse_gaushala_pdf
from src.vector_store import RetrievalEngine

# This folder will act as your local database
ARTIFACTS_DIR = "artifacts_store"

class IngestionManager:
    def __init__(self):
        """Initialize storage directory."""
        if not os.path.exists(ARTIFACTS_DIR):
            os.makedirs(ARTIFACTS_DIR)

    def _get_file_hash(self, file_bytes: bytes) -> str:
        """Generates a unique MD5 hash based on file content."""
        return hashlib.md5(file_bytes).hexdigest()

    def process_upload(self, file_bytes: bytes, file_name: str) -> Tuple[pd.DataFrame, RetrievalEngine]:
        """
        Orchestrates the data pipeline:
        1. Check if we have seen this file before (via Hash).
        2. If yes -> Load from Disk (Fast).
        3. If no -> Parse PDF -> Build Index -> Save to Disk (Slow first time).
        
        Returns:
            pd.DataFrame: The structured data.
            RetrievalEngine: The loaded/built vector store.
        """
        file_hash = self._get_file_hash(file_bytes)
        
        # Define paths for persistence
        parquet_path = os.path.join(ARTIFACTS_DIR, f"{file_hash}.parquet")
        index_path = os.path.join(ARTIFACTS_DIR, f"{file_hash}_index")

        # Initialize Engine (Models will load only if needed or forced)
        retrieval_engine = RetrievalEngine(load_models_now=True)

        # --- PATH 1: FAST LOAD (Cache Hit) ---
        if os.path.exists(parquet_path) and os.path.exists(index_path):
            print(f"🚀 Cache Hit! Loading existing data for {file_name}...")
            
            try:
                # Load Dataframe
                df = pd.read_parquet(parquet_path)
                
                # Load Vector Index
                retrieval_engine.load_local(index_path)
                
                return df, retrieval_engine
            except Exception as e:
                print(f"⚠️ Cache Corrupted ({e}). Reprocessing...")
                # Fall through to cold start if load fails

        # --- PATH 2: COLD START (New File) ---
        print(f"⚙️ New Data Detected. Processing {file_name}...")
        
        # 1. Convert bytes to file-like object for pdfplumber
        file_obj = io.BytesIO(file_bytes)
        
        # 2. Parse PDF (CPU Intensive)
        df = parse_gaushala_pdf(file_obj)
        
        if df.empty:
            # Return empty state rather than crashing, but log error
            print("❌ PDF extraction returned empty DataFrame.")
            raise ValueError("Could not extract data from PDF. Please check format.")

        # 3. Save Structured Data (Parquet preserves Int types)
        df.to_parquet(parquet_path)
        print(f"✅ Data saved to {parquet_path}")
        
        # 4. Build & Save Vector Index (GPU/CPU Intensive)
        retrieval_engine.build_index(df)
        retrieval_engine.save_local(index_path)
        
        return df, retrieval_engine
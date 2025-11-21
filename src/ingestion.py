import os
import hashlib
import io
import json
import shutil
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Optional

from src.data_processor import parse_gaushala_pdf
from src.vector_store import RetrievalEngine

# This folder will act as your local database
ARTIFACTS_DIR = "artifacts_store"
METADATA_FILE = "metadata.json"
GLOBAL_INDEX_DIR = os.path.join(ARTIFACTS_DIR, "global_index")

class IngestionManager:
    def __init__(self):
        """Initialize storage directory and metadata."""
        if not os.path.exists(ARTIFACTS_DIR):
            os.makedirs(ARTIFACTS_DIR)
        
        self.metadata_path = os.path.join(ARTIFACTS_DIR, METADATA_FILE)
        if not os.path.exists(self.metadata_path):
            self._save_metadata({})

    def _get_file_hash(self, file_bytes: bytes) -> str:
        """Generates a unique MD5 hash based on file content."""
        return hashlib.md5(file_bytes).hexdigest()

    def _load_metadata(self) -> Dict:
        """Loads metadata from JSON file."""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_metadata(self, metadata: Dict):
        """Saves metadata to JSON file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def get_all_artifacts(self) -> List[Dict]:
        """Returns a list of all stored artifacts (files)."""
        metadata = self._load_metadata()
        artifacts = [
            {
                "file_hash": k,
                **v
            }
            for k, v in metadata.items()
        ]
        # Sort by date (newest first)
        artifacts.sort(key=lambda x: x.get("upload_date", ""), reverse=True)
        return artifacts
        
    def get_file_path(self, file_hash: str) -> Optional[str]:
        """Returns the path to the original PDF file if it exists."""
        pdf_path = os.path.join(ARTIFACTS_DIR, f"{file_hash}.pdf")
        if os.path.exists(pdf_path):
            return pdf_path
        return None

    def delete_artifact(self, file_hash: str) -> bool:
        """Deletes an artifact and its associated files."""
        metadata = self._load_metadata()
        if file_hash in metadata:
            # Delete files
            paths = [
                os.path.join(ARTIFACTS_DIR, f"{file_hash}.parquet"),
                os.path.join(ARTIFACTS_DIR, f"{file_hash}.pdf")
            ]
            
            for p in paths:
                if os.path.exists(p):
                    os.remove(p)
            
            # Remove from metadata
            del metadata[file_hash]
            self._save_metadata(metadata)
            return True
        return False

    def load_global_data(self) -> pd.DataFrame:
        """Loads and concatenates ALL parquet files in the artifacts store."""
        metadata = self._load_metadata()
        dfs = []
        
        for file_hash in metadata.keys():
            parquet_path = os.path.join(ARTIFACTS_DIR, f"{file_hash}.parquet")
            if os.path.exists(parquet_path):
                try:
                    df = pd.read_parquet(parquet_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Error loading parquet {file_hash}: {e}")
        
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)

    def load_global_index(self) -> Tuple[Optional[pd.DataFrame], Optional[RetrievalEngine]]:
        """
        Attempts to load the Global Index from disk.
        If it doesn't exist, it returns None, None (caller should trigger rebuild).
        """
        # 1. Load All Data
        global_df = self.load_global_data()
        if global_df.empty:
            return None, None
            
        # 2. Check if Index exists
        if os.path.exists(GLOBAL_INDEX_DIR):
            try:
                print("🚀 Loading Global Index...")
                retrieval_engine = RetrievalEngine(load_models_now=True)
                retrieval_engine.load_local(GLOBAL_INDEX_DIR)
                return global_df, retrieval_engine
            except Exception as e:
                print(f"⚠️ Global Index Corrupted ({e}). Needs rebuild.")
                return None, None
        
        return None, None

    def rebuild_global_index(self) -> Tuple[pd.DataFrame, RetrievalEngine]:
        """
        Force rebuilds the global index from all current artifacts.
        """
        print("🔄 Rebuilding Global Index...")
        global_df = self.load_global_data()
        
        if global_df.empty:
            raise ValueError("No data available to build index.")
            
        retrieval_engine = RetrievalEngine(load_models_now=True)
        retrieval_engine.build_index(global_df)
        retrieval_engine.save_local(GLOBAL_INDEX_DIR)
        
        return global_df, retrieval_engine

    def process_upload(self, file_bytes: bytes, file_name: str) -> bool:
        """
        Processes a new upload:
        1. Parse PDF
        2. Save Parquet & PDF
        3. Update Metadata
        
        Note: This does NOT rebuild the index immediately. 
        The app should call rebuild_global_index() after upload is done.
        """
        file_hash = self._get_file_hash(file_bytes)
        metadata = self._load_metadata()
        
        # Define paths
        parquet_path = os.path.join(ARTIFACTS_DIR, f"{file_hash}.parquet")
        pdf_path = os.path.join(ARTIFACTS_DIR, f"{file_hash}.pdf")

        # Always overwrite/save to ensure freshness
        file_obj = io.BytesIO(file_bytes)
        df = parse_gaushala_pdf(file_obj)
        
        if df.empty:
            raise ValueError("Could not extract data from PDF.")

        # Save
        df.to_parquet(parquet_path)
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
            
        # Update Metadata
        metadata[file_hash] = {
            "filename": file_name,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "row_count": len(df),
            "file_size": len(file_bytes)
        }
        self._save_metadata(metadata)
        
        return True

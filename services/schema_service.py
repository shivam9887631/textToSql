import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any

from config import (
    SCHEMA_FILE, 
    VECTOR_INDEX_FILE, 
    VECTOR_NAMES_FILE, 
    VECTOR_DESC_FILE,
    LAST_FETCH_FILE
)
from .embedding_service import LocalEmbedder
from .database_service import DatabaseService

class SchemaService:
    def __init__(self, model_name: str):
        self.embedder = LocalEmbedder(model_name)
        self.index = None
        self.table_data = []
        self.table_names = []
    
    def load_schema_data(self, schema_json: Dict):
        self.table_data = []
        self.table_names = []
        
        for table_name, columns in schema_json.items():
            self.table_names.append(table_name)
            
            column_descriptions = [
                f"{col['column_name']} ({col['data_type']}, {'NULL' if col['nullable'] == 'Y' else 'NOT NULL'})" 
                for col in columns
            ]
            
            table_desc = f"Table {table_name} with columns: {', '.join(column_descriptions)}"
            self.table_data.append(table_desc)
        
        if self._is_index_valid():
            self.index = faiss.read_index(VECTOR_INDEX_FILE)
        else:
            self.build_index()
        
        with open(VECTOR_NAMES_FILE, 'w') as f:
            json.dump(self.table_names, f)
        
        with open(VECTOR_DESC_FILE, 'w') as f:
            json.dump(self.table_data, f)
    
    def _is_index_valid(self) -> bool:
        """Check if existing FAISS index is valid"""
        if not all(os.path.exists(f) for f in [VECTOR_INDEX_FILE, VECTOR_NAMES_FILE, VECTOR_DESC_FILE]):
            return False
        
        try:
            with open(VECTOR_NAMES_FILE, 'r') as f:
                existing_names = json.load(f)
            
            with open(VECTOR_DESC_FILE, 'r') as f:
                existing_descriptions = json.load(f)
            
            if (len(existing_names) != len(self.table_names) or 
                set(existing_names) != set(self.table_names) or
                existing_descriptions != self.table_data):
                return False
            
            test_index = faiss.read_index(VECTOR_INDEX_FILE)
            return test_index.ntotal == len(self.table_names)
            
        except Exception:
            return False
    
    def build_index(self):
        if not self.table_data:
            self.index = None
            return
        
        table_vectors = self.embedder.encode(self.table_data)
        vector_dimension = table_vectors.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)
        self.index.add(table_vectors)
        faiss.write_index(self.index, VECTOR_INDEX_FILE)
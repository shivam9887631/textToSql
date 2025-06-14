from typing import List, Dict, Any
from services.schema_service import SchemaService

class VectorSearch(SchemaService):
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword search when vector search fails"""
        query_lower = query.lower()
        results = []
        
        for i, table_name in enumerate(self.table_names):
            score = 0
            table_lower = table_name.lower()
            
            if table_lower in query_lower or query_lower in table_lower:
                score += 10
            
            if i < len(self.table_data):
                desc_lower = self.table_data[i].lower()
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 2 and word in desc_lower:
                        score += 3
            
            if score > 0:
                results.append({
                    "table_name": table_name,
                    "similarity_score": float(score),
                    "description": self.table_data[i] if i < len(self.table_data) else f"Table {table_name}"
                })
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant tables using vector similarity"""
        if self.index is not None and self.index.ntotal > 0:
            try:
                query_vector = self.embedder.encode([query])
                scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(self.table_names) and scores[0][i] > 0:
                        results.append({
                            "table_name": self.table_names[idx],
                            "similarity_score": float(scores[0][i]),
                            "description": self.table_data[idx]
                        })
                
                results.sort(key=lambda x: x["similarity_score"], reverse=True)
                
                if results:
                    return results
                    
            except Exception:
                pass
        
        return self.keyword_search(query, top_k)
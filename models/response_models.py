from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class SQLResponse(BaseModel):
    nl_query: str
    sql_query: str
    relevant_tables: List[Dict[str, Any]]
    execution_time: float

class StatusResponse(BaseModel):
    db_connected: bool
    last_updated: Optional[str] = None
    schema_tables_count: Optional[int] = None
from pydantic import BaseModel

class NLQueryRequest(BaseModel):
    query: str
    model: str = "mistral-large-latest"
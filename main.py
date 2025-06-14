import os
import json
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String
from pydantic import BaseModel
import uvicorn
import logging

from config import (
    SCHEMA_FILE,
    LAST_FETCH_FILE,
    MISTRAL_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DB_CONFIG
)
from models.request_models import NLQueryRequest
from models.response_models import SQLResponse, StatusResponse
from utils.logger import setup_logger
from utils.vector_search import VectorSearch
from services.sql_generation_service import SQLGenerationService
from services.database_service import DatabaseService
from database import Base, engine, get_db

# ---------- Setup ----------
app = FastAPI(
    title="Natural Language to SQL API",
    description="Convert natural language queries to SQL using AI",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = setup_logger()
# Ensure uvicorn logs go to our logger
logging.getLogger("uvicorn.error").handlers = logger.handlers

schema_search = VectorSearch(DEFAULT_EMBEDDING_MODEL)
schema_loaded = False

# ---------- ORM User Table ----------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    name: str
    email: str

@app.post("/users", tags=["Users"])
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(name=user.name, email=user.email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.get("/users", tags=["Users"])
def read_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.get("/", tags=["Status"])
async def root():
    return {"message": "API running (schema via Oracle/local fallback)"}

# ---------- Schema Loader with Oracle + Local fallback ----------
async def get_schema_search():
    global schema_loaded, schema_search
    if schema_loaded:
        return schema_search

    # First try local file if exists
    if os.path.exists(SCHEMA_FILE):
        try:
            with open(SCHEMA_FILE, 'r') as f:
                schema_json = json.load(f)
            logger.info(f"Loaded schema from local file {SCHEMA_FILE}")
            schema_search.load_schema_data(schema_json)
            schema_loaded = True
            return schema_search
        except Exception as e:
            logger.warning(f"Failed loading local schema.json: {e}")

    # If no local file or failed to load, try Oracle via DatabaseService
    try:
        logger.info("Attempting to fetch schema from Oracle")
        schema_json, error = DatabaseService.fetch_schema()
        if error:
            raise Exception(error)
        # Save locally for next time
        try:
            with open(SCHEMA_FILE, 'w') as f:
                json.dump(schema_json, f, indent=2)
            with open(LAST_FETCH_FILE, 'w') as f:
                f.write(datetime.now().isoformat())
            logger.info(f"Fetched schema from Oracle and saved to {SCHEMA_FILE}")
        except Exception as save_err:
            logger.warning(f"Fetched schema but failed to save locally: {save_err}")
        schema_search.load_schema_data(schema_json)
        schema_loaded = True
        return schema_search
    except Exception as e:
        # Final fallback: if still a local file existed but failed earlier, or no schema at all:
        msg = str(e)
        logger.error(f"Oracle schema fetch failed: {msg}")
        # If local file exists but was invalid earlier, return error:
        if os.path.exists(SCHEMA_FILE):
            raise HTTPException(status_code=500, detail=f"Oracle fetch failed: {msg}. Local schema file exists but failed to load.")
        else:
            raise HTTPException(status_code=503, detail=f"Schema not available: {msg}. Provide a valid local {SCHEMA_FILE} or fix Oracle connection.")

# ---------- Endpoint to force-refresh schema from Oracle ----------
@app.post("/update-schema", tags=["Schema"])
async def update_schema():
    global schema_loaded, schema_search
    try:
        schema_json, error = DatabaseService.fetch_schema()
        if error:
            raise Exception(error)
        # Save to local file
        with open(SCHEMA_FILE, "w") as f:
            json.dump(schema_json, f, indent=2)
        with open(LAST_FETCH_FILE, "w") as f:
            f.write(datetime.now().isoformat())
        schema_search.load_schema_data(schema_json)
        schema_loaded = True
        logger.info("Schema updated via /update-schema")
        return {
            "status": "success",
            "message": "Schema updated successfully",
            "tables_count": len(schema_json)
        }
    except Exception as e:
        msg = str(e)
        logger.error(f"update-schema failed: {msg}")
        raise HTTPException(status_code=500, detail=f"Error fetching schema: {msg}")

# ---------- Generate SQL Endpoint ----------
@app.post("/generate-sql", response_model=SQLResponse, tags=["SQL"])
async def generate_sql_endpoint(
    request: NLQueryRequest,
    background_tasks: BackgroundTasks,
    schema_search_dep: VectorSearch = Depends(get_schema_search)
):
    start_time = time.time()
    nl_query = request.query
    model = request.model

    if model not in MISTRAL_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {', '.join(MISTRAL_MODELS)}")

    # Kick off background refresh but do not block
    background_tasks.add_task(lambda: None)  # placeholder if you want periodic refresh

    try:
        relevant_tables = schema_search_dep.search(nl_query)
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching schema: {e}")

    if not relevant_tables:
        raise HTTPException(status_code=404, detail="No relevant tables found for your query")

    # Build schema text for Mistral prompt
    try:
        # Load full schema structure
        with open(SCHEMA_FILE, 'r') as f:
            schema_json = json.load(f)
    except Exception as e:
        logger.error(f"Failed reading {SCHEMA_FILE}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read schema file: {e}")

    schema_text = ""
    for table in relevant_tables:
        table_name = table.get('table_name')
        if table_name and table_name in schema_json:
            schema_text += f"Table: {table_name}\nColumns:\n"
            for col in schema_json[table_name]:
                nullable = "NULL" if col.get('nullable') == 'Y' else "NOT NULL"
                schema_text += f"- {col.get('column_name')} ({col.get('data_type')}, {nullable})\n"
            schema_text += "\n"

    # Debug log
    logger.debug(f"Schema text sent to SQLGenerationService:\n{schema_text}")

    # Call Mistral
    sql_query, error = SQLGenerationService.generate_sql(schema_text, nl_query, model)
    if error:
        logger.error(f"SQLGenerationService error: {error}")
        raise HTTPException(status_code=500, detail=error)

    execution_time = round(time.time() - start_time, 3)
    return SQLResponse(
        nl_query=nl_query,
        sql_query=sql_query,
        relevant_tables=relevant_tables,
        execution_time=execution_time
    )

@app.get("/models", tags=["Models"])
async def get_models():
    return {"models": MISTRAL_MODELS}

@app.get("/schema-tables", tags=["Schema"])
async def get_schema_tables(schema_search_dep: VectorSearch = Depends(get_schema_search)):
    try:
        with open(SCHEMA_FILE, 'r') as f:
            schema_json = json.load(f)
        return {
            "tables_count": len(schema_json),
            "tables": schema_json
        }
    except Exception as e:
        logger.error(f"Error loading schema-tables: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading schema: {e}")

@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status():
    is_available = os.path.exists(SCHEMA_FILE)
    tables_count = None
    last_updated = None
    if is_available:
        try:
            with open(SCHEMA_FILE, 'r') as f:
                schema_json = json.load(f)
                tables_count = len(schema_json)
            if os.path.exists(LAST_FETCH_FILE):
                with open(LAST_FETCH_FILE, 'r') as f:
                    lf = f.read().strip()
                    last = datetime.fromisoformat(lf)
                    last_updated = last.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.error(f"Error in status endpoint: {e}")
    return StatusResponse(
        db_connected=is_available,
        last_updated=last_updated,
        schema_tables_count=tables_count
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

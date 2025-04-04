import os

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from pydantic import BaseModel, Field

class QueryInput(BaseModel):
    query: str = Field(description="Korean query to search in the vector DB")

@tool("pdf_search", args_schema=QueryInput)
def search_vector_db(query):
    """Search the vector database pdf documents using the provided query."""
    password = os.getenv("POSTGRESQL_PASSWORD")
    connection = f"postgresql+psycopg://promartians:{password}@192.168.40.202:5432/ragnar"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    pgvector = PGVector(
        embeddings=embeddings,
        collection_name="54",
        connection=connection,
        use_jsonb=True
    )
    return pgvector.similarity_search_with_score(
                query=query,
                k=5
            )
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm import answer_question


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Research RAG API",
    version="1.0.0",
    description="Hybrid RAG API using FAISS, BM25, RRF and Gemini",
)


# ============================================================
# CORS
# Allows the Next.js frontend to communicate with FastAPI
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Research RAG API is running",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Research RAG API",
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        result = answer_question(question)

        # ----------------------------------------------------
        # If answer_question already returns a dictionary
        # ----------------------------------------------------

        if isinstance(result, dict):

            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
            }

        # ----------------------------------------------------
        # If answer_question returns a plain string
        # ----------------------------------------------------

        return {
            "answer": str(result),
            "sources": [],
        }

    except Exception as e:

        print("RAG ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}",
        )


# ============================================================
# APPLICATION INFO
# ============================================================

@app.get("/status")
def status():

    return {
        "api": "Research RAG API",
        "status": "running",
        "retrieval": [
            "BGE Embeddings",
            "FAISS",
            "BM25",
            "Reciprocal Rank Fusion",
        ],
        "generation": "Gemini",
    }
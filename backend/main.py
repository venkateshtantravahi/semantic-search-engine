from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.search.keyword import KeywordSearcher
from backend.search.semantic import SemanticSearcher
from backend.search.hybrid import HybridSearcher

app = FastAPI(title="AI Semantic Search Engine")

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now (restrict in prod)
    allow_methods=["*"],
    allow_headers=["*"],
)

keyword_searcher = KeywordSearcher()
semantic_searcher = SemanticSearcher()
hybrid_searcher = HybridSearcher(alpha=0.5)


@app.get("/search/keyword")
def keyword_search(query: str = Query(..., min_length=2), top_k: int = 5):
    return keyword_searcher.search(query, top_k)


@app.get("/search/semantic")
def semantic_search(query: str = Query(..., min_length=2), top_k: int = 5):
    return semantic_searcher.search(query, top_k)


@app.get("/search/blended")
def blended_search(query: str = Query(..., min_length=2), top_k: int = 5):
    return hybrid_searcher.search(query, top_k)
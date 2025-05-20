import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SemanticSearcher:
    """
    Handles semantic search using dense vector similarity (FAISS + BERT).
    """

    def __init__(self, data_path="/Users/venkateshtantravahi/Downloads/semantic-search-engine/data/clean_arxiv.csv",
                 index_path="/Users/venkateshtantravahi/Downloads/semantic-search-engine/models/faiss_index.bin",
                 embedding_path="/Users/venkateshtantravahi/Downloads/semantic-search-engine/models/semantic_embeddings.npy",
                 model_name="all-MiniLM-L6-v2"):
        self.df = pd.read_csv(data_path)
        self.embeddings = np.load(embedding_path)
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5):
        """
        Returns top_k semantically relevant documents using FAISS.

        Args:
            query (str): User query
            top_k (int): Number of results
        Returns:
            list: Top-k document dicts
        """
        query_vec = self.model.encode([query])
        _, indices = self.index.search(np.array(query_vec), top_k)

        results = []
        for idx in indices[0]:
            results.append({
                "id": self.df.iloc[idx]["id"],
                "title": self.df.iloc[idx]["clean_title"],
                "abstract": self.df.iloc[idx]["clean_abstract"],
                "score": 1.0  # Optional: semantic similarity score
            })

        return results
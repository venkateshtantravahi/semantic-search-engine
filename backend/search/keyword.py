import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class KeywordSearcher:
    """
    Handles keyword-based search using TF-IDF vector similarity.
    """

    def __init__(self,
                 data_path="/Users/venkateshtantravahi/Downloads/semantic-search-engine/data/clean_arxiv.csv",
                 vectorizer_path="/Users/venkateshtantravahi/Downloads/semantic-search-engine/models/tfidf_vectorizer.pkl",
                 vector_path="/Users/venkateshtantravahi/Downloads/semantic-search-engine/models/tfidf_vectors.pkl"):
        self.df = pd.read_csv(data_path)
        self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
        self.vectors = pickle.load(open(vector_path, "rb"))


    def search(self, query: str, top_k: int = 5):
        """
        Returns top_k matching documents using TF-IDF similarity.

        Args:
            query (str): User query
            top_k (int): Number of results to return
        Returns:
            list: Top-k document dicts
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices = scores.argsort()[-top_k:][::1]

        results = []

        for idx in top_indices:
            results.append({
                "id": self.df.iloc[idx]["id"],
                "title": self.df.iloc[idx]["clean_title"],
                "abstract": self.df.iloc[idx]["clean_abstract"],
                "score": round(float(scores[idx]), 4)
            })


        return results
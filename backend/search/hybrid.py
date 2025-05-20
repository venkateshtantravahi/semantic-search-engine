from backend.search.keyword import KeywordSearcher
from backend.search.semantic import SemanticSearcher


class HybridSearcher:
    """
    Combines TF-IDF and Semantic similarity scores for hybrid ranking.
    """

    def __init__(self, alpha=0.5):
        """
        Args:
            alpha (float): Weight for TF-IDF score (between 0 and 1).
        """
        self.keyword_searcher = KeywordSearcher()
        self.semantic_searcher = SemanticSearcher()
        self.alpha = alpha

    def search(self, query: str, top_k: int = 5):
        """
        Performs blended scoring of keyword and semantic matches.

        Args:
            query (str): The user's query string.
            top_k (int): Number of top results to return.
        Returns:
            list: Ranked documents with blended score.
        """
        keyword_results = self.keyword_searcher.search(query, top_k=50)
        semantic_results = self.semantic_searcher.search(query, top_k=50)

        # Map papers by ID to scores
        keyword_map = {doc['id']: doc['score'] for doc in keyword_results}
        semantic_map = {doc['id']: doc['score'] for doc in semantic_results}

        all_ids = set(keyword_map) | set(semantic_map)

        merged = []
        for doc_id in all_ids:
            tfidf_score = keyword_map.get(doc_id, 0)
            sem_score = semantic_map.get(doc_id, 0)
            final_score = self.alpha * tfidf_score + (1 - self.alpha) * sem_score

            doc = next((item for item in semantic_results if item["id"] == doc_id), None) or \
                  next((item for item in keyword_results if item["id"] == doc_id), None)

            merged.append({
                "id": doc_id,
                "title": doc["title"],
                "abstract": doc["abstract"],
                "score": round(final_score, 4)
            })

        # Sort by blended score
        merged = sorted(merged, key=lambda x: x["score"], reverse=True)[:top_k]
        return merged

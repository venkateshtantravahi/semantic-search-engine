import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFIndexer:
    """
    Builds a TF-IDF-based inverted index from the cleaned arXiv abstracts.

    Attributes:
        input_path (str): Path to the cleaned dataset.
        model_path (str): Path to save the TF-IDF model.
        vector_path (str): Path to save the transformed vectors.
    """

    def __init__(self,
                 input_path="../data/clean_arxiv.csv",
                 model_path="../models/tfidf_vectorizer.pkl",
                 vector_path="../models/tfidf_vectors.pkl"):
        self.input_path = input_path
        self.model_path = model_path
        self.vector_path = vector_path
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)


    def load_data(self) -> pd.DataFrame:
        """
        Loads the cleaned abstract data.
        """
        return pd.read_csv(self.input_path)


    def build_index(self, texts: list):
        """
        Fits the TF-IDF model and transforms the text data.

        Args:
            texts (list): List of documents (abstracts).
        Returns:
            sparse matrix: TF-IDF vector representation of texts.
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return tfidf_matrix


    def save_model(self, matrix):
        """
        Saves the TF-IDF model and transformed vectors.

        Args:
            matrix: The TF-IDF matrix.
        """
        os.makedirs("../models", exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.vector_path, "wb") as f:
            pickle.dump(matrix, f)
        print("TF-IDF models and vectors saved.")


    def run(self):
        """
        Orchestrates the full indexing pipeline.
        """
        print("Building TF-IDF index ...")
        df = self.load_data()
        tfidf_matrix = self.build_index(df["clean_abstract"].tolist())
        self.save_model(tfidf_matrix)


if __name__ == "__main__":
    indexer = TFIDFIndexer()
    indexer.run()
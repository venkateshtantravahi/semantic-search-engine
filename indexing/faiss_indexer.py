import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer


class FAISSIndexer:
    """
    Builds a semantic vector index using BERT embeddings and FAISS.

    Attributes:
        input_path (str): Path to the cleaned dataset.
        index_path (str): Path to save the FAISS index.
        embedding_path (str): Path to save document embeddings.
    """
    def __init__(self,
                 input_path="../data/clean_arxiv.csv",
                 index_path="../models/faiss_index.bin",
                 embedding_path="../models/semantic_embeddings.npy",
                 model_name="all-MiniLM-L6-v2"):
        self.input_path = input_path
        self.index_path = index_path
        self.embedding_path = embedding_path
        self.model = SentenceTransformer(model_name_or_path=model_name)


    def load_data(self) -> pd.DataFrame:
        """
        Loads the cleaned abstract data.
        """
        return pd.read_csv(self.input_path)


    def build_embeddings(self, texts: list) -> np.ndarray:
        """
        Computes BERT embeddings for a list of texts.

        Args:
            texts (list): Abstracts.
        Returns:
            np.ndarray: Dense vector embeddings.
        """
        return np.array(self.model.encode(texts, show_progress_bar=True, batch_size=32))


    def build_faiss_index(self, vectors: np.ndarray):
        """
        Builds a FAISS index from dense vectors.

        Args:
            vectors (np.ndarray): Document embeddings.
        Returns:
            FAISS index
        """
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        return index


    def save(self, vectors, index):
        """
        Saves the embeddings and FAISS index.
        """
        os.makedirs("../models", exist_ok=True)
        np.save(self.embedding_path, vectors)
        faiss.write_index(index, self.index_path)
        print("Embeddings and FAISS index saved.")


    def run(self):
        """
        Runs the full semantic indexing pipeline.
        """
        print("Building FAISS semantic index...")
        df = self.load_data()
        texts = df["clean_abstract"].tolist()
        vectors = self.build_embeddings(texts)
        index = self.build_faiss_index(vectors)
        self.save(vectors, index)



if __name__ == "__main__":
    indexer = FAISSIndexer()
    indexer.run()
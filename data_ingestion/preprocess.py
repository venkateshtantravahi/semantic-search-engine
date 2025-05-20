import pandas as pd
import os
import re


class ArxivPreprocessor:
    """
    Cleans and preprocesses arXiv paper metadata for indexing and semantic search.

    Attributes:
        input_path (str): Path to the raw dataset.
        output_path (str): Path to save the cleaned dataset.
    """

    def __init__(self, input_path="../data/raw_airxiv.csv", output_path="../data/clean_arxiv.csv"):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the raw dataset from a CSV file.

        Returns:
            pd.DataFrame: Raw data.
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found at: {self.input_path}")
        return pd.read_csv(self.input_path)

    def clean_text(self, text: str) -> str:
        """
        Cleans a string by removing LaTeX, punctuation, and extra spaces.

        Args:
            text (str): Input string.
        Returns:
            str: Cleaned string.
        """
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\$.*?\$', '', text)                    # Remove LaTeX math
        text = re.sub(r'\n', ' ', text)                        # Remove line breaks
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)              # Remove special characters
        text = re.sub(r'\s+', ' ', text)                       # Remove multiple spaces
        return text.strip().lower()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning to all relevant text fields in the dataset.

        Args:
            df (pd.DataFrame): Raw data.
        Returns:
            pd.DataFrame: Cleaned data.
        """
        df = df.drop_duplicates(subset="title")
        df["clean_title"] = df["title"].apply(self.clean_text)
        df["clean_abstract"] = df["summary"].apply(self.clean_text)
        return df[["id", "published", "authors", "clean_title", "clean_abstract"]]

    def save(self, df: pd.DataFrame):
        """
        Saves the cleaned dataset to a CSV file.

        Args:
            df (pd.DataFrame): Cleaned data.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Cleaned dataset saved to {self.output_path}")

    def run(self):
        """
        Full preprocessing pipeline: load → clean → save.
        """
        print("🧼 Preprocessing raw arXiv dataset...")
        raw_df = self.load_data()
        clean_df = self.preprocess(raw_df)
        self.save(clean_df)


if __name__ == "__main__":
    processor = ArxivPreprocessor()
    processor.run()
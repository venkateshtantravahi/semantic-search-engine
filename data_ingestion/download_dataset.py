import requests
import xml.etree.ElementTree as ET
import os
import pandas as pd

class AirxivDatasetDownloder:
    """
    Downloads metadata for academic research papaers from airxiv API.

    Attributes:
        category (str): ArXiv category to filter (e.g., cs.CL, cs.AI).
        max_results (int): Number of papers to download.
        output_path (str): File path to save the data.
    """

    def __init__(self, category="cs.CL", max_results=100, output_path="data/raw_airxiv.csv"):
        self.category = category
        self.max_results = max_results
        self.output_path = output_path
        self.base_url = "http://export.arxiv.org/api/query"


    def fetch(self):
        """
        Sends request to arXiv API and parses response into structured metadata.
        Returns:
            pd.DataFrame: DataFrame containing paper title, abstract, authors, etc.
        """
        params  = {
            "search_query": f"cat:{self.category}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.status_code}")

        root = ET.fromstring(response.content)
        entries = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            paper = {
                "id": entry.find("{http://www.w3.org/2005/Atom}id").text,
                "title": entry.find("{http://www.w3.org/2005/Atom}title").text.strip(),
                "summary": entry.find("{http://www.w3.org/2005/Atom}summary").text.strip(),
                "published": entry.find("{http://www.w3.org/2005/Atom}published").text,
                "authors": ", ".join(
                    [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
                )
            }
            entries.append(paper)

        return pd.DataFrame(entries)


    def save(self, df: pd.DataFrame):
        """
        Saves the DataFrame to a CSV file.
        Args:
            df (pd.DataFrame): The metadata DataFrame to save.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Dataset saved to {self.output_path} successfully.")


    def run(self):
        """
        Orchestrates the download and save process.
        """
        print(f"📥 Fetching {self.max_results} papers from arXiv category: {self.category}")
        df = self.fetch()
        self.save(df)



if __name__ == "__main__":
    downloader = AirxivDatasetDownloder(category="cs.CL", max_results=1000)
    downloader.run()
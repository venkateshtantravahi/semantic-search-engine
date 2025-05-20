# AI-Powered Semantic Search Engine

A hobby project that blends classic search engine architecture with modern AI/NLP to enable powerful **semantic search** over academic research papers (like arXiv abstracts). Built with love using **FastAPI**, **Streamlit**, **BERT**, **FAISS**, and **TF-IDF** — all modular, local, and open-source.

---

## About the Project

This project demonstrates how to build a lightweight semantic search engine from scratch using:

- **Traditional keyword-based search (TF-IDF + cosine similarity)**
- **Semantic vector search using BERT embeddings + FAISS**
- **A hybrid approach that blends both scores**

It includes:
- A Python-based **FastAPI backend** to serve search results
- A modern **Streamlit frontend** for a clean user experience
- Fully modular indexing, preprocessing, and search logic

---

## Features

### Core
- Full-text search over 1000+ academic paper abstracts
- Three search modes: `Semantic`, `Keyword`, `Blended`
- Top-k results with title, abstract, and score

### AI Enhancements
- BERT-based semantic search using `sentence-transformers`
- FAISS for high-speed vector similarity
- Score blending for hybrid relevance

### Frontend Experience
- Expandable abstracts
- Export search results to CSV
- Score filtering
- Local search history
- Keyword highlights
- Dark/light mode support

---

## Tech Stack

| Layer        | Tools / Libraries                              |
|-------------|-------------------------------------------------|
| Backend      | FastAPI, scikit-learn, faiss-cpu, sentence-transformers |
| Frontend     | Streamlit                                      |
| Indexing     | TF-IDF, FAISS                                  |
| Data         | arXiv metadata (fetched via API)               |
| Language     | Python 3.8+                                    |

---

## How to Use This Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/semantic-search-engine.git
cd semantic-search-engine
```

### 2. SettingUp Python Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Fetch and Prepare Data

```bash
python data_ingestion/download_dataset.py
python data_ingestion/preprocess.py
```


### 5. Build Indexes

```bash
python indexing/tfidf_indexer.py
python indexing/faiss_indexer.py
```

### 6. Start the Backend API
```bash
uvicorn backend.main:app --reload
```

### 7. Launch the Streamlit Frontend (in another terminal)
```bash
streamlit run frontend/app.py
```


## Example Search
1. Run the app.

2. Choose Semantic or Blended from the dropdown.

3. Try queries like:
    - `graph neural networks`
    - `language models for summarization`
    - `medical imaging with AI`

4. Adjust score filter or export results.

## Project Structure
```bash
semantic-search-engine/
├── backend/              # FastAPI backend & search logic
│   ├── main.py
│   └── search/
├── frontend/             # Streamlit app
│   └── app.py
├── data_ingestion/       # Dataset downloader and preprocessor
├── indexing/             # TF-IDF and FAISS indexing
├── notebooks/            # EDA & experiments
├── models/               # Saved vectorizers & indexes
├── data/                 # Raw and cleaned data
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes
- This project is not deployed, it's intended for local experimentation.
- You can change `search_mode` weight blending via `HybridSearcher(alpha=...)` in `hybrid.py`.


## Acknowledgments
- **arXiv** API for public access to paper metadata
- **HuggingFace** sentence-transformers
- FAISS by **Facebook** for scalable vector search
- **Streamlit** for an elegant frontend with minimal code


## Want to Learn More or Contribute?
Feel free to fork the repo, suggest improvements, or use parts of it in your own AI search projects.
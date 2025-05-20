import streamlit as st
import pandas as pd
import os
import requests
import re


API_URL = "http://localhost:8000"


# set page config
st.set_page_config(
    page_title="AI Semantic Search Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("🔍 AI Semantic Search Engine")
st.markdown("Search through academic paper abstracts using **Keyword**, **Semantic**, or **Blended** relevance.")


# Session State for search history
if "history" not in st.session_state:
    st.session_state.history = []


# side bar filters
with st.sidebar:
    st.header("⚙️ Search Configuration")
    search_mode = st.selectbox(
        "Search Type",
        ["Semantic", "Keyword", "Blended"],
        help="Semantic uses BERT embeddings, Keyword uses TF-IDF, and Blended combines both"
    )
    top_k = st.slider("Top K Results", 1, 20, 5)
    min_score = st.slider("Min Score Filter", 0.0, 1.0, 0.0, step=0.01)
    export = st.checkbox("Enable CSV Export")
    show_history = st.checkbox("Show Search History")


# Main Query Input
query = st.text_input("Enter your query:", placeholder="e.g., graph neural networks, NLP transformers...")


#Search Trigger
if st.button("Search") and query.strip():
    st.info(f"Searching using **{search_mode}** mode...")

    endpoint = f"{API_URL}/search/{search_mode.lower()}"
    params = {"query": query, "top_k": top_k}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        results = response.json()

        # Save to history
        st.session_state.history.append({"query": query, "mode": search_mode})

        # Filter by score
        results = [r for r in results if r.get("score", 0) >= min_score]

        if not results:
            st.warning("No results found above the score threshold.")
        else:
            st.success(f"Found {len(results)} results for: **{query}**")

            for idx, result in enumerate(results, 1):
                with st.expander(f"{idx}. {result['title']}"):
                    # Highlight keywords in abstract
                    abstract = result["abstract"]
                    for word in query.lower().split():
                        abstract = re.sub(fr"(?i)({re.escape(word)})", r"**\1**", abstract)

                    st.markdown(abstract)
                    st.caption(f"📊 Score: `{result.get('score', 0):.4f}` | 🆔 {result.get('id')}")


            # Export to CSV
            if export and results:
                df = pd.DataFrame(results)
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="search_results.csv",
                    mime="text/csv"
                )

    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")


# Show Search History
if show_history and st.session_state.history:
    st.markdown("---")
    st.subheader("🕓 Past Searches")
    for past in reversed(st.session_state.history[-5:]):
        st.markdown(f"🔹 `{past['mode']}` → **{past['query']}**")


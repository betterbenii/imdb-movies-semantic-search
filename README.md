# IMDB Semantic Searcher🎬

Welcome to IMDB movie semantic searcher,  This application helps you discover movies based on your preferences and provides detailed information about them. With a modern UI and dynamic data handling, 

## Features

- **Semantic Searching**: Query movies throughout the IMDB dataser
- **Interactive UI**: Enjoy a modern, responsive design with interactive elements like movie cards and chat-style messages.
- **Follow-Up Questions**: Easily explore related movies, ratings, and genres with quick follow-up options.
- **Search History**: Access your recent searches and rerun them with a single click.

## Installation

1. **Clone the Repository**:


2. **Set Up Virtual Environment**:

3. **Install Dependencies**:
 


5. **Run the Application**:
   Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- **`app.py`**: Main application file containing the Streamlit app logic.
- **`embeddings_cache.joblib`**: Cached embeddings for efficient similarity calculations.
- **`semantic_search_ready_imdb_with_embeddings.csv`**: Preprocessed movie data with embeddings.
- **`semantic_search_ready_imdb.csv`**: Original preprocessed movie data without embeddings.
- **`cleaned_imdb_2024.csv`**: Cleaned dataset of IMDB movies.
- **`search_engine.ipynb`**: Jupyter notebook for developing and testing the search engine.
- **`data_processing.ipynb`**: Jupyter notebook for data cleaning and preprocessing.

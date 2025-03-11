import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
from typing import List, Tuple, Dict

nltk.download('wordnet')
from nltk.corpus import wordnet

# Initialize session state for context retention and conversation
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'context' not in st.session_state:
    st.session_state.context = {}
if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = 'initial'
if 'current_movie_context' not in st.session_state:
    st.session_state.current_movie_context = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load preprocessed data with embeddings
@st.cache_data
def load_data():
    data = pd.read_csv('semantic_search_ready_imdb_with_embeddings.csv')
    cached_embeddings = joblib.load('embeddings_cache.joblib')
    return data, cached_embeddings

data, cached_embeddings = load_data()

# Initialize SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# Knowledge Graph Simulation
G = nx.DiGraph()
for _, row in data.iterrows():
    G.add_node(row['Movie_Name'], type='Movie', Overview=row['Overview'], Genres=row['Genres'])
    G.add_edge(row['Movie_Name'], row['Genres'], type='has_genre')

# Query Expansion Function
def expand_query(query: str) -> List[str]:
    try:
        terms = query.split()
        expanded_terms = []
        for term in terms:
            try:
                synsets = wordnet.synsets(term)
                if synsets:
                    # Add lemma names from the first synset
                    expanded_terms.extend([lemma.name() for lemma in synsets[0].lemmas()])
                else:
                    # If no synsets found, keep the original term
                    expanded_terms.append(term)
            except Exception as e:
                # If there's an error processing a term, keep the original term
                expanded_terms.append(term)
        
        # Remove duplicates and return
        return list(set(expanded_terms + terms))
    except Exception as e:
        # If anything goes wrong, return the original query terms
        st.warning(f"Query expansion encountered an error. Using original query terms.")
        return query.split()

# Search Function with Query Expansion
def search_with_expansion(query: str, data: pd.DataFrame, context: Dict = None) -> Tuple[List[int], List[float]]:
    # Use context to enhance the search if available
    if context and context.get('last_query') and context.get('last_results'):
        # Combine current query with context from previous query
        enhanced_query = f"{context['last_query']} {query}"
        expanded_query = expand_query(enhanced_query)
    else:
        expanded_query = expand_query(query)
    
    query_embedding = model.encode(' '.join(expanded_query))
    similarities = cosine_similarity([query_embedding], cached_embeddings)
    top_indices = similarities[0].argsort()[-5:][::-1]
    return top_indices, similarities[0][top_indices]

# Knowledge Graph Query Function
def query_knowledge_graph(query: str, G: nx.DiGraph) -> List[Tuple[str, str, str]]:
    results = []
    query_lower = query.lower()
    for node, attributes in G.nodes(data=True):
        if attributes.get('type') == 'Movie':
            if query_lower in attributes['Overview'].lower() or query_lower in attributes['Genres'].lower():
                results.append((node, attributes['Overview'], attributes['Genres']))
    return results[:5] if results else []

def update_context(query: str, results: List[Tuple[int, float]], selected_movie: dict = None) -> None:
    """Update the search context with the current query and results"""
    if len(st.session_state.search_history) >= 3:
        st.session_state.search_history.pop(0)
    st.session_state.search_history.append({
        'query': query,
        'results': results
    })
    st.session_state.last_results = results
    st.session_state.context = {
        'last_query': query,
        'last_results': results
    }
    if selected_movie:
        st.session_state.current_movie_context = selected_movie

def generate_follow_up_questions(movie_data: dict) -> List[str]:
    questions = [
        f"Would you like to know more about similar movies to '{movie_data['Movie_Name']}'?",
        f"Would you like to explore more {movie_data['Genres']} movies?",
        "Would you like to know about the ratings and reviews?",
        "Would you like to see movies with similar themes?"
    ]
    return questions

def handle_follow_up_response(response: str, movie_context: dict) -> str:
    if "similar movies" in response.lower():
        return f"Here are some movies similar to '{movie_context['Movie_Name']}'"
    elif "ratings" in response.lower():
        return f"Let me tell you about the ratings for '{movie_context['Movie_Name']}'"
    elif "explore more" in response.lower():
        return f"I'll show you more {movie_context['Genres']} movies"
    else:
        return "I'm not sure what you're asking about. Could you please rephrase your question?"

# Streamlit App UI
st.title('IMDB Movie Search Engine')

# Chat-like interface
st.markdown("### Chat History")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
query = st.text_input("What kind of movie are you looking for?")

# Search Button
if st.button('Search') or query:
    if query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Perform Search with context
        indices, scores = search_with_expansion(query, data, st.session_state.context)
        
        # Display results in chat format
        with st.chat_message("assistant"):
            st.write("I found these movies that might interest you:")
            for idx, score in zip(indices, scores):
                movie = data.iloc[idx]
                movie_data = {
                    'Movie_Name': movie['Movie_Name'],
                    'Genres': movie['Genres'],
                    'Overview': movie['Overview']
                }
                
                st.write(f"- **{movie['Movie_Name']}** - {movie['Genres']}")
                with st.expander("See details"):
                    st.write(f"Overview: {movie['Overview'][:200]}...")
                
                # Update context with the first movie as current context
                if idx == indices[0]:
                    update_context(query, list(zip(indices, scores)), movie_data)
        
        # Generate follow-up questions
        if st.session_state.current_movie_context:
            with st.chat_message("assistant"):
                st.write("Would you like to know more? You can ask about:")
                questions = generate_follow_up_questions(st.session_state.current_movie_context)
                for q in questions:
                    st.write(f"- {q}")
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Here are some movies that might interest you. Feel free to ask more specific questions about any of them!"
        })

# Handle follow-up queries
if st.session_state.current_movie_context and len(st.session_state.chat_history) > 0:
    follow_up = st.text_input("Ask me anything about these movies:", key="follow_up")
    
    # Add a button for follow-up search
    if st.button("Get Answer", key="follow_up_button"):
        if follow_up:
            # Add follow-up query to chat history
            st.session_state.chat_history.append({"role": "user", "content": follow_up})
            
            # Generate response based on context
            response = handle_follow_up_response(follow_up, st.session_state.current_movie_context)
            
            # Add initial response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Perform a new search based on the follow-up query using the embedded model
            with st.chat_message("assistant"):
                if "similar movies" in follow_up.lower():
                    # Create a query combining movie name, genres, and key terms from overview
                    movie_context = st.session_state.current_movie_context
                    similar_query = f"{movie_context['Movie_Name']} {movie_context['Genres']} {movie_context['Overview'][:100]}"
                    indices, scores = search_with_expansion(similar_query, data, None)
                    
                    st.write(f"📽️ **Similar Movies to '{movie_context['Movie_Name']}'**")
                    st.write("I found these movies that share similar themes, genres, or storytelling elements:")
                    
                    # Create columns for better organization
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        for idx, score in zip(indices, scores):
                            movie = data.iloc[idx]
                            if movie['Movie_Name'] != movie_context['Movie_Name']:
                                similarity_score = int(score * 100)
                                st.write(f"### {movie['Movie_Name']}")
                                st.write(f"**Genre:** {movie['Genres']}")
                                st.write(f"**Similarity Score:** {similarity_score}%")
                                with st.expander("Overview"):
                                    st.write(movie['Overview'][:600] + "...")
                                st.write("---")
                
                elif "ratings" in follow_up.lower():
                    # Display detailed ratings information with metrics
                    movie_name = st.session_state.current_movie_context['Movie_Name']
                    movie_data = data[data['Movie_Name'] == movie_name].iloc[0]
                    
                    st.write(f"📊 **Ratings Analysis for '{movie_name}'**")
                    
                    # Create three columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'Rating' in movie_data:
                            st.metric("IMDb Rating", f"{movie_data['Rating']}/10")
                    with col2:
                        if 'Votes' in movie_data:
                            st.metric("Total Votes", f"{movie_data['Votes']:,}")
                    with col3:
                        if 'Rating' in movie_data:
                            # Calculate a normalized score out of 100
                            normalized_score = int((float(movie_data['Rating']) / 10) * 100)
                            st.metric("Audience Score", f"{normalized_score}%")
                    
                    # Add rating distribution if available
                    st.write("\n### Rating Distribution")
                    st.write("Based on user votes and ratings:")
                    st.progress(float(movie_data['Rating']) / 10 if 'Rating' in movie_data else 0)
                
                elif "explore more" in follow_up.lower():
                    # Search for movies in the same genre using direct embedding search
                    current_genre = st.session_state.current_movie_context['Genres']
                    genre_query = f"best highly rated movies in genre {current_genre}"
                    indices, scores = search_with_expansion(genre_query, data, None)
                    
                    st.write(f"🎬 **Top {current_genre} Movies**")
                    st.write(f"Here are some highly-rated movies in the {current_genre} genre:")
                    
                    # Create a more organized layout
                    for idx, score in zip(indices, scores):
                        movie = data.iloc[idx]
                        if movie['Movie_Name'] != st.session_state.current_movie_context['Movie_Name']:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"### {movie['Movie_Name']}")
                                if 'Rating' in movie:
                                    st.write(f"**Rating:** ⭐ {movie['Rating']}/10")
                                st.write(f"**Genre Match:** {current_genre in movie['Genres']}")
                                with st.expander("Overview"):
                                    st.write(movie['Overview'][:600] + "...")
                            st.write("---")
                
                else:
                    # Enhanced general search with categorized results
                    indices, scores = search_with_expansion(follow_up, data, None)
                    st.write("🔍 **Search Results**")
                    st.write(f"Here's what I found based on your question: '{follow_up}'")
                    
                    # Group movies by genre for better organization
                    genres_found = set()
                    for idx, score in zip(indices, scores):
                        movie = data.iloc[idx]
                        genres_found.update(movie['Genres'].split(','))
                    
                    for genre in genres_found:
                        genre = genre.strip()
                        st.write(f"\n### {genre} Movies")
                        for idx, score in zip(indices, scores):
                            movie = data.iloc[idx]
                            if genre in movie['Genres']:
                                st.write(f"**{movie['Movie_Name']}**")
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    with st.expander("Overview"):
                                        st.write(movie['Overview'][:600] + "...")
                                with col2:
                                    if 'Rating' in movie:
                                        st.write(f"Rating: ⭐ {movie['Rating']}/10")
            
            # Update context with the new results while preserving previous context
            update_context(follow_up, list(zip(indices, scores)), st.session_state.current_movie_context)

# Display search history in sidebar
if st.session_state.search_history:
    st.sidebar.header("Recent Searches")
    for i, history in enumerate(st.session_state.search_history):
        st.sidebar.text(f"{i+1}. {history['query']}")

# Display context-aware suggestions
if st.session_state.last_results:
    st.sidebar.header("You might also like:")
    last_indices = [idx for idx, _ in st.session_state.last_results]
    for idx in last_indices[:3]:
        movie = data.iloc[idx]
        st.sidebar.text(f"- {movie['Movie_Name']}")

if __name__ == '__main__':
    st.write("Welcome! Ask me about any kind of movies you're interested in!")
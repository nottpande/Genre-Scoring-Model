import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


genre_embeddings = joblib.load('genre_embeddings.pkl')
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_summary_embedding(summary):
    """
    Convert summary to an embedding using SBERT.
    
    Args:
        summary (str): Movie summary text.
        
    Returns:
        np.array: Embedding of the summary.
    """
    return model.encode(summary,show_progress_bar=True)

def score_genres_for_summary(movie_name, summary, genre_embeddings):
    """
    Score genres based on the similarity between summary embedding and genre embeddings.
    
    Args:
        movie_name (str): Movie name.
        summary (str): Movie summary text.
        genre_embeddings (dict): Dictionary of genre embeddings.
        
    Returns:
        list: List of tuples (genre, similarity) sorted by descending similarity score.
    """
    summary_embedding = get_summary_embedding(summary)
    similarities = {}
    for genre, embeddings in genre_embeddings.items():
        similarity = cosine_similarity([summary_embedding], [embeddings])[0][0]
        similarities[genre] = similarity
    sorted_genres = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_genres

# Interactive input
movie_name = input("Enter the name of the movie: ")
summary = input("Enter the summary of the movie: ")
sorted_genres = score_genres_for_summary(movie_name, summary, genre_embeddings)
print(f"\nGenres scored for the movie '{movie_name}' based on the summary:")
for genre, score in sorted_genres:
    print(f"{genre}: {score:.4f}")

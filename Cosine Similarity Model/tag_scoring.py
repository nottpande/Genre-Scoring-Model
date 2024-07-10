import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


genre_embeddings = joblib.load('genre_embeddings.pkl')
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_tag_embeddings(tags):
    """
    Convert tags to embeddings using SBERT.
    
    Args:
        tags (list): List of tag strings.
        
    Returns:
        dict: Dictionary of tag embeddings.
    """
    return {tag: model.encode(tag) for tag in tags}

def find_most_similar_genres(tag_embedding, genre_embeddings, threshold):
    """
    Find the most similar genres for a given tag embedding.
    
    Args:
        tag_embedding (np.array): Embedding of the tag.
        genre_embeddings (dict): Dictionary of genre embeddings.
        threshold (float): Similarity threshold to consider multiple genres.
        
    Returns:
        list: List of tuples (genre, similarity) that are most similar to the tag embedding.
    """
    similarities = {}
    for genre, embeddings in genre_embeddings.items():
        similarity = cosine_similarity([tag_embedding], [embeddings])[0][0]
        similarities[genre] = similarity
    max_similarity = max(similarities.values())
    most_similar_genres = [(genre, similarity) for genre, similarity in similarities.items() if max_similarity - similarity <= threshold]
    
    return most_similar_genres

def score_tags(tags, genre_embeddings, threshold=0.1):
    """
    Score the tags based on their similarity to genre embeddings.
    
    Args:
        tags (list): List of tag strings.
        genre_embeddings (dict): Dictionary of genre embeddings.
        threshold (float): Similarity threshold to consider multiple genres.
        
    Returns:
        dict: Dictionary of tags and their corresponding genres and similarity scores.
    """
    tag_embeddings = get_tag_embeddings(tags)
    tag_scores = {}
    for tag, embedding in tag_embeddings.items():
        genres = find_most_similar_genres(embedding, genre_embeddings, threshold)
        tag_scores[tag] = genres
    return tag_scores

def process_and_score_tags(tag_string, genre_embeddings, threshold=0.1):
    """
    Process the input string to extract tags and score them.
    
    Args:
        tag_string (str): String of comma-separated tags.
        genre_embeddings (dict): Dictionary of genre embeddings.
        threshold (float): Similarity threshold to consider multiple genres.
        
    Returns:
        dict: Dictionary of tags and their corresponding genres and similarity scores.
    """
    tags = [tag.strip() for tag in tag_string.split(',')]
    tag_scores = score_tags(tags, genre_embeddings, threshold)
    return tag_scores

# Interactive input
tag_string = input("Enter the tags separated by commas: ")
threshold = float(input("Enter the similarity threshold: "))
tag_scores = process_and_score_tags(tag_string, genre_embeddings, threshold)
for tag, scores in tag_scores.items():
    print(f"{tag}:")
    for genre, score in scores:
        print(f"  {genre}: {score:.4f}")

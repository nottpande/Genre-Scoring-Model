import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class GenreScorer:
    def __init__(self, genre_embeddings_file, model):
        self.genre_embeddings = joblib.load(genre_embeddings_file)
        self.model = model
    
    def get_summary_embedding(self, summary):
        """
        Convert summary to an embedding using SBERT.
        """
        print("Creating embeddings of the summary")
        return self.model.encode(summary, show_progress_bar=True)
    
    def score_genres(self, summary):
        """
        Score genres based on the similarity between summary embedding and genre embeddings.
        """
        summary_embedding = self.get_summary_embedding(summary)
        similarities = {}
        for genre, embeddings in self.genre_embeddings.items():
            similarity = cosine_similarity([summary_embedding], [embeddings])[0][0]
            similarities[genre] = similarity
        X_min = min(similarities.values())
        X_max = max(similarities.values())
        scaled_similarities = {genre: 1 + (similarity - X_min) / (X_max - X_min) * 9 for genre, similarity in similarities.items()}
        sorted_genres = sorted(scaled_similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_genres

class TagScorer:
    def __init__(self, genre_embeddings_file, model):
        self.genre_embeddings = joblib.load(genre_embeddings_file)
        self.model = model
        self.threshold = 0.1  # You can adjust this as needed
    
    def get_tag_embeddings(self, tags):
        """
        Convert tags to embeddings using SBERT.
        """
        print("Creating embeddings of tags")
        return {tag: self.model.encode(tag,show_progress_bar=True) for tag in tags}
    
    def find_most_similar_genres(self, tag_embedding):
        """
        Find the most similar genres for a given tag embedding.
        """
        similarities = {}
        for genre, embeddings in self.genre_embeddings.items():
            similarity = cosine_similarity([tag_embedding], [embeddings])[0][0]
            similarities[genre] = similarity
        max_similarity = max(similarities.values())
        most_similar_genres = [(genre, similarity) for genre, similarity in similarities.items() if max_similarity - similarity <= self.threshold]
        
        return most_similar_genres
    
    def score_tags(self, tags, genre_scores):
        tag_embeddings = self.get_tag_embeddings(tags)
        tag_scores = {}

        for tag, embedding in tag_embeddings.items():
            genres = self.find_most_similar_genres(embedding)
            weighted_scores = [(genre, score * genre_scores.get(genre, 0.0)) for genre, score in genres]

            scores = [score for _, score in weighted_scores]
            if scores:
                score_min = min(scores)
                score_max = max(scores)
                if score_max != score_min:
                    scaled_scores = {genre: 1 + (score - score_min) / (score_max - score_min) * 9 for genre, score in weighted_scores}
                else:
                    scaled_scores = {genre: 1 for genre, score in weighted_scores}
                tag_scores[tag] = scaled_scores
            else:
                tag_scores[tag] = {}

        return tag_scores

    def process_and_score_tags(self, tag_string, genre_scores):
        """
        Process the input string to extract tags and score them.
        
        Args:
            tag_string (str): String of comma-separated tags.
            genre_scores (dict): Dictionary of genre scores.
            
        Returns:
            dict: Dictionary of tags and their corresponding genres and similarity scores.
        """
        tags = [tag.strip() for tag in tag_string.split(',')]
        tag_scores = self.score_tags(tags, genre_scores)
        return tag_scores

class MovieGenreTagPipeline:
    def __init__(self, genre_embeddings_file):
        try:
            print("Loading the model for encoding")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model Loaded Successfully!")
            
            print("Initializing the model for scoring the Genres")
            self.genre_scorer = GenreScorer(genre_embeddings_file, self.model)
            print("Initializing the model for scoring the tags")
            self.tag_scorer = TagScorer(genre_embeddings_file, self.model)
            
        except Exception as e:
            print("An error has occurred.")
            print(f"Error : {e}")
    
    def process_movie(self, summary, tags):
        """
        Process the movie to score genres and tags.
        """
        genre_scores = self.genre_scorer.score_genres(summary)
        genre_scores_dict = {genre: score for genre, score in genre_scores}
        tag_scores = self.tag_scorer.process_and_score_tags(tags, genre_scores_dict)
        
        return genre_scores, tag_scores
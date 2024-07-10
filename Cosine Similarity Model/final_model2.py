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
    
    def score_genres(self, movie_name, summary):
        """
        Score genres based on the similarity between summary embedding and genre embeddings.
        """
        summary_embedding = self.get_summary_embedding(summary)
        similarities = {}
        for genre, embeddings in self.genre_embeddings.items():
            similarity = cosine_similarity([summary_embedding], [embeddings])[0][0]
            similarities[genre] = similarity
        sorted_genres = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_genres

class TagScorer:
    def __init__(self, genre_embeddings_file, model):
        self.genre_embeddings = joblib.load(genre_embeddings_file)
        self.model = model
        self.threshold = 0.1
    
    def get_tag_embeddings(self, tags):
        """
        Convert tags to embeddings using SBERT.
        """
        print("Creating embeddings of tags")
        return {tag: self.model.encode(tag, show_progress_bar=True) for tag in tags}
    
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
        """
        Score the tags based on their similarity to genre embeddings, weighted by genre scores.
        """
        tag_embeddings = self.get_tag_embeddings(tags)
        tag_scores = {}
        for tag, embedding in tag_embeddings.items():
            genres = self.find_most_similar_genres(embedding)
            weighted_scores = [(genre, score * genre_scores.get(genre, 0.0)) for genre, score in genres]
            tag_scores[tag] = weighted_scores
        return tag_scores

def make_single_paragraph(summary):
    """
    Convert a multiline summary into a single paragraph by removing newlines and excess whitespace.
    """
    return ' '.join(summary.split()).strip()

class MovieGenreTagPipeline:
    def __init__(self, genre_embeddings_file):
        try:
            print("Loading the model for encoding")
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Model Loaded Successfully!")
            except Exception as e:
                print("An error has occured.")
                print(f"Error : {e}")
            try:
                print("Initializing the model for scoring the Genres")
                self.genre_scorer = GenreScorer(genre_embeddings_file, model)
                print("Initializing the model for scoring the tags")
                self.tag_scorer = TagScorer(genre_embeddings_file, model)
            except Exception as e:
                print("An error has occured.")
                print(f"Error : {e}")
        except Exception as e:
            print("An error has occured.")
            print(f"Error : {e}")

    
    def process_movie(self, movie_name, summary, tags):
        """
        Process the movie to score genres and tags.
        """
        genre_scores = self.genre_scorer.score_genres(movie_name, summary)
        tag_scores = self.tag_scorer.score_tags(tags, {genre: score for genre, score in genre_scores})
        
        return genre_scores, tag_scores


genre_embeddings_file = 'genre_embeddings.pkl'
pipeline = MovieGenreTagPipeline(genre_embeddings_file)

# Interactive input
movie_name = input("Enter the name of the movie: ")
print("Enter the summary of the movie (type 'END' in a new line, once you have entered the summary):")
summary_lines = []
while True:
    line = input()
    if line.strip().upper() == 'END':
        break
    summary_lines.append(line)
summary = '\n'.join(summary_lines)
summary = make_single_paragraph(summary)
tag_string = input("Enter the tags separated by commas: ")
genre_scores, tag_scores = pipeline.process_movie(movie_name, summary, tag_string.split(','))

print(f"\nGenres scored for the movie '{movie_name}' based on the summary:")
for genre, score in genre_scores:
    print(f"{genre}: {score:.4f}")

print(f"\nTags scored for the movie '{movie_name}' based on the genre understanding:")
for tag, scores in tag_scores.items():
    print(f"{tag}:")
    for genre, score in scores:
        print(f"  {genre}: {score:.4f}")

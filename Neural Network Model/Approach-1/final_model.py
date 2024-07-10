import sentence_transformers as st
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

class GenreClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(GenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def create_encodings(summary):
    model = st.SentenceTransformer("all-MiniLM-L6-v2")
    print("Creating Embeddings.")
    encoding = model.encode(summary, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    return encoding

def load_model(model_path='scoring_model_1.pth', input_size=384, output_size=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GenreClassifier(input_size, output_size)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_genres(genre_file='genres.joblib'):
    genre_file = joblib.load(genre_file)
    genre_labels = genre_file.columns.tolist()
    return genre_labels

def predict_genres(encoding, model):
    encoding = encoding.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        scores = model(encoding)
    return scores.squeeze(0)  # Remove batch dimension

def map_predictions_to_genres(scores, genre_names):
    genre_scores = {genre: score.item() for genre, score in zip(genre_names, scores)}
    return genre_scores

def main():
    movie_name = input("Enter the name of the movie: ")
    summary = input("Enter the summary of the movie: ")
    encoding = create_encodings(summary)
    genre_names = load_genres('genres.joblib')
    num_genres = len(genre_names)  # Get the number of genres
    model = load_model('scoring_model_1.pth', input_size=384, output_size=num_genres)
    scores = predict_genres(encoding, model)
    genre_scores = map_predictions_to_genres(scores, genre_names)
    
    return movie_name, genre_scores

movie_name, genre_scores = main()
print(f"Movie: {movie_name}")
print("Predicted Genres Scores:")
for genre, score in genre_scores.items():
    print(f"{genre}: {score}")

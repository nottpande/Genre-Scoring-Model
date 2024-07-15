from pipeline import MovieGenreTagPipeline
import pandas as pd
import numpy as np

dataset = pd.read_csv('final_data.csv',usecols=['node_id','Title','extracted_genres'])
id = pd.read_excel("Top 100 Movies.xlsx")
top_movies = pd.merge(dataset, id, on='node_id', how='inner')
df = pd.read_csv('../Data/all_tags_generated_LLm.csv',usecols=['Title','summary','tags_gpt'])
top_100 = pd.merge(top_movies, df, on='Title', how='inner').drop_duplicates().reset_index(drop=True)

def calculation(df, genre_embeddings_file):
    pipeline = MovieGenreTagPipeline(genre_embeddings_file)
    
    results = {
        'movie_name': [],
        'summary': [],
        'tags_gpt': [],
        'genre_scores': [],
        'tag_scores': []
    }
    
    for index, row in df.iterrows():
        movie_name = row['Title']
        summary = row['summary']
        tags = row['tags_gpt']
        
        genre_scores, tag_scores = pipeline.process_movie(summary,tags)
        
        formatted_tag_scores = []
        for tag, scores in tag_scores.items():
            score_str = f"{tag}:\n"
            for genre, score in scores:
                score_str += f"  {genre}: {score:.4f}\n"
            formatted_tag_scores.append(score_str.strip())
        
        results['movie_name'].append(movie_name)
        results['summary'].append(summary)
        results['tags_gpt'].append(tags)
        results['genre_scores'].append(genre_scores)
        results['tag_scores'].append("\n".join(formatted_tag_scores))
    
    return pd.DataFrame(results)

results_df = calculation(top_100, 'genre_embeddings.pkl')
results_df.to_csv('PREDICTIONS.csv')
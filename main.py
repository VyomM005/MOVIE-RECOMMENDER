import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ==========================================
# PHASE 1 & 2: DATA LOADING & PREPROCESSING
# ==========================================

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge on title
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Helper functions to clean JSON-like strings
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Apply cleaning
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Remove spaces to create unique tags (e.g., "Johnny Depp" -> "JohnnyDepp")
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

# Convert overview to list and create 'tags' column
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Final dataframe
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# ==========================================
# PHASE 3: MACHINE LEARNING (VECTORIZATION)
# ==========================================

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# ==========================================
# PHASE 4 & 5: RECOMMENDATION & EXPORT
# ==========================================

def recommend(movie):
    movie_lower = movie.lower()
    try:
        # Find index using case-insensitive search
        movie_index = new_df[new_df['title'].str.lower() == movie_lower].index[0]
        distances = similarity[movie_index]
        
        # Sort similarities
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        print(f"\nRecommendations for '{new_df.iloc[movie_index].title}':")
        for i in movies_list:
            print(f"- {new_df.iloc[i[0]].title}")
            
    except IndexError:
        print(f"\nError: '{movie}' not found. Please check your spelling!")


# Save the movie dictionary (the list of movies and tags)
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))

# Save the similarity matrix (the "brain")
pickle.dump(similarity, open('similarity.pkl', 'wb'))

# Export for Phase 5 (uncomment if you want to generate files for a UI later)
# pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
# pickle.dump(similarity, open('similarity.pkl', 'wb'))

# ==========================================
# INTERACTIVE USER INTERFACE
# ==========================================

print("\n--- Welcome to the Movie Recommender System ---")
while True:
    print("\n" + "="*40)
    user_input = input("Enter a movie name (or type 'quit' to exit): ").strip()
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    if user_input:
        recommend(user_input)
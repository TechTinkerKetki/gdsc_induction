import pandas as pd
import  os
import numpy as np
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

#extracting the zipfile

zip_path="Users\ketki\Projects\code1\python files\songs 2.zip"
song_data='songs 2'

with zipfile.ZipFile(zip_path,'r') as zip_ref:
    zip_ref.extractall(song_data)
print('files extracted')

#loading the csv into pandas
jls_extract_var = song_data
file_path=os.path.join(song_data,'tracks.csv')
df=pd.read_csv(file_path)

#Analysis
print(df.head())
print(df.columns)
df.fillna("", inplace=True)   #checking for missing values

#preprocessing
df["combined_features"] = (
   df["artist_genres"].astype(str) + " " +    
   df["tempos"].astype(str) + " " +  
   df["danceability"].astype(str) + " " +  
   df["energy"].astype(str) + " " +  
   df["valences"].astype(str)
)

X=TfidfVectorizer(stop_words='english')
X_matrix = X.fit_transform(df["combined_features"])

#cosine
cosine_sim = cosine_similarity(X_matrix, X_matrix)

def get_correct_song_name(song_title, df):
    if song_title in df["names"].values:
        return song_title  # Exact match found
    
    best_match, score = process.extractOne(song_title, df["names"].values)
    
    if score > 80:  # Threshold for good match
        return best_match
    else:
        return None  # No good match found

# Function to recommend similar songs
def recommend_songs(song_title, df, cosine_sim, top_n=6):
    corrected_song = get_correct_song_name(song_title, df)
    
    if corrected_song is None:
        return "Song not found!"

    print(f"Did you mean: {corrected_song}?")

    # Get song index
    idx = df[df["names"] == corrected_song].index[0]

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations (excluding itself)
    song_indices = [i[0] for i in sim_scores[1:top_n + 1]]

    # Return recommended songs
    return df.iloc[song_indices][["names", "artist_names"]]



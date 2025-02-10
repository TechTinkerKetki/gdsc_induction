import streamlit as st
import RS_gdsc  # Import the recommendation logic

st.title("🎵 Song Recommender System")
song_input = st.text_input("Enter a song name:")

if song_input:
    recommendations = RS_gdsc.recommend_songs(song_input)
    
    if isinstance(recommendations, str):  # If fuzzy matching suggests a correction
        st.write(recommendations)
    else:
        st.write("### Recommended Songs 🎶")
        st.dataframe(recommendations)

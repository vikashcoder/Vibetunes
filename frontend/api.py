import streamlit as st
import requests
from transformers import pipeline
import torch
import os
import time

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"  # Change this to your deployment URL

st.set_page_config(page_title="VibeTunes - Mood-Based Playlist Finder", page_icon="🎵", layout="centered")

# 🎨 Custom CSS for animations and styling
st.markdown(
    """
    <style>
    .playlist-card {
    border-radius: 15px; /* Rounded corners */
    transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
}

.playlist-card:hover {
    transform: scale(1.05); /* Slightly enlarges the card */
    box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.3); /* Adds a subtle shadow effect */
}


    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 1.5s ease-in-out;
    }
    
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
        animation: fadeIn 2s ease-in-out;
    }
    
    .sentiment-box {
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .positive { background-color: #d4edda; color: #155724; }
    .neutral { background-color: #fff3cd; color: #856404; }
    .negative { background-color: #f8d7da; color: #721c24; }
    
    .playlist-section {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        margin-bottom: 30px;
        box-shadow: 3px 3px 20px rgba(0, 0, 0, 0.1);
    }
    
    .playlist-card {
        background: linear-gradient(to right, #ff9a9e, #fad0c4);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .playlist-card h3 {
        color: #fff;
    }
    .playlist-card img {
    border-radius: 15px; /* Makes corners round */
    transition: transform 0.3s ease-in-out; /* Smooth transition */
}

    .playlist-card a {
        color: #fff;
        font-weight: bold;
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>🎶 VibeTunes: Discover Playlists That Match Your Mood</div>", unsafe_allow_html=True)

# ✅ Load trained sentiment model
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_DIR, device=0 if torch.cuda.is_available() else -1)

# ✅ Sentiment Analysis Function
def analyze_sentiment(text: str) -> str:
    result = sentiment_pipeline(text)[0]
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    return label_map.get(result['label'], "Neutral")

playlist_name = st.text_area("🎤 How are you feeling today? Share your vibe:")

# ✅ Show detected sentiment with animation
if playlist_name:
    sentiment = analyze_sentiment(playlist_name)
    sentiment_class = "positive" if sentiment == "Positive" else "neutral" if sentiment == "Neutral" else "negative"
    st.markdown(f"<div class='sentiment-box {sentiment_class} fade-in'>🎭 Detected Mood: **{sentiment}**</div>", unsafe_allow_html=True)
    time.sleep(0.5)  # Smooth transition effect

if st.button("🔍 Find Playlists"):
    if playlist_name:
        try:
            response = requests.post(
                f"{BACKEND_URL}/search_playlist", json={"name": playlist_name}
            )
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()

            if data and "playlists" in data and "items" in data["playlists"]:
                playlists = data["playlists"]["items"]
                if playlists:
                    st.markdown("<div class='playlist-section'>", unsafe_allow_html=True)
                    st.write("## 🎼 Recommended Playlists:")
                    for i, playlist in enumerate(playlists):
                        try:
                            st.markdown(f"""
                            <div class='playlist-card fade-in'>
                                <h3>🎵 {i+1}. {playlist['name']}</h3>
                                <p>👤 Owner: {playlist['owner']['display_name']}</p>
                                <a href='{playlist['external_urls']['spotify']}' target='_blank'  style='color: #ff6f61; font-weight: bold; text-decoration: underline;'>🔗 Listen on Spotify</a>
                                <br>
                                <img src='{playlist['images'][0]['url']}' width='250px'>
                            </div>
                            """, unsafe_allow_html=True)
                        except:
                            print("Failed to process image properly, check file")  # troubleshooting
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("😕 No playlists found matching your mood.")
            else:
                st.warning("😕 No playlists found matching your mood.")

        except requests.exceptions.ConnectionError as e:
            st.error(f"❌ Error: Could not connect to the backend at {BACKEND_URL}. Please ensure the backend server is running.")
            st.error(f"🔍 Details: {e}")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Error: The backend returned an error: {e}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error: A request exception occurred: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

    else:
        st.warning("⚠️ Please enter a description of how you're feeling.")

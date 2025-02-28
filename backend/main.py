
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import requests
# from transformers import pipeline
# import torch
# import os
# app = FastAPI()

# # ✅ Spotify API Setup
# SPOTIFY_ACCESS_TOKEN = "BQAFKr9d-K-DANlI_yYMwdqaFnF6MFdIsGpOtvt9ErW_EDs0o3cE_E-TQR0i89jie4Mu2nLImouzPsvDkcGWcO5mfwpEZXWOxyn3qpC26XTsH-84kbRtmENiYC1wFZdguKWrQ3VKXuw"
# SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

# # ✅ Load trained sentiment model
# MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_DIR, device=0 if torch.cuda.is_available() else -1)

# # ✅ Request model
# class SearchRequest(BaseModel):
#     name: str

# # ✅ Sentiment Analysis Function
# def analyze_sentiment(text: str) -> str:
#     result = sentiment_pipeline(text)[0]
#     label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
#     return label_map.get(result['label'], "neutral")

# # ✅ Search Playlist API
# @app.post("/search_playlist")
# async def search_playlist(request: SearchRequest):
#     headers = {"Authorization": f"Bearer {SPOTIFY_ACCESS_TOKEN}"}
    
#     # ✅ Analyze Sentiment
#     sentiment = analyze_sentiment(request.name)
#     print(f"🔍 Sentiment detected: {sentiment}")

#     # ✅ Adjust search term based on sentiment
#     search_term = request.name
#     if sentiment == "positive":
#         search_term = f"{request.name} happy"
#     elif sentiment == "negative":
#         search_term = f"{request.name} sad"

#     params = {"q": search_term, "type": "playlist"}

#     try:
#         response = requests.get(SPOTIFY_SEARCH_URL, headers=headers, params=params)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import requests
import os
from transformers import pipeline
import torch
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ✅ Load environment variables
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
SPOTIFY_ACCESS_TOKEN = None

# ✅ Load trained sentiment model
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_DIR, device=0 if torch.cuda.is_available() else -1)

# ✅ Request model
class SearchRequest(BaseModel):
    name: str

# ✅ Function to get a new access token
def get_spotify_token():
    global SPOTIFY_ACCESS_TOKEN
    auth_response = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    if auth_response.status_code == 200:
        SPOTIFY_ACCESS_TOKEN = auth_response.json()["access_token"]
    else:
        raise HTTPException(status_code=500, detail="Failed to retrieve Spotify token")

# ✅ Sentiment Analysis Function
def analyze_sentiment(text: str) -> str:
    result = sentiment_pipeline(text)[0]
    label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    return label_map.get(result['label'], "neutral")

# ✅ Search Playlist API
@app.post("/search_playlist")
async def search_playlist(request: SearchRequest):
    global SPOTIFY_ACCESS_TOKEN
    if not SPOTIFY_ACCESS_TOKEN:
        get_spotify_token()

    headers = {"Authorization": f"Bearer {SPOTIFY_ACCESS_TOKEN}"}
    
    # ✅ Analyze Sentiment
    sentiment = analyze_sentiment(request.name)
    print(f"🔍 Sentiment detected: {sentiment}")
    
    # ✅ Adjust search term based on sentiment
    search_term = request.name
    if sentiment == "positive":
        search_term = f"{request.name} happy"
    elif sentiment == "negative":
        search_term = f"{request.name} sad"
    
    params = {"q": search_term, "type": "playlist"}
    
    try:
        response = requests.get(SPOTIFY_SEARCH_URL, headers=headers, params=params)
        if response.status_code == 401:  # Unauthorized (token expired)
            get_spotify_token()
            headers = {"Authorization": f"Bearer {SPOTIFY_ACCESS_TOKEN}"}
            response = requests.get(SPOTIFY_SEARCH_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

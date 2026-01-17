import os
import json
import pandas as pd
import numpy as np
import requests
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# --- 1. LOAD SECRETS ---
# Load .env file for local dev. On Render, this does nothing (uses real Env Vars)
load_dotenv()

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# Hugging Face Config (Private Repo)
HF_TOKEN = os.getenv("HF_TOKEN") 
HF_REPO_ID = "ValerianFourel/seoul-medical-facilities"
HF_FILENAME = "facilities_metareviews_rag_ready.parquet"

# Paths
# On Render, we use a persistent disk at /var/lib/chroma. Locally, we use ./chroma_db
CHROMA_PATH = "/var/lib/chroma" if os.getenv("RENDER") else "./chroma_db"

# --- GLOBAL VARIABLES ---
vector_db = None
df_facilities = None

# --- HELPER FUNCTIONS ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculates distance (km) between two GPS points."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def get_naver_coordinates(address_text: str):
    """Converts 'Itaewon' -> [lat, lon] using Naver Cloud API."""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("⚠️ Naver Keys missing. Skipping geocoding.")
        return None
    
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    # Naver Geocoding Endpoint
    url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query={address_text}"
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data.get('addresses'):
            # Naver returns x=long, y=lat
            x = float(data['addresses'][0]['x']) 
            y = float(data['addresses'][0]['y']) 
            return [y, x] # Return [lat, lon]
    except Exception as e:
        print(f"❌ Naver API Error: {e}")
    return None

# --- LIFESPAN MANAGER (STARTUP LOGIC) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, df_facilities
    print("🚀 Booting SeoulMedBot Backend...")

    # 1. DOWNLOAD & LOAD DATA FROM HUGGING FACE
    try:
        print(f"📥 Downloading dataset from {HF_REPO_ID}...")
        
        local_parquet_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        df_facilities = pd.read_parquet(local_parquet_path)
        # Ensure IDs are strings for consistency
        df_facilities['place_id'] = df_facilities['place_id'].astype(str)
        print(f"✅ Loaded {len(df_facilities)} facilities from Hugging Face.")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR downloading data: {e}")
        # Load dummy data to prevent server crash
        df_facilities = pd.DataFrame([{
            "place_id": "9999", "name": "System Error Clinic", "category": "General",
            "lat": 37.5665, "lon": 126.9780, "english_confidence_score": 1,
            "Summaries": ["Data failed to load. Please check logs."], "Key_Highlights": []
        }])

    # 2. SETUP CHROMA DB (With OpenAI Embeddings)
    # This persists the vector store to disk so we don't rebuild every time
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # We use OpenAI because it's smarter at understanding medical nuance than local models
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    try:
        # Try to load existing DB
        vector_db = client.get_collection("seoul_med_v1", embedding_function=openai_ef)
        print("✅ Existing Embeddings Found. Ready to search.")
    except:
        print("⚠️ No Embeddings found. Generating now... (This uses OpenAI credits)")
        vector_db = client.create_collection("seoul_med_v1", embedding_function=openai_ef)
        
        ids, docs, metas = [], [], []
        
        # Prepare Data for Vectorization
        for _, row in df_facilities.iterrows():
            # Build a rich text blob for the AI to search against
            highlights = ""
            if isinstance(row['Key_Highlights'], list):
                highlights = ", ".join([h.get('topic', '') for h in row['Key_Highlights']])
            
            # BLOB FORMAT: "Name (Category). Summary. Highlights."
            text_blob = f"{row['name']} ({row['category']}). {row['Summaries'][0] if isinstance(row['Summaries'], list) and row['Summaries'] else ''}. {highlights}"
            
            ids.append(str(row['place_id']))
            docs.append(text_blob)
            metas.append({"category": row['category']})

        # Batch Upload to avoid timeouts
        batch_size = 100
        print(f"   Indexing {len(ids)} records in batches of {batch_size}...")
        
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            vector_db.add(
                ids=ids[i:end],
                documents=docs[i:end],
                metadatas=metas[i:end]
            )
            print(f"   Processed {end}/{len(ids)}")
            
        print("✅ Indexing Complete.")

    yield
    print("🛑 Shutting down SeoulMedBot.")

# --- APP SETUP ---
app = FastAPI(lifespan=lifespan)
client = Groq(api_key=GROQ_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class State(BaseModel):
    specialty: Optional[str] = None
    location: Optional[str] = None
    lat_lon: Optional[List[float]] = None
    willingness_to_travel: str = "Nearby"
    language_pref: str = "Korean is fine"
    keywords: List[str] = []
    ready_to_search: bool = False

class ChatRequest(BaseModel):
    message: str
    current_state: State

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are the logic engine for "SeoulMedBot".
Your goal is to extract medical intent and location from the user's message into a structured JSON state.

**State Variables:**
1. `specialty`: Medical category (e.g., "Dentist", "Dermatology", "Plastic Surgery").
2. `location`: User's text location (e.g., "Gangnam", "Itaewon"). IGNORE if `lat_lon` is already set.
3. `keywords`: Specific symptoms or needs (e.g., "implants", "botox", "english speaking").
4. `language_pref`: "English Preferred" if user types in English, else "Korean is fine".
5. `ready_to_search`: Set to TRUE only if you have a `specialty` AND (`location` OR `lat_lon`).

**Rules:**
- If `lat_lon` is present, you do not need to ask for location.
- If `specialty` is missing, ask for it in `response_text`.
- Output strictly JSON.

**JSON Structure:**
{
  "state": { ...updated variables... },
  "response_text": "Natural language reply confirming action or asking clarification."
}
"""

# --- CHAT ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # 1. LLM REASONING (Groq / Llama 3)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current State: {req.current_state.model_dump_json()}\nUser Message: {req.message}"}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        ai_data = json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"response": "I encountered an error processing your request.", "state": req.current_state}

    new_state = ai_data.get("state", {})
    response_text = ai_data.get("response_text", "")
    results = []

    # 2. GEOCODING (Text -> Lat/Lon)
    # If LLM extracted a location name but we don't have coords yet, use Naver
    if new_state.get("location") and not new_state.get("lat_lon"):
        print(f"📍 Geocoding location: {new_state['location']}")
        coords = get_naver_coordinates(new_state['location'])
        if coords:
            new_state['lat_lon'] = coords
            print(f"   -> Found coords: {coords}")
        else:
            # If Naver fails, we might need to ask user for clarity
            pass

    # 3. HYBRID SEARCH EXECUTION
    if new_state.get("ready_to_search"):
        print("🔎 Executing Hybrid Search...")
        temp_df = df_facilities.copy()
        
        # A. Semantic Search (Vector DB)
        # We combine specialty and keywords for a rich query
        search_query = f"{new_state.get('specialty', '')} {' '.join(new_state.get('keywords', []))}"
        
        if len(search_query.strip()) > 2:
            try:
                # Get Top 50 semantic matches
                rag_results = vector_db.query(query_texts=[search_query], n_results=50)
                relevant_ids = rag_results['ids'][0]
                # Filter DataFrame to only these IDs
                temp_df = temp_df[temp_df['place_id'].isin(relevant_ids)]
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}. Falling back to keyword match.")
                # Fallback: Simple string match on category
                temp_df = temp_df[temp_df['category'].str.contains(new_state.get('specialty', ''), case=False, na=False)]

        # B. Distance Filtering (Pandas)
        if new_state.get("lat_lon"):
            u_lat, u_lon = new_state['lat_lon']
            # Calculate distance for every candidate
            temp_df['distance'] = haversine(u_lat, u_lon, temp_df['lat'], temp_df['lon'])
            
            # Determine Radius from UI State
            radius_map = {
                "Neighborhood": 2.0,
                "Nearby": 5.0,
                "City-wide": 10.0,
                "Don't Care": 25.0
            }
            # Default to 5km if unknown
            max_dist = radius_map.get(req.current_state.willingness_to_travel, 5.0)
            
            # Apply Filter
            temp_df = temp_df[temp_df['distance'] <= max_dist]
            
            # Sort by Distance (Closest first)
            temp_df = temp_df.sort_values('distance')

        # C. Language Filtering (Optional Hard Filter)
        if new_state.get("language_pref") == "English Preferred":
             # Example: Only show clinics with score >= 4
             temp_df = temp_df[temp_df['english_confidence_score'] >= 4]

        # 4. FINALIZE RESULTS
        # Convert top 3 to list of dicts
        results = temp_df.head(3).to_dict(orient="records")
        
        if results:
            response_text = f"I found {len(results)} excellent options near you."
        else:
            response_text = "I couldn't find any clinics matching your exact criteria within that distance. Try increasing the search radius."

    return {
        "response": response_text,
        "state": new_state,
        "results": results
    }

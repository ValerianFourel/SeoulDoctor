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
from typing import List, Optional, Dict, Any
from groq import Groq
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HF_TOKEN = os.getenv("HF_TOKEN") 
HF_REPO_ID = "ValerianFourel/seoul-medical-facilities"
HF_FILENAME = "facilities_metareviews_rag_ready.parquet"

CHROMA_PATH = "/var/lib/chroma" if os.getenv("RENDER") else "./chroma_db"

# Defaults
DEFAULT_LAT = 37.5219  # Yeouido
DEFAULT_LON = 126.9243
vector_db = None
df_facilities = None

# --- 2. HELPER FUNCTIONS ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculates distance (km) between two GPS points."""
    R = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def get_naver_coordinates(address_text: str):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return None
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query={address_text}"
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data.get('addresses'):
            x = float(data['addresses'][0]['x']) 
            y = float(data['addresses'][0]['y']) 
            return [y, x]
    except Exception as e:
        print(f"❌ Naver API Error: {e}")
    return None

# --- 3. LIFESPAN (STARTUP) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, df_facilities
    print("🚀 Booting SeoulMedBot Backend...")

    # A. LOAD DATA
    try:
        print(f"📥 Downloading dataset from {HF_REPO_ID}...")
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset", token=HF_TOKEN)
        df_facilities = pd.read_parquet(local_path)
        df_facilities['place_id'] = df_facilities['place_id'].astype(str)
        print(f"✅ Loaded {len(df_facilities)} facilities.")
    except Exception as e:
        print(f"❌ DATA LOAD ERROR: {e}")
        df_facilities = pd.DataFrame()

    # B. SETUP RAG
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    try:
        vector_db = client.get_collection("seoul_med_v2", embedding_function=openai_ef)
        print("✅ Embeddings Found.")
    except:
        print("⚠️ Generating Embeddings...")
        vector_db = client.create_collection("seoul_med_v2", embedding_function=openai_ef)
        
        ids, docs, metas = [], [], []
        for _, row in df_facilities.iterrows():
            # Create a rich semantic blob
            highlights = ", ".join([h.get('topic', '') for h in row['Key_Highlights']]) if isinstance(row['Key_Highlights'], list) else ""
            summary_en = row['Summaries'][0] if isinstance(row['Summaries'], list) and row['Summaries'] else ""
            
            text_blob = f"{row['name']} ({row['category']}). {summary_en} {highlights}"
            
            ids.append(str(row['place_id']))
            docs.append(text_blob)
            metas.append({"category": row['category']})

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            vector_db.add(ids=ids[i:end], documents=docs[i:end], metadatas=metas[i:end])
            
        print("✅ Indexing Complete.")

    yield
    print("🛑 Shutting down.")

app = FastAPI(lifespan=lifespan)
client = Groq(api_key=GROQ_API_KEY)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 4. MODELS ---
class State(BaseModel):
    specialty: Optional[str] = None
    location: Optional[str] = None
    lat_lon: Optional[List[float]] = None
    willingness_to_travel: str = "Nearby"
    language_pref: str = "Korean is fine"
    keywords: List[str] = []
    asked_for_location: bool = False
    ready_to_search: bool = False

class ChatRequest(BaseModel):
    message: str
    current_state: State

# --- 5. SYSTEM PROMPTS ---

# PROMPT 1: LOGIC EXTRACTOR (Strict JSON)
EXTRACTION_PROMPT = """
You are "SeoulMedBot". Extract medical intent into JSON.
**State Variables:**
1. `specialty`: Medical category (e.g., "Dentist").
2. `location`: User text location.
3. `lat_lon`: GPS.
4. `asked_for_location`: Boolean.
5. `ready_to_search`: Boolean.
**Logic Rules:**
- If specialty missing -> ask.
- If location/lat_lon missing AND `asked_for_location` is false -> ask.
- If location/lat_lon missing AND `asked_for_location` is true -> default to Yeouido, set `ready_to_search`=true.
- Detect Language: If user speaks English, set `language_pref`="English Preferred".
**Format:** { "state": {...}, "response_text": "..." }
"""

# PROMPT 2: ENGAGING RESPONSE GENERATOR (Natural Language)
GENERATION_PROMPT = """
You are a helpful Medical Concierge. 
The user asked for: "{user_query}"
We have found these top facilities:
{results_context}

**Your Goal:**
Write an **engaging, warm response** recommending these places. 
- **Deterministic:** You MUST mention the `Name` of the top results clearly.
- **Non-Deterministic:** Use the `Summary` and `Highlights` to explain *why* they are good matches (e.g., "This place is great because...").
- **Constraint:** Do NOT list addresses, phone numbers, or hours (the UI cards show these).
- **Engagement Hook:** You must maximize time on site. End by asking if they want deeper details like "Amenity details" or "Medical equipment info".

**Tone:** Professional, caring, helpful.
**Language:** Respond in {language}.
"""

# --- 6. CHAT ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # --- STEP 1: EXTRACTION (LLM) ---
    msgs = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": f"State: {req.current_state.model_dump_json()}\nUser: {req.message}"}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="qwen-2.5-32b", 
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        ai_data = json.loads(completion.choices[0].message.content)
    except:
        return {"response": "System error.", "state": req.current_state}

    new_state = ai_data.get("state", {})
    response_text = ai_data.get("response_text", "")
    final_results = []

    # --- STEP 2: GEOCODING ---
    if new_state.get("location") and not new_state.get("lat_lon"):
        coords = get_naver_coordinates(new_state['location'])
        if coords: new_state['lat_lon'] = coords

    search_lat_lon = new_state.get("lat_lon") or [DEFAULT_LAT, DEFAULT_LON]

    # --- STEP 3: SEARCH EXECUTION ---
    if new_state.get("ready_to_search"):
        temp_df = df_facilities.copy()
        
        # A. Filter
        query_text = f"{new_state.get('specialty', '')} {' '.join(new_state.get('keywords', []))}"
        if len(query_text) > 2:
            try:
                rag_results = vector_db.query(query_texts=[query_text], n_results=50)
                temp_df = temp_df[temp_df['place_id'].isin(rag_results['ids'][0])]
            except: pass 

        # B. Distance
        u_lat, u_lon = search_lat_lon
        temp_df['distance'] = haversine(u_lat, u_lon, temp_df['lat'], temp_df['lon'])
        radius_map = {"Neighborhood": 2, "Nearby": 5, "City-wide": 10, "Don't Care": 25}
        temp_df = temp_df[temp_df['distance'] <= radius_map.get(new_state.get("willingness_to_travel"), 5)]
        temp_df = temp_df.sort_values('distance')

        # C. English
        if new_state.get("language_pref") in ["English Preferred", "Must speak English"]:
             temp_df = temp_df[temp_df['english_confidence_score'] > 2]

        # D. Get Top Results
        results_raw = temp_df.head(3).to_dict(orient="records")

        # --- STEP 4: GENERATE ENGAGING RESPONSE (LLM) ---
        if results_raw:
            # 1. Prepare Context for LLM (Deterministic + Non-Deterministic)
            results_context = ""
            language = "English" if new_state.get("language_pref") == "English Preferred" else "Korean"
            
            for r in results_raw:
                # Select correct language summary
                summary = r['Summaries'][0] if language == "English" and r['Summaries'] else (r['Summaries_Korean'][0] if r['Summaries_Korean'] else "No summary.")
                highlights = str(r['Key_Highlights'])
                results_context += f"- Name: {r['name']}\n  Summary: {summary}\n  Highlights: {highlights}\n\n"

            # 2. Call LLM for Natural Generation
            gen_messages = [{
                "role": "system", 
                "content": GENERATION_PROMPT.format(
                    user_query=req.message, 
                    results_context=results_context,
                    language=language
                )
            }]
            
            gen_completion = client.chat.completions.create(
                model="qwen-2.5-32b",
                messages=gen_messages,
                temperature=0.7 # Slight creativity for warmth
            )
            response_text = gen_completion.choices[0].message.content

        else:
            response_text = "I couldn't find any clinics matching those exact criteria. Would you like to try a different specialty or area?"

        # --- STEP 5: CLEANUP PAYLOAD FOR FRONTEND ---
        for r in results_raw:
            final_results.append({
                "place_id": r.get("place_id"),
                "name": r.get("name"), # Deterministic
                "category": r.get("category"),
                "address": r.get("address"), # Deterministic
                "distance": r.get("distance"),
                "phone": r.get("phone"), # Deterministic
                "business_hours": r.get("business_hours"), # Deterministic
                "website": r.get("website") or r.get("url"),
                "english_confidence_score": r.get("english_confidence_score"),
                "Summaries": r.get("Summaries"),
                "amenities": r.get("amenities"), # Hidden Info for "Show More"
                "medical_info_parsed": r.get("medical_info_parsed") # Hidden Info
            })

    return {
        "response": response_text,
        "state": new_state,
        "results": final_results
    }
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
    """
    Converts a text address (e.g., 'Gangnam Station') into [lat, lon].
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("⚠️ Naver API Keys missing. Skipping geocoding.")
        return None
    
    print(f"📍 Querying Naver Maps for: '{address_text}'...")
    
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query={address_text}"
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data.get('addresses'):
            # Naver returns x=longitude, y=latitude
            x = float(data['addresses'][0]['x']) 
            y = float(data['addresses'][0]['y']) 
            print(f"   ✅ Found Coordinates: [{y}, {x}]")
            return [y, x] # Return [lat, lon]
        else:
            print("   ❌ Location not found by Naver.")
    except Exception as e:
        print(f"   ❌ Naver API Error: {e}")
    return None

def disambiguate_location_with_llm(user_location: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Uses LLM to match user's location text to actual database values.
    Returns: {
        "matched_districts": [...],
        "matched_dongs": [...],
        "confidence": "high|medium|low",
        "reasoning": "..."
    }
    """
    if not user_location:
        return {"matched_districts": [], "matched_dongs": [], "confidence": "low", "reasoning": "No location provided"}
    
    # Extract unique values from database
    districts = []
    dongs = []
    
    if 'file_district' in df.columns:
        districts = sorted(df['file_district'].dropna().unique().tolist())
    if 'file_dong' in df.columns:
        dongs = sorted(df['file_dong'].dropna().unique().tolist())
    
    if not districts and not dongs:
        return {"matched_districts": [], "matched_dongs": [], "confidence": "low", "reasoning": "No location data in database"}
    
    # Format lists for LLM
    districts_str = ", ".join(districts[:50]) if districts else "None available"  # Limit to avoid token overflow
    dongs_str = ", ".join(dongs[:100]) if dongs else "None available"
    
    print(f"🧠 LLM Location Disambiguation for: '{user_location}'")
    
    prompt = LOCATION_DISAMBIGUATION_PROMPT.format(
        user_location=user_location,
        districts_list=districts_str,
        dongs_list=dongs_str
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"User said: '{user_location}'. What are the best matches?"}
            ],
            temperature=0.0,
            max_completion_tokens=512,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        print(f"   ✅ LLM matched: Districts={result.get('matched_districts')}, Dongs={result.get('matched_dongs')}")
        print(f"   Confidence: {result.get('confidence')}, Reasoning: {result.get('reasoning')}")
        return result
        
    except Exception as e:
        print(f"   ❌ LLM Disambiguation Error: {e}")
        return {"matched_districts": [], "matched_dongs": [], "confidence": "low", "reasoning": f"Error: {e}"}

# --- 3. LIFESPAN (STARTUP) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, df_facilities
    print("🚀 Booting SeoulMedBot Backend...")

    # A. LOAD DATA & FIX COLUMNS
    try:
        print(f"📥 Downloading dataset from {HF_REPO_ID}...")
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset", token=HF_TOKEN)
        df_facilities = pd.read_parquet(local_path)
        
        # --- FIX: Rename columns to standard 'lat' and 'lon' ---
        if 'latitude' in df_facilities.columns:
            df_facilities.rename(columns={'latitude': 'lat'}, inplace=True)
        if 'longitude' in df_facilities.columns:
            df_facilities.rename(columns={'longitude': 'lon'}, inplace=True)
            
        # Ensure they are floats if they exist
        if 'lat' in df_facilities.columns and 'lon' in df_facilities.columns:
            df_facilities['lat'] = df_facilities['lat'].astype(float)
            df_facilities['lon'] = df_facilities['lon'].astype(float)
            print(f"✅ Coordinates available: lat/lon columns found")
        else:
            print(f"⚠️ WARNING: Lat/Lon columns not in dataset. Will use text-based location matching.")
            print(f"   Available columns: {df_facilities.columns.tolist()}")

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

LOCATION_DISAMBIGUATION_PROMPT = """
You are a Seoul geography expert. The user mentioned: "{user_location}"

Here are the actual districts and neighborhoods in our database:
**Districts (Gu):**
{districts_list}

**Neighborhoods (Dong):**
{dongs_list}

**Your Task:**
Analyze the user's location text and select the BEST matching values from the lists above.

**Output JSON Format:**
{{
  "matched_districts": ["district1", "district2"],  // Empty list [] if no match
  "matched_dongs": ["dong1", "dong2"],              // Empty list [] if no match
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation of your choices"
}}

**Rules:**
1. ONLY select values that exist in the provided lists
2. Match variations (e.g., "Gangnam" → "강남구", "Geumcheon gu" → "금천구")
3. If user says "near Gangnam station", match the district containing it
4. Return EMPTY lists [] if no reasonable match found
5. Be generous with "medium" confidence - Korean place names have many variations
"""

GENERATION_PROMPT = """
You are a helpful Medical Concierge. 
The user asked for: "{user_query}"

**Search Context:**
- Location: {location_context}
- Specialty: {specialty}

We have found these top facilities:
{results_context}

**Your Goal:**
Write an **engaging, warm response** recommending these places. 
- **Deterministic:** You MUST mention the `Name` of the top results clearly.
- **Non-Deterministic:** Use the `Summary` and `Highlights` to explain *why* they are good matches.
- **Constraint:** Do NOT list addresses, phone numbers, or hours.
- **Engagement Hook:** End by asking if they want deeper details like "Amenity details".

**Tone:** Professional, caring, helpful.
**Language:** Respond in {language}.
"""

# --- 6. CHAT ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    
    # --- STEP 1: LOGIC EXTRACTION (LLM) ---
    msgs = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": f"State: {req.current_state.model_dump_json()}\nUser: {req.message}"}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=msgs,
            temperature=0.0,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        ai_data = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {"response": "System error.", "state": req.current_state}

    new_state = ai_data.get("state", {})
    response_text = ai_data.get("response_text", "")
    
    # [LOGGING] State Inspector
    print("\n--- 🧠 STATE UPDATE ---")
    print(json.dumps(new_state, indent=2))
    print("-----------------------\n")

    # --- STEP 2: LOCATION RESOLUTION (Naver + LLM Disambiguation) ---
    location_context = ""
    matched_districts = []
    matched_dongs = []
    
    if new_state.get("location") and not new_state.get("lat_lon"):
        # Try Naver geocoding first
        coords = get_naver_coordinates(new_state['location'])
        if coords: 
            new_state['lat_lon'] = coords
            location_context = f"Using GPS coordinates for {new_state['location']}"
            print(f"✅ Geocoding Successful: {new_state['location']} -> {coords}")
        else:
            # Fallback: LLM-based location disambiguation
            print(f"⚠️ Geocoding failed, using LLM disambiguation...")
            disambiguation = disambiguate_location_with_llm(new_state['location'], df_facilities)
            matched_districts = disambiguation.get('matched_districts', [])
            matched_dongs = disambiguation.get('matched_dongs', [])
            
            if matched_districts or matched_dongs:
                location_parts = []
                if matched_districts:
                    location_parts.append(f"Districts: {', '.join(matched_districts)}")
                if matched_dongs:
                    location_parts.append(f"Neighborhoods: {', '.join(matched_dongs)}")
                location_context = " | ".join(location_parts)
                print(f"✅ LLM Disambiguation: {location_context}")
            else:
                location_context = "Seoul-wide search (no specific location matched)"

    # Fallback Center (only used if we have coordinates)
    search_lat_lon = new_state.get("lat_lon") or [DEFAULT_LAT, DEFAULT_LON]

    # --- STEP 3: SEARCH EXECUTION ---
    final_results = []
    
    if new_state.get("ready_to_search"):
        temp_df = df_facilities.copy()
        
        # A. Semantic Filter
        query_text = f"{new_state.get('specialty', '')} {' '.join(new_state.get('keywords', []))}"
        if len(query_text) > 2:
            try:
                rag_results = vector_db.query(query_texts=[query_text], n_results=50)
                temp_df = temp_df[temp_df['place_id'].isin(rag_results['ids'][0])]
            except: pass 

        # B. Location Filter - SMART MULTI-MODE
        has_coords = 'lat' in temp_df.columns and 'lon' in temp_df.columns and new_state.get("lat_lon")
        
        if has_coords:
            # MODE 1: Geographic distance-based search
            u_lat, u_lon = search_lat_lon
            temp_df['distance'] = haversine(u_lat, u_lon, temp_df['lat'], temp_df['lon'])
            
            radius_map = {"Neighborhood": 2, "Nearby": 5, "City-wide": 10, "Don't Care": 25}
            temp_df = temp_df[temp_df['distance'] <= radius_map.get(new_state.get("willingness_to_travel"), 5)]
            temp_df = temp_df.sort_values('distance')
            print(f"✅ MODE 1: Distance-based search (radius: {radius_map.get(new_state.get('willingness_to_travel'), 5)}km)")
            
        elif matched_districts or matched_dongs:
            # MODE 2: LLM-guided precise filtering
            print(f"✅ MODE 2: LLM-guided location filtering")
            
            filters = []
            if matched_districts and 'file_district' in temp_df.columns:
                filters.append(temp_df['file_district'].isin(matched_districts))
                print(f"   Filtering by districts: {matched_districts}")
            
            if matched_dongs and 'file_dong' in temp_df.columns:
                filters.append(temp_df['file_dong'].isin(matched_dongs))
                print(f"   Filtering by neighborhoods: {matched_dongs}")
            
            if filters:
                # Combine filters with OR logic (match either district OR dong)
                combined_filter = filters[0]
                for f in filters[1:]:
                    combined_filter = combined_filter | f
                
                temp_df = temp_df[combined_filter]
                print(f"   ✅ Found {len(temp_df)} facilities in selected areas")
            
            # Sort by semantic relevance (from RAG) instead of distance
            temp_df['distance'] = 0  # Placeholder for display
            
        else:
            # MODE 3: No location filter (city-wide search)
            print(f"⚠️ MODE 3: City-wide search (no location filter)")
            temp_df['distance'] = 0  # Placeholder for consistency

        # C. English Filter
        if new_state.get("language_pref") in ["English Preferred", "Must speak English"]:
             temp_df = temp_df[temp_df['english_confidence_score'] > 2]

        # D. Top Results
        results_raw = temp_df.head(3).to_dict(orient="records")

        # --- STEP 4: GENERATION (LLM) ---
        if results_raw:
            results_context = ""
            language = "English" if new_state.get("language_pref") == "English Preferred" else "Korean"
            
            for r in results_raw:
                summary = r['Summaries'][0] if language == "English" and r['Summaries'] else (r['Summaries_Korean'][0] if r['Summaries_Korean'] else "No summary.")
                highlights = str(r['Key_Highlights'])
                results_context += f"- Name: {r['name']}\n  Summary: {summary}\n  Highlights: {highlights}\n\n"

            gen_messages = [{
                "role": "system", 
                "content": GENERATION_PROMPT.format(
                    user_query=req.message,
                    location_context=location_context or "Seoul area",
                    specialty=new_state.get('specialty', 'healthcare'),
                    results_context=results_context,
                    language=language
                )
            }]
            
            gen_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=gen_messages,
                temperature=1.0, 
                max_completion_tokens=1024,
                top_p=1,
                stream=False
            )
            response_text = gen_completion.choices[0].message.content
        else:
            response_text = "I couldn't find any clinics matching those exact criteria. Would you like to try a different specialty or area?"

        # --- STEP 5: CLEANUP ---
        for r in results_raw:
            final_results.append({
                "place_id": r.get("place_id"),
                "name": r.get("name"),
                "category": r.get("category"),
                "address": r.get("address"),
                "distance": r.get("distance"),
                "phone": r.get("phone"),
                "business_hours": r.get("business_hours"),
                "website": r.get("website") or r.get("url"),
                "english_confidence_score": r.get("english_confidence_score"),
                "Summaries": r.get("Summaries"),
                "amenities": r.get("amenities"),
                "medical_info_parsed": r.get("medical_info_parsed")
            })

    return {
        "response": response_text,
        "state": new_state,
        "results": final_results,
        "location_info": {
            "matched_districts": matched_districts,
            "matched_dongs": matched_dongs,
            "location_context": location_context
        }
    }
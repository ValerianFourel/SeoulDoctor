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

CHROMA_PATH = "./chroma_db"
LOCAL_PARQUET_PATH = "./local_facilities_cache.parquet"

# Defaults
DEFAULT_LAT = 37.5219  # Yeouido
DEFAULT_LON = 126.9243
vector_db = None
df_facilities = None  # Full dataset
df_filtered = None    # Filtered subset (Summaries not null)

# Medical specialty mapping: English → Korean category
SPECIALTY_CATEGORY_MAP = {
    # Dental
    'dentist': '치과',
    'dental': '치과',
    'orthodontics': '치과',
    
    # Medical
    'internal medicine': '내과',
    'family medicine': '가정의학과',
    'pediatrics': '소아청소년과',
    'pediatrician': '소아청소년과',
    'obstetrics': '산부인과',
    'gynecology': '산부인과',
    'ob/gyn': '산부인과',
    'obgyn': '산부인과',
    'dermatology': '피부과',
    'ophthalmology': '안과',
    'eye doctor': '안과',
    'ent': '이비인후과',
    'ear nose throat': '이비인후과',
    'orthopedics': '정형외과',
    'neurology': '신경과',
    'psychiatry': '정신건강의학과',
    'urology': '비뇨기과',
    'general surgery': '외과',
    'surgery': '외과',
    'plastic surgery': '성형외과',
    'radiology': '영상의학과',
    'anesthesiology': '마취통증의학과',
    'rehabilitation': '재활의학과',
    'physical therapy': '재활의학과',
    
    # Specialized
    'cardiology': '순환기내과',
    'gastroenterology': '소화기내과',
    'pulmonology': '호흡기내과',
    'nephrology': '신장내과',
    'endocrinology': '내분비내과',
    'rheumatology': '류마티스내과',
    'oncology': '종양내과',
    'hematology': '혈액내과',
    
    # Other
    'pharmacy': '약국',
    'oriental medicine': '한의원',
    'traditional medicine': '한의원',
}

# --- 2. HELPER FUNCTIONS ---

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two GPS coordinates."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def download_and_cache_parquet():
    """Download parquet from HuggingFace and cache locally."""
    if os.path.exists(LOCAL_PARQUET_PATH):
        print(f"✅ Loading cached parquet from {LOCAL_PARQUET_PATH}")
        df = pd.read_parquet(LOCAL_PARQUET_PATH)
    else:
        print(f"📥 Downloading parquet from HuggingFace...")
        remote_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=HF_FILENAME, 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        df = pd.read_parquet(remote_path)
        
        # Cache locally
        df.to_parquet(LOCAL_PARQUET_PATH)
        print(f"💾 Cached parquet locally to {LOCAL_PARQUET_PATH}")
    
    return df

def fuzzy_match_location(user_text: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuzzy match user's location text against file_district (roman) and file_dong (Hangul).
    Returns filtered dataframe.
    """
    if not user_text or len(df) == 0:
        return df
    
    user_lower = user_text.lower().strip()
    
    # Remove common suffixes for better matching
    user_clean = user_lower.replace('gu', '').replace('dong', '').replace('-', '').strip()
    
    print(f"🔍 Fuzzy matching location: '{user_text}'")
    
    matched_rows = []
    
    try:
        for idx, row in df.iterrows():
            score = 0
            
            # Match against file_district (roman characters)
            if 'file_district' in row.index and pd.notna(row['file_district']):
                district = str(row['file_district']).lower().replace('-', '')
                district_clean = district.replace('gu', '').strip()
                
                # Exact match
                if user_clean in district_clean or district_clean in user_clean:
                    score += 10
                # Partial match
                elif any(word in district_clean for word in user_clean.split() if len(word) > 2):
                    score += 5
            
            # Match against file_dong (Hangul)
            if 'file_dong' in row.index and pd.notna(row['file_dong']):
                dong = str(row['file_dong'])
                
                # Direct substring (works for Hangul too)
                if user_text in dong or dong in user_text:
                    score += 10
                # Check if user input is Hangul
                elif any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in user_text):
                    # User typed Hangul, try matching
                    if any(word in dong for word in user_text.split() if len(word) > 1):
                        score += 5
            
            # Match against address
            if 'address' in row.index and pd.notna(row['address']):
                address = str(row['address']).lower()
                if user_clean in address:
                    score += 3
            
            if score > 0:
                matched_rows.append((idx, score))
        
        if matched_rows:
            # Sort by score and get indices
            matched_rows.sort(key=lambda x: x[1], reverse=True)
            matched_indices = [idx for idx, score in matched_rows]
            
            result_df = df.loc[matched_indices]
            print(f"   ✅ Found {len(result_df)} facilities matching location")
            return result_df
        else:
            print(f"   ⚠️ No location matches found, using all facilities")
            return df
            
    except Exception as e:
        print(f"   ❌ Error in fuzzy matching: {e}, returning all facilities")
        return df

def build_context_for_llm(df_subset: pd.DataFrame, n_results: int = 10) -> str:
    """
    Build rich context from filtered parquet data for LLM.
    Uses both English (Summaries) and Korean (Summaries_Korean).
    """
    context_parts = []
    
    for idx, row in df_subset.head(n_results).iterrows():
        facility_info = []
        
        # Basic info
        facility_info.append(f"Name: {row['name']}")
        facility_info.append(f"Category: {row['category']}")
        
        # Location info
        if 'file_district' in row.index and pd.notna(row['file_district']):
            facility_info.append(f"District: {row['file_district']}")
        if 'file_dong' in row.index and pd.notna(row['file_dong']):
            facility_info.append(f"Neighborhood: {row['file_dong']}")
        
        # Distance (if calculated)
        if 'distance_km' in row.index and pd.notna(row['distance_km']) and row['distance_km'] > 0:
            facility_info.append(f"Distance: {row['distance_km']:.1f}km away")
        
        # English summary - avoid pd.isna on lists
        if 'Summaries' in row.index:
            summaries = row['Summaries']
            if isinstance(summaries, list) and len(summaries) > 0:
                facility_info.append(f"Summary (EN): {summaries[0]}")
        
        # Korean summary - avoid pd.isna on lists
        if 'Summaries_Korean' in row.index:
            summaries_kr = row['Summaries_Korean']
            if isinstance(summaries_kr, list) and len(summaries_kr) > 0:
                facility_info.append(f"Summary (KR): {summaries_kr[0]}")
        
        # Highlights - avoid pd.isna on lists
        if 'Key_Highlights' in row.index:
            highlights_data = row['Key_Highlights']
            if isinstance(highlights_data, list) and len(highlights_data) > 0:
                # Extract topic_en from each highlight dict
                topics = []
                for h in highlights_data[:5]:  # Top 5 highlights
                    if isinstance(h, dict) and 'topic_en' in h:
                        topics.append(h['topic_en'])
                if topics:
                    facility_info.append(f"Highlights: {', '.join(topics)}")
        
        # English capability
        if 'has_english' in row.index and row['has_english']:
            facility_info.append(f"English Speaking: Yes")
        
        # Medical info - avoid pd.isna on dicts
        if 'medical_info_parsed' in row.index:
            medical_data = row['medical_info_parsed']
            if isinstance(medical_data, dict) and len(medical_data) > 0:
                medical_keys = list(medical_data.keys())[:3]  # First 3 keys
                if medical_keys:
                    facility_info.append(f"Medical Services: {', '.join(medical_keys)}")
        
        context_parts.append("\n".join(facility_info))
    
    return "\n\n---\n\n".join(context_parts)

# --- 3. LIFESPAN (STARTUP) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, df_facilities, df_filtered
    print("🚀 Booting SeoulMedBot Backend...")

    # A. DOWNLOAD & CACHE PARQUET
    try:
        df_facilities = download_and_cache_parquet()
        print(f"✅ Loaded {len(df_facilities)} facilities from parquet")
        
        # Show data structure
        print(f"📊 Columns: {df_facilities.columns.tolist()[:10]}...")
        
        # CRITICAL FILTER: Only use facilities with Summaries
        df_filtered = df_facilities[df_facilities['Summaries'].notna()].copy()
        print(f"✅ Filtered to {len(df_filtered)} facilities with summaries")
        
        # Show location data samples
        if 'file_district' in df_filtered.columns:
            unique_districts = df_filtered['file_district'].dropna().unique()
            print(f"📍 Districts ({len(unique_districts)}): {sorted(unique_districts)[:5]}...")
        if 'file_dong' in df_filtered.columns:
            unique_dongs = df_filtered['file_dong'].dropna().unique()
            print(f"📍 Dongs ({len(unique_dongs)}): {list(unique_dongs)[:5]}...")
        
        # Check for GPS coordinates
        has_lat_lon = 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns
        if has_lat_lon:
            coords_count = df_filtered[df_filtered['latitude'].notna() & df_filtered['longitude'].notna()].shape[0]
            print(f"🌐 GPS coordinates available for {coords_count}/{len(df_filtered)} facilities")
        else:
            print(f"⚠️ No GPS coordinates in dataset - will use text-based location matching only")
        df_filtered['place_id'] = df_filtered['place_id'].fillna('').astype(str)        
    except Exception as e:
        print(f"❌ PARQUET LOAD ERROR: {e}")
        df_facilities = pd.DataFrame()
        df_filtered = pd.DataFrame()

    # B. SETUP RAG (using filtered subset)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    try:
        vector_db = client.get_collection("seoul_med_v3", embedding_function=openai_ef)
        print("✅ Embeddings collection found.")
    except:
        print("⚠️ Creating new embeddings from filtered dataset...")
        vector_db = client.create_collection("seoul_med_v3", embedding_function=openai_ef)
        
        ids, docs, metas = [], [], []
        for _, row in df_filtered.iterrows():
            # Build semantic blob from English + Korean
            summary_en = row['Summaries'][0] if isinstance(row['Summaries'], list) and row['Summaries'] else ""
            summary_kr = row['Summaries_Korean'][0] if isinstance(row['Summaries_Korean'], list) and row['Summaries_Korean'] else ""
            highlights = ", ".join([h.get('topic', '') for h in row['Key_Highlights']]) if isinstance(row['Key_Highlights'], list) else ""
            
            # Combine English + Korean for better semantic search
            text_blob = f"{row['name']} ({row['category']}). {summary_en} {summary_kr} {highlights}"
            
            ids.append(str(row['place_id']))
            docs.append(text_blob)
            metas.append({
                "category": row['category'],
                "district": str(row.get('file_district', '')),
                "has_english": bool(row.get('has_english', False))
            })

        # Batch insert
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            vector_db.add(ids=ids[i:end], documents=docs[i:end], metadatas=metas[i:end])
            
        print("✅ Embeddings indexed.")

    yield
    print("🛑 Shutting down.")

app = FastAPI(lifespan=lifespan)
client = Groq(api_key=GROQ_API_KEY)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 4. MODELS ---
class State(BaseModel):
    specialty: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None   # Optional - for distance-based search
    longitude: Optional[float] = None  # Optional - for distance-based search
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
1. `specialty`: Medical category (e.g., "Dentist", "Internal Medicine").
2. `location`: User's text location (e.g., "Gangnam", "Geumcheon gu").
3. `latitude`: Optional GPS latitude (float).
4. `longitude`: Optional GPS longitude (float).
5. `asked_for_location`: Boolean.
6. `ready_to_search`: Boolean.
7. `language_pref`: CRITICAL - Detect user's language!

**Logic Rules:**
- If specialty missing -> ask.
- If location/latitude/longitude missing AND `asked_for_location` is false -> ask for location.
- If location provided OR latitude/longitude provided -> set `ready_to_search`=true.

**LANGUAGE DETECTION (CRITICAL):**
- If user writes in English (ANY English words) -> set `language_pref`="English Preferred"
- If user writes in Korean (한글) -> set `language_pref`="Korean"
- When responding in `response_text`, use the SAME language as the user's message
- Examples:
  - "Find a dentist" → language_pref="English Preferred", response in English
  - "치과 찾아줘" → language_pref="Korean", response in Korean
  - "Gangnam dentist" → language_pref="English Preferred", response in English

**Notes:**
- latitude/longitude are OPTIONAL - most queries will only have text location
- ALWAYS match the user's language in your response_text

**Format:** { "state": {...}, "response_text": "..." }
"""

GENERATION_PROMPT = """
You are a helpful Medical Concierge for Seoul.

**User Query:** {user_query}
**Location Context:** {location_context}
**Language Preference:** {language}

**Available Facilities (ranked by relevance):**
{facilities_context}

**Your Task:**
1. Recommend the TOP 3 most relevant facilities from the list above
2. The facilities are already ranked by semantic relevance to the user's query
3. Use both English and Korean information to make your recommendation
4. Explain WHY each facility is a good match (use Summary and Highlights)
5. DO NOT include addresses, phone numbers, or hours
6. Ask if they want detailed amenity information

**Tone:** Warm, professional, helpful
**Language:** Respond in {language}
"""

# --- 6. CHAT ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    
    # --- STEP 1: EXTRACT INTENT (LLM) ---
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
            response_format={"type": "json_object"}
        )
        ai_data = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"response": "System error.", "state": req.current_state}

    new_state = ai_data.get("state", {})
    response_text = ai_data.get("response_text", "")
    
    # FALLBACK: Ensure language_pref is set
    if not new_state.get("language_pref") or new_state.get("language_pref") == "Korean is fine":
        # Detect from user message if LLM didn't set it properly
        user_message = req.message.lower()
        # Check if message contains English letters (basic heuristic)
        has_english = any(char.isalpha() and ord(char) < 128 for char in user_message)
        has_korean = any(0xAC00 <= ord(char) <= 0xD7A3 for char in req.message)
        
        if has_english and not has_korean:
            new_state["language_pref"] = "English Preferred"
            print("   🌐 Language detected: English (fallback)")
        elif has_korean:
            new_state["language_pref"] = "Korean"
            print("   🌐 Language detected: Korean (fallback)")
        else:
            new_state["language_pref"] = "Korean"  # Default to Korean
            print("   🌐 Language default: Korean")
    
    print("\n--- 🧠 STATE ---")
    print(json.dumps(new_state, indent=2))
    print("---------------\n")

    # --- STEP 2: FILTER PARQUET BY LOCATION ---
    location_context = ""
    
    if new_state.get("ready_to_search"):
        # Start with filtered dataset (Summaries not null)
        working_df = df_filtered.copy()
        print(f"📊 Starting with {len(working_df)} facilities (with summaries)")
        
        # LOCATION FILTERING - Two modes:
        # MODE 1: GPS coordinates available (latitude + longitude)
        # MODE 2: Text location only (fuzzy matching)
        
        has_coords = new_state.get("latitude") is not None and new_state.get("longitude") is not None
        
        if has_coords:
            # MODE 1: Distance-based filtering with GPS
            user_lat = new_state['latitude']
            user_lon = new_state['longitude']
            
            print(f"📍 MODE 1: GPS-based search at ({user_lat}, {user_lon})")
            
            # Calculate distances for all facilities
            working_df['distance_km'] = working_df.apply(
                lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']) 
                if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')) 
                else 999,  # Far distance for facilities without coords
                axis=1
            )
            
            # Filter by radius based on willingness to travel
            radius_map = {
                "Neighborhood": 2,
                "Nearby": 5,
                "City-wide": 10,
                "Don't Care": 25
            }
            max_distance = radius_map.get(new_state.get("willingness_to_travel"), 5)
            
            before_count = len(working_df)
            working_df = working_df[working_df['distance_km'] <= max_distance]
            working_df = working_df.sort_values('distance_km')  # Closest first
            
            print(f"   Filtered by {max_distance}km radius: {before_count} → {len(working_df)} facilities")
            location_context = f"within {max_distance}km"
            
        elif new_state.get("location"):
            # MODE 2: Text-based fuzzy matching
            print(f"📍 MODE 2: Text-based location search")
            working_df = fuzzy_match_location(new_state['location'], working_df)
            location_context = f"in {new_state['location']}"
            working_df['distance_km'] = 0  # Placeholder
            
        else:
            # MODE 3: No location filter
            print(f"📍 MODE 3: City-wide search (no location filter)")
            location_context = "across Seoul"
            working_df['distance_km'] = 0  # Placeholder
        
        # --- STEP 3: CATEGORY FILTER (Primary) ---
        # Try to match specialty to Korean category first
        korean_category = None
        if new_state.get("specialty"):
            specialty_lower = new_state['specialty'].lower().strip()
            if specialty_lower in SPECIALTY_CATEGORY_MAP:
                korean_category = SPECIALTY_CATEGORY_MAP[specialty_lower]
                print(f"🏥 Specialty '{new_state['specialty']}' → Korean category '{korean_category}'")
        
        if korean_category and 'category' in working_df.columns:
            # Filter by exact Korean category match
            before_count = len(working_df)
            working_df = working_df[working_df['category'].str.contains(korean_category, na=False, case=False)]
            print(f"   ✅ Category filter: {before_count} → {len(working_df)} facilities")
        
        # --- STEP 4: SEMANTIC RANKING (RAG) - NOT filtering! ---
        # Use RAG to RANK results by relevance, not to filter them out
        query_text = f"{new_state.get('specialty', '')} {' '.join(new_state.get('keywords', []))}"
        
        if len(query_text.strip()) > 2 and len(working_df) > 0:
            try:
                print(f"🔍 RAG semantic ranking for: '{query_text}'")
                
                # Get available place_ids from working_df
                available_ids = working_df['place_id'].tolist()
                
                # Query RAG with more results than we have
                n_rag_results = min(200, len(vector_db.get()['ids']))
                rag_results = vector_db.query(query_texts=[query_text], n_results=n_rag_results)
                
                if rag_results and 'ids' in rag_results and len(rag_results['ids']) > 0:
                    # Create ranking: place_id → rank (lower is better)
                    rag_ranking = {place_id: idx for idx, place_id in enumerate(rag_results['ids'][0])}
                    
                    # Add relevance_rank column (facilities not in RAG get high rank = low priority)
                    working_df['relevance_rank'] = working_df['place_id'].apply(
                        lambda pid: rag_ranking.get(pid, 9999)
                    )
                    
                    # Sort by relevance (lowest rank = most relevant)
                    working_df = working_df.sort_values('relevance_rank')
                    
                    # Count how many were actually ranked
                    ranked_count = (working_df['relevance_rank'] < 9999).sum()
                    print(f"   ✅ RAG ranked {ranked_count}/{len(working_df)} facilities by relevance")
                else:
                    print(f"   ⚠️ RAG returned no results, keeping original order")
                    working_df['relevance_rank'] = 9999
                    
            except Exception as e:
                print(f"   ⚠️ RAG error: {e}, keeping original order")
                working_df['relevance_rank'] = 9999
        else:
            print(f"   ⏭️ Skipping RAG ranking (query too short or no results)")
            working_df['relevance_rank'] = 9999
        
        # --- STEP 5: ENGLISH FILTER ---
        language_pref = new_state.get("language_pref", "Korean")
        if "English" in language_pref:
            if 'has_english' in working_df.columns:
                before = len(working_df)
                working_df = working_df[working_df['has_english'] == True]
                print(f"🌐 English filter: {before} → {len(working_df)} facilities")
            else:
                print(f"   ⚠️ 'has_english' column not found, skipping English filter")
        
        # --- STEP 5.5: FINAL SORTING ---
        # Combine distance and relevance for optimal ordering
        if len(working_df) > 0:
            if has_coords and 'distance_km' in working_df.columns:
                # For GPS mode: Sort by distance primarily, relevance secondarily
                working_df = working_df.sort_values(['distance_km', 'relevance_rank'])
                print(f"   📊 Sorted by distance + relevance")
            elif 'relevance_rank' in working_df.columns:
                # For text mode: Sort by relevance
                working_df = working_df.sort_values('relevance_rank')
                print(f"   📊 Sorted by relevance")
        
        # --- STEP 6: BUILD CONTEXT FROM PARQUET ---
        if len(working_df) > 0:
            print(f"✅ Building context from {len(working_df)} filtered facilities")
            facilities_context = build_context_for_llm(working_df, n_results=10)
            
            # --- STEP 7: GENERATE RECOMMENDATION (LLM) ---
            # Determine response language based on user's language preference
            language_pref = new_state.get("language_pref", "Korean")
            if "English" in language_pref:
                language = "English"
            else:
                language = "Korean"
            
            print(f"   🗣️ Generating response in: {language}")
            
            gen_messages = [{
                "role": "system",
                "content": GENERATION_PROMPT.format(
                    user_query=req.message,
                    location_context=location_context,
                    language=language,
                    facilities_context=facilities_context
                )
            }]
            
            gen_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=gen_messages,
                temperature=1.0,
                max_completion_tokens=1024
            )
            response_text = gen_completion.choices[0].message.content
            
            # Prepare results for frontend
            results = []
            for _, row in working_df.head(3).iterrows():
                result = {
                    "place_id": str(row['place_id']),
                    "name": row['name'],
                    "category": row['category'],
                }
                
                # Add optional fields - check type instead of pd.notna for complex types
                simple_fields = ['address', 'phone', 'business_hours', 'latitude', 'longitude', 'english_confidence_score']
                for field in simple_fields:
                    if field in row.index and pd.notna(row[field]):
                        result[field] = row[field]
                
                # String fields with different names
                if 'file_district' in row.index and pd.notna(row['file_district']):
                    result['district'] = row['file_district']
                if 'file_dong' in row.index and pd.notna(row['file_dong']):
                    result['dong'] = row['file_dong']
                
                # Website (try website, fallback to url)
                if 'website' in row.index and pd.notna(row['website']):
                    result['website'] = row['website']
                elif 'url' in row.index and pd.notna(row['url']):
                    result['website'] = row['url']
                
                # Complex fields - check type instead of pd.notna
                if 'Summaries' in row.index and isinstance(row['Summaries'], list):
                    result['Summaries'] = row['Summaries']
                
                if 'Summaries_Korean' in row.index and isinstance(row['Summaries_Korean'], list):
                    result['Summaries_Korean'] = row['Summaries_Korean']
                
                if 'Key_Highlights' in row.index and isinstance(row['Key_Highlights'], list):
                    result['Key_Highlights'] = row['Key_Highlights']
                
                if 'amenities' in row.index and isinstance(row['amenities'], (list, dict)):
                    result['amenities'] = row['amenities']
                
                if 'medical_info_parsed' in row.index and isinstance(row['medical_info_parsed'], dict):
                    result['medical_info_parsed'] = row['medical_info_parsed']
                
                # Boolean/numeric fields
                result['has_english'] = bool(row['has_english']) if 'has_english' in row.index else False
                result['distance_km'] = float(row['distance_km']) if 'distance_km' in row.index and pd.notna(row['distance_km']) else 0
                result['relevance_rank'] = int(row['relevance_rank']) if 'relevance_rank' in row.index and pd.notna(row['relevance_rank']) else 9999
                
                results.append(result)
        else:
            # No results found - respond in user's language
            language_pref = new_state.get("language_pref", "Korean")
            if "English" in language_pref:
                response_text = "I couldn't find any facilities matching your criteria. Would you like to try a different specialty or area?"
            else:
                response_text = "검색 조건에 맞는 시설을 찾을 수 없습니다. 다른 전문 분야나 지역을 시도해 보시겠어요?"
            results = []
    else:
        results = []

    return {
        "response": response_text,
        "state": new_state,
        "results": results
    }
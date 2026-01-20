import os
import json
import pandas as pd
import numpy as np
import requests
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
from groq import Groq
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from specialties import SPECIALTY_CATEGORY_MAP
from distance import haversine, fuzzy_match_location
from models import ChatRequest, State
from utils import safe_convert_to_python
from prompt import ROUTER_PROMPT, EXTRACTION_PROMPT_V2, GENERATION_PROMPT
from deterministic import get_greeting_message,get_reset_confirmation,generate_change_acknowledgment, ask_for_missing_info,ask_for_specialty_clarification,generate_chit_chat_response, generate_recovery_prompt,format_response 
import re


# --- 1. CONFIGURATION ---
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")  # Add to .env
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


# --- 2. KAKAO API FUNCTIONS ---

def kakao_geocode(address: str) -> Optional[Tuple[float, float]]:
    """
    Convert address to coordinates using Kakao API.
    Returns (latitude, longitude) or None if not found.
    """
    if not KAKAO_REST_API_KEY:
        print("⚠️ KAKAO_REST_API_KEY not set - skipping geocoding")
        return None
    
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": address}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('documents') and len(data['documents']) > 0:
            doc = data['documents'][0]
            
            # Try road address first, then jibun address
            if doc.get('road_address'):
                lat = float(doc['road_address']['y'])
                lon = float(doc['road_address']['x'])
            elif doc.get('address'):
                lat = float(doc['address']['y'])
                lon = float(doc['address']['x'])
            else:
                return None
            
            print(f"   🗺️ Kakao geocoded '{address}' → ({lat}, {lon})")
            return (lat, lon)
        else:
            print(f"   ⚠️ Kakao geocoding: No results for '{address}'")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Kakao geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        print(f"   ❌ Kakao geocoding parse error: {e}")
        return None


def kakao_reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Convert coordinates to address using Kakao API.
    Returns address string or None if not found.
    """
    if not KAKAO_REST_API_KEY:
        print("⚠️ KAKAO_REST_API_KEY not set - skipping reverse geocoding")
        return None
    
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"x": lon, "y": lat}  # Note: Kakao uses x=lon, y=lat
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('documents') and len(data['documents']) > 0:
            doc = data['documents'][0]
            
            # Try road address first, then jibun address
            if doc.get('road_address'):
                address = doc['road_address']['address_name']
            elif doc.get('address'):
                address = doc['address']['address_name']
            else:
                return None
            
            print(f"   🗺️ Kakao reverse geocoded ({lat}, {lon}) → '{address}'")
            return address
        else:
            print(f"   ⚠️ Kakao reverse geocoding: No results for ({lat}, {lon})")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Kakao reverse geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        print(f"   ❌ Kakao reverse geocoding parse error: {e}")
        return None


def geocode_location(location_text: str) -> Optional[Tuple[float, float]]:
    """
    Smart geocoding that handles district names and full addresses.
    Returns (latitude, longitude) or None.
    """
    if not location_text:
        return None
    
    # First try direct geocoding
    coords = kakao_geocode(location_text)
    if coords:
        return coords
    
    # If that fails, try adding "서울" prefix for district names
    if not location_text.startswith("서울"):
        coords = kakao_geocode(f"서울 {location_text}")
        if coords:
            return coords
    
    # Try adding "구" suffix for districts
    if "구" not in location_text and "동" not in location_text:
        coords = kakao_geocode(f"서울 {location_text}구")
        if coords:
            return coords
    
    print(f"   ⚠️ Could not geocode: {location_text}")
    return None


# --- 3. DATA LOADING ---

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


def build_context_for_llm(df_subset: pd.DataFrame, n_results: int = 10) -> str:
    """Build rich context from filtered parquet data for LLM."""
    context_parts = []
    
    for idx, row in df_subset.head(n_results).iterrows():
        facility_info = []
        
        facility_info.append(f"Name: {row['name']}")
        facility_info.append(f"Category: {row['category']}")
        
        if 'file_district' in row.index and pd.notna(row['file_district']):
            facility_info.append(f"District: {row['file_district']}")
        if 'file_dong' in row.index and pd.notna(row['file_dong']):
            facility_info.append(f"Neighborhood: {row['file_dong']}")
        
        if 'distance_km' in row.index and pd.notna(row['distance_km']) and row['distance_km'] > 0:
            facility_info.append(f"Distance: {row['distance_km']:.1f}km away")
        
        # Handle Summaries (ndarray)
        if 'Summaries' in row.index:
            summaries = row['Summaries']
            if isinstance(summaries, (list, np.ndarray)) and len(summaries) > 0:
                facility_info.append(f"Summary (EN): {summaries[0]}")
        
        # Handle Summaries_Korean (ndarray)
        if 'Summaries_Korean' in row.index:
            summaries_kr = row['Summaries_Korean']
            if isinstance(summaries_kr, (list, np.ndarray)) and len(summaries_kr) > 0:
                facility_info.append(f"Summary (KR): {summaries_kr[0]}")
        
        # Handle Key_Highlights (ndarray)
        if 'Key_Highlights' in row.index:
            highlights_data = row['Key_Highlights']
            if isinstance(highlights_data, (list, np.ndarray)) and len(highlights_data) > 0:
                topics = []
                for h in highlights_data[:5]:
                    if isinstance(h, dict) and 'topic_en' in h:
                        topics.append(h['topic_en'])
                if topics:
                    facility_info.append(f"Highlights: {', '.join(topics)}")
        
        if 'has_english' in row.index and row['has_english']:
            facility_info.append(f"English Speaking: Yes")
        
        context_parts.append("\n".join(facility_info))
    
    return "\n\n---\n\n".join(context_parts)


# --- 4. LIFESPAN (STARTUP) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, df_facilities, df_filtered
    print("🚀 Booting SeoulMedBot Backend (Router-Controller + Kakao Geocoding)...")

    try:
        df_facilities = download_and_cache_parquet()
        print(f"✅ Loaded {len(df_facilities)} facilities from parquet")
        
        # FILTER: Only use facilities with Summaries
        df_filtered = df_facilities[df_facilities['Summaries'].notna()].copy()
        print(f"✅ Filtered to {len(df_filtered)} facilities with summaries")
        
        # Show location data
        if 'file_district' in df_filtered.columns:
            unique_districts = df_filtered['file_district'].dropna().unique()
            print(f"📍 Districts ({len(unique_districts)}): {sorted(unique_districts)[:5]}...")
        
        # Check for GPS coordinates in parquet
        if 'lat' in df_filtered.columns and 'lon' in df_filtered.columns:
            coords_count = df_filtered[
                (df_filtered['lat'].notna()) & 
                (df_filtered['lon'].notna())
            ].shape[0]
            print(f"🌐 GPS coordinates available for {coords_count}/{len(df_filtered)} facilities ({coords_count/len(df_filtered)*100:.1f}%)")
        else:
            print(f"⚠️ No lat/lon columns - will use text-based location matching only")
        
        # Check Kakao API
        if KAKAO_REST_API_KEY:
            print(f"🗺️ Kakao geocoding API: ENABLED")
        else:
            print(f"⚠️ Kakao geocoding API: DISABLED (add KAKAO_REST_API_KEY to .env)")
        
        df_filtered['place_id'] = df_filtered['place_id'].fillna('').astype(str)        
    except Exception as e:
        print(f"❌ PARQUET LOAD ERROR: {e}")
        import traceback
        traceback.print_exc()
        df_facilities = pd.DataFrame()
        df_filtered = pd.DataFrame()

    # Setup RAG
    client_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    try:
        vector_db = client_chroma.get_collection("seoul_med_v3", embedding_function=openai_ef)
        print("✅ Embeddings collection found.")
    except:
        print("⚠️ Creating new embeddings from filtered dataset...")
        vector_db = client_chroma.create_collection("seoul_med_v3", embedding_function=openai_ef)
        
        ids, docs, metas = [], [], []
        for _, row in df_filtered.iterrows():
            summary_en = ""
            summary_kr = ""
            highlights = ""
            
            if isinstance(row['Summaries'], (list, np.ndarray)) and len(row['Summaries']) > 0:
                summary_en = str(row['Summaries'][0])
            if isinstance(row['Summaries_Korean'], (list, np.ndarray)) and len(row['Summaries_Korean']) > 0:
                summary_kr = str(row['Summaries_Korean'][0])
            if isinstance(row['Key_Highlights'], (list, np.ndarray)) and len(row['Key_Highlights']) > 0:
                highlights = ", ".join([h.get('topic', '') for h in row['Key_Highlights'] if isinstance(h, dict)])
            
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
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def detect_language(message: str) -> str:
    """Simple language detection."""
    has_korean = any(0xAC00 <= ord(char) <= 0xD7A3 for char in message)
    has_english = any(char.isalpha() and ord(char) < 128 for char in message)
    
    if has_korean:
        return "Korean"
    elif has_english:
        return "English Preferred"
    else:
        return "Korean"  # Default


def cleanse_state_for_change(current_state: State, user_message: str) -> State:
    """
    Programmatically cleanse state based on what the user is changing.
    This is executed BEFORE extraction to prevent bias.
    """
    message_lower = user_message.lower()
    
    # Detect what's being changed
    location_keywords = ["in ", "near ", "at ", "구", "동", "역", "gangnam", "songpa", "mapo", "jung", "강남", "송파"]
    specialty_keywords = ["dentist", "doctor", "dermatologist", "pediatrician", "치과", "피부과", "병원", "의원", "내과"]
    
    has_location_change = any(kw in message_lower for kw in location_keywords)
    has_specialty_change = any(kw in message_lower for kw in specialty_keywords)
    
    new_state = current_state.model_copy()
    
    if has_specialty_change:
        print("   🧹 Cleansing: specialty")
        new_state.specialty = None
        new_state.specialty_confidence = 0.0
    
    if has_location_change:
        print("   🧹 Cleansing: location")
        new_state.location = None
        new_state.latitude = None
        new_state.longitude = None
    
    # Always reset search flags when changing
    new_state.ready_to_search = False
    new_state.search_executed = False
    new_state.conversation_phase = "gathering"
    
    return new_state


def extract_entities(user_message: str) -> Dict[str, Any]:
    """
    Call the EXTRACTION_PROMPT_V2 (pure extractor).
    This is a leaf node that only extracts, doesn't reason about flow.
    """
    extraction_messages = [{
        "role": "system",
        "content": EXTRACTION_PROMPT_V2.format(user_message=user_message)
    }]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=extraction_messages,
            temperature=0.0,
            max_completion_tokens=512,
            response_format={"type": "json_object"}
        )
        extracted = json.loads(completion.choices[0].message.content)
        
        # Log extraction
        gps_info = ""
        if extracted.get('latitude') and extracted.get('longitude'):
            gps_info = f", GPS=({extracted['latitude']}, {extracted['longitude']})"
        
        print(f"   📦 Extracted: specialty={extracted.get('specialty')}, location={extracted.get('location')}{gps_info}, confidence={extracted.get('specialty_confidence', 0):.2f}")
        
        # GEOCODE if we have location text but no GPS
        if extracted.get('location') and not extracted.get('latitude'):
            coords = geocode_location(extracted['location'])
            if coords:
                extracted['latitude'] = coords[0]
                extracted['longitude'] = coords[1]
                print(f"   🗺️ Geocoded location to ({coords[0]}, {coords[1]})")
        
        return extracted
        
    except Exception as e:
        print(f"   ❌ Extraction Error: {e}")
        return {}


def merge_extraction_into_state(state: State, extracted: Dict[str, Any]) -> State:
    """Merge extracted entities into state (only if they exist)."""
    
    if extracted.get('specialty'):
        state.specialty = extracted['specialty']
        state.specialty_confidence = extracted.get('specialty_confidence', 0.7)
    
    if extracted.get('location'):
        state.location = extracted['location']
    
    if extracted.get('latitude') is not None:
        state.latitude = extracted['latitude']
    
    if extracted.get('longitude') is not None:
        state.longitude = extracted['longitude']
    
    if extracted.get('language_pref'):
        state.language_pref = extracted['language_pref']
    
    return state


def execute_search(state: State, user_message: str) -> Tuple[str, List[Dict]]:
    """
    Execute the actual search logic with GPS and geocoding support.
    Returns (response_text, results_list).
    """
    working_df = df_filtered.copy()
    print(f"📊 Starting with {len(working_df)} facilities")
    
    location_context = ""
    user_lat = state.latitude
    user_lon = state.longitude
    
    # ===== LOCATION FILTERING WITH GPS =====
    if user_lat and user_lon:
        # GPS-based search with distance calculation
        print(f"📍 GPS-based search: ({user_lat}, {user_lon})")
        
        # Get address from coordinates for context
        user_address = kakao_reverse_geocode(user_lat, user_lon)
        if user_address:
            location_context = f"near {user_address}"
            print(f"   📍 Reverse geocoded to: {user_address}")
        else:
            location_context = f"near your location"
        
        # Calculate distances for all facilities with coordinates
        if 'lat' in working_df.columns and 'lon' in working_df.columns:
            def calc_distance(row):
                if pd.notna(row['lat']) and pd.notna(row['lon']):
                    return haversine(user_lat, user_lon, row['lat'], row['lon'])
                else:
                    return 999.0
            
            working_df['distance_km'] = working_df.apply(calc_distance, axis=1)
            
            # Filter to reasonable distance (within 10km by default)
            max_distance = 10.0
            before_count = len(working_df)
            working_df = working_df[working_df['distance_km'] < max_distance]
            print(f"   ✅ Distance filter (<{max_distance}km): {before_count} → {len(working_df)}")
            
            if len(working_df) == 0:
                # Expand search radius if nothing found
                print(f"   ⚠️ No results within {max_distance}km, expanding to 20km")
                working_df = df_filtered.copy()
                working_df['distance_km'] = working_df.apply(calc_distance, axis=1)
                working_df = working_df[working_df['distance_km'] < 20.0]
                location_context = f"within 20km of your location"
            
            # Sort by distance initially
            working_df = working_df.sort_values('distance_km')
        else:
            working_df['distance_km'] = 0
            location_context = "in Seoul"
    
    elif state.location:
        # Text-based location search with fuzzy matching
        print(f"📍 Text-based location: {state.location}")
        working_df = fuzzy_match_location(state.location, working_df)
        location_context = f"in {state.location}"
        working_df['distance_km'] = 0
    
    else:
        # City-wide search
        print(f"📍 City-wide search (no location specified)")
        location_context = "across Seoul"
        working_df['distance_km'] = 0
    
    # ===== CATEGORY FILTER =====
    korean_category = None
    if state.specialty:
        specialty_lower = state.specialty.lower().strip()
        if specialty_lower in SPECIALTY_CATEGORY_MAP:
            korean_category = SPECIALTY_CATEGORY_MAP[specialty_lower]
            print(f"🏥 Specialty mapping: '{state.specialty}' → '{korean_category}'")
    
    if korean_category and 'category' in working_df.columns:
        before_count = len(working_df)
        working_df = working_df[working_df['category'].str.contains(korean_category, na=False, case=False)]
        print(f"   ✅ Category filter: {before_count} → {len(working_df)}")
    
    # ===== SEMANTIC RANKING (RAG) =====
    query_text = state.specialty or ""
    
    if len(query_text.strip()) > 2 and len(working_df) > 0:
        try:
            print(f"🔍 RAG ranking for: '{query_text}'")
            n_rag_results = min(200, len(vector_db.get()['ids']))
            rag_results = vector_db.query(query_texts=[query_text], n_results=n_rag_results)
            
            if rag_results and 'ids' in rag_results and len(rag_results['ids']) > 0:
                rag_ranking = {place_id: idx for idx, place_id in enumerate(rag_results['ids'][0])}
                working_df['relevance_rank'] = working_df['place_id'].apply(lambda pid: rag_ranking.get(pid, 9999))
                
                # Combined ranking: distance + relevance
                if 'distance_km' in working_df.columns and working_df['distance_km'].max() > 0:
                    # Normalize both scores to 0-1 range
                    max_rank = working_df['relevance_rank'].max()
                    max_dist = working_df['distance_km'].max()
                    
                    if max_rank > 0:
                        working_df['relevance_score'] = 1 - (working_df['relevance_rank'] / max_rank)
                    else:
                        working_df['relevance_score'] = 1.0
                    
                    if max_dist > 0:
                        working_df['distance_score'] = 1 - (working_df['distance_km'] / max_dist)
                    else:
                        working_df['distance_score'] = 1.0
                    
                    # Weighted combination (60% relevance, 40% distance)
                    working_df['combined_score'] = 0.6 * working_df['relevance_score'] + 0.4 * working_df['distance_score']
                    working_df = working_df.sort_values('combined_score', ascending=False)
                    print(f"   ✅ Combined ranking (60% relevance + 40% distance)")
                else:
                    working_df = working_df.sort_values('relevance_rank')
                    print(f"   ✅ RAG ranked {(working_df['relevance_rank'] < 9999).sum()}/{len(working_df)}")
            else:
                working_df['relevance_rank'] = 9999
        except Exception as e:
            print(f"   ⚠️ RAG error: {e}")
            working_df['relevance_rank'] = 9999
    else:
        working_df['relevance_rank'] = 9999
    
    # ===== ENGLISH FILTER =====
    language_pref = state.language_pref
    if "English" in language_pref and 'has_english' in working_df.columns:
        before = len(working_df)
        working_df = working_df[working_df['has_english'] == True]
        print(f"🌐 English filter: {before} → {len(working_df)}")
    
    # ===== GENERATE RESPONSE =====
    results = []
    
    if len(working_df) > 0:
        print(f"✅ Building context from top {min(10, len(working_df))} facilities")
        facilities_context = build_context_for_llm(working_df, n_results=10)
        
        language = "English" if "English" in language_pref else "Korean"
        
        gen_messages = [{
            "role": "system",
            "content": GENERATION_PROMPT.format(
                user_query=user_message,
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
        
        # ===== PREPARE RESULTS =====
        for _, row in working_df.head(3).iterrows():
            result = {
                "place_id": safe_convert_to_python(row['place_id']),
                "name": safe_convert_to_python(row['name']),
                "category": safe_convert_to_python(row['category']),
            }
            
            # Simple fields
            simple_fields = ['address', 'phone', 'business_hours', 'english_confidence_score']
            for field in simple_fields:
                if field in row.index and pd.notna(row[field]):
                    result[field] = safe_convert_to_python(row[field])
            
            # Location fields
            if 'file_district' in row.index and pd.notna(row['file_district']):
                result['district'] = safe_convert_to_python(row['file_district'])
            if 'file_dong' in row.index and pd.notna(row['file_dong']):
                result['dong'] = safe_convert_to_python(row['file_dong'])
            
            # GPS coordinates from parquet
            if 'lat' in row.index and pd.notna(row['lat']):
                result['lat'] = safe_convert_to_python(row['lat'])
            if 'lon' in row.index and pd.notna(row['lon']):
                result['lon'] = safe_convert_to_python(row['lon'])
            
            # Website
            if 'website' in row.index and pd.notna(row['website']):
                result['website'] = safe_convert_to_python(row['website'])
            elif 'url' in row.index and pd.notna(row['url']):
                result['website'] = safe_convert_to_python(row['url'])
            
            # Complex fields
            if 'Summaries' in row.index and isinstance(row['Summaries'], (list, np.ndarray)):
                result['Summaries'] = safe_convert_to_python(row['Summaries'])
            
            if 'Summaries_Korean' in row.index and isinstance(row['Summaries_Korean'], (list, np.ndarray)):
                result['Summaries_Korean'] = safe_convert_to_python(row['Summaries_Korean'])
            
            if 'Key_Highlights' in row.index and isinstance(row['Key_Highlights'], (list, np.ndarray)):
                result['Key_Highlights'] = safe_convert_to_python(row['Key_Highlights'])
            
            if 'amenities' in row.index:
                result['amenities'] = safe_convert_to_python(row['amenities'])
            
            if 'medical_info_parsed' in row.index and isinstance(row['medical_info_parsed'], dict):
                result['medical_info_parsed'] = safe_convert_to_python(row['medical_info_parsed'])
            
            # Booleans and numbers
            result['has_english'] = safe_convert_to_python(row.get('has_english', False))
            result['distance_km'] = safe_convert_to_python(row.get('distance_km', 0))
            result['relevance_rank'] = safe_convert_to_python(row.get('relevance_rank', 9999))
            
            results.append(result)
    else:
        # No results found
        if "English" in language_pref:
            response_text = "I couldn't find any facilities matching your criteria. Would you like to try a different specialty or area?"
        else:
            response_text = "검색 조건에 맞는 시설을 찾을 수 없습니다. 다른 전문 분야나 지역을 시도해 보시겠어요?"
    
    return response_text, results


# ==========================================
# MAIN CHAT ENDPOINT (Router-Controller)
# ==========================================

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Router-Controller Architecture:
    Step 1: Route (classify intent)
    Step 2: Branch (execute appropriate logic)
    """
    
    current_turn = req.current_state.turn_count + 1
    
    # ==========================================
    # STEP 1: ROUTING (Root Node)
    # ==========================================
    
    router_messages = [{
        "role": "system",
        "content": ROUTER_PROMPT.format(
            specialty=req.current_state.specialty or "None",
            location=req.current_state.location or "None",
            ready_to_search=req.current_state.ready_to_search,
            turn_count=current_turn,
            search_executed=req.current_state.search_executed,
            user_message=req.message
        )
    }]
    
    try:
        router_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=router_messages,
            temperature=0.0,
            max_completion_tokens=256,
            response_format={"type": "json_object"}
        )
        route = json.loads(router_completion.choices[0].message.content)
        intent = route.get("intent")
        
        print(f"\n{'='*80}")
        print(f"🧭 ROUTER DECISION: {intent}")
        print(f"   Confidence: {route.get('confidence', 0):.2f}")
        print(f"   Reasoning: {route.get('reasoning', '')}")
        print(f"   Turn: {current_turn}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Router Error: {e}")
        intent = "PROVIDE_INFO"  # Safe fallback
    
    # ==========================================
    # STEP 2: CONTROLLER (Tree Traversal)
    # ==========================================
    
    # === BRANCH 1: NEW_SEARCH (Complete Reset) ===
    if intent == "NEW_SEARCH":
        print("🔄 BRANCH: NEW_SEARCH - Complete state reset to 0")
        
        # Detect language from current message before reset
        detected_lang = detect_language(req.message)
        
        # Create brand new state (all fields reset to default/None)
        new_state = State()
        new_state.turn_count = 0  # Reset to 0
        new_state.language_pref = detected_lang
        
        # Check if this is an explicit reset/quit command
        reset_keywords = ["reset", "restart", "quit", "exit", "stop", "cancel", 
                         "새로 시작", "처음부터", "다시 시작", "그만", "종료"]
        
        is_explicit_reset = any(kw in req.message.lower() for kw in reset_keywords)
        
        if is_explicit_reset:
            response = get_reset_confirmation(detected_lang)
        else:
            response = get_greeting_message(detected_lang)
        
        print(f"   ✅ State reset complete - all fields returned to initial values")
        print(f"   📊 New state: specialty=None, location=None, lat=None, lon=None")
        
        return {
            "response": response,
            "state": new_state.model_dump(),
            "results": []
        }
    
    # === BRANCH 2: CHANGE_CRITERIA ===
    elif intent == "CHANGE_CRITERIA":
        print("🔀 BRANCH: CHANGE_CRITERIA - Cleansing state")
        
        # Hard state cleansing
        cleansed_state = cleanse_state_for_change(req.current_state, req.message)
        cleansed_state.turn_count = current_turn
        
        # Extract new information (will auto-geocode if needed)
        extracted = extract_entities(req.message)
        
        # Merge extracted data into cleansed state
        new_state = merge_extraction_into_state(cleansed_state, extracted)
        
        # Check if we have enough to search (either location OR GPS)
        has_location = bool(new_state.location or (new_state.latitude and new_state.longitude))
        
        if new_state.specialty and has_location:
            # Check specialty confidence
            if new_state.specialty_confidence >= 0.5:
                new_state.ready_to_search = True
                new_state.conversation_phase = "searching"
            else:
                # Low confidence - ask for clarification
                response = ask_for_specialty_clarification(new_state)
                return {
                    "response": response,
                    "state": new_state.model_dump(),
                    "results": []
                }
        else:
            new_state.conversation_phase = "gathering"
            response = ask_for_missing_info(new_state)
            return {
                "response": response,
                "state": new_state.model_dump(),
                "results": []
            }
        
        # Execute search if ready
        results = []
        if new_state.ready_to_search:
            response, results = execute_search(new_state, req.message)
            new_state.search_executed = True
         

        return {
            "response": response,
            "state": new_state.model_dump(),
            "results": results
        }
    
    # === BRANCH 3: PROVIDE_INFO ===
    elif intent == "PROVIDE_INFO":
        print("📝 BRANCH: PROVIDE_INFO - Extracting entities")
        
        # Extract entities from user message (will auto-geocode if needed)
        extracted = extract_entities(req.message)
        
        # Merge into current state (accumulate information)
        new_state = req.current_state.model_copy()
        new_state.turn_count = current_turn
        new_state = merge_extraction_into_state(new_state, extracted)
        
        # Check if we have enough to search (either location OR GPS)
        has_location = bool(new_state.location or (new_state.latitude and new_state.longitude))
        
        # Determine conversation phase
        if new_state.specialty and has_location:
            # Check specialty confidence
            if new_state.specialty_confidence >= 0.5:
                new_state.ready_to_search = True
                new_state.conversation_phase = "searching"
            else:
                # Low confidence - ask for clarification
                response = ask_for_specialty_clarification(new_state)
                response = format_response(response)
                return {
                    "response": response,
                    "state": new_state.model_dump(),
                    "results": []
                }
        else:
            # Still gathering
            new_state.conversation_phase = "gathering"
            response = ask_for_missing_info(new_state)
            
            return {
                "response": response,
                "state": new_state.model_dump(),
                "results": []
            }
        
        # Execute search if ready
        results = []
        if new_state.ready_to_search:
            response, results = execute_search(new_state, req.message)
            new_state.search_executed = True
         
        return {
            "response": response,
            "state": new_state.model_dump(),
            "results": results
        }
    
    # === BRANCH 4: CHIT_CHAT ===
    elif intent == "CHIT_CHAT":
        print("💬 BRANCH: CHIT_CHAT - Simple response")
        
        new_state = req.current_state.model_copy()
        new_state.turn_count = current_turn
        
        # Detect language if not set
        if not new_state.language_pref or new_state.language_pref == "Korean is fine":
            new_state.language_pref = detect_language(req.message)
        
        # Check if this is a closing statement
        is_closing = any(word in req.message.lower() for word in 
                        ["thanks", "thank you", "감사합니다", "고마워"])
        
        if is_closing:
            new_state.conversation_phase = "complete"
        
        response = generate_chit_chat_response(req.message, new_state)
        

        return {
            "response": response,
            "state": new_state.model_dump(),
            "results": []
        }
    
    # === BRANCH 5: HELP_RECOVERY (Stagnation Check) ===
    elif intent == "HELP_RECOVERY":
        print("🆘 BRANCH: HELP_RECOVERY - User is stuck")
        
        new_state = req.current_state.model_copy()
        new_state.turn_count = current_turn
        
        # Detect language if not set
        if not new_state.language_pref or new_state.language_pref == "Korean is fine":
            new_state.language_pref = detect_language(req.message)
        
        response = generate_recovery_prompt(new_state)
        

        return {
            "response": response,
            "state": new_state.model_dump(),
            "results": []
        }
    
    # Fallback
    else:
        print(f"⚠️ Unknown intent: {intent}")
        
        # Try to detect language
        lang = detect_language(req.message)
        
        if "English" in lang:
            response = "I'm not sure how to help. Could you rephrase your request?"
        else:
            response = "잘 이해하지 못했습니다. 다시 말씀해 주시겠어요?"
        
        return {
            "response": response,
            "state": req.current_state.model_dump(),
            "results": []
        }


# ===== HEALTH CHECK =====
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    # Count facilities with GPS
    gps_count = 0
    if df_filtered is not None and 'lat' in df_filtered.columns:
        gps_count = df_filtered[
            (df_filtered['lat'].notna()) & 
            (df_filtered['lon'].notna())
        ].shape[0]
    
    return {
        "status": "healthy",
        "facilities_loaded": len(df_filtered) if df_filtered is not None else 0,
        "facilities_with_gps": gps_count,
        "gps_coverage_percent": round(gps_count / len(df_filtered) * 100, 1) if df_filtered is not None and len(df_filtered) > 0 else 0,
        "vector_db_initialized": vector_db is not None,
        "kakao_geocoding_enabled": bool(KAKAO_REST_API_KEY),
        "architecture": "Router-Controller (Tree-Based) + Kakao Geocoding"
    }
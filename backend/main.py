import os
import sys

# FORCE UNBUFFERED OUTPUT - Must be at the very top
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

"""
========================================
SEOUL MEDICAL FACILITY SEARCH BACKEND
========================================

ARCHITECTURE OVERVIEW:
---------------------

1. DATA SOURCE (Parquet)
   - Loads facilities from HuggingFace parquet file
   - Extracts unique specialties/categories dynamically
   - No hardcoded specialty mappings
   
2. SPECIALTY MATCHING (LLM-based)
   - LLM receives list of actual specialties from parquet
   - Matches user intent (English/Korean) to closest specialty
   - Confidence scoring for ambiguous matches
   - Examples: "dentist" → "치과", "dermatologist" → "피부과"

3. LOCATION IDENTIFICATION (Google Maps API + Kakao Fallback)
   - PRIMARY: Google Maps Geocoding API
     * Direct geocoding
     * Auto-append ", Seoul" if first attempt fails
     * Place search for landmarks/business names
   - FALLBACK: Kakao Maps API (if Google fails)
   
   - Supports multiple input types:
     * GPS coordinates (latitude, longitude)
     * Korean addresses (도로명주소, 지번주소)
     * Districts/neighborhoods (구, 동)
     * Place names and landmarks
     * English location keywords
     * Partial addresses (auto-completed with "Seoul")
   
   - Returns: latitude, longitude, address_korean, district, dong
   
   FALLBACK CHAIN:
   1. Google Maps direct geocoding
   2. Google Maps with ", Seoul" appended  
   3. Google Maps place search
   4. Kakao Maps geocoding (various strategies)
   5. If all fail → return None

4. SEARCH MODES
   a) Zone-based: "in Gangnam" → filters by district/neighborhood
   b) Distance-based: "near me" → filters by GPS radius
   
5. RANKING ALGORITHM (Radius-Adaptive)
   - Small radius (≤2km): 50% specialty relevance + 50% distance
   - Medium radius (5km): 70% specialty relevance + 30% distance
   - Large radius (10km): 85% specialty relevance + 15% distance
   - Very large (20km+): 95% specialty relevance + 5% distance
   
6. EXTRACTION OPTIMIZATION
   - Full extraction: Initial queries with both specialty + location
   - Quick specialty extraction: When user changes only specialty
   - Quick location extraction: When user changes only location
   - Shorter prompts = faster responses, lower costs

7. SEARCH FLOW
   Input → LLM Extract → Google Maps Verify (+ ", Seoul" fallback) → 
   Filter Parquet → RAG Rank → Distance Sort → Results

GOOGLE MAPS INTEGRATION:
------------------------
- google_maps_geocode(address, add_seoul=True/False)
  * Convert address/place to GPS coordinates
  * Optionally append ", Seoul" for better Seoul-specific results
  
- google_maps_reverse_geocode(lat, lon)
  * Convert GPS coordinates to Korean address
  * Extract district (구) and neighborhood (동)
  
- google_maps_place_search(query)
  * Search for landmarks, business names, places
  * Returns place details with GPS + address
  
- google_maps_place_details(place_id)
  * Get detailed address components from place_id
  
KAKAO MAPS (Fallback):
---------------------
- kakao_geocode(address): Address → GPS + district + dong
- kakao_reverse_geocode(lat, lon): GPS → Address + district + dong
- Used when Google Maps API is unavailable or returns no results

RAG SEMANTIC SEARCH:
-------------------
- ChromaDB with OpenAI embeddings
- Indexes facility summaries and highlights
- Returns semantic relevance ranking
- Combined with distance for final scoring
"""

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
from distance import haversine, fuzzy_match_location
from models import ChatRequest, State
from utils import safe_convert_to_python
from prompt import ROUTER_PROMPT, EXTRACTION_PROMPT_V2, GENERATION_PROMPT
from deterministic import get_greeting_message,get_reset_confirmation,generate_change_acknowledgment, ask_for_missing_info,ask_for_specialty_clarification,generate_chit_chat_response, generate_recovery_prompt,format_response 
import re
from datetime import datetime
from pathlib import Path
import logging

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set third-party loggers to WARNING to reduce noise
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- 1. CONFIGURATION ---
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN") 
HF_REPO_ID = "ValerianFourel/seoul-medical-facilities"
HF_FILENAME = "facilities_metareviews_rag_ready.parquet"

CHROMA_PATH = "./chroma_db"
LOCAL_PARQUET_PATH = "./local_facilities_cache.parquet"

# Defaults
DEFAULT_LAT = 37.5219  # Yeouido
DEFAULT_LON = 126.9243
DEFAULT_MAX_DISTANCE = 5.0  # km - default search radius

vector_db = None
df_facilities = None  # Full dataset
df_filtered = None    # Filtered subset (Summaries not null)
available_specialties = []  # Unique specialties from parquet data


# --- 2. ENHANCED LOGGING UTILITIES ---

def print_separator(char='=', length=100):
    """Print a visual separator."""
    logger.info(char * length)


# --- 3. GOOGLE MAPS API FUNCTIONS ---

def google_maps_geocode(address: str, add_seoul: bool = False) -> Optional[Dict[str, Any]]:
    """
    Convert address to coordinates using Google Maps Geocoding API.
    Returns dict with {lat, lon, address_korean, district, dong, formatted_address} or None.
    
    Args:
        address: Address or place name to geocode
        add_seoul: If True, appends ", Seoul" to the search query
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("GOOGLE_MAPS_API_KEY not set - skipping geocoding")
        return None
    
    # Add Seoul to query if requested
    search_query = f"{address}, Seoul" if add_seoul else address
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": search_query,
        "key": GOOGLE_MAPS_API_KEY,
        "language": "ko"  # Request Korean results
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            result_data = data['results'][0]
            location = result_data['geometry']['location']
            
            result = {
                'lat': location['lat'],
                'lon': location['lng'],
                'formatted_address': result_data['formatted_address']
            }
            
            # Parse address components for district and dong
            for component in result_data.get('address_components', []):
                types = component.get('types', [])
                
                # District (구) - sublocality_level_1
                if 'sublocality_level_1' in types:
                    result['district'] = component['long_name']
                
                # Neighborhood (동) - sublocality_level_2
                elif 'sublocality_level_2' in types:
                    result['dong'] = component['long_name']
            
            # Use formatted address as Korean address
            result['address_korean'] = result['formatted_address']
            
            # Set defaults if not found
            result.setdefault('district', '')
            result.setdefault('dong', '')
            
            logger.info(f"✓ Google geocoded: {address[:30]}... → {result.get('district', 'Unknown')}")
            return result
        else:
            logger.warning(f"Google geocoding failed: {data.get('status')} for '{address}'")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Maps geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Google Maps geocoding parse error: {e}")
        return None


def google_maps_reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Convert coordinates to address using Google Maps Reverse Geocoding API.
    Returns dict with {address_korean, district, dong, formatted_address} or None.
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("GOOGLE_MAPS_API_KEY not set - skipping reverse geocoding")
        return None
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": GOOGLE_MAPS_API_KEY,
        "language": "ko"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            result_data = data['results'][0]
            
            result = {
                'formatted_address': result_data['formatted_address'],
                'address_korean': result_data['formatted_address']
            }
            
            # Parse address components for district and dong
            for component in result_data.get('address_components', []):
                types = component.get('types', [])
                
                # District (구)
                if 'sublocality_level_1' in types:
                    result['district'] = component['long_name']
                
                # Neighborhood (동)
                elif 'sublocality_level_2' in types:
                    result['dong'] = component['long_name']
            
            # Set defaults if not found
            result.setdefault('district', '')
            result.setdefault('dong', '')
            
            logger.info(f"✓ Google reverse geocoded: ({lat:.4f}, {lon:.4f}) → {result.get('district', 'Unknown')}")
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Maps reverse geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Google Maps reverse geocoding parse error: {e}")
        return None


def google_maps_place_search(query: str) -> Optional[Dict[str, Any]]:
    """
    Search for a place using Google Maps Places API (Text Search).
    Returns dict with {lat, lon, address_korean, district, dong, place_name} or None.
    
    This is useful for landmarks, business names, etc.
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("GOOGLE_MAPS_API_KEY not set - skipping place search")
        return None
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{query}, Seoul",
        "key": GOOGLE_MAPS_API_KEY,
        "language": "ko"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            result_data = data['results'][0]
            location = result_data['geometry']['location']
            
            result = {
                'lat': location['lat'],
                'lon': location['lng'],
                'place_name': result_data.get('name', ''),
                'formatted_address': result_data.get('formatted_address', '')
            }
            
            # Get detailed info using place_id
            place_id = result_data.get('place_id')
            if place_id:
                details = google_maps_place_details(place_id)
                if details:
                    result.update(details)
            
            result['address_korean'] = result.get('formatted_address', '')
            result.setdefault('district', '')
            result.setdefault('dong', '')
            
            logger.info(f"✓ Google place search: {query[:30]}... → {result.get('place_name', 'Unknown')}")
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Maps place search error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Google Maps place search parse error: {e}")
        return None


def google_maps_place_details(place_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a place using Google Maps Place Details API.
    """
    if not GOOGLE_MAPS_API_KEY:
        return None
    
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_MAPS_API_KEY,
        "language": "ko",
        "fields": "address_components,formatted_address"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('result'):
            result_data = data['result']
            result = {}
            
            # Parse address components
            for component in result_data.get('address_components', []):
                types = component.get('types', [])
                
                if 'sublocality_level_1' in types:
                    result['district'] = component['long_name']
                elif 'sublocality_level_2' in types:
                    result['dong'] = component['long_name']
            
            return result
        else:
            return None
            
    except Exception as e:
        logger.error(f"Google Maps place details error: {e}")
        return None


# --- 4. ENHANCED KAKAO API FUNCTIONS (kept as fallback) ---

def kakao_geocode(address: str) -> Optional[Dict[str, Any]]:
    """
    Convert address to coordinates using Kakao API with full address standardization.
    Returns dict with {lat, lon, address_korean, district, dong, address_type} or None.
    """
    if not KAKAO_REST_API_KEY:
        logger.warning("KAKAO_REST_API_KEY not set - skipping geocoding")
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
            result = {}
            
            # Try road address first, then jibun address
            if doc.get('road_address'):
                addr = doc['road_address']
                result['lat'] = float(addr['y'])
                result['lon'] = float(addr['x'])
                result['address_korean'] = addr['address_name']
                result['address_type'] = 'road'
                result['district'] = addr.get('region_2depth_name', '')  # 구
                result['dong'] = addr.get('region_3depth_name', '')      # 동
            elif doc.get('address'):
                addr = doc['address']
                result['lat'] = float(addr['y'])
                result['lon'] = float(addr['x'])
                result['address_korean'] = addr['address_name']
                result['address_type'] = 'jibun'
                result['district'] = addr.get('region_2depth_name', '')  # 구
                result['dong'] = addr.get('region_3depth_name', '')      # 동
            else:
                return None
            
            logger.info(f"✓ Geocoded: {address[:30]}... → {result['district']}")
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Kakao geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Kakao geocoding parse error: {e}")
        return None


def kakao_reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Convert coordinates to address using Kakao API.
    Returns dict with {address_korean, district, dong} or None.
    """
    if not KAKAO_REST_API_KEY:
        logger.warning("KAKAO_REST_API_KEY not set - skipping reverse geocoding")
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
            result = {}
            
            # Try road address first, then jibun address
            if doc.get('road_address'):
                addr = doc['road_address']
                result['address_korean'] = addr['address_name']
                result['district'] = addr.get('region_2depth_name', '')
                result['dong'] = addr.get('region_3depth_name', '')
            elif doc.get('address'):
                addr = doc['address']
                result['address_korean'] = addr['address_name']
                result['district'] = addr.get('region_2depth_name', '')
                result['dong'] = addr.get('region_3depth_name', '')
            else:
                return None
            
            logger.info(f"✓ Reverse geocoded: ({lat:.4f}, {lon:.4f}) → {result['district']}")
            return result
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Kakao reverse geocoding error: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Kakao reverse geocoding parse error: {e}")
        return None


def verify_and_standardize_address(location_text: str) -> Optional[Dict[str, Any]]:
    """
    Verify and standardize any location input using Google Maps API with fallbacks.
    Returns standardized address info with coordinates and zone details.
    
    FALLBACK STRATEGY:
    1. Try Google Maps geocoding directly
    2. Try Google Maps with ", Seoul" added
    3. Try Google Maps place search (for landmarks, business names)
    4. Try Kakao Maps as final fallback (if available)
    
    This is the main entry point for address verification.
    """
    if not location_text:
        return None
    
    logger.debug(f"🔍 Verifying location: '{location_text}'")
    
    # ===== STRATEGY 1: Direct Google Maps Geocoding =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps direct geocoding...")
        result = google_maps_geocode(location_text, add_seoul=False)
        if result:
            logger.info(f"✅ Google Maps (direct): '{location_text}' → {result.get('district', 'Unknown')}")
            return result
    
    # ===== STRATEGY 2: Google Maps with ", Seoul" =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps with ', Seoul' appended...")
        result = google_maps_geocode(location_text, add_seoul=True)
        if result:
            logger.info(f"✅ Google Maps (+ Seoul): '{location_text}' → {result.get('district', 'Unknown')}")
            return result
    
    # ===== STRATEGY 3: Google Maps Place Search =====
    if GOOGLE_MAPS_API_KEY:
        logger.debug("Trying Google Maps place search...")
        result = google_maps_place_search(location_text)
        if result:
            logger.info(f"✅ Google Maps (place search): '{location_text}' → {result.get('place_name', 'Unknown')}")
            return result
    
    # ===== STRATEGY 4: Kakao Maps Fallback =====
    if KAKAO_REST_API_KEY:
        logger.debug("Trying Kakao Maps as fallback...")
        
        # Try direct Kakao geocoding
        result = kakao_geocode(location_text)
        if result:
            logger.info(f"✅ Kakao Maps (fallback): '{location_text}' → {result.get('district', 'Unknown')}")
            return result
        
        # Try adding "서울" prefix for Kakao
        if not location_text.startswith("서울") and not location_text.lower().startswith("seoul"):
            result = kakao_geocode(f"서울 {location_text}")
            if result:
                logger.info(f"✅ Kakao Maps (+ 서울): '{location_text}' → {result.get('district', 'Unknown')}")
                return result
        
        # Try adding "구" suffix for Kakao
        if "구" not in location_text and "동" not in location_text:
            result = kakao_geocode(f"서울 {location_text}구")
            if result:
                logger.info(f"✅ Kakao Maps (+ 구): '{location_text}' → {result.get('district', 'Unknown')}")
                return result
    
    # ===== ALL STRATEGIES FAILED =====
    logger.warning(f"❌ All geocoding attempts failed for: '{location_text}'")
    logger.warning("   → Google Maps API: " + ("Available" if GOOGLE_MAPS_API_KEY else "NOT CONFIGURED"))
    logger.warning("   → Kakao Maps API: " + ("Available" if KAKAO_REST_API_KEY else "NOT CONFIGURED"))
    
    return None


def detect_search_mode(location_text: str, state: State) -> str:
    """
    Detect if user wants zone-based or distance-based search.
    Returns 'zone' or 'distance'.
    """
    if not location_text:
        return 'distance'
    
    # Zone indicators
    zone_keywords = ["in ", "구", "동", "district", "area", "zone", "neighborhood"]
    
    # Distance indicators
    distance_keywords = ["near", "close", "nearby", "km", "meter", "around", "근처", "주변", "가까운"]
    
    location_lower = location_text.lower()
    
    has_zone_keyword = any(kw in location_lower for kw in zone_keywords)
    has_distance_keyword = any(kw in location_lower for kw in distance_keywords)
    
    # If both or neither, prefer zone if we have district/dong info
    if has_zone_keyword or (not has_distance_keyword and state.district):
        return 'zone'
    else:
        return 'distance'


# --- 4. DATA LOADING ---

def download_and_cache_parquet():
    """Download parquet from HuggingFace and cache locally."""
    if os.path.exists(LOCAL_PARQUET_PATH):
        logger.info(f"✅ Loading cached parquet from {LOCAL_PARQUET_PATH}")
        df = pd.read_parquet(LOCAL_PARQUET_PATH)
    else:
        logger.info("📥 Downloading parquet from HuggingFace...")
        remote_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=HF_FILENAME, 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        df = pd.read_parquet(remote_path)
        
        # Cache locally
        df.to_parquet(LOCAL_PARQUET_PATH)
        logger.info(f"💾 Cached parquet locally to {LOCAL_PARQUET_PATH}")
    
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


# --- 5. LIFESPAN (STARTUP) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, df_facilities, df_filtered, available_specialties
    logger.info("🚀 Booting SeoulMedBot Backend...")

    try:
        df_facilities = download_and_cache_parquet()
        logger.info(f"✅ Loaded {len(df_facilities)} facilities from parquet")
        
        # FILTER: Only use facilities with Summaries
        df_filtered = df_facilities[df_facilities['Summaries'].notna()].copy()
        logger.info(f"✅ Filtered to {len(df_filtered)} facilities with summaries")
        
        # Extract unique specialties/categories from parquet
        if 'category' in df_filtered.columns:
            available_specialties = sorted(df_filtered['category'].dropna().unique().tolist())
            logger.info(f"📋 Extracted {len(available_specialties)} unique specialties from data")
            logger.info(f"   Sample: {available_specialties[:5]}")
        else:
            logger.warning("⚠️ No 'category' column found in parquet")
            available_specialties = []
        
        # Show location data
        if 'file_district' in df_filtered.columns:
            unique_districts = df_filtered['file_district'].dropna().unique()
            logger.info(f"📍 Districts: {len(unique_districts)}")
        
        # Check for GPS coordinates in parquet
        if 'lat' in df_filtered.columns and 'lon' in df_filtered.columns:
            coords_count = df_filtered[
                (df_filtered['lat'].notna()) & 
                (df_filtered['lon'].notna())
            ].shape[0]
            logger.info(f"🌐 GPS coverage: {coords_count/len(df_filtered)*100:.1f}%")
        
        # Check geocoding services
        if GOOGLE_MAPS_API_KEY:
            logger.info("🗺️ Google Maps geocoding: ENABLED (PRIMARY)")
        else:
            logger.warning("⚠️ Google Maps geocoding: DISABLED")
        
        if KAKAO_REST_API_KEY:
            logger.info("🗺️ Kakao Maps geocoding: ENABLED (FALLBACK)")
        else:
            logger.warning("⚠️ Kakao Maps geocoding: DISABLED")
        
        if not GOOGLE_MAPS_API_KEY and not KAKAO_REST_API_KEY:
            logger.error("❌ NO GEOCODING SERVICE CONFIGURED - Location features will be limited!")
        
        df_filtered['place_id'] = df_filtered['place_id'].fillna('').astype(str)        
    except Exception as e:
        logger.error(f"PARQUET LOAD ERROR: {e}", exc_info=True)
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
        logger.info("✅ Embeddings collection found.")
    except:
        logger.info("⚠️ Creating new embeddings from filtered dataset...")
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
            
        logger.info("✅ Embeddings indexed.")

    yield
    logger.info("🛑 Shutting down.")


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

    if has_english:
        return "English"
    elif has_korean:
        return "Korean"
    else:
        return "English"  # Default


def _format_location_summary(state: State) -> str:
    """Helper to format location data for logging."""
    parts = []
    
    if state.location:
        parts.append(f"location={state.location}")
    
    if state.latitude and state.longitude:
        parts.append(f"GPS=({state.latitude:.4f},{state.longitude:.4f})")
    
    if state.district:
        parts.append(f"district={state.district}")
    
    if state.dong:
        parts.append(f"dong={state.dong}")
    
    if state.address_korean:
        parts.append(f"address={state.address_korean[:30]}...")
    
    return " | ".join(parts) if parts else "No location data"


def standardize_and_fill_state(state: State) -> State:
    """
    Standardize and fill missing location fields in state using Kakao API.
    This ensures all location data is complete and consistent.
    
    Priority:
    1. If we have address/location text but no GPS → geocode to get GPS + district/dong
    2. If we have GPS but no address/district/dong → reverse geocode
    3. If we have district/dong but no GPS → geocode district to get approximate GPS
    
    Returns enriched state with all available location data.
    """
    enriched_state = state.model_copy()
    
    logger.info("=" * 60)
    logger.info("🔧 STATE ENRICHMENT STARTED")
    logger.info("=" * 60)
    
    # Track what we're filling in
    filled_fields = []
    
    # ===== CASE 1: Have address/location text but missing GPS/district/dong =====
    if state.location and not (state.latitude and state.longitude):
        logger.info(f"📍 Case 1: Have location text '{state.location}', need GPS data")
        
        verified = verify_and_standardize_address(state.location)
        
        if verified:
            if not state.latitude:
                enriched_state.latitude = verified['lat']
                filled_fields.append('latitude')
            
            if not state.longitude:
                enriched_state.longitude = verified['lon']
                filled_fields.append('longitude')
            
            if not state.address_korean:
                enriched_state.address_korean = verified['address_korean']
                filled_fields.append('address_korean')
            
            if not state.district:
                enriched_state.district = verified['district']
                filled_fields.append('district')
            
            if not state.dong:
                enriched_state.dong = verified['dong']
                filled_fields.append('dong')
            
            # Set search mode if not already set
            if not state.search_mode:
                enriched_state.search_mode = detect_search_mode(state.location, enriched_state)
                filled_fields.append('search_mode')
            
            logger.info(f"✅ Filled from address: {', '.join(filled_fields)}")
        else:
            logger.warning(f"Could not geocode '{state.location}'")
    
    # ===== CASE 2: Have GPS but missing address/district/dong =====
    elif state.latitude and state.longitude and not (state.address_korean and state.district):
        logger.info(f"📍 Case 2: Have GPS ({state.latitude:.4f}, {state.longitude:.4f}), need address data")
        
        # Try Google Maps reverse geocoding first
        reverse_result = None
        if GOOGLE_MAPS_API_KEY:
            reverse_result = google_maps_reverse_geocode(state.latitude, state.longitude)
        
        # Fallback to Kakao if Google fails
        if not reverse_result and KAKAO_REST_API_KEY:
            reverse_result = kakao_reverse_geocode(state.latitude, state.longitude)
        
        if reverse_result:
            if not state.address_korean:
                enriched_state.address_korean = reverse_result['address_korean']
                filled_fields.append('address_korean')
            
            if not state.district:
                enriched_state.district = reverse_result['district']
                filled_fields.append('district')
            
            if not state.dong:
                enriched_state.dong = reverse_result['dong']
                filled_fields.append('dong')
            
            # Update location text if not set
            if not state.location:
                enriched_state.location = reverse_result['district']
                filled_fields.append('location')
            
            # Set search mode if not already set
            if not state.search_mode:
                enriched_state.search_mode = 'distance'  # GPS implies distance-based search
                filled_fields.append('search_mode')
            
            logger.info(f"✅ Filled from GPS: {', '.join(filled_fields)}")
        else:
            logger.warning("Could not reverse geocode GPS coordinates")
    
    # ===== CASE 3: Have district/dong but missing GPS =====
    elif state.district and not (state.latitude and state.longitude):
        logger.info(f"📍 Case 3: Have district '{state.district}', need GPS")
        
        # Try to geocode the district
        location_query = f"서울 {state.district}"
        if state.dong:
            location_query = f"서울 {state.district} {state.dong}"
        
        verified = verify_and_standardize_address(location_query)
        
        if verified:
            if not state.latitude:
                enriched_state.latitude = verified['lat']
                filled_fields.append('latitude')
            
            if not state.longitude:
                enriched_state.longitude = verified['lon']
                filled_fields.append('longitude')
            
            if not state.address_korean:
                enriched_state.address_korean = verified['address_korean']
                filled_fields.append('address_korean')
            
            # Update location text if not set
            if not state.location:
                enriched_state.location = state.district
                filled_fields.append('location')
            
            # Set search mode if not already set
            if not state.search_mode:
                enriched_state.search_mode = 'zone'  # District implies zone-based search
                filled_fields.append('search_mode')
            
            logger.info(f"✅ Filled from district: {', '.join(filled_fields)}")
        else:
            logger.warning(f"Could not geocode district '{state.district}'")
    
    # ===== CASE 4: Already complete =====
    else:
        if state.latitude and state.longitude and state.district:
            logger.info("✅ State already complete - no enrichment needed")
        else:
            logger.info("ℹ️ Insufficient data for enrichment")
    
    # ===== SUMMARY =====
    if filled_fields:
        logger.info("\n📋 ENRICHMENT SUMMARY:")
        logger.info(f"   Before: {_format_location_summary(state)}")
        logger.info(f"   After:  {_format_location_summary(enriched_state)}")
        logger.info(f"   Filled: {', '.join(filled_fields)}")
    
    logger.info("=" * 60 + "\n")
    
    return enriched_state


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
        logger.debug("🧹 Cleansing: specialty")
        new_state.specialty = None
        new_state.specialty_confidence = 0.0
    
    if has_location_change:
        logger.debug("🧹 Cleansing: location data")
        new_state.location = None
        new_state.latitude = None
        new_state.longitude = None
        new_state.address_korean = None
        new_state.district = None
        new_state.dong = None
        new_state.search_mode = None
    
    # Always reset search flags when changing
    new_state.ready_to_search = False
    new_state.search_executed = False
    new_state.conversation_phase = "gathering"
    
    return new_state


def extract_entities(user_message: str) -> Dict[str, Any]:
    """
    Call the EXTRACTION_PROMPT_V2 (pure extractor) and verify address via Kakao API.
    Matches user intent to actual specialties from parquet data.
    This is a leaf node that only extracts, doesn't reason about flow.
    """
    # Build list of available specialties for LLM matching
    specialty_list = ", ".join(available_specialties[:50]) if available_specialties else "No specialties available"
    
    # Create extraction prompt with actual specialty options
    extraction_prompt = f"""You are extracting medical specialty and location from user messages.

AVAILABLE SPECIALTIES (from actual data):
{specialty_list}

Extract from this message: "{user_message}"

Return JSON with:
{{
  "specialty": "exact match from available specialties above, or null",
  "specialty_confidence": 0.0-1.0 (how confident the match is),
  "location": "any location mentioned (address, district, place name, landmark)",
  "language_pref": "English Preferred" or "Korean"
}}

RULES:
- specialty: MUST match one of the available specialties listed above, or null if no medical intent
- specialty_confidence: 1.0 if exact match, 0.7-0.9 if close match, 0.3-0.6 if uncertain, 0.0 if no match
- location: Extract ANY location reference (addresses, districts, neighborhoods, landmarks, place names)
- Match Korean and English specialty names

Examples:
"치과 찾아줘" → {{"specialty": "치과", "specialty_confidence": 1.0, ...}}
"I need a dentist" → {{"specialty": "치과", "specialty_confidence": 0.9, ...}}  
"강남구에서 피부과" → {{"specialty": "피부과", "location": "강남구", ...}}
"""
    
    extraction_messages = [{
        "role": "system",
        "content": extraction_prompt
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
        
        # Log the extracted specialty match
        if extracted.get('specialty'):
            logger.info(f"🎯 Specialty match: '{extracted['specialty']}' (confidence: {extracted.get('specialty_confidence', 0):.2f})")
        
        # ENHANCED: Verify and standardize address via Kakao API
        if extracted.get('location'):
            logger.debug(f"📍 Location extraction: '{extracted['location']}'")
            verified = verify_and_standardize_address(extracted['location'])
            
            if verified:
                # Replace with verified Kakao data
                extracted['latitude'] = verified['lat']
                extracted['longitude'] = verified['lon']
                extracted['address_korean'] = verified['address_korean']
                extracted['district'] = verified['district']
                extracted['dong'] = verified['dong']
                
                logger.info(f"✅ Kakao verified: {verified['district']} ({verified['lat']:.4f}, {verified['lon']:.4f})")
            else:
                logger.warning(f"⚠️ Kakao could not verify: '{extracted['location']}'")
        
        return extracted
        
    except Exception as e:
        logger.error(f"Extraction Error: {e}", exc_info=True)
        return {}


def quick_extract_location_change(user_message: str) -> Dict[str, Any]:
    """
    SHORTER extraction for when user is just changing location.
    Uses minimal prompt for efficiency.
    """
    extraction_prompt = f"""Extract location from: "{user_message}"

Return JSON: {{"location": "extracted location or null"}}

Extract: districts (구), neighborhoods (동), addresses, place names, landmarks.
Examples: "강남구", "Gangnam", "서울역", "역삼동"
"""
    
    extraction_messages = [{"role": "system", "content": extraction_prompt}]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=extraction_messages,
            temperature=0.0,
            max_completion_tokens=128,
            response_format={"type": "json_object"}
        )
        extracted = json.loads(completion.choices[0].message.content)
        
        # Verify via Kakao
        if extracted.get('location'):
            verified = verify_and_standardize_address(extracted['location'])
            
            if verified:
                extracted['latitude'] = verified['lat']
                extracted['longitude'] = verified['lon']
                extracted['address_korean'] = verified['address_korean']
                extracted['district'] = verified['district']
                extracted['dong'] = verified['dong']
                logger.info(f"✅ Quick location change: {verified['district']}")
            else:
                logger.warning(f"⚠️ Could not verify: '{extracted['location']}'")
        
        return extracted
        
    except Exception as e:
        logger.error(f"Quick extraction error: {e}", exc_info=True)
        return {}


def quick_extract_specialty_change(user_message: str) -> Dict[str, Any]:
    """
    SHORTER extraction for when user is just changing specialty.
    Uses minimal prompt for efficiency.
    """
    specialty_list = ", ".join(available_specialties[:50]) if available_specialties else ""
    
    extraction_prompt = f"""Match user intent to specialty from: "{user_message}"

AVAILABLE: {specialty_list}

Return JSON: {{
  "specialty": "matched specialty or null",
  "specialty_confidence": 0.0-1.0
}}

Match Korean/English names. Examples: "dentist" → "치과", "dermatologist" → "피부과"
"""
    
    extraction_messages = [{"role": "system", "content": extraction_prompt}]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=extraction_messages,
            temperature=0.0,
            max_completion_tokens=128,
            response_format={"type": "json_object"}
        )
        extracted = json.loads(completion.choices[0].message.content)
        
        if extracted.get('specialty'):
            logger.info(f"✅ Quick specialty change: '{extracted['specialty']}' ({extracted.get('specialty_confidence', 0):.2f})")
        
        return extracted
        
    except Exception as e:
        logger.error(f"Quick extraction error: {e}", exc_info=True)
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
    
    if extracted.get('address_korean'):
        state.address_korean = extracted['address_korean']
    
    if extracted.get('district'):
        state.district = extracted['district']
    
    if extracted.get('dong'):
        state.dong = extracted['dong']
    
    if extracted.get('language_pref'):
        state.language_pref = extracted['language_pref']
    
    # Detect search mode based on location context
    if state.location:
        state.search_mode = detect_search_mode(state.location, state)
    
    return state


def filter_by_zone(df: pd.DataFrame, district: str, dong: Optional[str] = None) -> pd.DataFrame:
    """
    Filter facilities by zone (district and optionally dong).
    Uses fuzzy matching for flexibility.
    """
    if not district:
        return df
    
    logger.info(f"🏘️ Zone filter: {district} {dong or ''}")
    
    # Exact match first
    if 'file_district' in df.columns:
        mask = df['file_district'].str.contains(district, na=False, case=False)
        
        # Add dong filter if specified
        if dong and 'file_dong' in df.columns:
            mask = mask & df['file_dong'].str.contains(dong, na=False, case=False)
        
        result = df[mask]
        
        if len(result) > 0:
            logger.info(f"✓ Filtered: {len(df)} → {len(result)}")
            return result
    
    # Fallback: fuzzy match
    logger.warning("Using fuzzy matching")
    return fuzzy_match_location(district, df)


def validate_distance_criteria(df: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    """
    Validate that results respect distance criteria.
    Filters out facilities beyond max_distance.
    """
    if 'distance_km' not in df.columns:
        return df
    
    # Remove facilities with invalid/missing distance
    df_valid = df[df['distance_km'].notna()].copy()
    
    # Filter by max distance
    before = len(df_valid)
    df_valid = df_valid[df_valid['distance_km'] <= max_distance]
    
    if len(df_valid) < before:
        logger.info(f"✂️ Distance validation: {before} → {len(df_valid)} (max {max_distance}km)")
    
    return df_valid


def execute_search(state: State, user_message: str, max_distance: float = DEFAULT_MAX_DISTANCE) -> Tuple[str, List[Dict]]:
    """
    Execute the actual search logic with:
    - Address verification via Kakao
    - Zone-based OR distance-based search
    - Distance validation
    """
    logger.info("=" * 60)
    logger.info("🔍 SEARCH STARTED")
    logger.info(f"Query: \"{user_message[:50]}...\"")
    logger.info(f"Specialty: {state.specialty or 'Any'}")
    logger.info(f"Mode: {(state.search_mode or 'auto').upper()}")
    logger.info(f"Max Distance: {max_distance}km")
    logger.info("=" * 60)
    
    working_df = df_filtered.copy()
    location_context = ""
    user_lat = state.latitude
    user_lon = state.longitude
    
    # ===== LOCATION FILTERING =====
    
    if state.search_mode == 'zone' and state.district:
        # ZONE-BASED SEARCH
        logger.info(f"🏘️ Zone search: {state.district} {state.dong or ''}")
        
        working_df = filter_by_zone(working_df, state.district, state.dong)
        
        if state.dong:
            location_context = f"in {state.dong}, {state.district}"
        else:
            location_context = f"in {state.district}"
        
        # Still calculate distances for ranking if GPS available
        if user_lat and user_lon and 'lat' in working_df.columns and 'lon' in working_df.columns:
            def calc_distance(row):
                if pd.notna(row['lat']) and pd.notna(row['lon']):
                    return haversine(user_lat, user_lon, row['lat'], row['lon'])
                return 999.0
            
            working_df['distance_km'] = working_df.apply(calc_distance, axis=1)
            working_df = working_df.sort_values('distance_km')
        else:
            working_df['distance_km'] = 0
    
    elif user_lat and user_lon:
        # GPS-BASED DISTANCE SEARCH
        logger.info(f"📍 GPS search: ({user_lat:.4f}, {user_lon:.4f})")
        
        # Get address from coordinates for context (try Google Maps first)
        reverse_result = None
        if GOOGLE_MAPS_API_KEY:
            reverse_result = google_maps_reverse_geocode(user_lat, user_lon)
        
        # Fallback to Kakao
        if not reverse_result and KAKAO_REST_API_KEY:
            reverse_result = kakao_reverse_geocode(user_lat, user_lon)
        
        if reverse_result:
            location_context = f"near {reverse_result['address_korean']}"
            
            # Store district/dong if not already set
            if not state.district:
                state.district = reverse_result.get('district')
                state.dong = reverse_result.get('dong')
        else:
            location_context = f"near your location"
        
        # Calculate distances for all facilities with coordinates
        if 'lat' in working_df.columns and 'lon' in working_df.columns:
            def calc_distance(row):
                if pd.notna(row['lat']) and pd.notna(row['lon']):
                    return haversine(user_lat, user_lon, row['lat'], row['lon'])
                return 999.0
            
            working_df['distance_km'] = working_df.apply(calc_distance, axis=1)
            
            # Filter by distance
            before_count = len(working_df)
            working_df = working_df[working_df['distance_km'] < max_distance]
            logger.info(f"✓ Distance filter: {before_count} → {len(working_df)}")
            
            if len(working_df) == 0:
                # Expand search radius if nothing found
                expanded_radius = max_distance * 2
                logger.warning(f"Expanding to {expanded_radius}km")
                working_df = df_filtered.copy()
                working_df['distance_km'] = working_df.apply(calc_distance, axis=1)
                working_df = working_df[working_df['distance_km'] < expanded_radius]
                location_context = f"within {expanded_radius}km of your location"
            
            # Sort by distance
            working_df = working_df.sort_values('distance_km')
        else:
            working_df['distance_km'] = 0
            location_context = "in Seoul"
    
    elif state.location:
        # TEXT-BASED LOCATION SEARCH (fallback)
        logger.info(f"📍 Text location: {state.location}")
        working_df = fuzzy_match_location(state.location, working_df)
        location_context = f"in {state.location}"
        working_df['distance_km'] = 0
    
    else:
        # CITY-WIDE SEARCH
        logger.info("📍 City-wide search")
        location_context = "across Seoul"
        working_df['distance_km'] = 0
    
    # ===== CATEGORY FILTER =====
    # Specialty is already matched to parquet categories by LLM
    if state.specialty and 'category' in working_df.columns:
        before_count = len(working_df)
        # Direct match since LLM already matched to available specialties
        working_df = working_df[working_df['category'].str.contains(state.specialty, na=False, case=False)]
        logger.info(f"🏥 Category filter: '{state.specialty}' → {before_count} to {len(working_df)}")
    
    # ===== SEMANTIC RANKING (RAG) =====
    query_text = state.specialty or ""
    
    if len(query_text.strip()) > 2 and len(working_df) > 0:
        try:
            logger.debug("🔍 RAG ranking...")
            n_rag_results = min(200, len(vector_db.get()['ids']))
            rag_results = vector_db.query(query_texts=[query_text], n_results=n_rag_results)
            
            if rag_results and 'ids' in rag_results and len(rag_results['ids']) > 0:
                rag_ranking = {place_id: idx for idx, place_id in enumerate(rag_results['ids'][0])}
                working_df['relevance_rank'] = working_df['place_id'].apply(lambda pid: rag_ranking.get(pid, 9999))
                
                # Combined ranking: distance + relevance (only if distance mode)
                if state.search_mode != 'zone' and 'distance_km' in working_df.columns and working_df['distance_km'].max() > 0:
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
                    
                    # Dynamic weighting: larger radius = more emphasis on specialty relevance
                    # Formula: relevance_weight increases with search radius
                    # 2km → 50% relevance, 50% distance
                    # 5km → 70% relevance, 30% distance  
                    # 10km → 85% relevance, 15% distance
                    # 20km+ → 95% relevance, 5% distance
                    if max_distance <= 2:
                        relevance_weight = 0.50
                    elif max_distance <= 5:
                        # Linear interpolation between 2km and 5km
                        relevance_weight = 0.50 + (max_distance - 2) * (0.70 - 0.50) / (5 - 2)
                    elif max_distance <= 10:
                        # Linear interpolation between 5km and 10km
                        relevance_weight = 0.70 + (max_distance - 5) * (0.85 - 0.70) / (10 - 5)
                    else:
                        # Cap at 95% relevance for very large radii
                        relevance_weight = min(0.95, 0.85 + (max_distance - 10) * 0.01)
                    
                    distance_weight = 1 - relevance_weight
                    
                    # Weighted combination with dynamic weights
                    working_df['combined_score'] = (
                        relevance_weight * working_df['relevance_score'] + 
                        distance_weight * working_df['distance_score']
                    )
                    working_df = working_df.sort_values('combined_score', ascending=False)
                    logger.debug(f"✓ Combined ranking ({relevance_weight:.0%} relevance + {distance_weight:.0%} distance for {max_distance}km radius)")
                else:
                    # Zone mode: pure relevance ranking
                    working_df = working_df.sort_values('relevance_rank')
                    logger.debug(f"✓ RAG ranked: {(working_df['relevance_rank'] < 9999).sum()}/{len(working_df)}")
            else:
                working_df['relevance_rank'] = 9999
        except Exception as e:
            logger.warning(f"RAG error: {e}")
            working_df['relevance_rank'] = 9999
    else:
        working_df['relevance_rank'] = 9999
    
    # ===== ENGLISH FILTER =====
    language_pref = state.language_pref
    if "English" in language_pref and 'has_english' in working_df.columns:
        before = len(working_df)
        working_df = working_df[working_df['has_english'] == True]
        logger.info(f"🌐 English filter: {before} → {len(working_df)}")
    
    # ===== FINAL DISTANCE VALIDATION (for distance mode) =====
    if state.search_mode != 'zone' and 'distance_km' in working_df.columns:
        working_df = validate_distance_criteria(working_df, max_distance)
    
    # ===== GENERATE RESPONSE =====
    results = []
    
    if len(working_df) > 0:
        logger.info(f"✓ Building response from {min(10, len(working_df))} facilities")
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
            
            # ⭐ FIX: Add BOTH distance_km and distance fields for frontend compatibility
            distance_value = safe_convert_to_python(row.get('distance_km', 0))
            result['distance_km'] = distance_value
            result['distance'] = distance_value  # Frontend expects 'distance'
            
            result['relevance_rank'] = safe_convert_to_python(row.get('relevance_rank', 9999))
            
            results.append(result)
    else:
        # No results found
        if "English" in language_pref:
            response_text = "I couldn't find any facilities matching your criteria. Would you like to try a different specialty or area?"
        else:
            response_text = "검색 조건에 맞는 시설을 찾을 수 없습니다. 다른 전문 분야나 지역을 시도해 보시겠어요?"
    
    logger.info(f"✅ SEARCH COMPLETED: {len(results)} results")
    logger.info("=" * 60 + "\n")
    
    return response_text, results


# ==========================================
# MAIN CHAT ENDPOINT (Router-Controller)
# ==========================================

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Router-Controller Architecture:
    Step 0: Enrich state (fill in missing location data)
    Step 1: Route (classify intent)
    Step 2: Branch (execute appropriate logic)
    Step 3: Return enriched state to frontend
    """
    
    logger.info("=" * 60)
    logger.info("📨 NEW REQUEST")
    logger.info(f"Message: \"{req.message[:50]}...\"")
    logger.info("=" * 60)
    
    # ==========================================
    # STEP 0: STATE ENRICHMENT (Fill in blanks)
    # ==========================================
    enriched_state = standardize_and_fill_state(req.current_state)
    
    # Use enriched state for the rest of the processing
    current_turn = enriched_state.turn_count + 1
    
    # ==========================================
    # STEP 1: ROUTING (Root Node)
    # ==========================================
    
    router_messages = [{
        "role": "system",
        "content": ROUTER_PROMPT.format(
            specialty=enriched_state.specialty or "None",
            location=enriched_state.location or "None",
            ready_to_search=enriched_state.ready_to_search,
            turn_count=current_turn,
            search_executed=enriched_state.search_executed,
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
        
        logger.info(f"🧭 ROUTER: {intent} (confidence: {route.get('confidence', 0):.2f})")
        logger.info(f"   Turn: {current_turn}")
        
    except Exception as e:
        logger.error(f"Router Error: {e}", exc_info=True)
        intent = "PROVIDE_INFO"  # Safe fallback
    
    # ==========================================
    # STEP 2: CONTROLLER (Tree Traversal)
    # ==========================================

    # === BRANCH 1: NEW_SEARCH (Complete Reset) ===
    if intent == "NEW_SEARCH":
        logger.info("🔄 BRANCH: NEW_SEARCH")
        
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
        
        return {
            "response": response,
            "state": new_state.model_dump(),
            "results": []
        }
    
    # === BRANCH 2: CHANGE_CRITERIA ===
    elif intent == "CHANGE_CRITERIA":
        logger.info("🔀 BRANCH: CHANGE_CRITERIA")
        
        # Hard state cleansing (use enriched state)
        cleansed_state = cleanse_state_for_change(enriched_state, req.message)
        cleansed_state.turn_count = current_turn
        
        # Detect what's being changed for optimized extraction
        message_lower = req.message.lower()
        location_keywords = ["in ", "near ", "at ", "구", "동", "역", "gangnam", "songpa", "mapo", "강남", "송파", "location"]
        specialty_keywords = ["dentist", "doctor", "dermatologist", "pediatrician", "치과", "피부과", "병원", "의원", "내과", "specialty"]
        
        has_location_mention = any(kw in message_lower for kw in location_keywords)
        has_specialty_mention = any(kw in message_lower for kw in specialty_keywords)
        
        # Use optimized extraction based on what changed
        if has_location_mention and not has_specialty_mention:
            # Only location changed - use quick extraction
            logger.debug("🚀 Using quick location extraction")
            extracted = quick_extract_location_change(req.message)
        elif has_specialty_mention and not has_location_mention:
            # Only specialty changed - use quick extraction  
            logger.debug("🚀 Using quick specialty extraction")
            extracted = quick_extract_specialty_change(req.message)
        else:
            # Both or unclear - use full extraction
            logger.debug("🔍 Using full extraction")
            extracted = extract_entities(req.message)
        
        # Merge extracted data into cleansed state
        new_state = merge_extraction_into_state(cleansed_state, extracted)
        
        # ⭐ ENRICH AGAIN after extraction
        new_state = standardize_and_fill_state(new_state)
        
        # Check if we have enough to search (either location OR GPS OR zone)
        has_location = bool(new_state.location or (new_state.latitude and new_state.longitude) or new_state.district)
        
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
            response, results = execute_search(new_state, req.message, max_distance=new_state.max_distance_km)
            new_state.search_executed = True
             
        return {
            "response": response,
            "state": new_state.model_dump(),
            "results": results
        }
    
    # === BRANCH 3: PROVIDE_INFO ===
    elif intent == "PROVIDE_INFO":
        logger.info("📝 BRANCH: PROVIDE_INFO")
        
        # Extract entities from user message (will auto-verify address via Kakao)
        extracted = extract_entities(req.message)
        
        # Merge into enriched state (accumulate information)
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state = merge_extraction_into_state(new_state, extracted)
        
        # ⭐ ENRICH AGAIN after extraction
        new_state = standardize_and_fill_state(new_state)
        
        # Check if we have enough to search (either location OR GPS OR zone)
        has_location = bool(new_state.location or (new_state.latitude and new_state.longitude) or new_state.district)
        
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
            response, results = execute_search(new_state, req.message, max_distance=new_state.max_distance_km)
            new_state.search_executed = True

        return {
            "response": response,
            "state": new_state.model_dump(),
            "results": results
        }
    
    # === BRANCH 4: CHIT_CHAT ===
    elif intent == "CHIT_CHAT":
        logger.info("💬 BRANCH: CHIT_CHAT")
        
        new_state = enriched_state.model_copy()
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
        logger.info("🆘 BRANCH: HELP_RECOVERY")
        
        new_state = enriched_state.model_copy()
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
        logger.warning(f"Unknown intent: {intent}")
        
        # Try to detect language
        lang = detect_language(req.message)
        
        if "English" in lang:
            response = "I'm not sure how to help. Could you rephrase your request?"
        else:
            response = "잘 이해하지 못했습니다. 다시 말씀해 주시겠어요?"
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        
        return {
            "response": response,
            "state": new_state.model_dump(),
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
        "available_specialties_count": len(available_specialties),
        "sample_specialties": available_specialties[:10] if available_specialties else [],
        "vector_db_initialized": vector_db is not None,
        "geocoding_services": {
            "google_maps": "ENABLED ✓" if GOOGLE_MAPS_API_KEY else "disabled",
            "kakao_maps": "ENABLED ✓" if KAKAO_REST_API_KEY else "disabled"
        },
        "geocoding_strategy": "Google Maps (primary) → Kakao Maps (fallback)",
        "logging_mode": "STRUCTURED LOGGING (uvicorn compatible)",
        "architecture": "Dynamic Specialty Matching + Google Maps Location ID + Radius-Adaptive Relevance Ranking"
    }
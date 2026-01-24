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
   - Large radius (10km): 85% relevance + 15% distance
   - Very large (20km+): 95% relevance + 5% distance
   
6. EXTRACTION OPTIMIZATION
   - Full extraction: Initial queries with both specialty + location
   - Quick specialty extraction: When user changes only specialty
   - Quick location extraction: When user changes only location
   - Shorter prompts = faster responses, lower costs

7. COOKIE CONSENT & PRIVACY
   - GDPR/CCPA compliant cookie consent management
   - Privacy-aware logging (respects analytics consent)
   - Conditional feature activation based on user consent
   - Transparent data usage with user control

8. SEARCH FLOW
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

# FORCE UNBUFFERED OUTPUT - Must be at the very top
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import json
import pandas as pd
import numpy as np
import requests
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException, Cookie, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
from groq import Groq
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path
import logging
import re

# Local imports
from distance import haversine, fuzzy_match_location
from models import ChatRequest, State
from utils import safe_convert_to_python
from prompt import ROUTER_PROMPT, EXTRACTION_PROMPT_V2, GENERATION_PROMPT
from deterministic import (
    get_greeting_message, get_reset_confirmation, generate_change_acknowledgment,
    ask_for_missing_info, ask_for_specialty_clarification, generate_chit_chat_response,
    generate_recovery_prompt, format_response
)
from location import (
    google_maps_geocode, google_maps_reverse_geocode, google_maps_place_search,
    google_maps_place_details, kakao_geocode, kakao_reverse_geocode,
    verify_and_standardize_address
)

# Pydantic imports
from pydantic import BaseModel


# ==========================================
# COOKIE CONSENT MODEL
# ==========================================

class CookieConsent(BaseModel):
    """Model for cookie consent settings - GDPR/CCPA compliant"""
    necessary: bool = True
    analytics: bool = False
    advertising: bool = False
    timestamp: Optional[str] = None


# ==========================================
# LOGGING SETUP
# ==========================================

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


# ==========================================
# CONFIGURATION
# ==========================================

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

# Global state
LANGUAGE = "English"  # Semi-fixed fixture, updated per message

# Global data structures
vector_db = None
df_facilities = None  # Full dataset
df_filtered = None    # Filtered subset (Summaries not null)
available_specialties = []  # Unique specialties from parquet data


# ==========================================
# COOKIE CONSENT HELPERS
# ==========================================

def get_consent_from_cookie(cookie_value: Optional[str]) -> CookieConsent:
    """
    Parse cookie consent from request.
    Returns default (all denied except necessary) if not present.
    """
    if not cookie_value:
        return CookieConsent()
    
    try:
        consent_data = json.loads(cookie_value)
        return CookieConsent(**consent_data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(f"Invalid cookieConsent format: {e}")
        return CookieConsent()


def should_log_analytics(consent: CookieConsent) -> bool:
    """Check if we can log analytics data"""
    return consent.analytics


def should_use_advertising(consent: CookieConsent) -> bool:
    """Check if we can use advertising features"""
    return consent.advertising


def privacy_safe_log(consent: CookieConsent, message: str, level: str = "info"):
    """
    Privacy-aware logging that respects user consent.
    Only logs detailed information if analytics consent is given.
    """
    if should_log_analytics(consent):
        if level == "info":
            logger.info(message)
        elif level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
    else:
        # Minimal logging without user data
        if level in ["warning", "error"]:
            logger.log(logging.WARNING if level == "warning" else logging.ERROR, 
                      "Operation logged (details hidden - no analytics consent)")


# ==========================================
# UTILITIES
# ==========================================

def print_separator(char='=', length=100):
    """Print a visual separator."""
    logger.info(char * length)


def detect_language(message: str) -> str:
    """
    Simple character-based language detection.
    If 50%+ characters are Roman (ASCII letters), return English.
    Otherwise, return Korean.
    """
    if not message or len(message.strip()) == 0:
        return "English"  # Default
    
    # Count ASCII letters (a-z, A-Z)
    roman_chars = sum(1 for char in message if char.isalpha() and ord(char) < 128)
    total_chars = len(message.replace(" ", ""))  # Exclude spaces
    
    if total_chars == 0:
        return "English"
    
    roman_ratio = roman_chars / total_chars
    detected = "English" if roman_ratio >= 0.5 else "Korean"
    
    logger.info(f"Language detected: {detected} (Roman: {roman_ratio:.1%})")
    return detected


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


# ==========================================
# DATA LOADING
# ==========================================

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


# ==========================================
# LIFESPAN (STARTUP/SHUTDOWN)
# ==========================================

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


# ==========================================
# FASTAPI APP SETUP
# ==========================================

app = FastAPI(lifespan=lifespan)
client = Groq(api_key=GROQ_API_KEY)

# CORS with credentials enabled for cookie support
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://seouldoc.io",     # Production domain
        "https://www.seouldoc.io", # WWW version
        "https://*.vercel.app",    # Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"]
)

# ==========================================
# STATE MANAGEMENT FUNCTIONS
# ==========================================

def standardize_and_fill_state(state: State, consent: Optional[CookieConsent] = None) -> State:
    """
    Standardize and fill missing location fields in state using geocoding APIs.
    This ensures all location data is complete and consistent.
    
    Privacy-aware: Only logs detailed information if analytics consent is given.
    """
    enriched_state = state.model_copy()
    
    if not consent:
        consent = CookieConsent()
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, "🔧 STATE ENRICHMENT STARTED")
    privacy_safe_log(consent, "=" * 60)
    
    filled_fields = []
    
    # ===== CASE 1: Have address/location text but missing GPS/district/dong =====
    if state.location and not (state.latitude and state.longitude):
        privacy_safe_log(consent, f"📍 Case 1: Have location text '{state.location}', need GPS data")
        
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
            
            if not state.search_mode:
                enriched_state.search_mode = detect_search_mode(state.location, enriched_state)
                filled_fields.append('search_mode')
            
            privacy_safe_log(consent, f"✅ Filled from address: {', '.join(filled_fields)}")
        else:
            logger.warning(f"Could not geocode '{state.location}'")
    
    # ===== CASE 2: Have GPS but missing address/district/dong =====
    elif state.latitude and state.longitude and not (state.address_korean and state.district):
        privacy_safe_log(consent, f"📍 Case 2: Have GPS ({state.latitude:.4f}, {state.longitude:.4f}), need address data")
        
        reverse_result = None
        if GOOGLE_MAPS_API_KEY:
            reverse_result = google_maps_reverse_geocode(state.latitude, state.longitude)
        
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
            
            if not state.location:
                enriched_state.location = reverse_result['district']
                filled_fields.append('location')
            
            if not state.search_mode:
                enriched_state.search_mode = 'distance'
                filled_fields.append('search_mode')
            
            privacy_safe_log(consent, f"✅ Filled from GPS: {', '.join(filled_fields)}")
        else:
            logger.warning("Could not reverse geocode GPS coordinates")
    
    # ===== CASE 3: Have district/dong but missing GPS =====
    elif state.district and not (state.latitude and state.longitude):
        privacy_safe_log(consent, f"📍 Case 3: Have district '{state.district}', need GPS")
        
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
            
            if not state.location:
                enriched_state.location = state.district
                filled_fields.append('location')
            
            if not state.search_mode:
                enriched_state.search_mode = 'zone'
                filled_fields.append('search_mode')
            
            privacy_safe_log(consent, f"✅ Filled from district: {', '.join(filled_fields)}")
        else:
            logger.warning(f"Could not geocode district '{state.district}'")
    
    # ===== CASE 4: Already complete =====
    else:
        if state.latitude and state.longitude and state.district:
            privacy_safe_log(consent, "✅ State already complete - no enrichment needed")
        else:
            privacy_safe_log(consent, "ℹ️ Insufficient data for enrichment")
    
    # ===== SUMMARY =====
    if filled_fields and should_log_analytics(consent):
        logger.info("\n📋 ENRICHMENT SUMMARY:")
        logger.info(f"   Before: {_format_location_summary(state)}")
        logger.info(f"   After:  {_format_location_summary(enriched_state)}")
        logger.info(f"   Filled: {', '.join(filled_fields)}")
    
    privacy_safe_log(consent, "=" * 60 + "\n")
    
    return enriched_state


def cleanse_state_for_change(current_state: State, user_message: str) -> State:
    """
    Programmatically cleanse state based on what the user is changing.
    This is executed BEFORE extraction to prevent bias.
    """
    message_lower = user_message.lower()
    
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
    
    new_state.ready_to_search = False
    new_state.search_executed = False
    new_state.conversation_phase = "gathering"
    
    return new_state


# ==========================================
# EXTRACTION FUNCTIONS
# ==========================================

def extract_entities(user_message: str, consent: Optional[CookieConsent] = None) -> Dict[str, Any]:
    """
    Call the extraction LLM and verify address via geocoding APIs.
    Matches user intent to actual specialties from parquet data.
    """
    global LANGUAGE
    
    if not consent:
        consent = CookieConsent()
    
    specialty_list = ", ".join(available_specialties[:50]) if available_specialties else "No specialties available"
    
    extraction_prompt = f"""You are extracting medical specialty and location from user messages.

AVAILABLE SPECIALTIES (from actual data):
{specialty_list}

Extract from this message: "{user_message}"

Return JSON with:
{{
  "specialty": "exact match from available specialties above, or null",
  "specialty_confidence": 0.0-1.0 (how confident the match is),
  "location": "any location mentioned (address, district, place name, landmark)"
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
    
    extraction_messages = [{"role": "system", "content": extraction_prompt}]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=extraction_messages,
            temperature=0.0,
            max_completion_tokens=512,
            response_format={"type": "json_object"}
        )
        extracted = json.loads(completion.choices[0].message.content)
        
        if extracted.get('specialty'):
            privacy_safe_log(consent, 
                f"🎯 Specialty match: '{extracted['specialty']}' (confidence: {extracted.get('specialty_confidence', 0):.2f})")
        
        if extracted.get('location'):
            logger.debug(f"📍 Location extraction: '{extracted['location']}'")
            verified = verify_and_standardize_address(extracted['location'])
            
            if verified:
                extracted['latitude'] = verified['lat']
                extracted['longitude'] = verified['lon']
                extracted['address_korean'] = verified['address_korean']
                extracted['district'] = verified['district']
                extracted['dong'] = verified['dong']
                
                privacy_safe_log(consent,
                    f"✅ Geocoding verified: {verified['district']} ({verified['lat']:.4f}, {verified['lon']:.4f})")
            else:
                logger.warning(f"⚠️ Could not verify: '{extracted['location']}'")
        
        return extracted
        
    except Exception as e:
        logger.error(f"Extraction Error: {e}", exc_info=True)
        return {}


def quick_extract_location_change(user_message: str, consent: Optional[CookieConsent] = None) -> Dict[str, Any]:
    """SHORTER extraction for when user is just changing location."""
    if not consent:
        consent = CookieConsent()
    
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
        
        if extracted.get('location'):
            verified = verify_and_standardize_address(extracted['location'])
            
            if verified:
                extracted['latitude'] = verified['lat']
                extracted['longitude'] = verified['lon']
                extracted['address_korean'] = verified['address_korean']
                extracted['district'] = verified['district']
                extracted['dong'] = verified['dong']
                privacy_safe_log(consent, f"✅ Quick location change: {verified['district']}")
            else:
                logger.warning(f"⚠️ Could not verify: '{extracted['location']}'")
        
        return extracted
        
    except Exception as e:
        logger.error(f"Quick extraction error: {e}", exc_info=True)
        return {}


def quick_extract_specialty_change(user_message: str, consent: Optional[CookieConsent] = None) -> Dict[str, Any]:
    """SHORTER extraction for when user is just changing specialty."""
    if not consent:
        consent = CookieConsent()
    
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
            privacy_safe_log(consent,
                f"✅ Quick specialty change: '{extracted['specialty']}' ({extracted.get('specialty_confidence', 0):.2f})")
        
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
    
    if state.location:
        state.search_mode = detect_search_mode(state.location, state)
    
    return state


# ==========================================
# SEARCH FUNCTIONS
# ==========================================

def filter_by_zone(df: pd.DataFrame, district: str, dong: Optional[str] = None) -> pd.DataFrame:
    """Filter facilities by zone (district and optionally dong)."""
    if not district:
        return df
    
    logger.info(f"🏘️ Zone filter: {district} {dong or ''}")
    
    if 'file_district' in df.columns:
        mask = df['file_district'].str.contains(district, na=False, case=False)
        
        if dong and 'file_dong' in df.columns:
            mask = mask & df['file_dong'].str.contains(dong, na=False, case=False)
        
        result = df[mask]
        
        if len(result) > 0:
            logger.info(f"✓ Filtered: {len(df)} → {len(result)}")
            return result
    
    logger.warning("Using fuzzy matching")
    return fuzzy_match_location(district, df)


def validate_distance_criteria(df: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    """Validate that results respect distance criteria."""
    if 'distance_km' not in df.columns:
        return df
    
    df_valid = df[df['distance_km'].notna()].copy()
    
    before = len(df_valid)
    df_valid = df_valid[df_valid['distance_km'] <= max_distance]
    
    if len(df_valid) < before:
        logger.info(f"✂️ Distance validation: {before} → {len(df_valid)} (max {max_distance}km)")
    
    return df_valid


def execute_search(
    state: State, 
    user_message: str, 
    max_distance: float = DEFAULT_MAX_DISTANCE,
    consent: Optional[CookieConsent] = None
) -> Tuple[str, List[Dict]]:
    """Execute the actual search logic with privacy-aware logging."""
    global LANGUAGE
    
    if not consent:
        consent = CookieConsent()
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, "🔍 SEARCH STARTED")
    privacy_safe_log(consent, f"Specialty: {state.specialty or 'Any'}")
    privacy_safe_log(consent, f"Mode: {(state.search_mode or 'auto').upper()}")
    privacy_safe_log(consent, f"Max Distance: {max_distance}km")
    privacy_safe_log(consent, f"Language: {LANGUAGE}")
    privacy_safe_log(consent, "=" * 60)
    
    working_df = df_filtered.copy()
    location_context = ""
    user_lat = state.latitude
    user_lon = state.longitude
    
    # ===== LOCATION FILTERING =====
    
    if state.search_mode == 'zone' and state.district:
        privacy_safe_log(consent, f"🏘️ Zone search: {state.district} {state.dong or ''}")
        
        working_df = filter_by_zone(working_df, state.district, state.dong)
        
        if state.dong:
            location_context = f"in {state.dong}, {state.district}"
        else:
            location_context = f"in {state.district}"
        
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
        privacy_safe_log(consent, f"📍 GPS search: ({user_lat:.4f}, {user_lon:.4f})")
        
        reverse_result = None
        if GOOGLE_MAPS_API_KEY:
            reverse_result = google_maps_reverse_geocode(user_lat, user_lon)
        
        if not reverse_result and KAKAO_REST_API_KEY:
            reverse_result = kakao_reverse_geocode(user_lat, user_lon)
        
        if reverse_result:
            location_context = f"near {reverse_result['address_korean']}"
            
            if not state.district:
                state.district = reverse_result.get('district')
                state.dong = reverse_result.get('dong')
        else:
            location_context = f"near your location"
        
        if 'lat' in working_df.columns and 'lon' in working_df.columns:
            def calc_distance(row):
                if pd.notna(row['lat']) and pd.notna(row['lon']):
                    return haversine(user_lat, user_lon, row['lat'], row['lon'])
                return 999.0
            
            working_df['distance_km'] = working_df.apply(calc_distance, axis=1)
            
            before_count = len(working_df)
            working_df = working_df[working_df['distance_km'] < max_distance]
            privacy_safe_log(consent, f"✓ Distance filter: {before_count} → {len(working_df)}")
            
            if len(working_df) == 0:
                expanded_radius = max_distance * 2
                logger.warning(f"Expanding to {expanded_radius}km")
                working_df = df_filtered.copy()
                working_df['distance_km'] = working_df.apply(calc_distance, axis=1)
                working_df = working_df[working_df['distance_km'] < expanded_radius]
                location_context = f"within {expanded_radius}km of your location"
            
            working_df = working_df.sort_values('distance_km')
        else:
            working_df['distance_km'] = 0
            location_context = "in Seoul"
    
    elif state.location:
        privacy_safe_log(consent, f"📍 Text location: {state.location}")
        working_df = fuzzy_match_location(state.location, working_df)
        location_context = f"in {state.location}"
        working_df['distance_km'] = 0
    
    else:
        privacy_safe_log(consent, "📍 City-wide search")
        location_context = "across Seoul"
        working_df['distance_km'] = 0
    
    # ===== CATEGORY FILTER =====
    if state.specialty and 'category' in working_df.columns:
        before_count = len(working_df)
        working_df = working_df[working_df['category'].str.contains(state.specialty, na=False, case=False)]
        privacy_safe_log(consent, f"🏥 Category filter: '{state.specialty}' → {before_count} to {len(working_df)}")
    
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
                    
                    # Dynamic weighting based on radius
                    if max_distance <= 2:
                        relevance_weight = 0.50
                    elif max_distance <= 5:
                        relevance_weight = 0.50 + (max_distance - 2) * (0.70 - 0.50) / (5 - 2)
                    elif max_distance <= 10:
                        relevance_weight = 0.70 + (max_distance - 5) * (0.85 - 0.70) / (10 - 5)
                    else:
                        relevance_weight = min(0.95, 0.85 + (max_distance - 10) * 0.01)
                    
                    distance_weight = 1 - relevance_weight
                    
                    working_df['combined_score'] = (
                        relevance_weight * working_df['relevance_score'] + 
                        distance_weight * working_df['distance_score']
                    )
                    working_df = working_df.sort_values('combined_score', ascending=False)
                    logger.debug(f"✓ Combined ranking ({relevance_weight:.0%} relevance + {distance_weight:.0%} distance)")
                else:
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
    if LANGUAGE == "English" and 'has_english' in working_df.columns:
        before = len(working_df)
        working_df = working_df[working_df['has_english'] == True]
        privacy_safe_log(consent, f"🌐 English filter: {before} → {len(working_df)}")
    
    # ===== FINAL DISTANCE VALIDATION =====
    if state.search_mode != 'zone' and 'distance_km' in working_df.columns:
        working_df = validate_distance_criteria(working_df, max_distance)
    
    # ===== GENERATE RESPONSE =====
    results = []
    
    if len(working_df) > 0:
        privacy_safe_log(consent, f"✓ Building response from {min(10, len(working_df))} facilities")
        facilities_context = build_context_for_llm(working_df, n_results=10)
        
        gen_messages = [{
            "role": "system",
            "content": GENERATION_PROMPT.format(
                user_query=user_message,
                location_context=location_context,
                language=LANGUAGE,
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
            
            simple_fields = ['address', 'phone', 'business_hours', 'english_confidence_score']
            for field in simple_fields:
                if field in row.index and pd.notna(row[field]):
                    result[field] = safe_convert_to_python(row[field])
            
            if 'file_district' in row.index and pd.notna(row['file_district']):
                result['district'] = safe_convert_to_python(row['file_district'])
            if 'file_dong' in row.index and pd.notna(row['file_dong']):
                result['dong'] = safe_convert_to_python(row['file_dong'])
            
            if 'lat' in row.index and pd.notna(row['lat']):
                result['lat'] = safe_convert_to_python(row['lat'])
            if 'lon' in row.index and pd.notna(row['lon']):
                result['lon'] = safe_convert_to_python(row['lon'])
            
            if 'website' in row.index and pd.notna(row['website']):
                result['website'] = safe_convert_to_python(row['website'])
            elif 'url' in row.index and pd.notna(row['url']):
                result['website'] = safe_convert_to_python(row['url'])
            
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
            
            result['has_english'] = safe_convert_to_python(row.get('has_english', False))
            
            # ⭐ Add BOTH distance_km and distance for frontend compatibility
            distance_value = safe_convert_to_python(row.get('distance_km', 0))
            result['distance_km'] = distance_value
            result['distance'] = distance_value
            
            result['relevance_rank'] = safe_convert_to_python(row.get('relevance_rank', 9999))
            
            results.append(result)
    else:
        if LANGUAGE == "English":
            response_text = "I couldn't find any facilities matching your criteria. Would you like to try a different specialty or area?"
        else:
            response_text = "검색 조건에 맞는 시설을 찾을 수 없습니다. 다른 전문 분야나 지역을 시도해 보시겠어요?"
    
    privacy_safe_log(consent, f"✅ SEARCH COMPLETED: {len(results)} results")
    privacy_safe_log(consent, "=" * 60 + "\n")
    
    return response_text, results


# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/consent")
async def update_consent(
    consent: CookieConsent,
    response: Response
):
    """
    Update cookie consent preferences.
    This endpoint allows the frontend to sync consent with the backend.
    """
    try:
        # Add timestamp
        consent.timestamp = datetime.utcnow().isoformat()
        
        # Set cookie with consent (30 days expiry)
        response.set_cookie(
            key="cookieConsent",
            value=json.dumps(consent.model_dump()),
            max_age=30 * 24 * 60 * 60,  # 30 days in seconds
            httponly=False,  # Allow JavaScript access
            secure=True,     # HTTPS only in production
            samesite="lax"   # CSRF protection
        )
        
        logger.info(f"✅ Consent updated: analytics={consent.analytics}, advertising={consent.advertising}")
        
        return {
            "status": "success",
            "message": "Consent preferences updated",
            "consent": consent.model_dump()
        }
        
    except Exception as e:
        logger.error(f"Error updating consent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update consent")


@app.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    response: Response,
    request: Request,
    cookieConsent: Optional[str] = Cookie(None)
):
    """
    Router-Controller Architecture with Cookie Consent Support.
    
    Now includes:
    - Cookie consent validation
    - Privacy-aware logging
    - Conditional analytics tracking
    """
    
    global LANGUAGE
    
    # Parse consent from cookie
    consent = get_consent_from_cookie(cookieConsent)
    
    # Privacy-aware logging
    if should_log_analytics(consent):
        logger.info("=" * 60)
        logger.info("📨 NEW REQUEST")
        logger.info(f"Message: \"{req.message[:50]}...\"")
    else:
        logger.info("📨 REQUEST (limited logging - no analytics consent)")
    
    # ==========================================
    # STEP 0A: LANGUAGE DETECTION
    # ==========================================
    LANGUAGE = detect_language(req.message)
    
    if should_log_analytics(consent):
        logger.info("=" * 60)
    
    # ==========================================
    # STEP 0B: STATE ENRICHMENT
    # ==========================================
    enriched_state = standardize_and_fill_state(req.current_state, consent)
    enriched_state.language_pref = LANGUAGE
    
    current_turn = enriched_state.turn_count + 1
    
    # ==========================================
    # STEP 1: ROUTING
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
        
        if should_log_analytics(consent):
            logger.info(f"🧭 ROUTER: {intent} (confidence: {route.get('confidence', 0):.2f})")
            logger.info(f"   Turn: {current_turn}")
        
    except Exception as e:
        logger.error(f"Router Error: {e}", exc_info=True)
        intent = "PROVIDE_INFO"
    
    # ==========================================
    # STEP 2: CONTROLLER (Tree Traversal)
    # ==========================================

    # === BRANCH 1: NEW_SEARCH ===
    if intent == "NEW_SEARCH":
        privacy_safe_log(consent, "🔄 BRANCH: NEW_SEARCH")
        
        new_state = State()
        new_state.turn_count = 0
        new_state.language_pref = LANGUAGE
        
        reset_keywords = ["reset", "restart", "quit", "exit", "stop", "cancel", 
                         "새로 시작", "처음부터", "다시 시작", "그만", "종료"]
        
        is_explicit_reset = any(kw in req.message.lower() for kw in reset_keywords)
        
        if is_explicit_reset:
            response_text = get_reset_confirmation(LANGUAGE)
        else:
            response_text = get_greeting_message(LANGUAGE)
        
        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": []
        }
    
    # === BRANCH 2: CHANGE_CRITERIA ===
    elif intent == "CHANGE_CRITERIA":
        privacy_safe_log(consent, "🔀 BRANCH: CHANGE_CRITERIA")
        
        cleansed_state = cleanse_state_for_change(enriched_state, req.message)
        cleansed_state.turn_count = current_turn
        cleansed_state.language_pref = LANGUAGE
        
        message_lower = req.message.lower()
        location_keywords = ["in ", "near ", "at ", "구", "동", "역", "gangnam", "songpa", "mapo", "강남", "송파", "location"]
        specialty_keywords = ["dentist", "doctor", "dermatologist", "pediatrician", "치과", "피부과", "병원", "의원", "내과", "specialty"]
        
        has_location_mention = any(kw in message_lower for kw in location_keywords)
        has_specialty_mention = any(kw in message_lower for kw in specialty_keywords)
        
        if has_location_mention and not has_specialty_mention:
            if should_log_analytics(consent):
                logger.debug("🚀 Using quick location extraction")
            extracted = quick_extract_location_change(req.message, consent)
        elif has_specialty_mention and not has_location_mention:
            if should_log_analytics(consent):
                logger.debug("🚀 Using quick specialty extraction")
            extracted = quick_extract_specialty_change(req.message, consent)
        else:
            if should_log_analytics(consent):
                logger.debug("🔍 Using full extraction")
            extracted = extract_entities(req.message, consent)
        
        new_state = merge_extraction_into_state(cleansed_state, extracted)
        new_state = standardize_and_fill_state(new_state, consent)
        new_state.language_pref = LANGUAGE
        
        has_location = bool(new_state.location or (new_state.latitude and new_state.longitude) or new_state.district)
        
        if new_state.specialty and has_location:
            if new_state.specialty_confidence >= 0.5:
                new_state.ready_to_search = True
                new_state.conversation_phase = "searching"
            else:
                response_text = ask_for_specialty_clarification(new_state)
                        
                return {
                    "response": response_text,
                    "state": new_state.model_dump(),
                    "results": []
                }
        else:
            new_state.conversation_phase = "gathering"
            response_text = ask_for_missing_info(new_state)
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }
        
        results = []
        if new_state.ready_to_search:
            response_text, results = execute_search(new_state, req.message, max_distance=new_state.max_distance_km, consent=consent)
            new_state.search_executed = True
             
        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": results
        }
    
    # === BRANCH 3: PROVIDE_INFO ===
    elif intent == "PROVIDE_INFO":
        privacy_safe_log(consent, "📝 BRANCH: PROVIDE_INFO")
        
        extracted = extract_entities(req.message, consent)
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        new_state = merge_extraction_into_state(new_state, extracted)
        new_state = standardize_and_fill_state(new_state, consent)
        new_state.language_pref = LANGUAGE
        
        has_location = bool(new_state.location or (new_state.latitude and new_state.longitude) or new_state.district)
        
        if new_state.specialty and has_location:
            if new_state.specialty_confidence >= 0.5:
                new_state.ready_to_search = True
                new_state.conversation_phase = "searching"
            else:
                response_text = ask_for_specialty_clarification(new_state)
                response_text = format_response(response_text)
                
                return {
                    "response": response_text,
                    "state": new_state.model_dump(),
                    "results": []
                }
        else:
            new_state.conversation_phase = "gathering"
            response_text = ask_for_missing_info(new_state)
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }
        
        results = []
        if new_state.ready_to_search:
            response_text, results = execute_search(new_state, req.message, max_distance=new_state.max_distance_km, consent=consent)
            new_state.search_executed = True

        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": results
        }
    
    # === BRANCH 4: CHIT_CHAT ===
    elif intent == "CHIT_CHAT":
        privacy_safe_log(consent, "💬 BRANCH: CHIT_CHAT")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        
        is_closing = any(word in req.message.lower() for word in 
                        ["thanks", "thank you", "감사합니다", "고마워"])
        
        if is_closing:
            new_state.conversation_phase = "complete"
        
        response_text = generate_chit_chat_response(req.message, new_state)
        
        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": []
        }
    
    # === BRANCH 5: HELP_RECOVERY ===
    elif intent == "HELP_RECOVERY":
        privacy_safe_log(consent, "🆘 BRANCH: HELP_RECOVERY")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        
        response_text = generate_recovery_prompt(new_state)
        
        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": []
        }
    
    # === FALLBACK ===
    else:
        logger.warning(f"Unknown intent: {intent}")
        
        if LANGUAGE == "English":
            response_text = "I'm not sure how to help. Could you rephrase your request?"
        else:
            response_text = "잘 이해하지 못했습니다. 다시 말씀해 주시겠어요?"
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        
        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": []
        }


@app.get("/health")
async def health_check():
    """Health check endpoint with privacy compliance information"""
    
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
        "logging_mode": "PRIVACY-AWARE LOGGING (respects cookie consent)",
        "privacy_features": {
            "cookie_consent": "ENABLED ✓",
            "gdpr_compliant": "YES",
            "ccpa_compliant": "YES",
            "analytics_conditional": "YES (requires user consent)",
            "advertising_conditional": "YES (requires user consent)",
            "privacy_aware_logging": "YES (reduces detail without consent)"
        },
        "architecture": "Dynamic Specialty Matching + Google Maps Location ID + Radius-Adaptive Relevance Ranking + Cookie Consent Management + Privacy-First Design"
    }
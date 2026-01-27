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
from fastapi import FastAPI, HTTPException, Cookie, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
from groq import Groq
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from datetime import datetime
import logging

# Local imports
from distance import haversine, fuzzy_match_location
from models import ChatRequest, State
from utils import safe_convert_to_python, DISTANCE_MAPPING
from prompt import (
    ROUTER_PROMPT, 
    EXTRACTION_PROMPT_V2, 
    GENERATION_PROMPT,
    FIELD_CHANGE_DETECTION_PROMPT
)
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
from cookies import (
    CookieConsent, should_log_analytics, get_consent_from_cookie, 
    should_use_advertising, privacy_safe_log, ensure_consent_object
)

# Import RAG Pipeline
from rag_pipeline import RAGPipeline

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
rag_pipeline = None  # RAG Pipeline instance
df_facilities = None  # Full dataset
df_filtered = None    # Filtered subset (Summaries not null)
available_specialties = []  # Unique specialties from parquet data


# ==========================================
# UTILITIES
# ==========================================

def print_separator(char='=', length=100):
    """Print a visual separator."""
    logger.info(char * length)

def detect_field_changes(
    current_state: State, 
    user_message: str,
    consent: Optional[CookieConsent] = None
) -> Dict[str, str]:
    """
    Use LLM to intelligently detect which fields user wants to change.
    Returns dict with keys: specialty, location, distance (values: "change" or "keep")
    """
    if not consent:
        consent = CookieConsent()
    
    detection_prompt = FIELD_CHANGE_DETECTION_PROMPT.format(
        specialty=current_state.specialty or "None",
        location=current_state.location or "None",
        district=current_state.district or "None",
        dong=current_state.dong or "None",
        max_distance_km=current_state.max_distance_km,
        search_mode=current_state.search_mode or "auto",
        user_message=user_message
    )
    
    detection_messages = [{"role": "system", "content": detection_prompt}]
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=detection_messages,
            temperature=0.0,
            max_completion_tokens=256,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        
        privacy_safe_log(consent, "🔍 Field change detection:")
        privacy_safe_log(consent, f"   Specialty: {result.get('specialty', 'keep')}")
        privacy_safe_log(consent, f"   Location: {result.get('location', 'keep')}")
        privacy_safe_log(consent, f"   Distance: {result.get('distance', 'keep')}")
        privacy_safe_log(consent, f"   Reasoning: {result.get('reasoning', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Field change detection error: {e}", exc_info=True)
        return {
            "specialty": "keep",
            "location": "keep", 
            "distance": "keep",
            "reasoning": "Error in detection, preserving all fields"
        }


def smart_cleanse_state(
    current_state: State, 
    change_detection: Dict[str, str]
) -> State:
    """
    Cleanse state based on LLM-detected field changes.
    Only clears fields that were marked as "change".
    """
    new_state = current_state.model_copy()
    
    if change_detection.get('specialty') == 'change':
        logger.debug("🧹 LLM detected: specialty change")
        new_state.specialty = None
        new_state.specialty_confidence = 0.0
    
    if change_detection.get('location') == 'change':
        logger.debug("🧹 LLM detected: location change")
        new_state.location = None
        new_state.latitude = None
        new_state.longitude = None
        new_state.address_korean = None
        new_state.district = None
        new_state.dong = None
        new_state.search_mode = None
    
    if change_detection.get('distance') == 'change':
        logger.debug("🧹 LLM detected: distance/travel preference change")
        new_state.max_distance_km = DEFAULT_MAX_DISTANCE
        new_state.travel_label = "Moderate"
    
    needs_revalidation = any(
        change_detection.get(field) == 'change' 
        for field in ['specialty', 'location']
    )
    
    if needs_revalidation:
        new_state.ready_to_search = False
        new_state.search_executed = False
        new_state.conversation_phase = "gathering"
        logger.debug("🔄 State requires revalidation due to changes")
    else:
        logger.debug("✓ No required fields changed, preserving search state")
    
    return new_state

def detect_language(message: str) -> str:
    """
    Simple character-based language detection.
    If 50%+ characters are Roman (ASCII letters), return English.
    Otherwise, return Korean.
    """
    if not message or len(message.strip()) == 0:
        return "English"
    
    roman_chars = sum(1 for char in message if char.isalpha() and ord(char) < 128)
    total_chars = len(message.replace(" ", ""))
    
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
    
    zone_keywords = ["in ", "구", "동", "district", "area", "zone", "neighborhood"]
    distance_keywords = ["near", "close", "nearby", "km", "meter", "around", "근처", "주변", "가까운"]
    
    location_lower = location_text.lower()
    
    has_zone_keyword = any(kw in location_lower for kw in zone_keywords)
    has_distance_keyword = any(kw in location_lower for kw in distance_keywords)
    
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
        
        df.to_parquet(LOCAL_PARQUET_PATH)
        logger.info(f"💾 Cached parquet locally to {LOCAL_PARQUET_PATH}")
    
    return df


# ==========================================
# LIFESPAN (STARTUP/SHUTDOWN)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, df_facilities, df_filtered, available_specialties, client
    logger.info("🚀 Booting SeoulMedBot Backend...")

    try:
        df_facilities = download_and_cache_parquet()
        logger.info(f"✅ Loaded {len(df_facilities)} facilities from parquet")
        
        df_filtered = df_facilities[df_facilities['Summaries'].notna()].copy()
        logger.info(f"✅ Filtered to {len(df_filtered)} facilities with summaries")
        
        if 'category' in df_filtered.columns:
            available_specialties = sorted(df_filtered['category'].dropna().unique().tolist())
            logger.info(f"📋 Extracted {len(available_specialties)} unique specialties from data")
            logger.info(f"   Sample: {available_specialties[:5]}")
        else:
            logger.warning("⚠️ No 'category' column found in parquet")
            available_specialties = []
        
        if 'file_district' in df_filtered.columns:
            unique_districts = df_filtered['file_district'].dropna().unique()
            logger.info(f"📍 Districts: {len(unique_districts)}")
        
        if 'lat' in df_filtered.columns and 'lon' in df_filtered.columns:
            coords_count = df_filtered[
                (df_filtered['lat'].notna()) & 
                (df_filtered['lon'].notna())
            ].shape[0]
            logger.info(f"🌐 GPS coverage: {coords_count/len(df_filtered)*100:.1f}%")
        
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
        
        logger.info("🤖 Initializing RAG Pipeline with Hybrid Search (BM25 + Vector)...")
        rag_pipeline = RAGPipeline(
            chroma_path=CHROMA_PATH,
            openai_api_key=OPENAI_API_KEY,
            groq_client=client,
            collection_name="seoul_med_v3",
            embedding_model="text-embedding-3-small"
        )
        
        rag_pipeline.initialize_collection(df_filtered, force_recreate=False)
        
        rag_stats = rag_pipeline.get_statistics()
        logger.info(f"✅ RAG Pipeline ready:")
        logger.info(f"   Vector documents: {rag_stats['document_count']}")
        logger.info(f"   BM25 documents: {rag_stats['bm25_document_count']}")
        logger.info(f"   Hybrid search: {'ENABLED ✓' if rag_stats['hybrid_search_enabled'] else 'DISABLED'}")
        
    except Exception as e:
        logger.error(f"STARTUP ERROR: {e}", exc_info=True)
        df_facilities = pd.DataFrame()
        df_filtered = pd.DataFrame()

    yield
    logger.info("🛑 Shutting down.")


# ==========================================
# FASTAPI APP SETUP
# ==========================================

app = FastAPI(lifespan=lifespan)
client = Groq(api_key=GROQ_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://seouldoc.io",
        "https://www.seouldoc.io",
        "https://seoul-doctor.vercel.app",
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
    """
    enriched_state = state.model_copy()
    
    if not consent:
        consent = CookieConsent()
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, "🔧 STATE ENRICHMENT STARTED")
    privacy_safe_log(consent, "=" * 60)
    
    filled_fields = []
    
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
    
    else:
        if state.latitude and state.longitude and state.district:
            privacy_safe_log(consent, "✅ State already complete - no enrichment needed")
        else:
            privacy_safe_log(consent, "ℹ️ Insufficient data for enrichment")
    
    if filled_fields and should_log_analytics(consent):
        logger.info("\n📋 ENRICHMENT SUMMARY:")
        logger.info(f"   Before: {_format_location_summary(state)}")
        logger.info(f"   After:  {_format_location_summary(enriched_state)}")
        logger.info(f"   Filled: {', '.join(filled_fields)}")
    
    privacy_safe_log(consent, "=" * 60 + "\n")
    
    return enriched_state


# ==========================================
# EXTRACTION FUNCTIONS
# ==========================================

def extract_entities(user_message: str, consent: Optional[CookieConsent] = None) -> Dict[str, Any]:
    """
    Call the extraction LLM with keyword extraction (hard + soft keywords).
    """
    global LANGUAGE
    
    if not consent:
        consent = CookieConsent()
    
    specialty_list = ", ".join(available_specialties[:50]) if available_specialties else "No specialties available"
    travel_labels_list = ", ".join([f'"{label}" ({dist}km)' for label, dist in DISTANCE_MAPPING.items()])
    
    extraction_prompt = EXTRACTION_PROMPT_V2.format(
        user_message=user_message,
        specialty_list=specialty_list,
        travel_labels_list=travel_labels_list
    )
    
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
        
        if extracted.get('travel_label'):
            travel_label = extracted['travel_label']
            if travel_label in DISTANCE_MAPPING:
                distance_km = DISTANCE_MAPPING[travel_label]
                privacy_safe_log(consent,
                    f"🚶 Travel preference: '{travel_label}' → {distance_km}km radius")
            else:
                extracted['travel_label'] = "Moderate"
                privacy_safe_log(consent,
                    f"⚠️ Invalid travel label, defaulting to 'Moderate' (5km)")
        else:
            extracted['travel_label'] = "Moderate"
            privacy_safe_log(consent, "ℹ️ No travel preference specified, defaulting to 'Moderate' (5km)")
        
        # ⭐ Log keyword extraction
        if extracted.get('hard_keywords'):
            privacy_safe_log(consent, f"🔒 Hard keywords (MUST): {extracted['hard_keywords']}")
        
        if extracted.get('soft_keywords'):
            privacy_safe_log(consent, f"💭 Soft keywords (preferences): {extracted['soft_keywords']}")
        
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
        return {"travel_label": "Moderate", "hard_keywords": [], "soft_keywords": []}


def quick_extract_location_change(user_message: str, consent: Optional[CookieConsent] = None) -> Dict[str, Any]:
    """SHORTER extraction for when user is just changing location OR travel distance."""
    if not consent:
        consent = CookieConsent()
    
    travel_labels_list = ", ".join([f'"{label}"' for label in DISTANCE_MAPPING.keys()])
    
    extraction_prompt = f"""Extract location AND travel preference from: "{user_message}"

Available travel labels: {travel_labels_list}

Return JSON: {{
    "location": "extracted location or null",
    "travel_label": "extracted travel preference or null"
}}
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
        
        if extracted.get('travel_label') and extracted['travel_label'] in DISTANCE_MAPPING:
            privacy_safe_log(consent, 
                f"🚶 Quick travel update: '{extracted['travel_label']}' → {DISTANCE_MAPPING[extracted['travel_label']]}km")
        
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
    """Merge extracted entities into state (including hard/soft keywords)."""
    
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

    # ⭐ Handle travel label
    if extracted.get('travel_label'):
        state.travel_label = extracted['travel_label']
        
        if state.travel_label in DISTANCE_MAPPING:
            state.max_distance_km = DISTANCE_MAPPING[state.travel_label]
            logger.info(f"📏 Distance updated: {state.travel_label} → {state.max_distance_km}km")
        else:
            logger.warning(f"⚠️ Unknown travel label '{state.travel_label}', using default 5km")
            state.travel_label = "Moderate"
            state.max_distance_km = 5.0
    
    # ⭐ NEW: Extract keywords (hard and soft)
    if extracted.get('hard_keywords'):
        existing_hard = set(state.hard_keywords)
        new_hard = set(extracted['hard_keywords'])
        state.hard_keywords = list(existing_hard.union(new_hard))
        logger.info(f"🔒 Hard keywords: {state.hard_keywords}")
    
    if extracted.get('soft_keywords'):
        existing_soft = set(state.keywords)
        new_soft = set(extracted['soft_keywords'])
        state.keywords = list(existing_soft.union(new_soft))
        logger.info(f"💭 Soft keywords: {state.keywords}")

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
    """
    HYBRID SEARCH: Path A (strict) + Path B (additive) fallback.
    
    Strategy:
    1. Path A: Try strict hard keyword filtering first (best precision)
    2. If Path A returns < 3 results: Run Path B with additive scoring to fill gap
    3. Merge results, deduplicate, ensure 3-5 total results
    
    Features:
    - Hard keywords: Strict filter (Path A), fallback to priority boost (Path B)
    - Soft keywords: Always semantic ranking
    - Specialty-preserving with radius expansion
    - Intelligent result merging without duplicates
    """
    global LANGUAGE, rag_pipeline
    
    if not consent:
        consent = CookieConsent()
    
    state.last_search_query = user_message
    state.last_search_timestamp = datetime.utcnow().isoformat()
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, "🔍 HYBRID SEARCH STARTED (Path A → Path B fallback)")
    privacy_safe_log(consent, f"Query: \"{user_message[:50]}...\"")
    privacy_safe_log(consent, f"Specialty: {state.specialty or 'Any'}")
    privacy_safe_log(consent, f"Mode: {(state.search_mode or 'auto').upper()}")
    privacy_safe_log(consent, f"Max Distance: {max_distance}km")
    privacy_safe_log(consent, f"Language: {LANGUAGE}")
    
    if state.keywords:
        privacy_safe_log(consent, f"💭 Soft Keywords: {', '.join(state.keywords)}")
    if state.hard_keywords:
        privacy_safe_log(consent, f"🔒 Hard Keywords: {', '.join(state.hard_keywords)}")
    if state.manual_search_mode:
        privacy_safe_log(consent, f"Manual Override: {state.manual_search_mode}")
    
    privacy_safe_log(consent, "=" * 60)
    
    working_df = df_filtered.copy()
    location_context = ""
    user_lat = state.latitude
    user_lon = state.longitude
    relaxed_filters = []
    
    # ===== LOCATION FILTERING (SHARED BY BOTH PATHS) =====
    
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
            privacy_safe_log(consent, f"✓ Calculated distances for {len(working_df)} zone facilities")
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
                privacy_safe_log(consent, f"✓ Reverse geocoded: {state.district} {state.dong or ''}")
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
            privacy_safe_log(consent, f"✓ Distance filter ({max_distance}km): {before_count} → {len(working_df)}")
            working_df = working_df.sort_values('distance_km')
        else:
            working_df['distance_km'] = 0
            location_context = "in Seoul"
            logger.warning("⚠️ No GPS coordinates in dataset, using city-wide search")
    
    elif state.location:
        privacy_safe_log(consent, f"📍 Text location: {state.location}")
        working_df = fuzzy_match_location(state.location, working_df)
        location_context = f"in {state.location}"
        working_df['distance_km'] = 0
        privacy_safe_log(consent, f"✓ Fuzzy location match: {len(working_df)} facilities")
    
    else:
        privacy_safe_log(consent, "📍 City-wide search (no location specified)")
        location_context = "across Seoul"
        working_df['distance_km'] = 0
    
    # ===== SPECIALTY FILTER WITH EXPANSION (SHARED) =====
    
    if state.specialty and 'category' in working_df.columns:
        before_count = len(working_df)
        working_df = working_df[working_df['category'].str.contains(state.specialty, na=False, case=False)]
        privacy_safe_log(consent, f"🏥 Category filter: '{state.specialty}' → {before_count} to {len(working_df)}")
        
        if len(working_df) < 3 and user_lat and user_lon and 'distance_km' in working_df.columns:
            logger.warning(f"⚠️ Only {len(working_df)} {state.specialty} in current area, expanding search...")
            
            expanded_df = df_filtered.copy()
            expanded_df = expanded_df[expanded_df['category'].str.contains(state.specialty, na=False, case=False)]
            
            if len(expanded_df) > 0:
                def calc_distance_expanded(row):
                    if pd.notna(row['lat']) and pd.notna(row['lon']):
                        return haversine(user_lat, user_lon, row['lat'], row['lon'])
                    return 999.0
                
                expanded_df['distance_km'] = expanded_df.apply(calc_distance_expanded, axis=1)
                expanded_df = expanded_df.sort_values('distance_km')
                
                expansion_radii = [max_distance * 1.5, max_distance * 2, max_distance * 3, max_distance * 5, 50]
                
                for expanded_radius in expansion_radii:
                    expanded_working = expanded_df[expanded_df['distance_km'] < expanded_radius].copy()
                    
                    if len(expanded_working) >= 3:
                        working_df = expanded_working
                        location_context = f"within {expanded_radius:.1f}km"
                        relaxed_filters.append(f"expanded search to {expanded_radius:.1f}km to find more {state.specialty}")
                        privacy_safe_log(consent, f"✓ Expanded to {expanded_radius:.1f}km: {len(working_df)} {state.specialty} found")
                        break
                
                if len(working_df) < 3 and len(expanded_df) > 0:
                    working_df = expanded_df.head(10)
                    location_context = f"across Seoul"
                    privacy_safe_log(consent, f"⚠️ Limited {state.specialty} availability: showing {len(working_df)} closest")
        
        if len(working_df) == 0:
            logger.warning(f"⚠️ No {state.specialty} facilities found")
            privacy_safe_log(consent, f"❌ No {state.specialty} available")
    
    # ===== PATH A: STRICT HARD KEYWORD FILTERING =====
    
    path_a_results_df = pd.DataFrame()
    path_a_place_ids = set()
    
    if state.hard_keywords and len(working_df) > 0:
        privacy_safe_log(consent, "🎯 PATH A: Strict hard keyword filtering")
        
        search_df = df_filtered.copy()
        
        # Preserve distance filter
        if 'distance_km' in working_df.columns and user_lat and user_lon:
            def calc_distance(row):
                if pd.notna(row['lat']) and pd.notna(row['lon']):
                    return haversine(user_lat, user_lon, row['lat'], row['lon'])
                return 999.0
            
            search_df['distance_km'] = search_df.apply(calc_distance, axis=1)
            search_df = search_df[search_df['distance_km'] < max_distance]
            privacy_safe_log(consent, f"   Distance preserved: {len(search_df)} facilities within {max_distance}km")
        
        # Apply strict hard keyword filter
        for keyword in state.hard_keywords:
            keyword_lower = keyword.lower()
            mask = pd.Series([False] * len(search_df), index=search_df.index)
            
            searchable_fields = ['name', 'category', 'address', 'Summaries', 'Summaries_Korean', 'Key_Highlights', 'amenities']
            
            for field in searchable_fields:
                if field not in search_df.columns:
                    continue
                
                if field in ['Summaries', 'Summaries_Korean', 'Key_Highlights']:
                    def check_list_field(val):
                        if isinstance(val, (list, np.ndarray)):
                            return any(keyword_lower in str(item).lower() for item in val)
                        return False
                    mask = mask | search_df[field].apply(check_list_field)
                
                elif field == 'amenities':
                    def check_amenities(val):
                        if isinstance(val, dict):
                            return any(keyword_lower in str(v).lower() for v in val.values())
                        return False
                    mask = mask | search_df[field].apply(check_amenities)
                
                else:
                    mask = mask | search_df[field].fillna('').astype(str).str.lower().str.contains(keyword_lower, na=False)
            
            search_df = search_df[mask]
        
        privacy_safe_log(consent, f"   Path A result: {len(search_df)} facilities match ALL hard keywords")
        
        if len(search_df) >= 3:
            path_a_results_df = search_df.copy()
            path_a_place_ids = set(path_a_results_df['place_id'].tolist())
            privacy_safe_log(consent, f"✓ Path A SUCCESS: {len(path_a_results_df)} results (≥3)")
        else:
            privacy_safe_log(consent, f"⚠️ Path A insufficient: {len(search_df)} results (<3) → Will try Path B")
            if len(search_df) > 0:
                path_a_results_df = search_df.copy()
                path_a_place_ids = set(path_a_results_df['place_id'].tolist())
    else:
        # No hard keywords, use working_df directly
        path_a_results_df = working_df.copy()
        path_a_place_ids = set(path_a_results_df['place_id'].tolist()) if len(path_a_results_df) > 0 else set()
        privacy_safe_log(consent, f"ℹ️  No hard keywords → Path A uses {len(path_a_results_df)} specialty-filtered results")
    
    # ===== PATH B: ADDITIVE SCORING (ONLY IF PATH A < 3) =====
    
    path_b_results_df = pd.DataFrame()
    needs_path_b = len(path_a_results_df) < 3
    
    if needs_path_b:
        privacy_safe_log(consent, f"🔄 PATH B: Additive scoring to fill gap to 3-5 results")
        
        # Start with full working_df (exclude Path A results)
        path_b_pool = working_df[~working_df['place_id'].isin(path_a_place_ids)].copy()
        privacy_safe_log(consent, f"   Path B pool: {len(path_b_pool)} facilities (excluding Path A results)")
        
        if len(path_b_pool) > 0:
            # Robust keyword matching function
            def calculate_field_boost(value, keyword_lower: str) -> bool:
                """
                Robust keyword matching across all field types (str, list, dict, np.ndarray, pd.Series).
                Fixes ValueError: "truth value of array with >1 element is ambiguous"
                """
                # Handle NumPy arrays explicitly (common source of the error)
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        return False
                    # Check each element in flattened array
                    for item in value.flat:
                        if not pd.isna(item) and keyword_lower in str(item).lower():
                            return True
                    return False
                
                # Handle pandas Series (sometimes appears in apply operations)
                if isinstance(value, pd.Series):
                    if value.empty:
                        return False
                    for item in value.values:
                        if not pd.isna(item) and keyword_lower in str(item).lower():
                            return True
                    return False
                
                # Handle scalar NaN/None first (MUST come before pd.isna check)
                if value is None or value == "":
                    return False
                
                # Safe scalar NaN check
                try:
                    if pd.isna(value):
                        return False
                except (ValueError, TypeError):
                    # pd.isna failed on this type, continue processing
                    pass
                
                # Handle lists/tuples
                if isinstance(value, (list, tuple)):
                    for item in value:
                        try:
                            if pd.isna(item):
                                continue
                        except (ValueError, TypeError):
                            pass
                        if keyword_lower in str(item).lower():
                            return True
                    return False
                
                # Handle dicts (amenities, medical_info_parsed)
                if isinstance(value, dict):
                    for v in value.values():
                        # ⭐ CRITICAL FIX: Recursively handle complex dict values (arrays, nested dicts)
                        if isinstance(v, (np.ndarray, list, tuple, dict, pd.Series)):
                            if calculate_field_boost(v, keyword_lower):
                                return True
                        else:
                            # Scalar value in dict
                            try:
                                if pd.isna(v):
                                    continue
                            except (ValueError, TypeError):
                                pass
                            if keyword_lower in str(v).lower():
                                return True
                    return False
                
                # Handle strings/scalars
                return keyword_lower in str(value).lower()

            
            def apply_keyword_boost(df: pd.DataFrame, keywords: List[str], weight: float, label: str) -> pd.Series:
                boost = pd.Series(0.0, index=df.index)
                searchable_fields = ['name', 'category', 'address', 'Summaries', 'Summaries_Korean', 'Key_Highlights', 'amenities', 'medical_info_parsed']
                
                for kw in keywords:
                    kw_lower = str(kw).lower().strip()
                    if not kw_lower:
                        continue
                    
                    field_matches = pd.Series(False, index=df.index)
                    for field in searchable_fields:
                        if field in df.columns:
                            field_matches |= df[field].apply(lambda x, kw=kw_lower: calculate_field_boost(x, kw))
                    
                    boost[field_matches] += weight
                    matched_count = field_matches.sum()
                    if matched_count > 0:
                        privacy_safe_log(consent, f"      +{weight} → '{kw}' matched {matched_count} facilities ({label})")
                
                return boost
            
            # Initialize scoring
            path_b_pool['relevance_boost'] = 0.0
            
            # 1. Hard keywords: Priority boost
            if state.hard_keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(path_b_pool, state.hard_keywords, 500.0, "HARD")
            
            # 2. Soft keywords: Preference boost
            if state.keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(path_b_pool, state.keywords, 50.0, "SOFT")
            
            # 3. Specialty match boost
            if state.specialty:
                spec_mask = path_b_pool['category'].fillna('').astype(str).str.lower().str.contains(state.specialty.lower(), na=False)
                path_b_pool.loc[spec_mask, 'relevance_boost'] += 200.0
            
            # 4. Distance decay
            if 'distance_km' in path_b_pool.columns:
                max_dist = path_b_pool['distance_km'].max() or 1.0
                if max_dist > 0:
                    distance_penalty = (path_b_pool['distance_km'] / max_dist) * 30
                    path_b_pool['relevance_boost'] -= distance_penalty
            
            # Sort by boost then distance
            path_b_pool = path_b_pool.sort_values(by=['relevance_boost', 'distance_km'], ascending=[False, True]).reset_index(drop=True)
            
            path_b_results_df = path_b_pool.copy()
            privacy_safe_log(consent, f"✓ Path B scored {len(path_b_results_df)} facilities")
    
    # ===== MERGE PATH A + PATH B RESULTS =====
    
    if len(path_a_results_df) >= 3:
        # Path A sufficient
        final_df = path_a_results_df.copy()
        privacy_safe_log(consent, f"✅ Using Path A only: {len(final_df)} results")
    
    elif len(path_a_results_df) > 0 and len(path_b_results_df) > 0:
        # Merge: Path A first, then Path B to fill gap
        needed = 5 - len(path_a_results_df)
        path_b_top = path_b_results_df.head(needed)
        
        final_df = pd.concat([path_a_results_df, path_b_top], ignore_index=True)
        privacy_safe_log(consent, f"✅ MERGED: {len(path_a_results_df)} from Path A + {len(path_b_top)} from Path B = {len(final_df)} total")
        relaxed_filters.append(f"included {len(path_b_top)} additional results using priority scoring")
    
    elif len(path_b_results_df) > 0:
        # Only Path B (Path A failed completely)
        final_df = path_b_results_df.copy()
        privacy_safe_log(consent, f"✅ Using Path B only: {len(final_df)} results")
        relaxed_filters.append("used priority scoring (strict filtering found no matches)")
    
    else:
        # Both failed
        final_df = working_df.copy()
        privacy_safe_log(consent, f"⚠️ Both paths failed, using {len(final_df)} base results")
    
    # ===== HYBRID RAG SEMANTIC RANKING =====
    
    query_components = []
    if state.specialty:
        query_components.append(state.specialty)
    if state.keywords:
        query_components.extend(state.keywords)
    
    query_text = " ".join(query_components) if query_components else user_message
    
    if len(query_text.strip()) > 2 and len(final_df) > 0 and rag_pipeline:
        privacy_safe_log(consent, "🔀 Applying Hybrid RAG (BM25 + Vector)...")
        
        try:
            route_decision = rag_pipeline.route_query(query_text)
            state.query_intent = route_decision.get('intent', 'MIXED')
            state.suggested_alpha = route_decision.get('suggested_alpha', 0.7)
            
            actual_alpha = rag_pipeline.calculate_alpha(route_decision, state.manual_search_mode)
            state.hybrid_alpha = actual_alpha
            
            privacy_safe_log(consent, f"   🎯 Router: {state.query_intent} (α={actual_alpha:.2f})")
            
            final_df = rag_pipeline.apply_combined_ranking(
                df=final_df,
                query_text=query_text,
                max_distance=max_distance,
                search_mode=state.search_mode or 'distance',
                n_results=200,
                use_hybrid=True,
                manual_mode=state.manual_search_mode
            )
            
            privacy_safe_log(consent, f"✓ Hybrid ranking applied")
            
        except Exception as e:
            logger.error(f"Hybrid ranking error: {e}", exc_info=True)
            final_df['relevance_rank'] = 9999
            if 'distance_km' in final_df.columns:
                final_df = final_df.sort_values('distance_km')
    else:
        state.query_intent = None
        state.suggested_alpha = None
        state.hybrid_alpha = None
    
    # ===== ENGLISH FILTER (with fallback) =====
    
    pre_english_df = final_df.copy()
    
    if LANGUAGE == "English" and 'has_english' in final_df.columns:
        before = len(final_df)
        final_df = final_df[final_df['has_english'] == True]
        privacy_safe_log(consent, f"🌐 English filter: {before} → {len(final_df)}")
        
        if len(final_df) < 3:
            final_df = pre_english_df
            relaxed_filters.append("included facilities with limited English")
    
    # ===== FINAL DISTANCE VALIDATION =====
    
    if state.search_mode != 'zone' and 'distance_km' in final_df.columns:
        pre_validation_df = final_df.copy()
        final_df = final_df[final_df['distance_km'] < max_distance]
        
        if len(final_df) < 3 and len(pre_validation_df) >= 3:
            final_df = pre_validation_df.head(5)
            relaxed_filters.append("included slightly farther facilities")
    
    # ===== RESULT SELECTION (3-5) =====
    
    def select_optimal_result_count(df: pd.DataFrame, min_results: int = 3, max_results: int = 5) -> int:
        total = len(df)
        if total < min_results:
            return total
        elif total <= max_results:
            return total
        elif total <= 10:
            return 4
        else:
            return max_results
    
    # ===== GENERATE RESPONSE =====
    
    results = []
    response_text = ""
    
    if len(final_df) > 0:
        n_results = select_optimal_result_count(final_df, min_results=3, max_results=5)
        privacy_safe_log(consent, f"✓ Selecting {n_results} results from {len(final_df)} facilities")
        
        try:
            facilities_context = rag_pipeline.build_context_for_llm(final_df, n_results=n_results, language=LANGUAGE)
        except Exception as e:
            logger.error(f"Context building error: {e}", exc_info=True)
            facilities_context = "Error building context"
        
        gen_messages = [{
            "role": "system",
            "content": GENERATION_PROMPT.format(
                user_query=user_message,
                location_context=location_context,
                language=LANGUAGE,
                facilities_context=facilities_context
            )
        }]
        
        try:
            gen_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=gen_messages,
                temperature=1.0,
                max_completion_tokens=1024
            )
            response_text = gen_completion.choices[0].message.content
            
            # Add disclaimers
            if len(relaxed_filters) > 0 and n_results >= 3:
                if LANGUAGE == "English":
                    if any("expanded search" in f for f in relaxed_filters):
                        disclaimer = f"\n\n💡 Note: I expanded the search area to find more options for you."
                    else:
                        disclaimer = "\n\n💡 Note: To show you 3+ options, I " + " and ".join(relaxed_filters) + "."
                else:
                    disclaimer = "\n\n💡 참고: 더 많은 옵션을 보여드리기 위해 검색 범위를 확대했습니다."
                response_text += disclaimer
            
            elif n_results < 3:
                if LANGUAGE == "English":
                    response_text += f"\n\n⚠️ Limited availability: Only {n_results} facilities found. Consider expanding your search area."
                else:
                    response_text += f"\n\n⚠️ 제한된 옵션: {n_results}개만 찾았습니다. 검색 범위를 확대해 보세요."
            
        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            if LANGUAGE == "English":
                response_text = f"I found {len(final_df)} facilities. Here are the top {n_results}:"
            else:
                response_text = f"{len(final_df)}개의 시설을 찾았습니다. 상위 {n_results}개:"
        
        # ===== PREPARE RESULTS FOR FRONTEND =====
        
        for idx, (_, row) in enumerate(final_df.head(n_results).iterrows(), start=1):
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
            
            distance_value = safe_convert_to_python(row.get('distance_km', 0))
            result['distance_km'] = distance_value
            result['distance'] = distance_value
            
            result['relevance_rank'] = safe_convert_to_python(row.get('relevance_rank', 9999))
            
            if 'relevance_boost' in row.index:
                result['relevance_boost'] = safe_convert_to_python(row['relevance_boost'])
            
            if 'combined_score' in row.index:
                result['combined_score'] = safe_convert_to_python(row['combined_score'])
            
            results.append(result)
            
            boost_value = result.get('relevance_boost')
            boost_info = f" boost={boost_value:.1f}" if boost_value is not None else ""
            privacy_safe_log(consent, f"   #{idx}: {result['name']} ({result['category']}) dist={distance_value:.1f}km{boost_info}")
    
    else:
        logger.warning("⚠️ No facilities found")
        
        if state.specialty:
            if LANGUAGE == "English":
                response_text = (
                    f"I couldn't find any {state.specialty} facilities in your search area.\n\n"
                    f"Try expanding your search radius or choosing a different area."
                )
            else:
                response_text = f"검색 지역에서 {state.specialty} 시설을 찾을 수 없습니다.\n\n검색 범위를 확대해 보세요."
        else:
            if LANGUAGE == "English":
                response_text = "I couldn't find any facilities matching your criteria.\n\nTry adjusting your search."
            else:
                response_text = "조건에 맞는 시설을 찾을 수 없습니다.\n\n검색 조건을 조정해 보세요."
    
    # ===== UPDATE STATE METADATA =====
    
    state.last_results_count = len(results)
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, f"✅ HYBRID SEARCH COMPLETED")
    privacy_safe_log(consent, f"   Path A results: {len(path_a_results_df)}")
    privacy_safe_log(consent, f"   Path B results: {len(path_b_results_df)}")
    privacy_safe_log(consent, f"   Final returned: {len(results)}")
    if state.query_intent:
        privacy_safe_log(consent, f"   Query intent: {state.query_intent} (α={state.hybrid_alpha:.2f})")
    if relaxed_filters:
        privacy_safe_log(consent, f"   Adjustments: {', '.join(relaxed_filters)}")
    if len(results) < 3:
        privacy_safe_log(consent, f"   ⚠️ WARNING: Only {len(results)} results")
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
    """Update cookie consent preferences."""
    try:
        consent.timestamp = datetime.utcnow().isoformat()
        
        response.set_cookie(
            key="cookieConsent",
            value=json.dumps(consent.model_dump()),
            max_age=30 * 24 * 60 * 60,
            httponly=False,
            secure=True,
            samesite="lax"
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
    Router-Controller Architecture with Hybrid RAG and keyword filtering.
    """
    
    global LANGUAGE
    
    consent = get_consent_from_cookie(cookieConsent)
    consent = ensure_consent_object(consent)

    if should_log_analytics(consent):
        logger.info("=" * 60)
        logger.info("📨 NEW REQUEST")
        logger.info(f"Message: \"{req.message[:50]}...\"")
    else:
        logger.info("📨 REQUEST (limited logging - no analytics consent)")
    
    LANGUAGE = detect_language(req.message)
    
    if should_log_analytics(consent):
        logger.info("=" * 60)
    
    enriched_state = standardize_and_fill_state(req.current_state, consent)
    enriched_state.language_pref = LANGUAGE
    
    current_turn = enriched_state.turn_count + 1
    
    # ===== ROUTING =====
    
    router_messages = [{
        "role": "system",
        "content": ROUTER_PROMPT.format(
            specialty=enriched_state.specialty or "None",
            specialty_confidence=enriched_state.specialty_confidence,
            location=enriched_state.location or "None",
            ready_to_search=enriched_state.ready_to_search,
            turn_count=current_turn,
            search_executed=enriched_state.search_executed,
            conversation_phase=enriched_state.conversation_phase or "gathering",
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
            logger.info(f"   Reasoning: {route.get('reasoning', 'N/A')[:100]}")
            logger.info(f"   Turn: {current_turn}")
        
    except Exception as e:
        logger.error(f"Router Error: {e}", exc_info=True)
        intent = "PROVIDE_INFO"
    
    # ===== CONTROLLER =====
    
    # CONFIRMATION
    if intent == "CONFIRMATION":
        privacy_safe_log(consent, "✅ BRANCH: CONFIRMATION")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        
        if new_state.specialty and new_state.specialty_confidence < 1.0:
            old_confidence = new_state.specialty_confidence
            new_state.specialty_confidence = 1.0
            privacy_safe_log(consent, f"📈 Confidence boosted: {old_confidence:.2f} → 1.0")
        
        has_location = bool(new_state.location or (new_state.latitude and new_state.longitude) or new_state.district)
        
        if new_state.specialty and has_location:
            new_state.ready_to_search = True
            new_state.conversation_phase = "searching"
            
            response_text, results = execute_search(
                new_state, 
                req.message, 
                max_distance=new_state.max_distance_km, 
                consent=consent
            )
            new_state.search_executed = True
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": results
            }
        else:
            new_state.conversation_phase = "gathering"
            response_text = ask_for_missing_info(new_state)
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }

    # NEW_SEARCH
    elif intent == "NEW_SEARCH":
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

    # CHANGE_CRITERIA
    elif intent == "CHANGE_CRITERIA":
        privacy_safe_log(consent, "🔀 BRANCH: CHANGE_CRITERIA")
        
        message_lower = req.message.lower()
        
        is_negation = message_lower.startswith("no ") or message_lower.startswith("not ") or message_lower.startswith("아니")
        
        if is_negation:
            privacy_safe_log(consent, "⛔ Negation detected - user correcting previous info")
            
            extracted = extract_entities(req.message, consent)
            
            new_state = enriched_state.model_copy()
            new_state.turn_count = current_turn
            new_state.language_pref = LANGUAGE
            
            if extracted.get('specialty'):
                new_state.specialty = extracted['specialty']
                new_state.specialty_confidence = extracted.get('specialty_confidence', 0.9)
                privacy_safe_log(consent, f"✓ Corrected specialty: {new_state.specialty}")
            
            if extracted.get('location'):
                new_state = merge_extraction_into_state(new_state, extracted)
                new_state = standardize_and_fill_state(new_state, consent)
                privacy_safe_log(consent, f"✓ Updated location: {new_state.district or new_state.location}")
            
            if extracted.get('travel_label'):
                new_state.travel_label = extracted['travel_label']
                if new_state.travel_label in DISTANCE_MAPPING:
                    new_state.max_distance_km = DISTANCE_MAPPING[new_state.travel_label]
            
            has_location = bool(new_state.location or (new_state.latitude and new_state.longitude) or new_state.district)
            
            if new_state.specialty and has_location:
                if new_state.specialty_confidence >= 0.5:
                    new_state.ready_to_search = True
                    new_state.conversation_phase = "searching"
                    
                    response_text, results = execute_search(
                        new_state,
                        req.message,
                        max_distance=new_state.max_distance_km,
                        consent=consent
                    )
                    new_state.search_executed = True
                    
                    return {
                        "response": response_text,
                        "state": new_state.model_dump(),
                        "results": results
                    }
                else:
                    response_text = ask_for_specialty_clarification(new_state)
                    return {
                        "response": response_text,
                        "state": new_state.model_dump(),
                        "results": []
                    }
            
            new_state.conversation_phase = "gathering"
            response_text = ask_for_missing_info(new_state)
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }
        
        is_city_wide_request = any(phrase in message_lower for phrase in [
            "city wide", "citywide", "city-wide", "all seoul", "entire seoul",
            "서울 전체", "서울전체", "서울 어디든", "어디든지"
        ])
        
        if is_city_wide_request:
            privacy_safe_log(consent, "🌆 City-wide search requested")
            
            new_state = enriched_state.model_copy()
            new_state.turn_count = current_turn
            new_state.language_pref = LANGUAGE
            
            new_state.search_mode = 'distance'
            new_state.max_distance_km = 25.0
            new_state.travel_label = "Anywhere in Seoul"
            
            privacy_safe_log(consent, f"✓ Expanded to city-wide: {new_state.max_distance_km}km")
            
            if new_state.specialty and new_state.specialty_confidence >= 0.5:
                new_state.ready_to_search = True
                new_state.conversation_phase = "searching"
                
                response_text, results = execute_search(
                    new_state,
                    req.message,
                    max_distance=new_state.max_distance_km,
                    consent=consent
                )
                new_state.search_executed = True
                
                return {
                    "response": response_text,
                    "state": new_state.model_dump(),
                    "results": results
                }
            else:
                response_text = ask_for_missing_info(new_state)
                return {
                    "response": response_text,
                    "state": new_state.model_dump(),
                    "results": []
                }
        
        privacy_safe_log(consent, "🤖 Detecting which fields to change...")
        change_detection = detect_field_changes(enriched_state, req.message, consent)
        
        cleansed_state = smart_cleanse_state(enriched_state, change_detection)
        cleansed_state.turn_count = current_turn
        cleansed_state.language_pref = LANGUAGE
        
        specialty_will_change = change_detection.get('specialty') == 'change'
        location_will_change = change_detection.get('location') == 'change'
        
        if location_will_change and not specialty_will_change:
            privacy_safe_log(consent, "🚀 Quick location extraction")
            extracted = quick_extract_location_change(req.message, consent)
        elif specialty_will_change and not location_will_change:
            privacy_safe_log(consent, "🚀 Quick specialty extraction")
            extracted = quick_extract_specialty_change(req.message, consent)
        else:
            privacy_safe_log(consent, "🔍 Full extraction")
            extracted = extract_entities(req.message, consent)
        
        new_state = merge_extraction_into_state(cleansed_state, extracted)
        
        if change_detection.get('specialty') == 'keep' and enriched_state.specialty:
            new_state.specialty = enriched_state.specialty
            new_state.specialty_confidence = enriched_state.specialty_confidence
            privacy_safe_log(consent, f"✓ Preserved specialty: {new_state.specialty}")
        
        if change_detection.get('location') == 'keep':
            if enriched_state.location:
                new_state.location = enriched_state.location
            if enriched_state.latitude:
                new_state.latitude = enriched_state.latitude
                new_state.longitude = enriched_state.longitude
            if enriched_state.district:
                new_state.district = enriched_state.district
                new_state.dong = enriched_state.dong
            if enriched_state.address_korean:
                new_state.address_korean = enriched_state.address_korean
            privacy_safe_log(consent, f"✓ Preserved location: {new_state.district or new_state.location}")
        
        if change_detection.get('distance') == 'keep':
            new_state.max_distance_km = enriched_state.max_distance_km
            new_state.travel_label = enriched_state.travel_label
            privacy_safe_log(consent, f"✓ Preserved distance: {new_state.max_distance_km}km")
        
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
            response_text, results = execute_search(
                new_state, 
                req.message, 
                max_distance=new_state.max_distance_km, 
                consent=consent
            )
            new_state.search_executed = True
            
        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": results
        }

    # PROVIDE_INFO
    elif intent == "PROVIDE_INFO":
        privacy_safe_log(consent, "📝 BRANCH: PROVIDE_INFO")
        
        extracted = extract_entities(req.message, consent)
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        
        if not extracted.get('specialty') and enriched_state.specialty:
            extracted['specialty'] = enriched_state.specialty
            extracted['specialty_confidence'] = enriched_state.specialty_confidence
            privacy_safe_log(consent, f"✓ Preserved existing specialty: {enriched_state.specialty}")
        
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
            response_text, results = execute_search(
                new_state, 
                req.message, 
                max_distance=new_state.max_distance_km, 
                consent=consent
            )
            new_state.search_executed = True

        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": results
        }
    
    # CHIT_CHAT
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
    
    # HELP_RECOVERY
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
    
    # FALLBACK
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
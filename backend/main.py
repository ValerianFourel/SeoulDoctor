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
import re
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
from utils import (
    safe_convert_to_python, DISTANCE_MAPPING, clean_llm_response,
    standardize_and_fill_state,detect_search_mode,detect_language,
    smart_cleanse_state,detect_field_changes,print_separator,
    has_vague_medical_term,user_wants_any_specialty, 
    validate_distance_criteria,download_and_cache_parquet, DEFAULT_MAX_DISTANCE
)
from prompt import (
    ROUTER_PROMPT, 
    EXTRACTION_PROMPT_V2, 
    GENERATION_PROMPT,
    SPECIALTY_MAPPING
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

CHROMA_PATH = "./chroma_db"

# Defaults
DEFAULT_LAT = 37.5219  # Yeouido
DEFAULT_LON = 126.9243

# Global state
LANGUAGE = "English"  # Semi-fixed fixture, updated per message

# Global data structures
rag_pipeline = None  # RAG Pipeline instance
df_facilities = None  # Full dataset
df_filtered = None    # Filtered subset (Summaries not null)
available_specialties = []  # Unique specialties from parquet data


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def ensure_city_wide_defaults(state: State, consent: CookieConsent) -> State:
    """
    If no location specified, default to city-wide Seoul search.
    Prevents falling back to 중구 or other arbitrary defaults.
    """
    if not state.location and not state.latitude and not state.district:
        privacy_safe_log(consent, "🌆 No location specified → defaulting to city-wide Seoul")
        state.location = "Seoul"
        state.latitude = 37.5665  # Seoul City Hall
        state.longitude = 126.9780
        state.max_distance_km = 25.0
        state.search_mode = 'distance'
        state.travel_label = "Anywhere in Seoul"
        state.travel_confidence = 1.0  # High confidence for default
    
    return state


def keyword_matches_word_boundary(keyword_lower: str, text: str) -> bool:
    """
    ⭐ FIX: Word boundary matching instead of substring.
    Prevents "unprofessional" matching "professional".
    
    Returns True if keyword appears as a complete word in text.
    """
    if not text or pd.isna(text):
        return False
    
    text_lower = str(text).lower()
    
    # Create word boundary pattern
    # \b matches word boundaries (spaces, punctuation, start/end)
    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
    
    return bool(re.search(pattern, text_lower))


def calculate_field_boost(value, keyword_lower: str) -> bool:
    """
    ⭐ FIXED: Robust keyword matching with word boundaries.
    Handles all field types (str, list, dict, np.ndarray, pd.Series).
    """
    # Handle NumPy arrays
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        for item in value.flat:
            if not pd.isna(item) and keyword_matches_word_boundary(keyword_lower, str(item)):
                return True
        return False
    
    # Handle pandas Series
    if isinstance(value, pd.Series):
        if value.empty:
            return False
        for item in value.values:
            if not pd.isna(item) and keyword_matches_word_boundary(keyword_lower, str(item)):
                return True
        return False
    
    # Handle scalar NaN/None
    if value is None or value == "":
        return False
    
    try:
        if pd.isna(value):
            return False
    except (ValueError, TypeError):
        pass
    
    # Handle lists/tuples
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                if pd.isna(item):
                    continue
            except (ValueError, TypeError):
                pass
            if keyword_matches_word_boundary(keyword_lower, str(item)):
                return True
        return False
    
    # Handle dicts
    if isinstance(value, dict):
        for v in value.values():
            # Recursively handle complex dict values
            if isinstance(v, (np.ndarray, list, tuple, dict, pd.Series)):
                if calculate_field_boost(v, keyword_lower):
                    return True
            else:
                try:
                    if pd.isna(v):
                        continue
                except (ValueError, TypeError):
                    pass
                if keyword_matches_word_boundary(keyword_lower, str(v)):
                    return True
        return False
    
    # Handle strings/scalars
    return keyword_matches_word_boundary(keyword_lower, str(value))


# ==========================================
# LIFESPAN (STARTUP/SHUTDOWN)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, df_facilities, df_filtered, available_specialties, client
    logger.info("🚀 Booting Seoul Med Match Backend...")

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
# EXTRACTION FUNCTIONS
# ==========================================

def extract_entities(user_message: str, consent: Optional[CookieConsent] = None) -> Dict[str, Any]:
    """
    Call the extraction LLM with keyword extraction (hard + soft + negative keywords).
    """
    global LANGUAGE
    
    if not consent:
        consent = CookieConsent()
    
    specialty_list = ", ".join(available_specialties[:50]) if available_specialties else "No specialties available"
    travel_labels_list = ", ".join([f'"{label}" ({dist}km)' for label, dist in DISTANCE_MAPPING.items()])
    
    extraction_prompt = EXTRACTION_PROMPT_V2.format(
        SPECIALTY_MAPPING=SPECIALTY_MAPPING,
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
        
        # Log keyword extraction
        if extracted.get('hard_keywords'):
            privacy_safe_log(consent, f"🔒 Hard keywords (MUST): {extracted['hard_keywords']}")
        
        if extracted.get('soft_keywords'):
            privacy_safe_log(consent, f"💭 Soft keywords (preferences): {extracted['soft_keywords']}")
        
        if extracted.get('negative_hard_keywords'):
            privacy_safe_log(consent, f"⛔ Negative hard keywords (EXCLUDE): {extracted['negative_hard_keywords']}")
        
        if extracted.get('negative_keywords'):
            privacy_safe_log(consent, f"🚫 Negative keywords (avoid): {extracted['negative_keywords']}")
        
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
        return {
            "travel_label": "Moderate", 
            "hard_keywords": [], 
            "soft_keywords": [],
            "negative_hard_keywords": [],
            "negative_keywords": []
        }


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


def merge_extraction_into_state(state: State, extracted: Dict[str, Any], replace_keywords: bool = True) -> State:
    """
    Merge extracted entities into state (with keyword replacement option).
    
    Args:
        state: Current state
        extracted: Extracted entities from LLM
        replace_keywords: If True, replace keywords entirely. If False, add to existing.
    """
    
    # ===== SPECIALTY =====
    if extracted.get('specialty'):
        state.specialty = extracted['specialty']
        state.specialty_confidence = extracted.get('specialty_confidence', 0.7)
    
    # ===== LOCATION =====
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

    # ===== TRAVEL LABEL & CONFIDENCE =====
    if extracted.get('travel_label'):
        state.travel_label = extracted['travel_label']
        
        # ⭐ NEW: Set confidence based on how it was provided
        if extracted.get('travel_confidence'):
            # Explicitly set (from widget = 1.0)
            state.travel_confidence = extracted['travel_confidence']
        else:
            # From text extraction = 0.6
            state.travel_confidence = 0.6
        
        if state.travel_label in DISTANCE_MAPPING:
            state.max_distance_km = DISTANCE_MAPPING[state.travel_label]
            logger.info(f"📏 Distance updated: {state.travel_label} → {state.max_distance_km}km (confidence: {state.travel_confidence})")
        else:
            logger.warning(f"⚠️ Unknown travel label '{state.travel_label}', using default 5km")
            state.travel_label = "Moderate"
            state.max_distance_km = 5.0
            state.travel_confidence = 0.5
    
    # ===== KEYWORDS (POSITIVE) =====
    if replace_keywords:
        # REPLACEMENT MODE (default)
        if extracted.get('hard_keywords') is not None:
            state.hard_keywords = extracted['hard_keywords']
            logger.info(f"🔄 Replaced hard keywords: {state.hard_keywords}")
        
        if extracted.get('soft_keywords') is not None:
            state.keywords = extracted['soft_keywords']
            logger.info(f"🔄 Replaced soft keywords: {state.keywords}")
    else:
        # ADDITIVE MODE (only when explicitly refining)
        if extracted.get('hard_keywords'):
            existing_hard = set(state.hard_keywords)
            new_hard = set(extracted['hard_keywords'])
            state.hard_keywords = list(existing_hard.union(new_hard))
            logger.info(f"➕ Added hard keywords: {state.hard_keywords}")
        
        if extracted.get('soft_keywords'):
            existing_soft = set(state.keywords)
            new_soft = set(extracted['soft_keywords'])
            state.keywords = list(existing_soft.union(new_soft))
            logger.info(f"➕ Added soft keywords: {state.keywords}")
    
    # ===== NEGATIVE KEYWORDS =====
    if replace_keywords:
        if extracted.get('negative_hard_keywords') is not None:
            state.negative_hard_keywords = extracted['negative_hard_keywords']
            logger.info(f"🚫 Replaced negative hard keywords: {state.negative_hard_keywords}")
        
        if extracted.get('negative_keywords') is not None:
            state.negative_keywords = extracted['negative_keywords']
            logger.info(f"🚫 Replaced negative keywords: {state.negative_keywords}")
    else:
        if extracted.get('negative_hard_keywords'):
            existing_neg_hard = set(state.negative_hard_keywords)
            new_neg_hard = set(extracted['negative_hard_keywords'])
            state.negative_hard_keywords = list(existing_neg_hard.union(new_neg_hard))
            logger.info(f"➕ Added negative hard keywords: {state.negative_hard_keywords}")
        
        if extracted.get('negative_keywords'):
            existing_neg = set(state.negative_keywords)
            new_neg = set(extracted['negative_keywords'])
            state.negative_keywords = list(existing_neg.union(new_neg))
            logger.info(f"➕ Added negative keywords: {state.negative_keywords}")

    # ===== SEARCH MODE =====
    if state.location:
        state.search_mode = detect_search_mode(state.location, state)
    
    return state


# ==========================================
# SEARCH FUNCTIONS
# ==========================================

def execute_emergency_search(
    state: State, 
    user_message: str,
    consent: Optional[CookieConsent] = None
) -> Tuple[str, List[Dict]]:
    """
    EMERGENCY MODE: Find the 3 closest emergency-capable facilities.
    Returns facilities from: 응급실, 종합병원, 국립병원, 시립,도립병원
    Sorted purely by distance - no priority, no summary requirements.
    """
    global LANGUAGE, df_facilities
    
    if not consent:
        consent = CookieConsent()
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, "🚨 EMERGENCY SEARCH ACTIVATED")
    privacy_safe_log(consent, "=" * 60)
    
    # Filter for emergency-capable facilities
    emergency_categories = ['응급실', '종합병원', '국립병원', '시립,도립병원']
    emergency_df = df_facilities[
        df_facilities['category'].isin(emergency_categories)
    ].copy()
    
    privacy_safe_log(consent, f"🏥 Found {len(emergency_df)} emergency-capable facilities")
    
    if len(emergency_df) == 0:
        logger.error("❌ CRITICAL: No emergency facilities in database!")
        
        if LANGUAGE == "English":
            response_text = (
                "🚨 **CALL 119 IMMEDIATELY!**\n\n"
                "I couldn't find emergency facility data.\n"
                "Please call 119 (Korea's emergency number) right now."
            )
        else:
            response_text = (
                "🚨 **지금 바로 119에 전화하세요!**\n\n"
                "응급 시설 데이터를 찾을 수 없습니다.\n"
                "한국 응급 전화번호 119로 즉시 연락하세요."
            )
        
        return response_text, []
    
    # Check user location
    user_lat = state.latitude
    user_lon = state.longitude
    
    if not (user_lat and user_lon):
        privacy_safe_log(consent, "⚠️ No location - requesting from user")
        
        if LANGUAGE == "English":
            response_text = (
                "🚨 **MEDICAL EMERGENCY**\n\n"
                "1️⃣ **CALL 119 NOW** (Korea's emergency number)\n\n"
                "2️⃣ Share your location:\n"
                "   • Click the location button\n"
                "   • Or type your address\n\n"
                "⚠️ **If life-threatening: CALL 119 FIRST**"
            )
        else:
            response_text = (
                "🚨 **응급 상황**\n\n"
                "1️⃣ **지금 바로 119에 전화하세요**\n\n"
                "2️⃣ 위치를 공유해 주세요:\n"
                "   • 아래 위치 버튼 클릭\n"
                "   • 또는 주소 입력\n\n"
                "⚠️ **생명 위급 시: 먼저 119 전화**"
            )
        
        return response_text, []
    
    # Calculate distances
    privacy_safe_log(consent, f"📍 User location: ({user_lat:.4f}, {user_lon:.4f})")
    
    def calc_distance(row):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            return haversine(user_lat, user_lon, row['lat'], row['lon'])
        return 999.0
    
    emergency_df['distance_km'] = emergency_df.apply(calc_distance, axis=1)
    
    # Sort by distance only
    emergency_df = emergency_df.sort_values('distance_km').reset_index(drop=True)
    
    # Filter out facilities with no GPS
    emergency_df = emergency_df[emergency_df['distance_km'] < 999.0]
    
    if len(emergency_df) == 0:
        logger.error("❌ No emergency facilities with GPS!")
        
        if LANGUAGE == "English":
            response_text = "🚨 **CALL 119 IMMEDIATELY!**\n\nNo GPS data available for emergency facilities."
        else:
            response_text = "🚨 **지금 바로 119에 전화하세요!**\n\n응급 시설 GPS 데이터 없음."
        
        return response_text, []
    
    # Get top 3 closest
    top_facilities = emergency_df.head(3)
    
    privacy_safe_log(consent, f"✅ Found {len(top_facilities)} closest emergency facilities")
    
    # Build results
    results = []
    for idx, (_, row) in enumerate(top_facilities.iterrows(), start=1):
        result = {
            "place_id": safe_convert_to_python(row.get('place_id', '')),
            "name": safe_convert_to_python(row['name']),
            "category": safe_convert_to_python(row['category']),
            "distance_km": safe_convert_to_python(row['distance_km']),
            "distance": safe_convert_to_python(row['distance_km']),
            "is_emergency": True,
        }
        
        # Add basic info (all optional)
        optional_fields = {
            'address': 'address',
            'phone': 'phone',
            'business_hours': 'business_hours',
            'website': 'website',
            'url': 'website',
            'file_district': 'district',
            'lat': 'lat',
            'lon': 'lon'
        }
        
        for src_field, dest_field in optional_fields.items():
            if src_field in row.index and pd.notna(row[src_field]):
                value = safe_convert_to_python(row[src_field])
                if value and (dest_field not in result or dest_field == 'website'):
                    result[dest_field] = value
        
        results.append(result)
        privacy_safe_log(consent, f"   #{idx}: {result['name']} ({result['category']}, {result['distance_km']:.2f}km)")
    
    # Generate response
    closest = results[0]
    distance = closest['distance_km']
    
    if LANGUAGE == "English":
        response_text = (
            f"🚨 **EMERGENCY: NEAREST FACILITIES**\n\n"
            f"📢 **Say:** \"Emergency. Need medical help. Location: {closest.get('district', 'Seoul')}.\"\n\n"
            f"1️⃣ **CALL 119 NOW** if life-threatening\n\n"
            f"2️⃣ **Closest facility:**\n"
            f"   🏥 {closest['name']}\n"
            f"   📍 **{distance:.2f} km** away\n"
        )
        
        if closest.get('phone'):
            response_text += f"   ☎️ {closest['phone']}\n"
        
        response_text += (
            f"\n⚠️ **Life-threatening emergency:**\n"
            f"   • Call 119 immediately\n"
            f"   • Stay calm, give your location\n"
            f"   • Follow dispatcher instructions\n\n"
            f"📍 See {len(results)} nearest facilities in cards below."
        )
    
    else:  # Korean
        response_text = (
            f"🚨 **응급: 가장 가까운 시설**\n\n"
            f"📢 **말하세요:** \"응급입니다. 의료 지원 필요. 위치: {closest.get('district', '서울')}.\"\n\n"
            f"1️⃣ **생명 위급 시 지금 바로 119 전화**\n\n"
            f"2️⃣ **가장 가까운 시설:**\n"
            f"   🏥 {closest['name']}\n"
            f"   📍 **{distance:.2f} km**\n"
        )
        
        if closest.get('phone'):
            response_text += f"   ☎️ {closest['phone']}\n"
        
        response_text += (
            f"\n⚠️ **생명 위급:**\n"
            f"   • 즉시 119 전화\n"
            f"   • 침착하게 위치 알림\n"
            f"   • 상황실 지시 따름\n\n"
            f"📍 아래 카드에서 가까운 {len(results)}곳 확인."
        )
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, "✅ EMERGENCY SEARCH COMPLETED")
    privacy_safe_log(consent, f"   Returned {len(results)} facilities")
    privacy_safe_log(consent, "=" * 60 + "\n")
    
    return response_text, results

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


def execute_search(
    state: State, 
    user_message: str, 
    max_distance: float = DEFAULT_MAX_DISTANCE,
    consent: Optional[CookieConsent] = None
) -> Tuple[str, List[Dict]]:
    """
    HYBRID SEARCH with Adaptive Precision Handling.
    
    Handles three search modes based on specialty confidence:
    - HIGH PRECISION (≥0.7): Strict specialty filter, semantic ranking
    - MEDIUM PRECISION (0.3-0.6): Specialty filter + general fallback
    - LOW PRECISION (≤0.3): Distance-first, all facility types
    
    Features:
    - ⭐ Gracefully handles vague/ambiguous queries
    - ⭐ Maintains precision for focused searches
    - ⭐ Progressive expansion when results are sparse
    - ⭐ Word-boundary keyword matching
    - ⭐ Negative keyword support
    """
    global LANGUAGE, rag_pipeline
    
    if not consent:
        consent = CookieConsent()
    
    state.last_search_query = user_message
    state.last_search_timestamp = datetime.utcnow().isoformat()
    
    # ===== DETERMINE SEARCH PRECISION LEVEL =====
    
    specialty_conf = state.specialty_confidence
    has_specialty = bool(state.specialty)
    is_general_search = getattr(state, 'is_general_search', False)
    
    # Classify search precision
    if is_general_search or not has_specialty or specialty_conf < 0.29:
        search_precision = "LOW"  # General/vague search
    elif specialty_conf < 0.7:
        search_precision = "MEDIUM"  # Fuzzy specialty match
    else:
        search_precision = "HIGH"  # Clear, focused search
    
    # ===== LOGGING =====
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, f"🔍 ADAPTIVE HYBRID SEARCH (Precision: {search_precision})")
    privacy_safe_log(consent, f"Query: \"{user_message[:50]}...\"")
    privacy_safe_log(consent, f"Specialty: {state.specialty or 'ANY (no filter)'}")
    privacy_safe_log(consent, f"Specialty Confidence: {specialty_conf:.2f}")
    privacy_safe_log(consent, f"Search Mode: {(state.search_mode or 'auto').upper()}")
    privacy_safe_log(consent, f"Max Distance: {max_distance}km")
    privacy_safe_log(consent, f"Language: {LANGUAGE}")
    
    if search_precision == "LOW":
        privacy_safe_log(consent, "🎲 LOW PRECISION → Distance-first, all facility types")
    elif search_precision == "MEDIUM":
        privacy_safe_log(consent, "🎯 MEDIUM PRECISION → Specialty + general fallback")
    else:
        privacy_safe_log(consent, "🔬 HIGH PRECISION → Strict specialty filter + semantic ranking")
    
    if state.keywords:
        privacy_safe_log(consent, f"💭 Soft Keywords: {', '.join(state.keywords)}")
    if state.hard_keywords:
        privacy_safe_log(consent, f"🔒 Hard Keywords: {', '.join(state.hard_keywords)}")
    if state.negative_keywords:
        privacy_safe_log(consent, f"🚫 Negative Keywords: {', '.join(state.negative_keywords)}")
    if state.negative_hard_keywords:
        privacy_safe_log(consent, f"⛔ Negative Hard Keywords: {', '.join(state.negative_hard_keywords)}")
    
    privacy_safe_log(consent, "=" * 60)
    
    working_df = df_filtered.copy()
    location_context = ""
    user_lat = state.latitude
    user_lon = state.longitude
    relaxed_filters = []
    
    # ===== LOCATION FILTERING =====
    
    if state.search_mode == 'zone' and state.district:
        privacy_safe_log(consent, f"🏘️ Zone search: {state.district} {state.dong or ''}")
        working_df = filter_by_zone(working_df, state.district, state.dong)
        
        location_context = f"in {state.dong}, {state.district}" if state.dong else f"in {state.district}"
        
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
        
        # Reverse geocode
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
            logger.warning("⚠️ No GPS coordinates in dataset")
    
    elif state.location:
        privacy_safe_log(consent, f"📍 Text location: {state.location}")
        working_df = fuzzy_match_location(state.location, working_df)
        location_context = f"in {state.location}"
        working_df['distance_km'] = 0
        privacy_safe_log(consent, f"✓ Fuzzy location match: {len(working_df)} facilities")
    
    else:
        privacy_safe_log(consent, "📍 City-wide search (no location)")
        location_context = "across Seoul"
        working_df['distance_km'] = 0
    
    # ===== ADAPTIVE SPECIALTY FILTERING =====
    
    specialty_filtered_df = pd.DataFrame()
    general_fallback_df = pd.DataFrame()
    
    if search_precision == "HIGH" and has_specialty and 'category' in working_df.columns:
        # HIGH PRECISION: Strict specialty filter only
        privacy_safe_log(consent, f"🔬 HIGH PRECISION: Strict '{state.specialty}' filter")
        
        before_count = len(working_df)
        specialty_filtered_df = working_df[
            working_df['category'].str.contains(state.specialty, na=False, case=False)
        ].copy()
        
        privacy_safe_log(consent, f"   Specialty filter: {before_count} → {len(specialty_filtered_df)}")
        
        # Expand search radius if too few results
        if len(specialty_filtered_df) < 3 and user_lat and user_lon:
            privacy_safe_log(consent, f"   ⚠️ Only {len(specialty_filtered_df)} results, expanding radius...")
            
            expanded_df = df_filtered[
                df_filtered['category'].str.contains(state.specialty, na=False, case=False)
            ].copy()
            
            if len(expanded_df) > 0:
                def calc_distance_expanded(row):
                    if pd.notna(row['lat']) and pd.notna(row['lon']):
                        return haversine(user_lat, user_lon, row['lat'], row['lon'])
                    return 999.0
                
                expanded_df['distance_km'] = expanded_df.apply(calc_distance_expanded, axis=1)
                expanded_df = expanded_df.sort_values('distance_km')
                
                for radius in [max_distance * 1.5, max_distance * 2, max_distance * 3, max_distance * 5, 50]:
                    expanded_working = expanded_df[expanded_df['distance_km'] < radius].copy()
                    
                    if len(expanded_working) >= 3:
                        specialty_filtered_df = expanded_working
                        location_context = f"within {radius:.1f}km"
                        relaxed_filters.append(f"expanded to {radius:.1f}km for more {state.specialty}")
                        privacy_safe_log(consent, f"   ✓ Expanded to {radius:.1f}km: {len(specialty_filtered_df)} found")
                        break
        
        working_df = specialty_filtered_df
    
    elif search_precision == "MEDIUM" and has_specialty and 'category' in working_df.columns:
        # MEDIUM PRECISION: Specialty filter + general fallback
        privacy_safe_log(consent, f"🎯 MEDIUM PRECISION: '{state.specialty}' + general fallback")
        
        before_count = len(working_df)
        specialty_filtered_df = working_df[
            working_df['category'].str.contains(state.specialty, na=False, case=False)
        ].copy()
        
        privacy_safe_log(consent, f"   Specialty filter: {before_count} → {len(specialty_filtered_df)}")
        
        # Add general facilities as fallback
        if len(specialty_filtered_df) < 5:
            privacy_safe_log(consent, f"   Adding general facilities as fallback...")
            
            # Get general facilities not already in specialty results
            specialty_place_ids = set(specialty_filtered_df['place_id'].tolist())
            general_fallback_df = working_df[
                ~working_df['place_id'].isin(specialty_place_ids)
            ].copy()
            
            if 'distance_km' in general_fallback_df.columns:
                general_fallback_df = general_fallback_df.sort_values('distance_km').head(10)
            
            privacy_safe_log(consent, f"   ✓ Added {len(general_fallback_df)} general facilities")
            
            # Merge: specialty first, then general
            working_df = pd.concat([specialty_filtered_df, general_fallback_df], ignore_index=True)
            relaxed_filters.append(f"included {len(general_fallback_df)} nearby general facilities")
        else:
            working_df = specialty_filtered_df
    
    else:
        # LOW PRECISION: No specialty filter, distance-first
        privacy_safe_log(consent, f"🎲 LOW PRECISION: All facility types (distance-first)")
        
        # Keep all facilities, just sort by distance
        if 'distance_km' in working_df.columns:
            working_df = working_df.sort_values('distance_km')
    
    # ===== HARD KEYWORD FILTERING (PATH A) =====
    
    path_a_results_df = pd.DataFrame()
    path_a_place_ids = set()
    
    if state.hard_keywords and len(working_df) > 0:
        privacy_safe_log(consent, "🎯 PATH A: Strict hard keyword filtering")
        
        search_df = working_df.copy()
        
        # Apply strict hard keyword filter
        for keyword in state.hard_keywords:
            keyword_lower = keyword.lower().strip()
            mask = pd.Series([False] * len(search_df), index=search_df.index)
            
            searchable_fields = ['name', 'category', 'address', 'Summaries', 'Summaries_Korean', 
                                'Key_Highlights', 'amenities', 'medical_info_parsed']
            
            for field in searchable_fields:
                if field not in search_df.columns:
                    continue
                
                if field in ['Summaries', 'Summaries_Korean', 'Key_Highlights']:
                    def check_list_field(val):
                        if isinstance(val, (list, np.ndarray)):
                            return any(keyword_matches_word_boundary(keyword_lower, str(item)) for item in val)
                        return False
                    mask = mask | search_df[field].apply(check_list_field)
                
                elif field in ['amenities', 'medical_info_parsed']:
                    def check_dict_field(val):
                        if isinstance(val, dict):
                            return any(keyword_matches_word_boundary(keyword_lower, str(v)) for v in val.values())
                        return False
                    mask = mask | search_df[field].apply(check_dict_field)
                
                else:
                    mask = mask | search_df[field].apply(
                        lambda x: keyword_matches_word_boundary(keyword_lower, str(x))
                    )
            
            search_df = search_df[mask]
            privacy_safe_log(consent, f"   Hard keyword '{keyword}': {len(search_df)} matches")
        
        # Apply negative hard keyword filter
        if state.negative_hard_keywords:
            privacy_safe_log(consent, f"⛔ Applying negative hard keyword exclusions")
            for neg_keyword in state.negative_hard_keywords:
                neg_keyword_lower = neg_keyword.lower().strip()
                exclude_mask = pd.Series([False] * len(search_df), index=search_df.index)
                
                searchable_fields = ['name', 'category', 'address', 'Summaries', 'Summaries_Korean', 
                                    'Key_Highlights', 'amenities', 'medical_info_parsed']
                
                for field in searchable_fields:
                    if field not in search_df.columns:
                        continue
                    
                    if field in ['Summaries', 'Summaries_Korean', 'Key_Highlights']:
                        def check_list_field(val):
                            if isinstance(val, (list, np.ndarray)):
                                return any(keyword_matches_word_boundary(neg_keyword_lower, str(item)) for item in val)
                            return False
                        exclude_mask = exclude_mask | search_df[field].apply(check_list_field)
                    
                    elif field in ['amenities', 'medical_info_parsed']:
                        def check_dict_field(val):
                            if isinstance(val, dict):
                                return any(keyword_matches_word_boundary(neg_keyword_lower, str(v)) for v in val.values())
                            return False
                        exclude_mask = exclude_mask | search_df[field].apply(check_dict_field)
                    
                    else:
                        exclude_mask = exclude_mask | search_df[field].apply(
                            lambda x: keyword_matches_word_boundary(neg_keyword_lower, str(x))
                        )
                
                before = len(search_df)
                search_df = search_df[~exclude_mask]
                privacy_safe_log(consent, f"   Excluded '{neg_keyword}': {before} → {len(search_df)}")
        
        if len(search_df) >= 3:
            path_a_results_df = search_df.copy()
            path_a_place_ids = set(path_a_results_df['place_id'].tolist())
            privacy_safe_log(consent, f"✓ Path A SUCCESS: {len(path_a_results_df)} results")
        else:
            privacy_safe_log(consent, f"⚠️ Path A insufficient: {len(search_df)} results → trying Path B")
            if len(search_df) > 0:
                path_a_results_df = search_df.copy()
                path_a_place_ids = set(path_a_results_df['place_id'].tolist())
    else:
        # No hard keywords, use working_df directly
        path_a_results_df = working_df.copy()
        path_a_place_ids = set(path_a_results_df['place_id'].tolist()) if len(path_a_results_df) > 0 else set()
        privacy_safe_log(consent, f"ℹ️  No hard keywords → Path A uses {len(path_a_results_df)} filtered results")
    
    # ===== ADDITIVE SCORING (PATH B) =====
    
    path_b_results_df = pd.DataFrame()
    needs_path_b = len(path_a_results_df) < 3
    
    if needs_path_b:
        privacy_safe_log(consent, f"🔄 PATH B: Additive scoring (filling to 3-5 results)")
        
        # Exclude Path A results
        path_b_pool = working_df[~working_df['place_id'].isin(path_a_place_ids)].copy()
        privacy_safe_log(consent, f"   Path B pool: {len(path_b_pool)} facilities")
        
        if len(path_b_pool) > 0:
            
            def apply_keyword_boost(df: pd.DataFrame, keywords: List[str], weight: float, label: str) -> pd.Series:
                boost = pd.Series(0.0, index=df.index)
                searchable_fields = ['name', 'category', 'address', 'Summaries', 'Summaries_Korean', 
                                    'Key_Highlights', 'amenities', 'medical_info_parsed']
                
                for kw in keywords:
                    kw_lower = str(kw).lower().strip()
                    if not kw_lower:
                        continue
                    
                    field_matches = pd.Series(False, index=df.index)
                    for field in searchable_fields:
                        if field in df.columns:
                            field_matches |= df[field].apply(
                                lambda x, kw=kw_lower: calculate_field_boost(x, kw)
                            )
                    
                    boost[field_matches] += weight
                    matched_count = field_matches.sum()
                    if matched_count > 0:
                        privacy_safe_log(consent, f"      '{kw}': +{weight} → {matched_count} matches ({label})")
                
                return boost
            
            # Initialize scoring
            path_b_pool['relevance_boost'] = 0.0
            
            # Hard keywords: Priority boost
            if state.hard_keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(
                    path_b_pool, state.hard_keywords, 500.0, "HARD"
                )
            
            # Soft keywords: Preference boost
            if state.keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(
                    path_b_pool, state.keywords, 50.0, "SOFT"
                )
            
            # Negative keywords: Penalty
            if state.negative_keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(
                    path_b_pool, state.negative_keywords, -300.0, "NEGATIVE"
                )
            
            # Negative hard keywords: Severe penalty
            if state.negative_hard_keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(
                    path_b_pool, state.negative_hard_keywords, -1000.0, "NEGATIVE_HARD"
                )
            
            # Specialty match boost (only for medium/high precision)
            if search_precision != "LOW" and state.specialty:
                spec_mask = path_b_pool['category'].fillna('').astype(str).str.lower().str.contains(
                    state.specialty.lower(), na=False
                )
                boost_amount = 200.0 * specialty_conf  # Scale by confidence
                path_b_pool.loc[spec_mask, 'relevance_boost'] += boost_amount
            
            # Distance decay
            if 'distance_km' in path_b_pool.columns:
                max_dist = path_b_pool['distance_km'].max() or 1.0
                if max_dist > 0:
                    distance_penalty = (path_b_pool['distance_km'] / max_dist) * 30
                    path_b_pool['relevance_boost'] -= distance_penalty
            
            # Sort by boost then distance
            path_b_pool = path_b_pool.sort_values(
                by=['relevance_boost', 'distance_km'], 
                ascending=[False, True]
            ).reset_index(drop=True)
            
            path_b_results_df = path_b_pool.copy()
            privacy_safe_log(consent, f"✓ Path B scored {len(path_b_results_df)} facilities")
    
    # ===== MERGE PATH A + PATH B =====
    
    if len(path_a_results_df) >= 3:
        final_df = path_a_results_df.copy()
        privacy_safe_log(consent, f"✅ Using Path A only: {len(final_df)} results")
    
    elif len(path_a_results_df) > 0 and len(path_b_results_df) > 0:
        needed = 5 - len(path_a_results_df)
        path_b_top = path_b_results_df.head(needed)
        
        final_df = pd.concat([path_a_results_df, path_b_top], ignore_index=True)
        privacy_safe_log(consent, f"✅ MERGED: {len(path_a_results_df)} (Path A) + {len(path_b_top)} (Path B)")
        relaxed_filters.append(f"added {len(path_b_top)} results via priority scoring")
    
    elif len(path_b_results_df) > 0:
        final_df = path_b_results_df.copy()
        privacy_safe_log(consent, f"✅ Using Path B only: {len(final_df)} results")
        relaxed_filters.append("used priority scoring (no strict matches)")
    
    else:
        final_df = working_df.copy()
        privacy_safe_log(consent, f"⚠️ Both paths failed, using {len(final_df)} base results")
    
    # ===== HYBRID RAG SEMANTIC RANKING =====
    
    if search_precision != "LOW":
        # Build query for semantic search
        query_components = []
        if state.specialty and search_precision == "HIGH":
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
                
                actual_alpha = rag_pipeline.calculate_alpha(
                    route_decision, 
                    state.manual_search_mode,
                )
                state.hybrid_alpha = actual_alpha
                
                privacy_safe_log(consent, f"   🎯 Router: {state.query_intent} (α={actual_alpha:.2f})")
                
                final_df = rag_pipeline.apply_combined_ranking(
                    df=final_df,
                    query_text=query_text,
                    max_distance=max_distance,
                    search_mode=state.search_mode or 'distance',
                    n_results=200,
                    use_hybrid=True,
                    manual_mode=state.manual_search_mode,
                    is_general_search=(search_precision == "LOW"),
                    specialty_confidence=specialty_conf
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
    else:
        # LOW PRECISION: Skip semantic ranking, pure distance sort
        privacy_safe_log(consent, "📍 LOW PRECISION → Skipping semantic ranking (distance-only)")
        if 'distance_km' in final_df.columns:
            final_df = final_df.sort_values('distance_km')
        state.query_intent = "GENERAL"
        state.hybrid_alpha = None
    
    # ===== ENGLISH FILTER =====
    
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
    
    # ===== RESULT SELECTION =====
    
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
            facilities_context = rag_pipeline.build_context_for_llm(
                final_df, 
                n_results=n_results, 
                language=LANGUAGE
            )
        except Exception as e:
            logger.error(f"Context building error: {e}", exc_info=True)
            facilities_context = "Error building context"
        
        gen_messages = [{
            "role": "system",
            "content": GENERATION_PROMPT.format(
                user_query=user_message,
                location_context=location_context,
                language=LANGUAGE,
                facilities_context=facilities_context,
                hard_keywords=", ".join(state.hard_keywords) if state.hard_keywords else "none",
                soft_keywords=", ".join(state.keywords) if state.keywords else "none"
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
            response_text = re.sub(r'[\u0400-\u04FF]+', 'Seoul', response_text)
            response_text = re.sub(r'대한민국 서울특별시 중구 세종대로 110', 'Seoul', response_text)
            response_text = re.sub(r'Seoul, 서울특별시 대한민국', '', response_text)


            # Add contextual disclaimers
            if search_precision == "LOW" and len(results) > 0:
                if LANGUAGE == "English":
                    response_text += "\n\n💡 Showing nearby facilities sorted by distance. These are general medical facilities that can help assess your needs or provide referrals."
                else:
                    response_text += "\n\n💡 거리순으로 근처 시설을 표시합니다. 진료 후 필요시 전문의에게 의뢰됩니다."
            
            elif search_precision == "MEDIUM" and len(general_fallback_df) > 0:
                if LANGUAGE == "English":
                    response_text += f"\n\n💡 I included some general clinics nearby since there were limited {state.specialty} options in your area."
                else:
                    response_text += f"\n\n💡 해당 지역에 {state.specialty} 시설이 제한적이어서 일반 병원도 포함했습니다."
            
            elif len(relaxed_filters) > 0 and n_results >= 3:
                if LANGUAGE == "English":
                    if any("expanded" in f for f in relaxed_filters):
                        response_text += "\n\n💡 I expanded the search area to find more options."
                    else:
                        response_text += "\n\n💡 To show you 3+ options, I " + " and ".join(relaxed_filters) + "."
                else:
                    response_text += "\n\n💡 더 많은 옵션을 위해 검색 범위를 조정했습니다."
            
            elif n_results < 3:
                if LANGUAGE == "English":
                    response_text += f"\n\n⚠️ Limited availability: Only {n_results} facilities found. Consider expanding your search area or criteria."
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
            privacy_safe_log(consent, f"   #{idx}: {result['name']} ({result['category']}) {distance_value:.1f}km{boost_info}")
    
    else:
        logger.warning("⚠️ No facilities found")
        
        if state.specialty and search_precision == "HIGH":
            if LANGUAGE == "English":
                response_text = (
                    f"I couldn't find any {state.specialty} facilities in your search area.\n\n"
                    f"Try:\n• Expanding your search radius\n• Searching 'any doctor' to see general facilities"
                )
            else:
                response_text = f"검색 지역에서 {state.specialty} 시설을 찾을 수 없습니다.\n\n검색 범위를 확대하거나 '아무 의사'로 검색해 보세요."
        else:
            if LANGUAGE == "English":
                response_text = "I couldn't find any facilities matching your criteria.\n\nTry adjusting your location or search area."
            else:
                response_text = "조건에 맞는 시설을 찾을 수 없습니다.\n\n위치나 검색 범위를 조정해 보세요."
    
    # ===== UPDATE STATE METADATA =====
    
    state.last_results_count = len(results)
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, f"✅ ADAPTIVE SEARCH COMPLETED (Precision: {search_precision})")
    privacy_safe_log(consent, f"   Path A results: {len(path_a_results_df)}")
    privacy_safe_log(consent, f"   Path B results: {len(path_b_results_df)}")
    privacy_safe_log(consent, f"   Final returned: {len(results)}")
    if state.query_intent:
        privacy_safe_log(consent, f"   Query intent: {state.query_intent} (α={state.hybrid_alpha:.2f})" if state.hybrid_alpha else f"   Query intent: {state.query_intent}")
    if relaxed_filters:
        privacy_safe_log(consent, f"   Adjustments: {', '.join(relaxed_filters)}")
    if len(results) < 3:
        privacy_safe_log(consent, f"   ⚠️ WARNING: Only {len(results)} results")
    privacy_safe_log(consent, "=" * 60 + "\n")

    response_text = clean_llm_response(response_text)
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


# ⭐ NEW ENDPOINT: Set Travel Preference from Widget
@app.post("/set_travel_preference")
async def set_travel_preference(
    req: dict,
    cookieConsent: Optional[str] = Cookie(None)
):
    """Direct endpoint for setting travel preference from UI widget."""
    global LANGUAGE
    
    consent = get_consent_from_cookie(cookieConsent)
    consent = ensure_consent_object(consent)
    
    travel_label = req.get('travel_label')
    current_state_dict = req.get('current_state', {})
    
    if not travel_label or travel_label not in DISTANCE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid travel label")
    
    # Reconstruct state
    try:
        state = State(**current_state_dict)
    except Exception as e:
        logger.error(f"State reconstruction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid state")
    
    # Update travel preferences with HIGH confidence (widget-based)
    state.travel_label = travel_label
    state.travel_confidence = 1.0  # ⭐ HIGH confidence from widget
    state.max_distance_km = DISTANCE_MAPPING[travel_label]
    
    # Detect language from state
    LANGUAGE = state.language_pref or "English"
    
    privacy_safe_log(consent, f"✅ Travel preference set via widget: {travel_label} → {state.max_distance_km}km (confidence: 1.0)")
    
    # Generate response
    if LANGUAGE == "English":
        response_text = f"Got it! I'll search within {state.max_distance_km}km ({travel_label})."
    else:
        response_text = f"알겠습니다! {state.max_distance_km}km 반경으로 검색하겠습니다 ({travel_label})."
    
    return {
        "response": response_text,
        "state": state.model_dump()
    }


# Seoul district names for early detection
SEOUL_DISTRICTS = {
    "gangnam", "songpa", "mapo", "jongno", "yongsan", "seodaemun", 
    "dongdaemun", "jungnang", "seongbuk", "gangbuk", "dobong", "nowon",
    "eunpyeong", "jung", "seongdong", "gwangjin", "gangseo", "yangcheon", 
    "guro", "geumcheon", "yeongdeungpo", "dongjak", "gwanak", "seocho", "gangdong",
    "강남", "송파", "마포", "종로", "용산", "서대문", "동대문", "중랑",
    "성북", "강북", "도봉", "노원", "은평", "중구", "성동", "광진",
    "강서", "양천", "구로", "금천", "영등포", "동작", "관악", "서초", "강동",
    "gangnam-gu", "songpa-gu", "mapo-gu", "jongno-gu", "강남구", "송파구", "마포구"
}

CITY_WIDE_KEYWORDS = {
    "citywide", "city wide", "city-wide", "anywhere", "all seoul", 
    "entire seoul", "전체", "서울전체", "서울 전체", "아무데나", "어디든", 
    "어디든지", "상관없", "just citywide", "doesnt matter", "doesn't matter"
}


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
    
    # ============================================
    # ⭐ EARLY DETECTION: City-wide or district responses
    # ============================================
    message_lower = req.message.lower().strip()
    message_normalized = message_lower.replace("-", "").replace(" ", "").replace("gu", "").replace("구", "")
    
    is_city_wide_response = any(kw in message_lower for kw in CITY_WIDE_KEYWORDS)
    is_simple_district = (
        message_normalized in {d.replace("-", "").replace(" ", "").replace("gu", "").replace("구", "") for d in SEOUL_DISTRICTS} or
        message_lower in SEOUL_DISTRICTS
    )
    
    # ===== HANDLE CITY-WIDE RESPONSE =====
    if is_city_wide_response:
        privacy_safe_log(consent, "🌆 EARLY DETECTION: City-wide response")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        new_state.location = "Seoul"
        new_state.latitude = 37.5665
        new_state.longitude = 126.9780
        new_state.max_distance_km = 25.0
        new_state.travel_label = "Anywhere in Seoul"
        new_state.travel_confidence = 0.6  # Text-based
        new_state.search_mode = 'distance'
        
        # ⭐ FIX: Allow null specialty if user wants "any"
        can_proceed = (
            (new_state.specialty and new_state.specialty_confidence >= 0.5) or
            (not new_state.specialty and new_state.specialty_confidence >= 0.3)
        )
        
        if can_proceed:
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
            if LANGUAGE == "English":
                response_text = "I'll search all of Seoul. What type of doctor do you need? (or say 'any' for all facilities)"
            else:
                response_text = "서울 전역에서 검색하겠습니다. 어떤 종류의 의사가 필요하신가요? (또는 '상관없음'이라고 말씀하세요)"
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }
    
    # ===== HANDLE SIMPLE DISTRICT RESPONSE =====
    if is_simple_district:
        privacy_safe_log(consent, f"📍 EARLY DETECTION: District response '{req.message}'")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        new_state.location = req.message.strip()
        
        verified = verify_and_standardize_address(f"서울 {req.message.strip()}")
        
        if verified:
            new_state.latitude = verified['lat']
            new_state.longitude = verified['lon']
            new_state.district = verified['district']
            new_state.dong = verified.get('dong')
            new_state.address_korean = verified['address_korean']
            new_state.search_mode = 'zone'
            privacy_safe_log(consent, f"✓ Geocoded to: {new_state.district}")
        else:
            new_state.search_mode = 'distance'
            new_state.max_distance_km = 10.0
        
        # ⭐ FIX: Allow null specialty
        can_proceed = (
            (new_state.specialty and new_state.specialty_confidence >= 0.5) or
            (not new_state.specialty and new_state.specialty_confidence >= 0.3)
        )
        
        if can_proceed:
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
            district_name = new_state.district or req.message.strip()
            if LANGUAGE == "English":
                response_text = f"Got it, I'll search in {district_name}. What type of doctor do you need? (or 'any')"
            else:
                response_text = f"{district_name}에서 검색하겠습니다. 어떤 종류의 의사가 필요하신가요? (또는 '상관없음')"
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }
    
    # ============================================
    # NORMAL ROUTING
    # ============================================
    
    router_messages = [{
        "role": "system",
        "content": ROUTER_PROMPT.format(
            SPECIALTY_MAPPING=SPECIALTY_MAPPING,
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
    
    # EMERGENCY
    if intent == "EMERGENCY":
        privacy_safe_log(consent, "🚨 BRANCH: EMERGENCY")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        new_state.conversation_phase = "emergency"
        
        if not (new_state.latitude and new_state.longitude):
            privacy_safe_log(consent, "Checking for location in emergency message...")
            extracted = extract_entities(req.message, consent)
            
            if extracted.get('location') or extracted.get('latitude'):
                new_state = merge_extraction_into_state(new_state, extracted)
                new_state = standardize_and_fill_state(new_state, consent)
        
        response_text, results = execute_emergency_search(
            new_state,
            req.message,
            consent=consent
        )
        
        return {
            "response": response_text,
            "state": new_state.model_dump(),
            "results": results
        }
    
    # CONFIRMATION
    # CONFIRMATION
    elif intent == "CONFIRMATION":
        privacy_safe_log(consent, "✅ BRANCH: CONFIRMATION")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        
        # ===== DETECT TRUE "ANY" KEYWORDS (NOT FACILITY TYPES) =====
        
        # Pure vague/random keywords (no semantic content)
        PURE_ANY_KEYWORDS = {
            # English - true vagueness
            "any", "anything", "whatever", "doesn't matter", "doesnt matter", 
            "don't care", "dont care", "no preference", "surprise me", "random",
            "just show me", "i don't mind", "i dont mind", "up to you",
            
            # Korean - true vagueness
            "아무거나", "상관없어", "상관없음", "아무", "랜덤", "아무데나",
            "다 좋아", "괜찮아", "뭐든지", "상관 없어요", "아무거나 좋아요"
        }
        
        # Facility type keywords (these are NOT random - they're specifications)
        FACILITY_TYPE_KEYWORDS = {
            "hospital", "hospitals", "clinic", "clinics", "doctor", "doctors",
            "의원", "병원", "한의원", "의사", "치과", "약국"
        }
        
        message_lower = req.message.lower().strip()
        message_words = set(message_lower.split())
        
        # Check if this is a pure "any" request
        is_pure_any = any(kw == message_lower for kw in PURE_ANY_KEYWORDS)  # Exact match
        is_any_phrase = any(phrase in message_lower for phrase in [
            "any doctor", "any clinic", "any hospital", "any facility",
            "any type", "any kind", "아무 의사", "아무 병원"
        ])
        
        # Check if user is specifying a facility type (NOT random)
        is_facility_type_request = any(kw in message_words for kw in FACILITY_TYPE_KEYWORDS)
        
        # Determine if this is truly a random request
        is_any_request = (is_pure_any or is_any_phrase) and not is_facility_type_request
        
        # ===== HANDLE TRUE "ANY" REQUEST → RANDOM/GENERAL SEARCH =====
        
        if is_any_request:
            privacy_safe_log(consent, "🎲 TRUE 'ANY' detected → Random/General search mode")
            
            # Clear specialty info for true random search
            new_state.specialty = None
            new_state.specialty_confidence = 0.2  # Low confidence triggers distance-first ranking
            new_state.is_general_search = True  # Explicit flag for general search
            
            # Clear keywords if this is a pure "any" request (not refinement)
            if is_pure_any or len(message_lower.split()) <= 3:
                new_state.keywords = []
                new_state.hard_keywords = []
                privacy_safe_log(consent, "   Cleared keywords (pure random search)")
            
            privacy_safe_log(consent, "   Mode: Distance-first, all facility types")
            
            # Ensure location is set (default to city-wide if missing)
            new_state = ensure_city_wide_defaults(new_state, consent)
            
            # ALWAYS proceed with random search
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
        
        # ===== HANDLE FACILITY TYPE REQUEST (병원/의원/한의원) =====
        
        elif is_facility_type_request and not new_state.specialty:
            privacy_safe_log(consent, "🏥 Facility type request detected")
            
            # Map common facility type requests to categories
            if any(kw in message_lower for kw in ["hospital", "병원"]):
                # User wants actual hospitals (not clinics)
                new_state.specialty = "병원"
                new_state.specialty_confidence = 0.7
                privacy_safe_log(consent, "   Mapped to: 병원 (hospitals)")
            
            elif any(kw in message_lower for kw in ["clinic", "의원"]):
                # User wants clinics
                new_state.specialty = "의원"
                new_state.specialty_confidence = 0.7
                privacy_safe_log(consent, "   Mapped to: 의원 (clinics)")
            
            elif any(kw in message_lower for kw in ["한의원", "oriental medicine", "korean medicine"]):
                # User wants traditional Korean medicine
                new_state.specialty = "한의원"
                new_state.specialty_confidence = 0.9
                privacy_safe_log(consent, "   Mapped to: 한의원")
            
            else:
                # Generic "doctor" - treat as general search
                new_state.specialty = "병원,의원"
                new_state.specialty_confidence = 0.5
                privacy_safe_log(consent, "   Mapped to: 병원,의원 (general medical facilities)")
            
            # Ensure location defaults
            new_state = ensure_city_wide_defaults(new_state, consent)
            
            # Proceed with search
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
        
        # ===== HANDLE NORMAL CONFIRMATION (WITH SPECIALTY) =====
        
        else:
            # Boost existing specialty confidence (user confirmed their intent)
            if new_state.specialty and new_state.specialty_confidence < 1.0:
                old_confidence = new_state.specialty_confidence
                
                # Boost confidence based on current level
                if new_state.specialty_confidence < 0.5:
                    # Low confidence → moderate boost (to medium precision)
                    new_state.specialty_confidence = min(0.6, new_state.specialty_confidence + 0.3)
                else:
                    # Medium confidence → small boost (to high precision)
                    new_state.specialty_confidence = min(0.9, new_state.specialty_confidence + 0.2)
                
                privacy_safe_log(consent, f"📈 Confidence boosted: {old_confidence:.2f} → {new_state.specialty_confidence:.2f}")
            
            # Ensure location defaults
            new_state = ensure_city_wide_defaults(new_state, consent)
            
            # ===== DETERMINE IF WE CAN PROCEED =====
            
            # Check if we have enough information to search
            has_location = bool(new_state.location or new_state.latitude or new_state.district)
            has_specialty = bool(new_state.specialty)
            has_keywords = bool(new_state.keywords or new_state.hard_keywords)
            
            # We can proceed if we have ANY of: location, specialty, or keywords
            can_proceed = has_location or has_specialty or has_keywords
            
            if can_proceed:
                new_state.ready_to_search = True
                new_state.conversation_phase = "searching"
                
                # Log search type
                if has_specialty:
                    precision = "HIGH" if new_state.specialty_confidence >= 0.7 else "MEDIUM" if new_state.specialty_confidence >= 0.3 else "LOW"
                    privacy_safe_log(consent, f"✓ Proceeding with {precision} precision specialty search")
                elif has_keywords:
                    privacy_safe_log(consent, f"✓ Proceeding with keyword-based search")
                else:
                    privacy_safe_log(consent, f"✓ Proceeding with general location-based search")
                
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
            
            # ===== NEED MORE INFORMATION =====
            
            else:
                # We have NOTHING - need at least location OR specialty
                new_state.conversation_phase = "gathering"
                
                privacy_safe_log(consent, "⚠️ Insufficient information - requesting clarification")
                
                if LANGUAGE == "English":
                    response_text = (
                        "I'd be happy to help! To find the right facility, I need:\n\n"
                        "**Where?**\n"
                        "• Share your location (button below)\n"
                        "• Or tell me a district (e.g., 'Gangnam')\n"
                        "• Or say 'citywide' for all Seoul\n\n"
                        "**What type?**\n"
                        "• Specialty (e.g., 'dermatologist', 'dentist', 'internal medicine')\n"
                        "• Or facility type ('hospital' vs 'clinic')\n"
                        "• Or say 'any doctor' to see nearby options"
                    )
                else:
                    response_text = (
                        "도와드리겠습니다! 다음 정보가 필요합니다:\n\n"
                        "**어디?**\n"
                        "• 위치 공유 (아래 버튼)\n"
                        "• 또는 구 이름 (예: '강남')\n"
                        "• 또는 '서울 전체'\n\n"
                        "**어떤 종류?**\n"
                        "• 전문 분야 (예: '피부과', '치과', '내과')\n"
                        "• 또는 시설 유형 ('병원' 또는 '의원')\n"
                        "• 또는 '아무 의사'라고 말씀하세요"
                    )
                
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
                new_state.travel_confidence = 0.6  # Text-based
                if new_state.travel_label in DISTANCE_MAPPING:
                    new_state.max_distance_km = DISTANCE_MAPPING[new_state.travel_label]
            
            new_state = ensure_city_wide_defaults(new_state, consent)
            
            # ⭐ FIX: Allow null specialty
            can_proceed = (
                (new_state.specialty and new_state.specialty_confidence >= 0.5) or
                (not new_state.specialty and new_state.specialty_confidence >= 0.3)
            )
            
            if can_proceed:
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
            elif new_state.specialty and new_state.specialty_confidence < 0.3:
                response_text = ask_for_specialty_clarification(new_state)
                return {
                    "response": response_text,
                    "state": new_state.model_dump(),
                    "results": []
                }
            
            new_state.conversation_phase = "gathering"
            if LANGUAGE == "English":
                response_text = "What type of medical facility are you looking for? (or 'any')"
            else:
                response_text = "어떤 종류의 의료 시설을 찾고 계신가요? (또는 '상관없음')"
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }
        
        # City-wide request detection (backup)
        is_city_wide_request = any(phrase in message_lower for phrase in CITY_WIDE_KEYWORDS)
        
        if is_city_wide_request:
            privacy_safe_log(consent, "🌆 City-wide search requested")
            
            new_state = enriched_state.model_copy()
            new_state.turn_count = current_turn
            new_state.language_pref = LANGUAGE
            new_state.search_mode = 'distance'
            new_state.max_distance_km = 25.0
            new_state.travel_label = "Anywhere in Seoul"
            new_state.travel_confidence = 0.6  # Text-based
            new_state.location = "Seoul"
            new_state.latitude = 37.5665
            new_state.longitude = 126.9780
            
            # ⭐ FIX: Allow null specialty
            can_proceed = (
                (new_state.specialty and new_state.specialty_confidence >= 0.5) or
                (not new_state.specialty and new_state.specialty_confidence >= 0.3)
            )
            
            if can_proceed:
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
                if LANGUAGE == "English":
                    response_text = "I'll search all of Seoul. What type of doctor do you need? (or 'any')"
                else:
                    response_text = "서울 전역에서 검색하겠습니다. 어떤 종류의 의사가 필요하신가요? (또는 '상관없음')"
                
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
            new_state.travel_confidence = enriched_state.travel_confidence
            privacy_safe_log(consent, f"✓ Preserved distance: {new_state.max_distance_km}km")
        
        new_state = standardize_and_fill_state(new_state, consent)
        new_state.language_pref = LANGUAGE
        
        new_state = ensure_city_wide_defaults(new_state, consent)
        
        # ⭐ FIX: Allow null specialty
        can_proceed = (
            (new_state.specialty and new_state.specialty_confidence >= 0.5) or
            (not new_state.specialty and new_state.specialty_confidence >= 0.3)
        )
        
        if can_proceed:
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
        elif new_state.specialty and new_state.specialty_confidence < 0.3:
            response_text = ask_for_specialty_clarification(new_state)
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }
        else:
            new_state.conversation_phase = "gathering"
            if LANGUAGE == "English":
                response_text = "What type of medical facility are you looking for? (or 'any')"
            else:
                response_text = "어떤 종류의 의료 시설을 찾고 계신가요? (또는 '상관없음')"
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }

    # PROVIDE_INFO
    # PROVIDE_INFO
    elif intent == "PROVIDE_INFO":
        privacy_safe_log(consent, "📝 BRANCH: PROVIDE_INFO")
        
        # ============================================
        # STEP 1: EXTRACT ENTITIES FROM USER MESSAGE
        # ============================================
        extracted = extract_entities(req.message, consent)
        
        # Check if this is a refinement (adding criteria) vs replacement
        is_refinement = any(word in req.message.lower() for word in 
                            ["also", "and", "additionally", "plus", "as well", 
                            "그리고", "또한", "추가로", "와", "과"])
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = LANGUAGE
        
        # ============================================
        # STEP 2: VALIDATE & CLEAN EXTRACTED KEYWORDS
        # ============================================
        # Remove hallucinated keywords that don't appear in user message
        if extracted.get('hard_keywords'):
            validated_hard = []
            for kw in extracted['hard_keywords']:
                # Check if keyword actually appears in user message (case-insensitive)
                if kw.lower() in req.message.lower():
                    validated_hard.append(kw)
                else:
                    privacy_safe_log(consent, f"⚠️ REMOVED hallucinated hard keyword: '{kw}'")
            extracted['hard_keywords'] = validated_hard
        
        if extracted.get('soft_keywords'):
            validated_soft = []
            for kw in extracted['soft_keywords']:
                if kw.lower() in req.message.lower():
                    validated_soft.append(kw)
                else:
                    privacy_safe_log(consent, f"⚠️ REMOVED hallucinated soft keyword: '{kw}'")
            extracted['soft_keywords'] = validated_soft
        
        # ============================================
        # STEP 3: OVERRIDE LLM FOR PURE FACILITY TYPE REQUESTS
        # ============================================
        message_lower = req.message.lower().strip()
        message_words = set(message_lower.split())
        
        # Pure facility type keywords (should override LLM)
        PURE_FACILITY_TYPES = {
            "clinic": ("의원", 0.8),
            "clinics": ("의원", 0.8),
            "hospital": ("병원", 0.8),
            "hospitals": ("병원", 0.8),
            "doctor": ("병원,의원", 0.7),
            "doctors": ("병원,의원", 0.7),
            "의원": ("의원", 0.9),
            "병원": ("병원", 0.9),
            "의사": ("병원,의원", 0.8),
            "한의원": ("한의원", 0.9)
        }
        
        # Specialty modifiers that indicate this is NOT a pure facility request
        SPECIALTY_MODIFIERS = [
            # English
            "dental", "derm", "cardio", "ortho", "pediatric", "eye", "heart", 
            "skin", "bone", "child", "mental", "foot", "ear", "nose", "throat", 
            "brain", "lung", "stomach", "kidney", "plastic", "cosmetic",
            # Korean
            "치과", "피부과", "내과", "정형외과", "소아과", "안과", "심장", 
            "피부", "뼈", "소아", "정신", "발", "귀", "코", "목", "뇌", 
            "폐", "위", "신장", "성형", "미용"
        ]
        
        # Check if this is a pure facility type request
        is_pure_facility_request = (
            len(message_lower.split()) <= 2 and
            not any(modifier in message_lower for modifier in SPECIALTY_MODIFIERS)
        )
        
        if is_pure_facility_request:
            for keyword, (specialty, confidence) in PURE_FACILITY_TYPES.items():
                if keyword in message_words:
                    extracted['specialty'] = specialty
                    extracted['specialty_confidence'] = confidence
                    extracted['hard_keywords'] = []  # Clear any hallucinated keywords
                    extracted['soft_keywords'] = []
                    privacy_safe_log(consent, f"   🎯 OVERRIDE: Pure '{keyword}' → {specialty} (conf={confidence})")
                    break
        
        # ============================================
        # STEP 3.5: POST-EXTRACTION VALIDATION FOR "DOCTOR"
        # ============================================
        # Catch cases where LLM hallucinated a specialty for simple "doctor" queries
        if ("doctor" in message_words or "의사" in message_words):
            word_count = len(message_words)
            
            # Check if this is a simple "doctor" request (not "eye doctor", "heart doctor", etc.)
            is_simple_doctor_request = (
                word_count <= 3 and  # Max 3 words (e.g., "find a doctor")
                not any(modifier in message_lower for modifier in SPECIALTY_MODIFIERS)
            )
            
            if is_simple_doctor_request:
                # Check if LLM extracted a specific specialty incorrectly
                current_specialty = extracted.get('specialty', '')
                is_general_specialty = current_specialty in ["병원,의원", "병원", "의원", ""]
                
                if not is_general_specialty:
                    # LLM hallucinated a specific specialty for "doctor" - override it
                    old_specialty = current_specialty
                    extracted['specialty'] = "병원,의원"
                    extracted['specialty_confidence'] = 0.7
                    extracted['hard_keywords'] = []
                    extracted['soft_keywords'] = []
                    privacy_safe_log(consent, 
                        f"   ⚠️ VALIDATION: Simple 'doctor' request detected, "
                        f"overriding '{old_specialty}' → 병원,의원")
        
        # ============================================
        # STEP 4: PRESERVE EXISTING SPECIALTY IF NOT REPLACED
        # ============================================
        if not extracted.get('specialty') and enriched_state.specialty:
            extracted['specialty'] = enriched_state.specialty
            extracted['specialty_confidence'] = enriched_state.specialty_confidence
            privacy_safe_log(consent, f"✓ Preserved existing specialty: {enriched_state.specialty}")
        
        # ============================================
        # STEP 5: ENHANCED FALLBACK DETECTION
        # ============================================
        # Only run if LLM didn't extract specialty or confidence is low
        if not extracted.get('specialty') or (extracted.get('specialty_confidence', 0) < 0.3):
            
            # PRIORITY CHECK: "doctor" or "의사" (highest priority)
            if ("doctor" in message_words or "의사" in message_words):
                # Double-check this isn't a specialty-modified request
                is_general_doctor = not any(modifier_phrase in message_lower for modifier_phrase in [
                    "eye doctor", "heart doctor", "skin doctor", "pediatric doctor",
                    "dental doctor", "foot doctor", "ear doctor", "brain doctor",
                    "안과의사", "심장의사", "피부과의사", "소아과의사", "치과의사"
                ])
                
                if is_general_doctor:
                    extracted['specialty'] = "병원,의원"
                    extracted['specialty_confidence'] = 0.7
                    privacy_safe_log(consent, "   Mapped 'doctor/의사' → 병원,의원 (general)")
            
            # Check for "hospital"
            elif "hospital" in message_words and "병원" not in message_lower:
                extracted['specialty'] = "병원"
                extracted['specialty_confidence'] = 0.7
                privacy_safe_log(consent, "   Mapped 'hospital' → 병원")
            
            # Check for "clinic"
            elif "clinic" in message_words and "의원" not in message_lower:
                extracted['specialty'] = "의원"
                extracted['specialty_confidence'] = 0.7
                privacy_safe_log(consent, "   Mapped 'clinic' → 의원")
            
            # Check for traditional Korean medicine
            elif any(kw in message_lower for kw in ["한의원", "oriental medicine", "korean medicine"]):
                extracted['specialty'] = "한의원"
                extracted['specialty_confidence'] = 0.9
                privacy_safe_log(consent, "   Mapped to 한의원")
            
            # Check for Korean "병원"
            elif "병원" in message_words:
                extracted['specialty'] = "병원"
                extracted['specialty_confidence'] = 0.8
                privacy_safe_log(consent, "   Mapped '병원' → 병원")
            
            # Check for Korean "의원"
            elif "의원" in message_words:
                extracted['specialty'] = "의원"
                extracted['specialty_confidence'] = 0.8
                privacy_safe_log(consent, "   Mapped '의원' → 의원")
        
        # ============================================
        # STEP 6: MERGE EXTRACTION INTO STATE
        # ============================================
        new_state = merge_extraction_into_state(
            new_state, 
            extracted, 
            replace_keywords=(not is_refinement)
        )
        new_state = standardize_and_fill_state(new_state, consent)
        new_state.language_pref = LANGUAGE
        
        # Ensure location defaults (city-wide if nothing specified)
        new_state = ensure_city_wide_defaults(new_state, consent)
        
        # ============================================
        # STEP 7: DETERMINE IF WE CAN PROCEED TO SEARCH
        # ============================================
        # We can proceed if we have:
        # - High confidence specialty (≥0.5), OR
        # - Medium confidence specialty (≥0.3) with location/keywords, OR
        # - No specialty but have location/keywords (general search)
        
        has_specialty = bool(new_state.specialty)
        has_location = bool(new_state.location or new_state.latitude or new_state.district)
        has_keywords = bool(new_state.keywords or new_state.hard_keywords)
        
        can_proceed = (
            # High confidence specialty alone
            (has_specialty and new_state.specialty_confidence >= 0.5) or
            # Medium confidence specialty with additional context
            (has_specialty and new_state.specialty_confidence >= 0.3 and (has_location or has_keywords)) or
            # No specialty but have location/keywords (general search)
            (not has_specialty and (has_location or has_keywords))
        )
        
        # ============================================
        # STEP 8A: EXECUTE SEARCH IF READY
        # ============================================
        if can_proceed:
            new_state.ready_to_search = True
            new_state.conversation_phase = "searching"
            
            privacy_safe_log(consent, "✅ Ready to search:")
            privacy_safe_log(consent, f"   Specialty: {new_state.specialty or 'ANY'} (conf={new_state.specialty_confidence:.2f})")
            privacy_safe_log(consent, f"   Location: {new_state.district or new_state.location or 'Seoul (city-wide)'}")
            privacy_safe_log(consent, f"   Keywords: {new_state.keywords or new_state.hard_keywords or 'none'}")
            
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
        
        # ============================================
        # STEP 8B: REQUEST SPECIALTY CLARIFICATION IF AMBIGUOUS
        # ============================================
        elif has_specialty and new_state.specialty_confidence < 0.3:
            # We have a specialty but confidence is too low
            new_state.conversation_phase = "gathering"
            
            privacy_safe_log(consent, 
                f"⚠️ Low specialty confidence ({new_state.specialty_confidence:.2f}) - requesting clarification")
            
            response_text = ask_for_specialty_clarification(new_state)
            response_text = format_response(response_text)
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
            }
        
        # ============================================
        # STEP 8C: REQUEST MORE INFORMATION
        # ============================================
        else:
            # We don't have enough information to search
            new_state.conversation_phase = "gathering"
            
            privacy_safe_log(consent, "⚠️ Insufficient information for search")
            privacy_safe_log(consent, f"   Has specialty: {has_specialty}")
            privacy_safe_log(consent, f"   Has location: {has_location}")
            privacy_safe_log(consent, f"   Has keywords: {has_keywords}")
            
            # Generate appropriate prompt based on what's missing
            if not has_specialty and not has_keywords:
                # Missing specialty/keywords
                if LANGUAGE == "English":
                    response_text = (
                        "What type of medical facility are you looking for?\n\n"
                        "Examples:\n"
                        "• 'Clinic' or 'Hospital' (general)\n"
                        "• 'Dentist' or 'Dermatologist' (specialty)\n"
                        "• 'Any doctor' (show nearby options)\n"
                        "• Specific needs like 'clinic with parking' or 'English-speaking doctor'"
                    )
                else:
                    response_text = (
                        "어떤 종류의 의료 시설을 찾고 계신가요?\n\n"
                        "예시:\n"
                        "• '의원' 또는 '병원' (일반)\n"
                        "• '치과' 또는 '피부과' (전문)\n"
                        "• '아무 의사' (근처 옵션 표시)\n"
                        "• '주차 가능한 병원' 또는 '영어 가능한 의사' 같은 구체적 요구사항"
                    )
            
            elif has_specialty and not has_location:
                # Have specialty but no location
                if LANGUAGE == "English":
                    response_text = (
                        f"Got it, you're looking for {new_state.specialty}. Where should I search?\n\n"
                        "• Click the location button below\n"
                        "• Type a district (e.g., 'Gangnam', 'Hongdae')\n"
                        "• Say 'citywide' to search all of Seoul"
                    )
                else:
                    response_text = (
                        f"{new_state.specialty}를 찾고 계시는군요. 어디에서 검색할까요?\n\n"
                        "• 아래 위치 버튼 클릭\n"
                        "• 구 이름 입력 (예: '강남', '홍대')\n"
                        "• '서울 전체'라고 말씀하세요"
                    )
            
            else:
                # Generic fallback
                if LANGUAGE == "English":
                    response_text = (
                        "What type of medical facility are you looking for?\n\n"
                        "Examples: clinic, hospital, dentist, dermatologist, internal medicine, or 'any doctor'"
                    )
                else:
                    response_text = (
                        "어떤 종류의 의료 시설을 찾고 계신가요?\n\n"
                        "예시: 의원, 병원, 치과, 피부과, 내과, 또는 '아무 의사'"
                    )
            
            return {
                "response": response_text,
                "state": new_state.model_dump(),
                "results": []
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
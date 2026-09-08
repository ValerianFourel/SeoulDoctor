import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from models import State
from cookies import (
    CookieConsent, should_log_analytics, privacy_safe_log
)
import sys
import os
from pathlib import Path
import json
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from location import (
    google_maps_geocode, google_maps_reverse_geocode, google_maps_place_search,
    google_maps_place_details, kakao_geocode, kakao_reverse_geocode,
    verify_and_standardize_address
)
from config import DISTANCE_MAPPING
LOCAL_PARQUET_PATH = os.getenv(
    "FACILITIES_CACHE_PATH",
    str(Path(__file__).resolve().parent / "local_facilities_cache.parquet"),
)

load_dotenv()
DEFAULT_MAX_DISTANCE = 5.0  # km - local search radius when the user omits distance

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

HF_REPO_ID = "ValerianFourel/seoul-medical-facilities"
HF_FILENAME = "facilities_metareviews_rag_ready.parquet"




HF_TOKEN = os.getenv("HF_TOKEN") 
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

# ==========================================
# UTILITIES
# ==========================================

# ==========================================
# ADD THIS HELPER FUNCTION AT THE TOP OF main.py (after imports, before lifespan)
# ==========================================

def normalize_seoul_to_null(location: Optional[str]) -> Optional[str]:
    """
    Normalize generic "Seoul" references to None for true city-wide search.
    
    This ensures that when users say just "Seoul" without specifying a district,
    we treat it the same as no location (city-wide search), preventing bias
    toward any specific coordinates or districts.
    
    Args:
        location: Location string to normalize
        
    Returns:
        None if location is just "Seoul" (any variant), otherwise returns original location
    
    Examples:
        >>> normalize_seoul_to_null("Seoul")
        None
        >>> normalize_seoul_to_null("서울")
        None
        >>> normalize_seoul_to_null("Gangnam")
        "Gangnam"
        >>> normalize_seoul_to_null(None)
        None
    """
    if not location:
        return None
    
    location_normalized = location.strip().lower()
    
    # List of generic Seoul references that should be treated as null
    GENERIC_SEOUL_REFS = {
        # English variants
        "seoul", 
        "seoul city", 
        "seoul, korea", 
        "seoul korea",
        "seoul south korea", 
        "seoul, south korea",
        "seoul-si",
        
        # Korean variants
        "서울", 
        "서울시", 
        "서울특별시",
        
        # Mixed/transliterated
        "seoul-si",
        "seoul si"
    }
    
    if location_normalized in GENERIC_SEOUL_REFS:
        logger.debug("Normalized generic Seoul reference to city-wide search")
        return None
    
    return location


def ensure_city_wide_defaults(state: State, consent: CookieConsent) -> State:
    """
    If no location specified (or just "Seoul"), default to city-wide Seoul search.
    ⭐ CRITICAL: Does NOT set GPS coordinates to avoid location bias.
    """
    # Check if location is effectively empty or just "Seoul"
    location_is_generic_seoul = False
    if state.location:
        location_lower = state.location.strip().lower()
        location_is_generic_seoul = location_lower in ["seoul", "서울", "서울시", "seoul city"]
    
    # Treat as city-wide if: no location, OR location is just "Seoul" without specifics
    if (not state.location or location_is_generic_seoul) and not state.latitude and not state.district:
        privacy_safe_log(consent, "🌆 No specific location → defaulting to city-wide Seoul")
        
        # ⚠️ Keep everything NULL for true city-wide (no display bias)
        state.location = None  # ← Set to None even if it was "Seoul"
        state.latitude = None
        state.longitude = None
        state.district = None
        state.dong = None
        state.address_korean = None
        
        state.max_distance_km = 25.0
        state.search_mode = 'distance'
        state.travel_label = "Anywhere in Seoul"
        state.travel_confidence = 1.0
        state.is_citywide_search = True
        
        privacy_safe_log(consent, "   ✓ City-wide mode: All location fields NULL, no GPS, no district bias")
    
    return state

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



# ==========================================
# STATE MANAGEMENT FUNCTIONS
# ==========================================

# REPLACEMENT for standardize_and_fill_state in utils.py
# Add this at the beginning of the function (after the consent check)

def standardize_and_fill_state(state: State, consent: Optional[CookieConsent] = None) -> State:
    """
    Standardize and fill missing location fields in state using geocoding APIs.
    
    ⭐ CRITICAL: Skips enrichment for city-wide searches to prevent bias toward specific districts.
    """
    enriched_state = state.model_copy()
    
    if not consent:
        consent = CookieConsent()
    
    # ===================================================================
    # ⭐ NORMALIZE "Seoul" to None FIRST (before any other processing)
    # ===================================================================
    enriched_state.location = normalize_seoul_to_null(enriched_state.location)
    
    # ===================================================================
    # ⭐ CITY-WIDE SEARCH DETECTION: Skip enrichment if no specific location
    # ===================================================================
    
    # Case 1: Explicitly marked as city-wide
    if state.is_citywide_search:
        privacy_safe_log(consent, "=" * 60)
        privacy_safe_log(consent, "🌆 CITY-WIDE SEARCH MODE (explicitly marked)")
        privacy_safe_log(consent, "   Skipping geocoding enrichment (no location bias)")
        privacy_safe_log(consent, "=" * 60 + "\n")
        
        # Ensure location is None (not "Seoul")
        enriched_state.location = None
        return enriched_state
    
    if not enriched_state.location and not enriched_state.latitude and not enriched_state.district:
        return enriched_state

    # ===================================================================
    # NORMAL ENRICHMENT FLOW (specific location provided)
    # ===================================================================
    # (Rest of the existing function continues here unchanged)
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, "🔧 STATE ENRICHMENT STARTED")
    privacy_safe_log(consent, "=" * 60)
    
    filled_fields = []
    
    # -------------------------------------------------------------------
    # ENRICHMENT CASE 1: Have location text, need GPS data
    # -------------------------------------------------------------------
    if state.location and not (state.latitude and state.longitude):
        privacy_safe_log(consent, f"📍 Enrichment Case 1: Have location text '{state.location}', need GPS data")
        
        verified = verify_and_standardize_address(state.location, consent=consent)
        
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
            
            # Mark as NOT city-wide since we have specific location
            enriched_state.is_citywide_search = False
            
            privacy_safe_log(consent, f"✅ Filled from address: {', '.join(filled_fields)}")
        else:
            logger.warning("Could not geocode location; preserving requested scope")
            enriched_state.is_citywide_search = False

    # -------------------------------------------------------------------
    # ENRICHMENT CASE 2: Have GPS, need address data
    # -------------------------------------------------------------------
    elif state.latitude and state.longitude and not (state.address_korean and state.district):
        privacy_safe_log(consent, f"📍 Enrichment Case 2: Have GPS ({state.latitude:.4f}, {state.longitude:.4f}), need address data")
        
        reverse_result = None
        if GOOGLE_MAPS_API_KEY:
            reverse_result = google_maps_reverse_geocode(state.latitude, state.longitude, consent=consent)
        
        if not reverse_result and KAKAO_REST_API_KEY:
            reverse_result = kakao_reverse_geocode(state.latitude, state.longitude, consent=consent)
        
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
            
            # Mark as NOT city-wide since we have specific GPS
            enriched_state.is_citywide_search = False
            
            privacy_safe_log(consent, f"✅ Filled from GPS: {', '.join(filled_fields)}")
        else:
            logger.warning("⚠️ Could not reverse geocode GPS coordinates")
    
    # -------------------------------------------------------------------
    # ENRICHMENT CASE 3: Have district, need GPS
    # -------------------------------------------------------------------
    elif state.district and not (state.latitude and state.longitude):
        privacy_safe_log(consent, f"📍 Enrichment Case 3: Have district '{state.district}', need GPS")
        
        location_query = f"서울 {state.district}"
        if state.dong:
            location_query = f"서울 {state.district} {state.dong}"
        
        verified = verify_and_standardize_address(location_query, consent=consent)
        
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
            
            # Mark as NOT city-wide since we have specific district
            enriched_state.is_citywide_search = False
            
            privacy_safe_log(consent, f"✅ Filled from district: {', '.join(filled_fields)}")
        else:
            logger.warning("Could not geocode district")
    
    # -------------------------------------------------------------------
    # NO ENRICHMENT NEEDED
    # -------------------------------------------------------------------
    else:
        if state.latitude and state.longitude and state.district:
            privacy_safe_log(consent, "✅ State already complete - no enrichment needed")
        else:
            privacy_safe_log(consent, "ℹ️ Insufficient data for enrichment")
    
    # -------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------
    if filled_fields and should_log_analytics(consent):
        logger.info("\n📋 ENRICHMENT SUMMARY:")
        logger.info(f"   Before: {_format_location_summary(state)}")
        logger.info(f"   After:  {_format_location_summary(enriched_state)}")
        logger.info(f"   Filled: {', '.join(filled_fields)}")
        logger.info(f"   City-wide: {enriched_state.is_citywide_search}")
    
    privacy_safe_log(consent, "=" * 60 + "\n")
    
    return enriched_state


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




def safe_convert_to_python(value):
    """Convert pandas/numpy types to native Python types for JSON serialization."""
    if value is None:
        return None
    
    # Handle numpy arrays and pandas Series BEFORE pd.isna()
    if isinstance(value, np.ndarray):
        return value.tolist()
    
    if isinstance(value, pd.Series):
        return value.tolist()
    
    # Handle NaN for scalars
    if isinstance(value, float) and np.isnan(value):
        return None
    
    # Handle pandas NA for scalars
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    
    # Handle numpy scalar types
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    
    # Handle lists recursively
    if isinstance(value, list):
        return [safe_convert_to_python(item) for item in value]
    
    # Handle dicts recursively
    if isinstance(value, dict):
        return {k: safe_convert_to_python(v) for k, v in value.items()}
    
    return value

def user_wants_any_specialty(message: str) -> bool:
    """Detect if user is okay with any specialty."""
    message_lower = message.lower()
    
    any_phrases = [
        "any specialty", "any doctor", "any type", "any kind",
        "doesn't matter", "doesnt matter", "don't care", "dont care",
        "whatever", "any hospital", "any clinic", "all types",
        "아무거나", "상관없", "어떤 의사든", "아무 병원", "아무 의사"
    ]
    
    return any(phrase in message_lower for phrase in any_phrases)


def has_vague_medical_term(message: str) -> bool:
    """Detect vague medical terms that don't specify specialty."""
    message_lower = message.lower()
    
    vague_terms = ["hospital", "clinic", "doctor", "medical facility", 
                   "병원", "의원", "의사", "진료소"]
    specific_terms = ["dentist", "dermatologist", "internal", "pediatric", "eye",
                      "치과", "피부과", "내과", "소아과", "안과"]
    
    has_vague = any(term in message_lower for term in vague_terms)
    has_specific = any(term in message_lower for term in specific_terms)
    
    return has_vague and not has_specific

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


def state_to_extraction_context(state: State) -> str:
    """
    Convert State to a concise context string for the extraction prompt.
    Shows current parameters so LLM knows what might be changing.
    """
    context = {
        "specialty": state.specialty,
        "specialty_confidence": state.specialty_confidence,
        "location": state.location or state.district,  # Use district if location not set
        "travel_label": state.travel_label,
        "hard_keywords": state.hard_keywords,
        "soft_keywords": state.keywords,  # State.keywords are soft keywords
        "negative_hard_keywords": state.negative_hard_keywords,
        "negative_keywords": state.negative_keywords,
        "language_pref": state.language_pref
    }
    
    # Build human-readable context
    parts = ["**CURRENT STATE (what's already set):**"]
    
    if context["specialty"]:
        parts.append(f"- Specialty: {context['specialty']} (confidence: {context['specialty_confidence']:.1f})")
    else:
        parts.append("- Specialty: Not set")
    
    if context["location"]:
        parts.append(f"- Location: {context['location']}")
    else:
        parts.append("- Location: Not set (will default to Yongsan-gu)")
    
    parts.append(f"- Travel Distance: {context['travel_label']}")
    
    if context["hard_keywords"]:
        parts.append(f"- Hard Keywords (must-have): {', '.join(context['hard_keywords'])}")
    else:
        parts.append("- Hard Keywords: None")
    
    if context["negative_hard_keywords"]:
        parts.append(f"- Negative Hard Keywords (must-not-have): {', '.join(context['negative_hard_keywords'])}")
    
    if context["soft_keywords"]:
        parts.append(f"- Soft Keywords (preferences): {', '.join(context['soft_keywords'])}")
    
    if context["negative_keywords"]:
        parts.append(f"- Negative Keywords (avoid): {', '.join(context['negative_keywords'])}")
    
    parts.append(f"- Language: {context['language_pref']}")
    parts.append("\n**IMPORTANT:** User may be updating ANY of these parameters. Extract ALL changes from the current message.")
    
    return "\n".join(parts)


# Alternative: Minimal JSON format (if you prefer compact)
def state_to_extraction_context_json(state: State) -> str:
    """
    Compact JSON representation of current state.
    """
    context = {
        "current_specialty": state.specialty,
        "current_location": state.location or state.district,
        "current_travel_label": state.travel_label,
        "current_hard_keywords": state.hard_keywords,
        "current_soft_keywords": state.keywords,
        "current_negative_hard_keywords": state.negative_hard_keywords,
        "current_negative_keywords": state.negative_keywords
    }
    
    import json
    return f"**CURRENT PARAMETERS:**\n```json\n{json.dumps(context, ensure_ascii=False, indent=2)}\n```\n\n**NOTE:** Extract ALL changes from user message. Multiple parameters can update simultaneously."

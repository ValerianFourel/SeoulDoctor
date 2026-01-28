import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from models import State
from cookies import (
    CookieConsent, should_log_analytics, privacy_safe_log
)
from prompt import FIELD_CHANGE_DETECTION_PROMPT
import sys
import os
import json
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from groq import Groq
from location import (
    google_maps_geocode, google_maps_reverse_geocode, google_maps_place_search,
    google_maps_place_details, kakao_geocode, kakao_reverse_geocode,
    verify_and_standardize_address
)
LOCAL_PARQUET_PATH = "./local_facilities_cache.parquet"

load_dotenv()
DEFAULT_MAX_DISTANCE = 25.0  # km - default search radius

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

HF_REPO_ID = "ValerianFourel/seoul-medical-facilities"
HF_FILENAME = "facilities_metareviews_rag_ready.parquet"


client = Groq(api_key=GROQ_API_KEY)


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

# Mapping semantic labels to numerical search radii
DISTANCE_MAPPING = {
    "Walking Distance": 0.5,  # 500m
    "Nearby": 1.0,           # 1km
    "Close": 2.0,            # 2km
    "Moderate": 5.0,         # 5km
    "Flexible": 10.0,        # 10km
    "Willing to Travel": 15.0, # 15km
    "Anywhere in Seoul": 25.0  # 25km (Approx max radius for Seoul)
}


# ==========================================
# UTILITIES
# ==========================================
def ensure_city_wide_defaults(state: State, consent=None) -> State:
    """Default to city-wide search if no location specified."""
    has_location = bool(
        state.location or 
        (state.latitude and state.longitude) or 
        state.district
    )
    
    if not has_location:
        state.location = "Seoul"
        state.latitude = 37.5665
        state.longitude = 126.9780
        state.max_distance_km = 25.0
        state.travel_label = "Anywhere in Seoul"
        state.search_mode = 'distance'
        if consent:
            privacy_safe_log(consent, "🌆 No location → city-wide search (25km)")
    
    return state

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
        keywords=", ".join(current_state.keywords) if current_state.keywords else "None",
        hard_keywords=", ".join(current_state.hard_keywords) if current_state.hard_keywords else "None",
        negative_keywords=", ".join(current_state.negative_keywords) if current_state.negative_keywords else "None",  # ← ADD
        negative_hard_keywords=", ".join(current_state.negative_hard_keywords) if current_state.negative_hard_keywords else "None",  # ← ADD
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
    change_detection: Dict[str, str],
    consent: Optional[CookieConsent] = None
) -> State:
    """
    Cleanse state based on LLM-detected field changes.
    Only clears fields that were marked as "change".
    
    Strategy:
    - Specialty change: Clear specialty + ALL keywords (major context shift)
    - Location change: Clear location fields (will be re-enriched by standardize_and_fill_state)
    - Distance change: Reset to default travel preferences
    - Keywords change: Clear keywords but preserve specialty (preference refinement)
    """
    if not consent:
        consent = CookieConsent()
    
    new_state = current_state.model_copy()
    cleared_fields = []
    
    privacy_safe_log(consent, "🧹 STATE CLEANSING STARTED")
    
    # ===== SPECIALTY CHANGE =====
    if change_detection.get('specialty') == 'change':
        privacy_safe_log(consent, "   ✂️ Specialty change detected")
        
        old_specialty = new_state.specialty
        new_state.specialty = None
        new_state.specialty_confidence = 0.0
        cleared_fields.append(f"specialty (was: {old_specialty})")
        
        # ⭐ Clear ALL keywords when specialty changes (major context shift)
        if new_state.keywords or new_state.hard_keywords:
            new_state.keywords = []
            new_state.hard_keywords = []
            new_state.negative_keywords = []
            new_state.negative_hard_keywords = []
            cleared_fields.append("all keywords (specialty changed)")
    
    # ===== LOCATION CHANGE =====
    if change_detection.get('location') == 'change':
        privacy_safe_log(consent, "   ✂️ Location change detected")
        
        # Clear all location-related fields
        # (standardize_and_fill_state will re-enrich from new location)
        location_fields = []
        
        if new_state.location:
            location_fields.append(f"location (was: {new_state.location})")
            new_state.location = None
        
        if new_state.latitude:
            location_fields.append(f"GPS (was: {new_state.latitude:.4f}, {new_state.longitude:.4f})")
            new_state.latitude = None
            new_state.longitude = None
        
        if new_state.address_korean:
            new_state.address_korean = None
            location_fields.append("address_korean")
        
        if new_state.district:
            location_fields.append(f"district (was: {new_state.district})")
            new_state.district = None
        
        if new_state.dong:
            location_fields.append(f"dong (was: {new_state.dong})")
            new_state.dong = None
        
        if new_state.search_mode:
            location_fields.append(f"search_mode (was: {new_state.search_mode})")
            new_state.search_mode = None
        
        cleared_fields.extend(location_fields)
    
    # ===== DISTANCE CHANGE =====
    if change_detection.get('distance') == 'change':
        privacy_safe_log(consent, "   ✂️ Distance/travel preference change detected")
        
        old_distance = new_state.max_distance_km
        old_label = new_state.travel_label
        
        new_state.max_distance_km = DEFAULT_MAX_DISTANCE
        new_state.travel_label = "Moderate"
        
        cleared_fields.append(f"distance (was: {old_distance}km '{old_label}' → default: 5km 'Moderate')")
    
    # ===== KEYWORD CHANGE (INDEPENDENT OF SPECIALTY) =====
    # This handles cases where user refines preferences but keeps same specialty
    # Example: "I need a clinic with parking" → "Actually, I need one with evening hours"
    if change_detection.get('keywords') == 'change' and change_detection.get('specialty') != 'change':
        privacy_safe_log(consent, "   ✂️ Keyword/preference change detected (specialty unchanged)")
        
        keyword_count = len(new_state.keywords) + len(new_state.hard_keywords)
        
        if keyword_count > 0:
            new_state.keywords = []
            new_state.hard_keywords = []
            new_state.negative_keywords = []
            new_state.negative_hard_keywords = []
            cleared_fields.append(f"keywords (cleared {keyword_count} preferences)")
    
    # ===== REVALIDATION CHECK =====
    # If critical fields changed, state needs revalidation before search
    needs_revalidation = any(
        change_detection.get(field) == 'change' 
        for field in ['specialty', 'location']
    )
    
    if needs_revalidation:
        old_phase = new_state.conversation_phase
        new_state.ready_to_search = False
        new_state.search_executed = False
        new_state.conversation_phase = "gathering"
        
        privacy_safe_log(consent, f"   🔄 State requires revalidation (phase: {old_phase} → gathering)")
        cleared_fields.append("ready_to_search, search_executed")
    else:
        privacy_safe_log(consent, "   ✓ No critical fields changed, preserving search state")
    
    # ===== SUMMARY =====
    if cleared_fields:
        privacy_safe_log(consent, f"✅ Cleared: {', '.join(cleared_fields)}")
    else:
        privacy_safe_log(consent, "✅ No fields needed clearing")
    
    privacy_safe_log(consent, "=" * 60 + "\n")
    
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



def clean_llm_response(response_text: str) -> str:
    """
    Remove common LLM formatting issues that break the output.
    
    ⭐ Fixes:
    - "Introduction" headers
    - Excessive bold formatting
    - Code blocks
    - Multiple blank lines
    """
    import re
    
    # Remove "Introduction" or "Facilities:" headers
    response_text = re.sub(r'^#+\s*(Introduction|Facilities|Options|Results).*?\n', '', response_text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove excessive bold formatting (keep some for emphasis)
    # Only remove bold from first line (intro)
    lines = response_text.split('\n')
    if lines:
        lines[0] = re.sub(r'\*\*([^*]+)\*\*', r'\1', lines[0])
    response_text = '\n'.join(lines)
    
    # Remove code blocks (```python, ```typescript, etc.)
    response_text = re.sub(r'```[a-z]*\n?', '', response_text)
    response_text = re.sub(r'```', '', response_text)
    
    # Remove inline code formatting for non-technical content
    response_text = re.sub(r'`([^`]+)`', r'\1', response_text)
    
    # Remove multiple blank lines (keep max 2)
    response_text = re.sub(r'\n{3,}', '\n\n', response_text)
    
    # Remove leading/trailing whitespace
    response_text = response_text.strip()
    
    # Remove checkmarks/crosses if they appear outside of lists
    lines = response_text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Keep checkmarks in numbered lists
        if re.match(r'^\d+\.', line.strip()):
            cleaned_lines.append(line)
        else:
            # Remove checkmarks from intro/outro
            line = re.sub(r'[✓✗❌✅⭐]', '', line)
            cleaned_lines.append(line)
    
    response_text = '\n'.join(cleaned_lines)
    
    return response_text


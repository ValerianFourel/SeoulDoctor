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
from pathlib import Path
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

import json
from functools import partial
from hashlib import sha256
from uuid import uuid4
import pandas as pd
import numpy as np
import re
from fastapi import FastAPI, HTTPException, Cookie, Header, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from datetime import datetime
from time import perf_counter
import logging

# Local imports
from distance import haversine, fuzzy_match_location
from models import (
    ChatRequest,
    State,
    serialize_results_for_chat,
    serialize_state_for_chat,
)
from utils import (
    safe_convert_to_python, DISTANCE_MAPPING,
    standardize_and_fill_state,detect_search_mode,
    print_separator,
    has_vague_medical_term,user_wants_any_specialty, ensure_city_wide_defaults,
    validate_distance_criteria,download_and_cache_parquet, DEFAULT_MAX_DISTANCE,
    LOCAL_PARQUET_PATH,
)
from prompt import (
    ROUTER_PROMPT, 
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
from evidence_response import answer_search
from review_presentation import response_language
from raw_review_store import ensure_raw_review_parquet
from config import (
    BGE_M3_RETRIEVER_API_TOKEN,
    BGE_M3_RETRIEVER_EXPIRES_AT,
    BGE_M3_MODEL_REVISION,
    BGE_M3_RETRIEVER_RELEASE_ID,
    BGE_M3_RETRIEVER_TIMEOUT_SECONDS,
    BGE_M3_RETRIEVER_URL,
    CHAT_RATE_LIMIT_REQUESTS,
    CHAT_RATE_LIMIT_WINDOW_SECONDS,
    ENABLE_RETRIEVAL_DEBUG,
    GROQ_AGENT_MODEL,
    GROQ_CHAT_MODEL,
    GROQ_REASONING_EFFORT,
    LLM_PROVIDER,
    RETRIEVAL_DEBUG_LIMIT,
    RERANKER_API_TOKEN,
    RERANKER_EXPIRES_AT,
    RERANKER_MAX_CANDIDATES,
    RERANKER_TIMEOUT_SECONDS,
    RERANKER_URL,
)
from llm_client import (
    build_llm_client,
    request_json_completion,
    request_answer_completion,
)
from query_facets import (
    augment_extracted_facets,
    retrieval_terms_from_state,
)
from rate_limit import SlidingWindowRateLimiter
from search.contracts import DistanceRule
from search.evidence_retrieval import EvidenceRecallPolicy
from search.indexes import IndexLoadError, IndexRepository
from search.indexes.manifest import sha256_file
from search.live_retrieval import (
    CandidateRetrievalAdapter,
    LEGACY_INDEX_VERSION,
    RetrievalQuery,
)
from search.rules import compile_legacy_state_rules, compile_shadow_rules
from search.reranker import RemoteEvidenceReranker
from search.readiness import probe_retrieval
from search.scope import ScopeBuilder
from search.semantic_retriever import RemoteBgeM3ReviewRetriever
from search.turn_delta import compile_turn_delta, reduce_search_state

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


class RequiredSearchIndexError(RuntimeError):
    """The configured mandatory Phase 3 release is absent or stale."""


def record_shadow_rule_compilation(
    user_message: str,
    extracted: Dict[str, Any],
) -> None:
    """Observe the new rule compiler without changing live search behavior."""
    try:
        compilation = compile_shadow_rules(user_message, extracted)
        logger.info(
            "Search-rule shadow compile status=%s issue_codes=%s",
            "compiled" if compilation.rules else "clarification",
            [issue.code for issue in compilation.issues],
        )
    except Exception:
        logger.exception("Search-rule shadow compiler failed")

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

CHROMA_PATH = os.getenv(
    "CHROMA_PATH", str(Path(__file__).resolve().parent / "chroma_db")
)
SEARCH_INDEX_ROOT = os.getenv("SEARCH_INDEX_ROOT", "").strip()
SEARCH_INDEX_REQUIRED = os.getenv("SEARCH_INDEX_REQUIRED", "false").strip().casefold() in {
    "1", "true", "yes", "on",
}

# Defaults
DEFAULT_LAT = 37.5219  # Yeouido
DEFAULT_LON = 126.9243

# Global data structures
rag_pipeline = None  # RAG Pipeline instance
search_index_release = None  # Validated Phase 3 release; Phase 4 will query it
df_facilities = None  # Full dataset
df_filtered = None    # Filtered subset (Summaries not null)
available_specialties = []  # Unique specialties from parquet data
evidence_reranker = (
    RemoteEvidenceReranker(
        base_url=RERANKER_URL,
        token=RERANKER_API_TOKEN,
        timeout_seconds=RERANKER_TIMEOUT_SECONDS,
        max_candidates=RERANKER_MAX_CANDIDATES,
        expires_at=RERANKER_EXPIRES_AT,
    )
    if RERANKER_URL
    else None
)
semantic_evidence_source = (
    RemoteBgeM3ReviewRetriever(
        base_url=BGE_M3_RETRIEVER_URL,
        token=BGE_M3_RETRIEVER_API_TOKEN,
        release_id=BGE_M3_RETRIEVER_RELEASE_ID,
        timeout_seconds=BGE_M3_RETRIEVER_TIMEOUT_SECONDS,
        expires_at=BGE_M3_RETRIEVER_EXPIRES_AT,
        model_revision=BGE_M3_MODEL_REVISION,
    )
    if BGE_M3_RETRIEVER_URL
    else None
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================




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
        for key, v in value.items():
            if isinstance(v, bool):
                if v and keyword_matches_word_boundary(keyword_lower, str(key).replace('_', ' ')):
                    return True
                continue
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
    global rag_pipeline, search_index_release, df_facilities, df_filtered
    global available_specialties, client
    logger.info(
        "🚀 Booting Seoul Med Match Backend (provider=%s, model=%s)",
        LLM_PROVIDER, GROQ_CHAT_MODEL,
    )

    try:
        df_facilities = download_and_cache_parquet()
        logger.info(f"✅ Loaded {len(df_facilities)} facilities from parquet")
        
        raw_reviews_path = ensure_raw_review_parquet()
        raw_review_source_sha256 = (
            sha256_file(Path(raw_reviews_path)) if raw_reviews_path else None
        )
        if raw_reviews_path:
            df_filtered = df_facilities.copy()
            logger.info(f"✅ Search coverage expanded to all {len(df_filtered)} facilities")
        else:
            df_filtered = df_facilities[df_facilities['Summaries'].notna()].copy()
            logger.info(f"✅ Filtered to {len(df_filtered)} facilities with summaries")
        
        df_filtered['place_id'] = df_filtered['place_id'].fillna('').astype(str)

        if SEARCH_INDEX_ROOT:
            try:
                candidate_release = IndexRepository(
                    Path(SEARCH_INDEX_ROOT)
                ).open_active()
                catalog_ids = set(df_filtered['place_id'])
                indexed_ids = set(candidate_release.facility_ids)
                if catalog_ids != indexed_ids:
                    raise IndexLoadError(
                        "active Phase 3 release does not match the facility catalog"
                    )
                if (
                    sha256_file(Path(LOCAL_PARQUET_PATH))
                    != candidate_release.manifest.facility_source_sha256
                    or not raw_reviews_path
                    or raw_review_source_sha256
                    != candidate_release.manifest.review_source_sha256
                ):
                    raise IndexLoadError(
                        "active Phase 3 release does not match mounted source snapshots"
                    )
                search_index_release = candidate_release
                logger.info(
                    "Phase 3 index validated: version=%s facilities=%s evidence=%s",
                    candidate_release.version,
                    candidate_release.manifest.facility_count,
                    candidate_release.manifest.evidence_count,
                )
            except (IndexLoadError, OSError) as exc:
                if 'candidate_release' in locals():
                    candidate_release.close()
                search_index_release = None
                if SEARCH_INDEX_REQUIRED:
                    raise RequiredSearchIndexError(
                        "required Phase 3 search index failed validation"
                    ) from exc
                logger.warning("Phase 3 index unavailable: %s", exc)

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
        
        logger.info("🤖 Initializing agentic RAG (dense general + BM25 specific)...")
        rag_pipeline = RAGPipeline(
            chroma_path=CHROMA_PATH,
            openai_api_key=OPENAI_API_KEY,
            groq_client=client,
            collection_name="seoul_med_agentic_v2",
            embedding_model="text-embedding-3-small",
            raw_reviews_path=raw_reviews_path,
            raw_review_source_sha256=raw_review_source_sha256,
        )
        
        rag_pipeline.initialize_collection(df_filtered, force_recreate=False)
        
        rag_stats = rag_pipeline.get_statistics()
        logger.info(f"✅ RAG Pipeline ready:")
        logger.info(f"   Vector documents: {rag_stats['document_count']}")
        logger.info(f"   BM25 documents: {rag_stats['bm25_document_count']}")
        logger.info(f"   Specific evidence chunks: {rag_stats['specific_bm25_document_count']}")
        logger.info(f"   Raw review comments: {'ENABLED' if rag_stats['raw_reviews_enabled'] else 'DISABLED'}")
        logger.info(f"   Hybrid search: {'ENABLED ✓' if rag_stats['hybrid_search_enabled'] else 'DISABLED'}")
        
    except RequiredSearchIndexError:
        logger.exception("Required Phase 3 search index failed startup validation")
        raise
    except Exception as e:
        logger.error(f"STARTUP ERROR: {e}", exc_info=True)
        df_facilities = pd.DataFrame()
        df_filtered = pd.DataFrame()

    yield
    if search_index_release is not None:
        search_index_release.close()
        search_index_release = None
    logger.info("🛑 Shutting down.")


# ==========================================
# FASTAPI APP SETUP
# ==========================================

app = FastAPI(lifespan=lifespan)
client = build_llm_client()
chat_rate_limiter = SlidingWindowRateLimiter(
    max_requests=CHAT_RATE_LIMIT_REQUESTS,
    window_seconds=CHAT_RATE_LIMIT_WINDOW_SECONDS,
)

@app.get("/")
def root_status() -> Dict[str, Any]:
    """Small public status route for container probes and human operators."""
    return {
        "service": "SeoulDoc API",
        "status": "ready" if rag_pipeline is not None and client is not None else "starting",
        "model_provider": LLM_PROVIDER,
        "model": GROQ_CHAT_MODEL,
        "agent_model": GROQ_AGENT_MODEL,
        "search_index_version": (
            search_index_release.version if search_index_release is not None else None
        ),
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Fail health checks when startup completed without usable search data."""
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=f"Configured LLM provider '{LLM_PROVIDER}' is unavailable",
        )
    ready = (
        rag_pipeline is not None
        and df_filtered is not None
        and not df_filtered.empty
    )
    if not ready:
        raise HTTPException(status_code=503, detail="Search indexes are not ready")
    stats = rag_pipeline.get_statistics()
    return {
        "status": "ok",
        "facilities": int(len(df_filtered)),
        "vector_documents": int(stats.get("document_count", 0)),
        "raw_reviews": int(stats.get("raw_review_count", 0)),
        "model_provider": LLM_PROVIDER,
        "model": GROQ_CHAT_MODEL,
        "agent_model": GROQ_AGENT_MODEL,
        "phase3_index": {
            "active": search_index_release is not None,
            "required": SEARCH_INDEX_REQUIRED,
            "version": (
                search_index_release.version
                if search_index_release is not None else None
            ),
            "facilities": (
                search_index_release.manifest.facility_count
                if search_index_release is not None else 0
            ),
            "evidence": (
                search_index_release.manifest.evidence_count
                if search_index_release is not None else 0
            ),
        },
    }


@app.get("/ready/retrieval")
def retrieval_readiness() -> Dict[str, Any]:
    health_check()
    if search_index_release is None:
        raise HTTPException(503, detail={"ready": False, "reason": "index_unavailable"})
    probe_state = State(location="Seoul", is_citywide_search=True)
    compilation = compile_legacy_state_rules(
        "진료 설명 consultation", probe_state, turn_id="readiness",
    )
    if compilation.rules is None:
        raise HTTPException(503, detail={"ready": False, "reason": "probe_scope_unavailable"})
    scope = ScopeBuilder().build(
        df_filtered, compilation.rules, index_version=search_index_release.version,
    )
    with search_index_release.within(scope) as scoped:
        report = probe_retrieval(scoped, semantic_evidence_source, evidence_reranker)
    if not report["ready"]:
        raise HTTPException(503, detail=report)
    return report


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://seouldoc.io",
        "https://www.seouldoc.io",
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
    Uses intent-based classification: "I want X" → positive, "avoid X" → negative
    """
    if not consent:
        consent = CookieConsent()
    
    specialty_list = ", ".join(available_specialties[:50]) if available_specialties else "No specialties available"
    travel_labels_list = ", ".join([f'"{label}" ({dist}km)' for label, dist in DISTANCE_MAPPING.items()])
    
    # ============================================
    # ULTRA-SIMPLE INTENT-BASED EXTRACTION PROMPT
    # ============================================
    extraction_prompt = f"""Extract from: "{user_message}"

Return a DELTA for this user turn, not a rewritten conversation state.

**TURN OPERATION:**
- "refine": add, set, or remove constraints in the current search.
- "replace_context": the user explicitly starts over or changes the whole request.

**DELTA SAFETY RULES:**
- "X is no longer required" or "remove X" → remove_terms: [X]. Never negate X.
- Opening times such as Tuesday evening go in required_hours, not review keywords.
- Copy an explicit numeric distance exactly into distance_km; never round it.

**KEYWORD INTENT RULES (READ CAREFULLY):**
1. "I want X" / "I need X" / "find X" / "looking for X" / "X doctor" → soft_keywords: [X] ✓
2. "avoid X" / "not X" / "without X" / "don't want X" → negative_keywords: [X] ✓

**X can be ANY quality - unfriendly, rude, expensive, rushed, cold, impolite, etc.**

**CRITICAL: Look at the VERB/INTENT, not whether the quality sounds "good" or "bad"!**

**EXAMPLES (STUDY THESE):**
✓ "I need an unfriendly doctor" → soft: ["unfriendly"], negative: []
✓ "rude doctor" → soft: ["rude"], negative: []
✓ "impolite doctor with parking" → soft: ["impolite"], hard: ["parking"], negative: []
✓ "cold and direct doctor" → soft: ["cold", "direct"], negative: []
✓ "expensive clinic" → soft: ["expensive"], negative: []
✓ "avoid friendly staff" → soft: [], negative: ["friendly"]
✓ "not polite doctors" → soft: [], negative: ["polite"]

**Available specialties:** {specialty_list}
**Travel labels:** {travel_labels_list}

**Return JSON with:**
- operation: "refine" or "replace_context"
- specialty: matched Korean specialty only when this turn explicitly requests or changes it; otherwise null. Never emit a generic default on a follow-up.
- specialty_confidence: 0.0-1.0
- location: extracted location or null (null if not mentioned)
- distance_km: exact numeric distance or null
- travel_label: one from travel labels list, or null when travel is not mentioned
- hard_keywords: explicitly mandatory factual requirements only. A question about availability is not a requirement.
- soft_keywords: subjective qualities user WANTS (can include unfriendly, rude, cold, expensive, etc.)
- negative_hard_keywords: factual exclusions (without parking, no weekend hours)
- negative_keywords: qualities to AVOID (avoid friendly, not polite)
- place_terms: literal place/district/landmark phrases
- gender_terms: requested doctor gender, preserving the user's wording
- disease_terms: diseases, conditions, or symptoms to treat
- comment_terms: qualities that should be supported by patient comments/reviews
- required_hours: canonical availability constraints
- remove_terms: explicitly withdrawn concepts. Waiting being acceptable withdraws a prior short-wait preference.
- visit_reason: current visit purpose, including a routine checkup, or null when unchanged
- inquiries: exact current-turn clauses asking factual questions without imposing a requirement
- term_operations: explicit changes with action add/remove/replace, field, term, optional replacement, and exact source_span
  Fields: keywords, hard_keywords, negative_keywords, negative_hard_keywords, comment_terms, gender_terms, disease_terms, required_hours.
  Removal and replacement require a clause explicitly withdrawing or replacing that concept.
  Do not classify words in "check directly" as a doctor-personality preference.
  Interpret polarity within each clause. Do not apply one clause's negation to the entire message.
  Retain restrained-prescribing preferences, staff roles and the distinction between questions and mandatory requirements.

**Format:**
{{
  "operation": "refine",
  "specialty": "string or null",
  "specialty_confidence": 0.7,
  "location": "string or null",
  "latitude": null,
  "longitude": null,
  "distance_km": null,
  "travel_label": null,
  "language_pref": "English Preferred",
  "hard_keywords": [],
  "soft_keywords": [],
  "negative_hard_keywords": [],
  "negative_keywords": [],
  "place_terms": [],
  "gender_terms": [],
  "disease_terms": [],
  "comment_terms": [],
  "required_hours": [],
  "remove_terms": [],
  "visit_reason": null,
  "inquiries": [],
  "term_operations": []
}}
"""
    
    extraction_messages = [{"role": "system", "content": extraction_prompt}]
    
    try:
        # ============================================
        # CALL LLM WITH HIGHER TEMPERATURE
        # ============================================
        extracted_payload, _ = request_json_completion(
            client,
            model=GROQ_CHAT_MODEL,
            messages=extraction_messages,
            temperature=0.2,
            max_completion_tokens=1024,
            required_keys=("specialty", "location", "travel_label"),
        )
        extracted = augment_extracted_facets(
            user_message,
            extracted_payload,
        )
        
        # ============================================
        # DEBUG LOGGING: RAW EXTRACTION
        # ============================================
        privacy_safe_log(consent, "=" * 60)
        privacy_safe_log(consent, "🔍 RAW EXTRACTION (before post-processing):")
        privacy_safe_log(consent, f"   📝 Input: '{user_message}'")
        privacy_safe_log(consent, f"   🏥 Specialty: {extracted.get('specialty', 'null')} (conf: {extracted.get('specialty_confidence', 0):.2f})")
        privacy_safe_log(consent, f"   🔒 Hard Keywords: {extracted.get('hard_keywords', [])}")
        privacy_safe_log(consent, f"   💭 Soft Keywords: {extracted.get('soft_keywords', [])}")
        privacy_safe_log(consent, f"   ⛔ Negative Hard: {extracted.get('negative_hard_keywords', [])}")
        privacy_safe_log(consent, f"   🚫 Negative Keywords: {extracted.get('negative_keywords', [])}")
        privacy_safe_log(consent, "=" * 60)
        
        # ============================================
        # STANDARD PROCESSING: SPECIALTY, TRAVEL, LOCATION
        # ============================================
        
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
                extracted['travel_label'] = None
                privacy_safe_log(consent,
                    "⚠️ Invalid travel label ignored")
        else:
            privacy_safe_log(consent, "ℹ️ No travel update in this message")
        
        # ============================================
        # FINAL KEYWORD LOGGING
        # ============================================
        if extracted.get('hard_keywords'):
            privacy_safe_log(consent, f"🔒 Hard keywords (MUST): {extracted['hard_keywords']}")
        
        if extracted.get('soft_keywords'):
            privacy_safe_log(consent, f"💭 Soft keywords (preferences): {extracted['soft_keywords']}")
        
        if extracted.get('negative_hard_keywords'):
            privacy_safe_log(consent, f"⛔ Negative hard keywords (EXCLUDE): {extracted['negative_hard_keywords']}")
        
        if extracted.get('negative_keywords'):
            privacy_safe_log(consent, f"🚫 Negative keywords (avoid): {extracted['negative_keywords']}")
        
        # ============================================
        # LOCATION VERIFICATION
        # ============================================
        if extracted.get('location'):
            privacy_safe_log(consent, "📍 Location extraction received")
            verified = verify_and_standardize_address(extracted['location'], consent=consent)
            
            if verified:
                extracted['latitude'] = verified['lat']
                extracted['longitude'] = verified['lon']
                extracted['address_korean'] = verified['address_korean']
                extracted['district'] = verified['district']
                extracted['dong'] = verified['dong']
                
                privacy_safe_log(consent,
                    f"✅ Geocoding verified: {verified['district']} ({verified['lat']:.4f}, {verified['lon']:.4f})")
            else:
                logger.warning("Could not verify an extracted location")
        
        privacy_safe_log(consent, "=" * 60 + "\n")
        record_shadow_rule_compilation(user_message, extracted)
        
        return extracted
        
    except Exception as e:
        logger.error(f"Extraction Error: {e}", exc_info=True)
        fallback = augment_extracted_facets(user_message, {
            "travel_label": "Moderate", 
            "hard_keywords": [], 
            "soft_keywords": [],
            "negative_hard_keywords": [],
            "negative_keywords": []
        })
        fallback["extraction_source"] = "deterministic_fallback"
        fallback["extraction_error"] = type(e).__name__
        record_shadow_rule_compilation(user_message, fallback)
        return fallback

def merge_extraction_into_state(
    state: State,
    extracted: Dict[str, Any],
    user_message: str = "",
) -> State:
    """Apply one model proposal through the server-owned state reducer."""
    delta = compile_turn_delta(user_message, extracted)
    next_state = reduce_search_state(state, delta)
    logger.info("Applied validated search-state delta")
    return next_state


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
    global df_facilities
    
    if not consent:
        consent = CookieConsent()

    language = state.language_pref or "English"
    
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
        
        if language == "English":
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
        
        if language == "English":
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
        
        if language == "English":
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
    
    if language == "English":
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
    
    return response_text, serialize_results_for_chat(
        results, include_debug=ENABLE_RETRIEVAL_DEBUG
    )

def filter_by_zone(df: pd.DataFrame, district: str, dong: Optional[str] = None) -> pd.DataFrame:
    """Filter facilities by zone (district and optionally dong)."""
    if not district:
        return df
    
    logger.info("Zone filter applied")
    
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


def select_optimal_result_count(frame: pd.DataFrame) -> int:
    count = len(frame)
    return min(count, 5) if count <= 5 or count > 10 else 4


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
    - ⭐ City-wide search support (no location bias, no distance calculation)
    """
    global rag_pipeline
    
    if not consent:
        consent = CookieConsent()

    state.clear_retrieval_telemetry()
    state.last_retrieval_run_id = uuid4().hex
    language = state.language_pref or "English"
    
    state.last_search_query = user_message
    state.last_search_timestamp = datetime.utcnow().isoformat()
    
    # ===== DETERMINE SEARCH PRECISION LEVEL =====
    
    specialty_conf = state.specialty_confidence
    has_specialty = bool(state.specialty)
    is_general_search = getattr(state, 'is_general_search', False)
    is_citywide = getattr(state, 'is_citywide_search', False)
    
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
    
    if is_citywide:
        privacy_safe_log(consent, f"City-wide: YES (no distance filtering)")
    else:
        privacy_safe_log(consent, f"Max Distance: {max_distance}km")
    
    privacy_safe_log(consent, f"Language: {language}")
    
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
    
    # ⭐ PRIORITY CHECK: City-wide search (NO distance calculation at all)
    if is_citywide:
        privacy_safe_log(consent, "🌆 CITY-WIDE SEARCH MODE")
        privacy_safe_log(consent, "   No location specified by user → No distance calculation")
        privacy_safe_log(consent, "   Searching all 25 districts equally")
        
        location_context = "across Seoul"
        
        # ⚠️ DO NOT add distance_km column for city-wide searches
        # Results will be ranked purely by relevance, not distance
        
        privacy_safe_log(consent, f"✓ City-wide pool: {len(working_df)} facilities (all districts)")
    
    # Zone-based search (specific district)
    elif state.search_mode == 'zone' and state.district:
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
    
    # GPS-based search (specific coordinates)
    elif user_lat and user_lon:
        privacy_safe_log(consent, f"📍 GPS search: ({user_lat:.4f}, {user_lon:.4f})")
        
        # Reverse geocode
        reverse_result = None
        if GOOGLE_MAPS_API_KEY:
            reverse_result = google_maps_reverse_geocode(user_lat, user_lon, consent=consent)
        
        if not reverse_result and KAKAO_REST_API_KEY:
            reverse_result = kakao_reverse_geocode(user_lat, user_lon, consent=consent)
        
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
            location_context = "null"
            logger.warning("⚠️ No GPS coordinates in dataset")
    
    # Text location (fuzzy match)
    elif state.location and state.location.strip().lower() not in ["seoul", "서울", "서울시"]:
        privacy_safe_log(consent, f"📍 Text location: {state.location}")
        working_df = fuzzy_match_location(state.location, working_df)
        location_context = f"in {state.location}"
        working_df['distance_km'] = 0
        privacy_safe_log(consent, f"✓ Fuzzy location match: {len(working_df)} facilities")
    
    # Fallback: No specific location (should have been caught by is_citywide, but safety check)
    else:
        privacy_safe_log(consent, "📍 No specific location (defaulting to city-wide)")
        location_context = "across Seoul"
        
        # ⚠️ DO NOT add distance_km column
        
        # Mark as city-wide if not already marked
        if not is_citywide:
            state.is_citywide_search = True
            is_citywide = True
            privacy_safe_log(consent, "   ⚠️ Auto-detecting as city-wide search")
    
    # ===== CHECK IF DISTANCE DATA EXISTS =====
    has_distance_data = 'distance_km' in working_df.columns
    
    if has_distance_data:
        privacy_safe_log(consent, f"✓ Distance data available for sorting/filtering")
    else:
        privacy_safe_log(consent, f"ℹ️ No distance data (city-wide search)")
    
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
            
            # Sort by distance only if we have distance data
            if has_distance_data:
                general_fallback_df = general_fallback_df.sort_values('distance_km').head(10)
            else:
                # City-wide: take first 10
                general_fallback_df = general_fallback_df.head(10)
            
            privacy_safe_log(consent, f"   ✓ Added {len(general_fallback_df)} general facilities")
            
            # Merge: specialty first, then general
            working_df = pd.concat([specialty_filtered_df, general_fallback_df], ignore_index=True)
            relaxed_filters.append(f"included {len(general_fallback_df)} nearby general facilities")
        else:
            working_df = specialty_filtered_df
    
    else:
        # LOW PRECISION: No specialty filter
        privacy_safe_log(consent, f"🎲 LOW PRECISION: All facility types")
        
        # Sort by distance only if we have distance data
        if has_distance_data:
            working_df = working_df.sort_values('distance_km')
            privacy_safe_log(consent, "   Sorted by distance")
        else:
            privacy_safe_log(consent, "   No distance sorting (city-wide search)")
    
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
                        return calculate_field_boost(val, keyword_lower)
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
                            return calculate_field_boost(val, neg_keyword_lower)
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

            # ⭐ MASSIVELY INCREASED KEYWORD WEIGHTS
            # Hard keywords: CRITICAL priority (was 500 → now 2000)
            if state.hard_keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(
                    path_b_pool, state.hard_keywords, 2000.0, "HARD"  # ⭐ 4x increase
                )

            # Soft keywords: STRONG preference (was 50 → now 500)
            if state.keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(
                    path_b_pool, state.keywords, 500.0, "SOFT"  # ⭐ 10x increase
                )

            # Negative keywords: SEVERE penalty (was -300 → now -1500)
            if state.negative_keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(
                    path_b_pool, state.negative_keywords, -1500.0, "NEGATIVE"  # ⭐ 5x increase
                )

            # Negative hard keywords: EXTREME penalty (was -1000 → now -5000)
            if state.negative_hard_keywords:
                path_b_pool['relevance_boost'] += apply_keyword_boost(
                    path_b_pool, state.negative_hard_keywords, -5000.0, "NEGATIVE_HARD"  # ⭐ 5x increase
                )

            # Specialty match boost (reduced to give keywords more relative importance)
            if search_precision != "LOW" and state.specialty:
                spec_mask = path_b_pool['category'].fillna('').astype(str).str.lower().str.contains(
                    state.specialty.lower(), na=False
                )
                boost_amount = 100.0 * specialty_conf  # ⭐ Reduced from 200 to 100 (keywords now dominate)
                path_b_pool.loc[spec_mask, 'relevance_boost'] += boost_amount

            # Distance decay (REDUCED to let keywords dominate more)
            if has_distance_data:
                max_dist = path_b_pool['distance_km'].max() or 1.0
                if max_dist > 0:
                    distance_penalty = (path_b_pool['distance_km'] / max_dist) * 10  # ⭐ Reduced from 30 to 10
                    path_b_pool['relevance_boost'] -= distance_penalty
                    privacy_safe_log(consent, "   Applied distance decay to scoring (reduced weight)")
            else:
                privacy_safe_log(consent, "   Skipping distance decay (no distance data)")
                        
                
            # Sort by boost (and distance if available)
            if has_distance_data:
                path_b_pool = path_b_pool.sort_values(
                    by=['relevance_boost', 'distance_km'], 
                    ascending=[False, True]
                ).reset_index(drop=True)
            else:
                path_b_pool = path_b_pool.sort_values(
                    by='relevance_boost', 
                    ascending=False
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

    # ScopeBuilder is authoritative during the legacy ranking migration. The
    # heuristic work above may rank candidates, but it cannot add an ID here.
    scope_compilation = compile_legacy_state_rules(
        user_message,
        state,
        turn_id=state.last_retrieval_run_id,
        allowed_specialties=available_specialties,
    )
    scope_selection = None
    scope_metadata = {}
    authoritative_max_distance = 5.0
    if scope_compilation.rules is None:
        logger.warning(
            "Authoritative scope needs clarification: %s",
            [issue.code for issue in scope_compilation.issues],
        )
        working_df = df_filtered.iloc[0:0].copy()
        final_df = working_df.copy()
        has_distance_data = False
        state.last_retrieval_metadata = {
            "retrieval_execution_status": "not_run",
            "retrieval_reason_codes": ["scope_unresolved"],
        }
    else:
        authoritative_rules = scope_compilation.rules
        scope_selection = ScopeBuilder().build(
            df_filtered,
            authoritative_rules,
            index_version=(
                search_index_release.version
                if search_index_release is not None
                else LEGACY_INDEX_VERSION
            ),
        )
        working_df = scope_selection.restrict_dataframe(df_filtered)
        geography = authoritative_rules.hard.geography
        if isinstance(geography, DistanceRule):
            authoritative_max_distance = geography.max_km
            state.max_distance_km = geography.max_km
            has_distance_data = "distance_km" in working_df.columns
            if has_distance_data:
                working_df = working_df.sort_values("distance_km")
            location_context = "near your location"
            is_citywide = False
        else:
            authoritative_max_distance = 5.0
            has_distance_data = False
            is_citywide = geography.area_kind == "city"
            location_context = (
                "across Seoul"
                if is_citywide
                else f"in {geography.display_name}"
            )
        final_df = working_df.copy()
        scope_metadata = {
            "scope_digest": scope_selection.descriptor.scope_digest,
            "rules_hash": scope_selection.descriptor.rules_hash,
            "candidate_scope_count": scope_selection.descriptor.facility_count,
        }
        state.last_retrieval_metadata = {
            "run_id": state.last_retrieval_run_id,
            **scope_metadata,
            "retrieval_status": "scope_built",
            "retrieval_execution_status": "not_run",
            "retrieval_reason_codes": [],
        }
        privacy_safe_log(
            consent,
            "Authoritative eligible scope: "
            f"{scope_selection.descriptor.facility_count} facilities",
        )

    path_a_results_df = final_df.copy()
    path_b_results_df = final_df.iloc[0:0].copy()
    specialty_filtered_df = final_df.copy()
    general_fallback_df = final_df.iloc[0:0].copy()
    relaxed_filters = []
    
    # ===== AGENTIC RAG: LLM → SEARCH → OBSERVATION LOOP =====

    exact_retrieval_terms = retrieval_terms_from_state(state)
    has_retrieval_preferences = bool(state.keywords or exact_retrieval_terms or state.inquiries or state.visit_reason)
    should_run_rag = search_index_release is not None or search_precision != "LOW" or has_retrieval_preferences

    if should_run_rag:
        query_components = [state.specialty or "medical facility"]
        if state.visit_reason:
            query_components.append(state.visit_reason)
        query_components.extend(state.place_terms)
        query_components.extend(state.inquiries)
        query_components.extend(state.keywords)
        query_components.extend(exact_retrieval_terms)
        query_text = " ".join(dict.fromkeys(part for part in query_components if part))
        
        if query_text.strip() and len(final_df) > 0 and rag_pipeline:
            # Give agentic retrieval the complete specialty/location/distance
            # scope, not only the 3-5 legacy heuristic finalists. This makes
            # every eligible facility's comments reachable by the multilingual
            # retrieval loop; evidence scoring still determines the final rank.
            if len(working_df) > len(final_df):
                final_ids = set(final_df["place_id"].astype(str))
                remaining_candidates = working_df[
                    ~working_df["place_id"].astype(str).isin(final_ids)
                ]
                final_df = pd.concat(
                    [final_df, remaining_candidates],
                    ignore_index=True,
                )
                privacy_safe_log(
                    consent,
                    f"🌐 Expanded RAG evidence scope to {len(final_df)} eligible facilities",
                )

            privacy_safe_log(consent, "🤖 Applying agentic RAG retrieval loop...")
            
            try:
                state.query_intent = "AGENTIC"
                state.suggested_alpha = None
                state.hybrid_alpha = None
                
                retrieval = CandidateRetrievalAdapter(
                    active_index=search_index_release,
                    legacy_pipeline=rag_pipeline,
                    evidence_reranker=evidence_reranker,
                    semantic_evidence_source=semantic_evidence_source,
                    evidence_policy=EvidenceRecallPolicy(require_remote_services=True),
                ).rank(
                    scope=scope_selection,
                    eligible=final_df,
                    rules=authoritative_rules,
                    query=RetrievalQuery(
                        text=query_text,
                        max_distance_km=authoritative_max_distance,
                        search_mode=state.search_mode or "distance",
                        specialty_confidence=specialty_conf,
                        exact_terms=tuple(exact_retrieval_terms),
                        target_language=language,
                        manual_mode=state.manual_search_mode,
                        display_limit=select_optimal_result_count(final_df),
                    ),
                )
                final_df = retrieval.dataframe
                state.last_retrieval_trace = final_df.attrs.get("rag_trace", [])
                state.last_retrieval_observations = final_df.attrs.get("rag_observations", [])
                state.last_retrieval_metadata = {
                    **final_df.attrs.get("rag_metadata", {}),
                    "run_id": state.last_retrieval_run_id,
                    **scope_metadata,
                }
                if ENABLE_RETRIEVAL_DEBUG:
                    retrieval_by_id = {
                        item["place_id"]: item
                        for item in final_df.attrs.get("rag_candidates", [])
                    }
                    state.last_retrieval_candidates = [
                        {
                            **retrieval_by_id.get(str(row["place_id"]), {
                                "place_id": str(row["place_id"]),
                                "retrieval_rank_1based": None,
                                "methods": [],
                            }),
                            "combined_rank_1based": combined_rank,
                        }
                        for combined_rank, (_, row) in enumerate(
                            final_df.head(RETRIEVAL_DEBUG_LIMIT).iterrows(),
                            start=1,
                        )
                    ]
                
                privacy_safe_log(
                    consent,
                    "Indexed scoped hybrid ranking applied "
                    f"({retrieval.telemetry.status})",
                )
                
            except Exception as e:
                logger.error(f"Agentic RAG ranking error: {e}", exc_info=True)
                state.last_retrieval_metadata = {
                    "run_id": state.last_retrieval_run_id,
                    **scope_metadata,
                    "retrieval_status": "error",
                    "retrieval_execution_status": "failed",
                    "retrieval_reason_codes": ["retrieval_exception"],
                    "termination_reason": type(e).__name__,
                }
                final_df['relevance_rank'] = 9999
                # Sort by distance only if we have it
                if has_distance_data:
                    final_df = final_df.sort_values('distance_km')
        else:
            state.query_intent = None
            state.suggested_alpha = None
            state.hybrid_alpha = None
            state.last_retrieval_metadata = {
                "run_id": state.last_retrieval_run_id, **scope_metadata,
                "retrieval_execution_status": "failed" if len(final_df) and not rag_pipeline else "not_run",
                "retrieval_reason_codes": ["retrieval_unavailable"] if len(final_df) and not rag_pipeline else (["scope_unresolved"] if scope_selection is None else []),
            }
    else:
        # Truly general request with no textual preferences: distance is enough.
        privacy_safe_log(consent, "📍 General request → distance-only ranking")
        
        # Sort by distance only if we have it
        if has_distance_data:
            final_df = final_df.sort_values('distance_km')
            privacy_safe_log(consent, "   Sorted by distance")
        else:
            privacy_safe_log(consent, "   Using natural order (no distance)")
        
        state.query_intent = "GENERAL"
        state.hybrid_alpha = None
        state.last_retrieval_metadata = {
            "run_id": state.last_retrieval_run_id,
            **scope_metadata,
            "retrieval_status": "not_run",
            "retrieval_execution_status": "not_run",
            "retrieval_reason_codes": [],
            "termination_reason": "general_distance_only",
            "candidate_scope_count": int(len(final_df)),
        }
    
    # ===== FINAL SCOPE VALIDATION =====

    if scope_selection is not None:
        scope_selection.assert_contains_only(final_df)
        final_df = scope_selection.restrict_dataframe(final_df)
    
    # ===== RESULT SELECTION =====
    
    # ===== GENERATE RESPONSE =====
    
    results = []
    if len(final_df) > 0:
        n_results = select_optimal_result_count(final_df)
        privacy_safe_log(consent, f"✓ Selecting {n_results} results from {len(final_df)} facilities")
        # ===== PREPARE RESULTS FOR FRONTEND =====
        
        for idx, (_, row) in enumerate(final_df.head(n_results).iterrows(), start=1):
            result = {
                "entity_type": "facility",
                "final_rank": idx,
                "place_id": safe_convert_to_python(row['place_id']),
                "name": safe_convert_to_python(row['name']),
                "category": safe_convert_to_python(row['category']),
            }
            
            simple_fields = ['address', 'phone', 'business_hours']
            for field in simple_fields:
                if field in row.index:
                    value = safe_convert_to_python(row[field])
                    if value is not None:
                        result[field] = value
            
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
            
            if 'amenities' in row.index:
                result['amenities'] = safe_convert_to_python(row['amenities'])
            
            if 'medical_info_parsed' in row.index and isinstance(row['medical_info_parsed'], dict):
                result['medical_info_parsed'] = safe_convert_to_python(row['medical_info_parsed'])

            if 'retrieval_evidence' in row.index and isinstance(row['retrieval_evidence'], list):
                result['retrieval_evidence'] = safe_convert_to_python(row['retrieval_evidence'])
            if (
                'retrieval_evidence_groups' in row.index
                and isinstance(row['retrieval_evidence_groups'], dict)
            ):
                result['retrieval_evidence_groups'] = safe_convert_to_python(
                    row['retrieval_evidence_groups']
                )


            if 'retrieval_methods' in row.index and isinstance(row['retrieval_methods'], list):
                result['retrieval_methods'] = safe_convert_to_python(row['retrieval_methods'])

            if 'retrieval_matched_terms' in row.index and isinstance(row['retrieval_matched_terms'], list):
                result['retrieval_matched_terms'] = safe_convert_to_python(row['retrieval_matched_terms'])

            rag_trace = final_df.attrs.get('rag_trace', [])
            if rag_trace:
                result['retrieval_trace'] = safe_convert_to_python(rag_trace)
            
            
            # ⭐ CRITICAL: Only include distance if it exists in the data
            if has_distance_data and 'distance_km' in row.index:
                distance_value = safe_convert_to_python(row['distance_km'])
                result['distance_km'] = distance_value
                result['distance'] = distance_value
            else:
                # ⚠️ DO NOT include distance fields for city-wide searches
                pass
            
            result['relevance_rank'] = safe_convert_to_python(row.get('relevance_rank', 9999))
            
            if 'relevance_boost' in row.index:
                result['relevance_boost'] = safe_convert_to_python(row['relevance_boost'])
            
            if 'combined_score' in row.index:
                result['combined_score'] = safe_convert_to_python(row['combined_score'])
            
            results.append(result)
            
            # Logging
            boost_value = result.get('relevance_boost')
            boost_info = f" boost={boost_value:.1f}" if boost_value is not None else ""
            
            if is_citywide or not has_distance_data:
                # City-wide: show district instead of distance
                district_info = result.get('district', 'Unknown')
                privacy_safe_log(consent, f"   #{idx}: {result['name']} ({result['category']}) in {district_info}{boost_info}")
            else:
                # Location-specific: show distance
                distance_value = result.get('distance_km', 0)
                privacy_safe_log(consent, f"   #{idx}: {result['name']} ({result['category']}) {distance_value:.1f}km{boost_info}")
    
    # ===== UPDATE STATE METADATA =====
    
    state.last_results_count = len(results)
    
    privacy_safe_log(consent, "=" * 60)
    privacy_safe_log(consent, f"✅ ADAPTIVE SEARCH COMPLETED (Precision: {search_precision})")
    privacy_safe_log(consent, f"   Mode: {'CITY-WIDE (no distance)' if is_citywide else 'LOCATION-SPECIFIC'}")
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

    outcome = answer_search(
        question=user_message,
        state=state,
        cards=results,
        metadata=state.last_retrieval_metadata,
        language=language,
        complete=partial(request_answer_completion, client, model=GROQ_CHAT_MODEL) if client else None,
        translation_api_key=os.getenv("GOOGLE_TRANSLATE_API_KEY", ""),
    )
    state.last_retrieval_metadata["answer"] = {"model": GROQ_CHAT_MODEL, **outcome.trace}
    return outcome.text, serialize_results_for_chat(
        outcome.cards, include_debug=ENABLE_RETRIEVAL_DEBUG,
    )

# ==========================================
# API ENDPOINTS
# ==========================================

@app.post("/consent")
async def update_consent(
    consent: CookieConsent,
    response: Response,
    request: Request,
):
    """Update cookie consent preferences."""
    try:
        consent.timestamp = datetime.utcnow().isoformat()
        
        is_secure_request = request.url.scheme == "https"
        response.set_cookie(
            key="cookieConsent",
            value=json.dumps(consent.model_dump()),
            max_age=30 * 24 * 60 * 60,
            httponly=True,
            secure=is_secure_request,
            samesite="none" if is_secure_request else "lax",
            path="/",
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
    cookieConsent: Optional[str] = Cookie(None),
    consent_header: Optional[str] = Header(None, alias="X-Cookie-Consent"),
):
    """Direct endpoint for setting travel preference from UI widget."""
    consent = get_consent_from_cookie(consent_header or cookieConsent)
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
    state.clear_retrieval_telemetry()
    
    # Update travel preferences with HIGH confidence (widget-based)
    state.travel_label = travel_label
    state.travel_confidence = 1.0  # ⭐ HIGH confidence from widget
    state.max_distance_km = DISTANCE_MAPPING[travel_label]
    
    # Detect language from state
    language = state.language_pref or "English"
    
    privacy_safe_log(consent, f"✅ Travel preference set via widget: {travel_label} → {state.max_distance_km}km (confidence: 1.0)")
    
    # Generate response
    if language == "English":
        response_text = f"Got it! I'll search within {state.max_distance_km}km ({travel_label})."
    else:
        response_text = f"알겠습니다! {state.max_distance_km}km 반경으로 검색하겠습니다 ({travel_label})."
    
    return {
        "response": response_text,
        "state": serialize_state_for_chat(
            state, include_debug=ENABLE_RETRIEVAL_DEBUG
        ),
    }


@app.post("/chat")
def chat_endpoint(
    req: ChatRequest,
    response: Response,
    request: Request,
    cookieConsent: Optional[str] = Cookie(None),
    consent_header: Optional[str] = Header(None, alias="X-Cookie-Consent"),
):
    """
    Router-Controller Architecture with Hybrid RAG and keyword filtering.
    """
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=f"Configured LLM provider '{LLM_PROVIDER}' is unavailable",
        )

    remote_host = request.client.host if request.client else "unknown"
    client_key = sha256(remote_host.encode("utf-8")).hexdigest()
    rate_limit = chat_rate_limiter.check(client_key)
    rate_limit_headers = {
        "X-RateLimit-Limit": str(CHAT_RATE_LIMIT_REQUESTS),
        "X-RateLimit-Remaining": str(rate_limit.remaining),
    }
    if not rate_limit.allowed:
        rate_limit_headers["Retry-After"] = str(rate_limit.retry_after_seconds)
        raise HTTPException(
            status_code=429,
            detail="Chat rate limit exceeded. Try again later.",
            headers=rate_limit_headers,
        )
    for name, value in rate_limit_headers.items():
        response.headers[name] = value
    
    consent = get_consent_from_cookie(consent_header or cookieConsent)
    consent = ensure_consent_object(consent)

    if should_log_analytics(consent):
        logger.info("=" * 60)
        logger.info("📨 NEW REQUEST")
        logger.info(f"Message: \"{req.message[:50]}...\"")
    else:
        logger.info("📨 REQUEST (limited logging - no analytics consent)")
    
    language, explicit_language = response_language(
        req.message, req.current_state.language_pref,
        established=req.current_state.turn_count > 0,
        explicit=req.current_state.explicit_response_language,
    )
    
    if should_log_analytics(consent):
        logger.info("=" * 60)
    
    enriched_state = standardize_and_fill_state(req.current_state, consent)
    enriched_state.clear_retrieval_telemetry()
    enriched_state.language_pref = language
    enriched_state.explicit_response_language = explicit_language
    
    current_turn = enriched_state.turn_count + 1
    
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
        route, _ = request_json_completion(
            client,
            model=GROQ_CHAT_MODEL,
            messages=router_messages,
            temperature=0.0,
            max_completion_tokens=768,
            required_keys=("intent",),
        )
        intent = route.get("intent")
        
        if should_log_analytics(consent):
            logger.info(f"🧭 ROUTER: {intent} (confidence: {route.get('confidence', 0):.2f})")
            logger.info(f"   Reasoning: {route.get('reasoning', 'N/A')[:100]}")
            logger.info(f"   Turn: {current_turn}")
        
    except Exception as e:
        logger.error(f"Router Error: {e}", exc_info=True)
        intent = "PROVIDE_INFO"
    
    if intent == "NEW_SEARCH":
        plain_reset = re.fullmatch(
            r"(?:reset|restart|start over|new search|reset (?:the |my )?search|새로 시작|처음부터|다시 시작)[.!\s]*",
            req.message.strip(), re.I,
        )
        if not plain_reset or compile_turn_delta(req.message, {}).operation != "replace_context":
            intent = "PROVIDE_INFO"

    if intent in {"CONFIRMATION", "CHANGE_CRITERIA"}:
        intent = "PROVIDE_INFO"

    # ===== CONTROLLER =====
    
    # EMERGENCY
    if intent == "EMERGENCY":
        privacy_safe_log(consent, "🚨 BRANCH: EMERGENCY")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = language
        new_state.conversation_phase = "emergency"
        
        if not (new_state.latitude and new_state.longitude):
            privacy_safe_log(consent, "Checking for location in emergency message...")
            extracted = extract_entities(req.message, consent)
            
            if extracted.get('location') or extracted.get('latitude'):
                new_state = merge_extraction_into_state(
                    new_state, extracted, user_message=req.message
                )
                new_state = standardize_and_fill_state(new_state, consent)
        
        response_text, results = execute_emergency_search(
            new_state,
            req.message,
            consent=consent
        )
        
        return {
            "response": response_text,
            "state": serialize_state_for_chat(
                    new_state, include_debug=ENABLE_RETRIEVAL_DEBUG
                ),
            "results": results
        }
    
    # NEW_SEARCH
    elif intent == "NEW_SEARCH":
        privacy_safe_log(consent, "🔄 BRANCH: NEW_SEARCH")
        
        new_state = State()
        new_state.turn_count = 0
        new_state.language_pref = language
        
        reset_keywords = ["reset", "restart", "quit", "exit", "stop", "cancel", 
                         "새로 시작", "처음부터", "다시 시작", "그만", "종료"]
        
        is_explicit_reset = any(kw in req.message.lower() for kw in reset_keywords)
        
        if is_explicit_reset:
            response_text = get_reset_confirmation(language)
        else:
            response_text = get_greeting_message(language)
        
        return {
            "response": response_text,
            "state": serialize_state_for_chat(
                    new_state, include_debug=ENABLE_RETRIEVAL_DEBUG
                ),
            "results": []
        }

    elif intent == "PROVIDE_INFO":
        privacy_safe_log(consent, "📝 BRANCH: PROVIDE_INFO")
        
        # ============================================
        # STEP 1: EXTRACT ENTITIES FROM USER MESSAGE
        # ============================================
        plain_confirmation = re.fullmatch(r"(?:yes|ok|okay|sure|네|예|응)[.!\s]*", req.message.strip(), re.I)
        extracted = {} if plain_confirmation else extract_entities(req.message, consent)
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = language
        
        new_state = merge_extraction_into_state(
            new_state, 
            extracted, 
            user_message=req.message,
        )
        new_state = standardize_and_fill_state(new_state, consent)
        new_state.language_pref = language
        
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
        has_keywords = bool(
            new_state.keywords
            or new_state.hard_keywords
            or new_state.gender_terms
            or new_state.disease_terms
            or new_state.comment_terms
        )
        
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
            privacy_safe_log(consent, f"   Hard Keywords: {new_state.hard_keywords or 'none'}")
            privacy_safe_log(consent, f"   Soft Keywords: {new_state.keywords or 'none'}")
            privacy_safe_log(consent, f"   Negative Hard: {new_state.negative_hard_keywords or 'none'}")
            privacy_safe_log(consent, f"   Negative Soft: {new_state.negative_keywords or 'none'}")
            privacy_safe_log(consent, f"   Place Facets: {new_state.place_terms or 'none'}")
            privacy_safe_log(consent, f"   Gender Facets: {new_state.gender_terms or 'none'}")
            privacy_safe_log(consent, f"   Disease Facets: {new_state.disease_terms or 'none'}")
            privacy_safe_log(consent, f"   Comment Facets: {new_state.comment_terms or 'none'}")
            
            response_text, results = execute_search(
                new_state, 
                req.message, 
                max_distance=new_state.max_distance_km, 
                consent=consent
            )
            new_state.search_executed = True
            
            return {
                "response": response_text,
                "state": serialize_state_for_chat(
                    new_state, include_debug=ENABLE_RETRIEVAL_DEBUG
                ),
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
                "state": serialize_state_for_chat(
                    new_state, include_debug=ENABLE_RETRIEVAL_DEBUG
                ),
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
                if language == "English":
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
                if language == "English":
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
                if language == "English":
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
                "state": serialize_state_for_chat(
                    new_state, include_debug=ENABLE_RETRIEVAL_DEBUG
                ),
                "results": []
            }
    
    # CHIT_CHAT
    elif intent == "CHIT_CHAT":
        privacy_safe_log(consent, "💬 BRANCH: CHIT_CHAT")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = language
        
        is_closing = any(word in req.message.lower() for word in 
                        ["thanks", "thank you", "감사합니다", "고마워"])
        
        if is_closing:
            new_state.conversation_phase = "complete"
        
        response_text = generate_chit_chat_response(req.message, new_state)
        
        return {
            "response": response_text,
            "state": serialize_state_for_chat(
                    new_state, include_debug=ENABLE_RETRIEVAL_DEBUG
                ),
            "results": []
        }
    
    # HELP_RECOVERY
    elif intent == "HELP_RECOVERY":
        privacy_safe_log(consent, "🆘 BRANCH: HELP_RECOVERY")
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = language
        
        response_text = generate_recovery_prompt(new_state)
        
        return {
            "response": response_text,
            "state": serialize_state_for_chat(
                    new_state, include_debug=ENABLE_RETRIEVAL_DEBUG
                ),
            "results": []
        }
    
    # FALLBACK
    else:
        logger.warning(f"Unknown intent: {intent}")
        
        if language == "English":
            response_text = "I'm not sure how to help. Could you rephrase your request?"
        else:
            response_text = "잘 이해하지 못했습니다. 다시 말씀해 주시겠어요?"
        
        new_state = enriched_state.model_copy()
        new_state.turn_count = current_turn
        new_state.language_pref = language
        
        return {
            "response": response_text,
            "state": serialize_state_for_chat(
                    new_state, include_debug=ENABLE_RETRIEVAL_DEBUG
                ),
            "results": []
        }

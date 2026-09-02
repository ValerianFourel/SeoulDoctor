from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated, Optional, List, Dict, Any


ChatMessage = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]

class State(BaseModel):
    """
    Comprehensive conversation state with complete search parameters.
    Maintains 1:1 mapping between frontend and backend.
    """
    # ===== SPECIALTY INFORMATION =====
    specialty: Optional[str] = None
    specialty_confidence: float = 0.0
    
    # ===== LOCATION INFORMATION =====
    # User-provided location (text)
    location: Optional[str] = "Seoul"
    
    # GPS coordinates
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Verified address data (from geocoding APIs)
    address_korean: Optional[str] = None  # Standardized Korean address
    district: Optional[str] = None  # 구 (district)
    dong: Optional[str] = None  # 동 (neighborhood)
    
    # ===== SEARCH PARAMETERS =====
    # Search mode: 'zone' (area-based) or 'distance' (radius-based)
    search_mode: Optional[str] = None  # 'zone' or 'distance'
    
    # Maximum search distance (km) for distance-based search
    max_distance_km: float = 5.0
    # User's travel label (semantic label)
    travel_label: str = "Moderate"  # Default label
    travel_confidence: float = 0.5  # NEW: Track how travel preference was set (0.6 for text, 1.0 for widget)

    
    # ===== KEYWORD FILTERING =====
    # Soft keywords: used for semantic search and ranking (POSITIVE preferences)
    keywords: List[str] = Field(default_factory=list)
    
    # Hard keywords: MUST appear in results (strict filtering) (POSITIVE requirements)
    hard_keywords: List[str] = Field(default_factory=list)
    
    # ⭐ NEW: Negative keywords (things to AVOID)
    negative_keywords: List[str] = Field(default_factory=list)  # Soft negatives (avoid these qualities)
    negative_hard_keywords: List[str] = Field(default_factory=list)  # Hard negatives (must NOT have)

    # First-class facets used by the bilingual retrieval/debug pipeline.
    place_terms: List[str] = Field(default_factory=list)
    gender_terms: List[str] = Field(default_factory=list)
    disease_terms: List[str] = Field(default_factory=list)
    comment_terms: List[str] = Field(default_factory=list)
    extraction_source: Optional[str] = None
    extraction_error: Optional[str] = None
    # ⭐ NEW: Track if this is a general/random search
    is_general_search: bool = False
    # ⭐ NEW: Track if this is a city-wide search (no specific location)
    is_citywide_search: bool = False
    
    # ===== HYBRID SEARCH PARAMETERS =====
    # Alpha coefficient for hybrid search (0.0 = pure keyword, 1.0 = pure semantic)
    # Automatically calculated by query router, but can be overridden
    hybrid_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Query classification: "FACTUAL" or "MIXED"
    query_intent: Optional[str] = None  # "FACTUAL" or "MIXED"
    
    # Suggested alpha from query router (before clipping)
    suggested_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Manual override for search mode (bypasses automatic routing)
    manual_search_mode: Optional[str] = None  # "FACTUAL_ONLY", "MIXED", or None for auto
    
    # ===== USER PREFERENCES =====
    language_pref: str = "English"
    
    # ===== CONVERSATION FLOW =====
    turn_count: int = 0
    ready_to_search: bool = False
    search_executed: bool = False
    conversation_phase: str = "greeting"  # greeting, gathering, searching, complete
    
    # ===== SEARCH RESULTS METADATA =====
    # Track last search parameters for debugging
    last_search_query: Optional[str] = None
    last_results_count: Optional[int] = None
    last_search_timestamp: Optional[str] = None
    last_retrieval_trace: List[Dict[str, Any]] = Field(default_factory=list)
    last_retrieval_observations: List[Dict[str, Any]] = Field(default_factory=list)
    last_retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    last_retrieval_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    last_retrieval_run_id: Optional[str] = None

    def clear_retrieval_telemetry(self) -> None:
        """Discard client-carried diagnostics before processing a new request."""
        self.last_retrieval_trace = []
        self.last_retrieval_observations = []
        self.last_retrieval_metadata = {}
        self.last_retrieval_candidates = []
        self.last_retrieval_run_id = None
    
    class Config:
        arbitrary_types_allowed = True


PRIVATE_CHAT_STATE_FIELDS = frozenset({
    "last_search_query",
    "last_results_count",
    "last_search_timestamp",
    "last_retrieval_trace",
    "last_retrieval_observations",
    "last_retrieval_metadata",
    "last_retrieval_candidates",
    "last_retrieval_run_id",
})
PRIVATE_RESULT_FIELDS = frozenset({
    "combined_score",
    "relevance_boost",
    "relevance_rank",
    "retrieval_matched_terms",
    "retrieval_methods",
    "retrieval_trace",
})
PUBLIC_RESULT_FIELDS = frozenset({
    "Key_Highlights",
    "Summaries",
    "Summaries_Korean",
    "address",
    "amenities",
    "business_hours",
    "category",
    "distance",
    "distance_km",
    "district",
    "dong",
    "english_confidence_score",
    "entity_type",
    "final_rank",
    "has_english",
    "is_emergency",
    "lat",
    "lon",
    "medical_info_parsed",
    "name",
    "phone",
    "place_id",
    "retrieval_evidence",
    "website",
})
PUBLIC_EVIDENCE_FIELDS = frozenset({
    "evidence_id",
    "is_verbatim",
    "language",
    "place_id",
    "relevance_reason",
    "source_field",
    "source_type",
    "text",
    "translated_text",
    "visit_date",
})


def serialize_state_for_chat(
    state: State,
    *,
    include_debug: bool,
) -> Dict[str, Any]:
    """Serialize patient state without evaluation-only telemetry."""
    serialized = state.model_dump()
    if include_debug:
        return serialized
    return {
        field: value
        for field, value in serialized.items()
        if field not in PRIVATE_CHAT_STATE_FIELDS
    }


def serialize_results_for_chat(
    results: List[Dict[str, Any]],
    *,
    include_debug: bool,
) -> List[Dict[str, Any]]:
    """Serialize recommendations while retaining patient-facing review excerpts."""
    if include_debug:
        return results

    serialized_results: List[Dict[str, Any]] = []
    for result in results:
        public_result = {
            field: value
            for field, value in result.items()
            if field in PUBLIC_RESULT_FIELDS
        }
        evidence = result.get("retrieval_evidence")
        if isinstance(evidence, list):
            public_result["retrieval_evidence"] = [
                {
                    field: value
                    for field, value in item.items()
                    if field in PUBLIC_EVIDENCE_FIELDS
                }
                for item in evidence
                if isinstance(item, dict)
            ]
        serialized_results.append(public_result)
    return serialized_results


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: ChatMessage
    current_state: State


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    state: State
    results: List[dict] = Field(default_factory=list)

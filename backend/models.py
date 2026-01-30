from pydantic import BaseModel, Field
from typing import Optional, List

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
    
    class Config:
        arbitrary_types_allowed = True


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    current_state: State


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    state: State
    results: List[dict] = []
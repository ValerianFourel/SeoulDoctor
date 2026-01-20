from pydantic import BaseModel, Field
from typing import Optional, List


class State(BaseModel):
    """
    Enhanced conversation state with address verification and zone search support.
    """
    # Core search parameters
    specialty: Optional[str] = None
    specialty_confidence: float = 0.0
    location: Optional[str] = None
    
    # GPS coordinates
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Verified address data (from Kakao API)
    address_korean: Optional[str] = None  # Standardized Korean address
    district: Optional[str] = None  # 구 (district)
    dong: Optional[str] = None  # 동 (neighborhood)
    
    # Search mode
    search_mode: Optional[str] = None  # 'zone' or 'distance'
    max_distance_km: float = 5.0  # Default search radius for distance mode
    
    # User preferences
    language_pref: str = "Korean is fine"
    
    # Conversation flow
    turn_count: int = 0
    ready_to_search: bool = False
    search_executed: bool = False
    conversation_phase: str = "greeting"  # greeting, gathering, searching, complete
    
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
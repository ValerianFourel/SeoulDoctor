# --- 4. MODELS ---
class State(BaseModel):
    specialty: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    willingness_to_travel: str = "Nearby"
    language_pref: str = "Korean is fine"
    keywords: List[str] = []
    asked_for_location: bool = False
    ready_to_search: bool = False

class ChatRequest(BaseModel):
    message: str
    current_state: State

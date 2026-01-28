// components/ChatInterface.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import { Send, MapPin, Sparkles, Globe, Bug, ChevronDown, ChevronUp, X } from "lucide-react";
import Link from 'next/link';

type State = {
  // ===== SPECIALTY INFORMATION =====
  specialty: string | null;
  specialty_confidence: number;
  
  // ===== LOCATION INFORMATION =====
  location: string | null;
  latitude: number | null;
  longitude: number | null;
  address_korean: string | null;
  district: string | null;
  dong: string | null;
  
  // ===== SEARCH PARAMETERS =====
  search_mode: string | null; // 'zone' or 'distance'
  max_distance_km: number;
  willingness_to_travel: string;
  
  // ===== KEYWORD FILTERING =====
  keywords: string[];
  hard_keywords: string[];
  negative_keywords: string[];  // ← ADD THIS
  negative_hard_keywords: string[];  // ← ADD THIS
  
  // ===== HYBRID SEARCH PARAMETERS =====
  hybrid_alpha: number | null;
  query_intent: string | null; // "FACTUAL" or "MIXED"
  suggested_alpha: number | null;
  manual_search_mode: string | null; // "FACTUAL_ONLY", "MIXED", or null
  
  // ===== USER PREFERENCES =====
  language_pref: string;
  
  // ===== CONVERSATION FLOW =====
  turn_count: number;
  ready_to_search: boolean;
  search_executed: boolean;
  conversation_phase: string;
  
  // ===== SEARCH RESULTS METADATA =====
  last_search_query: string | null;
  last_results_count: number | null;
  last_search_timestamp: string | null;
};


type Message = {
  role: "user" | "ai";
  content: string;
  results?: FacilityResult[];
  timestamp?: string;
};

type FacilityResult = {
  place_id: string;
  name: string;
  category: string;
  distance: number;
  english_confidence_score: number;
  Summaries: string[];
  address?: string;
  website?: string;
  relevance_rank?: number;
};

type DebugInfo = {
  lastRequest?: any;
  lastResponse?: any;
  apiError?: string;
  requestTimestamp?: string;
  responseTime?: number;
};

// --- HELPER FUNCTION FOR FORMATTING AI RESPONSES ---
const formatAIResponse = (text: string): string => {
  let formatted = text.replace(/\*\*([^*]+)\*\*/g, '$1\n');
  formatted = formatted.replace(/(\d+\.)/g, '\n$1');
  return formatted;
};


// --- CATEGORY TRANSLATION HELPER ---
const getCategoryEnglish = (koreanCategory: string): string => {
  const categoryMap: Record<string, string> = {
    // --- Core medical specialties ---
    '내과': 'Internal Medicine',
    '치과': 'Dentist',
    '산부인과': 'OB/GYN',
    '정형외과': 'Orthopedics',
    '피부과': 'Dermatology',
    '안과': 'Ophthalmology',
    '이비인후과': 'ENT',
    '외과': 'Surgery',
    '신경과': 'Neurology',
    '신경외과': 'Neurosurgery',
    '정신건강의학과': 'Psychiatry',
    '가정의학과': 'Family Medicine',
    '비뇨의학과': 'Urology',
    '비뇨기과': 'Urology',
    '소아청소년과': 'Pediatrics',
    '마취통증의학과': 'Anesthesiology & Pain Medicine',
    '재활의학과': 'Rehabilitation Medicine',
    '영상의학과': 'Radiology',
    '흉부외과': 'Thoracic Surgery',
    '대장,항문과': 'Colorectal Surgery',

    // --- Hospitals & facilities ---
    '병원,의원': 'Clinic / Hospital',
    '종합병원': 'General Hospital',
    '국립병원': 'National Hospital',
    '시립,도립병원': 'Public Hospital',
    '요양병원': 'Long-Term Care Hospital',
    '노인전문병원': 'Geriatric Hospital',
    '여성전문병원': 'Women’s Hospital',
    '보훈병원': 'Veterans Hospital',
    '병원부속시설': 'Hospital Facility',
    '응급실': 'Emergency Room',

    // --- Public health ---
    '보건소': 'Public Health Center',
    '보건지소': 'Community Health Subcenter',

    // --- Traditional medicine ---
    '한의원': 'Korean Medicine Clinic',
    '한방병원': 'Korean Medicine Hospital',

    // --- Health services (non-department) ---
    '건강검진': 'Health Checkup',
    '건강관리': 'Health Management',
    '조산원': 'Birthing Center',
    '모유수유': 'Breastfeeding Support',
    '언어치료': 'Speech Therapy',
    '치료,재활': 'Therapy & Rehabilitation',
    '아동,청소년상담': 'Child & Adolescent Counseling',

    // --- Non-medical but present in data ---
    '장례식장': 'Funeral Hall',
    '보험': 'Insurance',
    '종합대행업체': 'General Service Agency',
    '건물,빌딩': 'Building',
    'N/A': 'Unknown',

    // --- Clearly non-medical noise (keep explicit) ---
    '백숙,삼계탕': 'Restaurant (Chicken Soup)',
    '자동차정비,수리': 'Auto Repair',
    '미용실': 'Hair Salon',
    '머리염색': 'Hair Coloring',
    '미용': 'Beauty Services',
    '피부,체형관리': 'Skin & Body Care',
    '미용기기,재료': 'Beauty Equipment & Supplies',
    '세탁소': 'Laundry',
    '헬스장': 'Gym',
    '교습학원,교습소': 'Academy / Tutoring Institute',
  };

  for (const [korean, english] of Object.entries(categoryMap)) {
    if (koreanCategory.includes(korean)) {
      return english;
    }
  }
  
  return 'Medical Facility';
};

export default function ChatInterface() {
  // --- DEBUG MODE STATE ---
  const [debugMode, setDebugMode] = useState(false); // true for it to work
  const [debugExpanded, setDebugExpanded] = useState(false);
  const [debugInfo, setDebugInfo] = useState<DebugInfo>({});

  const [messages, setMessages] = useState<Message[]>([
    { 
      role: "ai", 
      content: "Hello! I'm SeoulMedBot, your AI medical concierge. Tell me what kind of doctor you need, and I'll find the best options for you.",
      timestamp: new Date().toISOString()
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [expandedFacilities, setExpandedFacilities] = useState<Set<string>>(new Set());
  const [disclaimerVisible, setDisclaimerVisible] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-dismiss disclaimer after 15 seconds
  useEffect(() => {
    const timer = setTimeout(() => {
      setDisclaimerVisible(false);
    }, 15000);

    return () => clearTimeout(timer);
  }, []);

  // Hide disclaimer on scroll
  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (container.scrollTop > 50) {
        setDisclaimerVisible(false);
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  // Handle mobile keyboard and viewport changes
  useEffect(() => {
    const handleResize = () => {
      // Scroll input into view when keyboard appears on mobile
      if (document.activeElement === inputRef.current) {
        setTimeout(() => {
          inputRef.current?.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
          });
        }, 100);
      }
    };

    // Listen for viewport changes (mobile keyboard)
    if (typeof window !== 'undefined' && 'visualViewport' in window) {
      window.visualViewport?.addEventListener('resize', handleResize);
      return () => {
        window.visualViewport?.removeEventListener('resize', handleResize);
      };
    }
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messages.length > 1) {
      // Small delay to ensure content is rendered
      setTimeout(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }, 100);
    }
  }, [messages]);

  const [currentState, setCurrentState] = useState<State>({
  // Specialty
  specialty: null,
  specialty_confidence: 0,
  
  // Location
  location: null,
  latitude: null,
  longitude: null,
  address_korean: null,
  district: null,
  dong: null,
  
  // Search parameters
  search_mode: null,
  max_distance_km: 5,
  willingness_to_travel: "Nearby",
  
  // Keywords
  keywords: [],
  hard_keywords: [],
  negative_keywords: [],  // ← ADD THIS
  negative_hard_keywords: [],  // ← ADD THIS
  
  // Hybrid search
  hybrid_alpha: null,
  query_intent: null,
  suggested_alpha: null,
  manual_search_mode: null,
  
  // Preferences
  language_pref: "English",
  
  // Conversation flow
  turn_count: 0,
  ready_to_search: false,
  search_executed: false,
  conversation_phase: "greeting",
  
  // Metadata
  last_search_query: null,
  last_results_count: null,
  last_search_timestamp: null,
});

  const toggleFacilityExpand = (placeId: string) => {
    setExpandedFacilities(prev => {
      const newSet = new Set(prev);
      if (newSet.has(placeId)) {
        newSet.delete(placeId);
      } else {
        newSet.add(placeId);
      }
      return newSet;
    });
  };

  const scrollToBottom = () => {
    setTimeout(() => {
      scrollRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, 100);
  };

  const handleSendMessage = async (text: string, stateOverride?: Partial<State>) => {
    if (!text.trim()) return;

    const userMsg: Message = { 
      role: "user", 
      content: text,
      timestamp: new Date().toISOString()
    };
    if (!stateOverride) setMessages((prev) => [...prev, userMsg]);
    
    setInput("");
    setLoading(true);

    // Scroll immediately when user sends message
    scrollToBottom();

    const stateToSend = {
      ...currentState,
      ...stateOverride,
    };

    const requestPayload = {
      message: text,
      current_state: stateToSend,
    };

    // Debug: Log request
    const requestTime = Date.now();
    if (debugMode) {
      setDebugInfo(prev => ({
        ...prev,
        lastRequest: requestPayload,
        requestTimestamp: new Date().toISOString(),
        apiError: undefined
      }));
    }

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });

      const data = await response.json();
      const responseTime = Date.now() - requestTime;

      // Debug: Log response
      if (debugMode) {
        setDebugInfo(prev => ({
          ...prev,
          lastResponse: data,
          responseTime: responseTime,
          apiError: undefined
        }));
      }

      if (data.state) {
        setCurrentState(data.state);
      }
      
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: data.response,
          results: data.results,
          timestamp: new Date().toISOString()
        },
      ]);

      // Scroll again after AI response
      scrollToBottom();
    } catch (error) {
      console.error("API Error:", error);
      
      // Debug: Log error
      if (debugMode) {
        setDebugInfo(prev => ({
          ...prev,
          apiError: error instanceof Error ? error.message : 'Unknown error',
          responseTime: Date.now() - requestTime
        }));
      }

      setMessages((prev) => [
        ...prev,
        { 
          role: "ai", 
          content: "I'm having trouble connecting right now. Please try again in a moment.",
          timestamp: new Date().toISOString()
        },
      ]);
      scrollToBottom();
    } finally {
      setLoading(false);
    }
  };

  const handleLocationClick = () => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser.");
      return;
    }
    
    setMessages((prev) => [...prev, { 
      role: "user", 
      content: "📍 Sharing my location...",
      timestamp: new Date().toISOString()
    }]);
    setLoading(true);
    scrollToBottom();

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        
        // Debug: Log geolocation
        if (debugMode) {
          setDebugInfo(prev => ({
            ...prev,
            lastRequest: {
              ...prev.lastRequest,
              geolocation: { latitude, longitude }
            }
          }));
        }

        handleSendMessage("User shared location coordinates.", {
          latitude: latitude,
          longitude: longitude,
          location: "Current Location",
        });
      },
      (error) => {
        alert("Unable to retrieve your location.");
        setLoading(false);
        
        // Debug: Log geolocation error
        if (debugMode) {
          setDebugInfo(prev => ({
            ...prev,
            apiError: `Geolocation error: ${error.message}`
          }));
        }
      }
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(input);
    }
  };

  const handleInputFocus = () => {
    // Ensure input stays visible on mobile when keyboard appears
    setTimeout(() => {
      inputRef.current?.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'nearest' 
      });
    }, 300); // Delay to let keyboard animation finish
  };

  // Calculate disclaimer height for dynamic spacing
  const disclaimerHeight = disclaimerVisible ? 56 : 0;

  // Get message statistics for debug panel
  const getMessageStats = () => {
    const userMessages = messages.filter(m => m.role === 'user').length;
    const aiMessages = messages.filter(m => m.role === 'ai').length;
    const messagesWithResults = messages.filter(m => m.results && m.results.length > 0).length;
    return { userMessages, aiMessages, messagesWithResults, total: messages.length };
  };

  return (
    <div className="relative h-full w-full bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 flex flex-col">
      
      {/* --- DEBUG TOGGLE BUTTON (Floating) --- */}
      <button
        onClick={() => setDebugMode(!debugMode)}
        className={`fixed top-20 right-4 z-50 p-3 rounded-full shadow-lg transition-all ${
          debugMode 
            ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white' 
            : 'bg-white text-slate-600 hover:bg-slate-50'
        } border-2 ${debugMode ? 'border-purple-300' : 'border-slate-200'}`}
        title={debugMode ? "Debug Mode: ON" : "Debug Mode: OFF"}
      >
        <Bug size={20} />
      </button>

      {/* --- DEBUG PANEL --- */}
      {debugMode && (
        <div className="fixed top-32 right-4 z-40 w-96 max-h-[70vh] bg-slate-900 text-slate-100 rounded-lg shadow-2xl border-2 border-purple-500 overflow-hidden flex flex-col">
          {/* Debug Header */}
          <div className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-600 to-pink-600 border-b border-purple-400">
            <div className="flex items-center gap-2">
              <Bug size={16} />
              <span className="font-bold text-sm">Debug Console</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setDebugExpanded(!debugExpanded)}
                className="p-1 hover:bg-white/20 rounded transition-colors"
              >
                {debugExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
              <button
                onClick={() => setDebugMode(false)}
                className="p-1 hover:bg-white/20 rounded transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {debugExpanded && (
              <div className="overflow-y-auto flex-1 p-4 text-xs space-y-4">
                {/* Message Statistics */}
                <div className="bg-slate-800 rounded p-3 border border-slate-700">
                  <h3 className="font-bold text-purple-400 mb-2">📊 Message Stats</h3>
                  <div className="space-y-1 text-slate-300">
                    <div className="flex justify-between">
                      <span>Total Messages:</span>
                      <span className="font-mono">{getMessageStats().total}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>User Messages:</span>
                      <span className="font-mono">{getMessageStats().userMessages}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>AI Messages:</span>
                      <span className="font-mono">{getMessageStats().aiMessages}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>With Results:</span>
                      <span className="font-mono">{getMessageStats().messagesWithResults}</span>
                    </div>
                  </div>
                </div>

                {/* Current State - Specialty */}
                <div className="bg-slate-800 rounded p-3 border border-slate-700">
                  <h3 className="font-bold text-blue-400 mb-2">🏥 Specialty</h3>
                  <div className="space-y-1 text-slate-300">
                    <div className="flex justify-between">
                      <span>Specialty:</span>
                      <span className="font-mono text-green-400 truncate max-w-[180px]">
                        {currentState.specialty || 'null'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Confidence:</span>
                      <span className="font-mono text-yellow-400">
                        {(currentState.specialty_confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Location Information */}
                <div className="bg-slate-800 rounded p-3 border border-slate-700">
                  <h3 className="font-bold text-green-400 mb-2">📍 Location</h3>
                  <div className="space-y-1 text-slate-300">
                    <div className="flex justify-between">
                      <span>Location Text:</span>
                      <span className="font-mono text-green-400 truncate max-w-[150px]">
                        {currentState.location || 'null'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>District (구):</span>
                      <span className="font-mono text-green-400">
                        {currentState.district || 'null'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Dong (동):</span>
                      <span className="font-mono text-green-400">
                        {currentState.dong || 'null'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>GPS:</span>
                      <span className="font-mono text-green-400 text-[10px]">
                        {currentState.latitude && currentState.longitude 
                          ? `${currentState.latitude.toFixed(4)}, ${currentState.longitude.toFixed(4)}`
                          : 'null'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Address (KR):</span>
                      <span className="font-mono text-green-400 text-[10px] truncate max-w-[150px]">
                        {currentState.address_korean ? 
                          (currentState.address_korean.length > 20 
                            ? currentState.address_korean.substring(0, 20) + '...'
                            : currentState.address_korean)
                          : 'null'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Search Parameters */}
                <div className="bg-slate-800 rounded p-3 border border-slate-700">
                  <h3 className="font-bold text-orange-400 mb-2">🔍 Search Parameters</h3>
                  <div className="space-y-1 text-slate-300">
                    <div className="flex justify-between">
                      <span>Mode:</span>
                      <span className={`font-mono ${
                        currentState.search_mode === 'zone' ? 'text-purple-400' : 
                        currentState.search_mode === 'distance' ? 'text-blue-400' : 
                        'text-slate-500'
                      }`}>
                        {currentState.search_mode || 'auto'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Max Distance:</span>
                      <span className="font-mono text-orange-400">
                        {currentState.max_distance_km}km
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Travel Willingness:</span>
                      <span className="font-mono text-cyan-400 text-[10px]">
                        {currentState.willingness_to_travel}
                      </span>
                    </div>
                  </div>
                </div>

              {/* Keywords */}
              <div className="bg-slate-800 rounded p-3 border border-slate-700">
                <h3 className="font-bold text-yellow-400 mb-2">🔤 Keywords</h3>
                <div className="space-y-1 text-slate-300">
                  <div>
                    <span className="text-slate-400 text-[10px]">Soft Keywords (prefer):</span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {currentState.keywords.length > 0 ? (
                        currentState.keywords.map((kw, idx) => (
                          <span key={idx} className="px-1.5 py-0.5 bg-yellow-900/30 text-yellow-300 rounded text-[10px]">
                            {kw}
                          </span>
                        ))
                      ) : (
                        <span className="text-slate-500 text-[10px]">none</span>
                      )}
                    </div>
                  </div>
                  
                  <div>
                    <span className="text-slate-400 text-[10px]">Hard Keywords (MUST have):</span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {currentState.hard_keywords.length > 0 ? (
                        currentState.hard_keywords.map((kw, idx) => (
                          <span key={idx} className="px-1.5 py-0.5 bg-red-900/30 text-red-300 rounded text-[10px] font-bold">
                            {kw}
                          </span>
                        ))
                      ) : (
                        <span className="text-slate-500 text-[10px]">none</span>
                      )}
                    </div>
                  </div>
                  
                  {/* ⭐ ADD NEGATIVE KEYWORDS SECTION */}
                  <div>
                    <span className="text-slate-400 text-[10px]">Negative Keywords (avoid):</span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {currentState.negative_keywords.length > 0 ? (
                        currentState.negative_keywords.map((kw, idx) => (
                          <span key={idx} className="px-1.5 py-0.5 bg-orange-900/30 text-orange-300 rounded text-[10px]">
                            🚫 {kw}
                          </span>
                        ))
                      ) : (
                        <span className="text-slate-500 text-[10px]">none</span>
                      )}
                    </div>
                  </div>
                  
                  <div>
                    <span className="text-slate-400 text-[10px]">Hard Negatives (MUST NOT have):</span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {currentState.negative_hard_keywords.length > 0 ? (
                        currentState.negative_hard_keywords.map((kw, idx) => (
                          <span key={idx} className="px-1.5 py-0.5 bg-red-900/50 text-red-200 rounded text-[10px] font-bold">
                            ⛔ {kw}
                          </span>
                        ))
                      ) : (
                        <span className="text-slate-500 text-[10px]">none</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

                {/* Hybrid Search Configuration */}
                <div className="bg-slate-800 rounded p-3 border border-slate-700">
                  <h3 className="font-bold text-pink-400 mb-2">🔀 Hybrid Search</h3>
                  <div className="space-y-1 text-slate-300">
                    <div className="flex justify-between">
                      <span>Query Intent:</span>
                      <span className={`font-mono ${
                        currentState.query_intent === 'FACTUAL' ? 'text-blue-400' : 
                        currentState.query_intent === 'MIXED' ? 'text-purple-400' : 
                        'text-slate-500'
                      }`}>
                        {currentState.query_intent || 'not classified'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Suggested α:</span>
                      <span className="font-mono text-yellow-400">
                        {currentState.suggested_alpha !== null 
                          ? currentState.suggested_alpha.toFixed(2) 
                          : 'null'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Active α:</span>
                      <span className="font-mono text-pink-400">
                        {currentState.hybrid_alpha !== null 
                          ? currentState.hybrid_alpha.toFixed(2) 
                          : 'auto'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Manual Override:</span>
                      <span className="font-mono text-cyan-400 text-[10px]">
                        {currentState.manual_search_mode || 'none'}
                      </span>
                    </div>
                    <div className="mt-2 p-2 bg-slate-900 rounded">
                      <div className="text-[10px] text-slate-400 mb-1">Alpha Scale:</div>
                      <div className="relative h-2 bg-slate-700 rounded">
                        <div 
                          className="absolute h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded"
                          style={{ 
                            width: `${((currentState.hybrid_alpha || 0.5) * 100)}%` 
                          }}
                        />
                      </div>
                      <div className="flex justify-between text-[9px] text-slate-500 mt-1">
                        <span>0.0 (keyword)</span>
                        <span>0.5</span>
                        <span>1.0 (semantic)</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Conversation State */}
                <div className="bg-slate-800 rounded p-3 border border-slate-700">
                  <h3 className="font-bold text-cyan-400 mb-2">💬 Conversation</h3>
                  <div className="space-y-1 text-slate-300">
                    <div className="flex justify-between">
                      <span>Phase:</span>
                      <span className="font-mono text-purple-400">
                        {currentState.conversation_phase}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Turn Count:</span>
                      <span className="font-mono">{currentState.turn_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Ready to Search:</span>
                      <span className={`font-mono ${currentState.ready_to_search ? 'text-green-400' : 'text-red-400'}`}>
                        {currentState.ready_to_search ? 'true' : 'false'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Search Executed:</span>
                      <span className={`font-mono ${currentState.search_executed ? 'text-green-400' : 'text-red-400'}`}>
                        {currentState.search_executed ? 'true' : 'false'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Language:</span>
                      <span className="font-mono text-blue-400">
                        {currentState.language_pref}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Search Metadata */}
                {(currentState.last_search_query || currentState.last_results_count !== null) && (
                  <div className="bg-slate-800 rounded p-3 border border-slate-700">
                    <h3 className="font-bold text-indigo-400 mb-2">📊 Last Search</h3>
                    <div className="space-y-1 text-slate-300">
                      {currentState.last_search_query && (
                        <div>
                          <span className="text-slate-400 text-[10px]">Query:</span>
                          <p className="font-mono text-indigo-300 text-[10px] break-words">
                            {currentState.last_search_query}
                          </p>
                        </div>
                      )}
                      {currentState.last_results_count !== null && (
                        <div className="flex justify-between">
                          <span>Results:</span>
                          <span className="font-mono text-green-400">
                            {currentState.last_results_count}
                          </span>
                        </div>
                      )}
                      {currentState.last_search_timestamp && (
                        <div className="flex justify-between">
                          <span>Timestamp:</span>
                          <span className="font-mono text-slate-400 text-[10px]">
                            {new Date(currentState.last_search_timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* API Response Info */}
                {debugInfo.lastResponse && (
                  <div className="bg-slate-800 rounded p-3 border border-slate-700">
                    <h3 className="font-bold text-green-400 mb-2">✅ Last API Response</h3>
                    <div className="space-y-1 text-slate-300">
                      {debugInfo.responseTime && (
                        <div className="flex justify-between">
                          <span>Response Time:</span>
                          <span className="font-mono text-yellow-400">
                            {debugInfo.responseTime}ms
                          </span>
                        </div>
                      )}
                      {debugInfo.lastResponse.results && (
                        <div className="flex justify-between">
                          <span>Results Count:</span>
                          <span className="font-mono text-green-400">
                            {debugInfo.lastResponse.results.length}
                          </span>
                        </div>
                      )}
                      <div className="mt-2">
                        <span className="text-slate-400">Response Preview:</span>
                        <pre className="mt-1 p-2 bg-slate-900 rounded text-[10px] overflow-x-auto max-h-32">
                          {JSON.stringify(debugInfo.lastResponse, null, 2).slice(0, 500)}...
                        </pre>
                      </div>
                    </div>
                  </div>
                )}

                {/* Last Request */}
                {debugInfo.lastRequest && (
                  <div className="bg-slate-800 rounded p-3 border border-slate-700">
                    <h3 className="font-bold text-orange-400 mb-2">📤 Last Request</h3>
                    <div className="space-y-1 text-slate-300">
                      {debugInfo.requestTimestamp && (
                        <div className="text-slate-400 text-[10px]">
                          {new Date(debugInfo.requestTimestamp).toLocaleTimeString()}
                        </div>
                      )}
                      <pre className="mt-1 p-2 bg-slate-900 rounded text-[10px] overflow-x-auto max-h-32">
                        {JSON.stringify(debugInfo.lastRequest, null, 2).slice(0, 500)}...
                      </pre>
                    </div>
                  </div>
                )}

                {/* API Error */}
                {debugInfo.apiError && (
                  <div className="bg-red-900/30 rounded p-3 border border-red-700">
                    <h3 className="font-bold text-red-400 mb-2">❌ API Error</h3>
                    <div className="text-red-300 text-xs break-words">
                      {debugInfo.apiError}
                    </div>
                  </div>
                )}

                {/* Environment Info */}
                <div className="bg-slate-800 rounded p-3 border border-slate-700">
                  <h3 className="font-bold text-cyan-400 mb-2">⚙️ Environment</h3>
                  <div className="space-y-1 text-slate-300">
                    <div className="flex justify-between">
                      <span>API URL:</span>
                      <span className="font-mono text-cyan-400 text-[10px] truncate max-w-[180px]">
                        {process.env.NEXT_PUBLIC_API_URL || 'Not set'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Loading:</span>
                      <span className={`font-mono ${loading ? 'text-yellow-400' : 'text-green-400'}`}>
                        {loading ? 'true' : 'false'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Debug Mode:</span>
                      <span className="font-mono text-purple-400">
                        enabled
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
        </div>
      )}

      {/* --- ELEGANT DISCLAIMER BANNER --- */}
      <div 
        className={`absolute top-0 left-0 right-0 z-20 transition-all duration-500 ease-in-out ${
          disclaimerVisible 
            ? 'translate-y-0 opacity-100' 
            : '-translate-y-full opacity-0'
        }`}
        style={{ pointerEvents: disclaimerVisible ? 'auto' : 'none' }}
      >
        <div className="bg-gradient-to-r from-amber-50 to-yellow-50 border-b border-amber-200 shadow-sm backdrop-blur-sm">
          <div className="max-w-5xl mx-auto px-4 py-2.5">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <span className="text-lg flex-shrink-0">⚠️</span>
                <p className="text-xs sm:text-sm text-amber-900 leading-tight">
                  <strong>Info only</strong> - Not medical advice. 
                  <Link href="/disclaimer" className="underline hover:text-amber-700 font-semibold ml-1">
                    Full disclaimer
                  </Link>
                  {' • '}
                  <span className="font-semibold whitespace-nowrap">Emergency: 119</span>
                </p>
              </div>
              <button
                onClick={() => setDisclaimerVisible(false)}
                className="flex-shrink-0 text-amber-700 hover:text-amber-900 p-1 rounded-full hover:bg-amber-100 transition-colors"
                aria-label="Dismiss"
              >
                <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* --- CHAT MESSAGES AREA --- */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto min-h-0 transition-all duration-500 ease-in-out overscroll-behavior-contain"
        style={{ 
          paddingTop: `${disclaimerHeight}px`
        }}
      >
        <div className="max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
          <div className="space-y-4 sm:space-y-6">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fadeIn`}
              >
                <div className={`flex gap-2 sm:gap-3 max-w-[95%] sm:max-w-[85%] ${msg.role === "user" ? 'flex-row-reverse' : 'flex-row'}`}>
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center shadow-md ${
                    msg.role === "user" 
                      ? "bg-gradient-to-br from-slate-600 to-slate-800" 
                      : "bg-gradient-to-br from-blue-500 to-purple-600"
                  }`}>
                    {msg.role === "user" ? (
                      <span className="text-white text-xs sm:text-sm font-bold">You</span>
                    ) : (
                      <Sparkles className="text-white" size={16} />
                    )}
                  </div>

                  <div className="flex flex-col gap-2 sm:gap-3 flex-1 min-w-0">
                    {/* Message Bubble */}
                    <div
                      className={`px-4 py-3 sm:px-5 sm:py-3.5 rounded-2xl shadow-sm ${
                        msg.role === "user"
                          ? "bg-gradient-to-r from-slate-700 to-slate-900 text-white"
                          : "bg-white border border-slate-200 text-slate-800"
                      }`}
                    >
                      <p className="text-sm sm:text-base leading-relaxed whitespace-pre-wrap break-words">
                        {msg.role === "ai" ? formatAIResponse(msg.content) : msg.content}
                      </p>
                      {/* Debug: Show timestamp */}
                      {debugMode && msg.timestamp && (
                        <div className="mt-2 pt-2 border-t border-slate-300 text-xs text-slate-500 font-mono">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </div>
                      )}
                    </div>

                    {/* Results Cards */}
                    {msg.results && msg.results.length > 0 && (
                      <div className="space-y-3">
                        {msg.results.map((facility, facilityIdx) => {
                          const isExpanded = expandedFacilities.has(facility.place_id);
                          const summary = facility.Summaries?.[0] || "";
                          const needsExpansion = summary.length > 150;
                          const categoryEnglish = getCategoryEnglish(facility.category);
                          
                          // Create search query for Naver Map (name + address beginning)
                          const addressStart = facility.address 
                            ? facility.address.split(',')[0].trim().split(' ').slice(0, 3).join(' ')
                            : '';
                          const mapSearchQuery = addressStart 
                            ? `${facility.name} ${addressStart}`
                            : facility.name;

                          return (
                            <div
                              key={facility.place_id}
                              className="bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-all overflow-hidden"
                            >
                              <div className="p-4 sm:p-5">
                                <div className="flex justify-between items-start gap-3 mb-2">
                                  <div className="flex-1 min-w-0">
                                    <h3 className="font-bold text-slate-900 text-base sm:text-lg mb-1.5 break-words">
                                      {facility.name}
                                    </h3>
                                    
                                    <div className="flex flex-wrap gap-2 items-center">
                                      <span className="inline-block px-2.5 py-1 bg-blue-50 text-blue-700 text-xs font-semibold rounded-md">
                                        {facility.category}
                                      </span>
                                      <span className="inline-block px-2.5 py-1 bg-slate-50 text-slate-600 text-xs font-medium rounded-md">
                                        {categoryEnglish}
                                      </span>
                                      {/* Debug: Show relevance rank */}
                                      {debugMode && facility.relevance_rank !== undefined && (
                                        <span className="inline-block px-2.5 py-1 bg-purple-50 text-purple-700 text-xs font-mono rounded-md">
                                          Rank: {facility.relevance_rank}
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  {facility.english_confidence_score >= 4 && (
                                    <span className="flex items-center gap-1.5 bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-full text-xs font-bold whitespace-nowrap flex-shrink-0">
                                      <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span>
                                      English OK
                                    </span>
                                  )}
                                </div>

                                {/* Address Display */}
                                {facility.address && (
                                  <div className="my-2.5 flex items-start gap-2">
                                    <MapPin size={14} className="text-slate-400 flex-shrink-0 mt-0.5" />
                                    <p className="text-xs sm:text-sm text-slate-600 leading-relaxed break-words">
                                      {facility.address}
                                    </p>
                                  </div>
                                )}

                                {/* Website Display */}
                                {facility.website && (
                                  <div className="my-2">
                                    <a 
                                      href={facility.website}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="inline-flex items-center gap-1.5 text-xs sm:text-sm text-blue-600 hover:text-blue-700 font-medium hover:underline transition-colors"
                                    >
                                      <Globe size={14} className="flex-shrink-0" />
                                      <span>Visit Website</span>
                                    </a>
                                  </div>
                                )}

                                {summary && (
                                  <div className="my-3">
                                    <p className={`text-sm sm:text-base text-slate-600 leading-relaxed break-words ${
                                      !isExpanded && needsExpansion ? 'line-clamp-2' : ''
                                    }`}>
                                      {summary}
                                    </p>
                                    {needsExpansion && (
                                      <button
                                        onClick={() => toggleFacilityExpand(facility.place_id)}
                                        className="mt-2 flex items-center gap-1 text-xs sm:text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors"
                                      >
                                        {isExpanded ? (
                                          <>
                                            Show less <svg className="w-3 h-3 sm:w-4 sm:h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" /></svg>
                                          </>
                                        ) : (
                                          <>
                                            Read more <svg className="w-3 h-3 sm:w-4 sm:h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                                          </>
                                        )}
                                      </button>
                                    )}
                                  </div>
                                )}

                                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-3 border-t border-slate-100">
                                  <div className="flex items-center gap-2 text-slate-500">
                                    <MapPin size={16} className="text-blue-500 flex-shrink-0" />
                                    <span className="text-sm sm:text-base font-medium">
                                      {facility.distance != null ? `${facility.distance.toFixed(1)} km away` : 'Distance N/A'}
                                    </span>
                                  </div>
                                  <a 
                                    href={`https://map.naver.com/v5/search/${encodeURIComponent(mapSearchQuery)}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="px-4 py-2 sm:px-5 sm:py-2.5 bg-gradient-to-r from-blue-500 to-purple-600 text-white text-sm sm:text-base font-semibold rounded-lg hover:shadow-lg hover:shadow-blue-500/50 transition-all text-center"
                                  >
                                    View on Map
                                  </a>
                                </div>

                                {/* Debug: Show place_id */}
                                {debugMode && (
                                  <div className="mt-2 pt-2 border-t border-slate-200">
                                    <p className="text-xs text-slate-500 font-mono">
                                      ID: {facility.place_id}
                                    </p>
                                  </div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="flex justify-start animate-fadeIn">
                <div className="flex gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-md flex-shrink-0">
                    <Sparkles className="text-white" size={16} />
                  </div>
                  <div className="bg-white border border-slate-200 rounded-2xl px-5 py-3.5 shadow-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={scrollRef} />
          </div>
        </div>
      </div>

      {/* --- INPUT AREA AT BOTTOM --- */}
      <div className="flex-shrink-0 border-t border-slate-200/50 bg-white/95 backdrop-blur-md shadow-lg">
        {/* Input Controls */}
        <div className="max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-3 sm:py-4">
          <div className="flex items-center gap-2 sm:gap-3">
            <button
              onClick={handleLocationClick}
              className="flex-shrink-0 p-3 sm:p-3.5 rounded-xl bg-gradient-to-br from-slate-50 to-slate-100 hover:from-slate-100 hover:to-slate-200 border border-slate-200 hover:border-blue-300 transition-all text-slate-600 hover:text-blue-600 shadow-sm"
              title="Share Location"
            >
              <MapPin size={20} className="sm:w-5 sm:h-5" />
            </button>
            
            <div className="flex-1 relative min-w-0">
              <input
                ref={inputRef}
                type="text"
                className="w-full px-4 py-3 sm:px-5 sm:py-4 rounded-xl bg-white border-2 border-slate-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 transition-all text-slate-800 placeholder-slate-400 outline-none text-sm sm:text-base shadow-sm"
                placeholder="Describe what you need..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={handleInputFocus}
              />
            </div>

            <button
              onClick={() => handleSendMessage(input)}
              disabled={!input.trim() || loading}
              className={`flex-shrink-0 p-3 sm:p-3.5 rounded-xl transition-all shadow-md ${
                input.trim() && !loading 
                  ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:shadow-lg hover:scale-105 active:scale-95" 
                  : "bg-slate-200 text-slate-400 cursor-not-allowed"
              }`}
            >
              <Send size={20} className="sm:w-5 sm:h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  ); 
}

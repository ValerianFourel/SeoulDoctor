"use client";

import { useState, useRef, useEffect } from "react";
import { Send, MapPin, Sparkles, ChevronDown, ChevronUp, Bug } from "lucide-react";
import AdSlot from "./AdSlot";

// --- TYPES ---
type State = {
  specialty: string | null;
  specialty_confidence: number;
  location: string | null;
  latitude: number | null;
  longitude: number | null;
  address_korean: string | null;
  district: string | null;
  dong: string | null;
  language_pref: string;
  max_distance_km: number;
  willingness_to_travel: string;
  keywords: string[];
  ready_to_search: boolean;
  search_executed: boolean;
  search_mode: string | null;
  conversation_phase: string;
  turn_count: number;
};

type Message = {
  role: "user" | "ai";
  content: string;
  results?: FacilityResult[];
};

type FacilityResult = {
  place_id: string;
  name: string;
  category: string;
  distance: number;
  english_confidence_score: number;
  Summaries: string[];
};

const TRAVEL_OPTIONS = ["Neighborhood", "Nearby", "City-wide", "Don't Care"];

// ============ TEMPORARY DEBUG UTILITIES - REMOVE BEFORE PRODUCTION ============
const DEBUG_MODE = false; // Set to false to disable all debug features

const logStateToConsole = (state: State, label: string = "LLM Response State") => {
  if (!DEBUG_MODE) return;
  
  console.log(`\n${'='.repeat(80)}`);
  console.log(`🤖 ${label.toUpperCase()}`);
  console.log(`${'='.repeat(80)}`);
  
  // Group 1: Core Search Info
  console.log(`\n📋 SEARCH CRITERIA:`);
  console.log(`   Specialty: ${state.specialty || 'Not set'} (confidence: ${state.specialty_confidence || 0})`);
  console.log(`   Location: ${state.location || 'Not set'}`);
  console.log(`   Search Mode: ${state.search_mode || 'auto'}`);
  
  // Group 2: Location Details
  console.log(`\n📍 LOCATION DETAILS:`);
  console.log(`   GPS: ${state.latitude && state.longitude ? `${state.latitude.toFixed(4)}, ${state.longitude.toFixed(4)}` : 'Not available'}`);
  console.log(`   Address (KR): ${state.address_korean || 'Not set'}`);
  console.log(`   District: ${state.district || 'Not set'}`);
  console.log(`   Dong: ${state.dong || 'Not set'}`);
  console.log(`   Max Distance: ${state.max_distance_km}km`);
  
  // Group 3: Conversation State
  console.log(`\n💬 CONVERSATION STATE:`);
  console.log(`   Phase: ${state.conversation_phase}`);
  console.log(`   Turn: ${state.turn_count}`);
  console.log(`   Language: ${state.language_pref}`);
  console.log(`   Ready to Search: ${state.ready_to_search ? '✅' : '❌'}`);
  console.log(`   Search Executed: ${state.search_executed ? '✅' : '❌'}`);
  
  // Group 4: Full Object (collapsed)
  console.log(`\n🔍 FULL STATE OBJECT:`);
  console.log(state);
  
  console.log(`\n${'='.repeat(80)}\n`);
};

const StateDebugPanel = ({ state }: { state: State }) => {
  if (!DEBUG_MODE) return null;
  
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className="fixed bottom-20 right-4 z-50 max-w-md">
      <div className="bg-slate-900 text-white rounded-lg shadow-2xl border border-slate-700">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full px-4 py-2 flex items-center justify-between hover:bg-slate-800 transition-colors rounded-t-lg"
        >
          <div className="flex items-center gap-2">
            <Bug size={16} className="text-yellow-400" />
            <span className="text-xs font-bold">DEBUG STATE</span>
          </div>
          <ChevronDown 
            size={16} 
            className={`transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
          />
        </button>
        
        {isExpanded && (
          <div className="p-4 text-xs space-y-3 max-h-96 overflow-y-auto">
            {/* Core Search */}
            <div>
              <div className="text-yellow-400 font-bold mb-1">🔍 SEARCH</div>
              <div className="space-y-1 text-slate-300">
                <div>Specialty: <span className="text-white">{state.specialty || '—'}</span></div>
                <div>Confidence: <span className="text-white">{state.specialty_confidence || 0}</span></div>
                <div>Mode: <span className="text-white">{state.search_mode || 'auto'}</span></div>
              </div>
            </div>
            
            {/* Location */}
            <div>
              <div className="text-blue-400 font-bold mb-1">📍 LOCATION</div>
              <div className="space-y-1 text-slate-300">
                <div>Input: <span className="text-white">{state.location || '—'}</span></div>
                <div>GPS: <span className="text-white">
                  {state.latitude && state.longitude 
                    ? `${state.latitude.toFixed(4)}, ${state.longitude.toFixed(4)}`
                    : '—'}
                </span></div>
                <div>District: <span className="text-white">{state.district || '—'}</span></div>
                <div>Dong: <span className="text-white">{state.dong || '—'}</span></div>
                <div>Max Dist: <span className="text-white">{state.max_distance_km}km</span></div>
              </div>
            </div>
            
            {/* Conversation */}
            <div>
              <div className="text-green-400 font-bold mb-1">💬 CONVERSATION</div>
              <div className="space-y-1 text-slate-300">
                <div>Phase: <span className="text-white">{state.conversation_phase}</span></div>
                <div>Turn: <span className="text-white">{state.turn_count}</span></div>
                <div>Language: <span className="text-white">{state.language_pref}</span></div>
                <div>Ready: <span className={state.ready_to_search ? 'text-green-400' : 'text-red-400'}>
                  {state.ready_to_search ? '✅' : '❌'}
                </span></div>
                <div>Executed: <span className={state.search_executed ? 'text-green-400' : 'text-red-400'}>
                  {state.search_executed ? '✅' : '❌'}
                </span></div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
// ============================================================================

// --- HELPER FUNCTION FOR FORMATTING AI RESPONSES ---
const formatAIResponse = (text: string): string => {
  let formatted = text.replace(/\*\*([^*]+)\*\*/g, '$1\n');
  formatted = formatted.replace(/(\d+\.)/g, '\n$1');
  return formatted;
};

// --- CATEGORY TRANSLATION HELPER ---
const getCategoryEnglish = (koreanCategory: string): string => {
  const categoryMap: Record<string, string> = {
    '치과': 'Dentist',
    '피부과': 'Dermatology',
    '내과': 'Internal Medicine',
    '소아과': 'Pediatrics',
    '정형외과': 'Orthopedics',
    '안과': 'Ophthalmology',
    '이비인후과': 'ENT',
    '산부인과': 'OB/GYN',
    '성형외과': 'Plastic Surgery',
    '신경과': 'Neurology',
    '정신건강의학과': 'Psychiatry',
    '가정의학과': 'Family Medicine',
    '외과': 'Surgery',
    '비뇨기과': 'Urology',
  };

  for (const [korean, english] of Object.entries(categoryMap)) {
    if (koreanCategory.includes(korean)) {
      return english;
    }
  }
  
  return 'Medical Facility';
};

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    { 
      role: "ai", 
      content: "Hello! I'm SeoulMedBot, your AI medical concierge. Tell me what kind of doctor you need, and I'll find the best options for you." 
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [expandedFacilities, setExpandedFacilities] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);

  const [travelRadius, setTravelRadius] = useState("Nearby"); 
  const [currentState, setCurrentState] = useState<State>({
    specialty: null,
    specialty_confidence: 0,
    location: null,
    latitude: null,
    longitude: null,
    address_korean: null,
    district: null,
    dong: null,
    language_pref: "English",
    max_distance_km: 5,
    willingness_to_travel: "Nearby",
    keywords: [],
    ready_to_search: false,
    search_executed: false,
    search_mode: null,
    conversation_phase: "greeting",
    turn_count: 0,
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

  const handleSendMessage = async (text: string, stateOverride?: Partial<State>) => {
    if (!text.trim()) return;

    const userMsg: Message = { role: "user", content: text };
    if (!stateOverride) setMessages((prev) => [...prev, userMsg]);
    
    setInput("");
    setLoading(true);

    const stateToSend = {
      ...currentState,
      ...stateOverride,
      willingness_to_travel: travelRadius,
    };

    // ============ TEMPORARY LOGGING - REMOVE BEFORE PRODUCTION ============
    if (DEBUG_MODE) {
      console.log(`\n📤 SENDING STATE TO BACKEND:`);
      console.log(stateToSend);
    }
    // ======================================================================

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          current_state: stateToSend,
        }),
      });

      const data = await response.json();

      if (data.state) {
        setCurrentState(data.state);
        
        // ============ TEMPORARY LOGGING - REMOVE BEFORE PRODUCTION ============
        logStateToConsole(data.state, "LLM Response State");
        // ======================================================================
      }
      
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: data.response,
          results: data.results,
        },
      ]);
    } catch (error) {
      console.error("API Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "I'm having trouble connecting right now. Please try again in a moment." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleLocationClick = () => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser.");
      return;
    }
    
    setMessages((prev) => [...prev, { role: "user", content: "📍 Sharing my location..." }]);
    setLoading(true);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        handleSendMessage("User shared location coordinates.", {
          latitude: latitude,
          longitude: longitude,
          location: "Current Location",
        });
      },
      (error) => {
        alert("Unable to retrieve your location.");
        setLoading(false);
      }
    );
  };

  return (
    <div className="flex flex-col h-full w-full bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100">
      
      {/* ============ TEMPORARY DEBUG PANEL - REMOVE BEFORE PRODUCTION ============ */}
      <StateDebugPanel state={currentState} />
      {/* ========================================================================== */}
      
      {/* --- HEADER --- */}
      <div className="sticky top-0 z-10 backdrop-blur-xl bg-white/80 border-b border-slate-200/50">
        <div className="w-full px-4 sm:px-6 py-3 sm:py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl blur-sm opacity-75"></div>
                <div className="relative bg-gradient-to-br from-blue-500 to-purple-600 p-1.5 sm:p-2 rounded-xl">
                  <Sparkles className="text-white" size={16} />
                </div>
              </div>
              <div>
                <h1 className="font-bold text-base sm:text-xl bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
                  SeoulMedBot
                </h1>
                <p className="text-[10px] sm:text-xs text-slate-500">AI Medical Concierge</p>
              </div>
            </div>
            
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 transition-all text-sm font-medium text-slate-700"
            >
              <svg 
                className="w-5 h-5" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M4 6h16M4 12h16M4 18h16" 
                />
              </svg>
              <ChevronDown className={`hidden sm:block transition-transform ${showSettings ? 'rotate-180' : ''}`} size={14} />
            </button>
          </div>
          
          {/* Settings Panel */}
          {showSettings && (
            <div className="mt-3 sm:mt-4 p-3 sm:p-4 rounded-xl bg-slate-50 border border-slate-200 animate-slideDown">
              <div className="flex flex-col gap-2 sm:gap-3">
                <div>
                  <label className="text-[10px] sm:text-xs font-semibold text-slate-600 mb-1.5 sm:mb-2 block">Search Radius</label>
                  <div className="grid grid-cols-4 gap-1.5 sm:gap-2">
                    {TRAVEL_OPTIONS.map((opt) => (
                      <button
                        key={opt}
                        onClick={() => setTravelRadius(opt)}
                        className={`py-2 sm:py-2.5 px-2 sm:px-3 rounded-lg text-[10px] sm:text-xs font-semibold transition-all ${
                          travelRadius === opt
                            ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg shadow-blue-500/50"
                            : "bg-white text-slate-600 hover:bg-slate-100 border border-slate-200"
                        }`}
                      >
                        {opt === "Neighborhood" ? "< 2km" : opt === "Nearby" ? "< 5km" : opt === "City-wide" ? "< 10km" : "Any"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* --- CHAT AREA --- */}
      <div className="flex-1 overflow-y-auto">
        <div className="w-full px-3 sm:px-6 py-4 sm:py-8">
          <div className="space-y-4 sm:space-y-6">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fadeIn`}
              >
                <div className={`flex gap-2 sm:gap-3 max-w-[98%] sm:max-w-[85%] ${msg.role === "user" ? 'flex-row-reverse' : 'flex-row'}`}>
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full flex items-center justify-center ${
                    msg.role === "user" 
                      ? "bg-gradient-to-br from-slate-600 to-slate-800" 
                      : "bg-gradient-to-br from-blue-500 to-purple-600"
                  }`}>
                    {msg.role === "user" ? (
                      <span className="text-white text-[10px] sm:text-xs font-bold">You</span>
                    ) : (
                      <Sparkles className="text-white" size={12} />
                    )}
                  </div>

                  <div className="flex flex-col gap-2 sm:gap-3 flex-1">
                    {/* Message Bubble */}
                    <div
                      className={`px-3 py-2 sm:px-5 sm:py-3 rounded-2xl ${
                        msg.role === "user"
                          ? "bg-gradient-to-r from-slate-700 to-slate-900 text-white"
                          : "bg-white border border-slate-200 text-slate-800 shadow-sm"
                      }`}
                    >
                      <p className="text-xs sm:text-sm leading-relaxed whitespace-pre-wrap">
                        {msg.role === "ai" ? formatAIResponse(msg.content) : msg.content}
                      </p>
                    </div>

                    {/* Results Cards */}
                    {msg.results && msg.results.length > 0 && (
                      <div className="space-y-2 sm:space-y-3">
                        {msg.results.map((facility, facilityIdx) => {
                          const isExpanded = expandedFacilities.has(facility.place_id);
                          const summary = facility.Summaries?.[0] || "";
                          const needsExpansion = summary.length > 150;
                          const categoryEnglish = getCategoryEnglish(facility.category);

                          return (
                            <div
                              key={facility.place_id}
                              className="bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-all overflow-hidden group animate-fadeIn"
                              style={{ animationDelay: `${facilityIdx * 100}ms` }}
                            >
                              <div className="p-3 sm:p-4">
                                <div className="flex justify-between items-start mb-1.5 sm:mb-2">
                                  <div className="flex-1">
                                    {/* Korean Name */}
                                    <h3 className="font-bold text-slate-900 text-sm sm:text-base mb-1">
                                      {facility.name}
                                    </h3>
                                    
                                    {/* Category in Korean and English */}
                                    <div className="flex flex-wrap gap-1.5 sm:gap-2 items-center">
                                      <span className="inline-block px-1.5 py-0.5 sm:px-2 sm:py-1 bg-blue-50 text-blue-600 text-[10px] sm:text-xs font-semibold rounded-md">
                                        {facility.category}
                                      </span>
                                      <span className="inline-block px-1.5 py-0.5 sm:px-2 sm:py-1 bg-slate-50 text-slate-600 text-[10px] sm:text-xs font-medium rounded-md">
                                        {categoryEnglish}
                                      </span>
                                    </div>
                                  </div>
                                  {facility.english_confidence_score >= 4 && (
                                    <span className="flex items-center gap-1 bg-emerald-50 text-emerald-700 px-2 py-1 sm:px-3 sm:py-1.5 rounded-full text-[10px] sm:text-xs font-bold">
                                      <span className="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-emerald-500 rounded-full"></span>
                                      English OK
                                    </span>
                                  )}
                                </div>

                                {summary && (
                                  <div className="my-2 sm:my-3">
                                    <p className={`text-xs sm:text-sm text-slate-600 leading-relaxed ${
                                      !isExpanded && needsExpansion ? 'line-clamp-2' : ''
                                    }`}>
                                      {summary}
                                    </p>
                                    {needsExpansion && (
                                      <button
                                        onClick={() => toggleFacilityExpand(facility.place_id)}
                                        className="mt-1.5 sm:mt-2 flex items-center gap-1 text-[10px] sm:text-xs font-semibold text-blue-600 hover:text-blue-700 transition-colors"
                                      >
                                        {isExpanded ? (
                                          <>
                                            Show less <ChevronUp size={12} />
                                          </>
                                        ) : (
                                          <>
                                            Read more <ChevronDown size={12} />
                                          </>
                                        )}
                                      </button>
                                    )}
                                  </div>
                                )}

                                <div className="flex items-center justify-between pt-2 sm:pt-3 border-t border-slate-100">
                                  <div className="flex items-center gap-1.5 sm:gap-2 text-slate-500">
                                    <MapPin size={12} className="text-blue-500" />
                                    <span className="text-xs sm:text-sm font-medium">
                                      {facility.distance != null ? `${facility.distance.toFixed(1)} km away` : 'Distance N/A'}
                                    </span>
                                  </div>
                                  <a 
                                    href={`https://map.naver.com/v5/search/${encodeURIComponent(facility.name)}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="px-3 py-1.5 sm:px-4 sm:py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white text-xs sm:text-sm font-semibold rounded-lg hover:shadow-lg hover:shadow-blue-500/50 transition-all"
                                  >
                                    View Map
                                  </a>
                                </div>
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
                <div className="flex gap-2 sm:gap-3">
                  <div className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <Sparkles className="text-white" size={12} />
                  </div>
                  <div className="bg-white border border-slate-200 rounded-2xl px-3 py-2 sm:px-5 sm:py-3 shadow-sm">
                    <div className="flex items-center gap-1.5 sm:gap-2">
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-blue-500 rounded-full animate-bounce"></div>
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={scrollRef} />
          </div>
        </div>
      </div>

      {/* --- INPUT AREA & MOBILE AD --- */}
      <div className="sticky bottom-0 backdrop-blur-xl bg-white/80 border-t border-slate-200/50 z-20">
        <div className="w-full px-3 sm:px-6 py-3 sm:py-4">
          
          {/* ▼▼▼ MOBILE ONLY AD SLOT - SAME WIDTH AS INPUT ▼▼▼ */}
          <div className="block xl:hidden w-full mb-3">
             <div className="overflow-hidden rounded-lg bg-slate-50/80 border border-slate-200/50 flex items-center justify-center" style={{ height: '60px' }}>
                <div className="scale-90 origin-center">
                  <AdSlot />
                </div>
             </div>
          </div>

          {/* Input Controls */}
          <div className="flex items-end gap-2 sm:gap-3">
            <button
              onClick={handleLocationClick}
              className="p-2.5 sm:p-3 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 hover:border-blue-300 transition-all text-slate-600 hover:text-blue-600 group"
              title="Share Location"
            >
              <MapPin size={18} className="sm:w-5 sm:h-5 group-hover:scale-110 transition-transform" />
            </button>
            
            <div className="flex-1 relative">
              <input
                type="text"
                className="w-full px-3 py-2.5 sm:px-5 sm:py-3.5 pr-10 sm:pr-12 rounded-xl bg-white border-2 border-slate-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 transition-all text-slate-800 placeholder-slate-400 outline-none text-xs sm:text-base"
                placeholder="Describe what you need..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage(input)}
              />
            </div>

            <button
              onClick={() => handleSendMessage(input)}
              disabled={!input.trim() || loading}
              className={`p-2.5 sm:p-3.5 rounded-xl transition-all ${
                input.trim() && !loading 
                  ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg shadow-blue-500/50 hover:shadow-xl hover:shadow-blue-500/50 hover:scale-105" 
                  : "bg-slate-200 text-slate-400 cursor-not-allowed"
              }`}
            >
              <Send size={18} className="sm:w-5 sm:h-5" />
            </button>
          </div>
          
          <p className="text-[10px] sm:text-xs text-slate-400 text-center mt-2 sm:mt-3">
            SeoulMedBot can make mistakes. Verify important medical information.
          </p>
          
        </div>
      </div>
    </div>
  );
}
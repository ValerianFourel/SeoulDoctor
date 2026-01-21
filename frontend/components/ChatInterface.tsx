"use client";

import { useState, useRef } from "react";
import { Send, MapPin, Sparkles } from "lucide-react";
import AdSlot from "./AdSlot";
import Footer from "./Footer";
import DisclaimerBanner from "./DisclaimerBanner";

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
  const [expandedFacilities, setExpandedFacilities] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);

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
    };

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
      
      {/* --- HEADER --- */}
      <div className="sticky top-0 z-10 backdrop-blur-xl bg-white/80 border-b border-slate-200/50">
        <div className="w-full px-4 sm:px-6 py-3 sm:py-4">
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
        </div>
      </div>

      {/* --- CHAT AREA (SCROLLABLE WITH FOOTER INSIDE) --- */}
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
                                    <h3 className="font-bold text-slate-900 text-sm sm:text-base mb-1">
                                      {facility.name}
                                    </h3>
                                    
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
                                            Show less <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" /></svg>
                                          </>
                                        ) : (
                                          <>
                                            Read more <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
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

        {/* DISCLAIMER AND FOOTER - INSIDE SCROLLABLE AREA */}
        <div className="w-full px-3 sm:px-6 pb-4">
          <DisclaimerBanner />
        </div>
        <Footer />
      </div>

      {/* --- AD BANNER + INPUT AREA (STICKY BOTTOM) --- */}
      <div className="sticky bottom-0 bg-white border-t border-slate-200/50 z-20">
        
        {/* MOBILE AD BANNER */}
        <div className="block xl:hidden w-full border-b border-slate-200/30">
          <div className="w-full h-[50px] flex items-center justify-center bg-gradient-to-r from-slate-50/50 to-slate-100/50">
            <div className="scale-[0.6] origin-center">
              <AdSlot />
            </div>
          </div>
        </div>
        
        {/* Input Controls */}
        <div className="w-full px-3 sm:px-6 py-2 sm:py-3">
          <div className="flex items-center gap-2 sm:gap-3">
            <button
              onClick={handleLocationClick}
              className="p-2 sm:p-3 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 hover:border-blue-300 transition-all text-slate-600 hover:text-blue-600"
              title="Share Location"
            >
              <MapPin size={18} className="sm:w-5 sm:h-5" />
            </button>
            
            <div className="flex-1 relative">
              <input
                type="text"
                className="w-full px-3 py-2 sm:px-5 sm:py-3 rounded-xl bg-slate-50 border border-slate-200 focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-500/20 transition-all text-slate-800 placeholder-slate-400 outline-none text-sm sm:text-base"
                placeholder="Describe what you need..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage(input)}
              />
            </div>

            <button
              onClick={() => handleSendMessage(input)}
              disabled={!input.trim() || loading}
              className={`p-2 sm:p-3 rounded-xl transition-all ${
                input.trim() && !loading 
                  ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-md hover:shadow-lg hover:scale-105" 
                  : "bg-slate-200 text-slate-400 cursor-not-allowed"
              }`}
            >
              <Send size={18} className="sm:w-5 sm:h-5" />
            </button>
          </div>
        </div>
        
      </div>
    </div>
  ); 
}
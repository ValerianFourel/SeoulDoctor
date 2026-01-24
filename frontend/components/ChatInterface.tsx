// components/ChatInterface.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import { Send, MapPin, Sparkles, Globe } from "lucide-react";
import Link from 'next/link';

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
  address?: string;
  website?: string;
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

  const scrollToBottom = () => {
    setTimeout(() => {
      scrollRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, 100);
  };

  const handleSendMessage = async (text: string, stateOverride?: Partial<State>) => {
    if (!text.trim()) return;

    const userMsg: Message = { role: "user", content: text };
    if (!stateOverride) setMessages((prev) => [...prev, userMsg]);
    
    setInput("");
    setLoading(true);

    // Scroll immediately when user sends message
    scrollToBottom();

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

      // Scroll again after AI response
      scrollToBottom();
    } catch (error) {
      console.error("API Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "I'm having trouble connecting right now. Please try again in a moment." },
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
    
    setMessages((prev) => [...prev, { role: "user", content: "📍 Sharing my location..." }]);
    setLoading(true);
    scrollToBottom();

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

  return (
    <div className="relative h-full w-full bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 flex flex-col">
      
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
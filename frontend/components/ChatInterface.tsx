"use client";

import { useState, useRef, useEffect } from "react";
import { Send, MapPin, Sparkles, ChevronDown } from "lucide-react";

// --- TYPES ---
type State = {
  specialty: string | null;
  location: string | null;
  lat_lon: number[] | null;
  willingness_to_travel: string;
  language_pref: string;
  keywords: string[];
  ready_to_search: boolean;
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
  const scrollRef = useRef<HTMLDivElement>(null);

  const [travelRadius, setTravelRadius] = useState("Nearby"); 
  const [currentState, setCurrentState] = useState<State>({
    specialty: null,
    location: null,
    lat_lon: null,
    willingness_to_travel: "Nearby",
    language_pref: "Korean is fine",
    keywords: [],
    ready_to_search: false,
  });

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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

      if (data.state) setCurrentState(data.state);
      
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
          lat_lon: [latitude, longitude],
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
    <div className="flex flex-col h-screen w-full bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100">
      
      {/* --- HEADER --- */}
      <div className="sticky top-0 z-10 backdrop-blur-xl bg-white/80 border-b border-slate-200/50">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl blur-sm opacity-75"></div>
                <div className="relative bg-gradient-to-br from-blue-500 to-purple-600 p-2 rounded-xl">
                  <Sparkles className="text-white" size={20} />
                </div>
              </div>
              <div>
                <h1 className="font-bold text-xl bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
                  SeoulMedBot
                </h1>
                <p className="text-xs text-slate-500">AI Medical Concierge</p>
              </div>
            </div>
            
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 transition-all text-sm font-medium text-slate-700"
            >
              <span className="hidden sm:inline">Settings</span>
              <ChevronDown className={`transition-transform ${showSettings ? 'rotate-180' : ''}`} size={16} />
            </button>
          </div>
          
          {/* Settings Panel */}
          {showSettings && (
            <div className="mt-4 p-4 rounded-xl bg-slate-50 border border-slate-200 animate-slideDown">
              <div className="flex flex-col gap-3">
                <div>
                  <label className="text-xs font-semibold text-slate-600 mb-2 block">Search Radius</label>
                  <div className="grid grid-cols-4 gap-2">
                    {TRAVEL_OPTIONS.map((opt) => (
                      <button
                        key={opt}
                        onClick={() => setTravelRadius(opt)}
                        className={`py-2.5 px-3 rounded-lg text-xs font-semibold transition-all ${
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
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="space-y-6">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fadeIn`}
              >
                <div className={`flex gap-3 max-w-[85%] ${msg.role === "user" ? 'flex-row-reverse' : 'flex-row'}`}>
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    msg.role === "user" 
                      ? "bg-gradient-to-br from-slate-600 to-slate-800" 
                      : "bg-gradient-to-br from-blue-500 to-purple-600"
                  }`}>
                    {msg.role === "user" ? (
                      <span className="text-white text-xs font-bold">You</span>
                    ) : (
                      <Sparkles className="text-white" size={14} />
                    )}
                  </div>

                  <div className="flex flex-col gap-3">
                    {/* Message Bubble */}
                    <div
                      className={`px-5 py-3 rounded-2xl ${
                        msg.role === "user"
                          ? "bg-gradient-to-r from-slate-700 to-slate-900 text-white"
                          : "bg-white border border-slate-200 text-slate-800 shadow-sm"
                      }`}
                    >
                      <p className="text-sm leading-relaxed">{msg.content}</p>
                    </div>

                    {/* Results Cards */}
                    {msg.results && msg.results.length > 0 && (
                      <div className="space-y-3">
                        {msg.results.map((facility, facilityIdx) => (
                          <div
                            key={facility.place_id}
                            className="bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-all overflow-hidden group animate-fadeIn"
                            style={{ animationDelay: `${facilityIdx * 100}ms` }}
                          >
                            <div className="p-4">
                              <div className="flex justify-between items-start mb-2">
                                <div className="flex-1">
                                  <h3 className="font-bold text-slate-900 text-base mb-1">{facility.name}</h3>
                                  <span className="inline-block px-2 py-1 bg-blue-50 text-blue-600 text-xs font-semibold rounded-md">
                                    {facility.category}
                                  </span>
                                </div>
                                {facility.english_confidence_score >= 4 && (
                                  <span className="flex items-center gap-1 bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-full text-xs font-bold">
                                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span>
                                    English OK
                                  </span>
                                )}
                              </div>

                              {facility.Summaries?.[0] && (
                                <p className="text-sm text-slate-600 leading-relaxed my-3 line-clamp-2">
                                  {facility.Summaries[0]}
                                </p>
                              )}

                              <div className="flex items-center justify-between pt-3 border-t border-slate-100">
                                <div className="flex items-center gap-2 text-slate-500">
                                  <MapPin size={14} className="text-blue-500" />
                                  <span className="text-sm font-medium">
                                    {facility.distance != null ? `${facility.distance.toFixed(1)} km away` : 'Distance N/A'}
                                  </span>
                                </div>
                                <a 
                                  href={`https://map.naver.com/v5/search/${encodeURIComponent(facility.name)}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white text-sm font-semibold rounded-lg hover:shadow-lg hover:shadow-blue-500/50 transition-all"
                                >
                                  View Map
                                </a>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="flex justify-start animate-fadeIn">
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <Sparkles className="text-white" size={14} />
                  </div>
                  <div className="bg-white border border-slate-200 rounded-2xl px-5 py-3 shadow-sm">
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

      {/* --- INPUT AREA --- */}
      <div className="sticky bottom-0 backdrop-blur-xl bg-white/80 border-t border-slate-200/50">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-end gap-3">
            <button
              onClick={handleLocationClick}
              className="p-3 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 hover:border-blue-300 transition-all text-slate-600 hover:text-blue-600 group"
              title="Share Location"
            >
              <MapPin size={20} className="group-hover:scale-110 transition-transform" />
            </button>
            
            <div className="flex-1 relative">
              <input
                type="text"
                className="w-full px-5 py-3.5 pr-12 rounded-xl bg-white border-2 border-slate-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 transition-all text-slate-800 placeholder-slate-400 outline-none"
                placeholder="Describe what you need..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage(input)}
              />
            </div>

            <button
              onClick={() => handleSendMessage(input)}
              disabled={!input.trim() || loading}
              className={`p-3.5 rounded-xl transition-all ${
                input.trim() && !loading 
                  ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg shadow-blue-500/50 hover:shadow-xl hover:shadow-blue-500/50 hover:scale-105" 
                  : "bg-slate-200 text-slate-400 cursor-not-allowed"
              }`}
            >
              <Send size={20} />
            </button>
          </div>
          <p className="text-xs text-slate-400 text-center mt-3">
            SeoulMedBot can make mistakes. Verify important medical information.
          </p>
        </div>
      </div>
    </div>
  );
}
"use client";

import { useState, useRef, useEffect } from "react";
import { Send, MapPin, Navigation, Bot, User } from "lucide-react";

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
  // Chat History
  const [messages, setMessages] = useState<Message[]>([
    { role: "ai", content: "Hello! I'm SeoulMedBot. Tell me what kind of doctor you need (e.g. 'English speaking dentist in Gangnam')." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // State Management (The "Clipboard")
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

  // Auto-scroll to bottom of chat
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- HANDLERS ---

  const handleSendMessage = async (text: string, stateOverride?: Partial<State>) => {
    if (!text.trim()) return;

    // 1. Add User Message to UI
    const userMsg: Message = { role: "user", content: text };
    // Only add to history if it's not a hidden location update
    if (!stateOverride) setMessages((prev) => [...prev, userMsg]);
    
    setInput("");
    setLoading(true);

    // 2. Prepare Payload (Enforce UI slider value)
    const stateToSend = {
      ...currentState,
      ...stateOverride,
      willingness_to_travel: travelRadius, // Always use the current slider value
    };

    try {
      // 3. Call Backend
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          current_state: stateToSend,
        }),
      });

      const data = await response.json();

      // 4. Update State & UI
      if (data.state) setCurrentState(data.state);
      
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: data.response,
          results: data.results, // Backend returns list of doctors here
        },
      ]);
    } catch (error) {
      console.error("API Error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "ai", content: "Sorry, I'm having trouble connecting to the server. Please try again." },
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
    
    // Optimistic UI update
    setMessages((prev) => [...prev, { role: "user", content: "📍 Shared precise location." }]);
    setLoading(true);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        // Send hidden message to backend with coords
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
    <div className="flex flex-col h-[700px] w-full max-w-lg bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-100">
      
      {/* --- HEADER --- */}
      <div className="bg-blue-600 p-4 text-white">
        <div className="flex items-center gap-2">
            <Bot size={24} />
            <h1 className="font-bold text-lg">SeoulMedBot</h1>
        </div>
        <p className="text-blue-100 text-xs mt-1">AI Concierge for Medical Services</p>
      </div>

      {/* --- SETTINGS BAR (Radius) --- */}
      <div className="bg-slate-50 p-3 border-b border-gray-200">
        <div className="flex justify-between items-center text-xs text-gray-500 mb-2 px-1">
            <span className="font-semibold">Search Radius</span>
            <span>{travelRadius === "Neighborhood" ? "< 2km" : travelRadius === "Nearby" ? "< 5km" : travelRadius === "City-wide" ? "< 10km" : "Any"}</span>
        </div>
        <div className="flex bg-gray-200 p-1 rounded-lg">
          {TRAVEL_OPTIONS.map((opt) => (
            <button
              key={opt}
              onClick={() => setTravelRadius(opt)}
              className={`flex-1 text-[10px] py-1.5 rounded-md transition-all font-medium ${
                travelRadius === opt
                  ? "bg-white text-blue-600 shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              {opt.replace("Neighborhood", "Close").replace("Don't Care", "Any")}
            </button>
          ))}
        </div>
      </div>

      {/* --- CHAT AREA --- */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 bg-gray-50">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}
          >
            {/* Bubble */}
            <div
              className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm shadow-sm ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-br-none"
                  : "bg-white text-gray-800 border border-gray-100 rounded-bl-none"
              }`}
            >
              {msg.content}
            </div>

            {/* Results Cards (Only if Backend sent results) */}
            {msg.results && msg.results.length > 0 && (
              <div className="mt-3 space-y-3 w-full max-w-[85%]">
                {msg.results.map((facility) => (
                  <div
                    key={facility.place_id}
                    className="bg-white p-3 rounded-xl border border-blue-100 shadow-sm hover:shadow-md transition-shadow"
                  >
                    <div className="flex justify-between items-start">
                        <div>
                            <div className="font-bold text-gray-900 text-sm">{facility.name}</div>
                            <div className="text-xs text-blue-600 font-medium mb-1">{facility.category}</div>
                        </div>
                        {facility.english_confidence_score >= 4 && (
                            <span className="text-[10px] bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-bold">
                                English OK
                            </span>
                        )}
                    </div>
                    
                    <div className="text-xs text-gray-500 my-2 line-clamp-2">
                        {facility.Summaries?.[0] || "No summary available."}
                    </div>

                    <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-50">
                        <span className="text-xs text-gray-400 flex items-center gap-1">
                            <Navigation size={12} /> {facility.distance.toFixed(1)} km away
                        </span>
                        <a 
                            href={`https://map.naver.com/v5/search/${encodeURIComponent(facility.name)}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[10px] bg-blue-50 text-blue-600 px-3 py-1.5 rounded-md font-semibold hover:bg-blue-100 transition-colors"
                        >
                            Open Map
                        </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        
        {loading && (
            <div className="flex items-center gap-2 text-xs text-gray-400 ml-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce delay-75" />
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce delay-150" />
                Thinking...
            </div>
        )}
        <div ref={scrollRef} />
      </div>

      {/* --- INPUT AREA --- */}
      <div className="p-4 bg-white border-t border-gray-100">
        <div className="flex items-center gap-2 bg-gray-50 p-1.5 rounded-xl border border-gray-200 focus-within:ring-2 focus-within:ring-blue-100 transition-all">
          <button
            onClick={handleLocationClick}
            className="p-2 text-gray-500 hover:text-blue-600 hover:bg-white rounded-lg transition-colors"
            title="Share Location"
          >
            <MapPin size={20} />
          </button>
          
          <input
            type="text"
            className="flex-1 bg-transparent border-none focus:ring-0 text-sm px-2 text-gray-700 placeholder-gray-400"
            placeholder="Describe your symptoms..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSendMessage(input)}
          />

          <button
            onClick={() => handleSendMessage(input)}
            disabled={!input.trim() || loading}
            className={`p-2 rounded-lg transition-colors ${
                input.trim() && !loading 
                ? "bg-blue-600 text-white hover:bg-blue-700 shadow-sm" 
                : "bg-gray-200 text-gray-400 cursor-not-allowed"
            }`}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}

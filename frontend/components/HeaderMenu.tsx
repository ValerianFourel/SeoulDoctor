'use client';

import Link from 'next/link';
import { useState, useRef, useEffect } from 'react';

export default function HeaderMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Clear timeout to prevent memory leaks if component unmounts
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  const handleMouseEnter = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setIsOpen(true);
  };

  const handleMouseLeave = () => {
    // Add a small delay before closing to allow moving mouse across gaps
    timeoutRef.current = setTimeout(() => {
      setIsOpen(false);
    }, 300); // 300ms grace period
  };

  return (
    <div 
      className="fixed top-4 right-4 z-[100]"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="relative">
        {/* Hamburger Menu Button */}
        <button
          className={`bg-white text-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 transition-all duration-200 ${
            isOpen ? 'bg-gray-100 ring-2 ring-blue-100' : 'hover:bg-gray-50'
          }`}
          aria-label="Menu"
          onClick={() => setIsOpen(!isOpen)} // Toggle on click for mobile/touch support
        >
          {/* Three lines icon */}
          <svg 
            className="w-6 h-6" 
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
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <div className="absolute top-full right-0 min-w-[220px]">
            {/* INVISIBLE BRIDGE: 
              The 'pt-2' here creates an invisible fill between the button 
              and the menu so the mouse never "leaves" the hover area.
            */}
            <div className="pt-2">
              <div className="bg-white text-gray-800 rounded-lg shadow-2xl border border-gray-200 py-2 overflow-hidden animate-in fade-in zoom-in-95 duration-100 origin-top-right">
                
                <Link 
                  href="/" 
                  onClick={() => setIsOpen(false)}
                  className="block px-4 py-3 text-sm hover:bg-blue-50 transition-colors flex items-center gap-2"
                >
                  <span>🏠</span> Home
                </Link>
                
                <div className="border-t border-gray-200 my-1"></div>
                
                <Link 
                  href="/about" 
                  onClick={() => setIsOpen(false)}
                  className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors flex items-center gap-2"
                >
                  <span>📖</span> About
                </Link>
                <Link 
                  href="/how-it-works" 
                  onClick={() => setIsOpen(false)}
                  className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors flex items-center gap-2"
                >
                  <span>⚙️</span> How It Works
                </Link>
                <Link 
                  href="/faq" 
                  onClick={() => setIsOpen(false)}
                  className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors flex items-center gap-2"
                >
                  <span>❓</span> FAQ
                </Link>
                
                <div className="border-t border-gray-200 my-1"></div>
                
                <Link 
                  href="/disclaimer" 
                  onClick={() => setIsOpen(false)}
                  className="block px-4 py-2.5 text-sm hover:bg-amber-50 transition-colors text-amber-700 font-medium flex items-center gap-2"
                >
                  <span>⚠️</span> Disclaimer
                </Link>
                <Link 
                  href="/privacy" 
                  onClick={() => setIsOpen(false)}
                  className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors flex items-center gap-2"
                >
                  <span>🔒</span> Privacy Policy
                </Link>
                
                <div className="border-t border-gray-200 my-1"></div>
                
                <div className="px-4 py-3 text-xs text-gray-500 bg-gray-50">
                  <p className="font-bold text-red-600 mb-1 flex items-center gap-1">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                    </span>
                    Emergency
                  </p>
                  <p>119 (Services) | 1339 (Hotline)</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
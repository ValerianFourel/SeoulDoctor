// components/HeaderMenu.tsx
'use client';

import Link from 'next/link';
import { useState, useRef, useEffect } from 'react';

export default function HeaderMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

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
    timeoutRef.current = setTimeout(() => {
      setIsOpen(false);
    }, 300);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-[100] bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
        {/* Logo/Brand */}
          <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            {/* I removed the div with the gradient background here */}
            <img 
              src="/img/logo_v3.svg" 
              alt="Logo" 
              className="w-8 h-8 object-contain" // Slightly increased size since the padding is gone
            />
            <div>
              <h1 className="font-bold text-lg bg-gradient-to-r from-slate-800 to-slate-600 bg-clip-text text-transparent">
                Seoul Doc
              </h1>
            </div>
          </Link>

        <div className="flex items-center gap-3">
          {/* PayPal Donate Button */}
          <a 
            href="https://paypal.me/vfseoul"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-all duration-200 shadow-sm hover:shadow-md font-medium text-sm"
          >
            <div className="w-5 h-5 flex items-center justify-center border-2 border-white rounded-full">
              <span className="text-xs font-bold">₩</span>
            </div>
            <span>Donate via PayPal</span>
          </a>

          {/* Menu Button */}
          <div 
            className="relative"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
          >
            <button
              className={`bg-white text-gray-800 p-2 rounded-lg border border-gray-200 transition-all duration-200 ${
                isOpen ? 'bg-gray-100 ring-2 ring-blue-100' : 'hover:bg-gray-50'
              }`}
              aria-label="Menu"
              onClick={() => setIsOpen(!isOpen)}
            >
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
              <div className="absolute top-full right-0 mt-2 min-w-[220px]">
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
                  {/* ADD THIS NEW LINK */}
                  <Link 
                    href="/contact" 
                    onClick={() => setIsOpen(false)}
                    className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors flex items-center gap-2"
                  >
                    <span>✉️</span> Contact Us
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
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
'use client';

import Link from 'next/link';
import { useState } from 'react';

export default function HeaderMenu() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="fixed top-4 right-4 z-50">
      <div className="relative">
        {/* Hamburger Menu Button */}
        <button
          onMouseEnter={() => setIsOpen(true)}
          onMouseLeave={() => setIsOpen(false)}
          className="bg-white hover:bg-gray-100 text-gray-800 p-3 rounded-lg shadow-lg transition-all duration-200 border border-gray-200"
          aria-label="Menu"
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

        {/* Dropdown Menu (appears below button on hover) */}
        {isOpen && (
          <div
            onMouseEnter={() => setIsOpen(true)}
            onMouseLeave={() => setIsOpen(false)}
            className="absolute top-full right-0 mt-2 bg-white text-gray-800 rounded-lg shadow-2xl border border-gray-200 py-2 min-w-[200px]"
          >
            <Link 
              href="/" 
              className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors"
            >
              🏠 Home
            </Link>
            
            <div className="border-t border-gray-200 my-1"></div>
            
            <Link 
              href="/about" 
              className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors"
            >
              📖 About
            </Link>
            <Link 
              href="/how-it-works" 
              className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors"
            >
              ⚙️ How It Works
            </Link>
            <Link 
              href="/faq" 
              className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors"
            >
              ❓ FAQ
            </Link>
            
            <div className="border-t border-gray-200 my-1"></div>
            
            <Link 
              href="/disclaimer" 
              className="block px-4 py-2.5 text-sm hover:bg-amber-50 transition-colors text-amber-700 font-medium"
            >
              ⚠️ Disclaimer
            </Link>
            <Link 
              href="/privacy" 
              className="block px-4 py-2.5 text-sm hover:bg-blue-50 transition-colors"
            >
              🔒 Privacy Policy
            </Link>
            
            <div className="border-t border-gray-200 my-1"></div>
            
            <div className="px-4 py-2 text-xs text-gray-500">
              <p className="font-semibold text-red-600">Emergency:</p>
              <p>119 (Services) | 1339 (Hotline)</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
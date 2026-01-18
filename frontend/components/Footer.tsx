'use client';

import Link from 'next/link';
import { useState } from 'react';

export default function Footer() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <footer className="bg-gray-800 text-white py-4 mt-auto relative">
      <div className="container mx-auto px-4">
        {/* Compact Main Footer */}
        <div className="flex flex-col md:flex-row justify-between items-center gap-3">
          {/* Left: Branding + Emergency (always visible) */}
          <div className="text-center md:text-left">
            <p className="text-sm font-semibold">Seoul Medical Finder</p>
            <p className="text-xs text-gray-400">
              🚨 Emergency: <strong>119</strong> | Medical: <strong>1339</strong>
            </p>
          </div>

          {/* Center: Quick Links Button (hover to expand) */}
          <div className="relative">
            <button
              onMouseEnter={() => setIsMenuOpen(true)}
              onMouseLeave={() => setIsMenuOpen(false)}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center gap-2"
            >
              <span>Pages</span>
              <svg 
                className={`w-4 h-4 transition-transform duration-200 ${isMenuOpen ? 'rotate-180' : ''}`}
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {/* Dropdown Menu (appears on hover) */}
            {isMenuOpen && (
              <div
                onMouseEnter={() => setIsMenuOpen(true)}
                onMouseLeave={() => setIsMenuOpen(false)}
                className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 bg-white text-gray-800 rounded-lg shadow-xl border border-gray-200 py-2 min-w-[180px] z-50"
              >
                <Link 
                  href="/about" 
                  className="block px-4 py-2 text-sm hover:bg-blue-50 transition-colors"
                >
                  📖 About
                </Link>
                <Link 
                  href="/how-it-works" 
                  className="block px-4 py-2 text-sm hover:bg-blue-50 transition-colors"
                >
                  ⚙️ How It Works
                </Link>
                <Link 
                  href="/faq" 
                  className="block px-4 py-2 text-sm hover:bg-blue-50 transition-colors"
                >
                  ❓ FAQ
                </Link>
                <div className="border-t border-gray-200 my-1"></div>
                <Link 
                  href="/disclaimer" 
                  className="block px-4 py-2 text-sm hover:bg-amber-50 transition-colors text-amber-700 font-medium"
                >
                  ⚠️ Disclaimer
                </Link>
                <Link 
                  href="/privacy" 
                  className="block px-4 py-2 text-sm hover:bg-blue-50 transition-colors"
                >
                  🔒 Privacy Policy
                </Link>
              </div>
            )}
          </div>

          {/* Right: Copyright */}
          <div className="text-center md:text-right">
            <p className="text-xs text-gray-400">
              © {new Date().getFullYear()} Seoul Medical Finder
            </p>
            <p className="text-xs text-gray-500">For info only • Not medical advice</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
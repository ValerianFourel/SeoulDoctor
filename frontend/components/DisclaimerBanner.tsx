'use client';

import Link from 'next/link';
import { useState } from 'react';

export default function DisclaimerBanner() {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <div className="bg-amber-50 border-b border-amber-300 px-3 py-2 sm:px-4 sm:py-2.5">
      <div className="flex items-center justify-between gap-2 max-w-7xl mx-auto">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className="text-base sm:text-lg flex-shrink-0">⚠️</span>
          <div className="flex-1 min-w-0">
            <p className="text-[10px] sm:text-xs text-amber-800 leading-tight">
              <strong>Info only</strong> - Not medical advice. Always consult healthcare professionals. 
              <Link href="/disclaimer" className="underline hover:text-amber-900 font-semibold ml-1">
                Full disclaimer
              </Link>
              {' • '}
              <span className="font-semibold whitespace-nowrap">Emergency: 119</span>
            </p>
          </div>
        </div>
        <button
          onClick={() => setIsVisible(false)}
          className="flex-shrink-0 text-amber-600 hover:text-amber-800 p-1"
          aria-label="Dismiss"
        >
          <svg className="h-3 w-3 sm:h-4 sm:w-4" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>
    </div>
  );
}
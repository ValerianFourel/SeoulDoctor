'use client';

import Link from 'next/link';
import { useState } from 'react';

export default function DisclaimerBanner() {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <div className="bg-amber-50 border-l-4 border-amber-500 p-4 mb-4 rounded-r-lg shadow-sm">
      <div className="flex items-start justify-between">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <span className="text-2xl">⚠️</span>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-semibold text-amber-900">
              Important Notice
            </h3>
            <p className="text-sm text-amber-800 mt-1">
              This chatbot helps you <strong>find medical facilities</strong> in Seoul. 
              It does <strong>NOT provide medical diagnosis or treatment advice</strong>. 
              Always consult qualified healthcare professionals for medical decisions.
            </p>
            <p className="text-xs text-amber-700 mt-2">
              <Link href="/disclaimer" className="underline hover:text-amber-900 font-semibold">
                Read full medical disclaimer
              </Link>
              {' • '}
              <span className="font-semibold">Emergency? Call 119</span>
            </p>
          </div>
        </div>
        <button
          onClick={() => setIsVisible(false)}
          className="flex-shrink-0 ml-4 text-amber-600 hover:text-amber-800"
          aria-label="Dismiss"
        >
          <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>
    </div>
  );
}
// components/AdSlotWrapper.tsx
'use client';

import { useEffect, useState } from 'react';
import AdSlot from './AdSlot';

export default function AdSlotWrapper() {
  const [hasConsent, setHasConsent] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    
    const checkConsent = () => {
      const savedConsent = localStorage.getItem('cookieConsent');
      if (savedConsent) {
        try {
          const parsed = JSON.parse(savedConsent);
          setHasConsent(parsed.advertising === true);
        } catch (error) {
          console.error('Error parsing consent:', error);
        }
      }
    };

    checkConsent();
    
    // Listen for consent changes
    window.addEventListener('storage', checkConsent);
    return () => window.removeEventListener('storage', checkConsent);
  }, []);

  if (!mounted || !hasConsent) {
    return null;
  }

  return (
    <div 
      className="w-[300px] min-w-[300px]" 
      style={{ width: '300px', minWidth: '300px' }}
    >
      <AdSlot />
    </div>
  );
}
// components/ConsentScripts.tsx
'use client';

import { useEffect, useState } from 'react';
import Script from 'next/script';

export default function ConsentScripts() {
  const [hasConsent, setHasConsent] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    
    const savedConsent = localStorage.getItem('cookieConsent');
    if (savedConsent) {
      try {
        const parsed = JSON.parse(savedConsent);
        setHasConsent(parsed.advertising === true);
      } catch (error) {
        console.error('Error parsing consent:', error);
        setHasConsent(false);
      }
    }
  }, []);

  // Don't render script tags until mounted (client-side only)
  if (!mounted || !hasConsent) return null;

  return (
    <>
      <Script
        async
        src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2786202112029582"
        crossOrigin="anonymous"
        strategy="afterInteractive"
      />
    </>
  );
}
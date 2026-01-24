// components/ConsentScripts.tsx
'use client';

import { useEffect, useState } from 'react';
import type { ReactElement } from 'react';

const ADSENSE_APPROVED = false; // Set to true once Google approves you

export default function ConsentScripts(): ReactElement | null {
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

  useEffect(() => {
    if (!mounted || !hasConsent || !ADSENSE_APPROVED) {
      return;
    }

    // Remove any existing AdSense script to avoid duplicates
    const existingScript = document.querySelector(
      'script[src*="adsbygoogle.js"]'
    );
    if (existingScript) {
      existingScript.remove();
    }

    // Create and inject AdSense script manually
    const script = document.createElement('script');
    script.src = 'https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2786202112029582';
    script.async = true;
    script.crossOrigin = 'anonymous';
    
    document.head.appendChild(script);
    
    console.log('✅ AdSense script loaded');
  }, [mounted, hasConsent]);

  return null;
}
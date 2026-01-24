// components/AdSlot.tsx
'use client';

import { useEffect, useRef, useState } from 'react';

export default function AdSlot() {
  const adRef = useRef<HTMLDivElement>(null);
  const [adLoaded, setAdLoaded] = useState(false);

  useEffect(() => {
    const loadAd = () => {
      const adContainer = adRef.current;
      if (!adContainer) {
        console.warn('Ad container not found');
        return;
      }

      // Check if container has width
      const rect = adContainer.getBoundingClientRect();
      if (rect.width === 0) {
        console.warn('Ad slot has no width, not initializing');
        return;
      }

      // Check if AdSense is loaded
      if (typeof window === 'undefined' || !(window as any).adsbygoogle) {
        console.warn('AdSense script not loaded yet');
        return;
      }

      // Initialize ad
      try {
        ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
        setAdLoaded(true);
        console.log('✅ Ad initialized');
      } catch (error) {
        console.error('Error initializing ad:', error);
      }
    };

    // Wait a bit for AdSense script to load
    const timer = setTimeout(loadAd, 1000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div 
      ref={adRef}
      className="w-[300px] h-[250px] bg-gray-100 flex items-center justify-center"
      style={{ width: '300px', height: '250px' }}
    >
      <ins
        className="adsbygoogle"
        style={{ display: 'block', width: '300px', height: '250px' }}
        data-ad-client="ca-pub-2786202112029582"
        data-ad-slot="YOUR_AD_SLOT_ID"
        data-ad-format="auto"
        data-full-width-responsive="false"
      />
      {!adLoaded && (
        <div className="text-gray-400 text-sm">Advertisement</div>
      )}
    </div>
  );
}
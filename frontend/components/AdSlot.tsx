// components/AdSlot.tsx
'use client';

import { useEffect, useRef, useState } from 'react';

export default function AdSlot() {
  const adRef = useRef<HTMLDivElement>(null);
  const [adState, setAdState] = useState<'loading' | 'loaded' | 'failed'>('loading');

  useEffect(() => {
    const loadAd = () => {
      const adContainer = adRef.current;
      if (!adContainer) {
        console.warn('Ad container not found');
        setAdState('failed');
        return;
      }

      // Check if container has width
      const rect = adContainer.getBoundingClientRect();
      if (rect.width === 0) {
        console.warn('Ad slot has no width, not initializing');
        setAdState('failed');
        return;
      }

      // Check if AdSense is loaded
      if (typeof window === 'undefined' || !(window as any).adsbygoogle) {
        console.warn('AdSense script not loaded yet');
        setAdState('failed');
        return;
      }

      // Initialize ad
      try {
        ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
        setAdState('loaded');
        console.log('✅ Ad initialized');
      } catch (error) {
        console.error('Error initializing ad:', error);
        setAdState('failed');
      }
    };

    // Wait a bit for AdSense script to load
    const timer = setTimeout(loadAd, 1000);

    // Fallback timeout - if ad doesn't load in 5 seconds, show placeholder
    const fallbackTimer = setTimeout(() => {
      if (adState === 'loading') {
        setAdState('failed');
      }
    }, 5000);

    return () => {
      clearTimeout(timer);
      clearTimeout(fallbackTimer);
    };
  }, [adState]);

  return (
    <div 
      ref={adRef}
      className="relative w-[300px] h-[250px]"
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
      
      {/* Placeholder shown while loading or if ad fails */}
      {adState !== 'loaded' && (
        <div 
          className="absolute inset-0 w-[300px] h-[250px] bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg border border-slate-200 flex items-center justify-center"
          style={{ width: '300px', height: '250px' }}
        >
          {adState === 'loading' && (
            <div className="flex flex-col items-center gap-2">
              <div className="w-8 h-8 border-2 border-slate-300 border-t-blue-500 rounded-full animate-spin"></div>
              <span className="text-xs text-slate-400">Loading ad...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
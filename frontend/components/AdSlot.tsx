"use client";
import { useEffect, useRef, useState } from "react";

export default function AdSlot() {
  const adRef = useRef<HTMLModElement>(null);
  const [shouldLoad, setShouldLoad] = useState(false);

  useEffect(() => {
    // Check if element is visible and has width
    const checkVisibility = () => {
      if (!adRef.current) return false;
      
      const rect = adRef.current.getBoundingClientRect();
      const styles = window.getComputedStyle(adRef.current.parentElement || adRef.current);
      
      // Element must be visible and have width
      return (
        styles.display !== 'none' &&
        styles.visibility !== 'hidden' &&
        rect.width > 0
      );
    };

    // Wait for layout and check visibility
    const timer = setTimeout(() => {
      if (checkVisibility()) {
        setShouldLoad(true);
      }
    }, 100);

    // Also listen for resize events (in case screen size changes)
    const handleResize = () => {
      if (!shouldLoad && checkVisibility()) {
        setShouldLoad(true);
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
    };
  }, [shouldLoad]);

  useEffect(() => {
    if (!shouldLoad || !adRef.current) return;

    try {
      // Double-check element still has width before pushing
      const rect = adRef.current.getBoundingClientRect();
      if (rect.width === 0) {
        console.log("Ad container has no width, skipping");
        return;
      }

      // Check if already initialized
      if (adRef.current.getAttribute('data-ad-status')) {
        return;
      }

      adRef.current.setAttribute('data-ad-status', 'filled');
      ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
      
    } catch (e) {
      console.error("AdSense error:", e);
    }
  }, [shouldLoad]);

  return (
    <div className="w-full flex justify-center">
      <ins
        ref={adRef}
        className="adsbygoogle"
        style={{ 
          display: "block",
          minWidth: "250px",
          minHeight: "250px",
          width: "100%"
        }}
        data-ad-client="ca-pub-2786202112029582"
        data-ad-slot="6649005456"
        data-ad-format="auto"
        data-full-width-responsive="true"
      ></ins>
    </div>
  );
}
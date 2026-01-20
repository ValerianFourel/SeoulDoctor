"use client";
import { useEffect, useRef } from "react";

export default function AdSlot() {
  const adRef = useRef<HTMLModElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const hasLoaded = useRef(false);

  useEffect(() => {
    if (hasLoaded.current) return;

    const loadAd = () => {
      if (!adRef.current || !containerRef.current) return;

      // Verify the container actually has dimensions
      const containerRect = containerRef.current.getBoundingClientRect();
      if (containerRect.width === 0 || containerRect.height === 0) {
        console.log("Container not ready, width:", containerRect.width);
        return;
      }

      try {
        hasLoaded.current = true;
        ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
      } catch (err) {
        console.error("AdSense error:", err);
        hasLoaded.current = false; // Allow retry
      }
    };

    // Use IntersectionObserver to ensure element is actually visible
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.intersectionRatio > 0) {
            // Element is visible, wait a bit for layout to settle
            setTimeout(loadAd, 200);
            observer.disconnect();
          }
        });
      },
      { threshold: 0.01 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, []);

  return (
    <div ref={containerRef} className="w-full min-w-[250px]">
      <ins
        ref={adRef}
        className="adsbygoogle"
        style={{ 
          display: "block",
          minWidth: "250px",
          minHeight: "250px"
        }}
        data-ad-client="ca-pub-2786202112029582"
        data-ad-slot="6649005456"
        data-ad-format="auto"
        data-full-width-responsive="true"
      ></ins>
    </div>
  );
}
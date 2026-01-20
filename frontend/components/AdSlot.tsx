"use client";
import { useEffect, useRef, useState } from "react";

export default function AdSlot() {
  const adRef = useRef<HTMLModElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isClient, setIsClient] = useState(false);
  const hasLoaded = useRef(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isClient || hasLoaded.current) return;

    const loadAd = () => {
      if (!adRef.current || !containerRef.current) return;

      // Verify the container actually has dimensions
      const containerRect = containerRef.current.getBoundingClientRect();
      if (containerRect.width === 0 || containerRect.height === 0) {
        console.log("Container not ready, skipping ad load");
        return;
      }

      try {
        hasLoaded.current = true;
        ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
      } catch (err) {
        console.error("AdSense error:", err);
        hasLoaded.current = false;
      }
    };

    // Use IntersectionObserver to ensure element is visible
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.intersectionRatio > 0) {
            setTimeout(loadAd, 300);
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
  }, [isClient]);

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
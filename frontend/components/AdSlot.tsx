"use client";
import { useEffect, useRef } from "react";

export default function AdSlot() {
  const adRef = useRef<HTMLModElement>(null);
  const hasLoaded = useRef(false);

  useEffect(() => {
    if (hasLoaded.current) return;

    // Wait for layout to settle
    const timer = setTimeout(() => {
      try {
        if (!adRef.current) return;

        // Final safety check
        const rect = adRef.current.getBoundingClientRect();
        if (rect.width === 0) {
          console.warn("Ad slot has no width, not initializing");
          return;
        }

        console.log("Initializing AdSense with width:", rect.width);
        hasLoaded.current = true;
        ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
      } catch (err) {
        console.error("AdSense error:", err);
      }
    }, 600);

    return () => clearTimeout(timer);
  }, []);

  return (
    <ins
      ref={adRef}
      className="adsbygoogle"
      style={{ 
        display: "inline-block",
        width: "100%",
        height: "50px"
      }}
      data-ad-client="ca-pub-2786202112029582"
      data-ad-slot="6649005456"
      data-ad-format="horizontal"
      data-full-width-responsive="false"
    ></ins>
  );
}
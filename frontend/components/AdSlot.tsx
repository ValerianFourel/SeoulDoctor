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
      if (!adRef.current || !containerRef.current) {
        console.log("Ad refs not ready");
        return;
      }

      // Check if parent container is actually visible (not hidden by CSS)
      const container = containerRef.current;
      const computedStyle = window.getComputedStyle(container);
      
      if (computedStyle.display === 'none' || computedStyle.visibility === 'hidden') {
        console.log("Ad container is hidden, skipping initialization");
        return;
      }

      // Check actual dimensions
      const rect = container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        console.log("Ad container has no dimensions:", rect.width, "x", rect.height);
        return;
      }

      // Additional check: is the aside parent visible?
      let parent = container.parentElement;
      while (parent) {
        const parentStyle = window.getComputedStyle(parent);
        if (parentStyle.display === 'none') {
          console.log("Parent element is hidden, skipping ad");
          return;
        }
        parent = parent.parentElement;
        // Stop at body
        if (parent?.tagName === 'BODY') break;
      }

      try {
        console.log("Initializing ad - container width:", rect.width);
        hasLoaded.current = true;
        ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
      } catch (err) {
        console.error("AdSense error:", err);
        hasLoaded.current = false;
      }
    };

    // Use IntersectionObserver to detect when element is visible
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.intersectionRatio > 0) {
            // Use requestAnimationFrame to ensure layout is complete
            requestAnimationFrame(() => {
              setTimeout(loadAd, 500);
            });
            observer.disconnect();
          }
        });
      },
      { 
        threshold: 0.01,
        rootMargin: '0px'
      }
    );

    if (containerRef.current) {
      // Double-check visibility before observing
      const rect = containerRef.current.getBoundingClientRect();
      if (rect.width > 0) {
        observer.observe(containerRef.current);
      } else {
        console.log("Container not visible on mount, not observing");
      }
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
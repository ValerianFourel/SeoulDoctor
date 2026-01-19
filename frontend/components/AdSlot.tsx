"use client";
import { useEffect, useRef } from "react";

export default function AdSlot() {
  const adRef = useRef<HTMLModElement>(null);

  useEffect(() => {
    try {
      // Check if the ad has already been loaded in this slot to prevent errors
      if (adRef.current && adRef.current.innerHTML === "") {
        ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
      }
    } catch (e) {
      console.error("AdSense error", e);
    }
  }, []);

  return (
    // The container constrains the width, but allows height to grow
    <div className="my-4 w-full flex justify-center overflow-hidden">
      <ins
        ref={adRef}
        className="adsbygoogle"
        // Changed style to block so 'data-ad-format="auto"' works correctly
        style={{ display: "block", minWidth: "250px", width: "100%" }} 
        data-ad-client="ca-pub-2786202112029582"
        data-ad-slot="6649005456" 
        data-ad-format="auto"
        data-full-width-responsive="true"
      ></ins>
    </div>
  );
}
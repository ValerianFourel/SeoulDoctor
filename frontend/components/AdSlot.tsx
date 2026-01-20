// components/TestAd.tsx
"use client";
import { useEffect } from "react";

export default function TestAd() {
  useEffect(() => {
    try {
      ((window as any).adsbygoogle = (window as any).adsbygoogle || []).push({});
    } catch (e) {
      console.error("Ad error:", e);
    }
  }, []);

  return (
    <div style={{ width: "100%", maxWidth: "728px", margin: "20px auto" }}>
      <ins
        className="adsbygoogle"
        style={{ display: "block" }}
        data-ad-client="ca-pub-2786202112029582"
        data-ad-slot="6649005456"
        data-ad-format="auto"
        data-full-width-responsive="true"
      />
    </div>
  );
}
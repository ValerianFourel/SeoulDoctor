"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

// Dynamically import AdSlot with no SSR
const AdSlot = dynamic(() => import("./AdSlot"), {
  ssr: false,
  loading: () => <div className="w-[250px] h-[250px]" /> // Placeholder
});

export default function ConditionalAdSlot() {
  const [mounted, setMounted] = useState(false);
  const [isLargeScreen, setIsLargeScreen] = useState(false);

  useEffect(() => {
    setMounted(true);
    
    const checkScreen = () => {
      setIsLargeScreen(window.innerWidth >= 1280);
    };

    checkScreen();
    window.addEventListener("resize", checkScreen);
    
    return () => window.removeEventListener("resize", checkScreen);
  }, []);

  // Don't render until client-side
  if (!mounted || !isLargeScreen) {
    return null;
  }

  return <AdSlot />;
}
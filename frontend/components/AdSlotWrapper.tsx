"use client";
import { useEffect, useState } from "react";
import AdSlot from "./AdSlot";

export default function AdSlotWrapper() {
  const [shouldRender, setShouldRender] = useState(false);

  useEffect(() => {
    const checkScreen = () => {
      // Only render if screen is >= 1280px (xl breakpoint)
      setShouldRender(window.innerWidth >= 1280);
    };

    // Check immediately
    checkScreen();

    // Listen for resize
    window.addEventListener('resize', checkScreen);

    return () => window.removeEventListener('resize', checkScreen);
  }, []);

  // Don't render AdSlot component at all on small screens
  if (!shouldRender) {
    return null;
  }

  return <AdSlot />;
}
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Footer from "../components/Footer";
import { Analytics } from "@vercel/analytics/react";
import { useEffect } from 'react';
import { usePathname } from 'next/navigation';

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export function ScrollToTop() {
  const pathname = usePathname();

  useEffect(() => {
    // Scroll to top whenever the route changes
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
}

export const metadata: Metadata = {
  title: "Seoul Medical Facility Finder | AI Medical Concierge",
  description: "Find the best English-speaking medical facilities in Seoul with AI assistance. Search by specialty, location, and language capabilities.",
  keywords: "Seoul medical facilities, English-speaking doctors Seoul, Korea healthcare, Seoul hospitals, medical concierge Seoul",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} flex flex-col min-h-screen`}>
        {/* Scroll to top on route change */}
        <ScrollToTop />
        
        {/* Main content area - takes up available space */}
        <main className="flex-grow">
          {children}
        </main>
        
        {/* Compact footer with hover menu */}
        <Footer />
        
        {/* Vercel Analytics - tracks page views automatically */}
        <Analytics />
      </body>
    </html>
  );
}
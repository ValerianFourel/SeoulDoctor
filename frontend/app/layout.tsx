import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import HeaderMenu from "../components/HeaderMenu";
import { Analytics } from "@vercel/analytics/react";
import Script from "next/script";
import AdSlotWrapper from "../components/AdSlotWrapper";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL || 
    process.env.VERCEL_URL 
      ? `https://${process.env.VERCEL_URL}`
      : 'http://localhost:3000'
  ),
  title: "Seoul Medical Facility Finder | AI Medical Concierge",
  description: "Find the best English-speaking medical facilities in Seoul with AI assistance. Search by specialty, location, and language capabilities.",
  keywords: "Seoul medical facilities, English-speaking doctors Seoul, Korea healthcare, Seoul hospitals, medical concierge Seoul",
  openGraph: {
    title: "Seoul Medical Facility Finder",
    description: "Find the best English-speaking medical facilities in Seoul with AI assistance",
    type: "website",
    locale: "en_US",
    siteName: "Seoul Medical Facility Finder",
  },
  twitter: {
    card: "summary_large_image",
    title: "Seoul Medical Facility Finder",
    description: "Find the best English-speaking medical facilities in Seoul with AI assistance",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* AdSense Script */}
        <Script
          async
          src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2786202112029582"
          crossOrigin="anonymous"
          strategy="lazyOnload"
        />
      </head>
      <body className={`${inter.className} flex flex-col min-h-screen bg-gray-50`}>
        <HeaderMenu />
        
        <main className="flex-grow flex justify-center w-full">
          
          {/* Left Ad Column (Hidden on Mobile) */}
          <aside className="hidden xl:flex w-[320px] shrink-0 flex-col items-end pr-4 pt-6">
            <div className="sticky top-24">
              <AdSlotWrapper />
            </div>
          </aside>

          {/* Center Content (Chat Interface) */}
          <div className="w-full max-w-4xl flex flex-col bg-white shadow-sm min-h-screen">
            {children}
          </div>

          {/* Right Ad Column (Hidden on Mobile) */}
          <aside className="hidden xl:flex w-[320px] shrink-0 flex-col items-start pl-4 pt-6">
            <div className="sticky top-24">
              <AdSlotWrapper />
            </div>
          </aside>

        </main>
        
        <Analytics />
      </body>
    </html>
  );
}
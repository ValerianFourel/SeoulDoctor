import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Script from "next/script";
import { Analytics } from "@vercel/analytics/react";

import "./globals.css";
import HeaderMenu from "../components/HeaderMenu";
import Footer from "../components/Footer";
import AdSlotWrapper from "../components/AdSlotWrapper";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : "http://localhost:3000");

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),

  title: "Seoul Medical Facility Finder | AI Medical Concierge",
  description:
    "Find the best English-speaking medical facilities in Seoul with AI assistance. Search by specialty, location, and language capabilities.",

  keywords: [
    "Seoul medical facilities",
    "English-speaking doctors Seoul",
    "Korea healthcare",
    "Seoul hospitals",
    "medical concierge Seoul",
  ],

  other: {
    "google-adsense-account": "ca-pub-2786202112029582",
  },

  openGraph: {
    title: "Seoul Medical Facility Finder",
    description:
      "Find the best English-speaking medical facilities in Seoul with AI assistance",
    type: "website",
    locale: "en_US",
    siteName: "Seoul Medical Facility Finder",
    url: siteUrl,
  },

  twitter: {
    card: "summary_large_image",
    title: "Seoul Medical Facility Finder",
    description:
      "Find the best English-speaking medical facilities in Seoul with AI assistance",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Google AdSense */}
        <Script
          async
          src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2786202112029582"
          crossOrigin="anonymous"
          strategy="afterInteractive"
        />
      </head>

      <body
        className={`${inter.className} flex flex-col min-h-screen bg-gray-50`}
      >
        <HeaderMenu />

        <main className="flex-grow flex justify-center w-full min-h-[calc(100vh-64px)]">
          {/* Left Ad Column */}
          <aside className="hidden xl:flex w-[320px] shrink-0 flex-col items-end pr-4 pt-6">
            <div className="sticky top-24">
              <AdSlotWrapper />
            </div>
          </aside>

          {/* Main Content */}
          <div className="w-full max-w-4xl flex flex-col bg-white shadow-sm">
            {children}
          </div>

          {/* Right Ad Column */}
          <aside className="hidden xl:flex w-[320px] shrink-0 flex-col items-start pl-4 pt-6">
            <div className="sticky top-24">
              <AdSlotWrapper />
            </div>
          </aside>
        </main>

        <Footer />
        <Analytics />
      </body>
    </html>
  );
}

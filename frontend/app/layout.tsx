import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Footer from "../components/Footer";
import HeaderMenu from "../components/HeaderMenu";
import { Analytics } from "@vercel/analytics/react";

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
      <body className={`${inter.className} flex flex-col min-h-screen`}>
        {/* Floating header menu button (top right) */}
        <HeaderMenu />
        
        {/* Main content area - takes up available space */}
        <main className="flex-grow">
          {children}
        </main>
        
        {/* Footer - 25% more compact */}
        <Footer />
        
        {/* Vercel Analytics - tracks page views automatically */}
        <Analytics />
      </body>
    </html>
  );
}
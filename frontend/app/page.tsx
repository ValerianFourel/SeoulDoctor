import type { Metadata } from "next";
import ChatInterface from "../components/ChatInterface";

export const metadata: Metadata = {
  title: "Seoul Medical Facility Finder | Find English-Speaking Doctors in Seoul",
  description: "AI-powered chatbot to find medical facilities in Seoul. Search by specialty, location, and English language capabilities. Free and easy to use.",
  keywords: "Seoul doctors, English-speaking hospitals Seoul, Korea medical facilities, Seoul healthcare finder",
  icons: {
    icon: "/img/logo.ico",
  },
  openGraph: {
    title: "Seoul Medical Facility Finder",
    description: "Find the best medical facilities in Seoul with AI assistance",
    type: "website",
  },
};

export default function Home() {
  return <ChatInterface />;
}

/*
NOTES:
- Enhanced metadata for better SEO
- The Footer (from layout.tsx) will automatically appear at the bottom
- The ChatInterface should have DisclaimerBanner at the top
- All navigation is handled by the Footer component

OPTIONAL: If you want quick links visible on the home page above the chat,
you could add them here like:

export default function Home() {
  return (
    <>
      <div className="bg-blue-50 py-2 text-center text-sm">
        <a href="/about" className="mx-2 hover:underline">About</a>
        <a href="/how-it-works" className="mx-2 hover:underline">How It Works</a>
        <a href="/faq" className="mx-2 hover:underline">FAQ</a>
      </div>
      <ChatInterface />
    </>
  );
}
*/
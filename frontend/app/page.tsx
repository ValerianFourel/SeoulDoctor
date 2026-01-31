import type { Metadata } from "next";
import ChatInterface from "../components/ChatInterface";

export const metadata: Metadata = {
  title: "Seoul Medical Facility Finder | Find English-Speaking Doctors in Seoul",
  description: "AI-powered chatbot to find medical facilities in Seoul. Search by specialty, location, and English language capabilities. Free and easy to use.",
  keywords: "Seoul doctors, English-speaking hospitals Seoul, Korea medical facilities, Seoul healthcare finder",
  icons: {
    icon: "/img/favicon.ico",
  },
  openGraph: {
    title: "Seoul Medical Facility Finder",
    description: "Find the best medical facilities in Seoul with AI assistance",
    type: "website",
  },
};

export default function Home() {
  return (
    <div className="h-full flex flex-col">
       <ChatInterface />
    </div>
  );
}

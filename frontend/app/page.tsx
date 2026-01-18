import type { Metadata } from "next";
import ChatInterface from "../components/ChatInterface";

export const metadata: Metadata = {
  title: "Home", // You can customize the page title here
  icons: {
    icon: "/img/logo.ico",
  },
};

export default function Home() {
  return <ChatInterface />;
}
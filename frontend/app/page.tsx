import ChatInterface from "@/components/ChatInterface";
import AdSlot from "@/components/AdSlot";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-4 md:p-8 bg-slate-100">
      
      {/* Title Section */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-slate-800 tracking-tight">
          Seoul<span className="text-blue-600">Med</span>Bot
        </h1>
        <p className="text-slate-500 text-sm mt-2">
          Your AI Medical Concierge in South Korea 🇰🇷
        </p>
      </div>

      <div className="w-full max-w-6xl flex flex-col lg:flex-row gap-8 justify-center items-start">
        
        {/* Left Sidebar (Ads) - Hidden on Mobile */}
        <div className="hidden lg:flex flex-col gap-6 w-[320px]">
          <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
            <h3 className="text-xs font-bold text-gray-400 mb-2 uppercase tracking-wider">Sponsored</h3>
            <AdSlot />
          </div>
          <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
            <AdSlot />
          </div>
        </div>

        {/* Main Chat App */}
        <div className="flex-1 w-full max-w-lg mx-auto">
          <ChatInterface />
        </div>

        {/* Right Sidebar (Ads) - Hidden on Mobile */}
        <div className="hidden lg:flex flex-col gap-6 w-[320px]">
          <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
             <h3 className="text-xs font-bold text-gray-400 mb-2 uppercase tracking-wider">Partners</h3>
            <AdSlot />
          </div>
        </div>

      </div>

      {/* Mobile Ad (Bottom) */}
      <div className="lg:hidden mt-8 w-full max-w-xs">
        <AdSlot />
      </div>

    </main>
  );
}

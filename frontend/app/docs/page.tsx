import type { Metadata } from "next";
import Link from "next/link";
import {
  ArrowRight,
  Bot,
  CheckCircle2,
  Code2,
  Database,
  ExternalLink,
  KeyRound,
  MapPin,
  Search,
  Server,
  ShieldCheck,
  Terminal,
} from "lucide-react";

export const metadata: Metadata = {
  title: "Documentation | Seoul Doc",
  description:
    "Set up, run, and integrate Seoul Doc, the AI-powered medical facility finder for Seoul.",
};

const navigation = [
  ["Overview", "overview"],
  ["How it works", "architecture"],
  ["Quick start", "quick-start"],
  ["Configuration", "configuration"],
  ["API", "api"],
  ["Safety", "safety"],
] as const;

const backendEnvironment = [
  ["GROQ_API_KEY", "Required", "Powers intent extraction and response generation."],
  ["OPENAI_API_KEY", "Required", "Creates embeddings for semantic search."],
  ["GOOGLE_MAPS_API_KEY", "Required", "Geocodes locations and resolves place names."],
  ["KAKAO_REST_API_KEY", "Recommended", "Provides Korea-focused geocoding fallback."],
  ["NAVER_CLIENT_ID", "Optional", "Reserved for Naver integrations."],
  ["NAVER_CLIENT_SECRET", "Optional", "Secret paired with the Naver client ID."],
] as const;

const CodeBlock = ({ children }: { children: string }) => (
  <pre className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-950 p-5 text-sm leading-7 text-slate-200 shadow-inner">
    <code>{children}</code>
  </pre>
);

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-white text-slate-900">
      <section className="relative overflow-hidden border-b border-slate-200 bg-slate-950 px-6 py-20 text-white sm:px-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_rgba(59,130,246,0.25),_transparent_42%)]" />
        <div className="relative mx-auto max-w-5xl">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-400/30 bg-blue-400/10 px-3 py-1 text-sm font-medium text-blue-200">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Developer documentation
          </div>
          <h1 className="max-w-3xl text-4xl font-bold tracking-tight sm:text-6xl">
            Build with Seoul Doc.
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
            Everything you need to run the conversational medical facility finder,
            understand its retrieval pipeline, and connect a client to the API.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a href="#quick-start" className="inline-flex items-center gap-2 rounded-xl bg-blue-500 px-5 py-3 font-semibold text-white transition hover:bg-blue-400">
              Quick start <ArrowRight className="h-4 w-4" />
            </a>
            <Link href="/" className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-5 py-3 font-semibold text-slate-200 transition hover:border-slate-500">
              Open the app <ExternalLink className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      <div className="mx-auto grid max-w-7xl grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)]">
        <aside className="border-b border-slate-200 px-6 py-6 lg:border-b-0 lg:border-r lg:px-8 lg:py-12">
          <nav className="flex gap-2 overflow-x-auto lg:sticky lg:top-24 lg:flex-col" aria-label="Documentation sections">
            {navigation.map(([label, target]) => (
              <a key={target} href={`#${target}`} className="whitespace-nowrap rounded-lg px-3 py-2 text-sm font-medium text-slate-600 transition hover:bg-blue-50 hover:text-blue-700">
                {label}
              </a>
            ))}
          </nav>
        </aside>

        <main className="min-w-0 px-6 py-12 sm:px-10 lg:px-14">
          <div className="max-w-4xl space-y-20">
            <section id="overview" className="scroll-mt-24">
              <p className="text-sm font-bold uppercase tracking-widest text-blue-600">Overview</p>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Medical search that understands context</h2>
              <p className="mt-5 text-lg leading-8 text-slate-600">
                Seoul Doc helps English and Korean speakers find medical facilities in Seoul. It combines conversational intent extraction, location intelligence, exact evidence retrieval, and semantic ranking to turn natural-language requests into useful facility recommendations.
              </p>
              <div className="mt-8 grid gap-4 sm:grid-cols-2">
                {[
                  [Bot, "Conversational", "Maintains specialty, location, travel, and preference context across turns."],
                  [Search, "Hybrid retrieval", "Combines specific BM25 evidence with dense semantic search."],
                  [MapPin, "Location aware", "Supports GPS, districts, neighborhoods, addresses, and landmarks."],
                  [ShieldCheck, "Safety first", "Detects emergency language and clearly separates search from medical advice."],
                ].map(([Icon, title, copy]) => {
                  const FeatureIcon = Icon as typeof Bot;
                  return (
                    <article key={title as string} className="rounded-2xl border border-slate-200 p-5 shadow-sm">
                      <FeatureIcon className="h-6 w-6 text-blue-600" />
                      <h3 className="mt-4 font-bold">{title as string}</h3>
                      <p className="mt-2 text-sm leading-6 text-slate-600">{copy as string}</p>
                    </article>
                  );
                })}
              </div>
            </section>

            <section id="architecture" className="scroll-mt-24">
              <p className="text-sm font-bold uppercase tracking-widest text-blue-600">How it works</p>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">From question to recommendation</h2>
              <div className="mt-8 space-y-4">
                {[
                  ["01", "Understand", "The router detects language and intent, then extracts specialty, location, travel distance, and preferences."],
                  ["02", "Resolve", "Google Maps resolves the location, with Kakao Maps available as a Korea-focused fallback."],
                  ["03", "Retrieve", "An agentic loop chooses broad semantic search, specific BM25 evidence search, refinement, or completion."],
                  ["04", "Rank and answer", "Candidates are filtered and ranked before the response is formatted into practical facility suggestions."],
                ].map(([number, title, copy]) => (
                  <div key={number} className="flex gap-5 rounded-2xl bg-slate-50 p-5">
                    <span className="font-mono text-sm font-bold text-blue-600">{number}</span>
                    <div><h3 className="font-bold">{title}</h3><p className="mt-1 leading-7 text-slate-600">{copy}</p></div>
                  </div>
                ))}
              </div>
            </section>

            <section id="quick-start" className="scroll-mt-24">
              <p className="text-sm font-bold uppercase tracking-widest text-blue-600">Quick start</p>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Run locally</h2>
              <p className="mt-4 leading-7 text-slate-600">Use Python 3.12+ and Node.js 18+. Start the API and web client in separate terminals.</p>
              <div className="mt-7 space-y-7">
                <div><h3 className="mb-3 flex items-center gap-2 font-bold"><Server className="h-5 w-5 text-blue-600" />1. Start the backend</h3><CodeBlock>{`cd backend\npython -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\nuvicorn main:app --reload --host 0.0.0.0 --port 8000`}</CodeBlock></div>
                <div><h3 className="mb-3 flex items-center gap-2 font-bold"><Code2 className="h-5 w-5 text-blue-600" />2. Start the frontend</h3><CodeBlock>{`cd frontend\nnpm ci\nnpm run dev`}</CodeBlock></div>
                <div className="flex gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-5 text-emerald-950">
                  <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
                  <p className="text-sm leading-6">Open <strong>http://localhost:3000</strong>. The first backend startup downloads the facility dataset and initializes its local search index, so it may take longer than later starts.</p>
                </div>
              </div>
            </section>

            <section id="configuration" className="scroll-mt-24">
              <p className="text-sm font-bold uppercase tracking-widest text-blue-600">Configuration</p>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Environment variables</h2>
              <h3 className="mt-8 mb-3 flex items-center gap-2 font-bold"><KeyRound className="h-5 w-5 text-blue-600" />Backend <code className="text-sm text-slate-500">backend/.env</code></h3>
              <div className="overflow-hidden rounded-2xl border border-slate-200">
                {backendEnvironment.map(([name, status, description]) => (
                  <div key={name} className="grid gap-1 border-b border-slate-200 p-4 last:border-0 sm:grid-cols-[190px_100px_1fr] sm:gap-4">
                    <code className="text-sm font-semibold text-blue-700">{name}</code><span className="text-xs font-bold uppercase tracking-wide text-slate-500">{status}</span><p className="text-sm text-slate-600">{description}</p>
                  </div>
                ))}
              </div>
              <h3 className="mt-8 mb-3 font-bold">Frontend <code className="text-sm text-slate-500">frontend/.env.local</code></h3>
              <CodeBlock>{`NEXT_PUBLIC_API_URL=http://localhost:8000\nNEXT_PUBLIC_SITE_URL=http://localhost:3000`}</CodeBlock>
            </section>

            <section id="api" className="scroll-mt-24">
              <p className="text-sm font-bold uppercase tracking-widest text-blue-600">API reference</p>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">FastAPI endpoints</h2>
              <p className="mt-4 leading-7 text-slate-600">When the backend is running, interactive OpenAPI documentation is available at <code className="rounded bg-slate-100 px-1.5 py-1 text-sm">http://localhost:8000/docs</code>.</p>
              <div className="mt-7 space-y-3">
                {[
                  ["POST", "/chat", "Send a message and conversation state; receive the assistant response and updated state."],
                  ["POST", "/set_travel_preference", "Update the maximum travel preference for a conversation."],
                  ["POST", "/consent", "Persist analytics, advertising, and functional cookie choices."],
                ].map(([method, path, description]) => (
                  <article key={path} className="rounded-2xl border border-slate-200 p-5">
                    <div className="flex items-center gap-3"><span className="rounded-md bg-blue-100 px-2 py-1 font-mono text-xs font-bold text-blue-700">{method}</span><code className="font-semibold">{path}</code></div>
                    <p className="mt-3 text-sm leading-6 text-slate-600">{description}</p>
                  </article>
                ))}
              </div>
              <h3 className="mt-8 mb-3 flex items-center gap-2 font-bold"><Terminal className="h-5 w-5 text-blue-600" />Example request</h3>
              <CodeBlock>{`curl -X POST http://localhost:8000/chat \\\n  -H "Content-Type: application/json" \\\n  -d '{"message":"Find an English-speaking dentist in Gangnam","current_state":{}}'`}</CodeBlock>
            </section>

            <section id="safety" className="scroll-mt-24 rounded-3xl bg-amber-50 p-7 sm:p-9">
              <p className="text-sm font-bold uppercase tracking-widest text-amber-700">Safety</p>
              <h2 className="mt-3 text-2xl font-bold">A search tool, not medical advice</h2>
              <p className="mt-4 leading-7 text-amber-950/80">Seoul Doc helps people discover facilities; it does not diagnose conditions or replace a qualified clinician. Facility details may change and should be confirmed directly. For a medical emergency in Korea, call <strong>119</strong>; for the medical information hotline, call <strong>1339</strong>.</p>
            </section>

            <section className="border-t border-slate-200 pt-10 text-center">
              <Database className="mx-auto h-8 w-8 text-blue-600" />
              <h2 className="mt-4 text-2xl font-bold">Ready to explore?</h2>
              <p className="mt-2 text-slate-600">Try the live experience or return to the setup guide.</p>
              <div className="mt-6 flex justify-center gap-3"><Link href="/" className="rounded-xl bg-blue-600 px-5 py-3 font-semibold text-white hover:bg-blue-700">Open Seoul Doc</Link><a href="#quick-start" className="inline-flex items-center gap-2 rounded-xl border border-slate-300 px-5 py-3 font-semibold hover:bg-slate-50"><Terminal className="h-4 w-4" />Setup guide</a></div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}

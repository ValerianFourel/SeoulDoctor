import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";


export const metadata: Metadata = {
  title: "Privacy and Data Processing - Seoul Medical Facility Finder",
  description:
    "How Seoul Medical Facility Finder processes chat, location, and technical data.",
  icons: {
    icon: "/img/favicon.ico",
  },
};

const externalLinkClass =
  "font-medium text-blue-700 underline decoration-blue-300 underline-offset-2 hover:text-blue-900";

function PolicySection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="mb-8">
      <h2 className="mb-4 text-2xl font-semibold text-gray-800">{title}</h2>
      {children}
    </section>
  );
}

export default function Privacy() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 px-4 py-12 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl rounded-lg bg-white p-8 shadow-xl md:p-12">
        <Link
          href="/"
          className="mb-6 inline-flex text-blue-600 transition-colors hover:text-blue-800"
        >
          ← Back to Chat
        </Link>

        <h1 className="mb-6 text-4xl font-bold text-gray-900">
          Privacy and Data Processing
        </h1>

        <div className="prose prose-lg max-w-none">
          <p className="mb-8 text-gray-600">
            <strong>Effective and last updated:</strong> September 2, 2026
          </p>

          <div className="mb-8 rounded-lg border-l-4 border-amber-500 bg-amber-50 p-6">
            <h2 className="mb-2 text-xl font-semibold text-amber-950">
              Read this before entering medical information
            </h2>
            <p className="text-gray-800">
              Your chat can include symptoms, conditions, and care preferences. The
              service sends that text to external AI providers to generate a reply.
              Do not enter your name, identification numbers, medical-record numbers,
              payment details, or information you do not want processed externally.
              This service is a facility-search aid, not a medical-record system or an
              emergency service.
            </p>
          </div>

          <PolicySection title="Information processed">
            <ul className="ml-4 space-y-2 text-gray-700">
              <li>
                • <strong>Chat and search content:</strong> Messages, medical
                specialties, symptoms or conditions you mention, language preference,
                facility preferences, and the conversation state returned by the app.
              </li>
              <li>
                • <strong>Location:</strong> A neighborhood, address, or coordinates
                only when you type or share them for a location-based search.
              </li>
              <li>
                • <strong>Technical data:</strong> Hosting and API infrastructure may
                process IP addresses, timestamps, request metadata, browser details,
                and operational logs. The app hashes the apparent client address in
                memory to enforce a short-term rate limit.
              </li>
              <li>
                • <strong>Consent preferences:</strong> Cookie choices are stored in
                your browser&apos;s local storage.
              </li>
              <li>
                • <strong>Contact requests:</strong> If you use the contact form, it
                sends your name, email address, and message through Web3Forms to the
                SeoulDoc contact mailbox.
              </li>
            </ul>
          </PolicySection>

          <PolicySection title="AI and infrastructure providers">
            <p className="mb-3 text-gray-700">
              SeoulDoc uses the following services to answer searches:
            </p>
            <ul className="ml-4 space-y-3 text-gray-700">
              <li>
                • <strong>OpenRouter and its selected model provider:</strong> The app
                sends chat messages, derived search criteria, and selected facility or
                review evidence to OpenRouter for language-model processing. Provider
                retention and training policies can vary. Review the{" "}
                <a
                  className={externalLinkClass}
                  href="https://openrouter.ai/docs/guides/privacy/data-collection"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  OpenRouter data-collection guide
                </a>{" "}
                and{" "}
                <a
                  className={externalLinkClass}
                  href="https://openrouter.ai/docs/guides/privacy/provider-logging"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  provider policy guide
                </a>
                .
              </li>
              <li>
                • <strong>OpenAI embeddings:</strong> Search text and facility index
                text can be sent to the OpenAI embeddings API for semantic retrieval.
                OpenAI documents its API retention controls in its{" "}
                <a
                  className={externalLinkClass}
                  href="https://platform.openai.com/docs/models/default-usage-policies-by-endpoint"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  API data-controls guide
                </a>
                .
              </li>
              <li>
                • <strong>Hugging Face:</strong> Hugging Face hosts the application,
                dataset, and attached storage used by the deployment.
              </li>
              <li>
                • <strong>Google Maps or Kakao Maps:</strong> When configured, a
                location query can be sent to these services for geocoding. Links to
                map pages leave SeoulDoc and are governed by the destination&apos;s policy.
              </li>
              <li>
                • <strong>Web3Forms:</strong> Web3Forms processes contact-form
                submissions and forwards them by email. Web3Forms says submission
                storage depends on the plan: 30 days for free plans and one year for
                pro plans. See its{' '}
                <a
                  className={externalLinkClass}
                  href="https://web3forms.com/platforms/javascript-contact-form"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  contact-form documentation
                </a>
                .
              </li>
            </ul>
          </PolicySection>

          <PolicySection title="Storage and retention">
            <ul className="ml-4 space-y-2 text-gray-700">
              <li>
                • The normal chat interface keeps the visible conversation in page
                memory. Reloading the page clears that interface state.
              </li>
              <li>
                • SeoulDoc does not include an application database for user chat
                histories. Limited request text may enter application logs only when
                analytics consent is enabled.
              </li>
              <li>
                • Hosting, AI, embedding, analytics, and geocoding providers apply
                their own logging and retention policies. Their settings and terms
                control data held by those providers.
              </li>
              <li>
                • Contact messages can remain in the recipient mailbox and with
                Web3Forms according to its current plan and retention policy.
              </li>
            </ul>
          </PolicySection>

          <PolicySection title="Cookies, analytics, and advertising">
            <p className="text-gray-700">
              Essential consent settings support the service. Analytics and advertising
              scripts remain denied until you grant the corresponding consent. You can
              change browser storage or cookie settings at any time. External analytics
              or advertising services, when enabled, apply their own privacy policies.
            </p>
          </PolicySection>

          <PolicySection title="Security and choices">
            <p className="mb-3 text-gray-700">
              The service uses HTTPS in deployment, owner-only permissions for local
              evaluation artifacts, bounded inputs, and request throttling. No internet
              service can guarantee absolute security.
            </p>
            <p className="text-gray-700">
              You can avoid providing optional location details, decline analytics and
              advertising consent, clear browser data, or stop using the service. Rights
              to access, correct, delete, restrict, or object to processing depend on
              your jurisdiction and the data held by each provider.
            </p>
          </PolicySection>

          <PolicySection title="Children and international processing">
            <p className="text-gray-700">
              The service is not directed to children under 18. Hosting and service
              providers may process data in countries outside your residence, where
              privacy laws may differ.
            </p>
          </PolicySection>

          <PolicySection title="Changes and contact">
            <p className="mb-3 text-gray-700">
              Material changes will appear on this page with an updated date.
            </p>
            <p className="text-gray-700">
              For privacy questions or requests, use the{" "}
              <Link className={externalLinkClass} href="/contact">
                contact page
              </Link>
              . Include no medical details in a support request unless they are needed
              to resolve it.
            </p>
          </PolicySection>
        </div>
      </div>
    </div>
  );
}

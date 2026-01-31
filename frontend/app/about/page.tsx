import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "About - Seoul Medical Facility Finder",
  description: "Learn about our mission to help residents and visitors find quality medical care in Seoul",
  icons: {
    icon: "/img/favicon.ico",
  },
};

export default function About() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-xl p-8 md:p-12">
        <Link 
          href="/" 
          className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-6 transition-colors"
        >
          ← Back to Chat
        </Link>
        
        <h1 className="text-4xl font-bold text-gray-900 mb-6">About Seoul Medical Facility Finder</h1>
        
        <div className="prose prose-lg max-w-none">
          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Our Mission</h2>
            <p className="text-gray-700 leading-relaxed mb-4">
              Finding quality medical care in a new city can be overwhelming, especially when dealing with language barriers and unfamiliar healthcare systems. Seoul Medical Facility Finder was created to bridge this gap by providing an intelligent, conversational way to discover medical facilities across Seoul that match your specific needs.
            </p>
            <p className="text-gray-700 leading-relaxed">
              Whether you're a resident, expat, or visitor, our platform helps you navigate Seoul's extensive healthcare landscape with confidence and ease.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">What We Provide</h2>
            <div className="bg-blue-50 border-l-4 border-blue-500 p-6 mb-4">
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span><strong>Intelligent facility matching</strong> based on specialty, location, and language capabilities</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span><strong>Comprehensive information</strong> including reviews, summaries, and key highlights from real patient experiences</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span><strong>English-speaking facility identification</strong> for international patients</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span><strong>Location-based search</strong> to find facilities in your neighborhood or area of interest</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span><strong>Bilingual support</strong> with responses available in both Korean and English</span>
                </li>
              </ul>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">What We Are NOT</h2>
            <div className="bg-amber-50 border-l-4 border-amber-500 p-6 mb-4">
              <p className="text-gray-700 leading-relaxed mb-3">
                <strong className="text-amber-800">Important:</strong> Seoul Medical Facility Finder is a <strong>facility directory and recommendation service</strong>, not a medical provider or diagnostic tool.
              </p>
              <ul className="space-y-2 text-gray-700">
                <li className="flex items-start">
                  <span className="text-amber-600 mr-2">✗</span>
                  <span>We do NOT provide medical diagnoses or treatment recommendations</span>
                </li>
                <li className="flex items-start">
                  <span className="text-amber-600 mr-2">✗</span>
                  <span>We do NOT replace professional medical advice</span>
                </li>
                <li className="flex items-start">
                  <span className="text-amber-600 mr-2">✗</span>
                  <span>We do NOT have access to your medical records or health information</span>
                </li>
                <li className="flex items-start">
                  <span className="text-amber-600 mr-2">✗</span>
                  <span>We are NOT affiliated with any specific medical facility or healthcare provider</span>
                </li>
              </ul>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Our Commitment</h2>
            <p className="text-gray-700 leading-relaxed mb-4">
              We are committed to maintaining accurate, up-to-date information about Seoul's medical facilities. Our platform aggregates data from public sources and user reviews to provide you with comprehensive insights that help you make informed decisions about where to seek care.
            </p>
            <p className="text-gray-700 leading-relaxed">
              This is an independent project developed to improve access to healthcare information in Seoul. We continuously work to expand our database and improve our recommendation algorithms to serve you better.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Who Should Use This Service</h2>
            <ul className="space-y-3 text-gray-700">
              <li className="flex items-start">
                <span className="text-green-600 mr-2">✓</span>
                <span>Expats and international residents seeking English-speaking healthcare providers</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-600 mr-2">✓</span>
                <span>Seoul residents exploring medical facilities in their neighborhood</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-600 mr-2">✓</span>
                <span>Visitors to Seoul who may need medical services during their stay</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-600 mr-2">✓</span>
                <span>Anyone looking for specialist care or specific medical services in Seoul</span>
              </li>
            </ul>
          </section>

          <section className="bg-gray-50 p-6 rounded-lg">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Contact & Feedback</h2>
            <p className="text-gray-700 leading-relaxed">
              This is an evolving project, and we welcome your feedback. If you encounter issues, have suggestions for improvement, or would like to contribute data about medical facilities, please reach out through the feedback mechanisms provided on our platform.
            </p>
          </section>
        </div>

        <div className="mt-8 pt-6 border-t border-gray-200">
          <p className="text-sm text-gray-600">
            <strong>Remember:</strong> For medical emergencies, always call 119 (Korea's emergency number) or go to the nearest emergency room immediately.
          </p>
        </div>
      </div>
    </div>
  );
}
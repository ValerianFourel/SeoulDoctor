import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How It Works - Seoul Medical Facility Finder",
  description: "Learn how our AI-powered chatbot helps you find the right medical facilities in Seoul",
  icons: {
    icon: "/img/logo.ico",
  },
};

export default function HowItWorks() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-xl p-8 md:p-12">
        <Link 
          href="/" 
          className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-6 transition-colors"
        >
          ← Back to Chat
        </Link>
        
        <h1 className="text-4xl font-bold text-gray-900 mb-6">How the Medical Chatbot Works</h1>
        
        <div className="prose prose-lg max-w-none">
          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Overview</h2>
            <p className="text-gray-700 leading-relaxed mb-4">
              Our chatbot uses advanced artificial intelligence and natural language processing to understand your healthcare needs and match you with appropriate medical facilities in Seoul. The system combines conversational AI with a comprehensive database of medical facilities to provide personalized recommendations.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">The Technology Behind the Chatbot</h2>
            
            <div className="space-y-6">
              <div className="bg-blue-50 p-6 rounded-lg">
                <h3 className="text-xl font-semibold text-blue-900 mb-3">1. Natural Language Understanding</h3>
                <p className="text-gray-700 leading-relaxed">
                  When you type a message, our AI system analyzes your text to understand:
                </p>
                <ul className="mt-3 space-y-2 text-gray-700 ml-4">
                  <li>• What type of medical care you're looking for (specialty, treatment type)</li>
                  <li>• Where you want to find care (neighborhood, district, or city-wide)</li>
                  <li>• Your language preference (English or Korean)</li>
                  <li>• Any specific requirements (English-speaking staff, specific services)</li>
                </ul>
              </div>

              <div className="bg-green-50 p-6 rounded-lg">
                <h3 className="text-xl font-semibold text-green-900 mb-3">2. Comprehensive Medical Facility Database</h3>
                <p className="text-gray-700 leading-relaxed mb-3">
                  Our database includes thousands of medical facilities across Seoul, with information gathered from:
                </p>
                <ul className="space-y-2 text-gray-700 ml-4">
                  <li>• <strong>Public health registries:</strong> Official data on licensed medical facilities</li>
                  <li>• <strong>Patient reviews:</strong> Aggregated feedback from platforms like Naver Maps</li>
                  <li>• <strong>Facility information:</strong> Services offered, operating hours, contact details</li>
                  <li>• <strong>Language capabilities:</strong> Identification of English-speaking facilities</li>
                </ul>
                <p className="text-gray-700 leading-relaxed mt-3">
                  Each facility entry includes summaries, key highlights, and curated information to help you make informed decisions.
                </p>
              </div>

              <div className="bg-purple-50 p-6 rounded-lg">
                <h3 className="text-xl font-semibold text-purple-900 mb-3">3. Intelligent Matching & Ranking</h3>
                <p className="text-gray-700 leading-relaxed mb-3">
                  The system uses multiple strategies to find the best matches for your needs:
                </p>
                <ul className="space-y-2 text-gray-700 ml-4">
                  <li>• <strong>Category filtering:</strong> Matches your specialty needs with facility types</li>
                  <li>• <strong>Location filtering:</strong> Identifies facilities in your preferred area using district and neighborhood data</li>
                  <li>• <strong>Semantic search:</strong> Uses AI embeddings to understand the meaning of your query and find relevant facilities beyond simple keyword matching</li>
                  <li>• <strong>Language filtering:</strong> Prioritizes English-speaking facilities when requested</li>
                </ul>
                <p className="text-gray-700 leading-relaxed mt-3">
                  The combination of these techniques ensures you see the most relevant facilities first.
                </p>
              </div>

              <div className="bg-amber-50 p-6 rounded-lg">
                <h3 className="text-xl font-semibold text-amber-900 mb-3">4. Conversational Response Generation</h3>
                <p className="text-gray-700 leading-relaxed">
                  Once relevant facilities are identified, our language model crafts a natural, conversational response that:
                </p>
                <ul className="mt-3 space-y-2 text-gray-700 ml-4">
                  <li>• Summarizes the top recommendations in easy-to-understand language</li>
                  <li>• Highlights key information like location, specialties, and patient feedback</li>
                  <li>• Responds in your preferred language (Korean or English)</li>
                  <li>• Provides actionable next steps (contact information, location details)</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">What Information We Use</h2>
            <p className="text-gray-700 leading-relaxed mb-4">
              To provide accurate recommendations, the chatbot analyzes:
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="border border-gray-200 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 mb-2">From Your Messages:</h4>
                <ul className="space-y-1 text-sm text-gray-600">
                  <li>• Medical specialty or condition mentioned</li>
                  <li>• Location preferences</li>
                  <li>• Language requirements</li>
                  <li>• Any specific keywords or requirements</li>
                </ul>
              </div>
              <div className="border border-gray-200 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 mb-2">From Our Database:</h4>
                <ul className="space-y-1 text-sm text-gray-600">
                  <li>• Facility category and specialties</li>
                  <li>• Location (district, neighborhood)</li>
                  <li>• Patient review summaries</li>
                  <li>• English language capabilities</li>
                  <li>• Operating hours and contact info</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Understanding the Limitations</h2>
            <div className="bg-red-50 border-l-4 border-red-500 p-6">
              <h3 className="text-lg font-semibold text-red-900 mb-3">What the Chatbot Cannot Do:</h3>
              <ul className="space-y-2 text-gray-700">
                <li className="flex items-start">
                  <span className="text-red-600 mr-2">✗</span>
                  <span><strong>Make medical diagnoses</strong> – The chatbot cannot assess symptoms or diagnose conditions</span>
                </li>
                <li className="flex items-start">
                  <span className="text-red-600 mr-2">✗</span>
                  <span><strong>Provide medical advice</strong> – It recommends facilities, not treatments or medications</span>
                </li>
                <li className="flex items-start">
                  <span className="text-red-600 mr-2">✗</span>
                  <span><strong>Make appointments</strong> – You'll need to contact facilities directly to schedule visits</span>
                </li>
                <li className="flex items-start">
                  <span className="text-red-600 mr-2">✗</span>
                  <span><strong>Access real-time availability</strong> – Information may not reflect current wait times or availability</span>
                </li>
                <li className="flex items-start">
                  <span className="text-red-600 mr-2">✗</span>
                  <span><strong>Guarantee accuracy</strong> – While we strive for accuracy, facility information may change</span>
                </li>
              </ul>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">How to Get the Best Results</h2>
            <div className="bg-green-50 p-6 rounded-lg">
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">1.</span>
                  <span><strong>Be specific</strong> – Mention the type of care you need (e.g., "dermatologist" rather than just "doctor")</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">2.</span>
                  <span><strong>Include location</strong> – Name your neighborhood or district for more relevant results</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">3.</span>
                  <span><strong>State language needs</strong> – If you need English-speaking care, mention it early</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">4.</span>
                  <span><strong>Ask follow-up questions</strong> – Refine your search by asking for more options or different areas</span>
                </li>
              </ul>
            </div>
          </section>

          <section className="bg-gray-50 p-6 rounded-lg">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Data Accuracy & Updates</h2>
            <p className="text-gray-700 leading-relaxed mb-3">
              We regularly update our database to maintain accuracy. However, medical facilities may change their services, hours, or contact information without notice. We recommend:
            </p>
            <ul className="space-y-2 text-gray-700 ml-4">
              <li>• Calling facilities to confirm current information before visiting</li>
              <li>• Verifying insurance acceptance if applicable</li>
              <li>• Checking current operating hours, especially on holidays</li>
            </ul>
          </section>
        </div>

        <div className="mt-8 pt-6 border-t border-gray-200">
          <p className="text-sm text-gray-600 text-center">
            Have questions about how the chatbot works? Feel free to ask directly in the chat interface!
          </p>
        </div>
      </div>
    </div>
  );
}
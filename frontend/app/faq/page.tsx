import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "FAQ - Seoul Medical Facility Finder",
  description: "Frequently asked questions about finding medical facilities in Seoul",
  icons: {
    icon: "/img/convertico-logo_v3.ico",
  },
};

export default function FAQ() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-xl p-8 md:p-12">
        <Link 
          href="/" 
          className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-6 transition-colors"
        >
          ← Back to Chat
        </Link>
        
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Frequently Asked Questions</h1>
        <p className="text-gray-600 mb-8">Find answers to common questions about using Seoul Medical Facility Finder</p>
        
        <div className="space-y-6">
          {/* General Questions */}
          <section>
            <h2 className="text-2xl font-semibold text-blue-800 mb-4 border-b-2 border-blue-200 pb-2">
              General Questions
            </h2>
            
            <div className="space-y-6">
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  What is Seoul Medical Facility Finder?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Seoul Medical Facility Finder is an AI-powered chatbot that helps you discover medical facilities (hospitals, clinics, specialists) across Seoul based on your needs. Simply describe what you're looking for in natural language, and the chatbot will recommend relevant facilities with detailed information including reviews, location, and language capabilities.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Is this service free?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Yes! Seoul Medical Facility Finder is completely free to use. The service is supported by contextual advertising, which allows us to maintain the platform without charging users.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Who created this service?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Seoul Medical Facility Finder is an independent project developed to improve access to healthcare information in Seoul. It is not affiliated with any specific medical facility, healthcare provider, or government agency. Our goal is to make it easier for residents and visitors to navigate Seoul's healthcare landscape.
                </p>
              </div>
            </div>
          </section>

          {/* Using the Service */}
          <section>
            <h2 className="text-2xl font-semibold text-blue-800 mb-4 border-b-2 border-blue-200 pb-2">
              Using the Service
            </h2>
            
            <div className="space-y-6">
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  What types of questions can I ask?
                </h3>
                <p className="text-gray-700 leading-relaxed mb-3">
                  You can ask about finding medical facilities based on:
                </p>
                <ul className="space-y-2 text-gray-700 ml-6">
                  <li>• <strong>Specialty:</strong> "Find a dermatologist," "I need an orthopedic surgeon"</li>
                  <li>• <strong>Location:</strong> "Clinics in Gangnam," "hospitals near Hongdae"</li>
                  <li>• <strong>Language:</strong> "English-speaking dentist," "facilities with English staff"</li>
                  <li>• <strong>Services:</strong> "Physical therapy," "mental health counseling"</li>
                  <li>• <strong>Combinations:</strong> "English-speaking pediatrician in Itaewon"</li>
                </ul>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  What types of questions should I NOT ask?
                </h3>
                <p className="text-gray-700 leading-relaxed mb-3">
                  This service is NOT appropriate for:
                </p>
                <ul className="space-y-2 text-gray-700 ml-6">
                  <li>• <strong>Medical diagnosis:</strong> "Do I have the flu?" "What's wrong with my back?"</li>
                  <li>• <strong>Treatment advice:</strong> "What medication should I take?" "How do I treat this symptom?"</li>
                  <li>• <strong>Emergency situations:</strong> If you're experiencing a medical emergency, call 119 immediately</li>
                  <li>• <strong>Personal medical advice:</strong> "Should I see a doctor for this?"</li>
                </ul>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Can I use this chatbot in Korean?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Yes! The chatbot supports both Korean and English. You can ask questions in either language, and the chatbot will respond in the same language you use. The system automatically detects your language preference.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  How do I get the best results?
                </h3>
                <p className="text-gray-700 leading-relaxed mb-3">
                  For optimal recommendations:
                </p>
                <ul className="space-y-2 text-gray-700 ml-6">
                  <li>1. <strong>Be specific</strong> about the type of care you need</li>
                  <li>2. <strong>Mention location</strong> if you have a preferred area</li>
                  <li>3. <strong>State language needs</strong> upfront if you need English-speaking care</li>
                  <li>4. <strong>Ask follow-up questions</strong> to refine results</li>
                  <li>5. <strong>Provide context</strong> about any specific requirements</li>
                </ul>
              </div>
            </div>
          </section>

          {/* About Medical Facilities */}
          <section>
            <h2 className="text-2xl font-semibold text-blue-800 mb-4 border-b-2 border-blue-200 pb-2">
              About Medical Facilities
            </h2>
            
            <div className="space-y-6">
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  How many facilities are in your database?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Our database includes thousands of medical facilities across all districts of Seoul, including hospitals, clinics, specialized centers, and private practices. We continuously update our database to include new facilities and updated information.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Where does your facility information come from?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  We aggregate information from multiple sources including official health registries, public databases, and user review platforms like Naver Maps. This combination allows us to provide comprehensive information including official details and real patient experiences.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  How accurate is the "English-speaking" designation?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  English-speaking designations are based on available data from facility information and patient reviews. However, English proficiency can vary, and staffing may change. We recommend calling ahead to confirm English language availability if this is critical to your needs.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Can you make appointments for me?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  No, we do not make appointments or have any connection with the medical facilities. You'll need to contact facilities directly using the phone numbers and contact information provided in the search results to schedule appointments.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Do you guarantee the quality of recommended facilities?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  No. We provide information to help you make informed decisions, but we do not endorse, recommend, or verify the quality of care provided by any facility. The recommendations are based on matching your search criteria with available facility data. Always research facilities independently and consult with healthcare professionals when making medical decisions.
                </p>
              </div>
            </div>
          </section>

          {/* Technical Questions */}
          <section>
            <h2 className="text-2xl font-semibold text-blue-800 mb-4 border-b-2 border-blue-200 pb-2">
              Technical Questions
            </h2>
            
            <div className="space-y-6">
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Do you store my medical information?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  No. We do not collect, store, or have access to your personal medical information, health records, or medical history. The chatbot only processes your search queries to find relevant facilities. For more details, please see our <Link href="/privacy" className="text-blue-600 hover:underline">Privacy Policy</Link>.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Is my conversation with the chatbot private?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Your chat messages are processed to provide facility recommendations and may be logged to improve our service. However, we do not share identifiable information with medical facilities or third parties (except as required by law). See our <Link href="/privacy" className="text-blue-600 hover:underline">Privacy Policy</Link> for complete details.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Why do I see advertisements?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  This service is free and supported by contextual advertising. We display ads through trusted partners like Google AdSense. These ads help us maintain and improve the platform without charging users. We do not sell your personal information to advertisers.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  What browsers and devices are supported?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Seoul Medical Facility Finder works on all modern web browsers (Chrome, Firefox, Safari, Edge) and is optimized for both desktop and mobile devices. For the best experience, we recommend using an updated browser version.
                </p>
              </div>
            </div>
          </section>

          {/* Safety & Emergencies */}
          <section>
            <h2 className="text-2xl font-semibold text-red-800 mb-4 border-b-2 border-red-200 pb-2">
              Safety & Emergencies
            </h2>
            
            <div className="space-y-6">
              <div className="bg-red-50 p-6 rounded-lg border-l-4 border-red-500">
                <h3 className="text-lg font-semibold text-red-900 mb-2">
                  What should I do in a medical emergency?
                </h3>
                <p className="text-gray-700 leading-relaxed mb-3">
                  <strong className="text-red-700">DO NOT use this chatbot for medical emergencies.</strong> If you or someone else is experiencing a medical emergency:
                </p>
                <ul className="space-y-2 text-gray-700 ml-6">
                  <li>• <strong>Call 119</strong> – Emergency medical services (ambulance)</li>
                  <li>• <strong>Call 1339</strong> – 24/7 medical emergency hotline</li>
                  <li>• <strong>Go to the nearest emergency room</strong> immediately</li>
                </ul>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Can this chatbot diagnose medical conditions?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  <strong>No, absolutely not.</strong> This chatbot is a facility directory tool, not a medical diagnostic tool. It cannot and does not diagnose medical conditions, assess symptoms, or provide medical advice. Only licensed healthcare professionals can provide medical diagnoses. For more information, see our <Link href="/disclaimer" className="text-blue-600 hover:underline">Medical Disclaimer</Link>.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Should I follow the chatbot's recommendations without consulting a doctor?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  No. The chatbot helps you <em>find</em> medical facilities, but you should always consult with qualified healthcare professionals for medical advice and decisions. The recommendations are informational only and do not replace professional medical consultation.
                </p>
              </div>
            </div>
          </section>

          {/* Problems & Support */}
          <section>
            <h2 className="text-2xl font-semibold text-blue-800 mb-4 border-b-2 border-blue-200 pb-2">
              Problems & Support
            </h2>
            
            <div className="space-y-6">
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  The chatbot didn't find any results. Why?
                </h3>
                <p className="text-gray-700 leading-relaxed mb-3">
                  This could happen for several reasons:
                </p>
                <ul className="space-y-2 text-gray-700 ml-6">
                  <li>• Your search criteria may be too specific or narrow</li>
                  <li>• The specialty you're looking for may use different terminology</li>
                  <li>• There may be limited facilities with English-speaking staff in the specified area</li>
                  <li>• Try broadening your search (e.g., expand the location area or try alternative specialty names)</li>
                </ul>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  The facility information seems outdated. What should I do?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  While we strive to maintain current information, facilities may update their details without notification. Always call the facility directly to confirm hours, services, insurance acceptance, and other important information before visiting. If you notice incorrect information, you can provide feedback through our platform.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  How can I provide feedback or report issues?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  We welcome feedback to improve our service. You can provide feedback about the chatbot, report incorrect facility information, or suggest improvements through the feedback mechanisms available on our platform. Your input helps us serve the community better.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  I have a question that's not answered here. What should I do?
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  Feel free to ask the chatbot directly! You can also review our <Link href="/about" className="text-blue-600 hover:underline">About</Link>, <Link href="/how-it-works" className="text-blue-600 hover:underline">How It Works</Link>, <Link href="/disclaimer" className="text-blue-600 hover:underline">Disclaimer</Link>, and <Link href="/privacy" className="text-blue-600 hover:underline">Privacy Policy</Link> pages for more detailed information about the service.
                </p>
              </div>
            </div>
          </section>

          {/* Quick Reference */}
          <section className="bg-blue-50 p-6 rounded-lg border-2 border-blue-200">
            <h2 className="text-2xl font-semibold text-blue-900 mb-4">
              Quick Reference: Korea Emergency Numbers
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded-lg">
                <p className="text-lg font-bold text-red-600">119</p>
                <p className="text-gray-700">Emergency Services (Fire, Ambulance, Rescue)</p>
              </div>
              <div className="bg-white p-4 rounded-lg">
                <p className="text-lg font-bold text-red-600">1339</p>
                <p className="text-gray-700">Emergency Medical Information Center (24/7)</p>
              </div>
              <div className="bg-white p-4 rounded-lg">
                <p className="text-lg font-bold text-blue-600">1345</p>
                <p className="text-gray-700">Foreign Language Interpretation (Immigration)</p>
              </div>
              <div className="bg-white p-4 rounded-lg">
                <p className="text-lg font-bold text-blue-600">1330</p>
                <p className="text-gray-700">Korea Travel Hotline (Tourism assistance)</p>
              </div>
            </div>
          </section>
        </div>

        <div className="mt-8 pt-6 border-t border-gray-200 text-center">
          <p className="text-sm text-gray-600">
            Still have questions? <Link href="/" className="text-blue-600 hover:underline font-semibold">Ask the chatbot directly</Link> or review our other information pages.
          </p>
        </div>
      </div>
    </div>
  );
}
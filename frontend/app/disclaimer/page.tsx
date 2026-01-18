import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Medical Disclaimer - Seoul Medical Facility Finder",
  description: "Important medical disclaimer and limitations of our facility finder service",
  icons: {
    icon: "/img/logo.ico",
  },
};

export default function Disclaimer() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-xl p-8 md:p-12">
        <Link 
          href="/" 
          className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-6 transition-colors"
        >
          ← Back to Chat
        </Link>
        
        <div className="bg-red-50 border-2 border-red-300 rounded-lg p-6 mb-8">
          <h1 className="text-3xl font-bold text-red-900 mb-4">⚠️ Important Medical Disclaimer</h1>
          <p className="text-red-800 text-lg font-semibold">
            Please read this disclaimer carefully before using our service.
          </p>
        </div>
        
        <div className="prose prose-lg max-w-none">
          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Not a Substitute for Professional Medical Advice</h2>
            <div className="bg-amber-50 border-l-4 border-amber-500 p-6 mb-4">
              <p className="text-gray-700 leading-relaxed mb-3">
                <strong>The Seoul Medical Facility Finder chatbot is a directory and information service ONLY.</strong> It is designed to help you locate medical facilities in Seoul based on your search criteria.
              </p>
              <p className="text-gray-700 leading-relaxed">
                This service <strong>does NOT provide medical advice, diagnosis, or treatment recommendations</strong>. The chatbot cannot and does not replace the judgment and expertise of qualified healthcare professionals.
              </p>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">What This Service Does NOT Do</h2>
            <div className="space-y-3 text-gray-700">
              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-red-600 text-2xl mr-3">✗</span>
                <div>
                  <strong className="text-gray-900">No Medical Diagnosis</strong>
                  <p className="text-sm mt-1">The chatbot cannot diagnose medical conditions, assess symptoms, or determine the severity of health issues. Only licensed healthcare professionals can provide medical diagnoses.</p>
                </div>
              </div>

              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-red-600 text-2xl mr-3">✗</span>
                <div>
                  <strong className="text-gray-900">No Treatment Recommendations</strong>
                  <p className="text-sm mt-1">We do not recommend treatments, medications, procedures, or medical interventions. All treatment decisions should be made in consultation with qualified medical professionals.</p>
                </div>
              </div>

              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-red-600 text-2xl mr-3">✗</span>
                <div>
                  <strong className="text-gray-900">No Doctor-Patient Relationship</strong>
                  <p className="text-sm mt-1">Using this chatbot does NOT establish a doctor-patient relationship. The information provided is general in nature and not tailored to your specific medical condition or circumstances.</p>
                </div>
              </div>

              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-red-600 text-2xl mr-3">✗</span>
                <div>
                  <strong className="text-gray-900">No Emergency Services</strong>
                  <p className="text-sm mt-1">This service is not designed for medical emergencies. Do not use this chatbot if you are experiencing a medical emergency.</p>
                </div>
              </div>

              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-red-600 text-2xl mr-3">✗</span>
                <div>
                  <strong className="text-gray-900">No Endorsement of Facilities</strong>
                  <p className="text-sm mt-1">Listing a facility on our platform does not constitute an endorsement, recommendation, or verification of the quality of care provided. We are not responsible for the actions, services, or quality of care provided by any medical facility.</p>
                </div>
              </div>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">What You Should Do Instead</h2>
            <div className="bg-green-50 border-l-4 border-green-500 p-6">
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">✓</span>
                  <span><strong>Consult Healthcare Professionals:</strong> Always seek the advice of a qualified physician or other healthcare provider with any questions regarding a medical condition.</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">✓</span>
                  <span><strong>Never Disregard Professional Advice:</strong> Do not disregard professional medical advice or delay seeking it because of information provided by this chatbot.</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">✓</span>
                  <span><strong>Verify All Information:</strong> Contact medical facilities directly to confirm services, hours, insurance acceptance, and any other important details before visiting.</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">✓</span>
                  <span><strong>Use Your Judgment:</strong> Research facilities independently and make informed decisions based on your personal circumstances and needs.</span>
                </li>
              </ul>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Medical Emergencies</h2>
            <div className="bg-red-100 border-2 border-red-400 p-6 rounded-lg">
              <h3 className="text-xl font-bold text-red-900 mb-3">🚨 If You Are Experiencing a Medical Emergency:</h3>
              <div className="space-y-2 text-gray-900">
                <p className="font-semibold text-lg">In South Korea, call:</p>
                <ul className="ml-6 space-y-2">
                  <li><strong className="text-red-700">119</strong> – Emergency Medical Services (ambulance, fire, rescue)</li>
                  <li><strong className="text-red-700">1339</strong> – Medical Emergency Hotline (24/7 medical consultation and guidance)</li>
                </ul>
                <p className="mt-4 font-semibold">For immediate life-threatening situations:</p>
                <ul className="ml-6 space-y-1">
                  <li>• Go to the nearest emergency room (응급실)</li>
                  <li>• Do not wait or attempt to search for facilities online</li>
                  <li>• Call emergency services immediately</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Information Accuracy & Limitations</h2>
            <p className="text-gray-700 leading-relaxed mb-4">
              While we make reasonable efforts to maintain accurate and up-to-date information about medical facilities:
            </p>
            <ul className="space-y-2 text-gray-700 ml-4">
              <li>• <strong>Information may be outdated:</strong> Facilities may change services, hours, contact information, or cease operations without notice</li>
              <li>• <strong>No guarantee of accuracy:</strong> We cannot guarantee the accuracy, completeness, or timeliness of facility information</li>
              <li>• <strong>User-generated content:</strong> Reviews and summaries may reflect subjective opinions and individual experiences</li>
              <li>• <strong>Translation limitations:</strong> Translations between Korean and English may not capture all nuances</li>
              <li>• <strong>Language capabilities:</strong> "English-speaking" designations are based on available data and may not reflect current staffing</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Third-Party Content</h2>
            <p className="text-gray-700 leading-relaxed">
              Our platform aggregates information from various public sources including government registries and user review platforms. We are not responsible for the accuracy of third-party content and do not endorse or guarantee the information provided by external sources.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Limitation of Liability</h2>
            <div className="bg-gray-50 p-6 rounded-lg border border-gray-300">
              <p className="text-gray-700 leading-relaxed mb-3">
                <strong>By using this service, you acknowledge and agree that:</strong>
              </p>
              <ul className="space-y-2 text-gray-700 ml-4">
                <li>• Seoul Medical Facility Finder and its operators are not liable for any harm, injury, or adverse outcomes resulting from your use of this service</li>
                <li>• We are not responsible for the quality of care provided by any medical facility listed on our platform</li>
                <li>• You use this service entirely at your own risk</li>
                <li>• We make no warranties, express or implied, regarding the accuracy, reliability, or suitability of the information provided</li>
                <li>• Your sole remedy for dissatisfaction with the service is to stop using it</li>
              </ul>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Appropriate Use of This Service</h2>
            <div className="bg-blue-50 p-6 rounded-lg">
              <p className="text-gray-700 leading-relaxed mb-3">
                <strong>This service is appropriate for:</strong>
              </p>
              <ul className="space-y-2 text-gray-700 ml-4">
                <li>✓ Finding medical facilities in a specific area or specialty</li>
                <li>✓ Learning about facility locations, services, and basic information</li>
                <li>✓ Identifying English-speaking medical facilities</li>
                <li>✓ General orientation to Seoul's healthcare landscape</li>
              </ul>
              <p className="text-gray-700 leading-relaxed mt-4 mb-3">
                <strong>This service is NOT appropriate for:</strong>
              </p>
              <ul className="space-y-2 text-gray-700 ml-4">
                <li>✗ Seeking medical advice or diagnosis</li>
                <li>✗ Making urgent healthcare decisions</li>
                <li>✗ Replacing professional medical consultation</li>
                <li>✗ Emergency medical situations</li>
              </ul>
            </div>
          </section>

          <section className="bg-gray-100 p-6 rounded-lg">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Your Responsibility</h2>
            <p className="text-gray-700 leading-relaxed">
              By using Seoul Medical Facility Finder, you acknowledge that you have read, understood, and agree to this medical disclaimer. You accept full responsibility for any decisions you make regarding medical care based on information obtained through this service. You agree to consult with qualified healthcare professionals for all medical decisions and to independently verify any facility information before making healthcare appointments or decisions.
            </p>
          </section>

          <div className="mt-8 p-6 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg">
            <p className="text-gray-700 leading-relaxed">
              <strong>Last Updated:</strong> January 2026
            </p>
            <p className="text-gray-700 leading-relaxed mt-2">
              This disclaimer may be updated periodically. Continued use of the service after changes constitutes acceptance of the updated disclaimer.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
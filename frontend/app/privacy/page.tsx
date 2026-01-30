import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy Policy - Seoul Medical Facility Finder",
  description: "How we collect, use, and protect your data when using our medical facility finder service",
  icons: {
    icon: "/img/convertico-logo_v3.ico",
  },
};

export default function Privacy() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-xl p-8 md:p-12">
        <Link 
          href="/" 
          className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-6 transition-colors"
        >
          ← Back to Chat
        </Link>
        
        <h1 className="text-4xl font-bold text-gray-900 mb-6">Privacy Policy</h1>
        
        <div className="prose prose-lg max-w-none">
          <p className="text-gray-600 mb-8">
            <strong>Effective Date:</strong> January 2026<br/>
            <strong>Last Updated:</strong> January 2026
          </p>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Introduction</h2>
            <p className="text-gray-700 leading-relaxed">
              Seoul Medical Facility Finder ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you use our medical facility finder chatbot service (the "Service").
            </p>
            <p className="text-gray-700 leading-relaxed mt-3">
              By using our Service, you consent to the data practices described in this policy. If you do not agree with this policy, please do not use our Service.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Information We Collect</h2>
            
            <div className="bg-blue-50 p-6 rounded-lg mb-4">
              <h3 className="text-xl font-semibold text-blue-900 mb-3">1. Information You Provide</h3>
              <p className="text-gray-700 mb-3">When you interact with our chatbot, we may collect:</p>
              <ul className="space-y-2 text-gray-700 ml-4">
                <li>• <strong>Chat messages:</strong> The text you enter in conversation with the chatbot</li>
                <li>• <strong>Search queries:</strong> Information about medical specialties, locations, and preferences you express</li>
                <li>• <strong>Feedback:</strong> Any feedback or ratings you provide about our service or facility recommendations</li>
              </ul>
              <p className="text-gray-700 mt-3 font-semibold">
                Important: We do NOT collect or store personal medical information, health records, or specific details about your medical conditions or symptoms.
              </p>
            </div>

            <div className="bg-green-50 p-6 rounded-lg mb-4">
              <h3 className="text-xl font-semibold text-green-900 mb-3">2. Automatically Collected Information</h3>
              <p className="text-gray-700 mb-3">When you access our Service, we automatically collect certain technical information:</p>
              <ul className="space-y-2 text-gray-700 ml-4">
                <li>• <strong>Device information:</strong> Browser type, operating system, device type</li>
                <li>• <strong>Usage data:</strong> Pages visited, time spent on the Service, interaction patterns</li>
                <li>• <strong>IP address:</strong> Your internet protocol address (which may indicate general location)</li>
                <li>• <strong>Cookies and tracking technologies:</strong> Data collected through cookies and similar technologies</li>
              </ul>
            </div>

            <div className="bg-purple-50 p-6 rounded-lg">
              <h3 className="text-xl font-semibold text-purple-900 mb-3">3. Information We Do NOT Collect</h3>
              <ul className="space-y-2 text-gray-700 ml-4">
                <li>• Personal health records or medical history</li>
                <li>• Social security numbers or government ID numbers</li>
                <li>• Payment or financial information (our service is free)</li>
                <li>• Precise real-time location tracking</li>
                <li>• Personal contact information unless voluntarily provided for support purposes</li>
              </ul>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">How We Use Your Information</h2>
            <p className="text-gray-700 leading-relaxed mb-3">
              We use the information we collect for the following purposes:
            </p>
            <div className="space-y-3">
              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-blue-600 font-bold text-xl mr-3">1.</span>
                <div>
                  <strong className="text-gray-900">To Provide the Service</strong>
                  <p className="text-sm text-gray-700 mt-1">Process your queries and provide relevant medical facility recommendations based on your search criteria.</p>
                </div>
              </div>

              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-blue-600 font-bold text-xl mr-3">2.</span>
                <div>
                  <strong className="text-gray-900">To Improve the Service</strong>
                  <p className="text-sm text-gray-700 mt-1">Analyze usage patterns to improve chatbot responses, ranking algorithms, and user experience.</p>
                </div>
              </div>

              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-blue-600 font-bold text-xl mr-3">3.</span>
                <div>
                  <strong className="text-gray-900">To Maintain Security</strong>
                  <p className="text-sm text-gray-700 mt-1">Monitor for fraudulent activity, abuse, and technical issues to maintain service integrity.</p>
                </div>
              </div>

              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-blue-600 font-bold text-xl mr-3">4.</span>
                <div>
                  <strong className="text-gray-900">To Comply with Legal Obligations</strong>
                  <p className="text-sm text-gray-700 mt-1">Respond to legal requests and prevent illegal activities as required by law.</p>
                </div>
              </div>

              <div className="flex items-start p-4 bg-gray-50 rounded-lg">
                <span className="text-blue-600 font-bold text-xl mr-3">5.</span>
                <div>
                  <strong className="text-gray-900">For Analytics and Advertising</strong>
                  <p className="text-sm text-gray-700 mt-1">Generate aggregate statistics about Service usage and display contextually relevant advertisements.</p>
                </div>
              </div>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Cookies and Tracking Technologies</h2>
            <p className="text-gray-700 leading-relaxed mb-3">
              We use cookies and similar tracking technologies to enhance your experience:
            </p>
            <ul className="space-y-2 text-gray-700 ml-4 mb-4">
              <li>• <strong>Essential cookies:</strong> Required for the Service to function properly</li>
              <li>• <strong>Analytics cookies:</strong> Help us understand how users interact with the Service</li>
              <li>• <strong>Advertising cookies:</strong> Used by advertising partners to display relevant ads</li>
            </ul>
            <p className="text-gray-700 leading-relaxed">
              You can control cookie preferences through your browser settings. However, disabling certain cookies may limit your ability to use some features of the Service.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Third-Party Services and Advertising</h2>
            
            <div className="bg-amber-50 border-l-4 border-amber-500 p-6 mb-4">
              <h3 className="text-lg font-semibold text-amber-900 mb-3">Advertising Partners</h3>
              <p className="text-gray-700 leading-relaxed mb-3">
                We may use third-party advertising services (such as Google AdSense) to display advertisements on our Service. These advertising partners may use cookies and similar technologies to:
              </p>
              <ul className="space-y-2 text-gray-700 ml-4">
                <li>• Collect information about your visits to our Service and other websites</li>
                <li>• Display ads based on your interests and browsing patterns</li>
                <li>• Measure ad effectiveness and optimize ad delivery</li>
              </ul>
              <p className="text-gray-700 leading-relaxed mt-3">
                We do not control these third-party technologies or the information they collect. Please review the privacy policies of our advertising partners for more information about their data practices.
              </p>
            </div>

            <div className="bg-blue-50 p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">Analytics Services</h3>
              <p className="text-gray-700 leading-relaxed">
                We may use analytics services (such as Google Analytics) to collect and analyze information about Service usage. These services help us understand user behavior and improve our Service.
              </p>
            </div>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">How We Share Your Information</h2>
            <p className="text-gray-700 leading-relaxed mb-3">
              We do not sell your personal information. We may share information in the following circumstances:
            </p>
            <ul className="space-y-2 text-gray-700 ml-4">
              <li>• <strong>Service providers:</strong> With trusted third parties who help us operate the Service (hosting, analytics, advertising)</li>
              <li>• <strong>Aggregated data:</strong> We may share anonymized, aggregated data that cannot identify individual users</li>
              <li>• <strong>Legal requirements:</strong> When required by law, court order, or government request</li>
              <li>• <strong>Protection of rights:</strong> To protect our rights, safety, or property, or that of our users or others</li>
              <li>• <strong>Business transfers:</strong> In connection with a merger, acquisition, or sale of assets</li>
            </ul>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Data Security</h2>
            <p className="text-gray-700 leading-relaxed mb-3">
              We implement reasonable technical and organizational measures to protect your information:
            </p>
            <ul className="space-y-2 text-gray-700 ml-4">
              <li>• Encryption of data in transit using HTTPS/SSL</li>
              <li>• Regular security assessments and updates</li>
              <li>• Limited access to personal information by authorized personnel only</li>
              <li>• Monitoring for security incidents and unauthorized access</li>
            </ul>
            <p className="text-gray-700 leading-relaxed mt-3">
              However, no method of transmission over the internet or electronic storage is 100% secure. While we strive to protect your information, we cannot guarantee absolute security.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Data Retention</h2>
            <p className="text-gray-700 leading-relaxed">
              We retain your information only as long as necessary to provide the Service and fulfill the purposes described in this policy. Chat logs and usage data may be retained for a reasonable period to improve our Service and comply with legal obligations. We periodically review and delete data that is no longer needed.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Your Rights and Choices</h2>
            <p className="text-gray-700 leading-relaxed mb-3">
              Depending on your location, you may have certain rights regarding your personal information:
            </p>
            <div className="bg-green-50 p-6 rounded-lg">
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">•</span>
                  <span><strong>Access:</strong> Request access to the personal information we hold about you</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">•</span>
                  <span><strong>Correction:</strong> Request correction of inaccurate information</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">•</span>
                  <span><strong>Deletion:</strong> Request deletion of your personal information (subject to legal obligations)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">•</span>
                  <span><strong>Opt-out:</strong> Opt out of certain data collection (e.g., cookies, advertising)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-600 font-bold mr-3">•</span>
                  <span><strong>Objection:</strong> Object to certain types of processing</span>
                </li>
              </ul>
            </div>
            <p className="text-gray-700 leading-relaxed mt-4">
              To exercise these rights, please contact us using the information provided at the end of this policy. We will respond to your request in accordance with applicable laws.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">International Users</h2>
            <p className="text-gray-700 leading-relaxed">
              Our Service is hosted and operated from servers that may be located in various jurisdictions. If you access the Service from outside the hosting jurisdiction, please be aware that your information may be transferred to, stored, and processed in countries with different privacy laws than your country of residence. By using the Service, you consent to such transfer and processing.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Children's Privacy</h2>
            <p className="text-gray-700 leading-relaxed">
              Our Service is not directed to individuals under the age of 18. We do not knowingly collect personal information from children. If you believe we have inadvertently collected information from a child, please contact us immediately, and we will take steps to delete such information.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Changes to This Privacy Policy</h2>
            <p className="text-gray-700 leading-relaxed">
              We may update this Privacy Policy from time to time to reflect changes in our practices or legal requirements. We will notify you of significant changes by posting the updated policy on this page with a new "Last Updated" date. Your continued use of the Service after changes are posted constitutes acceptance of the updated policy.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Third-Party Links</h2>
            <p className="text-gray-700 leading-relaxed">
              Our Service may contain links to medical facility websites and other third-party sites. We are not responsible for the privacy practices of these external sites. We encourage you to review the privacy policies of any third-party sites you visit.
            </p>
          </section>

          <section className="bg-gray-100 p-6 rounded-lg">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Contact Us</h2>
            <p className="text-gray-700 leading-relaxed mb-3">
              If you have questions, concerns, or requests regarding this Privacy Policy or our data practices, please contact us:
            </p>
            <div className="bg-white p-4 rounded border border-gray-300">
              <p className="text-gray-700">
                <strong>Seoul Medical Facility Finder</strong><br/>
                Privacy Inquiries<br/>
                <em>Contact information can be provided through the feedback mechanism on our platform</em>
              </p>
            </div>
          </section>

          <div className="mt-8 p-6 bg-blue-50 border-l-4 border-blue-500 rounded-r-lg">
            <p className="text-gray-700 text-sm">
              <strong>Note:</strong> This privacy policy is designed to be compliant with general privacy principles. Users in specific jurisdictions (such as the EU/GDPR or California/CCPA) may have additional rights under local law.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
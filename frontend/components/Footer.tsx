import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        {/* Emergency Banner */}
        <div className="bg-red-900 text-white p-4 rounded-lg mb-6 text-center">
          <p className="font-bold text-base mb-1">🚨 Medical Emergency?</p>
          <p className="text-sm">
            Call <strong className="text-lg">119</strong> (Emergency Services) or <strong className="text-lg">1339</strong> (Medical Hotline)
          </p>
        </div>

        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-6">
          {/* About Section */}
          <div>
            <h3 className="font-bold text-lg mb-3">Seoul Medical Finder</h3>
            <p className="text-sm text-gray-300 leading-relaxed">
              Find quality medical facilities across Seoul with AI-powered recommendations.
            </p>
          </div>

          {/* Information Links */}
          <div>
            <h4 className="font-semibold text-base mb-3">Information</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/about" className="text-gray-300 hover:text-white transition-colors">
                  About Us
                </Link>
              </li>
              <li>
                <Link href="/how-it-works" className="text-gray-300 hover:text-white transition-colors">
                  How It Works
                </Link>
              </li>
              <li>
                <Link href="/faq" className="text-gray-300 hover:text-white transition-colors">
                  FAQ
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal Links */}
          <div>
            <h4 className="font-semibold text-base mb-3">Legal</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/disclaimer" className="text-gray-300 hover:text-white transition-colors">
                  ⚠️ Medical Disclaimer
                </Link>
              </li>
              <li>
                <Link href="/privacy" className="text-gray-300 hover:text-white transition-colors">
                  Privacy Policy
                </Link>
              </li>
            </ul>
          </div>

          {/* Emergency Info */}
          <div>
            <h4 className="font-semibold text-base mb-3">Emergency Numbers</h4>
            <ul className="space-y-2 text-sm text-gray-300">
              <li><strong className="text-white">119</strong> - Emergency Services</li>
              <li><strong className="text-white">1339</strong> - Medical Hotline (24/7)</li>
              <li><strong className="text-white">1345</strong> - Immigration Help</li>
              <li><strong className="text-white">1330</strong> - Korea Travel Hotline</li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-700 pt-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-3 text-sm text-gray-400">
            <p>
              © {new Date().getFullYear()} Seoul Medical Facility Finder. All rights reserved.
            </p>
            <p>
              For informational purposes only. Not medical advice.
            </p>
          </div>
          
          {/* AdSense Disclosure */}
          <div className="text-center mt-4 text-xs text-gray-500">
            <p>This site is supported by advertising. We do not sell personal information.</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
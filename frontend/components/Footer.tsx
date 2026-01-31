'use client';

import Link from 'next/link';

export default function Footer() {
  const resetCookieConsent = () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('cookieConsent');
      window.location.reload();
    }
  };

  return (
    <footer className="bg-gradient-to-r from-blue-50 to-blue-100 text-gray-800 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        {/* Emergency Banner - Blue Theme with Green Accent Numbers */}
        <div className="bg-gradient-to-r from-blue-100 to-blue-200 border-2 border-blue-400 text-blue-900 p-4 rounded-lg mb-6 text-center shadow-md">
          <p className="font-bold text-base mb-1">🚨 Medical Emergency?</p>
          <p className="text-sm">
            Call <strong className="text-lg text-green-700">119</strong> (Emergency Services) or{' '}
            <strong className="text-lg text-green-700">1339</strong> (Medical Hotline)
          </p>
        </div>

        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-6">
          {/* About Section */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-700">Seoul Medical Finder</h3>
            <p className="text-sm text-gray-700 leading-relaxed mb-4">
              Find quality medical facilities across Seoul with AI-powered
              recommendations.
            </p>
            <div className="text-sm">
              <h4 className="font-semibold text-blue-700 mb-1">Contact Us</h4>
              <a 
                href="mailto:seouldoc.io@gmail.com" 
                className="text-blue-600 hover:text-blue-800 transition-colors"
              >
                seouldoc.io@gmail.com
              </a>
            </div>
          </div>

          {/* Information Links */}
          <div>
            <h4 className="font-semibold text-base mb-3 text-blue-700">Information</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link
                  href="/about"
                  className="text-gray-700 hover:text-blue-600 transition-colors"
                >
                  About Us
                </Link>
              </li>
              <li>
                <Link
                  href="/how-it-works"
                  className="text-gray-700 hover:text-blue-600 transition-colors"
                >
                  How It Works
                </Link>
              </li>
              <li>
                <Link
                  href="/faq"
                  className="text-gray-700 hover:text-blue-600 transition-colors"
                >
                  FAQ
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal Links */}
          <div>
            <h4 className="font-semibold text-base mb-3 text-blue-700">Legal</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link
                  href="/disclaimer"
                  className="text-gray-700 hover:text-blue-600 transition-colors"
                >
                  ⚠️ Medical Disclaimer
                </Link>
              </li>
              <li>
                <Link
                  href="/privacy"
                  className="text-gray-700 hover:text-blue-600 transition-colors"
                >
                  Privacy Policy
                </Link>
              </li>
              <li>
                <button
                  onClick={resetCookieConsent}
                  className="text-gray-700 hover:text-blue-600 transition-colors text-left"
                >
                  🍪 Cookie Settings
                </button>
              </li>
            </ul>
          </div>

          {/* Emergency Info */}
          <div>
            <h4 className="font-semibold text-base mb-3 text-blue-700">Emergency Numbers</h4>
            <ul className="space-y-2 text-sm text-gray-700">
              <li>
                <strong className="text-blue-700">119</strong> - Emergency Services
              </li>
              <li>
                <strong className="text-blue-700">1339</strong> - Medical Hotline
                (24/7)
              </li>
              <li>
                <strong className="text-blue-700">1345</strong> - Immigration Help
              </li>
              <li>
                <strong className="text-blue-700">1330</strong> - Korea Travel
                Hotline
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-blue-200 pt-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-3 text-sm text-gray-600">
            <p>
              © {new Date().getFullYear()} Seoul Medical Facility Finder. All
              rights reserved.
            </p>
            <p>For informational purposes only. Not medical advice.</p>
          </div>

          {/* AdSense Disclosure */}
          <div className="text-center mt-4 text-xs text-gray-500">
            <p>
              This site is supported by donations. We do not sell personal
              information.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
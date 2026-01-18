import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-6 mt-auto">
      <div className="container mx-auto px-4">
        {/* Emergency Banner - More Compact */}
        <div className="bg-red-900 text-white p-3 rounded-lg mb-4 text-center">
          <p className="font-semibold text-sm">🚨 Medical Emergency?</p>
          <p className="text-xs mt-1">
            Call <strong className="text-base">119</strong> (Emergency Services) or <strong className="text-base">1339</strong> (Medical Hotline)
          </p>
        </div>

        {/* Main Footer Content - Reduced Spacing */}
        <div className="grid md:grid-cols-4 gap-6 mb-4">
          {/* About Section */}
          <div>
            <h3 className="font-bold text-base mb-2">Seoul Medical Finder</h3>
            <p className="text-xs text-gray-300">
              Find quality medical facilities across Seoul with AI-powered recommendations.
            </p>
          </div>

          {/* Information Links */}
          <div>
            <h4 className="font-semibold text-sm mb-2">Information</h4>
            <ul className="space-y-1 text-xs">
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
            <h4 className="font-semibold text-sm mb-2">Legal</h4>
            <ul className="space-y-1 text-xs">
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
            <h4 className="font-semibold text-sm mb-2">Emergency Numbers</h4>
            <ul className="space-y-1 text-xs text-gray-300">
              <li><strong>119</strong> - Emergency Services</li>
              <li><strong>1339</strong> - Medical Hotline (24/7)</li>
              <li><strong>1345</strong> - Immigration Help</li>
              <li><strong>1330</strong> - Korea Travel Hotline</li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar - More Compact */}
        <div className="border-t border-gray-700 pt-4">
          <div className="flex flex-col md:flex-row justify-between items-center text-xs text-gray-400">
            <p>
              © {new Date().getFullYear()} Seoul Medical Facility Finder. All rights reserved.
            </p>
            <p className="mt-1 md:mt-0">
              For informational purposes only. Not medical advice.
            </p>
          </div>
          
          {/* AdSense Disclosure */}
          <div className="text-center mt-2 text-xs text-gray-500">
            <p>This site is supported by advertising. We do not sell personal information.</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
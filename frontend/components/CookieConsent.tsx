// components/CookieConsent.tsx
'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface ConsentSettings {
  necessary: boolean;
  analytics: boolean;
  advertising: boolean;
}

export default function CookieConsent() {
  const [showBanner, setShowBanner] = useState(false);
  const [consent, setConsent] = useState<ConsentSettings>({
    necessary: true,
    analytics: false,
    advertising: false,
  });
  const [showPreferences, setShowPreferences] = useState(false);

  useEffect(() => {
    const savedConsent = localStorage.getItem('cookieConsent');
    if (!savedConsent) {
      setShowBanner(true);
    } else {
      try {
        const parsed = JSON.parse(savedConsent);
        setConsent(parsed);
        loadScripts(parsed);
      } catch (error) {
        console.error('Error parsing consent:', error);
        setShowBanner(true);
      }
    }
  }, []);

  const loadScripts = (consentData: ConsentSettings) => {
    if (typeof window === 'undefined') return;

    // Load analytics if consented
    if (consentData.analytics) {
      const gtag = (window as any).gtag;
      if (gtag) {
        gtag('consent', 'update', {
          analytics_storage: 'granted',
        });
      }
    }

    // Load advertising if consented
    if (consentData.advertising) {
      const gtag = (window as any).gtag;
      if (gtag) {
        gtag('consent', 'update', {
          ad_storage: 'granted',
          ad_user_data: 'granted',
          ad_personalization: 'granted',
        });
      }
    }
  };

  const handleAcceptAll = () => {
    const newConsent = {
      necessary: true,
      analytics: true,
      advertising: true,
    };
    saveConsent(newConsent);
  };

  const handleRejectAll = () => {
    const newConsent = {
      necessary: true,
      analytics: false,
      advertising: false,
    };
    saveConsent(newConsent);
  };

  const handleSavePreferences = () => {
    saveConsent(consent);
  };

  const saveConsent = (consentData: ConsentSettings) => {
    localStorage.setItem('cookieConsent', JSON.stringify(consentData));
    setConsent(consentData);
    setShowBanner(false);
    setShowPreferences(false);
    loadScripts(consentData);

    // Reload page to apply changes (ensures ads load correctly)
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  };

  // Don't render anything until we check localStorage
  if (!showBanner) return null;

  return (
    <>
      {/* Main Banner */}
      {!showPreferences && (
        <div className="fixed bottom-0 left-0 right-0 z-50 bg-white border-t border-gray-200 shadow-lg">
          <div className="max-w-7xl mx-auto p-4 sm:p-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  🍪 Cookie Preferences
                </h3>
                <p className="text-sm text-gray-600">
                  We use cookies to enhance your browsing experience, serve
                  personalized ads or content, and analyze our traffic. By
                  clicking &quot;Accept All&quot;, you consent to our use of cookies.{' '}
                  <Link
                    href="/privacy-policy"
                    className="text-blue-600 hover:underline"
                  >
                    Read our Privacy Policy
                  </Link>
                  .
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
                <button
                  onClick={() => setShowPreferences(true)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                >
                  Customize
                </button>
                <button
                  onClick={handleRejectAll}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                >
                  Reject All
                </button>
                <button
                  onClick={handleAcceptAll}
                  className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
                >
                  Accept All
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Preferences Modal */}
      {showPreferences && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Cookie Preferences
              </h2>
              <p className="text-sm text-gray-600 mb-6">
                We use cookies and similar technologies to help personalize
                content, tailor and measure ads, and provide a better
                experience. Select your cookie preferences below.
              </p>

              {/* Necessary Cookies */}
              <div className="mb-6 pb-6 border-b border-gray-200">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900 mb-1">
                      Necessary Cookies
                    </h3>
                    <p className="text-sm text-gray-600">
                      These cookies are essential for the website to function
                      properly. They cannot be disabled.
                    </p>
                  </div>
                  <div className="ml-4">
                    <input
                      type="checkbox"
                      checked={true}
                      disabled
                      className="h-5 w-5 text-blue-600 rounded cursor-not-allowed"
                    />
                  </div>
                </div>
              </div>

              {/* Analytics Cookies */}
              <div className="mb-6 pb-6 border-b border-gray-200">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900 mb-1">
                      Analytics Cookies
                    </h3>
                    <p className="text-sm text-gray-600">
                      These cookies help us understand how visitors interact
                      with our website by collecting and reporting information
                      anonymously.
                    </p>
                  </div>
                  <div className="ml-4">
                    <input
                      type="checkbox"
                      checked={consent.analytics}
                      onChange={(e) =>
                        setConsent({ ...consent, analytics: e.target.checked })
                      }
                      className="h-5 w-5 text-blue-600 rounded cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              {/* Advertising Cookies */}
              <div className="mb-6">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900 mb-1">
                      Advertising Cookies
                    </h3>
                    <p className="text-sm text-gray-600">
                      These cookies are used to make advertising messages more
                      relevant to you and your interests. They also perform
                      functions like preventing the same ad from continuously
                      reappearing.
                    </p>
                  </div>
                  <div className="ml-4">
                    <input
                      type="checkbox"
                      checked={consent.advertising}
                      onChange={(e) =>
                        setConsent({
                          ...consent,
                          advertising: e.target.checked,
                        })
                      }
                      className="h-5 w-5 text-blue-600 rounded cursor-pointer"
                    />
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-3 mt-6">
                <button
                  onClick={() => setShowPreferences(false)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSavePreferences}
                  className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors flex-1 sm:flex-initial"
                >
                  Save Preferences
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
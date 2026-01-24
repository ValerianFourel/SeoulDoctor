// components/GoogleConsent.tsx
'use client';

import Script from 'next/script';

export default function GoogleConsent() {
  return (
    <Script
      id="google-consent-mode"
      strategy="afterInteractive"
      dangerouslySetInnerHTML={{
        __html: `
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          
          gtag('consent', 'default', {
            'ad_storage': 'denied',
            'ad_user_data': 'denied',
            'ad_personalization': 'denied',
            'analytics_storage': 'denied'
          });
        `,
      }}
    />
  );
}
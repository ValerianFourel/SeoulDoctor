'use client';
import { useState } from 'react';
import Link from 'next/link';


export default function ContactPage() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');

 const onSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
  // 1. THIS IS THE MOST IMPORTANT LINE
  event.preventDefault(); 
  
  setIsSubmitting(true);
  setStatus('idle');

  // 2. We use FormData to gather the info
  const formData = new FormData(event.currentTarget);
  formData.append("access_key", "910f6abd-3031-4cae-b81f-0fb2598ec199");

  try {
    const response = await fetch("https://api.web3forms.com/submit", {
      method: "POST", // 3. Ensure this is POST
      body: formData,
      // 4. This header ensures the browser knows we are talking to an API
      headers: {
        Accept: "application/json",
      },
    });

    const data = await response.json();
    if (data.success) {
      setStatus('success');
      (event.target as HTMLFormElement).reset();
    } else {
      setStatus('error');
    }
  } catch (error) {
    setStatus('error');
  } finally {
    setIsSubmitting(false);
  }
};

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-xl p-8 md:p-12">
        <Link 
          href="/" 
          className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-6 transition-colors"
        >
          ← Back to Home
        </Link>

        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Contact Us</h1>
          <p className="text-gray-600">
            Have questions? Send us a message and we'll get back to you at{' '}
            <span className="font-medium text-blue-600">seouldoc.io@gmail.com</span>.
          </p>
        </div>

        {status === 'success' ? (
          <div className="bg-green-50 border border-green-200 text-green-800 p-8 rounded-lg text-center animate-in fade-in zoom-in-95">
            <div className="text-4xl mb-4">✉️</div>
            <h2 className="text-xl font-bold mb-2">Message Received!</h2>
            <p className="mb-6">Thank you. We'll review your inquiry and respond shortly.</p>
            <button 
              onClick={() => setStatus('idle')}
              className="text-sm font-semibold text-blue-600 hover:text-blue-800 underline transition-colors"
            >
              Send another message
            </button>
          </div>
        ) : (
          <form onSubmit={onSubmit} className="space-y-5">
            {/* Honeypot Spam Protection */}
            <input type="checkbox" name="botcheck" className="hidden" style={{ display: 'none' }} />

            <div>
              <label htmlFor="name" className="block text-sm font-semibold text-gray-700 mb-1">
                Full Name
              </label>
              <input
                type="text"
                name="name"
                id="name"
                required
                placeholder="Your Name"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-semibold text-gray-700 mb-1">
                Email Address
              </label>
              <input
                type="email"
                name="email"
                id="email"
                required
                placeholder="email@example.com"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              />
            </div>

            <div>
              <label htmlFor="message" className="block text-sm font-semibold text-gray-700 mb-1">
                Message
              </label>
              <textarea
                name="message"
                id="message"
                required
                rows={5}
                placeholder="How can we help you?"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all resize-none"
              ></textarea>
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className={`w-full py-4 rounded-lg font-bold text-white transition-all shadow-md ${
                isSubmitting 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700 active:scale-[0.98]'
              }`}
            >
              {isSubmitting ? 'Sending...' : 'Send Message'}
            </button>

            {status === 'error' && (
              <div className="p-3 bg-red-50 text-red-600 text-sm rounded-lg text-center border border-red-100">
                ⚠️ Something went wrong. Please try again or email us directly.
              </div>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
// src/app/page.tsx
'use client';

import dynamic from 'next/dynamic';

// Dynamically import TradeLens component with no SSR
const TradeLens = dynamic(() => import('@/components/TradeLens'), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
        <p className="mt-4 text-xl font-semibold text-gray-700">Loading TradeLens...</p>
      </div>
    </div>
  ),
});

export default function Home() {
  return <TradeLens />;
}
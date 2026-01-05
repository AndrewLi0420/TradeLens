// src/app/page.tsx
'use client';

import dynamic from 'next/dynamic';

// Dynamically import TradeLens component with no SSR
const TradeLens = dynamic(() => import('@/components/TradeLens'), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-tl-space flex items-center justify-center">
      <div className="text-center">
        <div className="relative w-16 h-16 mx-auto mb-4">
          <div className="absolute inset-0 rounded-full border-2 border-tl-border animate-ping opacity-20" />
          <div className="absolute inset-0 rounded-full border-2 border-tl-smart animate-spin" style={{ borderTopColor: 'transparent' }} />
          <div className="absolute inset-2 rounded-full bg-tl-surface-elevated flex items-center justify-center">
            <span className="font-mono text-tl-smart text-lg font-bold">TL</span>
          </div>
        </div>
        <p className="text-data-sm font-medium text-tl-text-secondary">Initializing TradeLens...</p>
        <div className="mt-3 w-32 mx-auto h-1 bg-tl-surface rounded-full overflow-hidden">
          <div className="h-full w-1/2 bg-tl-smart rounded-full animate-pulse" />
        </div>
      </div>
    </div>
  ),
});

export default function Home() {
  return <TradeLens />;
}

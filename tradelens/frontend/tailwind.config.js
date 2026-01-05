/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // TradeLens Core Palette
        'tl-space': '#1B264F',
        'tl-space-light': '#243366',
        'tl-space-dark': '#141d3d',
        'tl-french': '#274690',
        'tl-french-light': '#3157a8',
        'tl-french-dark': '#1e3873',
        'tl-smart': '#576CA8',
        'tl-smart-light': '#6b7fba',
        'tl-graphite': '#302B27',
        'tl-graphite-light': '#3d3733',
        'tl-smoke': '#F5F3F5',
        'tl-smoke-dim': '#E8E5E8',
        
        // Semantic Colors
        'tl-positive': '#00C896',
        'tl-positive-muted': 'rgba(0, 200, 150, 0.125)',
        'tl-negative': '#FF4757',
        'tl-negative-muted': 'rgba(255, 71, 87, 0.125)',
        'tl-neutral': '#576CA8',
        'tl-warning': '#FFB830',
        'tl-warning-muted': 'rgba(255, 184, 48, 0.125)',
        
        // Surface Colors
        'tl-surface': '#1E2235',
        'tl-surface-elevated': '#252A40',
        'tl-surface-overlay': '#2D3350',
        'tl-border': '#3A4060',
        'tl-border-subtle': '#2E3448',
        
        // Legacy support
        border: "hsl(var(--border))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
      },
      textColor: {
        // Text Colors (for @apply directives)
        'tl-primary': '#F5F3F5',
        'tl-secondary': '#A8AEBF',
        'tl-tertiary': '#6B7280',
        'tl-muted': '#4B5563',
      },
      fontFamily: {
        'sans': ['var(--font-sans)', 'Inter', 'SF Pro Display', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        'mono': ['var(--font-mono)', 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Monaco', 'Consolas', 'monospace'],
      },
      fontSize: {
        'xxs': ['0.65rem', { lineHeight: '0.875rem' }],
        'data-xs': ['0.7rem', { lineHeight: '1rem', letterSpacing: '0.02em' }],
        'data-sm': ['0.8rem', { lineHeight: '1.125rem', letterSpacing: '0.01em' }],
        'data-base': ['0.875rem', { lineHeight: '1.25rem', letterSpacing: '0.01em' }],
        'data-lg': ['1rem', { lineHeight: '1.375rem' }],
        'data-xl': ['1.125rem', { lineHeight: '1.5rem' }],
        'data-2xl': ['1.375rem', { lineHeight: '1.75rem' }],
        'data-3xl': ['1.75rem', { lineHeight: '2rem' }],
      },
      spacing: {
        '0.5': '0.125rem',
        '1.5': '0.375rem',
        '2.5': '0.625rem',
        '3.5': '0.875rem',
        '4.5': '1.125rem',
        '13': '3.25rem',
        '15': '3.75rem',
        '18': '4.5rem',
        '22': '5.5rem',
      },
      borderRadius: {
        'sm': '0.25rem',
        'DEFAULT': '0.375rem',
        'md': '0.5rem',
        'lg': '0.625rem',
        'xl': '0.75rem',
      },
      boxShadow: {
        'tl-sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
        'tl': '0 2px 4px rgba(0, 0, 0, 0.25), 0 1px 2px rgba(0, 0, 0, 0.35)',
        'tl-md': '0 4px 8px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2)',
        'tl-lg': '0 8px 16px rgba(0, 0, 0, 0.35), 0 4px 8px rgba(0, 0, 0, 0.25)',
        'tl-glow-positive': '0 0 12px rgba(0, 200, 150, 0.3)',
        'tl-glow-negative': '0 0 12px rgba(255, 71, 87, 0.3)',
        'tl-inset': 'inset 0 1px 2px rgba(0, 0, 0, 0.2)',
      },
      backgroundImage: {
        'tl-gradient': 'linear-gradient(135deg, #1B264F 0%, #274690 50%, #1B264F 100%)',
        'tl-gradient-surface': 'linear-gradient(180deg, #252A40 0%, #1E2235 100%)',
        'tl-gradient-header': 'linear-gradient(90deg, #1B264F 0%, #274690 100%)',
        'tl-grid': 'linear-gradient(rgba(87, 108, 168, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(87, 108, 168, 0.05) 1px, transparent 1px)',
        'tl-scanline': 'repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0px, rgba(255,255,255,0.02) 1px, transparent 1px, transparent 2px)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'price-flash-up': 'priceFlashUp 0.6s ease-out',
        'price-flash-down': 'priceFlashDown 0.6s ease-out',
        'ticker-scroll': 'tickerScroll 30s linear infinite',
        'data-pulse': 'dataPulse 2s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'fade-in': 'fadeIn 0.2s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
        'shimmer': 'shimmer 2s linear infinite',
        'spin-slow': 'spin 2s linear infinite',
      },
      keyframes: {
        priceFlashUp: {
          '0%': { backgroundColor: 'rgba(0, 200, 150, 0.3)' },
          '100%': { backgroundColor: 'transparent' },
        },
        priceFlashDown: {
          '0%': { backgroundColor: 'rgba(255, 71, 87, 0.3)' },
          '100%': { backgroundColor: 'transparent' },
        },
        tickerScroll: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        dataPulse: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.7 },
        },
        glow: {
          '0%': { boxShadow: '0 0 4px rgba(87, 108, 168, 0.3)' },
          '100%': { boxShadow: '0 0 12px rgba(87, 108, 168, 0.5)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
        fadeIn: {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: 0 },
          '100%': { transform: 'scale(1)', opacity: 1 },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      transitionDuration: {
        '150': '150ms',
        '250': '250ms',
        '400': '400ms',
      },
      backdropBlur: {
        'xs': '2px',
      },
    },
  },
  plugins: [],
}

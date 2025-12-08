# TradeLens Frontend

Next.js frontend for TradeLens - an educational quantitative trading platform.

## Features

- **Market Dashboard**: Real-time market overview
- **Stock Analysis**: Price charts, fundamentals, technical indicators
- **ML Predictions**: Interactive ML model configuration and results
- **AI Assistant**: Chat-based trading education
- **News Feed**: Stock-specific news aggregation

## Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local
# Edit .env.local with your backend URL

# Run development server
npm run dev
```

Visit http://localhost:3000

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | Backend API URL |

## Deployment

### Vercel (Recommended)

1. Push this repo to GitHub
2. Connect to Vercel
3. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = Your deployed backend URL (e.g., `https://tradelens-api.onrender.com`)
4. Deploy!

### Manual Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Connecting to Backend

The frontend connects to the backend via the `NEXT_PUBLIC_API_URL` environment variable.

**Local Development:**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Production:**
```
NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
```

Make sure your backend has CORS configured to allow requests from your frontend domain.

## Tech Stack

- **Framework**: Next.js 14
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **State Management**: TanStack Query
- **HTTP Client**: Axios
- **Icons**: Lucide React


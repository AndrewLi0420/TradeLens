# TradeLens - System Context & Onboarding

## Project Overview
TradeLens is a quantitative trading education platform providing real-time market data, sentiment analysis, and ML predictions. It is designed to teach users about trading concepts using live data.

## System Architecture

### High-Level Components
1.  **Frontend**: Next.js (React) application.
2.  **Backend**: FastAPI (Python) application.
3.  **Data Layer**: PostgreSQL (primary DB), Redis (caching), External APIs (yfinance, Anthropic).

### Architecture Diagram
```mermaid
graph TD
    User[User] --> Frontend[Frontend (Next.js)]
    Frontend -->|REST API| Backend[Backend (FastAPI)]
    
    subgraph "Backend Services"
        Backend --> MarketSvc[Market Data Service]
        Backend --> SentimentSvc[Sentiment Pipeline]
        Backend --> MLEngine[ML Engine]
        Backend --> LLMSvc[LLM Service]
    end
    
    subgraph "Data Layer"
        Backend --> Redis[Redis Cache]
        Backend --> Postgres[PostgreSQL]
    end
    
    subgraph "External APIs"
        MarketSvc --> YFinance[yfinance]
        SentimentSvc --> FinBERT[HuggingFace FinBERT]
        LLMSvc --> Claude[Anthropic Claude]
    end
```

## Technology Stack

### Backend
*   **Language**: Python 3.9+
*   **Framework**: FastAPI (Async)
*   **Core Libraries**:
    *   `yfinance`: Market data
    *   `pandas`, `numpy`: Data manipulation
    *   `scikit-learn`, `pytorch`: ML models
    *   `transformers`: FinBERT for NLP
    *   `anthropic`: LLM integration
    *   `sqlalchemy`, `psycopg2`: Database ORM/Drivers
    *   `redis`: Caching

### Frontend
*   **Framework**: Next.js 14 / React 18
*   **Language**: TypeScript
*   **Styling**: Tailwind CSS
*   **State Management**: React Query (@tanstack/react-query)
*   **Charting**: Recharts
*   **HTTP Client**: Axios

### Infrastructure
*   **Database**: PostgreSQL 14+
*   **Cache**: Redis 6+
*   **Containerization**: Docker
*   **Deployment Targets**: Render (Backend), Vercel (Frontend)

## Key Directories & Files

### Root
*   `SEPARATION_GUIDE.md`: Guide for splitting monorepo.
*   `docs/`: Detailed documentation resources.

### Backend (`tradelens/backend/`)
*   `main.py`: Entry point, API routes, and core service orchestration.
*   `sentiment_pipeline.py`: News fetching and FinBERT analysis logic.
*   `llm_service.py`: Interface with Claude API for explanations.
*   `advanced_features.py`: Backtesting, portfolio optimization logic.
*   `database.py`: DB models and connection logic.
*   `requirements.txt`: Python dependencies.

### Frontend (`tradelens/frontend/`)
*   `src/app/`: Next.js App Router pages.
    *   `page.tsx`: Dashboard.
    *   `stock/[ticker]/`: Stock detail view.
    *   `sandbox/`: ML interaction playground.
*   `src/components/`: Reusable UI components.
*   `src/lib/api.ts`: Typed API client methods.

## Data Flow Examples

### 1. Market Data Fetch
`GET /api/stock/{ticker}/price`
1.  Check Redis cache for `<ticker>_price`.
2.  If miss, call `yfinance` to fetch OHLCV data.
3.  Format data and store in Redis (5m TTL).
4.  Return JSON response.

### 2. ML Prediction
`POST /api/ml/predict`
1.  Frontend sends ticker and model config.
2.  Backend fetches 2y historical data (yfinance).
3.  Computes technical indicators (RSI, MACD, etc.).
4.  Trains selected model (e.g., RandomForest) on fly.
5.  Generates prediction + feature importance.
6.  Calls `llm_service` to generate text explanation of results.
7.  Returns combined payload.

## Development Setup
1.  **Backend**:
    ```bash
    cd tradelens/backend
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    uvicorn main:app --reload
    ```
2.  **Frontend**:
    ```bash
    cd tradelens/frontend
    npm install
    npm run dev
    ```
3.  **Env Vars**:
    *   Backend: `DATABASE_URL`, `REDIS_HOST`, `ANTHROPIC_API_KEY`.
    *   Frontend: `NEXT_PUBLIC_API_URL`.

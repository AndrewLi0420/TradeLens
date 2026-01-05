# TradeLens Backend API

FastAPI backend for TradeLens - an educational quantitative trading platform.

## Features

- **Market Data**: Real-time stock prices, fundamentals via yfinance
- **ML Predictions**: Random Forest and Linear Regression models for price prediction
- **Technical Indicators**: MA, RSI, MACD, Bollinger Bands, and more
- **AI Integration**: Claude-powered explanations and chat (requires API key)
- **News & Sentiment**: Stock news aggregation

## Quick Start

### Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Run development server
python main.py

#uvicorn main:app --reload --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Optional | For AI explanations |
| `HUGGINGFACE_TOKEN` | Optional | For FinBERT sentiment |
| `FRONTEND_URL` | Yes | Your frontend URL for CORS |
| `DATABASE_URL` | Optional | PostgreSQL connection string |
| `REDIS_HOST` | Optional | Redis host for caching |

## Deployment

### Render.com (Recommended)

1. Push this repo to GitHub
2. Connect to Render.com
3. Create new Web Service
4. Set environment variables in dashboard
5. Deploy!

### Railway

1. Push to GitHub
2. Connect to Railway
3. Add environment variables
4. Deploy automatically

### Docker

```bash
docker build -t tradelens-api .
docker run -p 8000:8000 --env-file .env tradelens-api
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market/overview` | GET | Major indices overview |
| `/api/stock/{ticker}/price` | GET | Stock price history |
| `/api/stock/{ticker}/fundamentals` | GET | Company fundamentals |
| `/api/news/{ticker}` | GET | Recent news |
| `/api/ml/predict` | POST | ML price prediction |
| `/api/ml/features/available` | GET | Available ML features |
| `/api/ai/explain-movement` | POST | AI movement explanation |
| `/api/ai/chat` | POST | AI chat assistant |
| `/health` | GET | Health check |

## CORS Configuration

The backend allows all origins by default. For production, update `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```


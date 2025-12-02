# TradeLens Deployment & Setup Guide

## 📋 Prerequisites

### System Requirements
- Python 3.9+
- Node.js 18+
- PostgreSQL 14+
- Redis 6+
- 8GB RAM minimum (16GB recommended for ML models)

### Required API Keys
1. **Anthropic API Key** (for Claude integration)
   - Get from: https://console.anthropic.com
   - Set as: `ANTHROPIC_API_KEY`

2. **HuggingFace Token** (for FinBERT)
   - Get from: https://huggingface.co/settings/tokens
   - Set as: `HUGGINGFACE_TOKEN`

---

## 🔧 Backend Setup

### 1. Create Virtual Environment

```bash
# Create project directory
mkdir tradelens && cd tradelens

# Create backend directory
mkdir backend && cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
# Create requirements.txt
cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
yfinance==0.2.32
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
torch==2.1.1
transformers==4.35.2
beautifulsoup4==4.12.2
aiohttp==3.9.1
redis==5.0.1
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
anthropic==0.7.8
python-dotenv==1.0.0
EOF

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup PostgreSQL Database

```bash
# Create database
psql -U postgres

CREATE DATABASE tradelens;
CREATE USER tradelens_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE tradelens TO tradelens_user;
\q
```

### 4. Create Environment Configuration

```bash
# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=postgresql://tradelens_user:your_secure_password@localhost:5432/tradelens

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Security
SECRET_KEY=$(openssl rand -hex 32)
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment
ENVIRONMENT=development
DEBUG=True
EOF
```

### 5. Initialize Database

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")
```

Run: `python database.py`

### 6. Start Backend Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🎨 Frontend Setup

### 1. Create Next.js Application

```bash
# Navigate to project root
cd ..

# Create Next.js app with TypeScript
npx create-next-app@latest frontend --typescript --tailwind --app --src-dir

cd frontend
```

### 2. Install Additional Dependencies

```bash
npm install recharts lucide-react @tanstack/react-query axios
npm install -D @types/node
```

### 3. Configure API Integration

```typescript
// src/lib/api.ts
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// API functions
export const marketAPI = {
  getOverview: () => api.get('/api/market/overview'),
  getStockPrice: (ticker: string, range: string) => 
    api.get(`/api/stock/${ticker}/price?range=${range}`),
  getFundamentals: (ticker: string) => 
    api.get(`/api/stock/${ticker}/fundamentals`),
};

export const sentimentAPI = {
  getSentiment: (ticker: string, days: number = 7) => 
    api.get(`/api/sentiment/${ticker}?days=${days}`),
  getNews: (ticker: string, limit: number = 10) => 
    api.get(`/api/news/${ticker}?limit=${limit}`),
};

export const mlAPI = {
  predict: (data: any) => api.post('/api/ml/predict', data),
  getFeatures: () => api.get('/api/ml/features/available'),
};

export const aiAPI = {
  explainMovement: (data: any) => api.post('/api/ai/explain-movement', data),
  chat: (data: any) => api.post('/api/ai/chat', data),
};
```

### 4. Create Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 5. Start Development Server

```bash
npm run dev
```

Frontend will be available at: http://localhost:3000

---

## 🤖 Claude AI Integration

### Complete LLM Service Implementation

```python
# llm_service.py
import os
from anthropic import Anthropic
from typing import Dict, List, Optional
import json

class ClaudeService:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-20250514"
    
    def explain_stock_movement(
        self,
        ticker: str,
        price_data: Dict,
        news_data: List[Dict],
        sentiment: Dict,
        fundamentals: Dict
    ) -> str:
        """Generate educational explanation of stock movement"""
        
        context = f"""
Stock: {ticker}
Current Price: ${price_data.get('current_price', 'N/A')}
Price Change: {price_data.get('change_percent', 'N/A')}%
Volume Change: {price_data.get('volume_change', 'N/A')}%

Fundamentals:
- Sector: {fundamentals.get('sector', 'N/A')}
- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}
- Market Cap: {fundamentals.get('market_cap', 'N/A')}

Sentiment Analysis:
- Overall: {sentiment.get('overall_sentiment', 'neutral')}
- Positive: {sentiment.get('avg_positive', 0):.1%}
- Negative: {sentiment.get('avg_negative', 0):.1%}

Recent News Headlines:
"""
        for i, article in enumerate(news_data[:5], 1):
            context += f"\n{i}. {article.get('title', '')}"
        
        prompt = f"""You are an educational trading tutor helping beginners understand stock movements.

{context}

Provide a clear, beginner-friendly explanation of why this stock moved today. Cover:
1. Primary catalysts (earnings, news, sector movement)
2. Technical factors (volume, momentum)
3. Market context
4. What this means for traders

Keep it educational, accurate, and easy to understand. Avoid jargon or explain any terms used.
"""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def explain_ml_prediction(
        self,
        ticker: str,
        model_type: str,
        prediction_result: Dict,
        features_used: List[str]
    ) -> str:
        """Explain ML model behavior and predictions"""
        
        prompt = f"""You are teaching someone about machine learning in trading.

Model: {model_type}
Stock: {ticker}
Prediction: ${prediction_result.get('predicted_price', 'N/A')}
Current Price: ${prediction_result.get('current_price', 'N/A')}
Change: {prediction_result.get('prediction_change', 'N/A')}%

Features Used: {', '.join(features_used)}

Metrics:
- RMSE: {prediction_result.get('metrics', {}).get('rmse', 'N/A')}
- Directional Accuracy: {prediction_result.get('metrics', {}).get('directional_accuracy', 'N/A')}%

Explain:
1. How this model works in simple terms
2. What these features tell us
3. How to interpret these metrics
4. Limitations and caveats
5. What to consider when using ML for trading

Be educational and honest about both capabilities and limitations.
"""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            return f"Unable to generate explanation: {str(e)}"
    
    def chat(
        self,
        user_message: str,
        conversation_history: List[Dict],
        context: Optional[Dict] = None
    ) -> str:
        """Conversational AI assistant for trading education"""
        
        system_prompt = """You are an expert trading educator and tutor. Your role is to:

1. Teach quantitative trading concepts clearly
2. Explain technical indicators and their uses
3. Help users understand market movements
4. Explain ML and data science in trading
5. Always prioritize education over predictions

Guidelines:
- Be accurate and factually correct
- Explain concepts simply but thoroughly
- Use examples when helpful
- Acknowledge limitations and risks
- Never provide financial advice, only education
- Encourage critical thinking and research"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add context if provided
        context_str = ""
        if context:
            context_str = f"\n\nContext: {json.dumps(context, indent=2)}"
        
        messages.append({
            "role": "user",
            "content": user_message + context_str
        })
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=messages
            )
            
            return message.content[0].text
            
        except Exception as e:
            return f"I apologize, but I'm having trouble responding right now: {str(e)}"

# Integration with FastAPI
from fastapi import APIRouter

llm_router = APIRouter()
claude_service = ClaudeService()

@llm_router.post("/api/ai/explain-movement")
async def explain_movement_endpoint(request: AIExplainRequest):
    # Fetch required data
    ticker = request.ticker.upper()
    
    # Get price data, news, sentiment, fundamentals
    # (Use existing endpoints)
    
    explanation = claude_service.explain_stock_movement(
        ticker=ticker,
        price_data={},  # Populate from API calls
        news_data=[],   # Populate from API calls
        sentiment={},   # Populate from API calls
        fundamentals={} # Populate from API calls
    )
    
    return {"ticker": ticker, "explanation": explanation}

@llm_router.post("/api/ai/chat")
async def chat_endpoint(request: ChatRequest):
    response = claude_service.chat(
        user_message=request.message,
        conversation_history=[],  # Maintain in session
        context=request.context
    )
    
    return {"response": response}
```

---

## 🚀 Production Deployment

### Option 1: Vercel (Frontend) + Render (Backend)

#### Deploy Backend to Render

1. Create `render.yaml`:

```yaml
services:
  - type: web
    name: tradelens-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: REDIS_HOST
        sync: false
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: HUGGINGFACE_TOKEN
        sync: false

databases:
  - name: tradelens-db
    databaseName: tradelens
    user: tradelens_user

  - name: tradelens-redis
    plan: starter
```

2. Push to GitHub and connect to Render
3. Set environment variables in Render dashboard

#### Deploy Frontend to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd frontend
vercel

# Set environment variables
vercel env add NEXT_PUBLIC_API_URL production
```

### Option 2: AWS ECS (Docker)

Create `Dockerfile` for backend:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_HOST=redis
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - db
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=tradelens
      - POSTGRES_USER=tradelens_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

---

## 🔍 Testing

### Backend Tests

```bash
# Install pytest
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v
```

### Frontend Tests

```bash
# Install testing libraries
npm install -D @testing-library/react @testing-library/jest-dom vitest

# Run tests
npm test
```

---

## 📊 Monitoring & Logging

### Setup Application Monitoring

```python
# monitoring.py
import logging
from prometheus_client import Counter, Histogram
import time

# Metrics
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
api_latency = Histogram('api_latency_seconds', 'API latency', ['endpoint'])

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tradelens.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🔐 Security Checklist

- [ ] Environment variables secured
- [ ] Database credentials rotated
- [ ] API rate limiting implemented
- [ ] CORS properly configured
- [ ] SQL injection prevention (using ORMs)
- [ ] Input validation on all endpoints
- [ ] HTTPS enabled in production
- [ ] Authentication & authorization implemented
- [ ] API keys encrypted at rest
- [ ] Regular dependency updates

---

## 📈 Performance Optimization

1. **Caching Strategy**
   - Redis for frequently accessed data
   - Cache invalidation on updates
   - 5min cache for real-time data
   - 1hr cache for fundamental data

2. **Database Optimization**
   - Indexed columns: ticker, date, user_id
   - Connection pooling
   - Query optimization

3. **API Optimization**
   - Async operations for I/O
   - Batch requests where possible
   - Pagination for large datasets

---

## 📚 Next Steps

1. Implement user authentication
2. Add portfolio tracking
3. Build backtesting engine
4. Create mobile app
5. Add real-time WebSocket updates
6. Implement paper trading
7. Build community features

---

## 🆘 Troubleshooting

### Common Issues

**Issue: FinBERT model loading fails**
```bash
# Solution: Download model manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ProsusAI/finbert')"
```

**Issue: Redis connection fails**
```bash
# Solution: Start Redis service
sudo service redis-server start  # Linux
brew services start redis         # Mac
```

**Issue: Database connection timeout**
```bash
# Solution: Increase connection pool
# In database.py, add: pool_size=20, max_overflow=0
```

---

## 📞 Support

- Documentation: https://tradelens.io/docs
- GitHub Issues: https://github.com/yourusername/tradelens/issues
- Email: support@tradelens.io

---

**Built with ❤️ for quantitative trading education**
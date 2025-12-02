# main.py - FastAPI Backend for TradeLens
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache
import logging

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Redis for caching (optional)
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TradeLens API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection (optional - graceful fallback if unavailable)
redis_client = None
try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()  # Test connection
    logger.info("✅ Redis connected successfully")
except Exception as e:
    logger.warning(f"⚠️ Redis not available, caching disabled: {e}")
    redis_client = None


class RedisCache:
    """Simple cache wrapper that handles Redis being unavailable"""
    
    @staticmethod
    def get(key: str):
        if redis_client:
            try:
                return redis_client.get(key)
            except:
                return None
        return None
    
    @staticmethod
    def setex(key: str, ttl: int, value: str):
        if redis_client:
            try:
                redis_client.setex(key, ttl, value)
            except:
                pass


cache = RedisCache()

# ============================================================================
# DATA MODELS
# ============================================================================

class StockPriceRequest(BaseModel):
    ticker: str
    range: str = "1m"  # 1d, 5d, 1m, 3m, 6m, 1y, 5y, max

class MLPredictionRequest(BaseModel):
    ticker: str
    model_type: str = "random_forest"  # random_forest, linear, lstm
    features: List[str] = ["ma_20", "rsi", "volume"]
    prediction_window: str = "1w"  # 1d, 3d, 1w, 1m

class AIExplainRequest(BaseModel):
    ticker: str
    date: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    ticker: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

# ============================================================================
# MARKET DATA SERVICE
# ============================================================================

def get_range_params(range_str: str):
    """Convert range string to yfinance parameters"""
    range_map = {
        "1d": ("1d", "5m"),
        "5d": ("5d", "15m"),
        "1m": ("1mo", "1h"),
        "3m": ("3mo", "1d"),
        "6m": ("6mo", "1d"),
        "1y": ("1y", "1d"),
        "5y": ("5y", "1wk"),
        "max": ("max", "1wk")
    }
    return range_map.get(range_str, ("1mo", "1d"))

@app.get("/api/market/overview")
async def get_market_overview():
    """Get major indices overview"""
    tickers = ["^GSPC", "^IXIC", "^DJI", "^VIX"]
    names = ["S&P 500", "NASDAQ", "Dow Jones", "VIX"]
    
    results = []
    for ticker, name in zip(tickers, names):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                
                results.append({
                    "name": name,
                    "ticker": ticker,
                    "price": round(current, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_pct, 2)
                })
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            
    return {"indices": results, "updated_at": datetime.now().isoformat()}

@app.get("/api/stock/{ticker}/price")
async def get_stock_price(ticker: str, range: str = "1m"):
    """Get stock price data with OHLCV"""
    cache_key = f"price:{ticker}:{range}"
    
    # Check cache (5 min expiry for recent data)
    cached = cache.get(cache_key)
    if cached and range in ["1d", "5d"]:
        return json.loads(cached)
    
    try:
        period, interval = get_range_params(range)
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found for ticker")
        
        # Format data for frontend
        data = []
        for idx, row in hist.iterrows():
            data.append({
                "time": int(idx.timestamp()),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
        
        result = {
            "ticker": ticker.upper(),
            "range": range,
            "data": data,
            "current_price": round(hist['Close'].iloc[-1], 2),
            "fetched_at": datetime.now().isoformat()
        }
        
        # Cache result
        cache.setex(cache_key, 300, json.dumps(result))
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{ticker}/fundamentals")
async def get_fundamentals(ticker: str):
    """Get fundamental data"""
    cache_key = f"fundamentals:{ticker}"
    cached = cache.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fundamentals = {
            "ticker": ticker.upper(),
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "peg_ratio": info.get("pegRatio", None),
            "beta": info.get("beta", None),
            "eps": info.get("trailingEps", None),
            "dividend_yield": info.get("dividendYield", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
            "description": info.get("longBusinessSummary", "")
        }
        
        # Cache for 1 hour
        cache.setex(cache_key, 3600, json.dumps(fundamentals))
        
        return fundamentals
        
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for ML features"""
    df = df.copy()
    
    # Moving averages
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Volume indicators
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    return df

@app.get("/api/ml/features/available")
async def get_available_features():
    """List all available features for ML models"""
    return {
        "features": [
            {"name": "ma_5", "description": "5-day moving average", "category": "trend"},
            {"name": "ma_10", "description": "10-day moving average", "category": "trend"},
            {"name": "ma_20", "description": "20-day moving average", "category": "trend"},
            {"name": "ma_50", "description": "50-day moving average", "category": "trend"},
            {"name": "rsi", "description": "Relative Strength Index (14-day)", "category": "momentum"},
            {"name": "macd", "description": "MACD indicator", "category": "momentum"},
            {"name": "macd_signal", "description": "MACD signal line", "category": "momentum"},
            {"name": "bb_upper", "description": "Bollinger Band upper", "category": "volatility"},
            {"name": "bb_lower", "description": "Bollinger Band lower", "category": "volatility"},
            {"name": "volume_ratio", "description": "Volume / 20-day avg volume", "category": "volume"},
            {"name": "returns", "description": "Daily returns", "category": "returns"},
            {"name": "volatility", "description": "20-day volatility", "category": "volatility"}
        ]
    }

# ============================================================================
# ML PREDICTION ENGINE
# ============================================================================

@app.post("/api/ml/predict")
async def create_prediction(request: MLPredictionRequest):
    """Generate ML prediction"""
    try:
        # Fetch historical data
        stock = yf.Ticker(request.ticker)
        hist = stock.history(period="2y")  # Get 2 years for feature calculation
        
        if len(hist) < 100:
            raise HTTPException(status_code=400, detail="Insufficient historical data")
        
        # Calculate features
        df = calculate_technical_indicators(hist)
        df = df.dropna()
        
        # Determine prediction horizon
        horizon_map = {"1d": 1, "3d": 3, "1w": 5, "1m": 20}
        horizon = horizon_map.get(request.prediction_window, 5)
        
        # Create target variable (future returns)
        df['target'] = df['Close'].shift(-horizon)
        df = df.dropna()
        
        # Select features
        available_features = ['ma_5', 'ma_10', 'ma_20', 'ma_50', 'rsi', 'macd', 
                            'macd_signal', 'volume_ratio', 'volatility']
        selected_features = [f for f in request.features if f in available_features]
        
        if not selected_features:
            raise HTTPException(status_code=400, detail="No valid features selected")
        
        X = df[selected_features].values
        y = df['target'].values
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if request.model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif request.model_type == "linear":
            model = LinearRegression()
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        
        # Directional accuracy
        direction_actual = np.sign(y_test - df['Close'].values[split_idx:split_idx+len(y_test)])
        direction_pred = np.sign(y_pred_test - df['Close'].values[split_idx:split_idx+len(y_pred_test)])
        directional_accuracy = np.mean(direction_actual == direction_pred)
        
        # Current prediction
        current_features = df[selected_features].iloc[-1:].values
        current_scaled = scaler.transform(current_features)
        current_prediction = model.predict(current_scaled)[0]
        
        # Feature importance (for tree-based models)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = {
                feat: float(imp) 
                for feat, imp in zip(selected_features, model.feature_importances_)
            }
        
        return {
            "ticker": request.ticker.upper(),
            "model_type": request.model_type,
            "prediction_window": request.prediction_window,
            "current_price": round(df['Close'].iloc[-1], 2),
            "predicted_price": round(current_prediction, 2),
            "prediction_change": round(((current_prediction / df['Close'].iloc[-1]) - 1) * 100, 2),
            "metrics": {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "directional_accuracy": round(directional_accuracy * 100, 2)
            },
            "feature_importance": feature_importance,
            "features_used": selected_features,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# NEWS & SENTIMENT (Placeholder - requires additional scraping setup)
# ============================================================================

@app.get("/api/news/{ticker}")
async def get_news(ticker: str, limit: int = 10):
    """Get news for ticker"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:limit]
        
        formatted_news = []
        for article in news:
            formatted_news.append({
                "title": article.get("title", ""),
                "publisher": article.get("publisher", ""),
                "link": article.get("link", ""),
                "published_at": datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat(),
                "thumbnail": article.get("thumbnail", {}).get("resolutions", [{}])[0].get("url", "")
            })
        
        return {
            "ticker": ticker.upper(),
            "news": formatted_news,
            "count": len(formatted_news)
        }
        
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AI ASSISTANT (Uses Claude API - requires ANTHROPIC_API_KEY env var)
# ============================================================================

@app.post("/api/ai/explain-movement")
async def explain_movement(request: AIExplainRequest):
    """Generate AI explanation of stock movement"""
    # This would integrate with Claude API
    # For now, return structured response
    
    ticker = request.ticker.upper()
    
    try:
        # Fetch recent data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        news = stock.news[:5]
        info = stock.info
        
        # Calculate movement
        if len(hist) >= 2:
            current = hist['Close'].iloc[-1]
            previous = hist['Close'].iloc[-2]
            change_pct = ((current - previous) / previous) * 100
            
            return {
                "ticker": ticker,
                "movement": round(change_pct, 2),
                "explanation": f"Educational explanation would be generated here using Claude API with context about {ticker}'s recent performance, news, and market conditions.",
                "data_points": {
                    "current_price": round(current, 2),
                    "previous_close": round(previous, 2),
                    "volume_change": round((hist['Volume'].iloc[-1] / hist['Volume'].iloc[-2] - 1) * 100, 2),
                    "sector": info.get("sector", "N/A"),
                    "recent_news_count": len(news)
                }
            }
    
    except Exception as e:
        logger.error(f"Error explaining movement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/chat")
async def ai_chat(request: ChatRequest):
    """Chat with AI assistant"""
    # Integration with Claude API
    return {
        "response": "AI assistant response would be generated here using Claude API.",
        "context_used": request.ticker or "general",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
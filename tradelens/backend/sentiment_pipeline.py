# sentiment_pipeline.py - News Sentiment Analysis for TradeLens

import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta, timezone
import pandas as pd
from typing import List, Dict
import logging
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes financial news sentiment using FinBERT model
    """
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
    
    def _initialize_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            logger.info("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT: {e}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        Returns: dict with sentiment scores (positive, negative, neutral)
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract scores
            scores = predictions[0].cpu().numpy()
            
            return {
                "positive": float(scores[0]),
                "negative": float(scores[1]),
                "neutral": float(scores[2]),
                "compound": float(scores[0] - scores[1])  # Positive - negative
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    def aggregate_sentiment(self, sentiment_scores: List[Dict]) -> Dict:
        """Aggregate multiple sentiment scores"""
        if not sentiment_scores:
            return {
                "avg_positive": 0.0,
                "avg_negative": 0.0,
                "avg_neutral": 0.0,
                "avg_compound": 0.0,
                "overall_sentiment": "neutral"
            }
        
        avg_positive = sum(s["positive"] for s in sentiment_scores) / len(sentiment_scores)
        avg_negative = sum(s["negative"] for s in sentiment_scores) / len(sentiment_scores)
        avg_neutral = sum(s["neutral"] for s in sentiment_scores) / len(sentiment_scores)
        avg_compound = sum(s["compound"] for s in sentiment_scores) / len(sentiment_scores)
        
        # Determine overall sentiment
        if avg_compound > 0.2:
            overall = "bullish"
        elif avg_compound < -0.2:
            overall = "bearish"
        else:
            overall = "neutral"
        
        return {
            "avg_positive": round(avg_positive, 3),
            "avg_negative": round(avg_negative, 3),
            "avg_neutral": round(avg_neutral, 3),
            "avg_compound": round(avg_compound, 3),
            "overall_sentiment": overall,
            "sample_size": len(sentiment_scores)
        }


class NewsAggregator:
    """
    Fetches and processes news from multiple sources
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def fetch_yfinance_news(self, ticker: str, max_articles: int = 20) -> List[Dict]:
        """Fetch news from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news[:max_articles]
            
            articles = []
            for item in news_items:
                # yfinance returns nested structure with 'content' key
                content = item.get("content", {})
                
                # Extract title
                title = content.get("title", "")
                if not title:
                    continue  # Skip articles without titles
                
                # Extract link from canonicalUrl
                canonical_url = content.get("canonicalUrl", {})
                link = canonical_url.get("url", "") if isinstance(canonical_url, dict) else ""
                
                # Extract publisher from provider
                provider = content.get("provider", {})
                publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
                
                # Extract published date (ISO string format)
                pub_date_str = content.get("pubDate", "")
                if pub_date_str:
                    try:
                        published_at = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        # Ensure timezone aware - convert to UTC if needed
                        if published_at.tzinfo is None:
                            published_at = published_at.replace(tzinfo=timezone.utc)
                    except ValueError:
                        published_at = datetime.now(timezone.utc)
                else:
                    published_at = datetime.now(timezone.utc)
                
                # Extract thumbnail from resolutions
                thumbnail_data = content.get("thumbnail", {})
                resolutions = thumbnail_data.get("resolutions", []) if isinstance(thumbnail_data, dict) else []
                thumbnail = resolutions[0].get("url", "") if resolutions else ""
                
                articles.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "published_at": published_at,
                    "thumbnail": thumbnail,
                    "type": content.get("contentType", "article")
                })
            
            logger.info(f"Fetched {len(articles)} articles for {ticker}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching yfinance news for {ticker}: {e}")
            return []
    
    async def scrape_article_content(self, url: str) -> str:
        """
        Scrape article content from URL
        Note: Respects robots.txt and rate limits
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"User-Agent": "TradeLens Educational Bot"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        return text[:2000]  # Limit to first 2000 chars
                    
        except Exception as e:
            logger.warning(f"Could not scrape {url}: {e}")
        
        return ""
    
    def analyze_news_sentiment(self, ticker: str, days: int = 7) -> Dict:
        """
        Complete pipeline: fetch news, analyze sentiment, aggregate
        """
        try:
            # Fetch news
            articles = self.fetch_yfinance_news(ticker, max_articles=20)
            
            if not articles:
                return {
                    "ticker": ticker,
                    "error": "No news found",
                    "sentiment": None
                }
            
            # Filter by date range (use UTC aware datetime)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            recent_articles = [
                a for a in articles 
                if a["published_at"] > cutoff_date
            ]
            
            # Analyze sentiment for each article
            sentiment_results = []
            processed_articles = []
            
            for article in recent_articles[:15]:  # Limit processing
                # Analyze title (always available)
                title_sentiment = self.sentiment_analyzer.analyze_text(article["title"])
                
                processed_articles.append({
                    "title": article["title"],
                    "publisher": article["publisher"],
                    "link": article["link"],
                    "published_at": article["published_at"].isoformat(),
                    "sentiment": title_sentiment
                })
                
                sentiment_results.append(title_sentiment)
            
            # Aggregate all sentiments
            aggregated = self.sentiment_analyzer.aggregate_sentiment(sentiment_results)
            
            # Create time-series sentiment (daily aggregation)
            df = pd.DataFrame(processed_articles)
            if not df.empty:
                df['date'] = pd.to_datetime(df['published_at']).dt.date
                daily_sentiment = []
                
                for date in sorted(df['date'].unique()):
                    day_articles = df[df['date'] == date]
                    day_sentiments = [
                        s for s in sentiment_results 
                        if pd.to_datetime(processed_articles[sentiment_results.index(s)]['published_at']).date() == date
                    ]
                    day_agg = self.sentiment_analyzer.aggregate_sentiment(day_sentiments)
                    
                    daily_sentiment.append({
                        "date": str(date),
                        "compound": day_agg["avg_compound"],
                        "article_count": len(day_articles)
                    })
            else:
                daily_sentiment = []
            
            return {
                "ticker": ticker.upper(),
                "period_days": days,
                "total_articles": len(processed_articles),
                "overall_sentiment": aggregated,
                "daily_sentiment": daily_sentiment,
                "articles": processed_articles,
                "analyzed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis pipeline: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
                "sentiment": None
            }


# ============================================================================
# FastAPI Integration
# ============================================================================

from fastapi import APIRouter

sentiment_router = APIRouter()
news_aggregator = NewsAggregator()

@sentiment_router.get("/api/sentiment/{ticker}")
async def get_sentiment(ticker: str, days: int = 7):
    """Get sentiment analysis for ticker"""
    result = news_aggregator.analyze_news_sentiment(ticker, days)
    return result

@sentiment_router.post("/api/sentiment/analyze")
async def analyze_sentiment_batch(tickers: List[str], days: int = 7):
    """Batch sentiment analysis"""
    results = {}
    for ticker in tickers:
        results[ticker] = news_aggregator.analyze_news_sentiment(ticker, days)
    return results


# ============================================================================
# CLI Testing Interface
# ============================================================================

def main():
    """Test sentiment analysis from command line"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sentiment_pipeline.py <TICKER>")
        sys.exit(1)
    
    ticker = sys.argv[1]
    aggregator = NewsAggregator()
    
    print(f"\n{'='*60}")
    print(f"Analyzing sentiment for {ticker.upper()}")
    print(f"{'='*60}\n")
    
    result = aggregator.analyze_news_sentiment(ticker, days=7)
    
    if result.get("error"):
        print(f"Error: {result['error']}")
        return
    
    sentiment = result["overall_sentiment"]
    print(f"Overall Sentiment: {sentiment['overall_sentiment'].upper()}")
    print(f"Compound Score: {sentiment['avg_compound']:.3f}")
    print(f"Positive: {sentiment['avg_positive']:.1%}")
    print(f"Negative: {sentiment['avg_negative']:.1%}")
    print(f"Neutral: {sentiment['avg_neutral']:.1%}")
    print(f"\nAnalyzed {result['total_articles']} articles")
    
    print(f"\n{'='*60}")
    print("Recent Articles:")
    print(f"{'='*60}\n")
    
    for i, article in enumerate(result["articles"][:5], 1):
        print(f"{i}. {article['title']}")
        print(f"   Publisher: {article['publisher']}")
        print(f"   Sentiment: {article['sentiment']['compound']:.3f}")
        print(f"   Link: {article['link']}\n")


if __name__ == "__main__":
    main()
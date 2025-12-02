import os
from anthropic import Anthropic
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class ClaudeService:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "sk-ant-placeholder":
            logger.warning("ANTHROPIC_API_KEY not set - AI features will return mock responses")
            self.client = None
        else:
            self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
    
    def explain_stock_movement(
        self,
        ticker: str,
        price_data: Dict,
        news_data: List[Dict],
        sentiment: Dict,
        fundamentals: Dict
    ) -> str:
        if not self.client:
            return f"AI explanation for {ticker} movement would appear here. Set ANTHROPIC_API_KEY in .env to enable real AI responses."
        
        context = f"""
Stock: {ticker}
Current Price: ${price_data.get('current_price', 'N/A')}
Price Change: {price_data.get('change_percent', 'N/A')}%

Fundamentals:
- Sector: {fundamentals.get('sector', 'N/A')}
- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}

Sentiment: {sentiment.get('overall_sentiment', 'neutral')}

Recent News:
"""
        for i, article in enumerate(news_data[:3], 1):
            context += f"\n{i}. {article.get('title', '')}"
        
        prompt = f"""You are an educational trading tutor. Explain why this stock moved today in a beginner-friendly way:

{context}

Provide a clear explanation covering catalysts, technical factors, and market context."""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Unable to generate AI explanation: {str(e)}"
    
    def chat(
        self,
        user_message: str,
        conversation_history: List[Dict] = None,
        context: Optional[Dict] = None
    ) -> str:
        if not self.client:
            return "AI chat is not available. Please set ANTHROPIC_API_KEY in your .env file."
        
        if conversation_history is None:
            conversation_history = []
        
        messages = []
        for msg in conversation_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=messages
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"I apologize, but I'm having trouble responding: {str(e)}"

# Singleton instance
claude_service = ClaudeService()


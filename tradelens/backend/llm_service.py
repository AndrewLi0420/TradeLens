import os
from google import genai
from typing import Dict, List, Optional
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ClaudeService:
    def __init__(self):
        # Try GEMINI_API_KEY first, fall back to ANTHROPIC_API_KEY
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            logger.warning("No API key found in environment")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key)
                logger.info("AI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize client: {e}")
                self.client = None
        self.model = "gemini-2.0-flash-exp"
    
    def explain_stock_movement(
        self,
        ticker: str,
        price_data: Dict,
        news_data: List[Dict],
        sentiment: Dict,
        fundamentals: Dict
    ) -> str:
        if not self.client:
            return f"⚠️ AI explanation for {ticker} is not available. To enable AI features:\n\n1. Get a free Gemini API key at: https://aistudio.google.com/apikey\n2. Add it to your backend/.env file as: GEMINI_API_KEY=your_key_here\n3. Restart the backend server"
        
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
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Unable to generate AI explanation: {str(e)}"
    
    def chat(
        self,
        user_message: str,
        conversation_history: List[Dict] = None,
        context: Optional[Dict] = None
    ) -> str:
        if not self.client:
            return "⚠️ AI chat is not configured. To enable AI features:\n\n1. Get a free Gemini API key at: https://aistudio.google.com/apikey\n2. Add it to your backend/.env file as: GEMINI_API_KEY=your_key_here\n3. Restart the backend server\n\nThe chatbot will work once configured!"
        
        if conversation_history is None:
            conversation_history = []
        
        # Build conversation context for Gemini
        conversation_text = ""
        for msg in conversation_history[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        conversation_text += f"User: {user_message}\n\nAssistant:"
        
        try:
            logger.info(f"Sending message to Gemini: {user_message[:50]}...")
            # For google-genai SDK
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message
            )
            
            if hasattr(response, 'text'):
                return response.text
            else:
                logger.error(f"Unexpected response format from Gemini: {response}")
                return "I received an empty response from the AI."
                
        except Exception as e:
            logger.error(f"Gemini API error during chat: {str(e)}", exc_info=True)
            return f"I apologize, but I'm having trouble responding right now. (Error: {str(e)})"

# Singleton instance
llm_service = ClaudeService()  # Kept class name for compatibility


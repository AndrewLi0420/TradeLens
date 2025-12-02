# advanced_features.py - Advanced TradeLens Features

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# ADVANCED ML FEATURES
# ============================================================================

class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trades = []
        self.portfolio_value = []
    
    def backtest_ml_strategy(
        self,
        ticker: str,
        predictions: List[Dict],
        actual_prices: pd.Series,
        threshold: float = 0.02  # 2% minimum predicted change
    ) -> Dict:
        """
        Backtest ML prediction strategy
        
        Strategy: Buy when predicted increase > threshold,
                 Sell when predicted decrease > threshold
        """
        
        capital = self.initial_capital
        shares = 0
        position_value = 0
        
        for i, pred in enumerate(predictions):
            if i >= len(actual_prices):
                break
                
            current_price = actual_prices.iloc[i]
            predicted_change = pred['predicted_change_percent'] / 100
            
            # Buy signal
            if predicted_change > threshold and shares == 0:
                shares = capital / current_price
                capital = 0
                position_value = shares * current_price
                
                self.trades.append({
                    'type': 'buy',
                    'date': actual_prices.index[i],
                    'price': current_price,
                    'shares': shares,
                    'reason': f'Predicted +{predicted_change:.2%}'
                })
            
            # Sell signal
            elif predicted_change < -threshold and shares > 0:
                capital = shares * current_price
                position_value = capital
                
                self.trades.append({
                    'type': 'sell',
                    'date': actual_prices.index[i],
                    'price': current_price,
                    'shares': shares,
                    'capital': capital,
                    'reason': f'Predicted {predicted_change:.2%}'
                })
                
                shares = 0
            
            # Calculate current portfolio value
            current_value = capital + (shares * current_price)
            self.portfolio_value.append({
                'date': actual_prices.index[i],
                'value': current_value
            })
        
        # Final portfolio value
        final_value = capital + (shares * actual_prices.iloc[-1])
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate buy-and-hold benchmark
        bh_shares = self.initial_capital / actual_prices.iloc[0]
        bh_final = bh_shares * actual_prices.iloc[-1]
        bh_return = (bh_final - self.initial_capital) / self.initial_capital
        
        # Calculate metrics
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'buy_hold_return': bh_return,
            'buy_hold_return_pct': bh_return * 100,
            'alpha': (total_return - bh_return) * 100,
            'num_trades': len(self.trades),
            'trades': self.trades,
            'portfolio_history': self.portfolio_value
        }
        
        # Calculate Sharpe ratio if enough data
        if len(self.portfolio_value) > 1:
            returns = pd.Series([pv['value'] for pv in self.portfolio_value]).pct_change().dropna()
            if len(returns) > 0 and returns.std() != 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                metrics['sharpe_ratio'] = sharpe
        
        return metrics


class FeatureImportanceAnalyzer:
    """
    Analyze and visualize feature importance across different models
    """
    
    @staticmethod
    def analyze_feature_stability(
        ticker: str,
        features: List[str],
        n_splits: int = 5
    ) -> Dict:
        """
        Use time-series cross-validation to assess feature stability
        """
        
        # Fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        
        # Calculate features
        from main import calculate_technical_indicators
        df = calculate_technical_indicators(hist)
        df = df.dropna()
        
        # Prepare data
        X = df[features].values
        y = df['Close'].shift(-5).values  # 5-day ahead prediction
        
        # Remove NaN target values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        feature_importance_history = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Store feature importance
            feature_importance_history.append({
                feat: imp for feat, imp in zip(features, model.feature_importances_)
            })
        
        # Calculate mean and std of feature importance
        importance_stats = {}
        for feature in features:
            importances = [fi[feature] for fi in feature_importance_history]
            importance_stats[feature] = {
                'mean': np.mean(importances),
                'std': np.std(importances),
                'stability_score': 1 - (np.std(importances) / (np.mean(importances) + 1e-10))
            }
        
        # Rank features
        ranked_features = sorted(
            importance_stats.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        
        return {
            'ticker': ticker,
            'feature_importance': importance_stats,
            'ranked_features': ranked_features,
            'n_splits': n_splits
        }


# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================

class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimization
    """
    
    @staticmethod
    def calculate_portfolio_metrics(
        tickers: List[str],
        weights: List[float],
        period: str = "1y"
    ) -> Dict:
        """
        Calculate portfolio risk and return metrics
        """
        
        # Fetch data for all tickers
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            data[ticker] = hist['Close']
        
        # Create returns dataframe
        df = pd.DataFrame(data)
        returns = df.pct_change().dropna()
        
        # Calculate portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252
        
        # Calculate portfolio volatility
        cov_matrix = returns.cov() * 252
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0.02)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'weights': dict(zip(tickers, weights))
        }
    
    @staticmethod
    def efficient_frontier(
        tickers: List[str],
        num_portfolios: int = 1000
    ) -> List[Dict]:
        """
        Generate efficient frontier portfolios
        """
        
        # Fetch data
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            data[ticker] = hist['Close']
        
        df = pd.DataFrame(data)
        returns = df.pct_change().dropna()
        
        # Generate random portfolios
        portfolios = []
        
        for _ in range(num_portfolios):
            # Random weights
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            
            # Calculate metrics
            portfolio_return = np.sum(returns.mean() * weights) * 252
            cov_matrix = returns.cov() * 252
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
            
            portfolios.append({
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe': sharpe_ratio,
                'weights': weights.tolist()
            })
        
        return portfolios


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

class RiskManager:
    """
    Risk management tools and metrics
    """
    
    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR)
        """
        var = RiskManager.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Dict:
        """
        Calculate maximum drawdown
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'date': max_dd_date,
            'drawdown_series': drawdown
        }
    
    @staticmethod
    def position_sizing_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Kelly Criterion for position sizing
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate  # Win probability
        q = 1 - p  # Loss probability
        
        kelly = (b * p - q) / b
        
        # Use fractional Kelly for safety
        return max(0, min(kelly * 0.25, 0.25))  # Cap at 25% and use 1/4 Kelly


# ============================================================================
# EDUCATIONAL CONTENT GENERATOR
# ============================================================================

class EducationalContentGenerator:
    """
    Generate educational content about trading concepts
    """
    
    @staticmethod
    def explain_indicator(indicator: str) -> Dict:
        """
        Detailed explanations of technical indicators
        """
        
        explanations = {
            'rsi': {
                'name': 'Relative Strength Index',
                'description': 'Momentum oscillator measuring speed and magnitude of price changes',
                'calculation': 'RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss',
                'interpretation': {
                    'above_70': 'Potentially overbought - price may decline',
                    'below_30': 'Potentially oversold - price may increase',
                    'divergence': 'Price and RSI moving in opposite directions signals reversal'
                },
                'example': 'If a stock has RSI of 75, it suggests strong buying pressure but may be due for a pullback',
                'limitations': 'Can remain overbought/oversold in strong trends. Use with other indicators.'
            },
            'macd': {
                'name': 'Moving Average Convergence Divergence',
                'description': 'Trend-following momentum indicator showing relationship between two EMAs',
                'calculation': 'MACD = 12-period EMA - 26-period EMA; Signal = 9-period EMA of MACD',
                'interpretation': {
                    'bullish_crossover': 'MACD crosses above signal line - potential buy signal',
                    'bearish_crossover': 'MACD crosses below signal line - potential sell signal',
                    'zero_line': 'MACD above zero = bullish; below zero = bearish'
                },
                'example': 'When MACD crosses above signal line with positive divergence, it suggests strengthening upward momentum',
                'limitations': 'Lagging indicator - signals may come late. False signals in sideways markets.'
            },
            'ma': {
                'name': 'Moving Average',
                'description': 'Average price over a specified time period, smoothing price action',
                'calculation': 'Sum of closing prices over N periods / N',
                'interpretation': {
                    'price_above': 'Price above MA suggests uptrend',
                    'price_below': 'Price below MA suggests downtrend',
                    'golden_cross': '50-day MA crosses above 200-day MA - strong bullish signal'
                },
                'example': 'If stock trades above 200-day MA, it indicates long-term uptrend',
                'limitations': 'Lagging indicator. Whipsaws in choppy markets.'
            }
        }
        
        return explanations.get(indicator.lower(), {'error': 'Indicator not found'})
    
    @staticmethod
    def explain_ml_concept(concept: str) -> Dict:
        """
        Explain ML concepts in trading context
        """
        
        concepts = {
            'overfitting': {
                'definition': 'Model learns training data too well, including noise, resulting in poor generalization',
                'in_trading': 'A model that perfectly predicts historical prices but fails on new data',
                'how_to_detect': 'Large gap between training and test accuracy',
                'prevention': [
                    'Use cross-validation',
                    'Keep model simple',
                    'Regularization techniques',
                    'More training data',
                    'Feature selection'
                ],
                'example': 'Model achieves 95% accuracy on historical data but only 45% on new data'
            },
            'feature_importance': {
                'definition': 'Measure of how much each feature contributes to model predictions',
                'in_trading': 'Identifies which indicators are most useful for predicting price movements',
                'interpretation': 'Higher values mean feature has greater impact on predictions',
                'use_cases': [
                    'Feature selection',
                    'Understanding model decisions',
                    'Strategy development',
                    'Risk assessment'
                ],
                'example': 'If RSI has 40% importance, it\'s the strongest predictor in your model'
            },
            'train_test_split': {
                'definition': 'Dividing data into training and testing sets to evaluate model performance',
                'in_trading': 'Train on historical data, test on recent unseen data',
                'time_series_warning': 'Must use chronological split, not random split',
                'best_practices': [
                    'Use 70-80% for training',
                    'Never test on training data',
                    'Walk-forward testing for time series',
                    'Keep test set representative'
                ],
                'example': 'Train on 2020-2023 data, test on 2024 data'
            }
        }
        
        return concepts.get(concept.lower(), {'error': 'Concept not found'})


# ============================================================================
# API INTEGRATION FOR ADVANCED FEATURES
# ============================================================================

from fastapi import APIRouter

advanced_router = APIRouter()

@advanced_router.post("/api/backtest")
async def run_backtest(
    ticker: str,
    model_config: Dict,
    start_date: str,
    end_date: str
):
    """Run backtest on ML strategy"""
    
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        # Generate predictions (simplified)
        predictions = []  # Would call ML prediction service
        
        # Run backtest
        engine = BacktestEngine()
        results = engine.backtest_ml_strategy(
            ticker=ticker,
            predictions=predictions,
            actual_prices=hist['Close']
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {"error": str(e)}

@advanced_router.get("/api/feature-analysis/{ticker}")
async def analyze_features(ticker: str, features: List[str]):
    """Analyze feature importance and stability"""
    
    try:
        analyzer = FeatureImportanceAnalyzer()
        results = analyzer.analyze_feature_stability(ticker, features)
        return results
        
    except Exception as e:
        logger.error(f"Feature analysis error: {e}")
        return {"error": str(e)}

@advanced_router.post("/api/portfolio/optimize")
async def optimize_portfolio(tickers: List[str], weights: Optional[List[float]] = None):
    """Portfolio optimization"""
    
    try:
        optimizer = PortfolioOptimizer()
        
        if weights:
            # Calculate metrics for given weights
            metrics = optimizer.calculate_portfolio_metrics(tickers, weights)
            return {"portfolio": metrics}
        else:
            # Generate efficient frontier
            frontier = optimizer.efficient_frontier(tickers)
            return {"efficient_frontier": frontier}
            
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        return {"error": str(e)}

@advanced_router.get("/api/education/indicator/{indicator}")
async def get_indicator_explanation(indicator: str):
    """Get educational content about indicator"""
    
    generator = EducationalContentGenerator()
    return generator.explain_indicator(indicator)

@advanced_router.get("/api/education/ml/{concept}")
async def get_ml_concept_explanation(concept: str):
    """Get educational content about ML concept"""
    
    generator = EducationalContentGenerator()
    return generator.explain_ml_concept(concept)


# ============================================================================
# TESTING UTILITIES
# ============================================================================

import pytest
from fastapi.testclient import TestClient

class TestTradeLens:
    """
    Comprehensive test suite for TradeLens
    """
    
    @pytest.fixture
    def client(self):
        from main import app
        return TestClient(app)
    
    def test_market_overview(self, client):
        """Test market overview endpoint"""
        response = client.get("/api/market/overview")
        assert response.status_code == 200
        data = response.json()
        assert 'indices' in data
        assert len(data['indices']) > 0
    
    def test_stock_price(self, client):
        """Test stock price endpoint"""
        response = client.get("/api/stock/AAPL/price?range=1m")
        assert response.status_code == 200
        data = response.json()
        assert data['ticker'] == 'AAPL'
        assert 'data' in data
        assert len(data['data']) > 0
    
    def test_ml_prediction(self, client):
        """Test ML prediction endpoint"""
        payload = {
            "ticker": "AAPL",
            "model_type": "random_forest",
            "features": ["ma_20", "rsi", "volume"],
            "prediction_window": "1w"
        }
        response = client.post("/api/ml/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert 'predicted_price' in data
        assert 'metrics' in data
    
    def test_invalid_ticker(self, client):
        """Test handling of invalid ticker"""
        response = client.get("/api/stock/INVALID123/price")
        assert response.status_code in [404, 500]
    
    def test_feature_importance_analysis(self):
        """Test feature importance analyzer"""
        analyzer = FeatureImportanceAnalyzer()
        features = ['ma_20', 'rsi', 'macd']
        
        results = analyzer.analyze_feature_stability('AAPL', features)
        
        assert 'feature_importance' in results
        assert all(f in results['feature_importance'] for f in features)
    
    def test_backtest_engine(self):
        """Test backtesting engine"""
        engine = BacktestEngine(initial_capital=100000)
        
        # Create mock data
        dates = pd.date_range('2024-01-01', periods=100)
        prices = pd.Series(150 + np.random.randn(100).cumsum(), index=dates)
        
        predictions = [
            {'predicted_change_percent': 3.0} if i % 10 < 5 else {'predicted_change_percent': -3.0}
            for i in range(100)
        ]
        
        results = engine.backtest_ml_strategy('TEST', predictions, prices)
        
        assert 'total_return' in results
        assert 'num_trades' in results
    
    def test_risk_metrics(self):
        """Test risk management calculations"""
        # Create mock returns
        returns = pd.Series(np.random.randn(252) * 0.02)
        
        var = RiskManager.calculate_var(returns, 0.95)
        cvar = RiskManager.calculate_cvar(returns, 0.95)
        
        assert var < 0  # VaR should be negative
        assert cvar < var  # CVaR should be more negative than VaR
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization"""
        optimizer = PortfolioOptimizer()
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        weights = [0.4, 0.3, 0.3]
        
        metrics = optimizer.calculate_portfolio_metrics(tickers, weights)
        
        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
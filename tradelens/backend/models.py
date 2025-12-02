# models.py - Complete Database Models for TradeLens
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, ForeignKey, Text, Date
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from database import Base

# ============================================================================
# USER MODELS
# ============================================================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    learning_sessions = relationship("LearningSession", back_populates="user", cascade="all, delete-orphan")
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")

# ============================================================================
# WATCHLIST MODELS
# ============================================================================

class Watchlist(Base):
    __tablename__ = "watchlists"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(10), nullable=False)
    name = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    alert_price_above = Column(Float, nullable=True)
    alert_price_below = Column(Float, nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="watchlists")
    
    # Composite index for fast lookups
    __table_args__ = (
        {'postgresql_ignore_search_path': True},
    )

# ============================================================================
# SENTIMENT CACHE MODELS
# ============================================================================

class SentimentCache(Base):
    __tablename__ = "sentiment_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=True)  # Compound score
    positive_score = Column(Float, nullable=True)
    negative_score = Column(Float, nullable=True)
    neutral_score = Column(Float, nullable=True)
    news_count = Column(Integer, default=0)
    headlines = Column(JSON, nullable=True)  # Array of headline objects
    overall_sentiment = Column(String(20), nullable=True)  # 'bullish', 'bearish', 'neutral'
    cached_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        {'postgresql_ignore_search_path': True},
    )

# ============================================================================
# ML PREDICTION MODELS
# ============================================================================

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    ticker = Column(String(10), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # 'random_forest', 'linear', 'lstm'
    prediction_window = Column(String(10), nullable=False)  # '1d', '1w', '1m'
    
    # Input data
    features_used = Column(JSON, nullable=False)  # Array of feature names
    current_price = Column(Float, nullable=False)
    
    # Prediction output
    predicted_price = Column(Float, nullable=False)
    predicted_change_percent = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=True)
    
    # Model performance metrics
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    directional_accuracy = Column(Float, nullable=True)
    
    # Feature importance
    feature_importance = Column(JSON, nullable=True)
    
    # Actual outcome (filled in later)
    actual_price = Column(Float, nullable=True)
    actual_change_percent = Column(Float, nullable=True)
    prediction_correct = Column(Boolean, nullable=True)
    
    # Metadata
    training_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    outcome_date = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="predictions")

# ============================================================================
# LEARNING SESSION MODELS
# ============================================================================

class LearningSession(Base):
    __tablename__ = "learning_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_type = Column(String(50), nullable=False)  # 'chat', 'ml_experiment', 'analysis', 'backtest'
    
    # Session content
    ticker = Column(String(10), nullable=True)
    content = Column(JSON, nullable=False)  # Flexible storage for different session types
    
    # For chat sessions
    messages = Column(JSON, nullable=True)  # Array of chat messages
    
    # For ML experiments
    experiment_config = Column(JSON, nullable=True)
    experiment_results = Column(JSON, nullable=True)
    
    # Metadata
    duration_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="learning_sessions")

# ============================================================================
# PORTFOLIO MODELS
# ============================================================================

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    initial_capital = Column(Float, default=100000.0)
    current_value = Column(Float, default=100000.0)
    cash_balance = Column(Float, default=100000.0)
    is_paper_trading = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")

# ============================================================================
# POSITION MODELS (for paper trading)
# ============================================================================

class Position(Base):
    __tablename__ = "positions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(10), nullable=False)
    shares = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    market_value = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    unrealized_pnl_percent = Column(Float, nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")

# ============================================================================
# TRANSACTION MODELS (for paper trading)
# ============================================================================

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(String(10), nullable=False)
    transaction_type = Column(String(10), nullable=False)  # 'buy', 'sell'
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")

# ============================================================================
# BACKTEST RESULTS MODELS
# ============================================================================

class BacktestResult(Base):
    __tablename__ = "backtest_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    name = Column(String(255), nullable=False)
    ticker = Column(String(10), nullable=False)
    
    # Strategy configuration
    strategy_config = Column(JSON, nullable=False)
    model_type = Column(String(50), nullable=True)
    features_used = Column(JSON, nullable=True)
    
    # Date range
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    
    # Performance metrics
    initial_capital = Column(Float, nullable=False)
    final_value = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    total_return_percent = Column(Float, nullable=False)
    
    # Benchmark comparison
    benchmark_return = Column(Float, nullable=True)
    benchmark_return_percent = Column(Float, nullable=True)
    alpha = Column(Float, nullable=True)
    
    # Risk metrics
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    
    # Trading statistics
    num_trades = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    avg_win = Column(Float, nullable=True)
    avg_loss = Column(Float, nullable=True)
    
    # Detailed results
    trades = Column(JSON, nullable=True)  # Array of trade objects
    portfolio_history = Column(JSON, nullable=True)  # Time series of portfolio values
    
    created_at = Column(DateTime, default=datetime.utcnow)

# ============================================================================
# NEWS ARTICLES CACHE (for faster retrieval)
# ============================================================================

class NewsArticle(Base):
    __tablename__ = "news_articles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    publisher = Column(String(255), nullable=True)
    link = Column(String(1000), nullable=True, unique=True)
    thumbnail = Column(String(1000), nullable=True)
    published_at = Column(DateTime, nullable=False, index=True)
    
    # Sentiment analysis
    sentiment_positive = Column(Float, nullable=True)
    sentiment_negative = Column(Float, nullable=True)
    sentiment_neutral = Column(Float, nullable=True)
    sentiment_compound = Column(Float, nullable=True)
    
    # Content
    summary = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)
    
    # Metadata
    fetched_at = Column(DateTime, default=datetime.utcnow)
    analyzed_at = Column(DateTime, nullable=True)

# ============================================================================
# USER PREFERENCES
# ============================================================================

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    # Display preferences
    default_chart_range = Column(String(10), default="1m")
    theme = Column(String(20), default="light")
    
    # Notification preferences
    email_notifications = Column(Boolean, default=True)
    price_alerts = Column(Boolean, default=True)
    ml_prediction_alerts = Column(Boolean, default=False)
    
    # Default settings
    default_model_type = Column(String(50), default="random_forest")
    default_features = Column(JSON, nullable=True)
    
    # Privacy
    share_predictions = Column(Boolean, default=False)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ============================================================================
# INITIALIZATION SCRIPT
# ============================================================================

if __name__ == "__main__":
    from database import engine, init_db
    
    print("Creating all database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    # Print all tables
    print("\n📊 Created tables:")
    for table in Base.metadata.sorted_tables:
        print(f"  - {table.name}")
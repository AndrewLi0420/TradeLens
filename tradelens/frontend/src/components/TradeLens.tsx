import React, { useState, useEffect, useCallback } from 'react';
import Image from 'next/image';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, Search, Brain, Activity, BarChart3, Sparkles, RefreshCw, Loader2, AlertCircle, ExternalLink, ChevronRight, Zap, Clock, Target, Layers, Radio } from 'lucide-react';
import { aiAPI } from '@/lib/api';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types
interface MarketIndex {
  name: string;
  ticker: string;
  price: number;
  change: number;
  change_percent: number;
}

interface PriceData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface NewsArticle {
  title: string;
  publisher: string;
  link: string;
  published_at: string;
  thumbnail: string;
}

interface Fundamentals {
  ticker: string;
  name: string;
  sector: string;
  industry: string;
  market_cap: number;
  pe_ratio: number | null;
  beta: number | null;
  eps: number | null;
  dividend_yield: number | null;
  '52w_high': number | null;
  '52w_low': number | null;
  description: string;
}

interface Prediction {
  ticker: string;
  current_price: number;
  predicted_price: number;
  prediction_change: number;
  metrics: {
    rmse: number;
    mae: number;
    directional_accuracy: number;
  };
  feature_importance: Record<string, number> | null;
  features_used: string[];
}

// Synthetic news generator based on ticker
const generateSyntheticNews = (ticker: string): NewsArticle[] => {
  const now = new Date();
  const companyNames: Record<string, string> = {
    'AAPL': 'Apple',
    'TSLA': 'Tesla',
    'NVDA': 'NVIDIA',
    'MSFT': 'Microsoft',
    'GOOGL': 'Google',
    'AMZN': 'Amazon',
    'META': 'Meta',
  };
  const company = companyNames[ticker] || ticker;
  
  const newsTemplates = [
    {
      title: `${company} Reports Strong Q4 Earnings, Beats Analyst Expectations`,
      publisher: 'MarketWatch',
      daysAgo: 0,
    },
    {
      title: `Analysts Upgrade ${ticker} Price Target Amid Growing AI Demand`,
      publisher: 'Bloomberg',
      daysAgo: 1,
    },
    {
      title: `${company} Announces New Strategic Partnership in Cloud Computing`,
      publisher: 'Reuters',
      daysAgo: 2,
    },
    {
      title: `Institutional Investors Increase Holdings in ${ticker} Stock`,
      publisher: 'Yahoo Finance',
      daysAgo: 3,
    },
    {
      title: `${company} CEO Discusses Growth Strategy at Industry Conference`,
      publisher: 'CNBC',
      daysAgo: 4,
    },
  ];

  return newsTemplates.map((template, i) => {
    const publishDate = new Date(now);
    publishDate.setDate(publishDate.getDate() - template.daysAgo);
    return {
      title: template.title,
      publisher: template.publisher,
      link: `#synthetic-${ticker}-${i}`,
      published_at: publishDate.toISOString(),
      thumbnail: '',
    };
  });
};


// Synthetic weekly sentiment data generator
const generateWeeklySentiment = (ticker: string) => {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const seed = ticker.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  
  return days.map((day, i) => {
    const basePositive = 35 + ((seed * (i + 1)) % 30);
    const baseNegative = 15 + ((seed * (i + 2)) % 20);
    const neutral = 100 - basePositive - baseNegative;
    
    return {
      day,
      positive: Math.round(basePositive),
      neutral: Math.round(neutral),
      negative: Math.round(baseNegative),
    };
  });
};

const TradeLens = () => {
  const [activeView, setActiveView] = useState('dashboard');
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState('1m');
  const [searchQuery, setSearchQuery] = useState('');
  const [watchlist, setWatchlist] = useState(['AAPL', 'TSLA', 'NVDA', 'MSFT']);
  
  // API Data States
  const [marketIndices, setMarketIndices] = useState<MarketIndex[]>([]);
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [news, setNews] = useState<NewsArticle[]>([]);
  const [fundamentals, setFundamentals] = useState<Fundamentals | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [watchlistPrices, setWatchlistPrices] = useState<Record<string, { price: number; change: number }>>({});
  const [watchlistFundamentals, setWatchlistFundamentals] = useState<Record<string, { sector: string; industry: string }>>({});
  
  // Loading States
  const [loadingMarket, setLoadingMarket] = useState(true);
  const [loadingPrice, setLoadingPrice] = useState(false);
  const [loadingNews, setLoadingNews] = useState(false);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loadingWatchlist, setLoadingWatchlist] = useState(true);
  
  // Error States
  const [error, setError] = useState<string | null>(null);

  const timeRanges = [
    { label: '1D', value: '1d' },
    { label: '5D', value: '5d' },
    { label: '1M', value: '1m' },
    { label: '3M', value: '3m' },
    { label: '6M', value: '6m' },
    { label: '1Y', value: '1y' },
    { label: '5Y', value: '5y' },
    { label: 'MAX', value: 'max' }
  ];

  // Fetch market overview
  const fetchMarketOverview = useCallback(async () => {
    setLoadingMarket(true);
    try {
      const response = await fetch(`${API_BASE}/api/market/overview`);
      if (!response.ok) throw new Error('Failed to fetch market data');
      const data = await response.json();
      setMarketIndices(data.indices || []);
      setError(null);
    } catch (err) {
      console.error('Market overview error:', err);
      setError('Failed to load market data');
    } finally {
      setLoadingMarket(false);
    }
  }, []);

  // Fetch stock price data
  const fetchStockPrice = useCallback(async (ticker: string, range: string) => {
    if (!ticker) return;
    setLoadingPrice(true);
    try {
      const response = await fetch(`${API_BASE}/api/stock/${ticker}/price?range=${range}`);
      if (!response.ok) throw new Error('Failed to fetch price data');
      const data = await response.json();
      setPriceData(data.data || []);
      setCurrentPrice(data.current_price);
      setError(null);
    } catch (err) {
      console.error('Price fetch error:', err);
      setError(`Failed to load price data for ${ticker}`);
    } finally {
      setLoadingPrice(false);
    }
  }, []);

  // Fetch news for a stock
  const fetchNews = useCallback(async (ticker: string) => {
    if (!ticker) return;
    setLoadingNews(true);
    try {
      const response = await fetch(`${API_BASE}/api/news/${ticker}?limit=5`);
      if (!response.ok) throw new Error('Failed to fetch news');
      const data = await response.json();
      setNews(data.news || []);
    } catch (err) {
      console.error('News fetch error:', err);
    } finally {
      setLoadingNews(false);
    }
  }, []);

  // Fetch fundamentals
  const fetchFundamentals = useCallback(async (ticker: string) => {
    if (!ticker) return;
    try {
      const response = await fetch(`${API_BASE}/api/stock/${ticker}/fundamentals`);
      if (!response.ok) throw new Error('Failed to fetch fundamentals');
      const data = await response.json();
      setFundamentals(data);
    } catch (err) {
      console.error('Fundamentals fetch error:', err);
    }
  }, []);

  // Fetch watchlist prices
  const fetchWatchlistPrices = useCallback(async () => {
    setLoadingWatchlist(true);
    const prices: Record<string, { price: number; change: number }> = {};
    
    await Promise.all(
      watchlist.map(async (ticker) => {
        try {
          const response = await fetch(`${API_BASE}/api/stock/${ticker}/price?range=1d`);
          if (response.ok) {
            const data = await response.json();
            const priceHistory = data.data || [];
            if (priceHistory.length >= 2) {
              const current = priceHistory[priceHistory.length - 1]?.close || data.current_price;
              const previous = priceHistory[0]?.close || current;
              const change = ((current - previous) / previous) * 100;
              prices[ticker] = { price: current, change };
            } else if (data.current_price) {
              prices[ticker] = { price: data.current_price, change: 0 };
            }
          }
        } catch (err) {
          console.error(`Failed to fetch ${ticker}:`, err);
        }
      })
    );
    
    setWatchlistPrices(prices);
    setLoadingWatchlist(false);
  }, [watchlist]);

  // Fetch watchlist fundamentals
  const fetchWatchlistFundamentals = useCallback(async () => {
    const results = await Promise.all(
      watchlist.map(async (ticker) => {
        try {
          const response = await fetch(`${API_BASE}/api/stock/${ticker}/fundamentals`);
          if (response.ok) {
            const data = await response.json();
            return { 
              ticker, 
              sector: data.sector || 'Unknown', 
              industry: data.industry || 'Unknown' 
            };
          }
        } catch (err) {
          console.error(`Failed to fetch fundamentals for ${ticker}:`, err);
        }
        return null;
      })
    );
    
    const newData: Record<string, { sector: string; industry: string }> = {};
    results.forEach(result => {
      if (result) {
        newData[result.ticker] = { sector: result.sector, industry: result.industry };
      }
    });
    
    if (Object.keys(newData).length > 0) {
      setWatchlistFundamentals(prev => ({ ...prev, ...newData }));
    }
  }, [watchlist]);

  // Run ML prediction
  const runPrediction = async (ticker: string, modelType: string, features: string[], window: string) => {
    setLoadingPrediction(true);
    setPrediction(null);
    try {
      const response = await fetch(`${API_BASE}/api/ml/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker,
          model_type: modelType,
          features,
          prediction_window: window
        })
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }
      const data = await response.json();
      setPrediction(data);
    } catch (err: any) {
      console.error('Prediction error:', err);
      setError(err.message || 'ML Prediction failed');
    } finally {
      setLoadingPrediction(false);
    }
  };

  // Initial load
  useEffect(() => {
    fetchMarketOverview();
    fetchWatchlistPrices();
    fetchWatchlistFundamentals();
    
    const interval = setInterval(fetchMarketOverview, 60000);
    return () => clearInterval(interval);
  }, [fetchMarketOverview, fetchWatchlistPrices, fetchWatchlistFundamentals]);

  // Fetch stock data when selection changes
  useEffect(() => {
    if (selectedStock) {
      fetchStockPrice(selectedStock, timeRange);
      fetchNews(selectedStock);
      fetchFundamentals(selectedStock);
    }
  }, [selectedStock, timeRange, fetchStockPrice, fetchNews, fetchFundamentals]);

  // Handle search
  const handleSearch = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && searchQuery.trim()) {
      setSelectedStock(searchQuery.toUpperCase().trim());
      setActiveView('stock-detail');
      setSearchQuery('');
    }
  };

  // Format large numbers
  const formatNumber = (num: number | null | undefined): string => {
    if (num === null || num === undefined) return 'N/A';
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    return `$${num.toFixed(2)}`;
  };

  // Format compact numbers
  const formatCompact = (num: number | null | undefined): string => {
    if (num === null || num === undefined) return '--';
    if (num >= 1e12) return `${(num / 1e12).toFixed(1)}T`;
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    return num.toFixed(2);
  };

  // Market Overview Component - Compact Ticker Strip
  const MarketOverview = () => (
    <div className="mb-4">
      <div className="flex items-center gap-2 overflow-x-auto pb-2 tl-scroll-area">
        {loadingMarket ? (
          Array(4).fill(0).map((_, i) => (
            <div key={i} className="flex-shrink-0 tl-card-compact animate-pulse min-w-[160px]">
              <div className="h-3 bg-tl-surface-overlay rounded w-16 mb-2" />
              <div className="h-5 bg-tl-surface-overlay rounded w-20 mb-1" />
              <div className="h-3 bg-tl-surface-overlay rounded w-12" />
            </div>
          ))
        ) : marketIndices.length > 0 ? (
          marketIndices.map((index) => {
            // Use the actual index ticker directly (e.g., ^GSPC, ^IXIC) for accurate index prices
            const indexTicker = index.ticker;
            return (
              <div 
                key={index.name} 
                className="flex-shrink-0 tl-card-compact min-w-[160px] hover:border-tl-smart transition-colors cursor-pointer"
                onClick={() => {
                  if (indexTicker) {
                    setSelectedStock(indexTicker);
                    setActiveView('stock-detail');
                  }
                }}
                title={indexTicker ? `Click to view ${index.name} chart` : undefined}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xxs uppercase tracking-wide text-tl-tertiary font-medium">{index.name}</span>
                  <div className={`tl-indicator ${index.change_percent >= 0 ? 'tl-indicator-positive' : 'tl-indicator-negative'}`} />
                </div>
                <div className="font-mono text-data-lg font-bold text-tl-smoke">
                  {index.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div className={`flex items-center gap-1 font-mono text-data-xs font-medium ${index.change_percent >= 0 ? 'text-tl-positive' : 'text-tl-negative'}`}>
                  {index.change_percent >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  <span>{index.change_percent >= 0 ? '+' : ''}{index.change_percent?.toFixed(2)}%</span>
                </div>
              </div>
            );
          })
        ) : (
          <div className="flex-1 flex items-center justify-center py-4 text-tl-tertiary">
            <AlertCircle className="w-4 h-4 mr-2" />
            <span className="text-data-xs">Unable to load market data</span>
          </div>
        )}
      </div>
    </div>
  );

  // Watchlist Component - High Density
  const Watchlist = () => {
    const [isAddingStock, setIsAddingStock] = useState(false);
    const [newStockTicker, setNewStockTicker] = useState('');
    const [addStockError, setAddStockError] = useState<string | null>(null);

    const handleAddStock = async (inputValue?: string) => {
      const ticker = (inputValue || newStockTicker).toUpperCase().trim();
      
      if (!ticker) {
        setAddStockError('Enter ticker');
        return;
      }
      
      if (watchlist.includes(ticker)) {
        setAddStockError('Already added');
        return;
      }

      try {
        const response = await fetch(`${API_BASE}/api/stock/${ticker}/price?range=1d`);
        if (!response.ok) {
          setAddStockError('Invalid ticker');
          return;
        }
        
        setWatchlist([...watchlist, ticker]);
        setNewStockTicker('');
        setIsAddingStock(false);
        setAddStockError(null);
        
        fetchWatchlistPrices();
        fetchWatchlistFundamentals();
      } catch (err) {
        setAddStockError('Failed to validate');
      }
    };

    const handleRemoveStock = (tickerToRemove: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setWatchlist(watchlist.filter(t => t !== tickerToRemove));
    };

    return (
      <div className="tl-panel h-full flex flex-col">
        <div className="tl-panel-header flex-shrink-0">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-tl-smart" />
            <h3 className="text-data-sm font-semibold text-tl-smoke">Watchlist</h3>
            <span className="tl-badge tl-badge-neutral">{watchlist.length}</span>
          </div>
          <button 
            onClick={fetchWatchlistPrices} 
            className="tl-btn-icon"
            disabled={loadingWatchlist}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingWatchlist ? 'animate-spin' : ''}`} />
          </button>
        </div>
        
        <div className="tl-panel-body p-0 flex-1 flex flex-col">
          <div className="divide-y divide-tl-border-subtle flex-1 overflow-y-auto tl-scroll-area">
            {watchlist.map((ticker) => {
              const priceInfo = watchlistPrices[ticker];
              return (
                <div
                  key={ticker}
                  onClick={() => {
                    setSelectedStock(ticker);
                    setActiveView('stock-detail');
                  }}
                  className="tl-ticker-row px-3 py-2.5 group cursor-pointer"
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <div className="tl-ticker-badge flex-shrink-0">
                      {ticker[0]}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-data-sm text-tl-smoke">{ticker}</span>
                        <ChevronRight className="w-3 h-3 text-tl-muted opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                      <div className="text-xxs text-tl-tertiary truncate">
                        {watchlistFundamentals[ticker]?.sector || 'Loading...'}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="text-right">
                      {loadingWatchlist && !priceInfo ? (
                        <div className="w-16 h-8 tl-skeleton rounded" />
                      ) : priceInfo ? (
                        <>
                          <div className="font-mono text-data-sm font-semibold text-tl-smoke">
                            ${priceInfo.price.toFixed(2)}
                          </div>
                          <div className={`font-mono text-data-xs font-medium flex items-center justify-end gap-0.5 ${priceInfo.change >= 0 ? 'text-tl-positive' : 'text-tl-negative'}`}>
                            {priceInfo.change >= 0 ? <TrendingUp className="w-2.5 h-2.5" /> : <TrendingDown className="w-2.5 h-2.5" />}
                            {priceInfo.change >= 0 ? '+' : ''}{priceInfo.change.toFixed(2)}%
                          </div>
                        </>
                      ) : (
                        <span className="text-tl-muted text-data-xs">--</span>
                      )}
                    </div>
                    <button
                      onClick={(e) => handleRemoveStock(ticker, e)}
                      className="opacity-0 group-hover:opacity-100 tl-btn-icon p-1 hover:bg-tl-negative-muted hover:text-tl-negative transition-all"
                      title="Remove"
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
          
          <div className="p-3 border-t border-tl-border-subtle">
            {isAddingStock ? (
              <div className="space-y-2">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newStockTicker}
                    onChange={(e) => {
                      setNewStockTicker(e.target.value.toUpperCase());
                      setAddStockError(null);
                    }}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handleAddStock((e.target as HTMLInputElement).value);
                      }
                    }}
                    placeholder="TICKER"
                    className="tl-input-compact flex-1 uppercase"
                    autoFocus
                  />
                  <button onClick={() => handleAddStock()} className="tl-btn-primary tl-btn-compact">
                    Add
                  </button>
                  <button
                    onClick={() => {
                      setIsAddingStock(false);
                      setNewStockTicker('');
                      setAddStockError(null);
                    }}
                    className="tl-btn-ghost tl-btn-compact"
                  >
                    ×
                  </button>
                </div>
                {addStockError && (
                  <p className="text-xxs text-tl-negative flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    {addStockError}
                  </p>
                )}
              </div>
            ) : (
              <button 
                onClick={() => setIsAddingStock(true)}
                className="w-full py-2 border border-dashed border-tl-border rounded-md text-tl-tertiary text-data-xs font-medium hover:border-tl-smart hover:text-tl-smart hover:bg-tl-surface-overlay transition-all"
              >
                + Add Symbol
              </button>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Stock Chart Component - Enhanced
  const StockChart = () => {
    const priceChange = priceData.length >= 2 
      ? ((priceData[priceData.length - 1]?.close - priceData[0]?.close) / priceData[0]?.close) * 100 
      : 0;
    const isPositive = priceChange >= 0;
    
    return (
      <div className="tl-panel">
        <div className="tl-panel-header">
          <div className="flex items-center gap-4">
            <div>
              <div className="flex items-center gap-2">
                {selectedStock && <div className="tl-ticker-badge text-xs">{selectedStock[0]}</div>}
                <h3 className="font-mono text-data-lg font-bold text-tl-smoke">
                  {selectedStock || 'Select Symbol'}
                </h3>
              </div>
              {currentPrice && (
                <div className="flex items-center gap-3 mt-1">
                  <span className="font-mono text-data-2xl font-bold text-tl-smoke">
                    ${currentPrice.toFixed(2)}
                  </span>
                  <span className={`tl-badge ${isPositive ? 'tl-badge-positive' : 'tl-badge-negative'}`}>
                    {isPositive ? '+' : ''}{priceChange.toFixed(2)}%
                  </span>
                </div>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-1">
            {timeRanges.map((range) => (
              <button
                key={range.value}
                onClick={() => setTimeRange(range.value)}
                className={`tl-time-btn ${
                  timeRange === range.value ? 'tl-time-btn-active' : 'tl-time-btn-inactive'
                }`}
              >
                {range.label}
              </button>
            ))}
          </div>
        </div>
        
        <div className="tl-panel-body p-4">
          {loadingPrice ? (
            <div className="h-[280px] flex items-center justify-center">
              <Loader2 className="w-6 h-6 animate-spin text-tl-smart" />
            </div>
          ) : priceData.length > 0 ? (
            <div className="tl-chart-container tl-chart-grid">
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={priceData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={isPositive ? '#00C896' : '#FF4757'} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={isPositive ? '#00C896' : '#FF4757'} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2E3448" />
                  <XAxis
                    dataKey="time"
                    tickFormatter={(time) => {
                      const date = new Date(time * 1000);
                      return timeRange === '1d' 
                        ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                        : date.toLocaleDateString([], { month: 'short', day: 'numeric' });
                    }}
                    stroke="#6B7280"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis 
                    stroke="#6B7280" 
                    fontSize={10}
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(val) => `$${val.toFixed(0)}`}
                    tickLine={false}
                    axisLine={false}
                    width={50}
                  />
                  <Tooltip
                    contentStyle={{ 
                      background: '#252A40', 
                      border: '1px solid #3A4060', 
                      borderRadius: '6px',
                      boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)'
                    }}
                    labelStyle={{ color: '#A8AEBF', fontSize: '11px', fontFamily: 'JetBrains Mono' }}
                    itemStyle={{ color: '#F5F3F5', fontSize: '12px', fontFamily: 'JetBrains Mono' }}
                    labelFormatter={(time) => new Date(time * 1000).toLocaleString()}
                    formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="close" 
                    stroke={isPositive ? '#00C896' : '#FF4757'}
                    strokeWidth={2}
                    fill="url(#chartGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-[280px] flex items-center justify-center text-tl-tertiary">
              <div className="text-center">
                <BarChart3 className="w-10 h-10 mx-auto mb-2 opacity-50" />
                <p className="text-data-sm">Search for a symbol to view chart</p>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // News Panel Component - Compact
  const NewsPanel = () => {
    const displayNews = news.length > 0 ? news : (selectedStock ? generateSyntheticNews(selectedStock) : []);
    const isSynthetic = news.length === 0;

    return (
      <div className="tl-panel h-full flex flex-col">
        <div className="tl-panel-header flex-shrink-0">
          <div className="flex items-center gap-2">
            <Radio className="w-4 h-4 text-tl-smart" />
            <h3 className="text-data-sm font-semibold text-tl-smoke">News Feed</h3>
            {isSynthetic && displayNews.length > 0 && (
              <span className="tl-badge tl-badge-warning">Sample</span>
            )}
          </div>
        </div>
        
        <div className="tl-panel-body p-0 tl-scroll-area flex-1 overflow-y-auto">
          {loadingNews ? (
            <div className="p-3 space-y-3">
              {Array(3).fill(0).map((_, i) => (
                <div key={i} className="space-y-2">
                  <div className="h-3 bg-tl-surface-overlay rounded w-full" />
                  <div className="h-3 bg-tl-surface-overlay rounded w-2/3" />
                  <div className="h-2 bg-tl-surface-overlay rounded w-24" />
                </div>
              ))}
            </div>
          ) : displayNews.length > 0 ? (
            <div className="divide-y divide-tl-border-subtle">
              {displayNews.map((article, i) => (
                <a 
                  key={i} 
                  href={article.link} 
                  target={isSynthetic ? "_self" : "_blank"}
                  rel="noopener noreferrer"
                  className="block px-3 py-2.5 hover:bg-tl-surface-overlay transition-colors group"
                  onClick={(e) => isSynthetic && e.preventDefault()}
                >
                  <div className="text-data-xs font-medium text-tl-smoke group-hover:text-tl-smart transition-colors line-clamp-2">
                    {article.title}
                  </div>
                  <div className="flex items-center gap-2 mt-1.5 text-xxs text-tl-tertiary">
                    <span className="font-medium">{article.publisher}</span>
                    <span>•</span>
                    <span>{new Date(article.published_at).toLocaleDateString()}</span>
                    {!isSynthetic && (
                      <ExternalLink className="w-3 h-3 ml-auto opacity-0 group-hover:opacity-100 transition-opacity" />
                    )}
                  </div>
                </a>
              ))}
            </div>
          ) : (
            <div className="flex items-center justify-center py-8 text-tl-tertiary">
              <p className="text-data-xs">No news available</p>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Weekly Sentiment Chart - Compact
  const WeeklySentimentChart = () => {
    const sentimentData = selectedStock ? generateWeeklySentiment(selectedStock) : [];
    
    if (!selectedStock) return null;

    const avgPositive = sentimentData.reduce((acc, d) => acc + d.positive, 0) / sentimentData.length;
    const avgNegative = sentimentData.reduce((acc, d) => acc + d.negative, 0) / sentimentData.length;
    const sentimentScore = Math.round(avgPositive - avgNegative);
    const sentimentLabel = sentimentScore > 20 ? 'Bullish' : sentimentScore > 5 ? 'Positive' : sentimentScore > -5 ? 'Neutral' : sentimentScore > -20 ? 'Bearish' : 'Very Bearish';

    return (
      <div className="tl-panel">
        <div className="tl-panel-header">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-tl-smart" />
            <h3 className="text-data-sm font-semibold text-tl-smoke">Sentiment Analysis</h3>
            <span className="tl-badge tl-badge-warning">Sample</span>
          </div>
          <div className="flex items-center gap-2">
            <span className={`tl-badge ${sentimentScore > 5 ? 'tl-badge-positive' : sentimentScore < -5 ? 'tl-badge-negative' : 'tl-badge-neutral'}`}>
              {sentimentLabel}
            </span>
            <span className="font-mono text-data-xs text-tl-secondary">
              {sentimentScore > 0 ? '+' : ''}{sentimentScore}
            </span>
          </div>
        </div>

        <div className="tl-panel-body">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={sentimentData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }} barSize={40}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2E3448" vertical={false} />
              <XAxis 
                dataKey="day" 
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#6B7280', fontSize: 10 }}
              />
              <YAxis 
                axisLine={false}
                tickLine={false}
                tick={{ fill: '#6B7280', fontSize: 10 }}
                tickFormatter={(val) => `${val}%`}
              />
              <Tooltip
                contentStyle={{ 
                  background: '#252A40', 
                  border: '1px solid #3A4060', 
                  borderRadius: '6px',
                  fontSize: '11px'
                }}
                formatter={(value: number, name: string) => [
                  `${value}%`, 
                  name.charAt(0).toUpperCase() + name.slice(1)
                ]}
              />
              <Bar dataKey="positive" stackId="sentiment" fill="#00C896" radius={[0, 0, 0, 0]} />
              <Bar dataKey="neutral" stackId="sentiment" fill="#576CA8" radius={[0, 0, 0, 0]} />
              <Bar dataKey="negative" stackId="sentiment" fill="#FF4757" radius={[0, 0, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-tl-border-subtle">
            <div className="tl-metric-card">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 bg-tl-positive rounded-sm" />
                <span className="tl-metric-label">Positive</span>
              </div>
              <div className="font-mono text-data-base font-semibold text-tl-positive">{Math.round(avgPositive)}%</div>
            </div>
            <div className="tl-metric-card">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 bg-tl-smart rounded-sm" />
                <span className="tl-metric-label">Neutral</span>
              </div>
              <div className="font-mono text-data-base font-semibold text-tl-secondary">{Math.round(100 - avgPositive - avgNegative)}%</div>
            </div>
            <div className="tl-metric-card">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 bg-tl-negative rounded-sm" />
                <span className="tl-metric-label">Negative</span>
              </div>
              <div className="font-mono text-data-base font-semibold text-tl-negative">{Math.round(avgNegative)}%</div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Fundamentals Panel - High Density Grid
  const FundamentalsPanel = () => (
    <div className="tl-panel">
      <div className="tl-panel-header">
        <div className="flex items-center gap-2">
          <Target className="w-4 h-4 text-tl-smart" />
          <h3 className="text-data-sm font-semibold text-tl-smoke">Fundamentals</h3>
        </div>
      </div>
      
      <div className="tl-panel-body">
        {fundamentals && fundamentals.ticker === selectedStock ? (
          <div className="space-y-4">
            {/* Company Header */}
            <div className="pb-3 border-b border-tl-border-subtle">
              <div className="font-semibold text-data-base text-tl-smoke">{fundamentals.name}</div>
              <div className="flex items-center gap-2 mt-1">
                <span className="tl-badge tl-badge-neutral">{fundamentals.sector}</span>
                <span className="text-xxs text-tl-tertiary">{fundamentals.industry}</span>
              </div>
            </div>
            
            {/* Metrics Grid */}
            <div className="grid grid-cols-3 gap-2">
              <div className="tl-metric-card">
                <span className="tl-metric-label">Mkt Cap</span>
                <span className="tl-metric-value">{formatCompact(fundamentals.market_cap)}</span>
              </div>
              <div className="tl-metric-card">
                <span className="tl-metric-label">P/E</span>
                <span className="tl-metric-value">{fundamentals.pe_ratio?.toFixed(1) || '--'}</span>
              </div>
              <div className="tl-metric-card">
                <span className="tl-metric-label">EPS</span>
                <span className="tl-metric-value">${fundamentals.eps?.toFixed(2) || '--'}</span>
              </div>
              <div className="tl-metric-card">
                <span className="tl-metric-label">Beta</span>
                <span className="tl-metric-value">{fundamentals.beta?.toFixed(2) || '--'}</span>
              </div>
              <div className="tl-metric-card">
                <span className="tl-metric-label">52W High</span>
                <span className="tl-metric-value text-tl-positive">${fundamentals['52w_high']?.toFixed(0) || '--'}</span>
              </div>
              <div className="tl-metric-card">
                <span className="tl-metric-label">52W Low</span>
                <span className="tl-metric-value text-tl-negative">${fundamentals['52w_low']?.toFixed(0) || '--'}</span>
              </div>
            </div>
            
            {/* 52 Week Range Bar */}
            {fundamentals['52w_high'] && fundamentals['52w_low'] && currentPrice && (
              <div className="mt-3">
                <div className="flex items-center justify-between text-xxs text-tl-tertiary mb-1">
                  <span>${fundamentals['52w_low']?.toFixed(0)}</span>
                  <span className="text-tl-smoke font-mono font-medium">52W Range</span>
                  <span>${fundamentals['52w_high']?.toFixed(0)}</span>
                </div>
                <div className="relative h-2 bg-tl-surface rounded-full overflow-hidden">
                  <div 
                    className="absolute h-full bg-gradient-to-r from-tl-negative via-tl-smart to-tl-positive rounded-full"
                    style={{ width: '100%' }}
                  />
                  <div 
                    className="absolute w-1 h-4 bg-tl-smoke rounded-full -top-1 shadow-md"
                    style={{ 
                      left: `${Math.min(100, Math.max(0, ((currentPrice - fundamentals['52w_low']) / (fundamentals['52w_high'] - fundamentals['52w_low'])) * 100))}%`,
                      transform: 'translateX(-50%)'
                    }}
                  />
                </div>
              </div>
            )}
            
            {/* Description */}
            {fundamentals.description && (
              <div className="pt-3 border-t border-tl-border-subtle">
                <p className="text-data-xs text-tl-secondary line-clamp-3">{fundamentals.description}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center py-8 text-tl-tertiary">
            <Loader2 className="w-5 h-5 animate-spin mr-2" />
            <span className="text-data-xs">Loading fundamentals...</span>
          </div>
        )}
      </div>
    </div>
  );

  // ML Prediction Sandbox - Enhanced
  const MLPredictionSandbox = () => {
    const [selectedFeatures, setSelectedFeatures] = useState(['ma_20', 'rsi', 'volatility']);
    const [modelType, setModelType] = useState('random_forest');
    const [predictionWindow, setPredictionWindow] = useState('1w');
    const [predictionTicker, setPredictionTicker] = useState(selectedStock || 'AAPL');

    const availableFeatures = [
      { id: 'ma_5', name: '5D MA', category: 'trend' },
      { id: 'ma_20', name: '20D MA', category: 'trend' },
      { id: 'ma_50', name: '50D MA', category: 'trend' },
      { id: 'rsi', name: 'RSI', category: 'momentum' },
      { id: 'macd', name: 'MACD', category: 'momentum' },
      { id: 'macd_signal', name: 'Signal', category: 'momentum' },
      { id: 'volume_ratio', name: 'Vol Ratio', category: 'volume' },
      { id: 'volatility', name: 'Volatility', category: 'volatility' }
    ];

    const handleRunPrediction = () => {
      runPrediction(predictionTicker, modelType, selectedFeatures, predictionWindow);
    };

    return (
      <div className="tl-panel">
        <div className="tl-panel-header">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-tl-smart" />
            <h3 className="text-data-sm font-semibold text-tl-smoke">ML Prediction Sandbox</h3>
            <span className="tl-badge tl-badge-neutral">Educational</span>
          </div>
        </div>

        <div className="tl-panel-body">
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Configuration Panel */}
            <div className="space-y-4">
              <div>
                <label className="tl-metric-label mb-2 block">Symbol</label>
                <input
                  type="text"
                  value={predictionTicker}
                  onChange={(e) => setPredictionTicker(e.target.value.toUpperCase())}
                  className="tl-input font-mono uppercase"
                  placeholder="AAPL"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="tl-metric-label mb-2 block">Model</label>
                  <select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                    className="tl-select"
                  >
                    <option value="linear">Linear Reg.</option>
                    <option value="random_forest">Random Forest</option>
                  </select>
                </div>

                <div>
                  <label className="tl-metric-label mb-2 block">Window</label>
                  <select
                    value={predictionWindow}
                    onChange={(e) => setPredictionWindow(e.target.value)}
                    className="tl-select"
                  >
                    <option value="1d">1 Day</option>
                    <option value="3d">3 Days</option>
                    <option value="1w">1 Week</option>
                    <option value="1m">1 Month</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="tl-metric-label mb-2 block">Features ({selectedFeatures.length})</label>
                <div className="grid grid-cols-2 gap-1 p-2 bg-tl-surface rounded-md max-h-36 overflow-y-auto tl-scroll-area">
                  {availableFeatures.map((feature) => (
                    <label key={feature.id} className="flex items-center gap-2 p-1.5 rounded hover:bg-tl-surface-elevated cursor-pointer transition-colors">
                      <input
                        type="checkbox"
                        checked={selectedFeatures.includes(feature.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedFeatures([...selectedFeatures, feature.id]);
                          } else {
                            setSelectedFeatures(selectedFeatures.filter(f => f !== feature.id));
                          }
                        }}
                        className="w-3.5 h-3.5 rounded border-tl-border bg-tl-surface text-tl-smart focus:ring-tl-smart focus:ring-offset-0"
                      />
                      <span className="text-data-xs text-tl-secondary">{feature.name}</span>
                    </label>
                  ))}
                </div>
              </div>

              <button 
                onClick={handleRunPrediction}
                disabled={loadingPrediction || selectedFeatures.length === 0}
                className="w-full tl-btn-primary py-2.5 flex items-center justify-center gap-2"
              >
                {loadingPrediction ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    <span>Run Prediction</span>
                  </>
                )}
              </button>
            </div>

            {/* Results Panel */}
            <div>
              {prediction ? (
                <div className="space-y-4">
                  {/* Prediction Result */}
                  <div className="p-4 rounded-lg bg-gradient-to-br from-tl-surface to-tl-surface-elevated border border-tl-border-subtle">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xxs uppercase tracking-wide text-tl-tertiary">
                        Predicted ({prediction.ticker})
                      </span>
                      <Clock className="w-3.5 h-3.5 text-tl-muted" />
                    </div>
                    <div className="font-mono text-data-3xl font-bold text-tl-smoke">
                      ${prediction.predicted_price.toFixed(2)}
                    </div>
                    <div className={`flex items-center gap-2 mt-1 font-mono text-data-sm ${prediction.prediction_change >= 0 ? 'text-tl-positive' : 'text-tl-negative'}`}>
                      {prediction.prediction_change >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                      <span>{prediction.prediction_change >= 0 ? '+' : ''}{prediction.prediction_change.toFixed(2)}%</span>
                      <span className="text-tl-tertiary">from ${prediction.current_price.toFixed(2)}</span>
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-2 gap-2">
                    <div className="tl-metric-card">
                      <span className="tl-metric-label">RMSE</span>
                      <div className="flex items-center gap-2">
                        <span className="tl-metric-value">{prediction.metrics.rmse.toFixed(2)}</span>
                        <div className="flex-1 tl-progress">
                          <div 
                            className="tl-progress-bar tl-progress-positive" 
                            style={{ width: `${Math.max(0, 100 - prediction.metrics.rmse * 5)}%` }} 
                          />
                        </div>
                      </div>
                    </div>
                    <div className="tl-metric-card">
                      <span className="tl-metric-label">Dir. Accuracy</span>
                      <div className="flex items-center gap-2">
                        <span className="tl-metric-value">{prediction.metrics.directional_accuracy.toFixed(0)}%</span>
                        <div className="flex-1 tl-progress">
                          <div 
                            className="tl-progress-bar tl-progress-neutral" 
                            style={{ width: `${prediction.metrics.directional_accuracy}%` }} 
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Feature Importance */}
                  {prediction.feature_importance && (
                    <div className="p-3 bg-tl-surface rounded-md">
                      <span className="tl-metric-label mb-2 block">Feature Importance</span>
                      <div className="space-y-1.5">
                        {Object.entries(prediction.feature_importance)
                          .sort(([, a], [, b]) => b - a)
                          .slice(0, 5)
                          .map(([feature, importance]) => (
                            <div key={feature} className="flex items-center gap-2">
                              <span className="font-mono text-xxs text-tl-secondary w-16 truncate">{feature}</span>
                              <div className="flex-1 tl-progress h-1">
                                <div 
                                  className="h-full rounded-full bg-tl-smart" 
                                  style={{ width: `${importance * 100}%` }} 
                                />
                              </div>
                              <span className="font-mono text-xxs text-tl-tertiary w-10 text-right">
                                {(importance * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))
                        }
                      </div>
                    </div>
                  )}

                  {/* Disclaimer */}
                  <div className="flex items-start gap-2 p-3 bg-tl-warning-muted rounded-md border border-tl-warning/30">
                    <Sparkles className="w-4 h-4 text-tl-warning flex-shrink-0 mt-0.5" />
                    <p className="text-xxs text-tl-warning/90">
                      Educational tool only. ML predictions based on historical patterns should not be used for trading decisions.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full py-12 text-tl-tertiary">
                  <Brain className="w-12 h-12 mb-3 opacity-30" />
                  <p className="text-data-sm text-center">Configure model and click<br />"Run Prediction" to see results</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // AI Assistant - Compact Chat
  const AIAssistant = () => {
    const [messages, setMessages] = useState([
      { role: 'assistant', content: 'Hi! I\'m your AI trading tutor. Ask me anything about stocks, indicators, or trading strategies!' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = async () => {
      if (!input.trim() || isLoading) return;
      
      const userMessage = input;
      setInput('');
      
      // Add user message immediately
      const newMessages = [...messages, { role: 'user', content: userMessage }];
      setMessages(newMessages);
      
      // Add loading indicator
      setIsLoading(true);
      setMessages([...newMessages, { role: 'assistant', content: '...' }]);
      
      try {
        // Call the AI API with conversation history
        const response = await aiAPI.chat({
          message: userMessage,
          ticker: selectedTicker || undefined,
          context: selectedTicker ? { ticker: selectedTicker } : undefined
        });
        
        // Replace loading message with actual response
        setMessages([...newMessages, { role: 'assistant', content: response.data.response }]);
      } catch (error) {
        console.error('Error sending message:', error);
        setMessages([...newMessages, { 
          role: 'assistant', 
          content: 'Sorry, I\'m having trouble responding right now. Please make sure the API key is configured correctly.' 
        }]);
      } finally {
        setIsLoading(false);
      }
    };

    return (
      <div className="tl-panel h-full flex flex-col">
        <div className="tl-panel-header flex-shrink-0">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-tl-smart" />
            <h3 className="text-data-sm font-semibold text-tl-smoke">AI Trading Tutor</h3>
          </div>
        </div>
        
        <div className="tl-panel-body p-0 flex flex-col flex-1">
          <div className="flex-1 overflow-y-auto p-3 space-y-3 tl-scroll-area">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] px-3 py-2 rounded-lg text-data-xs ${
                  msg.role === 'user' 
                    ? 'bg-tl-french text-tl-smoke rounded-br-sm' 
                    : 'bg-tl-surface text-tl-secondary rounded-bl-sm'
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}
          </div>

          <div className="p-3 border-t border-tl-border-subtle">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
                placeholder="Ask about trading..."
                className="tl-input-compact flex-1"
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                className="tl-btn-primary tl-btn-compact"
                disabled={isLoading}
              >
                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Send'}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Learning Modules Panel
  const LearningModules = () => {
    const [selectedModule, setSelectedModule] = useState<string | null>(null);
    const [activeSection, setActiveSection] = useState(0);
    const [showQuiz, setShowQuiz] = useState(false);
    const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
    const [quizSubmitted, setQuizSubmitted] = useState(false);

    const moduleContent: Record<string, {
      title: string;
      duration: string;
      level: string;
      icon: any;
      overview: string;
      sections: { heading: string; content: string; keyPoints?: string[] }[];
      practicalTips: string[];
      quiz: { question: string; options: string[]; correct: number }[];
    }> = {
      'technical-analysis': {
        title: 'Technical Analysis Intro',
        duration: '30 min',
        level: 'Beginner',
        icon: BarChart3,
        overview: 'Technical analysis is the study of past market data, primarily price and volume, to forecast future price movements. Unlike fundamental analysis, which examines a company\'s financial health, technical analysis focuses solely on chart patterns and statistical indicators.',
        sections: [
          {
            heading: 'What is Technical Analysis?',
            content: 'Technical analysis is a trading discipline that evaluates investments by analyzing statistical trends gathered from trading activity, such as price movement and volume. Unlike fundamental analysts who attempt to evaluate a security\'s intrinsic value, technical analysts focus on patterns of price movements, trading signals, and various analytical charting tools.',
            keyPoints: [
              'Based on the premise that historical price movements predict future price behavior',
              'Uses charts and indicators to identify patterns',
              'Works on any timeframe from minutes to years',
              'Applicable to stocks, forex, commodities, and cryptocurrencies'
            ]
          },
          {
            heading: 'The Three Core Principles',
            content: 'Technical analysis is built on three fundamental assumptions that form the basis of all chart analysis and trading strategies.',
            keyPoints: [
              'Market Action Discounts Everything: All known information is already reflected in the price',
              'Prices Move in Trends: Once a trend is established, it\'s more likely to continue than reverse',
              'History Tends to Repeat Itself: Market psychology creates recognizable patterns over time'
            ]
          },
          {
            heading: 'Types of Charts',
            content: 'Charts are the foundation of technical analysis. The three most common types are line charts, bar charts, and candlestick charts. Each provides different levels of detail about price action.',
            keyPoints: [
              'Line Charts: Simple visualization connecting closing prices',
              'Bar Charts: Show open, high, low, and close (OHLC) for each period',
              'Candlestick Charts: Visual representation of OHLC with color-coded bodies',
              'Volume Charts: Display trading volume alongside price'
            ]
          },
          {
            heading: 'Support and Resistance',
            content: 'Support and resistance are key concepts in technical analysis. Support is a price level where buying pressure exceeds selling pressure, preventing further decline. Resistance is where selling pressure exceeds buying pressure, preventing further rise.',
            keyPoints: [
              'Support acts as a "floor" for prices',
              'Resistance acts as a "ceiling" for prices',
              'Broken support often becomes resistance (and vice versa)',
              'Multiple touches make levels more significant'
            ]
          }
        ],
        practicalTips: [
          'Start with longer timeframes (daily/weekly) before moving to shorter ones',
          'Always consider multiple indicators for confirmation',
          'Practice on paper trading before using real money',
          'Keep a trading journal to track your analysis and results'
        ],
        quiz: [
          {
            question: 'What does technical analysis primarily focus on?',
            options: ['Company earnings', 'Price and volume patterns', 'Economic indicators', 'Management quality'],
            correct: 1
          },
          {
            question: 'What is "support" in technical analysis?',
            options: ['A trend line', 'A price level where buying pressure prevents decline', 'A moving average', 'A volume indicator'],
            correct: 1
          }
        ]
      },
      'moving-averages': {
        title: 'Moving Averages',
        duration: '25 min',
        level: 'Beginner',
        icon: TrendingUp,
        overview: 'Moving averages are one of the most popular and versatile technical indicators. They smooth out price data by creating a constantly updated average price, helping traders identify trends and potential entry/exit points.',
        sections: [
          {
            heading: 'Understanding Moving Averages',
            content: 'A moving average calculates the average price of a security over a specific number of periods. As new data becomes available, the average "moves" by dropping the oldest data point and adding the newest one. This creates a smooth line that filters out noise from random price fluctuations.',
            keyPoints: [
              'Smooths price data to reveal underlying trends',
              'Lagging indicator - based on past prices',
              'Used for trend identification and support/resistance',
              'Common periods: 20, 50, 100, and 200 days'
            ]
          },
          {
            heading: 'Simple Moving Average (SMA)',
            content: 'The Simple Moving Average is calculated by adding up the closing prices for a specific number of periods and dividing by that number. For example, a 20-day SMA adds the last 20 closing prices and divides by 20. Each day, the oldest price drops off and the newest is added.',
            keyPoints: [
              'Equal weight given to all prices in the period',
              'Easy to calculate and understand',
              'Good for identifying overall trend direction',
              '200-day SMA is widely watched for long-term trends'
            ]
          },
          {
            heading: 'Exponential Moving Average (EMA)',
            content: 'The Exponential Moving Average gives more weight to recent prices, making it more responsive to new information. This weighting is applied exponentially, so the most recent price has the greatest impact while older prices have progressively less influence.',
            keyPoints: [
              'More weight on recent prices',
              'Reacts faster to price changes than SMA',
              'Better for short-term trading',
              'Popular periods: 12-day and 26-day for MACD'
            ]
          },
          {
            heading: 'Trading Signals: Crossovers',
            content: 'Moving average crossovers are one of the most common trading signals. A Golden Cross occurs when a shorter-term MA crosses above a longer-term MA, signaling bullish momentum. A Death Cross is the opposite - when the shorter MA crosses below the longer MA.',
            keyPoints: [
              'Golden Cross: Short-term MA crosses above long-term MA (bullish)',
              'Death Cross: Short-term MA crosses below long-term MA (bearish)',
              'Common pair: 50-day and 200-day moving averages',
              'Price crossing above/below MA also generates signals'
            ]
          },
          {
            heading: 'Moving Averages as Support/Resistance',
            content: 'Moving averages often act as dynamic support or resistance levels. In an uptrend, price tends to bounce off the MA when it pulls back. In a downtrend, the MA often acts as resistance. The 200-day SMA is particularly significant for institutional traders.',
            keyPoints: [
              'MA can act as dynamic support in uptrends',
              'MA can act as dynamic resistance in downtrends',
              'The more widely watched, the more significant',
              'Multiple MAs create stronger zones'
            ]
          }
        ],
        practicalTips: [
          'Use multiple MAs together (e.g., 20, 50, 200) for a complete picture',
          'Match MA length to your trading timeframe - shorter for day trading, longer for swing trading',
          'Don\'t use crossover signals in isolation - confirm with other indicators',
          'EMAs work better in trending markets; SMAs work better in ranging markets'
        ],
        quiz: [
          {
            question: 'What is a "Golden Cross"?',
            options: ['Price reaching all-time high', 'Short-term MA crossing above long-term MA', 'High trading volume', 'Two resistance levels meeting'],
            correct: 1
          },
          {
            question: 'Which MA type reacts faster to price changes?',
            options: ['Simple Moving Average', 'Exponential Moving Average', 'They react equally', 'Weighted Moving Average'],
            correct: 1
          }
        ]
      },
      'rsi-momentum': {
        title: 'RSI & Momentum',
        duration: '35 min',
        level: 'Intermediate',
        icon: Activity,
        overview: 'Momentum indicators measure the speed and magnitude of price movements. The Relative Strength Index (RSI) is one of the most popular momentum oscillators, helping traders identify overbought and oversold conditions as well as potential trend reversals.',
        sections: [
          {
            heading: 'What is Momentum?',
            content: 'Momentum in trading refers to the rate of acceleration of a security\'s price or volume. Momentum indicators help traders understand the strength behind price movements. Strong momentum suggests a trend is likely to continue, while weakening momentum may signal a potential reversal.',
            keyPoints: [
              'Measures the velocity of price changes',
              'Helps confirm trend strength',
              'Can signal potential reversals before price moves',
              'Leading indicators - can predict future price action'
            ]
          },
          {
            heading: 'Understanding RSI',
            content: 'The Relative Strength Index (RSI) was developed by J. Welles Wilder Jr. and measures the speed and change of price movements on a scale of 0 to 100. It compares the magnitude of recent gains to recent losses over a specified period (typically 14 days).',
            keyPoints: [
              'Oscillates between 0 and 100',
              'Above 70 = Overbought (potential sell signal)',
              'Below 30 = Oversold (potential buy signal)',
              'Default period is 14, but can be adjusted'
            ]
          },
          {
            heading: 'RSI Divergence',
            content: 'Divergence occurs when the RSI moves in the opposite direction of price. Bullish divergence happens when price makes a lower low while RSI makes a higher low, suggesting potential upward reversal. Bearish divergence is when price makes a higher high while RSI makes a lower high.',
            keyPoints: [
              'Bullish Divergence: Price lower low + RSI higher low',
              'Bearish Divergence: Price higher high + RSI lower high',
              'One of the most powerful reversal signals',
              'More reliable on longer timeframes'
            ]
          },
          {
            heading: 'RSI and Trend Analysis',
            content: 'In strong trends, RSI behavior changes. During uptrends, RSI tends to stay between 40-90 with the 40-50 zone acting as support. During downtrends, RSI typically ranges between 10-60 with 50-60 as resistance. Understanding this helps avoid false signals.',
            keyPoints: [
              'Uptrend: RSI tends to stay above 40',
              'Downtrend: RSI tends to stay below 60',
              'Centerline (50) crossovers confirm trend',
              'Adjust overbought/oversold levels based on trend'
            ]
          },
          {
            heading: 'Other Momentum Indicators',
            content: 'Besides RSI, several other momentum indicators are widely used. The Stochastic Oscillator compares closing price to price range. MACD shows the relationship between two EMAs. The Rate of Change (ROC) measures percentage change over a period.',
            keyPoints: [
              'Stochastic: Compares close to range (%K and %D lines)',
              'MACD: Difference between 12 and 26 period EMAs',
              'ROC: Percentage change over N periods',
              'Using multiple momentum indicators increases reliability'
            ]
          }
        ],
        practicalTips: [
          'Don\'t trade overbought/oversold signals alone - wait for confirmation',
          'RSI divergence is most powerful after extended trends',
          'Consider the overall trend before taking RSI signals',
          'Use longer RSI periods (21+) for swing trading, shorter (7-9) for day trading'
        ],
        quiz: [
          {
            question: 'What RSI level typically indicates overbought conditions?',
            options: ['Below 30', 'Above 50', 'Above 70', 'Below 50'],
            correct: 2
          },
          {
            question: 'What is bullish divergence?',
            options: ['Price and RSI both rising', 'Price making lower low while RSI makes higher low', 'RSI above 70', 'Price crossing moving average'],
            correct: 1
          }
        ]
      },
      'ml-trading': {
        title: 'ML in Trading',
        duration: '45 min',
        level: 'Advanced',
        icon: Brain,
        overview: 'Machine learning is revolutionizing quantitative trading by enabling computers to identify patterns in market data that humans might miss. This module explores how ML models are applied to financial markets, their limitations, and best practices for implementation.',
        sections: [
          {
            heading: 'Introduction to ML in Finance',
            content: 'Machine learning in trading involves using algorithms that learn from historical market data to make predictions or decisions. Unlike traditional rule-based systems, ML models can adapt to changing market conditions and discover complex, non-linear relationships in data.',
            keyPoints: [
              'ML can process vast amounts of data quickly',
              'Identifies patterns humans might miss',
              'Adapts to changing market conditions',
              'Used by hedge funds, banks, and retail traders'
            ]
          },
          {
            heading: 'Types of ML Models',
            content: 'Several machine learning approaches are used in trading. Supervised learning models like Random Forests and Neural Networks are trained on labeled historical data to predict future prices. Unsupervised learning finds hidden patterns. Reinforcement learning optimizes trading strategies through trial and error.',
            keyPoints: [
              'Linear Regression: Simple, interpretable price prediction',
              'Random Forest: Ensemble method, handles non-linearity',
              'Neural Networks: Deep learning for complex patterns',
              'LSTM: Specialized for time series data'
            ]
          },
          {
            heading: 'Feature Engineering',
            content: 'Feature engineering is crucial for ML trading models. Features are the input variables the model uses for prediction. Common features include technical indicators (MA, RSI, MACD), price transformations (returns, volatility), and volume metrics. The quality of features often matters more than the choice of model.',
            keyPoints: [
              'Technical indicators as features (MA, RSI, Bollinger Bands)',
              'Price transformations (log returns, volatility)',
              'Volume-based features (volume ratio, OBV)',
              'Lagged features (previous days\' data)',
              'Time-based features (day of week, month)'
            ]
          },
          {
            heading: 'Common Pitfalls',
            content: 'ML trading faces unique challenges. Overfitting occurs when a model learns noise instead of signal, performing well on historical data but poorly on new data. Look-ahead bias accidentally uses future information. Survivorship bias ignores failed companies. Markets are also non-stationary, meaning patterns change over time.',
            keyPoints: [
              'Overfitting: Model memorizes noise, fails on new data',
              'Look-ahead bias: Accidentally using future information',
              'Data snooping: Testing too many strategies on same data',
              'Non-stationarity: Market patterns change over time',
              'Transaction costs: Often ignored but crucial'
            ]
          },
          {
            heading: 'Model Validation',
            content: 'Proper validation is essential. Walk-forward validation trains on past data and tests on future data, moving the window forward. This mimics real trading conditions. Cross-validation must be adapted for time series to avoid look-ahead bias. Out-of-sample testing on completely unseen data is the final test.',
            keyPoints: [
              'Train/test split must respect time order',
              'Walk-forward optimization for parameter tuning',
              'Out-of-sample testing on unseen data',
              'Paper trading before live deployment',
              'Continuous monitoring and retraining'
            ]
          },
          {
            heading: 'Practical Implementation',
            content: 'Implementing ML trading systems requires careful consideration of infrastructure, execution, and risk management. Data quality is paramount. Models should be regularly retrained and monitored. Position sizing and risk controls must be integrated. Start with simple models before adding complexity.',
            keyPoints: [
              'Start simple: Linear models before deep learning',
              'Ensure data quality and consistency',
              'Implement proper backtesting framework',
              'Include transaction costs and slippage',
              'Never deploy without thorough testing'
            ]
          }
        ],
        practicalTips: [
          'Start with simple models - Linear Regression or Random Forest before Neural Networks',
          'Focus on feature engineering - good features matter more than fancy algorithms',
          'Always use walk-forward validation, never standard cross-validation',
          'Account for transaction costs, slippage, and market impact',
          'Paper trade extensively before risking real capital',
          'Be skeptical of backtests showing amazing returns'
        ],
        quiz: [
          {
            question: 'What is overfitting in ML trading?',
            options: ['Using too much data', 'Model learns noise instead of signal', 'Training too slowly', 'Using too many assets'],
            correct: 1
          },
          {
            question: 'Why is standard cross-validation problematic for trading?',
            options: ['It\'s too slow', 'It causes look-ahead bias', 'It uses too much memory', 'It\'s too simple'],
            correct: 1
          }
        ]
      },
      'sentiment-analysis': {
        title: 'Sentiment Analysis',
        duration: '30 min',
        level: 'Intermediate',
        icon: Radio,
        overview: 'Sentiment analysis uses natural language processing to gauge market mood from news, social media, and other text sources. Understanding market sentiment can provide an edge by capturing information before it\'s fully reflected in prices.',
        sections: [
          {
            heading: 'What is Market Sentiment?',
            content: 'Market sentiment refers to the overall attitude of investors toward a particular security or market. It\'s the psychological state that drives buying and selling decisions. Sentiment can be bullish (optimistic), bearish (pessimistic), or neutral. Understanding sentiment helps predict short-term price movements.',
            keyPoints: [
              'Reflects collective investor psychology',
              'Drives short-term price movements',
              'Can create overbought/oversold conditions',
              'Often contrarian indicator at extremes'
            ]
          },
          {
            heading: 'Sources of Sentiment Data',
            content: 'Sentiment can be extracted from various sources. Financial news articles provide professional analysis. Social media platforms like Twitter and Reddit capture retail investor sentiment. SEC filings, earnings call transcripts, and analyst reports offer institutional perspectives. Each source has different characteristics and predictive power.',
            keyPoints: [
              'Financial News: Reuters, Bloomberg, WSJ',
              'Social Media: Twitter, Reddit (r/wallstreetbets), StockTwits',
              'SEC Filings: 10-K, 10-Q, 8-K reports',
              'Analyst Reports: Upgrades, downgrades, price targets',
              'Options Market: Put/call ratios, implied volatility'
            ]
          },
          {
            heading: 'NLP for Sentiment',
            content: 'Natural Language Processing (NLP) techniques extract sentiment from text. Basic methods use sentiment dictionaries to count positive/negative words. Advanced methods use machine learning models trained on financial text. Transformer models like FinBERT are specifically trained on financial language.',
            keyPoints: [
              'Dictionary-based: Count positive/negative words',
              'Machine Learning: Naive Bayes, SVM classifiers',
              'Deep Learning: LSTM, Transformer models',
              'FinBERT: BERT fine-tuned for financial text',
              'Context matters: "Beating expectations" vs "beating up"'
            ]
          },
          {
            heading: 'Sentiment Indicators',
            content: 'Several quantitative sentiment indicators exist. The VIX (Fear Index) measures expected volatility. Put/Call ratios show options market sentiment. The AAII Sentiment Survey tracks individual investor attitudes. The CNN Fear & Greed Index combines multiple indicators.',
            keyPoints: [
              'VIX: Implied volatility, spikes during fear',
              'Put/Call Ratio: High ratio = bearish sentiment',
              'AAII Survey: Individual investor sentiment',
              'Margin Debt: High levels suggest euphoria',
              'Short Interest: High shorts can lead to squeezes'
            ]
          },
          {
            heading: 'Trading with Sentiment',
            content: 'Sentiment can be used as a trading signal in several ways. Momentum traders follow sentiment trends. Contrarian traders bet against extreme sentiment. Combining sentiment with technical and fundamental analysis provides more robust signals. Event-driven strategies react to sudden sentiment shifts.',
            keyPoints: [
              'Momentum: Trade in direction of sentiment trend',
              'Contrarian: Fade extreme sentiment readings',
              'News Trading: React to breaking news sentiment',
              'Combine with technicals for confirmation',
              'Watch for sentiment divergence from price'
            ]
          }
        ],
        practicalTips: [
          'Extreme sentiment often indicates potential reversals',
          'Combine sentiment with technical analysis for confirmation',
          'Social media sentiment is noisy - use large sample sizes',
          'News impact decays quickly - act fast or wait for confirmation',
          'Track sentiment changes over time, not just absolute levels'
        ],
        quiz: [
          {
            question: 'What does a high VIX typically indicate?',
            options: ['Bullish sentiment', 'Low volatility', 'Market fear/uncertainty', 'Strong economy'],
            correct: 2
          },
          {
            question: 'What is contrarian sentiment trading?',
            options: ['Following the crowd', 'Betting against extreme sentiment', 'Only trading news', 'Using technical analysis'],
            correct: 1
          }
        ]
      },
      'risk-management': {
        title: 'Risk Management',
        duration: '40 min',
        level: 'Beginner',
        icon: Target,
        overview: 'Risk management is the most important skill in trading. It\'s not about avoiding risk entirely, but about controlling it. Proper risk management ensures you stay in the game long enough to be profitable and prevents catastrophic losses.',
        sections: [
          {
            heading: 'Why Risk Management Matters',
            content: 'Even the best trading strategy will have losing trades. Without proper risk management, a series of losses can wipe out your account. Professional traders focus first on protecting capital, then on making profits. The math is simple: a 50% loss requires a 100% gain to break even.',
            keyPoints: [
              '50% loss needs 100% gain to recover',
              'Survival is the first priority',
              'Consistent small losses are better than rare large ones',
              'Risk management is the edge most traders lack'
            ]
          },
          {
            heading: 'Position Sizing',
            content: 'Position sizing determines how much capital to allocate to each trade. The most common rule is to never risk more than 1-2% of your account on a single trade. This means if your stop loss is 10%, your position size should be 10-20% of your account.',
            keyPoints: [
              '1-2% Rule: Never risk more than 1-2% per trade',
              'Position Size = (Account × Risk%) / (Entry - Stop Loss)',
              'Smaller positions in volatile markets',
              'Scale positions based on conviction level'
            ]
          },
          {
            heading: 'Stop Losses',
            content: 'Stop losses automatically exit a position at a predetermined price to limit losses. Technical stops are placed at logical chart levels. Volatility stops use ATR to account for normal price fluctuations. Time stops exit positions after a certain period regardless of price.',
            keyPoints: [
              'Technical Stops: Based on support/resistance levels',
              'Volatility Stops: Based on ATR (Average True Range)',
              'Percentage Stops: Fixed % below entry',
              'Time Stops: Exit after N days regardless of price',
              'Mental stops are dangerous - use real orders'
            ]
          },
          {
            heading: 'Risk-Reward Ratio',
            content: 'The risk-reward ratio compares potential profit to potential loss. A 1:3 ratio means you risk $1 to potentially make $3. With this ratio, you only need to win 25% of trades to break even. Professional traders typically aim for at least 1:2 risk-reward.',
            keyPoints: [
              'Calculate before entering any trade',
              '1:2 minimum (risk $1 to make $2)',
              'Higher ratios = lower required win rate',
              'Don\'t move stops to worsen your ratio'
            ]
          },
          {
            heading: 'Portfolio Risk',
            content: 'Beyond individual trade risk, manage overall portfolio risk. Diversification reduces exposure to any single asset. Correlation matters - uncorrelated assets provide better diversification. Maximum drawdown limits set thresholds for reducing exposure.',
            keyPoints: [
              'Diversify across sectors and asset classes',
              'Watch correlation between positions',
              'Set maximum portfolio heat (total open risk)',
              'Reduce size after drawdowns',
              'Consider hedging strategies'
            ]
          },
          {
            heading: 'Psychological Risk Management',
            content: 'Trading psychology is often the biggest risk factor. Fear and greed lead to poor decisions. Revenge trading after losses compounds problems. Having a written trading plan removes emotional decision-making. Regular breaks prevent burnout and maintain objectivity.',
            keyPoints: [
              'Create and follow a written trading plan',
              'Never trade to recover losses (revenge trading)',
              'Take breaks after large wins or losses',
              'Keep a trading journal for accountability',
              'Accept that losses are part of the process'
            ]
          }
        ],
        practicalTips: [
          'Calculate your risk before every trade - no exceptions',
          'Start with smaller position sizes while learning',
          'Use a trading journal to track your risk management',
          'Never move a stop loss to avoid taking a loss',
          'Take partial profits to reduce risk and lock in gains',
          'Have a maximum daily/weekly loss limit and stick to it'
        ],
        quiz: [
          {
            question: 'What is the commonly recommended maximum risk per trade?',
            options: ['5-10%', '10-20%', '1-2%', '50%'],
            correct: 2
          },
          {
            question: 'If you risk $100 with a 1:3 risk-reward ratio, what is your potential profit?',
            options: ['$100', '$300', '$33', '$50'],
            correct: 1
          }
        ]
      }
    };

    const modules = [
      { id: 'technical-analysis', title: 'Technical Analysis Intro', duration: '30 min', level: 'Beginner', icon: BarChart3 },
      { id: 'moving-averages', title: 'Moving Averages', duration: '25 min', level: 'Beginner', icon: TrendingUp },
      { id: 'rsi-momentum', title: 'RSI & Momentum', duration: '35 min', level: 'Intermediate', icon: Activity },
      { id: 'ml-trading', title: 'ML in Trading', duration: '45 min', level: 'Advanced', icon: Brain },
      { id: 'sentiment-analysis', title: 'Sentiment Analysis', duration: '30 min', level: 'Intermediate', icon: Radio },
      { id: 'risk-management', title: 'Risk Management', duration: '40 min', level: 'Beginner', icon: Target },
    ];

    // Module List View
    if (!selectedModule) {
      return (
        <div className="tl-panel">
          <div className="tl-panel-header">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-tl-smart" />
              <h3 className="text-data-sm font-semibold text-tl-smoke">Learning Modules</h3>
            </div>
          </div>
          
          <div className="tl-panel-body p-0">
            <div className="divide-y divide-tl-border-subtle">
              {modules.map((module) => (
                <div 
                  key={module.id} 
                  onClick={() => setSelectedModule(module.id)}
                  className="p-3 hover:bg-tl-surface-overlay cursor-pointer transition-colors group"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-md bg-tl-surface flex items-center justify-center text-tl-smart group-hover:bg-tl-smart group-hover:text-tl-smoke transition-colors">
                      <module.icon className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-data-sm font-medium text-tl-smoke group-hover:text-tl-smart transition-colors">
                        {module.title}
                      </div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-xxs text-tl-tertiary">{module.duration}</span>
                        <span className={`tl-badge ${
                          module.level === 'Beginner' ? 'tl-badge-positive' : 
                          module.level === 'Intermediate' ? 'tl-badge-warning' : 
                          'tl-badge-negative'
                        }`}>
                          {module.level}
                        </span>
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-tl-muted opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    }

    // Module Content View
    const content = moduleContent[selectedModule];

    return (
      <div className="tl-panel">
        <div className="tl-panel-header">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => {
                setSelectedModule(null);
                setActiveSection(0);
                setShowQuiz(false);
                setQuizAnswers({});
                setQuizSubmitted(false);
              }}
              className="tl-btn-icon"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <div className="flex items-center gap-2">
              <content.icon className="w-4 h-4 text-tl-smart" />
              <h3 className="text-data-sm font-semibold text-tl-smoke">{content.title}</h3>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xxs text-tl-tertiary">{content.duration}</span>
            <span className={`tl-badge ${
              content.level === 'Beginner' ? 'tl-badge-positive' : 
              content.level === 'Intermediate' ? 'tl-badge-warning' : 
              'tl-badge-negative'
            }`}>
              {content.level}
            </span>
          </div>
        </div>

        <div className="tl-panel-body p-0 max-h-[600px] overflow-y-auto tl-scroll-area">
          {!showQuiz ? (
            <>
              {/* Overview */}
              <div className="p-4 border-b border-tl-border-subtle bg-tl-surface/50">
                <p className="text-data-xs text-tl-secondary leading-relaxed">{content.overview}</p>
              </div>

              {/* Section Navigation */}
              <div className="flex overflow-x-auto tl-scroll-area border-b border-tl-border-subtle">
                {content.sections.map((section, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveSection(i)}
                    className={`px-4 py-2.5 text-data-xs font-medium whitespace-nowrap border-b-2 transition-colors ${
                      activeSection === i 
                        ? 'border-tl-smart text-tl-smart bg-tl-surface-overlay' 
                        : 'border-transparent text-tl-tertiary hover:text-tl-secondary'
                    }`}
                  >
                    {i + 1}. {section.heading.split(':')[0]}
                  </button>
                ))}
              </div>

              {/* Active Section Content */}
              <div className="p-4 space-y-4">
                <h4 className="text-data-base font-semibold text-tl-smoke">
                  {content.sections[activeSection].heading}
                </h4>
                <p className="text-data-xs text-tl-secondary leading-relaxed">
                  {content.sections[activeSection].content}
                </p>
                
                {content.sections[activeSection].keyPoints && (
                  <div className="mt-4 p-3 bg-tl-surface rounded-md border border-tl-border-subtle">
                    <h5 className="text-xxs uppercase tracking-wide text-tl-tertiary mb-2 font-semibold">
                      Key Points
                    </h5>
                    <ul className="space-y-2">
                      {content.sections[activeSection].keyPoints?.map((point, i) => (
                        <li key={i} className="flex items-start gap-2 text-data-xs text-tl-secondary">
                          <div className="w-1.5 h-1.5 rounded-full bg-tl-smart mt-1.5 flex-shrink-0" />
                          {point}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Section Navigation */}
                <div className="flex items-center justify-between pt-4 border-t border-tl-border-subtle">
                  <button
                    onClick={() => setActiveSection(Math.max(0, activeSection - 1))}
                    disabled={activeSection === 0}
                    className="tl-btn-ghost tl-btn-compact flex items-center gap-1 disabled:opacity-30"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    Previous
                  </button>
                  <span className="text-xxs text-tl-tertiary">
                    {activeSection + 1} / {content.sections.length}
                  </span>
                  {activeSection < content.sections.length - 1 ? (
                    <button
                      onClick={() => setActiveSection(activeSection + 1)}
                      className="tl-btn-ghost tl-btn-compact flex items-center gap-1"
                    >
                      Next
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </button>
                  ) : (
                    <button
                      onClick={() => setShowQuiz(true)}
                      className="tl-btn-primary tl-btn-compact flex items-center gap-1"
                    >
                      Take Quiz
                      <Zap className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>

              {/* Practical Tips */}
              <div className="p-4 border-t border-tl-border-subtle bg-gradient-to-r from-tl-smart/5 to-transparent">
                <h5 className="text-data-xs font-semibold text-tl-smoke mb-3 flex items-center gap-2">
                  <Sparkles className="w-3.5 h-3.5 text-tl-smart" />
                  Practical Tips
                </h5>
                <ul className="space-y-2">
                  {content.practicalTips.map((tip, i) => (
                    <li key={i} className="flex items-start gap-2 text-data-xs text-tl-secondary">
                      <span className="text-tl-smart font-mono">{i + 1}.</span>
                      {tip}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          ) : (
            /* Quiz View */
            <div className="p-4 space-y-6">
              <div className="flex items-center justify-between">
                <h4 className="text-data-base font-semibold text-tl-smoke">Knowledge Check</h4>
                {quizSubmitted && (
                  <span className={`tl-badge ${
                    Object.values(quizAnswers).filter((a, i) => a === content.quiz[i].correct).length === content.quiz.length
                      ? 'tl-badge-positive' : 'tl-badge-warning'
                  }`}>
                    {Object.values(quizAnswers).filter((a, i) => a === content.quiz[i].correct).length} / {content.quiz.length} Correct
                  </span>
                )}
              </div>

              {content.quiz.map((q, qIndex) => (
                <div key={qIndex} className="p-4 bg-tl-surface rounded-lg border border-tl-border-subtle">
                  <p className="text-data-sm font-medium text-tl-smoke mb-3">
                    {qIndex + 1}. {q.question}
                  </p>
                  <div className="space-y-2">
                    {q.options.map((option, oIndex) => {
                      const isSelected = quizAnswers[qIndex] === oIndex;
                      const isCorrect = oIndex === q.correct;
                      const showResult = quizSubmitted;
                      
                      return (
                        <button
                          key={oIndex}
                          onClick={() => !quizSubmitted && setQuizAnswers({...quizAnswers, [qIndex]: oIndex})}
                          disabled={quizSubmitted}
                          className={`w-full text-left p-3 rounded-md border transition-all text-data-xs ${
                            showResult
                              ? isCorrect
                                ? 'bg-tl-positive-muted border-tl-positive text-tl-positive'
                                : isSelected
                                  ? 'bg-tl-negative-muted border-tl-negative text-tl-negative'
                                  : 'bg-tl-surface border-tl-border-subtle text-tl-tertiary'
                              : isSelected
                                ? 'bg-tl-smart/20 border-tl-smart text-tl-smoke'
                                : 'bg-tl-surface-overlay border-tl-border hover:border-tl-smart/50 text-tl-secondary'
                          }`}
                        >
                          <span className="font-mono mr-2">{String.fromCharCode(65 + oIndex)}.</span>
                          {option}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}

              <div className="flex items-center justify-between pt-4 border-t border-tl-border-subtle">
                <button
                  onClick={() => {
                    setShowQuiz(false);
                    setQuizAnswers({});
                    setQuizSubmitted(false);
                  }}
                  className="tl-btn-ghost tl-btn-compact"
                >
                  ← Back to Content
                </button>
                {!quizSubmitted ? (
                  <button
                    onClick={() => setQuizSubmitted(true)}
                    disabled={Object.keys(quizAnswers).length !== content.quiz.length}
                    className="tl-btn-primary tl-btn-compact disabled:opacity-50"
                  >
                    Submit Answers
                  </button>
                ) : (
                  <button
                    onClick={() => {
                      setSelectedModule(null);
                      setActiveSection(0);
                      setShowQuiz(false);
                      setQuizAnswers({});
                      setQuizSubmitted(false);
                    }}
                    className="tl-btn-primary tl-btn-compact"
                  >
                    Complete Module ✓
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen">
      {/* Error Toast */}
      {error && (
        <div className="fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-3 bg-tl-negative-muted border border-tl-negative/50 rounded-lg shadow-tl-lg text-tl-negative animate-slide-down">
          <AlertCircle className="w-4 h-4" />
          <span className="text-data-sm font-medium">{error}</span>
          <button onClick={() => setError(null)} className="ml-2 hover:text-tl-smoke transition-colors">×</button>
        </div>
      )}

      {/* Header */}
      <header className="sticky top-0 z-40 bg-tl-space/95 backdrop-blur-md border-b border-tl-border">
        <div className="max-w-[1800px] mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            {/* Left: Logo & Brand */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Image src="/tradelens-logo.png" alt="TradeLens" width={28} height={28} />
                <span className="font-display text-lg font-bold text-tl-smoke">TradeLens</span>
              </div>
              <span className="hidden sm:inline-flex tl-badge tl-badge-neutral">Quant EdTech</span>
            </div>
            
            {/* Right: Search + Live Indicator */}
            <div className="flex items-center gap-4">
              {/* Search */}
              <div className="relative w-64 md:w-80">
                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-tl-muted" />
                <input
                  type="text"
                  placeholder="Search symbol (e.g., AAPL)..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value.toUpperCase())}
                  onKeyPress={handleSearch}
                  className="w-full pl-9 pr-4 py-2 bg-tl-surface border border-tl-border rounded-md text-data-sm font-mono text-tl-smoke placeholder-tl-text-muted focus:outline-none focus:border-tl-smart focus:ring-1 focus:ring-tl-smart transition-all"
                />
              </div>
              
              {/* Live Indicator */}
              <div className="hidden sm:flex items-center gap-2 text-xxs">
                <div className="tl-indicator tl-indicator-positive tl-indicator-pulse" />
                <span className="text-tl-tertiary uppercase tracking-wide">Live</span>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex gap-1 mt-3 -mb-px overflow-x-auto tl-scroll-area">
            {[
              { id: 'dashboard', label: 'Dashboard' },
              { id: 'stock-detail', label: 'Analysis' },
              { id: 'ml-sandbox', label: 'ML Sandbox' },
              { id: 'learn', label: 'Learn' }
            ].map((view) => (
              <button
                key={view.id}
                onClick={() => setActiveView(view.id)}
                className={`tl-nav-item whitespace-nowrap ${
                  activeView === view.id ? 'tl-nav-item-active' : 'tl-nav-item-inactive'
                }`}
              >
                {view.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1800px] mx-auto px-4 py-4">
        {activeView === 'dashboard' && (
          <div className="space-y-4">
            <MarketOverview />
            <StockChart />
            {/* AI Assistant + Watchlist - Same Row with matching heights */}
            <div className="grid lg:grid-cols-4 gap-4" style={{ height: '350px' }}>
              <div className="lg:col-span-3 h-full">
                <AIAssistant />
              </div>
              <div className="lg:col-span-1 h-full">
                <Watchlist />
              </div>
            </div>
          </div>
        )}

        {activeView === 'stock-detail' && (
          <div className="space-y-4">
            <MarketOverview />
            {/* Row 1: Stock Chart + News Feed */}
            <div className="grid lg:grid-cols-4 gap-4">
              <div className="lg:col-span-3">
                <StockChart />
              </div>
              <div className="lg:col-span-1">
                <NewsPanel />
              </div>
            </div>
            {/* Row 2: Sentiment/Fundamentals + Watchlist */}
            <div className="grid lg:grid-cols-4 gap-4">
              <div className="lg:col-span-3">
                <div className="grid md:grid-cols-2 gap-4 h-full">
                  <WeeklySentimentChart />
                  <FundamentalsPanel />
                </div>
              </div>
              <div className="lg:col-span-1">
                <Watchlist />
              </div>
            </div>
          </div>
        )}

        {activeView === 'ml-sandbox' && (
          <div className="space-y-4">
            <MarketOverview />
            <MLPredictionSandbox />
          </div>
        )}

        {activeView === 'learn' && (
          <div className="space-y-4">
            <MarketOverview />
            <div className="grid lg:grid-cols-2 gap-4">
              <LearningModules />
              <AIAssistant />
            </div>
          </div>
        )}
      </main>

      {/* Footer Status Bar */}
      <footer className="fixed bottom-0 left-0 right-0 bg-tl-space/95 backdrop-blur-md border-t border-tl-border py-1.5 px-4">
        <div className="max-w-[1800px] mx-auto flex items-center justify-between text-xxs text-tl-tertiary">
          <div className="flex items-center gap-4">
            <span className="font-mono">TradeLens v1.0</span>
            <span className="hidden sm:inline">•</span>
            <span className="hidden sm:inline">Educational Use Only</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="hidden sm:inline font-mono">
              {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
            <div className="flex items-center gap-1.5">
              <div className="tl-indicator tl-indicator-positive" />
              <span>Connected</span>
            </div>
          </div>
        </div>
      </footer>
      
      {/* Bottom padding for footer */}
      <div className="h-8" />
    </div>
  );
};

export default TradeLens;

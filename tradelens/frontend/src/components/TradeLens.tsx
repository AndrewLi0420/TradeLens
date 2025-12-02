import React, { useState, useEffect, useCallback } from 'react';
import Image from 'next/image';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, Search, Brain, Activity, BarChart3, Sparkles, RefreshCw, Loader2, AlertCircle, ExternalLink } from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

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

  // Fetch watchlist fundamentals (sector/industry for each stock)
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
    
    // Refresh market data every 60 seconds
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

  const MarketOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      {loadingMarket ? (
        Array(4).fill(0).map((_, i) => (
          <div key={i} className="bg-white rounded-lg shadow p-4 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-20 mb-2" />
            <div className="h-8 bg-gray-200 rounded w-28 mb-2" />
            <div className="h-4 bg-gray-200 rounded w-16" />
          </div>
        ))
      ) : marketIndices.length > 0 ? (
        marketIndices.map((index) => (
          <div key={index.name} className="bg-white rounded-lg shadow p-4 hover:shadow-md transition">
            <div className="text-sm text-gray-600">{index.name}</div>
            <div className="text-2xl font-bold mt-1">{index.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
            <div className={`flex items-center mt-2 ${index.change_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {index.change_percent >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              <span className="text-sm font-semibold">{index.change_percent >= 0 ? '+' : ''}{index.change_percent?.toFixed(2)}%</span>
            </div>
          </div>
        ))
      ) : (
        <div className="col-span-4 text-center py-8 text-gray-500">
          <AlertCircle className="w-8 h-8 mx-auto mb-2" />
          <p>Unable to load market data. Make sure the backend is running.</p>
        </div>
      )}
    </div>
  );

  const Watchlist = () => {
    const [isAddingStock, setIsAddingStock] = useState(false);
    const [newStockTicker, setNewStockTicker] = useState('');
    const [addStockError, setAddStockError] = useState<string | null>(null);

    const handleAddStock = async (inputValue?: string) => {
      const ticker = (inputValue || newStockTicker).toUpperCase().trim();
      
      if (!ticker) {
        setAddStockError('Please enter a ticker symbol');
        return;
      }
      
      if (watchlist.includes(ticker)) {
        setAddStockError('Stock already in watchlist');
        return;
      }

      // Validate the ticker by trying to fetch its price
      try {
        const response = await fetch(`${API_BASE}/api/stock/${ticker}/price?range=1d`);
        if (!response.ok) {
          setAddStockError('Invalid ticker symbol');
          return;
        }
        
        // Add to watchlist
        setWatchlist([...watchlist, ticker]);
        setNewStockTicker('');
        setIsAddingStock(false);
        setAddStockError(null);
        
        // Fetch price and fundamentals for the new stock
        fetchWatchlistPrices();
        fetchWatchlistFundamentals();
      } catch (err) {
        setAddStockError('Failed to validate ticker');
      }
    };

    const handleRemoveStock = (tickerToRemove: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setWatchlist(watchlist.filter(t => t !== tickerToRemove));
    };

    return (
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold">My Watchlist</h3>
          <button 
            onClick={fetchWatchlistPrices} 
            className="p-2 hover:bg-gray-100 rounded-full transition"
            disabled={loadingWatchlist}
          >
            <RefreshCw className={`w-4 h-4 text-gray-600 ${loadingWatchlist ? 'animate-spin' : ''}`} />
          </button>
        </div>
        <div className="space-y-3">
          {watchlist.map((ticker) => {
            const priceInfo = watchlistPrices[ticker];
            return (
              <div
                key={ticker}
                onClick={() => {
                  setSelectedStock(ticker);
                  setActiveView('stock-detail');
                }}
                className="flex items-center justify-between p-3 hover:bg-gray-50 rounded cursor-pointer transition group"
              >
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold mr-3">
                    {ticker[0]}
                  </div>
                  <div>
                    <div className="font-semibold">{ticker}</div>
                    <div className="text-sm text-gray-500">
                      {watchlistFundamentals[ticker]?.industry || watchlistFundamentals[ticker]?.sector || 'Loading...'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center">
                  <div className="text-right mr-2">
                    {loadingWatchlist && !priceInfo ? (
                      <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                    ) : priceInfo ? (
                      <>
                        <div className="font-semibold">${priceInfo.price.toFixed(2)}</div>
                        <div className={`text-sm ${priceInfo.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {priceInfo.change >= 0 ? '+' : ''}{priceInfo.change.toFixed(2)}%
                        </div>
                      </>
                    ) : (
                      <span className="text-gray-400 text-sm">--</span>
                    )}
                  </div>
                  <button
                    onClick={(e) => handleRemoveStock(ticker, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition text-gray-400 hover:text-red-500"
                    title="Remove from watchlist"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
        
        {isAddingStock ? (
          <div className="mt-4 space-y-2">
            <div className="flex space-x-2">
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
                placeholder="Enter ticker (e.g., GOOGL)"
                className="flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                autoFocus
              />
              <button
                onClick={() => handleAddStock()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition text-sm font-medium"
              >
                Add
              </button>
              <button
                onClick={() => {
                  setIsAddingStock(false);
                  setNewStockTicker('');
                  setAddStockError(null);
                }}
                className="px-3 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition text-sm"
              >
                Cancel
              </button>
            </div>
            {addStockError && (
              <p className="text-sm text-red-500 flex items-center">
                <AlertCircle className="w-3 h-3 mr-1" />
                {addStockError}
              </p>
            )}
          </div>
        ) : (
          <button 
            onClick={() => setIsAddingStock(true)}
            className="w-full mt-4 py-2 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-blue-400 hover:text-blue-600 hover:bg-blue-50 transition"
          >
            + Add Stock
          </button>
        )}
      </div>
    );
  };

  const StockChart = () => (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold">{selectedStock || 'Select a Stock'}</h3>
          {currentPrice && (
            <div className="text-2xl font-bold text-blue-600">${currentPrice.toFixed(2)}</div>
          )}
        </div>
        <div className="flex space-x-1">
          {timeRanges.map((range) => (
            <button
              key={range.value}
              onClick={() => setTimeRange(range.value)}
              className={`px-3 py-1 rounded text-sm font-medium transition ${
                timeRange === range.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {range.label}
            </button>
          ))}
        </div>
      </div>
      
      {loadingPrice ? (
        <div className="h-[300px] flex items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        </div>
      ) : priceData.length > 0 ? (
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={priceData}>
            <defs>
              <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="time"
              tickFormatter={(time) => {
                const date = new Date(time * 1000);
                return timeRange === '1d' 
                  ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                  : date.toLocaleDateString([], { month: 'short', day: 'numeric' });
              }}
              stroke="#999"
              fontSize={12}
            />
            <YAxis 
              stroke="#999" 
              fontSize={12}
              domain={['dataMin', 'dataMax']}
              tickFormatter={(val) => `$${val.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{ background: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
              labelFormatter={(time) => new Date(time * 1000).toLocaleString()}
              formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
            />
            <Area 
              type="monotone" 
              dataKey="close" 
              stroke="#3b82f6" 
              strokeWidth={2}
              fill="url(#colorClose)"
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div className="h-[300px] flex items-center justify-center text-gray-500">
          <div className="text-center">
            <BarChart3 className="w-12 h-12 mx-auto mb-2 text-gray-300" />
            <p>Search for a stock to view its chart</p>
          </div>
        </div>
      )}
    </div>
  );

  const NewsPanel = () => (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold flex items-center">
          <Activity className="w-5 h-5 mr-2 text-purple-600" />
          Latest News
        </h3>
      </div>
      
      {loadingNews ? (
        <div className="space-y-3">
          {Array(3).fill(0).map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-full mb-2" />
              <div className="h-3 bg-gray-200 rounded w-24" />
            </div>
          ))}
        </div>
      ) : news.length > 0 ? (
        <div className="space-y-4">
          {news.map((article, i) => (
            <a 
              key={i} 
              href={article.link} 
              target="_blank" 
              rel="noopener noreferrer"
              className="block p-3 hover:bg-gray-50 rounded-lg transition group"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="text-sm font-medium group-hover:text-blue-600 transition">
                    {article.title}
                  </div>
                  <div className="text-xs text-gray-500 mt-1 flex items-center">
                    <span>{article.publisher}</span>
                    <span className="mx-2">•</span>
                    <span>{new Date(article.published_at).toLocaleDateString()}</span>
                  </div>
                </div>
                <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-blue-600 flex-shrink-0 ml-2" />
              </div>
            </a>
          ))}
        </div>
      ) : (
        <div className="text-center py-6 text-gray-500">
          <p>No news available</p>
        </div>
      )}
    </div>
  );

  const FundamentalsPanel = () => (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h3 className="text-lg font-bold mb-4">Fundamentals</h3>
      {fundamentals && fundamentals.ticker === selectedStock ? (
        <div className="space-y-3">
          <div className="pb-3 border-b">
            <div className="font-semibold text-lg">{fundamentals.name}</div>
            <div className="text-sm text-gray-500">{fundamentals.sector} • {fundamentals.industry}</div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-gray-500">Market Cap</div>
              <div className="font-semibold">{formatNumber(fundamentals.market_cap)}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">P/E Ratio</div>
              <div className="font-semibold">{fundamentals.pe_ratio?.toFixed(2) || 'N/A'}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">EPS</div>
              <div className="font-semibold">${fundamentals.eps?.toFixed(2) || 'N/A'}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">Beta</div>
              <div className="font-semibold">{fundamentals.beta?.toFixed(2) || 'N/A'}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">52W High</div>
              <div className="font-semibold">${fundamentals['52w_high']?.toFixed(2) || 'N/A'}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">52W Low</div>
              <div className="font-semibold">${fundamentals['52w_low']?.toFixed(2) || 'N/A'}</div>
            </div>
          </div>
          {fundamentals.description && (
            <div className="pt-3 border-t">
              <div className="text-xs text-gray-500 mb-1">About</div>
              <div className="text-sm text-gray-700 line-clamp-3">{fundamentals.description}</div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-6 text-gray-500">
          <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
          <p>Loading fundamentals...</p>
        </div>
      )}
    </div>
  );

  const MLPredictionSandbox = () => {
    const [selectedFeatures, setSelectedFeatures] = useState(['ma_20', 'rsi', 'volatility']);
    const [modelType, setModelType] = useState('random_forest');
    const [predictionWindow, setPredictionWindow] = useState('1w');
    const [predictionTicker, setPredictionTicker] = useState(selectedStock || 'AAPL');

    const availableFeatures = [
      { id: 'ma_5', name: '5-day MA', category: 'trend' },
      { id: 'ma_20', name: '20-day MA', category: 'trend' },
      { id: 'ma_50', name: '50-day MA', category: 'trend' },
      { id: 'rsi', name: 'RSI', category: 'momentum' },
      { id: 'macd', name: 'MACD', category: 'momentum' },
      { id: 'macd_signal', name: 'MACD Signal', category: 'momentum' },
      { id: 'volume_ratio', name: 'Volume Ratio', category: 'volume' },
      { id: 'volatility', name: 'Volatility', category: 'volatility' }
    ];

    const handleRunPrediction = () => {
      runPrediction(predictionTicker, modelType, selectedFeatures, predictionWindow);
    };

    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-bold mb-4 flex items-center">
          <Brain className="w-5 h-5 mr-2 text-blue-600" />
          ML Prediction Sandbox
        </h3>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold mb-3">Model Configuration</h4>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Stock Ticker</label>
              <input
                type="text"
                value={predictionTicker}
                onChange={(e) => setPredictionTicker(e.target.value.toUpperCase())}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter ticker (e.g., AAPL)"
              />
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Model Type</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="linear">Linear Regression</option>
                <option value="random_forest">Random Forest</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Prediction Window</label>
              <select
                value={predictionWindow}
                onChange={(e) => setPredictionWindow(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="1d">1 Day</option>
                <option value="3d">3 Days</option>
                <option value="1w">1 Week</option>
                <option value="1m">1 Month</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Features</label>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {availableFeatures.map((feature) => (
                  <label key={feature.id} className="flex items-center space-x-2 cursor-pointer">
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
                      className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                    />
                    <span className="text-sm">{feature.name}</span>
                    <span className="text-xs text-gray-500">({feature.category})</span>
                  </label>
                ))}
              </div>
            </div>

            <button 
              onClick={handleRunPrediction}
              disabled={loadingPrediction || selectedFeatures.length === 0}
              className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition font-semibold disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loadingPrediction ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  Running Prediction...
                </>
              ) : (
                'Run Prediction'
              )}
            </button>
          </div>

          <div>
            <h4 className="font-semibold mb-3">Prediction Results</h4>
            
            {prediction ? (
              <>
                <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg p-4 mb-4">
                  <div className="text-sm text-gray-600 mb-1">
                    Predicted Price ({prediction.ticker})
                  </div>
                  <div className="text-3xl font-bold text-blue-600">
                    ${prediction.predicted_price.toFixed(2)}
                  </div>
                  <div className={`text-sm mt-1 ${prediction.prediction_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {prediction.prediction_change >= 0 ? '+' : ''}{prediction.prediction_change.toFixed(2)}% from ${prediction.current_price.toFixed(2)}
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span>RMSE</span>
                      <span className="font-semibold">{prediction.metrics.rmse.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full" 
                        style={{ width: `${Math.max(0, 100 - prediction.metrics.rmse * 5)}%` }} 
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span>Directional Accuracy</span>
                      <span className="font-semibold">{prediction.metrics.directional_accuracy.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full" 
                        style={{ width: `${prediction.metrics.directional_accuracy}%` }} 
                      />
                    </div>
                  </div>
                </div>

                {prediction.feature_importance && (
                  <div className="mt-4">
                    <div className="text-sm font-medium mb-2">Feature Importance</div>
                    <div className="space-y-1">
                      {Object.entries(prediction.feature_importance)
                        .sort(([, a], [, b]) => b - a)
                        .map(([feature, importance]) => (
                          <div key={feature} className="flex items-center text-xs">
                            <span className="w-24 text-gray-600">{feature}</span>
                            <div className="flex-1 bg-gray-200 rounded-full h-2 mx-2">
                              <div 
                                className="bg-purple-500 h-2 rounded-full" 
                                style={{ width: `${importance * 100}%` }} 
                              />
                            </div>
                            <span className="w-12 text-right">{(importance * 100).toFixed(1)}%</span>
                          </div>
                        ))
                      }
                    </div>
                  </div>
                )}

                <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <div className="flex items-start">
                    <Sparkles className="w-5 h-5 text-yellow-600 mr-2 flex-shrink-0 mt-0.5" />
                    <div className="text-sm text-yellow-800">
                      <strong>Note:</strong> This is an educational tool. ML predictions are based on historical patterns and should not be used for actual trading decisions.
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Brain className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                <p>Configure your model and click "Run Prediction" to see results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const AIAssistant = () => {
    const [messages, setMessages] = useState([
      { role: 'assistant', content: 'Hi! I\'m your AI trading tutor. Ask me anything about stocks, indicators, or trading strategies!' }
    ]);
    const [input, setInput] = useState('');

    const sendMessage = () => {
      if (!input.trim()) return;
      
      setMessages([...messages, 
        { role: 'user', content: input },
        { role: 'assistant', content: 'This is where Claude would provide educational explanations about your question, using real market data and context.' }
      ]);
      setInput('');
    };

    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-bold mb-4 flex items-center">
          <Sparkles className="w-5 h-5 mr-2 text-purple-600" />
          AI Trading Tutor
        </h3>
        
        <div className="h-64 overflow-y-auto mb-4 space-y-3">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] p-3 rounded-lg ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-800'
              }`}>
                {msg.content}
              </div>
            </div>
          ))}
        </div>

        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask me anything about trading..."
            className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
          <button
            onClick={sendMessage}
            className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-semibold"
          >
            Send
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Error Toast */}
      {error && (
        <div className="fixed top-4 right-4 bg-red-100 border border-red-300 text-red-700 px-4 py-3 rounded-lg shadow-lg z-50 flex items-center">
          <AlertCircle className="w-5 h-5 mr-2" />
          <span>{error}</span>
          <button onClick={() => setError(null)} className="ml-4 text-red-500 hover:text-red-700">×</button>
        </div>
      )}

      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Image src="/tradelens-logo.png" alt="TradeLens Logo" width={32} height={32} />
              <h1 className="text-2xl font-bold">TradeLens</h1>
              <span className="text-sm bg-white/20 px-2 py-1 rounded">Quant EdTech</span>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search stocks (e.g., AAPL)..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value.toUpperCase())}
                  onKeyPress={handleSearch}
                  className="pl-10 pr-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/50 w-64"
                />
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex space-x-6 mt-4">
            {[
              { id: 'dashboard', label: 'Dashboard' },
              { id: 'stock-detail', label: 'Stock Detail' },
              { id: 'ml-sandbox', label: 'ML Sandbox' },
              { id: 'learn', label: 'Learn' }
            ].map((view) => (
              <button
                key={view.id}
                onClick={() => setActiveView(view.id)}
                className={`pb-2 transition ${
                  activeView === view.id
                    ? 'border-b-2 border-white font-semibold'
                    : 'text-white/70 hover:text-white'
                }`}
              >
                {view.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {activeView === 'dashboard' && (
          <>
            <MarketOverview />
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <StockChart />
                <AIAssistant />
              </div>
              <div>
                <Watchlist />
              </div>
            </div>
          </>
        )}

        {activeView === 'stock-detail' && (
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <StockChart />
              <FundamentalsPanel />
            </div>
            <div>
              <NewsPanel />
              <Watchlist />
            </div>
          </div>
        )}

        {activeView === 'ml-sandbox' && (
          <MLPredictionSandbox />
        )}

        {activeView === 'learn' && (
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-bold mb-4">Learning Modules</h3>
              <div className="space-y-3">
                {[
                  'Introduction to Technical Analysis',
                  'Understanding Moving Averages',
                  'RSI and Momentum Indicators',
                  'Machine Learning in Trading',
                  'Sentiment Analysis Basics',
                  'Risk Management Strategies'
                ].map((module, i) => (
                  <div key={i} className="p-4 border border-gray-200 rounded-lg hover:border-blue-500 cursor-pointer transition">
                    <div className="font-semibold">{module}</div>
                    <div className="text-sm text-gray-500 mt-1">30 min • Beginner</div>
                  </div>
                ))}
              </div>
            </div>
            <AIAssistant />
          </div>
        )}
      </main>
    </div>
  );
};

export default TradeLens;

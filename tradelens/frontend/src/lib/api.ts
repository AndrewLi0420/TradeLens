import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

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


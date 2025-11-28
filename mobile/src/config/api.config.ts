/**
 * API Configuration for BreastCare AI Mobile App
 * Centralized configuration for all API endpoints
 */

// Environment types
type Environment = 'PRODUCTION' | 'DEVELOPMENT';

// Environment configuration
export const API_CONFIG = {
  // Production server on Render.com
  PRODUCTION: {
    BASE_URL: 'https://breast-cancer-28rl.onrender.com/api/v1',
    TIMEOUT: 60000, // 60 seconds for Render cold start
  },
  
  // Local development server
  DEVELOPMENT: {
    BASE_URL: 'http://192.168.1.181:8000/api/v1',
    TIMEOUT: 10000, // 10 seconds for local
  }
} as const;

// Current environment - change this to switch between environments
const CURRENT_ENV: Environment = 'PRODUCTION';

// Export current configuration
export const API_BASE_URL = API_CONFIG[CURRENT_ENV].BASE_URL;
export const API_TIMEOUT = API_CONFIG[CURRENT_ENV].TIMEOUT;

// Common headers for all requests
export const COMMON_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'User-Agent': 'BreastCare-Mobile-App/1.0.0',
};

// Utility function to get full endpoint URL
export const getEndpoint = (path: string): string => {
  return `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`;
};

// Debug logging
console.log(`[API Config] Using ${CURRENT_ENV} environment: ${API_BASE_URL}`);
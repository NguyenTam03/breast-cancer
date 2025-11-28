/**
 * API Service for BreastCare AI Mobile App
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { AnalysisResult, AnalysisHistory } from '../types/analysis.types';
import { API_BASE_URL, API_TIMEOUT, COMMON_HEADERS } from '../config/api.config';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: COMMON_HEADERS,
      // Add additional config for React Native
      validateStatus: function (status) {
        return status >= 200 && status < 300;
      },
    });

    // Request interceptor - automatically add auth token
    this.client.interceptors.request.use(
      async (config) => {
        try {
          // Get auth token from storage
          const token = await AsyncStorage.getItem('access_token');
          if (token) {
            config.headers.Authorization = `Bearer ${token}`;
          }
        } catch (error) {
          console.warn('Failed to get auth token:', error);
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor with retry logic for Render cold start
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;
        
        // If it's a network error and we haven't retried yet
        if (error.code === 'ECONNABORTED' || error.message === 'Network Error') {
          if (!originalRequest._retry) {
            originalRequest._retry = true;
            console.log('Render service might be sleeping, retrying in 5 seconds...');
            
            // Wait 5 seconds for Render to wake up
            await new Promise(resolve => setTimeout(resolve, 5000));
            
            return this.client(originalRequest);
          }
        }
        
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Check API health status
   */
  async checkHealth(): Promise<any> {
    const response = await this.client.get('/health/');
    return response.data;
  }

  /**
   * Analyze image for breast cancer detection
   */
  async analyzeImage(
    imageUri: string,
    notes?: string
  ): Promise<AnalysisResult> {
    const formData = new FormData();
    
    // Append image file
    formData.append('image', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'breast_scan.jpg',
    } as any);

    // Append notes if provided
    if (notes) {
      formData.append('notes', notes);
    }

    const response = await this.client.post('/analysis/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }


  /**
   * Get analysis history (all users - for testing)
   */
  async getAnalysisHistory(
    page: number = 1,
    pageSize: number = 10,
    filterPrediction?: string
  ): Promise<AnalysisHistory> {
    const params: any = { page, pageSize };
    if (filterPrediction) {
      params.filter_prediction = filterPrediction;
    }

    const response = await this.client.get('/analysis/history', { params });
    return response.data;
  }

  /**
   * Get analysis history for specific user
   */
  async getUserAnalysisHistory(
    userId: string,
    page: number = 1,
    pageSize: number = 10,
    filterPrediction?: string
  ): Promise<AnalysisHistory> {
    const params: any = { page, pageSize };
    if (filterPrediction) {
      params.filter_prediction = filterPrediction;
    }

    const response = await this.client.get(`/analysis/history/${userId}`, { params });
    return response.data;
  }

  /**
   * Get specific analysis details
   */
  async getAnalysisDetails(analysisId: string): Promise<AnalysisResult> {
    const response = await this.client.get(`/analysis/${analysisId}`);
    return response.data;
  }

  /**
   * Update analysis (notes, tags)
   */
  async updateAnalysis(
    analysisId: string,
    updateData: { userNotes?: string; tags?: string[] }
  ): Promise<any> {
    const response = await this.client.put(`/analysis/${analysisId}`, updateData);
    return response.data;
  }

  /**
   * Delete analysis
   */
  async deleteAnalysis(analysisId: string): Promise<any> {
    const response = await this.client.delete(`/analysis/${analysisId}`);
    return response.data;
  }

  /**
   * Toggle bookmark status
   */
  async toggleBookmark(analysisId: string): Promise<any> {
    const response = await this.client.post(`/analysis/${analysisId}/bookmark`);
    return response.data;
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;

/**
 * API Service for BreastCare AI Mobile App
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { AnalysisResult, AnalysisHistory } from '../types/analysis.types';

// API Configuration  
// For Android emulator, use 10.0.2.2 instead of localhost
// For physical device or iOS simulator, use your computer's IP address
const API_BASE_URL = 'http://10.0.2.2:8000/api/v1';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000, // 30 seconds timeout for image upload
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        // const token = getAuthToken();
        // if (token) {
        //   config.headers.Authorization = `Bearer ${token}`;
        // }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
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
   * Get analysis history
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

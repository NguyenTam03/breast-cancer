import { LoginRequest, RegisterRequest, TokenResponse, User, DeviceInfo } from '../types/auth.types';
import { Platform } from 'react-native';
import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = 'http://10.0.2.2:8000/api/v1'; // Android emulator URL

class AuthService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000, // 10 seconds timeout
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('Auth API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  private getDeviceInfo(): DeviceInfo {
    return {
      platform: Platform.OS === 'ios' ? 'ios' : Platform.OS === 'android' ? 'android' : 'web',
      deviceId: Math.random().toString(36).substring(7),
      appVersion: '1.0.0'
    };
  }

  async login(email: string, password: string): Promise<TokenResponse> {
    const loginRequest: LoginRequest = {
      email,
      password,
      deviceInfo: this.getDeviceInfo()
    };

    try {
      const response = await this.client.post('/auth/login', loginRequest);
      return response.data;
    } catch (error: any) {
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      } else if (error.code === 'ECONNABORTED') {
        throw new Error('Kết nối timeout. Vui lòng kiểm tra server.');
      } else if (error.code === 'NETWORK_ERROR') {
        throw new Error('Lỗi kết nối mạng. Vui lòng kiểm tra server backend.');
      } else {
        throw new Error('Đăng nhập thất bại: ' + error.message);
      }
    }
  }

  async register(
    email: string,
    password: string,
    firstName: string,
    lastName: string,
    role: 'doctor' | 'patient' = 'patient'
  ): Promise<TokenResponse> {
    const registerRequest: RegisterRequest = {
      email,
      password,
      firstName,
      lastName,
      role: role as any,
      deviceInfo: this.getDeviceInfo()
    };

    try {
      const response = await this.client.post('/auth/register', registerRequest);
      return response.data;
    } catch (error: any) {
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      } else {
        throw new Error('Đăng ký thất bại: ' + error.message);
      }
    }
  }

  async getCurrentUser(accessToken: string): Promise<User> {
    try {
      const response = await this.client.get('/auth/me', {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401) {
        throw new Error('Token không hợp lệ hoặc đã hết hạn');
      } else {
        throw new Error('Không thể lấy thông tin người dùng: ' + error.message);
      }
    }
  }

  async refreshToken(refreshToken: string): Promise<TokenResponse> {
    try {
      const response = await this.client.post('/auth/refresh', { 
        refresh_token: refreshToken 
      });
      return response.data;
    } catch (error: any) {
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      } else {
        throw new Error('Không thể làm mới token: ' + error.message);
      }
    }
  }

  async logout(accessToken: string): Promise<void> {
    try {
      await this.client.post('/auth/logout', {}, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
    } catch (error: any) {
      // Logout error is not critical, just log it
      console.warn('Logout error:', error.message);
    }
  }
}

export const authService = new AuthService();

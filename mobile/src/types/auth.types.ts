export interface User {
  id: string;
  email: string;
  profile: UserProfile;
  preferences: UserPreferences;
  createdAt: string;
  lastLogin?: string;
}

export interface UserProfile {
  firstName: string;
  lastName: string;
  dateOfBirth?: string;
  phone?: string;
  avatar?: string;
  gender?: string;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: 'vi' | 'en';
  notifications: boolean;
}

export interface LoginRequest {
  email: string;
  password: string;
  deviceInfo: DeviceInfo;
}

export interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  deviceInfo: DeviceInfo;
}

export interface DeviceInfo {
  platform: 'ios' | 'android' | 'web';
  deviceId: string;
  appVersion: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

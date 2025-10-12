import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { AuthState, User, TokenResponse } from '../types/auth.types';
import { authService } from '../services/authService';

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, firstName: string, lastName: string, role?: 'doctor' | 'patient') => Promise<void>;
  logout: () => Promise<void>;
  checkAuthState: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const STORAGE_KEYS = {
  ACCESS_TOKEN: 'access_token',
  REFRESH_TOKEN: 'refresh_token',
  USER_DATA: 'user_data',
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    accessToken: null,
    refreshToken: null,
    isLoading: true,
    isAuthenticated: false,
  });

  // Check if user is already authenticated
  const checkAuthState = async () => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true }));
      
      const [accessToken, refreshToken, userData] = await Promise.all([
        AsyncStorage.getItem(STORAGE_KEYS.ACCESS_TOKEN),
        AsyncStorage.getItem(STORAGE_KEYS.REFRESH_TOKEN),
        AsyncStorage.getItem(STORAGE_KEYS.USER_DATA),
      ]);

      if (accessToken && refreshToken) {
        try {
          // Try to get current user info to validate token
          const user = await authService.getCurrentUser(accessToken);
          
          setAuthState({
            user,
            accessToken,
            refreshToken,
            isLoading: false,
            isAuthenticated: true,
          });
        } catch (error) {
          console.log('Access token expired, attempting to refresh...');
          // Token might be expired, try to refresh
          if (refreshToken) {
            try {
              const tokenResponse = await authService.refreshToken(refreshToken);
              const user = await authService.getCurrentUser(tokenResponse.access_token);
              
              await saveTokens(tokenResponse, user);
              
              setAuthState({
                user,
                accessToken: tokenResponse.access_token,
                refreshToken: tokenResponse.refresh_token,
                isLoading: false,
                isAuthenticated: true,
              });
              console.log('Token refreshed successfully');
            } catch (refreshError: any) {
              // Refresh failed, clear stored data and log out user
              console.log('Refresh token expired or invalid, logging out user');
              await clearStorage();
              setAuthState({
                user: null,
                accessToken: null,
                refreshToken: null,
                isLoading: false,
                isAuthenticated: false,
              });
            }
          } else {
            // No refresh token available
            await clearStorage();
            setAuthState({
              user: null,
              accessToken: null,
              refreshToken: null,
              isLoading: false,
              isAuthenticated: false,
            });
          }
        }
      } else {
        // No tokens found, user is not authenticated
        await clearStorage();
        setAuthState(prev => ({
          ...prev,
          user: null,
          accessToken: null,
          refreshToken: null,
          isLoading: false,
          isAuthenticated: false,
        }));
      }
    } catch (error) {
      console.error('Error checking auth state:', error);
      await clearStorage();
      setAuthState({
        user: null,
        accessToken: null,
        refreshToken: null,
        isLoading: false,
        isAuthenticated: false,
      });
    }
  };

  const saveTokens = async (tokenResponse: TokenResponse, user: User) => {
    await Promise.all([
      AsyncStorage.setItem(STORAGE_KEYS.ACCESS_TOKEN, tokenResponse.access_token),
      AsyncStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, tokenResponse.refresh_token),
      AsyncStorage.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(user)),
    ]);
  };

  const clearStorage = async () => {
    await Promise.all([
      AsyncStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN),
      AsyncStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN),
      AsyncStorage.removeItem(STORAGE_KEYS.USER_DATA),
    ]);
  };

  const login = async (email: string, password: string) => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true }));
      
      const tokenResponse = await authService.login(email, password);
      const user = await authService.getCurrentUser(tokenResponse.access_token);
      
      await saveTokens(tokenResponse, user);
      
      setAuthState({
        user,
        accessToken: tokenResponse.access_token,
        refreshToken: tokenResponse.refresh_token,
        isLoading: false,
        isAuthenticated: true,
      });
    } catch (error) {
      setAuthState(prev => ({ ...prev, isLoading: false }));
      throw error;
    }
  };

  const register = async (
    email: string,
    password: string,
    firstName: string,
    lastName: string,
    role: 'doctor' | 'patient' = 'patient'
  ) => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true }));
      
      const tokenResponse = await authService.register(email, password, firstName, lastName, role);
      const user = await authService.getCurrentUser(tokenResponse.access_token);
      
      await saveTokens(tokenResponse, user);
      
      setAuthState({
        user,
        accessToken: tokenResponse.access_token,
        refreshToken: tokenResponse.refresh_token,
        isLoading: false,
        isAuthenticated: true,
      });
    } catch (error) {
      setAuthState(prev => ({ ...prev, isLoading: false }));
      throw error;
    }
  };

  const logout = async () => {
    try {
      if (authState.accessToken) {
        await authService.logout(authState.accessToken);
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      await clearStorage();
      setAuthState({
        user: null,
        accessToken: null,
        refreshToken: null,
        isLoading: false,
        isAuthenticated: false,
      });
    }
  };

  useEffect(() => {
    checkAuthState();
  }, []);

  const value: AuthContextType = {
    ...authState,
    login,
    register,
    logout,
    checkAuthState,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

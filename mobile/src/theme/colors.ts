/**
 * Dark Theme Color Palette
 */

export const colors = {
  // Background colors
  background: {
    primary: '#0A0B0D',
    secondary: '#1A1B1E',
    tertiary: '#2A2B2E',
    card: '#1E1F22',
    overlay: 'rgba(0, 0, 0, 0.8)',
  },

  // Text colors
  text: {
    primary: '#FFFFFF',
    secondary: '#B8BCC1',
    tertiary: '#8B8E93',
    disabled: '#5A5D62',
    accent: '#3B82F6',
  },

  // Border colors
  border: {
    primary: '#2A2B2E',
    secondary: '#3A3B3E',
    accent: '#3B82F6',
  },

  // Status colors
  status: {
    success: '#10B981',
    error: '#EF4444',
    warning: '#F59E0B',
    info: '#3B82F6',
  },

  // Button colors
  button: {
    primary: '#3B82F6',
    primaryHover: '#2563EB',
    secondary: '#374151',
    secondaryHover: '#4B5563',
    outline: 'transparent',
    danger: '#EF4444',
    dangerHover: '#DC2626',
    disabled: '#374151',
  },

  // Gradient colors
  gradient: {
    primary: ['#1E1F22', '#2A2B2E'],
    accent: ['#3B82F6', '#1D4ED8'],
    success: ['#10B981', '#059669'],
    error: ['#EF4444', '#DC2626'],
  },
};

export type Colors = typeof colors;

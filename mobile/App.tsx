import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { View, Text, ActivityIndicator } from 'react-native';

// Import auth
import { AuthProvider, useAuth } from './src/contexts/AuthContext';
import { LoginScreen } from './src/screens/LoginScreen';

// Import screens
import NewHomeScreen from './src/screens/NewHomeScreen';
import AnalysisScreen from './src/screens/AnalysisScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import ImagePreviewScreen from './src/screens/ImagePreviewScreen';
import ImageCropScreen from './src/screens/ImageCropScreen';

// Define navigation types
export type RootStackParamList = {
  Home: undefined;
  ImagePreview: { imageUri: string };
  ImageCrop: { imageUri: string };
  Analysis: { imageUri: string };
  HistoryMain: undefined;
};

// Create navigators
const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Home Stack Navigator
function HomeStack() {
  return (
    <Stack.Navigator 
      screenOptions={{ 
        headerShown: false,
        cardStyle: { backgroundColor: '#1a1a2e' }
      }}
    >
      <Stack.Screen name="Home" component={NewHomeScreen} />
      <Stack.Screen name="ImagePreview" component={ImagePreviewScreen} />
      <Stack.Screen name="ImageCrop" component={ImageCropScreen} />
      <Stack.Screen name="Analysis" component={AnalysisScreen} />
    </Stack.Navigator>
  );
}

// History Stack Navigator
function HistoryStack() {
  return (
    <Stack.Navigator 
      screenOptions={{ 
        headerShown: false,
        cardStyle: { backgroundColor: '#1a1a2e' }
      }}
    >
      <Stack.Screen name="HistoryMain" component={HistoryScreen} />
      <Stack.Screen name="Analysis" component={AnalysisScreen} />
    </Stack.Navigator>
  );
}

// Main Tab Navigator
function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#2a2a3e',
          borderTopWidth: 1,
          borderTopColor: 'rgba(255, 255, 255, 0.1)',
          paddingBottom: 8,
          paddingTop: 8,
          height: 80,
        },
        tabBarActiveTintColor: '#4c6ef5',
        tabBarInactiveTintColor: '#8e8e93',
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '500',
          marginTop: 4,
        },
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap;

          if (route.name === 'HomeTab') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'HistoryTab') {
            iconName = focused ? 'analytics' : 'analytics-outline';
          } else if (route.name === 'SettingsTab') {
            iconName = focused ? 'settings' : 'settings-outline';
          } else {
            iconName = 'help-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen 
        name="HomeTab" 
        component={HomeStack}
        options={{
          tabBarLabel: 'Trang chủ',
        }}
      />
      <Tab.Screen 
        name="HistoryTab" 
        component={HistoryStack}
        options={{
          tabBarLabel: 'Phân tích',
        }}
      />
      <Tab.Screen 
        name="SettingsTab" 
        component={SettingsScreen}
        options={{
          tabBarLabel: 'Cài đặt',
        }}
      />
    </Tab.Navigator>
  );
}

// Auth Stack Navigator for login flow
function AuthStack() {
  const AuthStackNav = createStackNavigator();
  return (
    <AuthStackNav.Navigator 
      screenOptions={{ 
        headerShown: false,
        cardStyle: { backgroundColor: '#1a1a2e' }
      }}
    >
      <AuthStackNav.Screen name="Login" component={LoginScreen} />
    </AuthStackNav.Navigator>
  );
}

// Loading screen
function LoadingScreen() {
  return (
    <View style={{ 
      flex: 1, 
      backgroundColor: '#1a1a2e',
      justifyContent: 'center', 
      alignItems: 'center' 
    }}>
      <ActivityIndicator size="large" color="#4c6ef5" />
      <Text style={{ 
        color: 'white', 
        fontSize: 16, 
        marginTop: 16 
      }}>
        Đang tải...
      </Text>
    </View>
  );
}

// App Navigator based on auth state
function AppNavigator() {
  const { isAuthenticated, isLoading, user } = useAuth();

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <NavigationContainer
      theme={{
        ...DefaultTheme,
        dark: true,
        colors: {
          ...DefaultTheme.colors,
          primary: '#4c6ef5',
          background: '#1a1a2e',
          card: '#2a2a3e',
          text: '#ffffff',
          border: 'rgba(255, 255, 255, 0.1)',
          notification: '#4c6ef5',
        },
      }}
    >
      {isAuthenticated ? <MainTabs /> : <AuthStack />}
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <AppNavigator />
        </AuthProvider>
      </QueryClientProvider>
    </SafeAreaProvider>
  );
}

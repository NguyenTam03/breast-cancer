import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { View, Text } from 'react-native';

// Import screens
import NewHomeScreen from './src/screens/NewHomeScreen';
import AnalysisScreen from './src/screens/AnalysisScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import SettingsScreen from './src/screens/SettingsScreen';

// Define navigation types
export type RootStackParamList = {
  Home: undefined;
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

// Appointment placeholder screen
function AppointmentScreen() {
  return (
    <View style={{ 
      flex: 1, 
      backgroundColor: '#1a1a2e',
      justifyContent: 'center', 
      alignItems: 'center' 
    }}>
      <Ionicons name="calendar-outline" size={48} color="#8e8e93" />
      <Text style={{ 
        color: 'white', 
        fontSize: 18, 
        fontWeight: 'bold', 
        marginTop: 16,
        marginBottom: 8
      }}>
        Lịch hẹn
      </Text>
      <Text style={{ 
        color: '#8e8e93', 
        fontSize: 14, 
        textAlign: 'center' 
      }}>
        Chức năng đang phát triển
      </Text>
    </View>
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
          } else if (route.name === 'AppointmentTab') {
            iconName = focused ? 'calendar' : 'calendar-outline';
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
        name="AppointmentTab" 
        component={AppointmentScreen}
        options={{
          tabBarLabel: 'Lịch hẹn',
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

export default function App() {
  return (
    <SafeAreaProvider>
      <QueryClientProvider client={queryClient}>
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
          <MainTabs />
        </NavigationContainer>
      </QueryClientProvider>
    </SafeAreaProvider>
  );
}

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Dimensions,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';
import { colors } from '../theme/colors';
import { apiService } from '../services/api';

const { width } = Dimensions.get('window');

interface FeatureCardProps {
  title: string;
  subtitle: string;
  icon: keyof typeof Ionicons.glyphMap;
  gradient: string[];
  onPress: () => void;
}

export default function NewHomeScreen({ navigation }: any) {
  const [searchQuery, setSearchQuery] = useState('');
  const [isApiHealthy, setIsApiHealthy] = useState(false);
  const [recentAnalysisCount, setRecentAnalysisCount] = useState(0);

  // Check API health and get recent analysis count on component mount
  useEffect(() => {
    checkApiHealth();
    getRecentAnalysisCount();
  }, []);

  const checkApiHealth = async () => {
    try {
      await apiService.checkHealth();
      setIsApiHealthy(true);
    } catch (error) {
      console.error('API health check failed:', error);
      setIsApiHealthy(false);
    }
  };

  const getRecentAnalysisCount = async () => {
    try {
      const history = await apiService.getAnalysisHistory(1, 5);
      setRecentAnalysisCount(history.totalCount || 0);
    } catch (error) {
      console.error('Failed to get analysis count:', error);
      setRecentAnalysisCount(0);
    }
  };

  const takePhoto = async () => {
    try {
      const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
      
      if (permissionResult.granted === false) {
        Alert.alert('Quyền truy cập', 'Cần quyền truy cập camera để chụp ảnh');
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        // Navigate to analysis screen with image
        navigation.navigate('Analysis', { imageUri: result.assets[0].uri });
      }
    } catch (error) {
      Alert.alert('Lỗi', 'Không thể chụp ảnh');
    }
  };

  const pickFromGallery = async () => {
    try {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (permissionResult.granted === false) {
        Alert.alert('Quyền truy cập', 'Cần quyền truy cập thư viện ảnh để chọn hình');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        // Navigate to analysis screen with image
        navigation.navigate('Analysis', { imageUri: result.assets[0].uri });
      }
    } catch (error) {
      Alert.alert('Lỗi', 'Không thể chọn hình ảnh');
    }
  };

  const FeatureCard: React.FC<FeatureCardProps> = ({ title, subtitle, icon, gradient, onPress }) => (
    <TouchableOpacity style={styles.featureCard} onPress={onPress} activeOpacity={0.8}>
      <LinearGradient colors={gradient as any} style={styles.featureGradient}>
        <View style={styles.featureContent}>
          <View style={styles.featureIconContainer}>
            <Ionicons name={icon} size={24} color="white" />
          </View>
          <View style={styles.featureText}>
            <Text style={styles.featureTitle}>{title}</Text>
            <Text style={styles.featureSubtitle}>{subtitle}</Text>
          </View>
          <Ionicons name="arrow-forward" size={20} color="white" style={styles.featureArrow} />
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
      
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.profileButton}>
            <View style={styles.profileImagePlaceholder}>
              <Ionicons name="person" size={20} color="#4c6ef5" />
            </View>
          </TouchableOpacity>
          
          <View style={styles.headerActions}>
            <TouchableOpacity style={styles.headerButton}>
              <Ionicons name="add" size={24} color="white" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.headerButton}>
              <Ionicons name="notifications-outline" size={24} color="white" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Search Bar */}
        <View style={styles.searchContainer}>
          <View style={styles.searchBar}>
            <Ionicons name="search" size={20} color="#8e8e93" style={styles.searchIcon} />
            <Text style={styles.searchPlaceholder}>Tìm kiếm</Text>
            <TouchableOpacity>
              <Ionicons name="mic" size={20} color="#8e8e93" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Main Feature Card */}
        <View style={styles.mainCardContainer}>
          <LinearGradient 
            colors={['#4c6ef5', '#7c3aed'] as any} 
            style={styles.mainCard}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <View style={styles.mainCardBadge}>
              <Ionicons name="camera" size={12} color="white" />
              <Text style={styles.mainCardBadgeText}>Phân tích ngay</Text>
            </View>
            
            <View style={styles.mainCardContent}>
              <View style={styles.mainCardText}>
                <Text style={styles.mainCardTitle}>Chụp ảnh để</Text>
                <Text style={styles.mainCardTitle}>Phân tích ngay</Text>
                <TouchableOpacity style={styles.mainCardButton} onPress={takePhoto}>
                  <Text style={styles.mainCardButtonText}>Chụp ngay</Text>
                </TouchableOpacity>
              </View>
              
              <View style={styles.mainCardImage}>
                <Image 
                  source={{ uri: 'https://via.placeholder.com/120x120/ffffff/4c6ef5?text=🏥' }}
                  style={styles.doctorImage}
                />
              </View>
            </View>
            
            {/* Dots indicator */}
            <View style={styles.dotsContainer}>
              <View style={[styles.dot, styles.activeDot]} />
              <View style={styles.dot} />
              <View style={styles.dot} />
            </View>
          </LinearGradient>
        </View>

        {/* Features Section */}
        <View style={styles.featuresSection}>
          <Text style={styles.sectionTitle}>Quản lý Phân tích Ung thư Vú</Text>
          
          <View style={styles.featuresGrid}>
            <View style={styles.featuresRow}>
              <FeatureCard
                title="Chụp ảnh phân tích"
                subtitle="Sử dụng camera để chụp ảnh."
                icon="camera"
                gradient={['#ff6b6b', '#ee5a24']}
                onPress={takePhoto}
              />
              
              <FeatureCard
                title="Xem lịch sử kết quả"
                subtitle={`${recentAnalysisCount} kết quả đã phân tích`}
                icon="document-text"
                gradient={['#4c6ef5', '#3742fa']}
                onPress={() => navigation.navigate('HistoryTab')}
              />
            </View>
            
            <View style={styles.featuresRow}>
              <FeatureCard
                title="Chọn từ thư viện"
                subtitle="Chọn ảnh từ thư viện của bạn."
                icon="images"
                gradient={['#a55eea', '#8c7ae6']}
                onPress={pickFromGallery}
              />
              
              <FeatureCard
                title="Đặt lịch tái khám"
                subtitle="Đặt lịch hẹn với bác sĩ."
                icon="calendar"
                gradient={['#ffa726', '#ff9800']}
                onPress={() => navigation.navigate('AppointmentTab')}
              />
            </View>
          </View>
        </View>
      </ScrollView>
      
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 20,
  },
  profileButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    overflow: 'hidden',
  },
  profileImage: {
    width: '100%',
    height: '100%',
  },
  profileImagePlaceholder: {
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerActions: {
    flexDirection: 'row',
    gap: 16,
  },
  headerButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  searchContainer: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2a2a3e',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 12,
  },
  searchIcon: {
    marginRight: 8,
  },
  searchPlaceholder: {
    flex: 1,
    color: '#8e8e93',
    fontSize: 16,
  },
  mainCardContainer: {
    paddingHorizontal: 20,
    marginBottom: 32,
  },
  mainCard: {
    borderRadius: 20,
    padding: 20,
    minHeight: 160,
  },
  mainCardBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    alignSelf: 'flex-start',
    marginBottom: 16,
    gap: 6,
  },
  mainCardBadgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  mainCardContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    flex: 1,
  },
  mainCardText: {
    flex: 1,
  },
  mainCardTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    lineHeight: 30,
  },
  mainCardButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    marginTop: 16,
    alignSelf: 'flex-start',
  },
  mainCardButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  mainCardImage: {
    width: 120,
    height: 120,
    justifyContent: 'center',
    alignItems: 'center',
  },
  doctorImage: {
    width: '100%',
    height: '100%',
    borderRadius: 60,
  },
  dotsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
    marginTop: 16,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  activeDot: {
    backgroundColor: 'white',
  },
  featuresSection: {
    paddingHorizontal: 20,
    marginBottom: 32,
  },
  sectionTitle: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  featuresGrid: {
    gap: 16,
  },
  featuresRow: {
    flexDirection: 'row',
    gap: 16,
  },
  featureCard: {
    flex: 1,
    borderRadius: 16,
    overflow: 'hidden',
  },
  featureGradient: {
    padding: 20,
    minHeight: 120,
  },
  featureContent: {
    flex: 1,
    justifyContent: 'space-between',
  },
  featureIconContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  featureText: {
    flex: 1,
  },
  featureTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
    lineHeight: 20,
  },
  featureSubtitle: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
    lineHeight: 16,
  },
  featureArrow: {
    alignSelf: 'flex-end',
    marginTop: 8,
  },
});

/**
 * Home Screen - Main screen with two analysis options
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  ScrollView,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import Card from '../components/Card';
import Button from '../components/Button';
import { colors } from '../theme/colors';

export default function HomeScreen({ navigation }: any) {

  const handleImageAnalysis = async () => {
    Alert.alert(
      'Phân tích hình ảnh',
      'Chọn cách thức tải hình ảnh',
      [
        {
          text: 'Chụp ảnh',
          onPress: () => takePhoto(),
        },
        {
          text: 'Chọn từ thư viện',
          onPress: () => pickFromLibrary(),
        },
        {
          text: 'Hủy',
          style: 'cancel',
        },
      ]
    );
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
        navigation.navigate('AnalysisScreen', {
          imageUri: result.assets[0].uri,
          isFromFeatures: false
        });
      }
    } catch (error) {
      Alert.alert('Lỗi', 'Không thể chụp ảnh');
    }
  };

  const pickFromLibrary = async () => {
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
        navigation.navigate('AnalysisScreen', {
          imageUri: result.assets[0].uri,
          isFromFeatures: false
        });
      }
    } catch (error) {
      Alert.alert('Lỗi', 'Không thể chọn hình ảnh');
    }
  };

  const handleFeatureAnalysis = () => {
    navigation.navigate('FeatureInputScreen');
  };

  const handleViewHistory = () => {
    navigation.navigate('HistoryTab');
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
      
      <LinearGradient colors={['#1a1a2e', '#16213e'] as any} style={styles.gradient}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.logoContainer}>
            <Ionicons name="medical" size={32} color="#4c6ef5" />
            <Text style={styles.title}>BreastCare AI</Text>
          </View>
          <Text style={styles.subtitle}>
            Hệ thống AI hỗ trợ phân tích ung thư vú
          </Text>
        </View>

        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Main Analysis Options */}
          <View style={styles.optionsContainer}>
            {/* Image Analysis Option */}
            <Card style={styles.optionCard} padding="large">
              <TouchableOpacity
                style={styles.optionContent}
                onPress={handleImageAnalysis}
                activeOpacity={0.8}
              >
                <View style={styles.optionIcon}>
                  <Ionicons name="camera" size={40} color="#4c6ef5" />
                </View>
                <View style={styles.optionInfo}>
                  <Text style={styles.optionTitle}>Phân tích hình ảnh</Text>
                  <Text style={styles.optionDescription}>
                    Sử dụng CNN-GWO để phân tích trực tiếp từ hình ảnh siêu âm
                  </Text>
                  <View style={styles.optionFeatures}>
                    <View style={styles.featureItem}>
                      <Ionicons name="checkmark-circle" size={16} color="#4ade80" />
                      <Text style={styles.featureText}>Nhanh chóng & chính xác</Text>
                    </View>
                    <View style={styles.featureItem}>
                      <Ionicons name="checkmark-circle" size={16} color="#4ade80" />
                      <Text style={styles.featureText}>Deep Learning CNN</Text>
                    </View>
                  </View>
                </View>
                <Ionicons name="chevron-forward" size={24} color="#8e8e93" />
              </TouchableOpacity>
            </Card>

            {/* Feature Analysis Option */}
            <Card style={styles.optionCard} padding="large">
              <TouchableOpacity
                style={styles.optionContent}
                onPress={handleFeatureAnalysis}
                activeOpacity={0.8}
              >
                <View style={styles.optionIcon}>
                  <Ionicons name="calculator" size={40} color="#f59e0b" />
                </View>
                <View style={styles.optionInfo}>
                  <Text style={styles.optionTitle}>Phân tích từ Features</Text>
                  <Text style={styles.optionDescription}>
                    Sử dụng GWO-SVM với các đặc trưng đã được tối ưu hóa
                  </Text>
                  <View style={styles.optionFeatures}>
                    <View style={styles.featureItem}>
                      <Ionicons name="checkmark-circle" size={16} color="#4ade80" />
                      <Text style={styles.featureText}>7 features được chọn</Text>
                    </View>
                    <View style={styles.featureItem}>
                      <Ionicons name="checkmark-circle" size={16} color="#4ade80" />
                      <Text style={styles.featureText}>Grey Wolf Optimizer</Text>
                    </View>
                  </View>
                </View>
                <Ionicons name="chevron-forward" size={24} color="#8e8e93" />
              </TouchableOpacity>
            </Card>
          </View>

          {/* Educational Section */}
          <Card style={styles.educationCard} padding="large">
            <View style={styles.educationHeader}>
              <Ionicons name="school" size={24} color="#8b5cf6" />
              <Text style={styles.educationTitle}>Tại sao có 2 phương pháp?</Text>
            </View>
            <Text style={styles.educationText}>
              • <Text style={styles.boldText}>Phân tích hình ảnh:</Text> Sử dụng mạng neural tích chập (CNN) để trực tiếp phân tích từ hình ảnh siêu âm
            </Text>
            <Text style={styles.educationText}>
              • <Text style={styles.boldText}>Phân tích features:</Text> Sử dụng thuật toán Grey Wolf Optimizer để chọn ra những đặc trưng quan trọng nhất từ Wisconsin Breast Cancer dataset
            </Text>
            <Text style={styles.educationNote}>
              Cả hai phương pháp đều đã được kiểm chứng và có độ chính xác cao trong nghiên cứu khoa học.
            </Text>
          </Card>

          {/* Quick Actions */}
          <View style={styles.quickActions}>
            <TouchableOpacity style={styles.quickActionButton} onPress={handleViewHistory}>
              <Ionicons name="time" size={20} color="#6b7280" />
              <Text style={styles.quickActionText}>Xem lịch sử</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.quickActionButton}>
              <Ionicons name="help-circle" size={20} color="#6b7280" />
              <Text style={styles.quickActionText}>Hướng dẫn</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.quickActionButton}>
              <Ionicons name="settings" size={20} color="#6b7280" />
              <Text style={styles.quickActionText}>Cài đặt</Text>
            </TouchableOpacity>
          </View>

          {/* Disclaimer */}
          <View style={styles.disclaimer}>
            <Ionicons name="warning" size={20} color="#f59e0b" />
            <Text style={styles.disclaimerText}>
              Kết quả chỉ mang tính chất tham khảo. Luôn tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác.
            </Text>
          </View>
        </ScrollView>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  gradient: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 24,
  },
  header: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 32,
    alignItems: 'center',
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    marginLeft: 12,
  },
  subtitle: {
    fontSize: 16,
    color: '#8e8e93',
    textAlign: 'center',
    lineHeight: 22,
  },
  optionsContainer: {
    paddingHorizontal: 20,
    gap: 16,
    marginBottom: 24,
  },
  optionCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  optionContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  optionIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  optionInfo: {
    flex: 1,
  },
  optionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: 'white',
    marginBottom: 6,
  },
  optionDescription: {
    fontSize: 14,
    color: '#8e8e93',
    lineHeight: 20,
    marginBottom: 12,
  },
  optionFeatures: {
    gap: 4,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  featureText: {
    fontSize: 12,
    color: '#8e8e93',
    marginLeft: 6,
  },
  educationCard: {
    marginHorizontal: 20,
    marginBottom: 24,
    backgroundColor: 'rgba(139, 92, 246, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(139, 92, 246, 0.2)',
  },
  educationHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  educationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: 'white',
    marginLeft: 8,
  },
  educationText: {
    fontSize: 14,
    color: '#8e8e93',
    lineHeight: 20,
    marginBottom: 8,
  },
  boldText: {
    fontWeight: '600',
    color: 'white',
  },
  educationNote: {
    fontSize: 13,
    color: '#8b5cf6',
    fontStyle: 'italic',
    marginTop: 8,
    lineHeight: 18,
  },
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  quickActionButton: {
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 12,
    minWidth: 80,
  },
  quickActionText: {
    fontSize: 12,
    color: '#8e8e93',
    marginTop: 6,
    textAlign: 'center',
  },
  disclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginHorizontal: 20,
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
  },
  disclaimerText: {
    flex: 1,
    fontSize: 13,
    color: '#8e8e93',
    lineHeight: 18,
    marginLeft: 12,
  },
});

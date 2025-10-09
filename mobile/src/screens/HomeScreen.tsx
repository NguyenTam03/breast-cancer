/**
 * Home Screen - Main screen with camera functionality
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  SafeAreaView,
  ScrollView,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';
import { useMutation } from '@tanstack/react-query';
import { apiService } from '../services/api';
import { AnalysisResult } from '../types/analysis.types';

export default function HomeScreen() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  // Mutation for image analysis
  const analysisMutation = useMutation({
    mutationFn: (data: { imageUri: string; notes?: string }) =>
      apiService.analyzeImage(data.imageUri, data.notes),
    onSuccess: (result) => {
      setAnalysisResult(result);
    },
    onError: (error: any) => {
      Alert.alert(
        'Lỗi Phân Tích',
        error.response?.data?.detail || 'Có lỗi xảy ra khi phân tích hình ảnh'
      );
    },
  });

  const pickImage = async () => {
    try {
      // Request permission
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (permissionResult.granted === false) {
        Alert.alert('Quyền truy cập', 'Cần quyền truy cập thư viện ảnh để chọn hình');
        return;
      }

      // Pick image
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedImage(result.assets[0].uri);
        setAnalysisResult(null); // Reset previous result
      }
    } catch (error) {
      Alert.alert('Lỗi', 'Không thể chọn hình ảnh');
    }
  };

  const takePhoto = async () => {
    try {
      // Request permission
      const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
      
      if (permissionResult.granted === false) {
        Alert.alert('Quyền truy cập', 'Cần quyền truy cập camera để chụp ảnh');
        return;
      }

      // Take photo
      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        setSelectedImage(result.assets[0].uri);
        setAnalysisResult(null); // Reset previous result
      }
    } catch (error) {
      Alert.alert('Lỗi', 'Không thể chụp ảnh');
    }
  };

  const analyzeImage = () => {
    if (!selectedImage) {
      Alert.alert('Thông báo', 'Vui lòng chọn hoặc chụp một hình ảnh trước');
      return;
    }

    analysisMutation.mutate({ imageUri: selectedImage });
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setAnalysisResult(null);
  };

  const getResultColor = (prediction: string) => {
    return prediction === 'BENIGN' ? '#10B981' : '#EF4444';
  };

  const getResultText = (prediction: string) => {
    return prediction === 'BENIGN' ? 'Lành tính' : 'Ác tính';
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="auto" />
      
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>BreastCare AI</Text>
          <Text style={styles.subtitle}>Phân tích hình ảnh ung thư vú</Text>
        </View>

        {/* Image Selection */}
        <View style={styles.imageSection}>
          {selectedImage ? (
            <View style={styles.imageContainer}>
              <Image source={{ uri: selectedImage }} style={styles.selectedImage} />
              <TouchableOpacity style={styles.resetButton} onPress={resetAnalysis}>
                <Text style={styles.resetButtonText}>Chọn ảnh khác</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <View style={styles.placeholderContainer}>
              <Text style={styles.placeholderText}>Chưa có hình ảnh được chọn</Text>
            </View>
          )}
        </View>

        {/* Action Buttons */}
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.actionButton} onPress={takePhoto}>
            <Text style={styles.buttonText}>Chụp ảnh</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.actionButton} onPress={pickImage}>
            <Text style={styles.buttonText}>Chọn từ thư viện</Text>
          </TouchableOpacity>
        </View>

        {/* Analyze Button */}
        {selectedImage && (
          <TouchableOpacity
            style={[
              styles.analyzeButton,
              analysisMutation.isPending && styles.analyzeButtonDisabled
            ]}
            onPress={analyzeImage}
            disabled={analysisMutation.isPending}
          >
            {analysisMutation.isPending ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <Text style={styles.analyzeButtonText}>Phân tích hình ảnh</Text>
            )}
          </TouchableOpacity>
        )}

        {/* Analysis Result */}
        {analysisResult && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>Kết quả phân tích</Text>
            
            <View style={[
              styles.resultCard,
              { borderLeftColor: getResultColor(analysisResult.prediction) }
            ]}>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Dự đoán:</Text>
                <Text style={[
                  styles.resultValue,
                  { color: getResultColor(analysisResult.prediction) }
                ]}>
                  {getResultText(analysisResult.prediction)}
                </Text>
              </View>
              
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Độ tin cậy:</Text>
                <Text style={styles.resultValue}>
                  {(analysisResult.confidence * 100).toFixed(1)}%
                </Text>
              </View>
              
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Thời gian xử lý:</Text>
                <Text style={styles.resultValue}>
                  {analysisResult.processingTime}ms
                </Text>
              </View>
              
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Ngày phân tích:</Text>
                <Text style={styles.resultValue}>
                  {new Date(analysisResult.analysisDate).toLocaleString('vi-VN')}
                </Text>
              </View>
            </View>

            <Text style={styles.disclaimer}>
              Kết quả này chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác.
            </Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#6B7280',
    textAlign: 'center',
  },
  imageSection: {
    marginBottom: 30,
  },
  imageContainer: {
    alignItems: 'center',
  },
  selectedImage: {
    width: 200,
    height: 200,
    borderRadius: 12,
    marginBottom: 15,
  },
  resetButton: {
    backgroundColor: '#EF4444',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 20,
  },
  resetButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '500',
  },
  placeholderContainer: {
    height: 200,
    backgroundColor: '#E5E7EB',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderStyle: 'dashed',
    borderColor: '#9CA3AF',
  },
  placeholderText: {
    color: '#6B7280',
    fontSize: 16,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    gap: 15,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#3B82F6',
    paddingVertical: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  analyzeButton: {
    backgroundColor: '#10B981',
    paddingVertical: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 30,
  },
  analyzeButtonDisabled: {
    backgroundColor: '#9CA3AF',
  },
  analyzeButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  resultContainer: {
    marginTop: 20,
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 15,
    textAlign: 'center',
  },
  resultCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 20,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
    marginBottom: 20,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  resultLabel: {
    fontSize: 16,
    color: '#6B7280',
    fontWeight: '500',
  },
  resultValue: {
    fontSize: 16,
    color: '#1F2937',
    fontWeight: '600',
  },
  disclaimer: {
    fontSize: 14,
    color: '#F59E0B',
    textAlign: 'center',
    fontStyle: 'italic',
    lineHeight: 20,
  },
});

/**
 * Dark Theme Home Screen with modern minimal design
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Alert,
  SafeAreaView,
  ScrollView,
  Dimensions,
  StatusBar,
  Animated,
  Switch,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';
import { useMutation } from '@tanstack/react-query';
import { apiService } from '../services/api';
import { AnalysisResult } from '../types/analysis.types';
import Button from '../components/Button';
import Card from '../components/Card';
import { colors } from '../theme/colors';

const { width } = Dimensions.get('window');

export default function DarkHomeScreen() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [fadeAnim] = useState(new Animated.Value(1));
  const [enableCrop, setEnableCrop] = useState<boolean>(false);

  // Mutation for image analysis
  const analysisMutation = useMutation({
    mutationFn: (data: { imageUri: string; notes?: string }) =>
      apiService.analyzeImage(data.imageUri, data.notes),
    onSuccess: (result) => {
      setAnalysisResult(result);
      // Animate result appearance
      Animated.sequence([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        }),
      ]).start();
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
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (permissionResult.granted === false) {
        Alert.alert('Quyền truy cập', 'Cần quyền truy cập thư viện ảnh để chọn hình');
        return;
      }

      const imagePickerOptions: any = {
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: enableCrop,
        quality: 0.8,
      };

      // Only add aspect ratio if cropping is enabled
      if (enableCrop) {
        imagePickerOptions.aspect = [1, 1];
      }

      const result = await ImagePicker.launchImageLibraryAsync(imagePickerOptions);

      if (!result.canceled && result.assets[0]) {
        setSelectedImage(result.assets[0].uri);
        setAnalysisResult(null);
      }
    } catch (error) {
      Alert.alert('Lỗi', 'Không thể chọn hình ảnh');
    }
  };

  const takePhoto = async () => {
    try {
      const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
      
      if (permissionResult.granted === false) {
        Alert.alert('Quyền truy cập', 'Cần quyền truy cập camera để chụp ảnh');
        return;
      }

      const cameraOptions: any = {
        allowsEditing: enableCrop,
        quality: 0.8,
      };

      // Only add aspect ratio if cropping is enabled
      if (enableCrop) {
        cameraOptions.aspect = [1, 1];
      }

      const result = await ImagePicker.launchCameraAsync(cameraOptions);

      if (!result.canceled && result.assets[0]) {
        setSelectedImage(result.assets[0].uri);
        setAnalysisResult(null);
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
    return prediction === 'BENIGN' ? colors.status.success : colors.status.error;
  };

  const getResultText = (prediction: string) => {
    return prediction === 'BENIGN' ? 'Lành tính' : 'Ác tính';
  };

  const getStatusIndicator = (prediction: string) => {
    return prediction === 'BENIGN' ? '●' : '●';
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.background.primary} />
      
      <LinearGradient
        colors={colors.gradient.primary as any}
        style={styles.gradient}
      >
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>BreastCare AI</Text>
            <Text style={styles.subtitle}>
              Hỗ trợ phân tích hình ảnh ung thư vú bằng trí tuệ nhân tạo
            </Text>
          </View>

          {/* Image Upload Section */}
          <Card style={styles.imageCard} padding="large">
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Tải lên hình ảnh</Text>
              <Text style={styles.sectionSubtitle}>
                Chọn hoặc chụp hình ảnh để phân tích
              </Text>
            </View>
            
            {selectedImage ? (
              <View style={styles.imageContainer}>
                <View style={styles.imageWrapper}>
                  <Image source={{ uri: selectedImage }} style={styles.selectedImage} />
                  <View style={styles.imageOverlay}>
                    <Button
                      title="Thay đổi"
                      onPress={resetAnalysis}
                      variant="outline"
                      size="small"
                    />
                  </View>
                </View>
              </View>
            ) : (
              <View style={styles.placeholderContainer}>
                <View style={styles.placeholderIcon}>
                  <Text style={styles.placeholderIconText}>+</Text>
                </View>
                <Text style={styles.placeholderText}>
                  Chưa có hình ảnh
                </Text>
                <Text style={styles.placeholderSubtext}>
                  Nhấn vào một trong các nút bên dưới để thêm hình ảnh
                </Text>
              </View>
            )}

            {/* Crop Option */}
            <View style={styles.optionContainer}>
              <View style={styles.optionRow}>
                <Text style={styles.optionLabel}>Cắt ảnh thành hình vuông</Text>
                <Switch
                  value={enableCrop}
                  onValueChange={setEnableCrop}
                  trackColor={{ 
                    false: colors.background.tertiary, 
                    true: colors.status.info 
                  }}
                  thumbColor={enableCrop ? colors.text.primary : colors.text.tertiary}
                />
              </View>
              <Text style={styles.optionDescription}>
                {enableCrop 
                  ? 'Ảnh sẽ được cắt thành hình vuông trước khi phân tích'
                  : 'Sử dụng toàn bộ khung hình để phân tích'
                }
              </Text>
            </View>

            <View style={styles.buttonContainer}>
              <Button
                title="Chụp ảnh"
                onPress={takePhoto}
                style={styles.actionButton}
              />
              
              <Button
                title="Thư viện"
                onPress={pickImage}
                variant="outline"
                style={styles.actionButton}
              />
            </View>
          </Card>

          {/* Analyze Section */}
          {selectedImage && (
            <Card style={styles.analyzeCard} padding="large">
              <Button
                title={analysisMutation.isPending ? "Đang phân tích..." : "Phân tích hình ảnh"}
                onPress={analyzeImage}
                loading={analysisMutation.isPending}
                size="large"
                style={styles.analyzeButton}
              />
            </Card>
          )}

          {/* Results Section */}
          {analysisResult && (
            <Animated.View style={{ opacity: fadeAnim }}>
              <Card style={styles.resultCard} padding="large">
                <View style={styles.resultHeader}>
                  <Text style={styles.resultTitle}>Kết quả phân tích</Text>
                  <View style={styles.resultBadge}>
                    <Text style={styles.resultId}>#{analysisResult.id.slice(0, 8)}</Text>
                  </View>
                </View>
                
                <View style={[
                  styles.predictionContainer,
                  { borderLeftColor: getResultColor(analysisResult.prediction) }
                ]}>
                  <View style={styles.predictionRow}>
                    <Text style={[
                      styles.statusIndicator,
                      { color: getResultColor(analysisResult.prediction) }
                    ]}>
                      {getStatusIndicator(analysisResult.prediction)}
                    </Text>
                    <View style={styles.predictionInfo}>
                      <Text style={styles.predictionLabel}>Kết quả dự đoán</Text>
                      <Text style={[
                        styles.predictionValue,
                        { color: getResultColor(analysisResult.prediction) }
                      ]}>
                        {getResultText(analysisResult.prediction)}
                      </Text>
                    </View>
                  </View>
                </View>

                <View style={styles.metricsContainer}>
                  <View style={styles.metric}>
                    <Text style={styles.metricValue}>
                      {(analysisResult.confidence * 100).toFixed(1)}%
                    </Text>
                    <Text style={styles.metricLabel}>Độ tin cậy</Text>
                  </View>
                  
                  <View style={styles.metricDivider} />
                  
                  <View style={styles.metric}>
                    <Text style={styles.metricValue}>
                      {analysisResult.processingTime}ms
                    </Text>
                    <Text style={styles.metricLabel}>Thời gian xử lý</Text>
                  </View>
                  
                  <View style={styles.metricDivider} />
                  
                  <View style={styles.metric}>
                    <Text style={styles.metricValue}>
                      {new Date(analysisResult.analysisDate).toLocaleDateString('vi-VN')}
                    </Text>
                    <Text style={styles.metricLabel}>Ngày phân tích</Text>
                  </View>
                </View>

                <View style={styles.disclaimer}>
                  <Text style={styles.disclaimerTitle}>Lưu ý quan trọng</Text>
                  <Text style={styles.disclaimerText}>
                    Kết quả này chỉ mang tính chất tham khảo và hỗ trợ. 
                    Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác.
                  </Text>
                </View>
              </Card>
            </Animated.View>
          )}

          <View style={styles.footer}>
            <Text style={styles.footerText}>
              Được hỗ trợ bởi công nghệ AI tiên tiến
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
    backgroundColor: colors.background.primary,
  },
  gradient: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
    paddingTop: 20,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: colors.text.primary,
    marginBottom: 12,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: colors.text.secondary,
    textAlign: 'center',
    lineHeight: 24,
    paddingHorizontal: 20,
  },
  imageCard: {
    marginBottom: 24,
  },
  sectionHeader: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: colors.text.primary,
    marginBottom: 8,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: colors.text.tertiary,
    lineHeight: 20,
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 24,
  },
  imageWrapper: {
    position: 'relative',
  },
  selectedImage: {
    width: width * 0.65,
    height: width * 0.65,
    borderRadius: 24,
    borderWidth: 2,
    borderColor: colors.border.secondary,
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 12,
    right: 12,
  },
  placeholderContainer: {
    alignItems: 'center',
    paddingVertical: 48,
    marginBottom: 24,
  },
  placeholderIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: colors.background.tertiary,
    borderWidth: 2,
    borderColor: colors.border.secondary,
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  placeholderIconText: {
    fontSize: 48,
    color: colors.text.tertiary,
    fontWeight: '300',
  },
  placeholderText: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.secondary,
    marginBottom: 8,
  },
  placeholderSubtext: {
    fontSize: 14,
    color: colors.text.tertiary,
    textAlign: 'center',
    lineHeight: 20,
    paddingHorizontal: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 16,
  },
  actionButton: {
    flex: 1,
  },
  optionContainer: {
    marginBottom: 20,
    padding: 16,
    backgroundColor: colors.background.tertiary,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  optionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  optionLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
    flex: 1,
  },
  optionDescription: {
    fontSize: 13,
    color: colors.text.tertiary,
    lineHeight: 18,
  },
  analyzeCard: {
    marginBottom: 24,
  },
  analyzeButton: {
    backgroundColor: colors.status.info,
  },
  resultCard: {
    marginBottom: 24,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  resultTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  resultBadge: {
    backgroundColor: colors.background.tertiary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  resultId: {
    fontSize: 12,
    color: colors.text.tertiary,
    fontWeight: '600',
  },
  predictionContainer: {
    borderLeftWidth: 4,
    paddingLeft: 20,
    marginBottom: 24,
    backgroundColor: colors.background.tertiary,
    padding: 20,
    borderRadius: 16,
  },
  predictionRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusIndicator: {
    fontSize: 32,
    marginRight: 16,
  },
  predictionInfo: {
    flex: 1,
  },
  predictionLabel: {
    fontSize: 14,
    color: colors.text.tertiary,
    marginBottom: 6,
    fontWeight: '500',
  },
  predictionValue: {
    fontSize: 28,
    fontWeight: 'bold',
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingVertical: 24,
    backgroundColor: colors.background.tertiary,
    borderRadius: 16,
    marginBottom: 24,
  },
  metric: {
    alignItems: 'center',
    flex: 1,
  },
  metricDivider: {
    width: 1,
    height: 40,
    backgroundColor: colors.border.secondary,
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.text.primary,
    marginBottom: 6,
  },
  metricLabel: {
    fontSize: 12,
    color: colors.text.tertiary,
    textAlign: 'center',
    fontWeight: '500',
  },
  disclaimer: {
    padding: 20,
    backgroundColor: colors.background.tertiary,
    borderRadius: 16,
    borderLeftWidth: 4,
    borderLeftColor: colors.status.warning,
  },
  disclaimerTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: 8,
  },
  disclaimerText: {
    fontSize: 14,
    color: colors.text.secondary,
    lineHeight: 22,
  },
  footer: {
    alignItems: 'center',
    marginTop: 32,
  },
  footerText: {
    color: colors.text.tertiary,
    fontSize: 14,
    fontStyle: 'italic',
  },
});

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Animated,
  Alert,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import { useMutation } from '@tanstack/react-query';
import { apiService } from '../services/api';
import { AnalysisResult } from '../types/analysis.types';
import Button from '../components/Button';
import Card from '../components/Card';
import { colors } from '../theme/colors';

export default function AnalysisScreen({ route, navigation }: any) {
  const { imageUri, analysisResult: existingResult, isFromHistory } = route.params;
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(existingResult || null);
  const [fadeAnim] = useState(new Animated.Value(1));

  // API Base URL for images
  const API_BASE_URL = 'http://10.0.2.2:8000/api/v1';

  // Determine the image source
  const imageSource = isFromHistory && existingResult?.imageUrl 
    ? `${API_BASE_URL}${existingResult.imageUrl}`
    : imageUri;

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

  const analyzeImage = () => {
    analysisMutation.mutate({ imageUri });
  };

  const getResultColor = (prediction: string) => {
    return prediction === 'BENIGN' ? '#4ade80' : '#f87171';
  };

  const getResultText = (prediction: string) => {
    return prediction === 'BENIGN' ? 'Lành tính' : 'Ác tính';
  };

  const getResultIcon = (prediction: string) => {
    return prediction === 'BENIGN' ? 'checkmark-circle' : 'alert-circle';
  };

  const shareResult = () => {
    // Implement share functionality
    Alert.alert('Chia sẻ', 'Tính năng chia sẻ sẽ được phát triển');
  };

  const saveResult = () => {
    if (!analysisResult) return;
    
    Alert.alert(
      'Lưu kết quả', 
      'Kết quả đã được lưu vào lịch sử',
      [
        {
          text: 'OK',
          onPress: () => {
            // Navigate to HistoryTab to show the saved result
            navigation.navigate('HistoryTab');
          }
        }
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
      
      <LinearGradient colors={['#1a1a2e', '#16213e'] as any} style={styles.gradient}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Phân tích hình ảnh</Text>
          <TouchableOpacity style={styles.headerButton}>
            <Ionicons name="help-circle-outline" size={24} color="white" />
          </TouchableOpacity>
        </View>

        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Image Display */}
          <View style={styles.imageSection}>
            <View style={styles.imageContainer}>
              <Image source={{ uri: imageSource }} style={styles.image} />
              <LinearGradient 
                colors={['transparent', 'rgba(0,0,0,0.7)'] as any}
                style={styles.imageOverlay}
              >
                <View style={styles.imageInfo}>
                  <Ionicons name="camera" size={16} color="white" />
                  <Text style={styles.imageInfoText}>
                    {isFromHistory ? 'Ảnh từ lịch sử' : 'Hình ảnh được chọn'}
                  </Text>
                </View>
              </LinearGradient>
            </View>
          </View>

          {/* Analysis Section */}
          {!analysisResult && !analysisMutation.isPending && (
            <Card style={styles.analysisCard} padding="large">
              <View style={styles.analysisContent}>
                <View style={styles.analysisIcon}>
                  <Ionicons name="analytics" size={48} color="#4c6ef5" />
                </View>
                <Text style={styles.analysisTitle}>Sẵn sàng phân tích</Text>
                <Text style={styles.analysisSubtitle}>
                  Hệ thống AI sẽ phân tích hình ảnh của bạn và đưa ra kết quả dự đoán
                </Text>
                <Button
                  title="Bắt đầu phân tích"
                  onPress={analyzeImage}
                  size="large"
                  style={styles.analyzeButton}
                />
              </View>
            </Card>
          )}

          {/* Loading Section */}
          {analysisMutation.isPending && (
            <Card style={styles.loadingCard} padding="large">
              <View style={styles.loadingContent}>
                <View style={styles.loadingIcon}>
                  <Animated.View style={{ transform: [{ rotate: '45deg' }] }}>
                    <Ionicons name="analytics" size={48} color="#4c6ef5" />
                  </Animated.View>
                </View>
                <Text style={styles.loadingTitle}>Đang phân tích...</Text>
                <Text style={styles.loadingSubtitle}>
                  Vui lòng đợi trong khi AI phân tích hình ảnh của bạn
                </Text>
                <View style={styles.progressBar}>
                  <View style={styles.progressIndicator} />
                </View>
              </View>
            </Card>
          )}

          {/* Results Section */}
          {analysisResult && (
            <Animated.View style={{ opacity: fadeAnim }}>
              <Card style={styles.resultCard} padding="large">
                <View style={styles.resultHeader}>
                  <View style={styles.resultIconContainer}>
                    <Ionicons 
                      name={getResultIcon(analysisResult.prediction)} 
                      size={32} 
                      color={getResultColor(analysisResult.prediction)} 
                    />
                  </View>
                  <View style={styles.resultInfo}>
                    <Text style={styles.resultLabel}>Kết quả phân tích</Text>
                    <Text style={[
                      styles.resultValue,
                      { color: getResultColor(analysisResult.prediction) }
                    ]}>
                      {getResultText(analysisResult.prediction)}
                    </Text>
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
                </View>

                <View style={styles.disclaimer}>
                  <Ionicons name="information-circle-outline" size={20} color="#fbbf24" />
                  <View style={styles.disclaimerText}>
                    <Text style={styles.disclaimerTitle}>Lưu ý quan trọng</Text>
                    <Text style={styles.disclaimerContent}>
                      Kết quả này chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa.
                    </Text>
                  </View>
                </View>

                <View style={styles.actionButtons}>
                  <Button
                    title="Lưu kết quả"
                    onPress={saveResult}
                    variant="outline"
                    style={styles.actionButton}
                  />
                  <Button
                    title="Chia sẻ"
                    onPress={shareResult}
                    style={styles.actionButton}
                  />
                </View>
              </Card>
            </Animated.View>
          )}
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 24,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  headerButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  imageSection: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  imageContainer: {
    position: 'relative',
    borderRadius: 20,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: 300,
    borderRadius: 20,
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 16,
  },
  imageInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  imageInfoText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  analysisCard: {
    marginHorizontal: 20,
    marginBottom: 24,
  },
  analysisContent: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  analysisIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(76, 110, 245, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  analysisTitle: {
    color: 'white',
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  analysisSubtitle: {
    color: '#8e8e93',
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 24,
  },
  analyzeButton: {
    backgroundColor: '#4c6ef5',
    width: '100%',
  },
  loadingCard: {
    marginHorizontal: 20,
    marginBottom: 24,
  },
  loadingContent: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  loadingIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(76, 110, 245, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  loadingTitle: {
    color: 'white',
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  loadingSubtitle: {
    color: '#8e8e93',
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 24,
  },
  progressBar: {
    width: '100%',
    height: 4,
    backgroundColor: 'rgba(76, 110, 245, 0.2)',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressIndicator: {
    width: '60%',
    height: '100%',
    backgroundColor: '#4c6ef5',
    borderRadius: 2,
  },
  resultCard: {
    marginHorizontal: 20,
    marginBottom: 24,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  resultIconContainer: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  resultInfo: {
    flex: 1,
  },
  resultLabel: {
    color: '#8e8e93',
    fontSize: 14,
    marginBottom: 4,
  },
  resultValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingVertical: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
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
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 6,
  },
  metricLabel: {
    fontSize: 12,
    color: '#8e8e93',
    textAlign: 'center',
    fontWeight: '500',
  },
  disclaimer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: 'rgba(251, 191, 36, 0.1)',
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#fbbf24',
    marginBottom: 24,
  },
  disclaimerText: {
    flex: 1,
    marginLeft: 12,
  },
  disclaimerTitle: {
    color: '#fbbf24',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  disclaimerContent: {
    color: '#8e8e93',
    fontSize: 13,
    lineHeight: 18,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
  },
});

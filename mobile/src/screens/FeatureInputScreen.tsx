import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  ScrollView,
  TextInput,
  Alert,
  Switch,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useMutation, useQuery } from '@tanstack/react-query';
import { apiService } from '../services/api';
import { FeatureInfo, AnalysisResult } from '../types/analysis.types';
import Button from '../components/Button';
import Card from '../components/Card';
import { colors } from '../theme/colors';

export default function FeatureInputScreen({ navigation }: any) {
  const [featureValues, setFeatureValues] = useState<Record<string, string>>({});
  const [useGWO, setUseGWO] = useState(true);
  const [notes, setNotes] = useState('');
  const [showEducationalInfo, setShowEducationalInfo] = useState(false);

  // Fetch features info from API
  const { data: featuresInfo, isLoading: isLoadingFeatures, error: featuresError } = useQuery({
    queryKey: ['features-info'],
    queryFn: () => apiService.getFeaturesInfo(),
  });

  // Initialize feature values with default values when data loads
  useEffect(() => {
    if (featuresInfo?.features) {
      const initialValues: Record<string, string> = {};
      featuresInfo.features.forEach((feature: FeatureInfo) => {
        initialValues[feature.name] = feature.default_value.toString();
      });
      setFeatureValues(initialValues);
    }
  }, [featuresInfo]);

  // Mutation for feature analysis
  const analysisMutation = useMutation({
    mutationFn: (data: { featureData: Record<string, number>; useGWO: boolean; notes?: string }) =>
      apiService.analyzeFeatures(data.featureData, data.useGWO, data.notes),
    onSuccess: (result: AnalysisResult) => {
      // Navigate to analysis results screen
      navigation.navigate('Analysis', {
        analysisResult: result,
        isFromFeatures: true
      });
    },
    onError: (error: any) => {
      Alert.alert(
        'Lỗi Phân Tích',
        error.response?.data?.detail || 'Có lỗi xảy ra khi phân tích features'
      );
    },
  });

  const handleFeatureChange = (featureName: string, value: string) => {
    setFeatureValues(prev => ({
      ...prev,
      [featureName]: value
    }));
  };

  const validateAndAnalyze = () => {
    if (!featuresInfo?.features) {
      Alert.alert('Lỗi', 'Không thể tải thông tin features');
      return;
    }

    // Convert string values to numbers and validate
    const numericFeatures: Record<string, number> = {};
    const errors: string[] = [];

    featuresInfo.features.forEach((feature: FeatureInfo) => {
      const value = featureValues[feature.name];
      
      // Check if value exists and is not empty
      if (!value || value.trim() === '') {
        if (feature.is_required) {
          errors.push(`${feature.description} là bắt buộc`);
        }
        return;
      }

      // Clean the input value
      const cleanedValue = value.trim().replace(/,/g, '.'); // Replace comma with dot for decimal
      
      // More robust number validation
      if (!/^-?\d*\.?\d+$/.test(cleanedValue)) {
        errors.push(`${feature.description} phải là số hợp lệ (ví dụ: 12.5)`);
        return;
      }

      const numValue = parseFloat(cleanedValue);
      
      // Check for valid number
      if (isNaN(numValue) || !isFinite(numValue)) {
        errors.push(`${feature.description} phải là số hợp lệ`);
      } else if (numValue < 0) {
        errors.push(`${feature.description} phải là số dương hoặc bằng 0`);
      } else {
        // Ensure we store as proper number
        numericFeatures[feature.name] = Number(numValue);
      }
    });

    if (errors.length > 0) {
      Alert.alert('Lỗi Validation', errors.join('\n'));
      return;
    }

    if (Object.keys(numericFeatures).length === 0) {
      Alert.alert('Lỗi', 'Vui lòng nhập ít nhất một feature hợp lệ');
      return;
    }

    // Log for debugging
    console.log('Sending numeric features:', numericFeatures);
    console.log('Feature types:', Object.entries(numericFeatures).map(([key, value]) => `${key}: ${typeof value}`));

    // Start analysis
    analysisMutation.mutate({
      featureData: numericFeatures,
      useGWO: useGWO,
      notes: notes.trim() || undefined
    });
  };

  const resetToDefaults = () => {
    if (featuresInfo?.features) {
      const defaultValues: Record<string, string> = {};
      featuresInfo.features.forEach((feature: FeatureInfo) => {
        defaultValues[feature.name] = feature.default_value.toString();
      });
      setFeatureValues(defaultValues);
    }
  };

  const renderFeatureInput = (feature: FeatureInfo) => (
    <View key={feature.name} style={styles.featureInputContainer}>
      <View style={styles.featureHeader}>
        <Text style={styles.featureLabel}>{feature.description}</Text>
        <TouchableOpacity
          style={styles.infoButton}
          onPress={() => Alert.alert(
            feature.description,
            `Tên kỹ thuật: ${feature.name}\nGiá trị mặc định: ${feature.default_value}${feature.is_required ? '\n*Bắt buộc' : ''}`
          )}
        >
          <Ionicons name="information-circle-outline" size={16} color="#8e8e93" />
        </TouchableOpacity>
      </View>
      <TextInput
        style={styles.featureInput}
        value={featureValues[feature.name] || ''}
        onChangeText={(value) => handleFeatureChange(feature.name, value)}
        placeholder={`Nhập giá trị (mặc định: ${feature.default_value})`}
        placeholderTextColor="#8e8e93"
        keyboardType="numeric"
      />
    </View>
  );

  if (isLoadingFeatures) {
    return (
      <SafeAreaView style={styles.container}>
        <LinearGradient colors={['#1a1a2e', '#16213e'] as any} style={styles.gradient}>
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4c6ef5" />
            <Text style={styles.loadingText}>Đang tải thông tin features...</Text>
          </View>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  if (featuresError) {
    return (
      <SafeAreaView style={styles.container}>
        <LinearGradient colors={['#1a1a2e', '#16213e'] as any} style={styles.gradient}>
          <View style={styles.errorContainer}>
            <Ionicons name="warning-outline" size={48} color="#f87171" />
            <Text style={styles.errorText}>Không thể tải thông tin features</Text>
            <Button title="Thử lại" onPress={() => navigation.goBack()} />
          </View>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
      
      <LinearGradient colors={['#1a1a2e', '#16213e'] as any} style={styles.gradient}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Nhập Features</Text>
          <TouchableOpacity 
            style={styles.headerButton}
            onPress={() => setShowEducationalInfo(!showEducationalInfo)}
          >
            <Ionicons name="school-outline" size={24} color="white" />
          </TouchableOpacity>
        </View>

        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Educational Info */}
          {showEducationalInfo && (
            <Card style={styles.educationalCard} padding="large">
              <View style={styles.educationalHeader}>
                <Ionicons name="bulb" size={24} color="#fbbf24" />
                <Text style={styles.educationalTitle}>Thông tin về Features</Text>
              </View>
              <Text style={styles.educationalText}>
                Các features này được chọn bởi Grey Wolf Optimizer (GWO) từ tập dữ liệu Wisconsin Breast Cancer. 
                Chúng là những đặc trưng quan trọng nhất để phân loại khối u.
              </Text>
              <Text style={styles.educationalSubtext}>
                • <Text style={styles.boldText}>Mean:</Text> Giá trị trung bình
              </Text>
              <Text style={styles.educationalSubtext}>
                • <Text style={styles.boldText}>SE:</Text> Sai số chuẩn (Standard Error)
              </Text>
              <Text style={styles.educationalSubtext}>
                • <Text style={styles.boldText}>Worst:</Text> Giá trị xấu nhất trong 3 giá trị lớn nhất
              </Text>
            </Card>
          )}

          {/* Method Selection */}
          <Card style={styles.methodCard} padding="large">
            <View style={styles.methodHeader}>
              <Text style={styles.methodTitle}>Phương pháp phân tích</Text>
              <Switch
                value={useGWO}
                onValueChange={setUseGWO}
                trackColor={{ false: '#8e8e93', true: '#4c6ef5' }}
                thumbColor={useGWO ? '#ffffff' : '#f4f3f4'}
              />
            </View>
            <Text style={styles.methodDescription}>
              {useGWO 
                ? 'Sử dụng GWO-SVM (7 features được chọn bởi Grey Wolf Optimizer)'
                : 'Sử dụng Full-SVM (tất cả 30 features)'
              }
            </Text>
          </Card>

          {/* Feature Inputs */}
          <Card style={styles.featuresCard} padding="large">
            <View style={styles.featuresHeader}>
              <Text style={styles.featuresTitle}>
                Features cần nhập ({featuresInfo?.features?.length || 0})
              </Text>
              <TouchableOpacity style={styles.resetButton} onPress={resetToDefaults}>
                <Ionicons name="refresh" size={16} color="#4c6ef5" />
                <Text style={styles.resetButtonText}>Reset</Text>
              </TouchableOpacity>
            </View>
            
            {featuresInfo?.features?.map(renderFeatureInput)}
          </Card>

          {/* Notes */}
          <Card style={styles.notesCard} padding="large">
            <Text style={styles.notesTitle}>Ghi chú (tùy chọn)</Text>
            <TextInput
              style={styles.notesInput}
              value={notes}
              onChangeText={setNotes}
              placeholder="Thêm ghi chú về phân tích..."
              placeholderTextColor="#8e8e93"
              multiline
              numberOfLines={3}
            />
          </Card>

          {/* Analyze Button */}
          <View style={styles.buttonContainer}>
            <Button
              title={analysisMutation.isPending ? "Đang phân tích..." : "Bắt đầu phân tích"}
              onPress={validateAndAnalyze}
              disabled={analysisMutation.isPending}
              size="large"
              style={styles.analyzeButton}
            />
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  loadingText: {
    color: 'white',
    fontSize: 16,
    marginTop: 16,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  errorText: {
    color: '#f87171',
    fontSize: 16,
    marginVertical: 16,
    textAlign: 'center',
  },
  educationalCard: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  educationalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  educationalTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
  educationalText: {
    color: '#8e8e93',
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 12,
  },
  educationalSubtext: {
    color: '#8e8e93',
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 4,
  },
  boldText: {
    fontWeight: '600',
    color: '#ffffff',
  },
  methodCard: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  methodHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  methodTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  methodDescription: {
    color: '#8e8e93',
    fontSize: 14,
    lineHeight: 18,
  },
  featuresCard: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  featuresHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  featuresTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: 'rgba(76, 110, 245, 0.1)',
    borderRadius: 8,
  },
  resetButtonText: {
    color: '#4c6ef5',
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  featureInputContainer: {
    marginBottom: 16,
  },
  featureHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  featureLabel: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
  },
  infoButton: {
    padding: 4,
  },
  featureInput: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: 'white',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  notesCard: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  notesTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  notesInput: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: 'white',
    fontSize: 14,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    minHeight: 80,
    textAlignVertical: 'top',
  },
  buttonContainer: {
    paddingHorizontal: 20,
    marginTop: 8,
  },
  analyzeButton: {
    backgroundColor: '#4c6ef5',
  },
});

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { colors } from '../theme/colors';
import { apiService } from '../services/api';

interface FeatureInfo {
  index: number;
  name: string;
  description: string;
  display_order: number;
}

interface FeatureInputScreenProps {
  navigation: any;
}

export default function FeatureInputScreen({ navigation }: FeatureInputScreenProps) {
  const [features, setFeatures] = useState<string[]>([]);
  const [featureInfo, setFeatureInfo] = useState<FeatureInfo[]>([]);
  const [notes, setNotes] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingInfo, setIsLoadingInfo] = useState(true);

  useEffect(() => {
    loadFeatureInfo();
  }, []);

  const loadFeatureInfo = async () => {
    try {
      setIsLoadingInfo(true);
      // Use apiService to get feature information
      const data = await apiService.getFeatureInfo();
      
      setFeatureInfo(data.selectedFeatures);
      // Initialize features array with empty strings
      setFeatures(new Array(data.selectedFeatures.length).fill(''));
    } catch (error) {
      console.error('Failed to load feature info:', error);
      Alert.alert('Lỗi', 'Không thể tải thông tin features. Sử dụng dữ liệu mặc định.');
      // Set default 7 features if API fails
      const defaultFeatures = [
        { index: 0, name: 'radius_mean', description: 'Bán kính trung bình', display_order: 0 },
        { index: 1, name: 'texture_se', description: 'Độ lệch chuẩn texture', display_order: 1 },
        { index: 2, name: 'compactness_se', description: 'Độ lệch chuẩn độ compact', display_order: 2 },
        { index: 3, name: 'radius_worst', description: 'Bán kính tệ nhất', display_order: 3 },
        { index: 4, name: 'texture_worst', description: 'Texture tệ nhất', display_order: 4 },
        { index: 5, name: 'smoothness_worst', description: 'Độ mịn tệ nhất', display_order: 5 },
        { index: 6, name: 'compactness_worst', description: 'Độ compact tệ nhất', display_order: 6 }
      ];
      setFeatureInfo(defaultFeatures);
      setFeatures(new Array(7).fill(''));
    } finally {
      setIsLoadingInfo(false);
    }
  };

  const handleFeatureChange = (index: number, value: string) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const validateFeatures = (): boolean => {
    for (let i = 0; i < features.length; i++) {
      const value = features[i].trim();
      if (value === '') {
        Alert.alert('Lỗi', `Vui lòng nhập giá trị cho ${featureInfo[i]?.description}`);
        return false;
      }
      
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        Alert.alert('Lỗi', `Giá trị "${value}" không hợp lệ cho ${featureInfo[i]?.description}`);
        return false;
      }
      
      if (numValue < 0) {
        Alert.alert('Lỗi', `Giá trị không được âm cho ${featureInfo[i]?.description}`);
        return false;
      }
    }
    return true;
  };

  const handleSubmit = async () => {
    if (!validateFeatures()) {
      return;
    }

    try {
      setIsLoading(true);
      
      // Convert string values to numbers
      const featureValues = features.map(f => parseFloat(f.trim()));
      
      // Call API to predict
      const response = await apiService.predictFeatures({
        features: featureValues,
        notes: notes.trim() || undefined
      });

      // Navigate to analysis screen with results
      navigation.navigate('Analysis', { 
        analysisResult: response,
        analysisType: 'features'
      });

    } catch (error: any) {
      console.error('Feature prediction failed:', error);
      Alert.alert('Lỗi', error.message || 'Không thể phân tích dữ liệu');
    } finally {
      setIsLoading(false);
    }
  };

  const clearAllFields = () => {
    setFeatures(new Array(features.length).fill(''));
    setNotes('');
  };

  const fillSampleData = () => {
    // Sample data for testing (these values should result in a prediction)
    const sampleValues = ['14.127', '0.905', '0.02592', '16.14', '23.96', '0.1149', '0.1876'];
    setFeatures(sampleValues);
    setNotes('Dữ liệu mẫu để test');
  };

  if (isLoadingInfo) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.button.primary} />
          <Text style={styles.loadingText}>Đang tải thông tin features...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView 
        style={styles.container} 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity 
            style={styles.backButton} 
            onPress={() => navigation.goBack()}
          >
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Phân tích bằng Features</Text>
          <TouchableOpacity style={styles.infoButton} onPress={() => Alert.alert('Thông tin', 'Nhập 7 giá trị features để phân tích ung thư vú bằng mô hình SVM với GWO feature selection')}>
            <Ionicons name="information-circle-outline" size={24} color="white" />
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          {/* Description Card */}
          <View style={styles.descriptionCard}>
            <Text style={styles.descriptionTitle}>7 Features được chọn bởi GWO</Text>
            <Text style={styles.descriptionText}>
              Nhập giá trị cho 7 features đã được tối ưu hóa bởi Grey Wolf Optimizer 
              từ 30 features gốc của Wisconsin Breast Cancer dataset.
            </Text>
          </View>

          {/* Feature Input Fields */}
          <View style={styles.inputSection}>
            <Text style={styles.sectionTitle}>Nhập Giá Trị Features</Text>
            
            {featureInfo.map((feature, index) => (
              <View key={feature.index} style={styles.inputCard}>
                <Text style={styles.inputLabel}>
                  {index + 1}. {feature.description}
                </Text>
                <Text style={styles.inputSubLabel}>({feature.name})</Text>
                <TextInput
                  style={styles.textInput}
                  value={features[index] || ''}
                  onChangeText={(value) => handleFeatureChange(index, value)}
                  placeholder="Nhập giá trị số..."
                  placeholderTextColor="#8e8e93"
                  keyboardType="decimal-pad"
                />
              </View>
            ))}
          </View>

          {/* Notes Section */}
          <View style={styles.inputSection}>
            <Text style={styles.sectionTitle}>Ghi Chú (Tùy chọn)</Text>
            <View style={styles.inputCard}>
              <TextInput
                style={[styles.textInput, styles.notesInput]}
                value={notes}
                onChangeText={setNotes}
                placeholder="Nhập ghi chú về dữ liệu..."
                placeholderTextColor="#8e8e93"
                multiline
                numberOfLines={3}
              />
            </View>
          </View>

          {/* Action Buttons */}
          <View style={styles.actionSection}>
            <TouchableOpacity style={styles.secondaryButton} onPress={fillSampleData}>
              <Ionicons name="flask-outline" size={20} color="#4c6ef5" />
              <Text style={styles.secondaryButtonText}>Dữ liệu mẫu</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.secondaryButton} onPress={clearAllFields}>
              <Ionicons name="refresh-outline" size={20} color="#ff6b6b" />
              <Text style={[styles.secondaryButtonText, { color: '#ff6b6b' }]}>Xóa tất cả</Text>
            </TouchableOpacity>
          </View>

          {/* Submit Button */}
          <TouchableOpacity 
            style={[styles.submitButton, isLoading && styles.submitButtonDisabled]} 
            onPress={handleSubmit}
            disabled={isLoading}
          >
            <LinearGradient
              colors={isLoading ? ['#ccc', '#aaa'] : ['#4c6ef5', '#7c3aed']}
              style={styles.submitGradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              {isLoading ? (
                <ActivityIndicator size="small" color="white" />
              ) : (
                <Ionicons name="analytics" size={24} color="white" />
              )}
              <Text style={styles.submitButtonText}>
                {isLoading ? 'Đang phân tích...' : 'Phân tích ngay'}
              </Text>
            </LinearGradient>
          </TouchableOpacity>

          <View style={styles.bottomPadding} />
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: 'white',
    marginTop: 16,
    fontSize: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
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
    fontWeight: 'bold',
  },
  infoButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollView: {
    flex: 1,
  },
  descriptionCard: {
    margin: 20,
    padding: 20,
    backgroundColor: '#2a2a3e',
    borderRadius: 16,
  },
  descriptionTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  descriptionText: {
    color: '#b0b0b0',
    fontSize: 14,
    lineHeight: 20,
  },
  inputSection: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  sectionTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  inputCard: {
    backgroundColor: '#2a2a3e',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  inputLabel: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  inputSubLabel: {
    color: '#8e8e93',
    fontSize: 12,
    marginBottom: 12,
  },
  textInput: {
    backgroundColor: '#1a1a2e',
    borderRadius: 8,
    padding: 12,
    color: 'white',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  notesInput: {
    height: 80,
    textAlignVertical: 'top',
  },
  actionSection: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    marginBottom: 24,
    gap: 12,
  },
  secondaryButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2a2a3e',
    borderRadius: 12,
    padding: 16,
    gap: 8,
  },
  secondaryButtonText: {
    color: '#4c6ef5',
    fontSize: 14,
    fontWeight: '600',
  },
  submitButton: {
    marginHorizontal: 20,
    marginBottom: 24,
    borderRadius: 16,
    overflow: 'hidden',
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 18,
    gap: 12,
  },
  submitButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  bottomPadding: {
    height: 40,
  },
});

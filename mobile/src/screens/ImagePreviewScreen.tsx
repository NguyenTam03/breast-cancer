import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import Button from '../components/Button';

const { width, height } = Dimensions.get('window');

export default function ImagePreviewScreen({ route, navigation }: any) {
  const { imageUri } = route.params;

  const useOriginalImage = () => {
    navigation.navigate('Analysis', { imageUri });
  };

  const cropImage = () => {
    navigation.navigate('ImageCrop', { imageUri });
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
          <Text style={styles.headerTitle}>Xem trước ảnh</Text>
          <View style={styles.headerPlaceholder} />
        </View>

        {/* Image Preview */}
        <View style={styles.imageContainer}>
          <Image source={{ uri: imageUri }} style={styles.image} />
          
          {/* Floating Crop Button */}
          <TouchableOpacity style={styles.cropButton} onPress={cropImage}>
            <Ionicons name="crop" size={24} color="white" />
          </TouchableOpacity>
          
          {/* Image Info Overlay */}
          <LinearGradient 
            colors={['transparent', 'rgba(0,0,0,0.7)'] as any}
            style={styles.imageOverlay}
          >
            <View style={styles.imageInfo}>
              <Text style={styles.imageTitle}>Ảnh đã chọn</Text>
              <Text style={styles.imageSubtitle}>
                Nhấn crop để chỉnh sửa hoặc phân tích ngay
              </Text>
            </View>
          </LinearGradient>
        </View>

        {/* Action Button */}
        <View style={styles.actionContainer}>
          <Button
            title="Phân tích ngay"
            onPress={useOriginalImage}
            size="large"
            style={styles.analyzeButton}
          />
        </View>
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 16,
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
  headerPlaceholder: {
    width: 40,
  },
  imageContainer: {
    flex: 1,
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 20,
    overflow: 'hidden',
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
    borderRadius: 20,
  },
  cropButton: {
    position: 'absolute',
    top: 16,
    right: 16,
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingHorizontal: 20,
    paddingBottom: 20,
    paddingTop: 40,
  },
  imageInfo: {
    alignItems: 'flex-start',
  },
  imageTitle: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  imageSubtitle: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    lineHeight: 18,
  },
  actionContainer: {
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  analyzeButton: {
    backgroundColor: '#4c6ef5',
  },
});

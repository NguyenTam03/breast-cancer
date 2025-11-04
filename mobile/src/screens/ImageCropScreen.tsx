import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Dimensions,
  Alert,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import * as ImageManipulator from 'expo-image-manipulator';
import {
  GestureHandlerRootView,
  PanGestureHandler,
  PinchGestureHandler,
  State,
} from 'react-native-gesture-handler';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  runOnJS,
  withSpring,
} from 'react-native-reanimated';
import Button from '../components/Button';

const { width, height } = Dimensions.get('window');
const imagePreviewWidth = width - 40;

interface CropRatio {
  id: string;
  name: string;
  ratio: [number, number];
  icon: keyof typeof Ionicons.glyphMap;
}

const cropRatios: CropRatio[] = [
  { id: 'square', name: '1:1', ratio: [1, 1], icon: 'square' },
  { id: 'portrait', name: '3:4', ratio: [3, 4], icon: 'phone-portrait' },
  { id: 'landscape', name: '4:3', ratio: [4, 3], icon: 'phone-landscape' },
  { id: 'wide', name: '16:9', ratio: [16, 9], icon: 'tv' },
  { id: 'free', name: 'Tự do', ratio: [0, 0], icon: 'crop' },
];

// Simple Interactive Crop Box Component
const InteractiveCropBox = ({ selectedRatio, imageContainerSize }: { 
  selectedRatio: CropRatio; 
  imageContainerSize: { width: number; height: number };
}) => {
  // Calculate crop box size based on ratio
  const getCropBoxStyle = () => {
    const { width: containerWidth, height: containerHeight } = imageContainerSize;
    const [ratioW, ratioH] = selectedRatio.ratio;
    
    if (selectedRatio.id === 'free') {
      return { width: 180, height: 180 };
    }
    
    const targetAspect = ratioW / ratioH;
    const maxSize = Math.min(containerWidth * 0.8, containerHeight * 0.8);
    
    let cropWidth, cropHeight;
    if (targetAspect > 1) {
      cropWidth = maxSize;
      cropHeight = maxSize / targetAspect;
    } else {
      cropHeight = maxSize;
      cropWidth = maxSize * targetAspect;
    }
    
    return { width: cropWidth, height: cropHeight };
  };
  
  const cropBoxStyle = getCropBoxStyle();
  
  return (
    <View style={styles.cropContainer}>
      <View style={[styles.interactiveCropFrame, cropBoxStyle]}>
        {/* Crop frame border */}
        <View style={styles.cropFrameBorder} />
        
        {/* Corner handles */}
        <View style={[styles.cropHandle, styles.topLeft]} />
        <View style={[styles.cropHandle, styles.topRight]} />
        <View style={[styles.cropHandle, styles.bottomLeft]} />
        <View style={[styles.cropHandle, styles.bottomRight]} />
        
        {/* Center drag indicator */}
        <View style={styles.centerDragIndicator}>
          <Ionicons name="move" size={16} color="white" />
        </View>
        
        {/* Instruction text */}
        <View style={styles.cropInstruction}>
          <Text style={styles.cropInstructionText}>
            Khung crop {selectedRatio.name}
          </Text>
        </View>
      </View>
    </View>
  );
};

export default function ImageCropScreen({ route, navigation }: any) {
  const { imageUri } = route.params;
  const [selectedRatio, setSelectedRatio] = useState<CropRatio>(cropRatios[0]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [croppedImage, setCroppedImage] = useState<string | null>(null);
  const [imageContainerSize] = useState({ width: imagePreviewWidth, height: 300 });

  const cropImage = async () => {
    if (!imageUri) return;

    setIsProcessing(true);
    try {
      // Get image info first
      const imageInfo = await ImageManipulator.manipulateAsync(imageUri, [], {
        format: ImageManipulator.SaveFormat.JPEG,
      });

      let cropData;
      
      if (selectedRatio.id === 'free') {
        // For free crop, use center crop
        const imageAspect = imageInfo.width / imageInfo.height;
        const targetSize = Math.min(imageInfo.width, imageInfo.height);
        
        cropData = {
          originX: (imageInfo.width - targetSize) / 2,
          originY: (imageInfo.height - targetSize) / 2,
          width: targetSize,
          height: targetSize,
        };
      } else {
        // Calculate crop dimensions based on ratio
        const [ratioW, ratioH] = selectedRatio.ratio;
        const imageAspect = imageInfo.width / imageInfo.height;
        const targetAspect = ratioW / ratioH;

        let cropWidth, cropHeight, originX, originY;

        if (imageAspect > targetAspect) {
          // Image is wider than target ratio
          cropHeight = imageInfo.height;
          cropWidth = cropHeight * targetAspect;
          originX = (imageInfo.width - cropWidth) / 2;
          originY = 0;
        } else {
          // Image is taller than target ratio
          cropWidth = imageInfo.width;
          cropHeight = cropWidth / targetAspect;
          originX = 0;
          originY = (imageInfo.height - cropHeight) / 2;
        }

        cropData = {
          originX: Math.max(0, originX),
          originY: Math.max(0, originY),
          width: Math.min(cropWidth, imageInfo.width),
          height: Math.min(cropHeight, imageInfo.height),
        };
      }

      const result = await ImageManipulator.manipulateAsync(
        imageUri,
        [{ crop: cropData }],
        {
          compress: 0.8,
          format: ImageManipulator.SaveFormat.JPEG,
        }
      );

      setCroppedImage(result.uri);
    } catch (error) {
      console.error('Crop error:', error);
      Alert.alert('Lỗi', 'Không thể crop ảnh');
    } finally {
      setIsProcessing(false);
    }
  };

  const useCroppedImage = () => {
    if (croppedImage) {
      navigation.navigate('Analysis', { imageUri: croppedImage });
    }
  };

  const resetCrop = () => {
    setCroppedImage(null);
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
          <Text style={styles.headerTitle}>Crop ảnh</Text>
          <TouchableOpacity style={styles.resetButton} onPress={resetCrop}>
            <Ionicons name="refresh" size={24} color="white" />
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          {/* Image Preview */}
          <View style={styles.imageSection}>
            <View style={styles.imageContainer}>
              <Image 
                source={{ uri: croppedImage || imageUri }} 
                style={styles.image} 
                contentFit="contain"
              />
              {!croppedImage && (
                <GestureHandlerRootView style={styles.cropOverlay}>
                  <InteractiveCropBox 
                    selectedRatio={selectedRatio} 
                    imageContainerSize={imageContainerSize}
                  />
                </GestureHandlerRootView>
              )}
            </View>
          </View>

          {/* Crop Ratios */}
          <View style={styles.ratiosSection}>
            <Text style={styles.sectionTitle}>Tỷ lệ crop</Text>
            <ScrollView 
              horizontal 
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.ratiosContainer}
            >
              {cropRatios.map((ratio) => (
                <TouchableOpacity
                  key={ratio.id}
                  style={[
                    styles.ratioCard,
                    selectedRatio.id === ratio.id && styles.ratioCardSelected
                  ]}
                  onPress={() => setSelectedRatio(ratio)}
                >
                  <View style={[
                    styles.ratioIconContainer,
                    selectedRatio.id === ratio.id && styles.ratioIconSelected
                  ]}>
                    <Ionicons 
                      name={ratio.icon} 
                      size={20} 
                      color={selectedRatio.id === ratio.id ? '#4c6ef5' : '#8e8e93'} 
                    />
                  </View>
                  <Text style={[
                    styles.ratioName,
                    selectedRatio.id === ratio.id && styles.ratioNameSelected
                  ]}>
                    {ratio.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>

          {/* Info Card */}
          <View style={styles.infoSection}>
            <View style={styles.infoCard}>
              <View style={styles.infoIconContainer}>
                <Ionicons name="information-circle" size={24} color="#fbbf24" />
              </View>
              <View style={styles.infoContent}>
                <Text style={styles.infoTitle}>Hướng dẫn</Text>
                <Text style={styles.infoText}>
                  Chọn tỷ lệ crop mong muốn và nhấn "Áp dụng crop" để cắt ảnh theo khung hiển thị.
                </Text>
              </View>
            </View>
          </View>
        </ScrollView>

        {/* Action Buttons */}
        <View style={styles.actionsContainer}>
          {!croppedImage ? (
            <Button
              title={isProcessing ? "Đang xử lý..." : "Áp dụng crop"}
              onPress={cropImage}
              disabled={isProcessing}
              size="large"
              style={styles.cropButton}
            />
          ) : (
            <View style={styles.actionButtonsRow}>
              <Button
                title="Crop lại"
                onPress={resetCrop}
                variant="outline"
                style={styles.actionButton}
              />
              <Button
                title="Sử dụng ảnh này"
                onPress={useCroppedImage}
                style={styles.actionButton}
              />
            </View>
          )}
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
  scrollView: {
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
  resetButton: {
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
    height: 300,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
    borderRadius: 20,
  },
  cropOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cropFrame: {
    borderWidth: 2,
    borderColor: '#4c6ef5',
    borderStyle: 'dashed',
    position: 'relative',
  },
  cropSquare: {
    width: 200,
    height: 200,
  },
  cropPortrait: {
    width: 150,
    height: 200,
  },
  cropLandscape: {
    width: 200,
    height: 150,
  },
  cropWide: {
    width: 250,
    height: 140,
  },
  cropFree: {
    width: 180,
    height: 180,
  },
  cropCorner: {
    position: 'absolute',
    top: -5,
    left: -5,
    width: 10,
    height: 10,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#4c6ef5',
  },
  ratiosSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    paddingHorizontal: 20,
  },
  ratiosContainer: {
    paddingHorizontal: 20,
    gap: 12,
  },
  ratioCard: {
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 12,
    padding: 12,
    minWidth: 80,
  },
  ratioCardSelected: {
    backgroundColor: 'rgba(76, 110, 245, 0.1)',
    borderWidth: 1,
    borderColor: '#4c6ef5',
  },
  ratioIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  ratioIconSelected: {
    backgroundColor: 'rgba(76, 110, 245, 0.2)',
  },
  ratioName: {
    color: '#8e8e93',
    fontSize: 12,
    fontWeight: '600',
  },
  ratioNameSelected: {
    color: '#4c6ef5',
  },
  infoSection: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  infoCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'rgba(251, 191, 36, 0.1)',
    borderRadius: 16,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#fbbf24',
  },
  infoIconContainer: {
    marginRight: 12,
    marginTop: 2,
  },
  infoContent: {
    flex: 1,
  },
  infoTitle: {
    color: '#fbbf24',
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  infoText: {
    color: '#8e8e93',
    fontSize: 13,
    lineHeight: 18,
  },
  actionsContainer: {
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  cropButton: {
    backgroundColor: '#4c6ef5',
  },
  actionButtonsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
  },
  // Interactive crop styles
  cropContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  interactiveCropFrame: {
    borderWidth: 2,
    borderColor: '#4c6ef5',
    borderStyle: 'dashed',
    position: 'relative',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cropFrameBorder: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderWidth: 1,
    borderColor: 'rgba(76, 110, 245, 0.3)',
  },
  cropHandle: {
    position: 'absolute',
    width: 12,
    height: 12,
    backgroundColor: '#4c6ef5',
    borderRadius: 6,
    borderWidth: 2,
    borderColor: 'white',
  },
  topLeft: {
    top: -6,
    left: -6,
  },
  topRight: {
    top: -6,
    right: -6,
  },
  bottomLeft: {
    bottom: -6,
    left: -6,
  },
  bottomRight: {
    bottom: -6,
    right: -6,
  },
  centerDragIndicator: {
    position: 'absolute',
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(76, 110, 245, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cropInstruction: {
    position: 'absolute',
    bottom: -30,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  cropInstructionText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
    backgroundColor: 'rgba(76, 110, 245, 0.8)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
});

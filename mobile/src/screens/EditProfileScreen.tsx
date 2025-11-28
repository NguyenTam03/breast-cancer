import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Alert,
  TextInput,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import Card from '../components/Card';
import Button from '../components/Button';
import { colors } from '../theme/colors';
import { useAuth } from '../contexts/AuthContext';

interface EditProfileScreenProps {
  navigation: any;
}

export default function EditProfileScreen({ navigation }: EditProfileScreenProps) {
  const { user, updateProfile } = useAuth();
  
  const [firstName, setFirstName] = useState(user?.profile?.firstName || '');
  const [lastName, setLastName] = useState(user?.profile?.lastName || '');
  const [isLoading, setIsLoading] = useState(false);

  const handleSave = async () => {
    if (!firstName.trim() || !lastName.trim()) {
      Alert.alert('Lỗi', 'Vui lòng nhập đầy đủ họ và tên');
      return;
    }

    try {
      setIsLoading(true);
      await updateProfile({
        firstName: firstName.trim(),
        lastName: lastName.trim(),
      });
      
      Alert.alert(
        'Thành công',
        'Cập nhật hồ sơ thành công',
        [
          {
            text: 'OK',
            onPress: () => navigation.goBack(),
          }
        ]
      );
    } catch (error: any) {
      Alert.alert('Lỗi', error.message || 'Không thể cập nhật hồ sơ');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    navigation.goBack();
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
      
      <LinearGradient colors={['#1a1a2e', '#16213e'] as any} style={styles.gradient}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.backButton} onPress={handleCancel}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Chỉnh sửa hồ sơ</Text>
          <TouchableOpacity style={styles.saveButton} onPress={handleSave} disabled={isLoading}>
            <Text style={[styles.saveButtonText, isLoading && styles.saveButtonTextDisabled]}>
              Lưu
            </Text>
          </TouchableOpacity>
        </View>

        <KeyboardAvoidingView 
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.keyboardView}
        >
          <ScrollView 
            style={styles.scrollView}
            contentContainerStyle={styles.scrollContent}
            showsVerticalScrollIndicator={false}
          >
            {/* Profile Image Section */}
            <View style={styles.imageSection}>
              <Card style={styles.imageCard} padding="large">
                <View style={styles.imageContainer}>
                  <View style={styles.profileImageContainer}>
                    <Image 
                      source={{ uri: 'https://via.placeholder.com/120x120/4c6ef5/ffffff?text=BC' }}
                      style={styles.profileImage}
                    />
                    <TouchableOpacity style={styles.editImageButton}>
                      <Ionicons name="camera" size={20} color="white" />
                    </TouchableOpacity>
                  </View>
                  <Text style={styles.imageHint}>Nhấn để thay đổi ảnh đại diện</Text>
                </View>
              </Card>
            </View>

            {/* Form Section */}
            <View style={styles.formSection}>
              <Card style={styles.formCard} padding="large">
                <Text style={styles.sectionTitle}>Thông tin cá nhân</Text>
                
                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>Họ</Text>
                  <View style={styles.inputContainer}>
                    <Ionicons name="person-outline" size={20} color="#8e8e93" style={styles.inputIcon} />
                    <TextInput
                      style={styles.textInput}
                      value={firstName}
                      onChangeText={setFirstName}
                      placeholder="Nhập họ của bạn"
                      placeholderTextColor="#8e8e93"
                      autoCapitalize="words"
                      editable={!isLoading}
                    />
                  </View>
                </View>

                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>Tên</Text>
                  <View style={styles.inputContainer}>
                    <Ionicons name="person-outline" size={20} color="#8e8e93" style={styles.inputIcon} />
                    <TextInput
                      style={styles.textInput}
                      value={lastName}
                      onChangeText={setLastName}
                      placeholder="Nhập tên của bạn"
                      placeholderTextColor="#8e8e93"
                      autoCapitalize="words"
                      editable={!isLoading}
                    />
                  </View>
                </View>

                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>Email</Text>
                  <View style={[styles.inputContainer, styles.inputDisabled]}>
                    <Ionicons name="mail-outline" size={20} color="#8e8e93" style={styles.inputIcon} />
                    <TextInput
                      style={[styles.textInput, styles.textInputDisabled]}
                      value={user?.email || ''}
                      placeholder="Email"
                      placeholderTextColor="#8e8e93"
                      editable={false}
                    />
                  </View>
                  <Text style={styles.inputHint}>Email không thể thay đổi</Text>
                </View>

                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>Vai trò</Text>
                  <View style={[styles.inputContainer, styles.inputDisabled]}>
                    <Ionicons 
                      name={user?.role === 'doctor' ? 'medical-outline' : 'person-outline'} 
                      size={20} 
                      color="#8e8e93" 
                      style={styles.inputIcon} 
                    />
                    <TextInput
                      style={[styles.textInput, styles.textInputDisabled]}
                      value={user?.role === 'doctor' ? 'Bác sĩ' : 'Bệnh nhân'}
                      placeholder="Vai trò"
                      placeholderTextColor="#8e8e93"
                      editable={false}
                    />
                  </View>
                  <Text style={styles.inputHint}>Vai trò không thể thay đổi</Text>
                </View>
              </Card>
            </View>

            {/* Action Buttons */}
            <View style={styles.actionSection}>
              <Button
                title="Lưu thay đổi"
                onPress={handleSave}
                loading={isLoading}
                style={styles.saveActionButton}
              />
              <Button
                title="Hủy"
                onPress={handleCancel}
                variant="outline"
                style={styles.cancelButton}
                disabled={isLoading}
              />
            </View>
          </ScrollView>
        </KeyboardAvoidingView>
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
  keyboardView: {
    flex: 1,
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
  saveButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: '#4c6ef5',
  },
  saveButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  saveButtonTextDisabled: {
    opacity: 0.5,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingBottom: 24,
  },
  imageSection: {
    marginBottom: 24,
  },
  imageCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  imageContainer: {
    alignItems: 'center',
  },
  profileImageContainer: {
    position: 'relative',
    marginBottom: 12,
  },
  profileImage: {
    width: 120,
    height: 120,
    borderRadius: 60,
  },
  editImageButton: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#4c6ef5',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#1a1a2e',
  },
  imageHint: {
    color: '#8e8e93',
    fontSize: 14,
    textAlign: 'center',
  },
  formSection: {
    marginBottom: 24,
  },
  formCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  sectionTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 20,
  },
  inputGroup: {
    marginBottom: 20,
  },
  inputLabel: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 8,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  inputDisabled: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    opacity: 0.7,
  },
  inputIcon: {
    marginRight: 12,
  },
  textInput: {
    flex: 1,
    color: 'white',
    fontSize: 16,
    paddingVertical: 4,
  },
  textInputDisabled: {
    color: '#8e8e93',
  },
  inputHint: {
    color: '#8e8e93',
    fontSize: 12,
    marginTop: 4,
    marginLeft: 4,
  },
  actionSection: {
    gap: 12,
  },
  saveActionButton: {
    backgroundColor: '#4c6ef5',
  },
  cancelButton: {
    borderColor: '#8e8e93',
  },
});
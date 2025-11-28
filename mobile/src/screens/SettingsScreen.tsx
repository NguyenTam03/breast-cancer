import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
  Alert,
  Switch,
  Linking,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import Card from '../components/Card';
import { colors } from '../theme/colors';
import { useAuth } from '../contexts/AuthContext';

interface SettingsScreenProps {
  navigation: any;
}

interface SettingItem {
  id: string;
  title: string;
  subtitle?: string;
  icon: keyof typeof Ionicons.glyphMap;
  type: 'navigation' | 'switch' | 'action';
  value?: boolean;
  onPress?: () => void;
  onToggle?: (value: boolean) => void;
}

export default function SettingsScreen({ navigation }: SettingsScreenProps) {
  const { user, logout } = useAuth();
  const [notifications, setNotifications] = useState(true);

  const handleLogout = async () => {
    Alert.alert(
      'Đăng xuất',
      'Bạn có chắc chắn muốn đăng xuất?',
      [
        { text: 'Hủy', style: 'cancel' },
        { 
          text: 'Đăng xuất', 
          style: 'destructive', 
          onPress: async () => {
            try {
              await logout();
              // Navigation sẽ được handle tự động bởi AuthContext
            } catch (error) {
              Alert.alert('Lỗi', 'Không thể đăng xuất. Vui lòng thử lại.');
            }
          }
        },
      ]
    );
  };

  const personalSettings: SettingItem[] = [
    {
      id: 'profile',
      title: 'Hồ sơ cá nhân',
      subtitle: 'Chỉnh sửa thông tin cá nhân',
      icon: 'person-outline',
      type: 'navigation',
      onPress: () => navigation.navigate('EditProfile'),
    },
  ];

  const appSettings: SettingItem[] = [
    {
      id: 'notifications',
      title: 'Thông báo',
      subtitle: 'Nhận thông báo về kết quả phân tích',
      icon: 'notifications-outline',
      type: 'switch',
      value: notifications,
      onToggle: setNotifications,
    },
  ];

  const dataSettings: SettingItem[] = [
    {
      id: 'clear',
      title: 'Xóa dữ liệu',
      subtitle: 'Xóa tất cả lịch sử phân tích',
      icon: 'trash-outline',
      type: 'action',
      onPress: () => {
        Alert.alert(
          'Xóa dữ liệu',
          'Bạn có chắc chắn muốn xóa tất cả dữ liệu? Hành động này không thể hoàn tác.',
          [
            { text: 'Hủy', style: 'cancel' },
            { text: 'Xóa', style: 'destructive', onPress: () => console.log('Clear data') },
          ]
        );
      },
    },
  ];

  const supportSettings: SettingItem[] = [
    {
      id: 'contact',
      title: 'Liên hệ hỗ trợ',
      subtitle: 'Gửi phản hồi hoặc báo lỗi',
      icon: 'mail-outline',
      type: 'navigation',
      onPress: () => {
        const email = 'n21dccn158@student.ptithcm.edu.vn';
        const subject = 'Hỗ trợ ứng dụng BreastCare AI';
        const body = 'Xin chào, tôi cần hỗ trợ về vấn đề: ';
        const url = `mailto:${email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
        
        Linking.canOpenURL(url)
          .then((supported) => {
            if (!supported) {
              Alert.alert('Lỗi', 'Không thể mở ứng dụng email');
            } else {
              return Linking.openURL(url);
            }
          })
          .catch((err) => console.error('An error occurred', err));
      },
    },
    {
      id: 'about',
      title: 'Về ứng dụng',
      subtitle: 'Phiên bản 1.0.0',
      icon: 'information-circle-outline',
      type: 'navigation',
      onPress: () => Alert.alert('Về ứng dụng', 'BreastCare AI v1.0.0\nPhát triển bởi Nhân & Tâm'),
    },
  ];

  const renderSettingItem = (item: SettingItem) => (
    <TouchableOpacity
      key={item.id}
      style={styles.settingItem}
      onPress={item.onPress}
      activeOpacity={item.type === 'switch' ? 1 : 0.7}
    >
      <View style={styles.settingIcon}>
        <Ionicons name={item.icon} size={24} color="#4c6ef5" />
      </View>
      
      <View style={styles.settingContent}>
        <Text style={styles.settingTitle}>{item.title}</Text>
        {item.subtitle && (
          <Text style={styles.settingSubtitle}>{item.subtitle}</Text>
        )}
      </View>
      
      <View style={styles.settingAction}>
        {item.type === 'switch' ? (
          <Switch
            value={item.value}
            onValueChange={item.onToggle}
            trackColor={{ 
              false: 'rgba(255, 255, 255, 0.1)', 
              true: '#4c6ef5' 
            }}
            thumbColor={item.value ? 'white' : '#8e8e93'}
          />
        ) : (
          <Ionicons name="chevron-forward" size={20} color="#8e8e93" />
        )}
      </View>
    </TouchableOpacity>
  );

  const renderSection = (title: string, items: SettingItem[]) => (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      <Card style={styles.sectionCard} padding="none">
        {items.map((item, index) => (
          <View key={item.id}>
            {renderSettingItem(item)}
            {index < items.length - 1 && <View style={styles.separator} />}
          </View>
        ))}
      </Card>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
      
      <LinearGradient colors={['#1a1a2e', '#16213e'] as any} style={styles.gradient}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Cài đặt</Text>
          <TouchableOpacity style={styles.headerButton}>
            <Ionicons name="search-outline" size={24} color="white" />
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          {/* Profile Section */}
          <View style={styles.profileSection}>
            <Card style={styles.profileCard} padding="large">
              <View style={styles.profileContent}>
                <View style={styles.profileImageContainer}>
                  <Image 
                    source={{ uri: 'https://via.placeholder.com/80x80/4c6ef5/ffffff?text=BC' }}
                    style={styles.profileImage}
                  />
                  <TouchableOpacity style={styles.editButton}>
                    <Ionicons name="camera" size={16} color="white" />
                  </TouchableOpacity>
                </View>
                
                <View style={styles.profileInfo}>
                  <Text style={styles.profileName}>
                    {user?.profile?.firstName && user?.profile?.lastName 
                      ? `${user.profile.firstName} ${user.profile.lastName}` 
                      : 'Người dùng'}
                  </Text>
                  <Text style={styles.profileEmail}>
                    {user?.email || 'user@example.com'}
                  </Text>
                  <View style={styles.roleBadge}>
                    <Text style={styles.roleText}>
                      {user?.role === 'doctor' ? '👨‍⚕️ Bác sĩ' : '👤 Bệnh nhân'}
                    </Text>
                  </View>
                  <TouchableOpacity
                    style={styles.editProfileButton}
                    onPress={() => navigation.navigate('EditProfile')}
                  >
                    <Text style={styles.editProfileText}>Chỉnh sửa hồ sơ</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </Card>
          </View>

          {/* Settings Sections */}
          {renderSection('Cá nhân', personalSettings)}
          {renderSection('Ứng dụng', appSettings)}
          {renderSection('Dữ liệu', dataSettings)}
          {renderSection('Hỗ trợ', supportSettings)}

          {/* Logout Section */}
          <View style={styles.section}>
            <Card style={styles.sectionCard} padding="none">
              <TouchableOpacity 
                style={styles.logoutButton}
                onPress={handleLogout}
              >
                <Ionicons name="log-out-outline" size={24} color="#f87171" />
                <Text style={styles.logoutText}>Đăng xuất</Text>
              </TouchableOpacity>
            </Card>
          </View>

          <View style={styles.footer}>
            <Text style={styles.footerText}>
              BreastCare AI • Phiên bản 1.0.0
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 24,
  },
  headerTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  headerButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollView: {
    flex: 1,
    paddingHorizontal: 20,
  },
  profileSection: {
    marginBottom: 32,
  },
  profileCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  profileContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  profileImageContainer: {
    position: 'relative',
    marginRight: 16,
  },
  profileImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
  },
  editButton: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#4c6ef5',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#1a1a2e',
  },
  profileInfo: {
    flex: 1,
  },
  profileName: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  profileEmail: {
    color: '#8e8e93',
    fontSize: 14,
    marginBottom: 8,
  },
  roleBadge: {
    backgroundColor: 'rgba(76, 110, 245, 0.2)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  roleText: {
    color: '#4c6ef5',
    fontSize: 12,
    fontWeight: '600',
  },
  editProfileButton: {
    backgroundColor: 'rgba(76, 110, 245, 0.2)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  editProfileText: {
    color: '#4c6ef5',
    fontSize: 14,
    fontWeight: '600',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    paddingHorizontal: 4,
  },
  sectionCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  settingIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(76, 110, 245, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  settingContent: {
    flex: 1,
  },
  settingTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 2,
  },
  settingSubtitle: {
    color: '#8e8e93',
    fontSize: 13,
    lineHeight: 18,
  },
  settingAction: {
    marginLeft: 12,
  },
  separator: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    marginLeft: 76,
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    gap: 16,
  },
  logoutText: {
    color: '#f87171',
    fontSize: 16,
    fontWeight: '500',
  },
  footer: {
    alignItems: 'center',
    paddingVertical: 24,
    marginBottom: 100,
  },
  footerText: {
    color: '#8e8e93',
    fontSize: 14,
    textAlign: 'center',
  },
});

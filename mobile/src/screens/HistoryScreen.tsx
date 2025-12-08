import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  StatusBar,
  Alert,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import { useFocusEffect } from '@react-navigation/native';
import Card from '../components/Card';
import { colors } from '../theme/colors';
import { apiService } from '../services/api';
import { AnalysisResult } from '../types/analysis.types';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE_URL } from '../config/api.config';

interface HistoryScreenProps {
  navigation: any;
}

export default function HistoryScreen({ navigation }: HistoryScreenProps) {
  const { user, isAuthenticated } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [historyData, setHistoryData] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [page, setPage] = useState(1);
  const [hasMoreData, setHasMoreData] = useState(true);

  // Load history data from API
  useEffect(() => {
    loadHistoryData();
  }, []);

  // Refresh data when screen comes into focus
  useFocusEffect(
    React.useCallback(() => {
      // Refresh data when user navigates back to this screen
      onRefresh();
    }, [])
  );

  const loadHistoryData = async (pageNum: number = 1, isRefresh: boolean = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      // Check if user is authenticated and has valid ID
      if (!isAuthenticated || !user?.id) {
        Alert.alert('Lỗi', 'Vui lòng đăng nhập để xem lịch sử');
        return;
      }

      // Use user-specific endpoint
      const response = await apiService.getUserAnalysisHistory(user.id, pageNum, 20);
      
      if (isRefresh || pageNum === 1) {
        setHistoryData(response.analyses);
      } else {
        setHistoryData(prev => [...prev, ...response.analyses]);
      }

      setHasMoreData(response.analyses.length === 20);
      setPage(pageNum);
    } catch (error) {
      console.error('Failed to load user history:', error);
      Alert.alert('Lỗi', 'Không thể tải lịch sử phân tích của bạn');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    loadHistoryData(1, true);
  };

  const loadMoreData = () => {
    if (!loading && hasMoreData) {
      loadHistoryData(page + 1);
    }
  };

  const getResultColor = (prediction: string) => {
    switch (prediction) {
      case 'BENIGN':
        return '#4ade80'; // Xanh lá - Lành tính
      case 'MALIGNANT':
        return '#f87171'; // Đỏ - Ác tính
      case 'NORMAL':
        return '#60a5fa'; // Xanh dương - Bình thường
      default:
        return '#8e8e93'; // Xám - Không xác định
    }
  };

  const getResultText = (prediction: string) => {
    switch (prediction) {
      case 'BENIGN':
        return 'Lành tính';
      case 'MALIGNANT':
        return 'Ác tính';
      case 'NORMAL':
        return 'Bình thường';
      default:
        return 'Không xác định';
    }
  };

  const getResultIcon = (prediction: string) => {
    switch (prediction) {
      case 'BENIGN':
        return 'checkmark-circle';
      case 'MALIGNANT':
        return 'alert-circle';
      case 'NORMAL':
        return 'shield-checkmark';
      default:
        return 'help-circle';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('vi-VN', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const handleItemPress = (item: AnalysisResult) => {
    Alert.alert(
      'Chi tiết kết quả',
      `Kết quả: ${getResultText(item.prediction)}\nĐộ tin cậy: ${(item.confidence * 100).toFixed(1)}%\nNgày: ${formatDate(item.analysisDate)}`,
      [
        { text: 'Đóng', style: 'cancel' },
        { 
          text: 'Xem chi tiết', 
          onPress: () => navigation.navigate('Analysis', { 
            analysisResult: item,
            isFromHistory: true 
          })
        },
        { text: 'Chia sẻ', onPress: () => shareResult(item) },
      ]
    );
  };

  const shareResult = (item: AnalysisResult) => {
    // TODO: Implement share functionality
    Alert.alert('Chia sẻ', 'Chức năng chia sẻ đang được phát triển');
  };

  const handleDeleteItem = async (item: AnalysisResult) => {
    Alert.alert(
      'Xóa kết quả',
      'Bạn có chắc chắn muốn xóa kết quả này?',
      [
        { text: 'Hủy', style: 'cancel' },
        { 
          text: 'Xóa', 
          style: 'destructive', 
          onPress: async () => {
            try {
              await apiService.deleteAnalysis(item.id);
              setHistoryData(prev => prev.filter(h => h.id !== item.id));
              Alert.alert('Thành công', 'Đã xóa kết quả phân tích');
            } catch (error) {
              Alert.alert('Lỗi', 'Không thể xóa kết quả');
            }
          }
        },
      ]
    );
  };

  const filteredData = historyData.filter((item) => {
    if (selectedFilter === 'benign') return item.prediction === 'BENIGN';
    if (selectedFilter === 'malignant') return item.prediction === 'MALIGNANT';
    if (selectedFilter === 'normal') return item.prediction === 'NORMAL';
    return true;
  });

  const FilterButton = ({ title, value, isActive }: { title: string; value: string; isActive: boolean }) => (
    <TouchableOpacity
      style={[styles.filterButton, isActive && styles.filterButtonActive]}
      onPress={() => setSelectedFilter(value)}
    >
      <Text style={[styles.filterButtonText, isActive && styles.filterButtonTextActive]}>
        {title}
      </Text>
    </TouchableOpacity>
  );

  const renderHistoryItem = ({ item }: { item: AnalysisResult }) => {
    // Construct proper image URL
    const getImageUrl = () => {
      if (!item.imageUrl) return null;
      
      // If imageUrl already contains full URL (starts with http), use as is
      if (item.imageUrl.startsWith('http')) {
        return item.imageUrl;
      }
      
      // Otherwise, prepend API_BASE_URL
      const baseUrl = API_BASE_URL.replace('/api/v1', ''); // Remove API path for image URLs
      return `${baseUrl}${item.imageUrl}`;
    };

    const imageUri = getImageUrl();

    return (
      <Card style={styles.historyItem} padding="medium">
        <TouchableOpacity onPress={() => handleItemPress(item)} activeOpacity={0.8}>
          <View style={styles.itemContent}>
            <View style={styles.itemImage}>
              {imageUri ? (
                <Image 
                  source={{ uri: imageUri }}
                  style={styles.thumbnail}
                  contentFit="cover"
                  transition={300}
                  // Enable caching
                  cachePolicy="memory-disk"
                  recyclingKey={item.id}
                  // Error fallback
                  onError={() => {
                    console.warn('Failed to load image:', imageUri);
                  }}
                />
              ) : (
                <View style={styles.thumbnailPlaceholder}>
                  <Ionicons name="image-outline" size={24} color="#8e8e93" />
                </View>
              )}
            </View>
            
            <View style={styles.itemInfo}>
              <View style={styles.itemHeader}>
                <View style={styles.resultBadge}>
                  <Ionicons 
                    name={getResultIcon(item.prediction)} 
                    size={16} 
                    color={getResultColor(item.prediction)} 
                  />
                  <Text style={[styles.resultText, { color: getResultColor(item.prediction) }]}>
                    {getResultText(item.prediction)}
                  </Text>
                </View>
                <TouchableOpacity onPress={() => handleDeleteItem(item)}>
                  <Ionicons name="trash-outline" size={20} color="#8e8e93" />
                </TouchableOpacity>
              </View>
              
              <View style={styles.itemDetails}>
                <View style={styles.detailRow}>
                  <Ionicons name="analytics-outline" size={14} color="#8e8e93" />
                  <Text style={styles.detailText}>
                    Độ tin cậy: {(item.confidence * 100).toFixed(1)}%
                  </Text>
                </View>
                
                <View style={styles.detailRow}>
                  <Ionicons name="time-outline" size={14} color="#8e8e93" />
                  <Text style={styles.detailText}>
                    {formatDate(item.analysisDate)}
                  </Text>
                </View>
                
                <View style={styles.detailRow}>
                  <Ionicons name="speedometer-outline" size={14} color="#8e8e93" />
                  <Text style={styles.detailText}>
                    Xử lý: {item.processingTime}ms
                  </Text>
                </View>

                {item.userNotes && (
                  <View style={styles.detailRow}>
                    <Ionicons name="document-text-outline" size={14} color="#8e8e93" />
                    <Text style={styles.detailText} numberOfLines={1}>
                      {item.userNotes}
                    </Text>
                  </View>
                )}

                {/* Debug URL - can be removed later */}
                {__DEV__ && imageUri && (
                  <View style={styles.detailRow}>
                    <Ionicons name="link-outline" size={14} color="#8e8e93" />
                    <Text style={[styles.detailText, { fontSize: 10 }]} numberOfLines={1}>
                      {imageUri}
                    </Text>
                  </View>
                )}
              </View>
            </View>
          </View>
        </TouchableOpacity>
      </Card>
    );
  };

  const renderLoadingFooter = () => {
    if (!loading || page === 1) return null;
    return (
      <View style={styles.loadingFooter}>
        <ActivityIndicator size="small" color="#4c6ef5" />
      </View>
    );
  };

  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <View style={styles.emptyIcon}>
        <Ionicons name="document-text-outline" size={48} color="#8e8e93" />
      </View>
      <Text style={styles.emptyTitle}>Chưa có lịch sử</Text>
      <Text style={styles.emptySubtitle}>
        Các kết quả phân tích của bạn sẽ xuất hiện ở đây
      </Text>
      <TouchableOpacity 
        style={styles.emptyButton}
        onPress={() => navigation.navigate('HomeTab')}
      >
        <Text style={styles.emptyButtonText}>Bắt đầu phân tích</Text>
      </TouchableOpacity>
    </View>
  );

  if (loading && page === 1) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
        <LinearGradient colors={['#1a1a2e', '#16213e'] as any} style={styles.gradient}>
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#4c6ef5" />
            <Text style={styles.loadingText}>Đang tải lịch sử...</Text>
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
          <Text style={styles.headerTitle}>Lịch sử phân tích</Text>
          <TouchableOpacity style={styles.headerButton} onPress={onRefresh}>
            <Ionicons name="refresh-outline" size={24} color="white" />
          </TouchableOpacity>
        </View>

        {/* Stats Section */}
        <View style={styles.statsSection}>
          <Card style={styles.statsCard} padding="medium">
            <View style={styles.statsRow}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{historyData.length}</Text>
                <Text style={styles.statLabel}>Tổng số</Text>
              </View>
              
              <View style={styles.statDivider} />
              
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {historyData.filter(item => item.prediction === 'BENIGN').length}
                </Text>
                <Text style={styles.statLabel}>Lành tính</Text>
              </View>
              
              <View style={styles.statDivider} />
              
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {historyData.filter(item => item.prediction === 'MALIGNANT').length}
                </Text>
                <Text style={styles.statLabel}>Ác tính</Text>
              </View>

              <View style={styles.statDivider} />
              
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {historyData.filter(item => item.prediction === 'NORMAL').length}
                </Text>
                <Text style={styles.statLabel}>Bình thường</Text>
              </View>
            </View>
          </Card>
        </View>

        {/* Filter Section */}
        <View style={styles.filterSection}>
          <View style={styles.filterButtons}>
            <FilterButton title="Tất cả" value="all" isActive={selectedFilter === 'all'} />
            <FilterButton title="Lành tính" value="benign" isActive={selectedFilter === 'benign'} />
            <FilterButton title="Ác tính" value="malignant" isActive={selectedFilter === 'malignant'} />
            <FilterButton title="Bình thường" value="normal" isActive={selectedFilter === 'normal'} />
          </View>
        </View>

        {/* History List */}
        <View style={styles.listContainer}>
          {filteredData.length > 0 ? (
            <FlatList
              data={filteredData}
              keyExtractor={(item) => item.id}
              renderItem={renderHistoryItem}
              showsVerticalScrollIndicator={false}
              contentContainerStyle={styles.listContent}
              refreshControl={
                <RefreshControl 
                  refreshing={refreshing} 
                  onRefresh={onRefresh}
                  tintColor="#4c6ef5"
                />
              }
              onEndReached={loadMoreData}
              onEndReachedThreshold={0.1}
              ListFooterComponent={renderLoadingFooter}
            />
          ) : (
            renderEmptyState()
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: 'white',
    fontSize: 16,
    marginTop: 16,
  },
  loadingFooter: {
    padding: 20,
    alignItems: 'center',
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
  statsSection: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  statsCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statDivider: {
    width: 1,
    height: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#8e8e93',
    fontWeight: '500',
  },
  filterSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  filterButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  filterButtonActive: {
    backgroundColor: '#4c6ef5',
  },
  filterButtonText: {
    color: '#8e8e93',
    fontSize: 14,
    fontWeight: '500',
  },
  filterButtonTextActive: {
    color: 'white',
  },
  listContainer: {
    flex: 1,
    paddingHorizontal: 20,
  },
  listContent: {
    paddingBottom: 100,
  },
  historyItem: {
    marginBottom: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  itemContent: {
    flexDirection: 'row',
  },
  itemImage: {
    marginRight: 16,
  },
  thumbnail: {
    width: 60,
    height: 60,
    borderRadius: 12,
  },
  thumbnailPlaceholder: {
    width: 60,
    height: 60,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  itemInfo: {
    flex: 1,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  resultBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  resultText: {
    fontSize: 16,
    fontWeight: '600',
  },
  itemDetails: {
    gap: 4,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  detailText: {
    fontSize: 12,
    color: '#8e8e93',
    flex: 1,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(142, 142, 147, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  emptyTitle: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  emptySubtitle: {
    color: '#8e8e93',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 22,
  },
  emptyButton: {
    backgroundColor: '#4c6ef5',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
  },
  emptyButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

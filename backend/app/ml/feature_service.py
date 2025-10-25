"""
Feature Service for managing GWO selected features
"""

import numpy as np
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FeatureService:
    def __init__(self):
        # Feature names từ Wisconsin Breast Cancer dataset (như trong notebook)
        self.all_feature_names = [
            "feature_0_mean", "feature_1_mean", "feature_2_mean", "feature_3_mean", "feature_4_mean",
            "feature_5_mean", "feature_6_mean", "feature_7_mean", "feature_8_mean", "feature_9_mean",
            "feature_0_se", "feature_1_se", "feature_2_se", "feature_3_se", "feature_4_se",
            "feature_5_se", "feature_6_se", "feature_7_se", "feature_8_se", "feature_9_se",
            "feature_0_worst", "feature_1_worst", "feature_2_worst", "feature_3_worst", "feature_4_worst",
            "feature_5_worst", "feature_6_worst", "feature_7_worst", "feature_8_worst", "feature_9_worst"
        ]
        
        # Mapping tên features thân thiện với người dùng
        self.feature_display_names = {
            "feature_0_mean": "Bán kính trung bình", "feature_1_mean": "Kết cấu trung bình", 
            "feature_2_mean": "Chu vi trung bình", "feature_3_mean": "Diện tích trung bình",
            "feature_4_mean": "Độ mượt trung bình", "feature_5_mean": "Độ nén trung bình",
            "feature_6_mean": "Độ lõm trung bình", "feature_7_mean": "Điểm lõm trung bình",
            "feature_8_mean": "Đối xứng trung bình", "feature_9_mean": "Chiều fractal trung bình",
            
            "feature_0_se": "Bán kính SE", "feature_1_se": "Kết cấu SE",
            "feature_2_se": "Chu vi SE", "feature_3_se": "Diện tích SE",
            "feature_4_se": "Độ mượt SE", "feature_5_se": "Độ nén SE",
            "feature_6_se": "Độ lõm SE", "feature_7_se": "Điểm lõm SE",
            "feature_8_se": "Đối xứng SE", "feature_9_se": "Chiều fractal SE",
            
            "feature_0_worst": "Bán kính xấu nhất", "feature_1_worst": "Kết cấu xấu nhất",
            "feature_2_worst": "Chu vi xấu nhất", "feature_3_worst": "Diện tích xấu nhất",
            "feature_4_worst": "Độ mượt xấu nhất", "feature_5_worst": "Độ nén xấu nhất",
            "feature_6_worst": "Độ lõm xấu nhất", "feature_7_worst": "Điểm lõm xấu nhất",
            "feature_8_worst": "Đối xứng xấu nhất", "feature_9_worst": "Chiều fractal xấu nhất"
        }
        
        # Features được chọn bởi GWO (từ kết quả notebook)
        # Random Forest selected features
        self.gwo_rf_features = [
            "feature_0_mean", "feature_1_se", "feature_6_se", "feature_0_worst", 
            "feature_1_worst", "feature_4_worst", "feature_5_worst"
        ]
        
        # SVM selected features  
        self.gwo_svm_features = [
            "feature_0_mean", "feature_1_se", "feature_6_se", "feature_0_worst",
            "feature_1_worst", "feature_4_worst", "feature_5_worst"
        ]
    
    def get_gwo_selected_features(self, model_type: str = "svm") -> Dict[str, Any]:
        """
        Lấy danh sách features được GWO chọn cho model type cụ thể
        
        Args:
            model_type: "rf" hoặc "svm"
            
        Returns:
            Dict chứa thông tin features được chọn
        """
        try:
            if model_type.lower() == "rf":
                selected_features = self.gwo_rf_features
            else:
                selected_features = self.gwo_svm_features
            
            # Tạo danh sách features với tên hiển thị
            feature_list = []
            for feature_name in selected_features:
                feature_info = {
                    "name": feature_name,
                    "displayName": self.feature_display_names.get(feature_name, feature_name),
                    "category": self._get_feature_category(feature_name),
                    "importance": self._get_feature_importance(feature_name, selected_features)
                }
                feature_list.append(feature_info)
            
            return {
                "modelType": model_type.upper(),
                "totalFeatures": len(self.all_feature_names),
                "selectedCount": len(selected_features),
                "selectionRatio": len(selected_features) / len(self.all_feature_names),
                "features": feature_list,
                "algorithm": "Grey Wolf Optimizer (GWO)"
            }
            
        except Exception as e:
            logger.error(f"Error getting GWO selected features: {e}")
            return {
                "modelType": model_type.upper(),
                "totalFeatures": 30,
                "selectedCount": 0,
                "selectionRatio": 0,
                "features": [],
                "algorithm": "Grey Wolf Optimizer (GWO)",
                "error": str(e)
            }
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Xác định category của feature"""
        if "_mean" in feature_name:
            return "Trung bình"
        elif "_se" in feature_name:
            return "Sai số chuẩn"
        elif "_worst" in feature_name:
            return "Xấu nhất"
        return "Khác"
    
    def _get_feature_importance(self, feature_name: str, selected_features: List[str]) -> float:
        """Tính importance score (đơn giản dựa trên thứ tự trong danh sách)"""
        try:
            # Importance giảm dần theo thứ tự trong danh sách
            index = selected_features.index(feature_name)
            return round(1.0 - (index * 0.1), 2)
        except ValueError:
            return 0.5
    
    def get_feature_comparison(self) -> Dict[str, Any]:
        """So sánh features được chọn giữa RF và SVM"""
        rf_features = set(self.gwo_rf_features)
        svm_features = set(self.gwo_svm_features)
        
        common_features = list(rf_features.intersection(svm_features))
        rf_only = list(rf_features - svm_features)
        svm_only = list(svm_features - rf_features)
        
        return {
            "commonFeatures": [
                {
                    "name": name,
                    "displayName": self.feature_display_names.get(name, name)
                } for name in common_features
            ],
            "rfOnlyFeatures": [
                {
                    "name": name,
                    "displayName": self.feature_display_names.get(name, name)
                } for name in rf_only
            ],
            "svmOnlyFeatures": [
                {
                    "name": name,
                    "displayName": self.feature_display_names.get(name, name)
                } for name in svm_only
            ],
            "totalCommon": len(common_features),
            "rfTotal": len(rf_features),
            "svmTotal": len(svm_features)
        }

# Global instance
feature_service = FeatureService()

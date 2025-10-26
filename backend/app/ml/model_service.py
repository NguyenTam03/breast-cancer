"""
Machine Learning Model Service for Breast Cancer Prediction
Supports two pathways:
- Direct CNN classifier on images (existing)
- GWO pipeline: CNN feature extractor + GWO-selected features + small classifier
"""

import os
import io
import time
import json
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Tuple, Dict, Any, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreastCancerPredictor:
    def __init__(self, model_path: str = r"models\model_gwo_selected_feature.h5"):
        """
        Initialize the breast cancer predictor with the trained CNN model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.input_shape = (224, 224, 3)  # Standard CNN input shape
        self.class_names = ["BENIGN", "MALIGNANT"]
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained CNN model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at: {self.model_path}")
                return False
            
            # Load the model
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Model loaded successfully from: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess the input image for model prediction
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            np.ndarray: Preprocessed image array ready for prediction
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((self.input_shape[0], self.input_shape[1]))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize pixel values to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Make prediction on the input image
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            Dict containing prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please check model file.")
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            start_time = time.time()
            predictions = self.model.predict(processed_image)
            processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            
            # Get prediction probability
            confidence = float(predictions[0][0])
            
            # Determine class based on threshold (0.5)
            predicted_class = "MALIGNANT" if confidence > 0.5 else "BENIGN"
            
            # If predicting malignant, confidence is the raw score
            # If predicting benign, confidence is 1 - raw_score
            final_confidence = confidence if predicted_class == "MALIGNANT" else (1 - confidence)
            
            return {
                "prediction": predicted_class,
                "confidence": round(final_confidence, 3),
                "processing_time": processing_time,
                "raw_score": round(confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict containing model information
        """
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        return {
            "status": "Model loaded",
            "input_shape": self.input_shape,
            "model_path": self.model_path,
            "classes": self.class_names,
            "model_summary": str(self.model.summary()) if self.model else None
        }

class GWOPredictor:
    """
    Predictor that uses CNN as feature extractor + GWO-selected features classifier.

    Artifacts expected:
    - CNN base: models/breast_cancer_cnn_model.h5 (or overridden)
    - GWO classifier: models/model_gwo_selected_feature.h5 or models/model_gwo.h5
    - Selected indices: gwo_selected_indices.npy (np.int32 array)
    - Feature extractor spec (optional): feature_extractor_spec.json
        {
          "cnn_model_path": "models/breast_cancer_cnn_model.h5",
          "layer_ref": {"type": "index", "value": -4} | {"type": "name", "value": "flatten_2"},
          "input_from_first_layer": true,
          "target_size": [224,224,3],
          "normalization": "scale_0_1"
        }
    """

    def __init__(
        self,
        cnn_model_path: str = "models/breast_cancer_cnn_model.h5",
        gwo_model_path: Optional[str] = None,
        selected_idx_path: str = "models/gwo_selected_indices.npy",
        extractor_spec_path: str = "models/feature_extractor_spec.json",
    ) -> None:
        self.cnn_model_path = self._resolve_first_existing([
            cnn_model_path,
            os.path.join("models", "breast_cancer_cnn_model.h5"),
            os.path.abspath(os.path.join("models", "breast_cancer_cnn_model.h5")),
            os.path.abspath(os.path.join("..", "models", "breast_cancer_cnn_model.h5")),
        ])
        self.gwo_model_path = self._resolve_first_existing([
            gwo_model_path or os.path.join("models", "model_gwo_selected_feature.h5"),
            os.path.join("models", "model_gwo.h5"),
            os.path.abspath(os.path.join("models", "model_gwo_selected_feature.h5")),
            os.path.abspath(os.path.join("models", "model_gwo.h5")),
        ])
        self.selected_idx_path = self._resolve_first_existing([
            selected_idx_path,
            os.path.join("models", "gwo_selected_indices.npy"),
            os.path.abspath(os.path.join("models", "gwo_selected_indices.npy")),
            os.path.abspath("gwo_selected_indices.npy"),
        ])
        self.extractor_spec_path = self._resolve_first_existing([
            extractor_spec_path,
            os.path.join("models", "feature_extractor_spec.json"),
            os.path.abspath(os.path.join("models", "feature_extractor_spec.json")),
        ], must_exist=False)

        self.cnn_model = None
        self.gwo_model = None
        self.feature_extractor = None
        self.selected_idx = None
        self.is_loaded = False
        self.input_shape = (224, 224, 3)
        self.class_names = ["BENIGN", "MALIGNANT"]

        self._load()

    def _resolve_first_existing(self, candidates, must_exist: bool = True) -> Optional[str]:
        for p in candidates:
            if p and os.path.exists(p):
                return p
        if must_exist:
            return candidates[0]  # return first, will fail later with clear error
        return None

    def _load(self) -> bool:
        try:
            
            if not os.path.exists(self.cnn_model_path):
                logger.error(f"CNN base model not found at: {self.cnn_model_path}")
                return False
            if not os.path.exists(self.gwo_model_path):
                logger.error(f"GWO classifier model not found at: {self.gwo_model_path}")
                return False
            if not os.path.exists(self.selected_idx_path):
                logger.error(f"GWO selected indices not found at: {self.selected_idx_path}")
                return False

            # Load base CNN
            self.cnn_model = tf.keras.models.load_model(self.cnn_model_path)

            # Determine extractor config
            layer_ref = {"type": "index", "value": -4}
            input_from_first = True
            target_size = list(self.input_shape)
            if self.extractor_spec_path and os.path.exists(self.extractor_spec_path):
                try:
                    with open(self.extractor_spec_path, "r", encoding="utf-8") as f:
                        spec = json.load(f)
                    layer_ref = spec.get("layer_ref", layer_ref)
                    input_from_first = spec.get("input_from_first_layer", input_from_first)
                    target_size = spec.get("target_size", target_size)
                except Exception as e:
                    logger.warning(f"Failed to read extractor spec: {e}. Using defaults.")

            self.input_shape = tuple(target_size)

            # Build feature extractor
            input_tensor = self.cnn_model.layers[0].input if input_from_first else self.cnn_model.input
            if layer_ref.get("type") == "name":
                feat_layer = self.cnn_model.get_layer(name=layer_ref.get("value", "flatten_2"))
            else:
                # default index -4
                feat_layer = self.cnn_model.get_layer(index=int(layer_ref.get("value", -4)))
            # Build feature extractor using tf.keras
            KModel = tf.keras.Model
            self.feature_extractor = KModel(inputs=input_tensor, outputs=feat_layer.output)

            # Load selected indices
            self.selected_idx = np.load(self.selected_idx_path)
            if self.selected_idx.ndim != 1:
                self.selected_idx = self.selected_idx.reshape(-1)

            # Load GWO classifier
            # Load GWO classifier using tf.keras
            self.gwo_model = tf.keras.models.load_model(self.gwo_model_path)

            self.is_loaded = True
            logger.info(
                f"GWO pipeline loaded. CNN: {self.cnn_model_path}, GWO: {self.gwo_model_path}, indices: {self.selected_idx_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading GWO pipeline: {e}")
            self.is_loaded = False
            return False

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((self.input_shape[1], self.input_shape[0]))
            arr = np.array(image).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            return arr
        except Exception as e:
            logger.error(f"Error preprocessing image (GWO): {e}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def predict(self, image_data: bytes) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("GWO pipeline is not loaded. Check model and indices files.")
        try:
            x = self.preprocess_image(image_data)  # (1, H, W, 3)

            # Extract features
            feat = self.feature_extractor.predict(x)
            feat = feat.reshape(1, -1)

            # Select GWO features
            feat_sel = feat[:, self.selected_idx]

            # Predict
            start = time.time()
            prob = float(self.gwo_model.predict(feat_sel)[0, 0])
            #predict model Cnn không co feat_sel
            # prob = float(self.cnn_model.predict(x)[0, 0])
            processing_time = int((time.time() - start) * 1000)

            predicted_class = "MALIGNANT" if prob > 0.5 else "BENIGN"
            final_confidence = prob if predicted_class == "MALIGNANT" else (1 - prob)

            return {
                "prediction": predicted_class,
                "confidence": round(final_confidence, 3),
                "processing_time": processing_time,
                "raw_score": round(prob, 3)
            }
        except Exception as e:
            logger.error(f"Error during GWO prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")


class FeatureBasedPredictor:
    """
    Predictor for feature-based analysis using rule-based approach with GWO-selected features.
    Uses 7 features selected by Grey Wolf Optimizer for direct classification.
    
    Note: Since model_gwo_selected_feature.h5 expects 1341 CNN features, 
    we use a simple rule-based classifier for direct 7-feature input.
    """
    
    def __init__(
        self,
        model_path: str = "models/breast_cancer_cnn_model.h5",  # Use CNN model as reference
        selected_idx_path: str = "models/gwo_selected_indices.npy"
    ) -> None:
        self.model_path = self._resolve_path(model_path)
        self.selected_idx_path = self._resolve_path(selected_idx_path)
        
        self.model = None
        self.selected_idx = None
        self.is_loaded = False
        self.class_names = ["BENIGN", "MALIGNANT"]
        
        # Feature names corresponding to the 30 original features
        self.feature_names = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", 
            "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]
        
        # Feature descriptions in Vietnamese
        self.feature_descriptions = {
            "radius_mean": "Bán kính trung bình",
            "texture_mean": "Texture trung bình", 
            "perimeter_mean": "Chu vi trung bình",
            "area_mean": "Diện tích trung bình",
            "smoothness_mean": "Độ mịn trung bình",
            "compactness_mean": "Độ compact trung bình",
            "concavity_mean": "Độ lõm trung bình",
            "concave_points_mean": "Điểm lõm trung bình",
            "symmetry_mean": "Độ đối xứng trung bình",
            "fractal_dimension_mean": "Chiều fractal trung bình",
            "radius_se": "Độ lệch chuẩn bán kính",
            "texture_se": "Độ lệch chuẩn texture",
            "perimeter_se": "Độ lệch chuẩn chu vi", 
            "area_se": "Độ lệch chuẩn diện tích",
            "smoothness_se": "Độ lệch chuẩn độ mịn",
            "compactness_se": "Độ lệch chuẩn độ compact",
            "concavity_se": "Độ lệch chuẩn độ lõm",
            "concave_points_se": "Độ lệch chuẩn điểm lõm",
            "symmetry_se": "Độ lệch chuẩn đối xứng",
            "fractal_dimension_se": "Độ lệch chuẩn chiều fractal",
            "radius_worst": "Bán kính tệ nhất",
            "texture_worst": "Texture tệ nhất",
            "perimeter_worst": "Chu vi tệ nhất",
            "area_worst": "Diện tích tệ nhất", 
            "smoothness_worst": "Độ mịn tệ nhất",
            "compactness_worst": "Độ compact tệ nhất",
            "concavity_worst": "Độ lõm tệ nhất",
            "concave_points_worst": "Điểm lõm tệ nhất",
            "symmetry_worst": "Độ đối xứng tệ nhất",
            "fractal_dimension_worst": "Chiều fractal tệ nhất"
        }
        
        self._load()
    
    def _resolve_path(self, path: str) -> str:
        """Resolve model file path"""
        candidates = [
            path,
            os.path.join("backend", path),
            os.path.abspath(path),
            os.path.abspath(os.path.join("backend", path))
        ]
        
        for p in candidates:
            if os.path.exists(p):
                return p
        return path  # Return original if none found
    
    def _load(self) -> bool:
        """Load for rule-based predictor"""
        try:
            # For rule-based predictor, we don't need to load complex models
            # Just validate that we can work with the feature selection
            self.is_loaded = True
            logger.info(f"Feature-based rule predictor loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading feature-based predictor: {e}")
            self.is_loaded = False
            return False
    
    def get_selected_feature_info(self) -> List[Dict[str, Any]]:
        """Get information about selected features"""
        if not self.is_loaded:
            return []
        
        # Use only the first 7 features for GWO selection (hardcoded for now)
        gwo_selected_indices = [0, 11, 16, 20, 21, 24, 25]  # Based on notebook GWO selection
        gwo_selected_names = [
            "radius_mean", "texture_se", "compactness_se", 
            "radius_worst", "texture_worst", "smoothness_worst", "compactness_worst"
        ]
        
        feature_info = []
        for i, (idx, name) in enumerate(zip(gwo_selected_indices, gwo_selected_names)):
            feature_info.append({
                "index": int(idx),
                "name": name,
                "description": self.feature_descriptions.get(name, name),
                "display_order": i
            })
        
        return feature_info
    
    def validate_features(self, features: List[float]) -> Tuple[bool, str]:
        """Validate input features"""
        if not self.is_loaded:
            return False, "Model not loaded"
        
        # Expect exactly 7 features for GWO selection
        expected_count = 7
        if len(features) != expected_count:
            return False, f"Expected {expected_count} features, got {len(features)}"
        
        # Check for valid numeric values
        for i, feature in enumerate(features):
            if not isinstance(feature, (int, float)) or np.isnan(feature) or np.isinf(feature):
                return False, f"Invalid value at feature {i}: {feature}"
        
        return True, "Valid"
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict breast cancer from feature values using rule-based approach
        
        Args:
            features: List of 7 feature values in order:
                      [radius_mean, texture_se, compactness_se, radius_worst, 
                       texture_worst, smoothness_worst, compactness_worst]
            
        Returns:
            Dict containing prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Feature-based model is not loaded. Please check model files.")
        
        # Validate input features
        is_valid, message = self.validate_features(features)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {message}")
        
        try:
            start_time = time.time()
            
            # Extract individual features
            radius_mean = features[0]      # Bán kính trung bình
            texture_se = features[1]       # Độ lệch chuẩn texture
            compactness_se = features[2]   # Độ lệch chuẩn độ compact
            radius_worst = features[3]     # Bán kính tệ nhất
            texture_worst = features[4]    # Texture tệ nhất
            smoothness_worst = features[5] # Độ mịn tệ nhất
            compactness_worst = features[6] # Độ compact tệ nhất
            
            # Rule-based scoring system based on Wisconsin Breast Cancer dataset thresholds
            malignant_score = 0.0
            total_weight = 0.0
            
            # Feature 1: radius_mean (weight: 0.20)
            # Malignant if > 14.0, Benign if < 12.0
            if radius_mean > 14.0:
                malignant_score += 0.20 * min((radius_mean - 14.0) / 6.0, 1.0)
            elif radius_mean < 12.0:
                malignant_score += 0.20 * max((12.0 - radius_mean) / 6.0, 0.0) * 0.1
            else:
                malignant_score += 0.20 * 0.5
            total_weight += 0.20
            
            # Feature 2: texture_se (weight: 0.15)
            # Malignant if > 1.2, Benign if < 0.8
            if texture_se > 1.2:
                malignant_score += 0.15 * min((texture_se - 1.2) / 2.0, 1.0)
            elif texture_se < 0.8:
                malignant_score += 0.15 * max((0.8 - texture_se) / 0.8, 0.0) * 0.1
            else:
                malignant_score += 0.15 * 0.5
            total_weight += 0.15
            
            # Feature 3: compactness_se (weight: 0.10)
            # Malignant if > 0.05, Benign if < 0.02
            if compactness_se > 0.05:
                malignant_score += 0.10 * min((compactness_se - 0.05) / 0.1, 1.0)
            elif compactness_se < 0.02:
                malignant_score += 0.10 * max((0.02 - compactness_se) / 0.02, 0.0) * 0.1
            else:
                malignant_score += 0.10 * 0.5
            total_weight += 0.10
            
            # Feature 4: radius_worst (weight: 0.25) - Most important
            # Malignant if > 16.0, Benign if < 13.0
            if radius_worst > 16.0:
                malignant_score += 0.25 * min((radius_worst - 16.0) / 10.0, 1.0)
            elif radius_worst < 13.0:
                malignant_score += 0.25 * max((13.0 - radius_worst) / 6.0, 0.0) * 0.1
            else:
                malignant_score += 0.25 * 0.5
            total_weight += 0.25
            
            # Feature 5: texture_worst (weight: 0.15)
            # Malignant if > 25.0, Benign if < 20.0
            if texture_worst > 25.0:
                malignant_score += 0.15 * min((texture_worst - 25.0) / 15.0, 1.0)
            elif texture_worst < 20.0:
                malignant_score += 0.15 * max((20.0 - texture_worst) / 10.0, 0.0) * 0.1
            else:
                malignant_score += 0.15 * 0.5
            total_weight += 0.15
            
            # Feature 6: smoothness_worst (weight: 0.10)
            # Malignant if > 0.13, Benign if < 0.10
            if smoothness_worst > 0.13:
                malignant_score += 0.10 * min((smoothness_worst - 0.13) / 0.1, 1.0)
            elif smoothness_worst < 0.10:
                malignant_score += 0.10 * max((0.10 - smoothness_worst) / 0.05, 0.0) * 0.1
            else:
                malignant_score += 0.10 * 0.5
            total_weight += 0.10
            
            # Feature 7: compactness_worst (weight: 0.05)
            # Malignant if > 0.25, Benign if < 0.15
            if compactness_worst > 0.25:
                malignant_score += 0.05 * min((compactness_worst - 0.25) / 0.5, 1.0)
            elif compactness_worst < 0.15:
                malignant_score += 0.05 * max((0.15 - compactness_worst) / 0.1, 0.0) * 0.1
            else:
                malignant_score += 0.05 * 0.5
            total_weight += 0.05
            
            # Normalize score
            if total_weight > 0:
                normalized_score = malignant_score / total_weight
            else:
                normalized_score = 0.5
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Apply sigmoid-like transformation for more realistic confidence
            confidence_raw = 1.0 / (1.0 + np.exp(-6.0 * (normalized_score - 0.5)))
            
            # Determine class based on threshold (0.5)
            predicted_class = "MALIGNANT" if confidence_raw > 0.5 else "BENIGN"
            
            # Calculate final confidence
            final_confidence = confidence_raw if predicted_class == "MALIGNANT" else (1 - confidence_raw)
            
            return {
                "prediction": predicted_class,
                "confidence": round(final_confidence, 3),
                "processing_time": processing_time,
                "raw_score": round(confidence_raw, 3),
                "features_used": len(features),
                "analysis_type": "features",
                "rule_based": True
            }
            
        except Exception as e:
            logger.error(f"Error during feature-based prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")


# Global model instances
_predictor_instance = None
_feature_predictor_instance = None

def get_predictor() -> GWOPredictor:
    """
    Get the global GWO predictor instance (singleton pattern)
    
    Returns:
        GWOPredictor: The GWO predictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = GWOPredictor()
    return _predictor_instance

def get_feature_predictor() -> FeatureBasedPredictor:
    """
    Get the global feature-based predictor instance (singleton pattern)
    
    Returns:
        FeatureBasedPredictor: The feature-based predictor instance
    """
    global _feature_predictor_instance
    if _feature_predictor_instance is None:
        _feature_predictor_instance = FeatureBasedPredictor()
    return _feature_predictor_instance

def predict_breast_cancer(image_data: bytes) -> Dict[str, Any]:
    """
    Convenience function to predict breast cancer from image data using GWO model
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        Dict containing prediction results
    """
    predictor = get_predictor()
    return predictor.predict(image_data)

def predict_breast_cancer_features(features: List[float]) -> Dict[str, Any]:
    """
    Convenience function to predict breast cancer from feature values
    
    Args:
        features: List of feature values
        
    Returns:
        Dict containing prediction results
    """
    predictor = get_feature_predictor()
    return predictor.predict(features)

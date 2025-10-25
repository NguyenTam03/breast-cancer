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
import pickle
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreastCancerPredictor:
    def __init__(self, model_path: str = "models\model_gwo_selected_feature.h5"):
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


# Global GWO model instance
_predictor_instance = None

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

class FeaturePredictor:
    """
    Predictor for manual feature input based on Wisconsin Breast Cancer dataset features.
    Supports both SVM full features and GWO-selected features approach.
    """
    
    def __init__(self):
        self.feature_names = [
            'feature_0_mean', 'feature_1_mean', 'feature_2_mean', 'feature_3_mean', 'feature_4_mean',
            'feature_5_mean', 'feature_6_mean', 'feature_7_mean', 'feature_8_mean', 'feature_9_mean',
            'feature_0_se', 'feature_1_se', 'feature_2_se', 'feature_3_se', 'feature_4_se',
            'feature_5_se', 'feature_6_se', 'feature_7_se', 'feature_8_se', 'feature_9_se',
            'feature_0_worst', 'feature_1_worst', 'feature_2_worst', 'feature_3_worst', 'feature_4_worst',
            'feature_5_worst', 'feature_6_worst', 'feature_7_worst', 'feature_8_worst', 'feature_9_worst'
        ]
        
        # GWO selected features for SVM (from notebook)
        self.gwo_selected_features = [
            'feature_0_mean', 'feature_1_se', 'feature_6_se', 
            'feature_0_worst', 'feature_1_worst', 'feature_4_worst', 'feature_5_worst'
        ]
        
        # Feature descriptions for educational purposes
        self.feature_descriptions = {
            'feature_0_mean': 'Radius (mean)',
            'feature_1_mean': 'Texture (mean)', 
            'feature_2_mean': 'Perimeter (mean)',
            'feature_3_mean': 'Area (mean)',
            'feature_4_mean': 'Smoothness (mean)',
            'feature_5_mean': 'Compactness (mean)',
            'feature_6_mean': 'Concavity (mean)',
            'feature_7_mean': 'Concave points (mean)',
            'feature_8_mean': 'Symmetry (mean)',
            'feature_9_mean': 'Fractal dimension (mean)',
            'feature_0_se': 'Radius (SE)',
            'feature_1_se': 'Texture (SE)',
            'feature_2_se': 'Perimeter (SE)',
            'feature_3_se': 'Area (SE)',
            'feature_4_se': 'Smoothness (SE)',
            'feature_5_se': 'Compactness (SE)',
            'feature_6_se': 'Concavity (SE)',
            'feature_7_se': 'Concave points (SE)',
            'feature_8_se': 'Symmetry (SE)',
            'feature_9_se': 'Fractal dimension (SE)',
            'feature_0_worst': 'Radius (worst)',
            'feature_1_worst': 'Texture (worst)',
            'feature_2_worst': 'Perimeter (worst)',
            'feature_3_worst': 'Area (worst)',
            'feature_4_worst': 'Smoothness (worst)',
            'feature_5_worst': 'Compactness (worst)',
            'feature_6_worst': 'Concavity (worst)',
            'feature_7_worst': 'Concave points (worst)',
            'feature_8_worst': 'Symmetry (worst)',
            'feature_9_worst': 'Fractal dimension (worst)'
        }
        
        # Feature means for filling missing values (typical dataset values)
        self.feature_means = {
            'feature_0_mean': 14.127, 'feature_1_mean': 19.289, 'feature_2_mean': 91.969,
            'feature_3_mean': 654.889, 'feature_4_mean': 0.096, 'feature_5_mean': 0.104,
            'feature_6_mean': 0.089, 'feature_7_mean': 0.048, 'feature_8_mean': 0.181,
            'feature_9_mean': 0.063, 'feature_0_se': 0.406, 'feature_1_se': 1.217,
            'feature_2_se': 2.866, 'feature_3_se': 40.337, 'feature_4_se': 0.007,
            'feature_5_se': 0.025, 'feature_6_se': 0.032, 'feature_7_se': 0.012,
            'feature_8_se': 0.021, 'feature_9_se': 0.004, 'feature_0_worst': 16.269,
            'feature_1_worst': 25.677, 'feature_2_worst': 107.261, 'feature_3_worst': 880.583,
            'feature_4_worst': 0.132, 'feature_5_worst': 0.254, 'feature_6_worst': 0.272,
            'feature_7_worst': 0.115, 'feature_8_worst': 0.290, 'feature_9_worst': 0.084
        }
        
        self.scaler_path = "models/scaler.pkl"
        self.svm_full_model_path = "models/svm_full_model.pkl"  
        self.svm_gwo_model_path = "models/svm_gwo_model.pkl"
        
        # Try to load models if available
        self.scaler = None
        self.svm_full_model = None
        self.svm_gwo_model = None
        self._load_models()
    
    def _load_models(self):
        """Load pickle models if available"""
        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
        
        try:
            if os.path.exists(self.svm_full_model_path):
                with open(self.svm_full_model_path, 'rb') as f:
                    self.svm_full_model = pickle.load(f)
                logger.info("SVM full model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SVM full model: {e}")
            
        try:
            if os.path.exists(self.svm_gwo_model_path):
                with open(self.svm_gwo_model_path, 'rb') as f:
                    self.svm_gwo_model = pickle.load(f)
                logger.info("SVM GWO model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load SVM GWO model: {e}")
    
    def get_gwo_features_info(self) -> List[Dict[str, Any]]:
        """Get information about GWO selected features for UI"""
        features_info = []
        for feature_name in self.gwo_selected_features:
            features_info.append({
                "name": feature_name,
                "description": self.feature_descriptions.get(feature_name, feature_name),
                "default_value": self.feature_means.get(feature_name, 0),
                "is_required": True
            })
        return features_info
    
    def predict_from_features(self, feature_input: Dict[str, float], use_gwo: bool = True) -> Dict[str, Any]:
        """
        Predict from manual feature input
        
        Args:
            feature_input: Dict with feature names as keys and values
            use_gwo: If True, use GWO-selected features only
            
        Returns:
            Dict containing prediction results
        """
        try:
            start_time = time.time()
            
            if use_gwo:
                # Use only GWO selected features
                if not self.svm_gwo_model:
                    # Fallback to simple rule-based prediction for demo
                    return self._rule_based_prediction(feature_input)
                
                # Prepare full feature vector with means as defaults
                full_features = [self.feature_means[fname] for fname in self.feature_names]
                
                # Update with provided values
                for fname, value in feature_input.items():
                    if fname in self.feature_names:
                        idx = self.feature_names.index(fname)
                        full_features[idx] = value
                
                # Scale features
                if self.scaler:
                    X_df = pd.DataFrame([full_features], columns=self.feature_names)
                    X_scaled = self.scaler.transform(X_df)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
                    
                    # Select GWO features
                    X_gwo = X_scaled_df[self.gwo_selected_features].values
                    
                    # Predict
                    prediction = self.svm_gwo_model.predict(X_gwo)[0]
                    probabilities = self.svm_gwo_model.predict_proba(X_gwo)[0]
                    confidence = probabilities[1] if prediction == 1 else probabilities[0]
                else:
                    return self._rule_based_prediction(feature_input)
            else:
                # Use all features
                if not self.svm_full_model or not self.scaler:
                    return self._rule_based_prediction(feature_input)
                
                # Prepare feature vector
                full_features = [feature_input.get(fname, self.feature_means[fname]) 
                               for fname in self.feature_names]
                
                # Scale and predict
                X_scaled = self.scaler.transform(np.array(full_features).reshape(1, -1))
                prediction = self.svm_full_model.predict(X_scaled)[0]
                probabilities = self.svm_full_model.predict_proba(X_scaled)[0]
                confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            processing_time = int((time.time() - start_time) * 1000)
            predicted_class = "MALIGNANT" if prediction == 1 else "BENIGN"
            
            return {
                "prediction": predicted_class,
                "confidence": round(float(confidence), 3),
                "processing_time": processing_time,
                "method": "GWO-SVM" if use_gwo else "Full-SVM",
                "features_used": len(self.gwo_selected_features) if use_gwo else len(self.feature_names)
            }
            
        except Exception as e:
            logger.error(f"Error in feature prediction: {e}")
            return self._rule_based_prediction(feature_input)
    
    def _rule_based_prediction(self, feature_input: Dict[str, float]) -> Dict[str, Any]:
        """Fallback rule-based prediction for demo purposes"""
        try:
            # Simple rule based on key features
            radius_mean = feature_input.get('feature_0_mean', 14.0)
            texture_worst = feature_input.get('feature_1_worst', 25.0)
            area_worst = feature_input.get('feature_3_worst', 800.0)
            
            # Simple thresholds (for demo only)
            risk_score = 0
            if radius_mean > 15.0:
                risk_score += 0.3
            if texture_worst > 25.0:
                risk_score += 0.3
            if area_worst > 900.0:
                risk_score += 0.4
            
            predicted_class = "MALIGNANT" if risk_score > 0.5 else "BENIGN"
            confidence = risk_score if predicted_class == "MALIGNANT" else (1 - risk_score)
            
            return {
                "prediction": predicted_class,
                "confidence": round(confidence, 3),
                "processing_time": 10,
                "method": "Rule-based (fallback)",
                "features_used": len(feature_input)
            }
        except Exception as e:
            logger.error(f"Error in rule-based prediction: {e}")
            return {
                "prediction": "BENIGN",
                "confidence": 0.5,
                "processing_time": 10,
                "method": "Default (error)",
                "features_used": 0
            }


# Global instances
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

def get_feature_predictor() -> FeaturePredictor:
    """
    Get the global feature predictor instance (singleton pattern)
    
    Returns:
        FeaturePredictor: The feature predictor instance
    """
    global _feature_predictor_instance
    if _feature_predictor_instance is None:
        _feature_predictor_instance = FeaturePredictor()
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

def predict_from_features(feature_input: Dict[str, float], use_gwo: bool = True) -> Dict[str, Any]:
    """
    Convenience function to predict breast cancer from manual features
    
    Args:
        feature_input: Dict with feature names and values
        use_gwo: Use GWO selected features only
        
    Returns:
        Dict containing prediction results
    """
    predictor = get_feature_predictor()
    return predictor.predict_from_features(feature_input, use_gwo)

def get_features_info() -> List[Dict[str, Any]]:
    """
    Get information about GWO selected features for UI
    
    Returns:
        List of feature information dictionaries
    """
    predictor = get_feature_predictor()
    return predictor.get_gwo_features_info()

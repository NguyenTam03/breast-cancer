"""
Machine Learning Model Service for Breast Cancer Prediction
Uses GWO-optimized pipeline (3-class classification):
Image → CNN Feature Extractor → GWO Selected Features → Classifier → 3 Classes
"""

import os
import io
import time
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GWOPredictor:
    """
    GWO-optimized breast cancer predictor
    
    Pipeline:
    1. CNN base model: Extract features from image
    2. GWO selected indices: Select important features (65 features)
    3. GWO classifier: Predict 3 classes [BENIGN, MALIGNANT, NORMAL]
    """

    def __init__(
        self,
        cnn_model_path: str = "models/breast_cancer_cnn_model.h5",
        gwo_model_path: str = "models/model_gwo_selected_feature.h5",
        selected_idx_path: str = "models/gwo_selected_indices.npy"
    ) -> None:
        """
        Initialize GWO predictor pipeline
        
        Args:
            cnn_model_path: Path to CNN feature extractor
            gwo_model_path: Path to GWO classifier
            selected_idx_path: Path to GWO selected feature indices
        """
        # Resolve paths
        self.cnn_model_path = self._resolve_first_existing([
            cnn_model_path,
            os.path.join("models", "breast_cancer_cnn_model.h5"),
            os.path.abspath(os.path.join("models", "breast_cancer_cnn_model.h5")),
            os.path.abspath(os.path.join("..", "models", "breast_cancer_cnn_model.h5")),
        ])
        
        self.gwo_model_path = self._resolve_first_existing([
            gwo_model_path,
            os.path.join("models", "model_gwo_selected_feature.h5"),
            os.path.abspath(os.path.join("models", "model_gwo_selected_feature.h5")),
            os.path.abspath(os.path.join("..", "models", "model_gwo_selected_feature.h5")),
        ])
        
        self.selected_idx_path = self._resolve_first_existing([
            selected_idx_path,
            os.path.join("models", "gwo_selected_indices.npy"),
            os.path.abspath(os.path.join("models", "gwo_selected_indices.npy")),
            os.path.abspath(os.path.join("..", "models", "gwo_selected_indices.npy")),
        ])
        
        self.cnn_model = None
        self.gwo_model = None
        self.feature_extractor = None
        self.selected_idx = None
        self.is_loaded = False
        self.input_shape = (224, 224, 3)
        self.class_names = ["BENIGN", "MALIGNANT", "NORMAL"]
        
        self._load()

    def _resolve_first_existing(self, candidates) -> str:
        """Find first existing file path from candidates"""
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return candidates[0]  # Return first as fallback

    def _load(self) -> bool:
        """Load all components of GWO pipeline"""
        try:
            # Check all required files exist
            if not os.path.exists(self.cnn_model_path):
                logger.error(f"CNN model not found at: {self.cnn_model_path}")
                return False
            if not os.path.exists(self.gwo_model_path):
                logger.error(f"GWO classifier not found at: {self.gwo_model_path}")
                return False
            if not os.path.exists(self.selected_idx_path):
                logger.error(f"GWO indices not found at: {self.selected_idx_path}")
                return False
            
            # Load CNN base model
            self.cnn_model = tf.keras.models.load_model(self.cnn_model_path, compile=False)
            
            # Build the model with input shape
            self.cnn_model.build((None, 224, 224, 3))
            
            # Get feature layer (layer -4)
            feature_layer = self.cnn_model.layers[-4]
            
            # Create new input layer
            input_tensor = tf.keras.Input(shape=(224, 224, 3))
            
            # Build feature extractor by calling model layers
            x = input_tensor
            for i, layer in enumerate(self.cnn_model.layers):
                x = layer(x)
                if i == len(self.cnn_model.layers) - 4:  # Stop at layer -4
                    break
            
            self.feature_extractor = tf.keras.Model(inputs=input_tensor, outputs=x)
            
            # Load GWO selected indices
            self.selected_idx = np.load(self.selected_idx_path)
            if self.selected_idx.ndim != 1:
                self.selected_idx = self.selected_idx.reshape(-1)
            
            # Load GWO classifier
            self.gwo_model = tf.keras.models.load_model(self.gwo_model_path)
            
            self.is_loaded = True
            logger.info(f"GWO pipeline loaded successfully")
            logger.info(f"  CNN: {self.cnn_model_path}")
            logger.info(f"  GWO: {self.gwo_model_path}")
            logger.info(f"  Selected features: {len(self.selected_idx)}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading GWO pipeline: {e}")
            self.is_loaded = False
            return False

    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """
        Preprocess image for model prediction
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            Preprocessed image array (1, 224, 224, 3)
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((self.input_shape[0], self.input_shape[1]))
            
            # Convert to array and normalize [0, 1]
            arr = np.array(image).astype(np.float32) / 255.0
            
            # Add batch dimension
            arr = np.expand_dims(arr, axis=0)
            
            return arr
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """
        Predict breast cancer class from image using GWO pipeline
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("GWO pipeline is not loaded. Check model files.")
        
        try:
            # Step 1: Preprocess image
            x = self.preprocess_image(image_data)  # (1, 224, 224, 3)
            
            # Step 2: Extract features using CNN
            features = self.feature_extractor.predict(x)  # (1, num_features)
            features = features.reshape(1, -1)  # Flatten if needed
            
            # Step 3: Select GWO features
            selected_features = features[:, self.selected_idx]  # (1, 65)
            
            # Step 4: Predict using GWO classifier
            start = time.time()
            probs = self.gwo_model.predict(selected_features)[0]  # shape: (3,)
            processing_time = int((time.time() - start) * 1000)
            
            # Get predicted class
            predicted_idx = int(np.argmax(probs))
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probs[predicted_idx])
            
            # Create probabilities dictionary
            probabilities = {
                self.class_names[i]: round(float(probs[i]), 3) 
                for i in range(len(self.class_names))
            }
            
            return {
                "prediction": predicted_class,
                "confidence": round(confidence, 3),
                "processing_time": processing_time,
                "probabilities": probabilities
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        return {
            "status": "Model loaded",
            "input_shape": self.input_shape,
            "model_path": self.model_path,
            "classes": self.class_names,
            "num_classes": len(self.class_names)
        }


# Global model instances
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

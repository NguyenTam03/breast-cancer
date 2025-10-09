"""
Machine Learning Model Service for Breast Cancer Prediction
"""

import os
import io
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreastCancerPredictor:
    def __init__(self, model_path: str = "models/breast_cancer_cnn_model.h5"):
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

# Global model instance
_predictor_instance = None

def get_predictor() -> BreastCancerPredictor:
    """
    Get the global predictor instance (singleton pattern)
    
    Returns:
        BreastCancerPredictor: The predictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = BreastCancerPredictor()
    return _predictor_instance

def predict_breast_cancer(image_data: bytes) -> Dict[str, Any]:
    """
    Convenience function to predict breast cancer from image data
    
    Args:
        image_data: Raw image data in bytes
        
    Returns:
        Dict containing prediction results
    """
    predictor = get_predictor()
    return predictor.predict(image_data)

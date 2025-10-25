"""
Analysis endpoints for BreastCare AI Backend
"""

import os
import io
import uuid
import aiofiles
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer
from typing import List, Optional
from PIL import Image
from beanie import PydanticObjectId

from app.models.analysis import Analysis, FeatureAnalysis, PredictionResult, ImageInfo, MLResults, AnalysisMetadata, FeatureData
from app.models.user import User
from app.ml.model_service import predict_breast_cancer, get_predictor, predict_from_features, get_features_info
from app.utils.dependencies import get_current_user
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()


class FeatureAnalysisRequest(BaseModel):
    feature_data: dict
    use_gwo: bool = True
    notes: Optional[str] = None


@router.post("/predict")
async def analyze_image(
    image: UploadFile = File(...),
    notes: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Analyze uploaded image for breast cancer detection"""
    
    # Validate image file
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Check file size (max 10MB)
    if image.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size too large. Maximum 10MB allowed."
        )
    
    try:
        # Read image data
        image_data = await image.read()
        
        # Get image dimensions
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            width, height = pil_image.size
            file_size = len(image_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file"
            )
        
        # Run ML model prediction
        try:
            prediction_result = predict_breast_cancer(image_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Save uploaded image to uploads directory
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create filename with analysis ID
        file_extension = os.path.splitext(image.filename)[1]
        saved_filename = f"{analysis_id}{file_extension}"
        file_path = os.path.join(upload_dir, saved_filename)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(image_data)
        
        # Prepare response
        analysis_result = {
            "id": analysis_id,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "processingTime": prediction_result["processing_time"],
            "analysisDate": datetime.utcnow().isoformat() + "Z",
            "imageInfo": {
                "originalName": image.filename,
                "savedPath": file_path,
                "fileSize": file_size,
                "mimeType": image.content_type,
                "dimensions": {"width": width, "height": height}
            },
            "userNotes": notes,
            "isBookmarked": False,
            "tags": [],
            "rawScore": prediction_result.get("raw_score", 0)
        }
        
        # Save analysis results to database
        try:
            # Create Analysis document
            analysis_doc = Analysis(
                userId=current_user.id,  # Use current authenticated user
                imageInfo=ImageInfo(
                    originalName=image.filename,
                    filePath=file_path,
                    fileSize=file_size,
                    mimeType=image.content_type,
                    dimensions={"width": width, "height": height}
                ),
                mlResults=MLResults(
                    prediction=prediction_result["prediction"],
                    confidence=prediction_result["confidence"],
                    processingTime=prediction_result["processing_time"],
                    rawOutput=prediction_result.get("raw_score", 0)
                ),
                metadata=AnalysisMetadata(
                    deviceInfo="Mobile",
                    appVersion="1.0.0"
                ),
                userNotes=notes
            )
            
            # Save to database
            saved_analysis = await analysis_doc.save()
            
            # Update response with saved ID
            analysis_result["id"] = str(saved_analysis.id)
            
        except Exception as e:
            logger.error(f"Failed to save analysis to database: {e}")
            # Continue with response even if database save fails
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/image/{analysis_id}")
async def get_analysis_image(analysis_id: str):
    """Get the image associated with an analysis"""
    try:
        # Find analysis by ID
        analysis = await Analysis.get(PydanticObjectId(analysis_id))
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Check if image file exists
        image_path = analysis.imageInfo.filePath
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image file not found"
            )
        
        # Return image file
        return FileResponse(
            path=image_path,
            media_type=analysis.imageInfo.mimeType,
            filename=analysis.imageInfo.originalName
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve image: {str(e)}"
        )


@router.get("/history")
async def get_analysis_history(
    page: int = 1,
    pageSize: int = 10,
    filter_prediction: Optional[str] = None
    # TODO: Re-enable authentication after testing
    # current_user: User = Depends(get_current_user)
):
    """Get user's analysis history - returns all analyses (for testing)"""
    try:
        # Build query filter
        query_filter = {}
        if filter_prediction:
            query_filter["mlResults.prediction"] = filter_prediction.upper()
        
        all_analyses = []
        
        # Get image analyses with error handling
        try:
            image_analyses = await Analysis.find(query_filter)\
                .sort("-createdAt")\
                .to_list()
            
            # Format image analyses
            for analysis in image_analyses:
                try:
                    formatted_analysis = {
                        "id": str(analysis.id),
                        "prediction": analysis.mlResults.prediction,
                        "confidence": analysis.mlResults.confidence,
                        "processingTime": analysis.mlResults.processingTime,
                        "analysisDate": analysis.metadata.analysisDate.isoformat() + "Z",
                        "analysisType": "image",
                        "imageInfo": {
                            "originalName": analysis.imageInfo.originalName,
                            "fileSize": analysis.imageInfo.fileSize,
                            "mimeType": analysis.imageInfo.mimeType,
                            "dimensions": analysis.imageInfo.dimensions
                        },
                        "imageUrl": f"/analysis/image/{str(analysis.id)}",
                        "userNotes": analysis.userNotes,
                        "isBookmarked": analysis.isBookmarked,
                        "tags": analysis.tags,
                        "createdAt": analysis.createdAt
                    }
                    all_analyses.append(formatted_analysis)
                except Exception as format_error:
                    logger.warning(f"Failed to format image analysis {analysis.id}: {format_error}")
                    continue
        except Exception as image_error:
            logger.warning(f"Failed to get image analyses: {image_error}")
        
        # Get feature analyses with error handling
        try:
            feature_analyses = await FeatureAnalysis.find(query_filter)\
                .sort("-createdAt")\
                .to_list()
            
            # Format feature analyses
            for analysis in feature_analyses:
                try:
                    formatted_analysis = {
                        "id": str(analysis.id),
                        "prediction": analysis.mlResults.prediction,
                        "confidence": analysis.mlResults.confidence,
                        "processingTime": analysis.mlResults.processingTime,
                        "analysisDate": analysis.metadata.analysisDate.isoformat() + "Z",
                        "analysisType": "features",
                        "method": analysis.featureData.method,
                        "featuresUsed": analysis.featureData.featuresUsed,
                        "useGWO": analysis.featureData.useGWO,
                        "inputFeatures": analysis.featureData.inputFeatures,
                        "userNotes": analysis.userNotes,
                        "isBookmarked": analysis.isBookmarked,
                        "tags": analysis.tags,
                        "createdAt": analysis.createdAt
                    }
                    all_analyses.append(formatted_analysis)
                except Exception as format_error:
                    logger.warning(f"Failed to format feature analysis {analysis.id}: {format_error}")
                    continue
        except Exception as feature_error:
            logger.warning(f"Failed to get feature analyses: {feature_error}")
        
        # Sort combined analyses by creation date (newest first)
        try:
            all_analyses.sort(key=lambda x: x["createdAt"], reverse=True)
        except Exception as sort_error:
            logger.warning(f"Failed to sort analyses: {sort_error}")
        
        # Calculate pagination
        total_count = len(all_analyses)
        skip = (page - 1) * pageSize
        paginated_analyses = all_analyses[skip:skip + pageSize]
        
        # Remove createdAt from final response (used only for sorting)
        for analysis in paginated_analyses:
            if "createdAt" in analysis:
                del analysis["createdAt"]
        
        return {
            "analyses": paginated_analyses,
            "totalCount": total_count,
            "page": page,
            "pageSize": pageSize
        }
        
    except Exception as e:
        logger.error(f"Failed to get analysis history: {e}")
        # Return empty result instead of error to prevent app crash
        return {
            "analyses": [],
            "totalCount": 0,
            "page": page,
            "pageSize": pageSize,
            "error": "Failed to retrieve analysis history"
        }


@router.get("/history/{user_id}")
async def get_user_analysis_history(
    user_id: str,
    page: int = 1,
    pageSize: int = 10,
    filter_prediction: Optional[str] = None
):
    """Get analysis history for specific user"""
    try:
        # Calculate skip for pagination
        skip = (page - 1) * pageSize
        
        # Build query filter for specific user
        query_filter = {"userId": PydanticObjectId(user_id)}
        if filter_prediction:
            query_filter["mlResults.prediction"] = filter_prediction.upper()
        
        # Get total count for this user
        total_count = await Analysis.find(query_filter).count()
        
        # Get analyses with pagination for this user
        analyses = await Analysis.find(query_filter)\
            .sort("-createdAt")\
            .skip(skip)\
            .limit(pageSize)\
            .to_list()
        
        # Format analyses to match mobile app expected structure
        formatted_analyses = []
        for analysis in analyses:
            formatted_analysis = {
                "id": str(analysis.id),
                "prediction": analysis.mlResults.prediction,
                "confidence": analysis.mlResults.confidence,
                "processingTime": analysis.mlResults.processingTime,
                "analysisDate": analysis.metadata.analysisDate.isoformat() + "Z",
                "imageInfo": {
                    "originalName": analysis.imageInfo.originalName,
                    "fileSize": analysis.imageInfo.fileSize,
                    "mimeType": analysis.imageInfo.mimeType,
                    "dimensions": analysis.imageInfo.dimensions
                },
                "imageUrl": f"/analysis/image/{str(analysis.id)}",  # Add image URL
                "userNotes": analysis.userNotes,
                "isBookmarked": analysis.isBookmarked,
                "tags": analysis.tags,
                "userId": str(analysis.userId)  # Include userId in response
            }
            formatted_analyses.append(formatted_analysis)
        
        return {
            "analyses": formatted_analyses,
            "totalCount": total_count,
            "page": page,
            "pageSize": pageSize,
            "userId": user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to get user analysis history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user analysis history: {str(e)}"
        )


@router.get("/{analysis_id}")
async def get_analysis_details(analysis_id: str):
    """Get details of a specific analysis"""
    # TODO: Get analysis by ID from database
    # TODO: Check if user owns this analysis
    
    return {
        "id": analysis_id,
        "prediction": "BENIGN",
        "confidence": 0.85,
        "processingTime": 1500,
        "analysisDate": "2024-10-04T18:23:00Z",
        "imageInfo": {
            "originalName": "sample.jpg",
            "fileSize": 1024000,
            "mimeType": "image/jpeg",
            "dimensions": {"width": 224, "height": 224}
        },
        "userNotes": "Sample analysis",
        "isBookmarked": False,
        "tags": ["sample"]
    }


@router.put("/{analysis_id}")
async def update_analysis(analysis_id: str, update_data: dict):
    """Update analysis notes or tags"""
    # TODO: Update analysis in database
    # TODO: Check if user owns this analysis
    
    return {
        "message": "Analysis updated successfully",
        "analysis": update_data
    }


@router.delete("/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis"""
    try:
        # Find analysis by ID
        analysis = await Analysis.get(PydanticObjectId(analysis_id))
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Delete associated image file if exists
        try:
            if analysis.imageInfo.filePath and os.path.exists(analysis.imageInfo.filePath):
                os.remove(analysis.imageInfo.filePath)
        except Exception as e:
            logger.warning(f"Failed to delete image file: {e}")
        
        # Delete analysis from database
        await analysis.delete()
        
        return {
            "message": "Analysis deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete analysis: {str(e)}"
        )


@router.post("/{analysis_id}/bookmark")
async def toggle_bookmark(analysis_id: str):
    """Toggle bookmark status of an analysis"""
    # TODO: Toggle bookmark status in database
    
    return {
        "message": "Bookmark toggled successfully",
        "isBookmarked": True
    }


@router.get("/features/info")
async def get_features_information():
    """Get information about GWO-selected features for UI form"""
    try:
        features_info = get_features_info()
        return {
            "success": True,
            "features": features_info,
            "totalFeatures": len(features_info),
            "description": "GWO-selected features for breast cancer diagnosis"
        }
    except Exception as e:
        logger.error(f"Failed to get features info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve features information: {str(e)}"
        )


@router.post("/predict/features")
async def analyze_features(
    request: FeatureAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze manually input features for breast cancer detection"""
    try:
        # Extract data from request
        feature_data = request.feature_data
        use_gwo = request.use_gwo
        notes = request.notes
        
        # Validate feature data
        if not feature_data or not isinstance(feature_data, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feature data must be a non-empty dictionary"
            )
        
        # Log received data for debugging
        logger.info(f"Received request: {request}")
        logger.info(f"Received feature_data: {feature_data}")
        
        # Convert all values to floats with better error handling
        cleaned_features = {}
        for key, value in feature_data.items():
            if value is None:
                logger.warning(f"Feature '{key}' has None value, skipping")
                continue
                
            # Handle various input types
            try:
                if isinstance(value, (int, float)):
                    cleaned_features[key] = float(value)
                elif isinstance(value, str):
                    # Remove whitespace and handle empty strings
                    cleaned_value = value.strip()
                    if not cleaned_value:
                        logger.warning(f"Feature '{key}' has empty string value, skipping")
                        continue
                    cleaned_features[key] = float(cleaned_value)
                else:
                    raise ValueError(f"Unsupported type: {type(value)}")
                    
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert feature '{key}' with value '{value}' to float: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid value for feature '{key}': must be a number (received: {value}, type: {type(value).__name__})"
                )
        
        if not cleaned_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid features provided after cleaning"
            )
        
        logger.info(f"Cleaned features: {cleaned_features}")
        
        # Run ML model prediction on features
        try:
            prediction_result = predict_from_features(cleaned_features, use_gwo)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feature prediction failed: {str(e)}"
            )
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Prepare response
        analysis_result = {
            "id": analysis_id,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "processingTime": prediction_result["processing_time"],
            "analysisDate": datetime.utcnow().isoformat() + "Z",
            "method": prediction_result.get("method", "GWO-SVM"),
            "featuresUsed": prediction_result.get("features_used", 0),
            "inputFeatures": cleaned_features,
            "useGWO": use_gwo,
            "userNotes": notes,
            "isBookmarked": False,
            "tags": ["manual-features"],
            "analysisType": "features"
        }
        
        # Save feature analysis results to database
        try:
            # Create FeatureAnalysis document
            feature_analysis_doc = FeatureAnalysis(
                userId=current_user.id,  # Use current authenticated user
                featureData=FeatureData(
                    inputFeatures=cleaned_features,
                    useGWO=use_gwo,
                    method=prediction_result.get("method", "GWO-SVM"),
                    featuresUsed=prediction_result.get("features_used", 0)
                ),
                mlResults=MLResults(
                    prediction=prediction_result["prediction"],
                    confidence=prediction_result["confidence"],
                    processingTime=prediction_result["processing_time"],
                    rawOutput=prediction_result.get("raw_score", 0.0),
                    modelVersion="gwo-svm-v1.0"
                ),
                metadata=AnalysisMetadata(
                    deviceInfo="Mobile",
                    appVersion="1.0.0",
                    analysisType="features"
                ),
                userNotes=notes,
                tags=["manual-features"]
            )
            
            # Save to database
            saved_analysis = await feature_analysis_doc.save()
            
            # Update response with saved ID
            analysis_result["id"] = str(saved_analysis.id)
            
            logger.info(f"Feature analysis saved to database with ID: {saved_analysis.id}")
            
        except Exception as e:
            logger.error(f"Failed to save feature analysis to database: {e}")
            # Continue with response even if database save fails
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature analysis failed: {str(e)}"
        )


@router.post("/compare")
async def compare_methods(
    image: Optional[UploadFile] = File(None),
    feature_data: Optional[dict] = None,
    use_gwo: bool = True,
    notes: Optional[str] = None
):
    """Compare image-based and feature-based predictions"""
    try:
        results = {}
        
        # Image-based prediction if image provided
        if image and image.filename:
            # Validate image
            if not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must be an image"
                )
            
            try:
                image_data = await image.read()
                image_prediction = predict_breast_cancer(image_data)
                results["image_prediction"] = {
                    "prediction": image_prediction["prediction"],
                    "confidence": image_prediction["confidence"],
                    "processing_time": image_prediction["processing_time"],
                    "method": "CNN-GWO"
                }
            except Exception as e:
                results["image_prediction"] = {
                    "error": f"Image prediction failed: {str(e)}"
                }
        
        # Feature-based prediction if features provided
        if feature_data and isinstance(feature_data, dict):
            try:
                # Clean feature data
                cleaned_features = {}
                for key, value in feature_data.items():
                    try:
                        cleaned_features[key] = float(value)
                    except (ValueError, TypeError):
                        continue
                
                if cleaned_features:
                    feature_prediction = predict_from_features(cleaned_features, use_gwo)
                    results["feature_prediction"] = {
                        "prediction": feature_prediction["prediction"],
                        "confidence": feature_prediction["confidence"],
                        "processing_time": feature_prediction["processing_time"],
                        "method": feature_prediction.get("method", "GWO-SVM"),
                        "features_used": feature_prediction.get("features_used", 0)
                    }
                else:
                    results["feature_prediction"] = {
                        "error": "No valid features provided"
                    }
            except Exception as e:
                results["feature_prediction"] = {
                    "error": f"Feature prediction failed: {str(e)}"
                }
        
        # Check if at least one method was used
        if not results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either image or feature data must be provided"
            )
        
        # Add comparison analysis if both methods worked
        if ("image_prediction" in results and "error" not in results["image_prediction"] and
            "feature_prediction" in results and "error" not in results["feature_prediction"]):
            
            img_pred = results["image_prediction"]
            feat_pred = results["feature_prediction"]
            
            results["comparison"] = {
                "agreement": img_pred["prediction"] == feat_pred["prediction"],
                "confidence_difference": abs(img_pred["confidence"] - feat_pred["confidence"]),
                "average_confidence": (img_pred["confidence"] + feat_pred["confidence"]) / 2
            }
        
        return {
            "success": True,
            "analysisDate": datetime.utcnow().isoformat() + "Z",
            "results": results,
            "notes": notes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison analysis failed: {str(e)}"
        )

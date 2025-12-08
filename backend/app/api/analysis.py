"""
Analysis endpoints for BreastCare AI Backend
"""

import os
import io
import uuid
import aiofiles
import base64
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPBearer
from typing import List, Optional
from PIL import Image
from beanie import PydanticObjectId
from pydantic import BaseModel, Field

from app.models.analysis import Analysis, PredictionResult, ImageInfo, MLResults, AnalysisMetadata, AnalysisType
from app.models.user import User
from app.ml.model_service import predict_breast_cancer, get_predictor
from app.utils.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()


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
            
            # 🔍 DEBUG: Print probabilities to console
            logger.info("="*60)
            logger.info("🔬 PREDICTION RESULTS:")
            logger.info(f"   Predicted Class: {prediction_result['prediction']}")
            logger.info(f"   Confidence: {prediction_result['confidence']*100:.2f}%")
            logger.info("   Probabilities for all classes:")
            if 'probabilities' in prediction_result:
                for class_name, prob in prediction_result['probabilities'].items():
                    logger.info(f"      {class_name:12s}: {prob*100:6.2f}%")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Convert image to base64 for MongoDB storage
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Keep file saving for backward compatibility (optional)
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_extension = os.path.splitext(image.filename)[1]
        saved_filename = f"{analysis_id}{file_extension}"
        file_path = os.path.join(upload_dir, saved_filename)
        
        # Save file (for local development/backup)
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(image_data)
        except Exception as e:
            logger.warning(f"Failed to save local file: {e}")
            file_path = None  # Will rely on base64 data
        
        # Prepare response
        analysis_result = {
            "id": analysis_id,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "probabilities": prediction_result.get("probabilities", {}),  # ✅ Add probabilities
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
                    filePath=file_path,  # Keep for compatibility
                    imageData=image_base64,  # ✅ Store base64 in MongoDB
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
        
        # ✅ Priority 1: Try to get image from base64 data (MongoDB)
        if analysis.imageInfo.imageData:
            try:
                # Decode base64 image data
                image_bytes = base64.b64decode(analysis.imageInfo.imageData)
                
                # Return image as Response
                return Response(
                    content=image_bytes,
                    media_type=analysis.imageInfo.mimeType,
                    headers={
                        "Content-Disposition": f"inline; filename={analysis.imageInfo.originalName}"
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to decode base64 image: {e}")
        
        # ✅ Fallback: Try to get image from file path (for backward compatibility)
        image_path = analysis.imageInfo.filePath
        if image_path and os.path.exists(image_path):
            return FileResponse(
                path=image_path,
                media_type=analysis.imageInfo.mimeType,
                filename=analysis.imageInfo.originalName
            )
        
        # No image available
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image data not found"
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
        # Calculate skip for pagination
        skip = (page - 1) * pageSize
        
        # TODO: Use current_user.id when auth is enabled
        # For now, get all analyses for testing
        query_filter = {}
        if filter_prediction:
            query_filter["mlResults.prediction"] = filter_prediction.upper()
        
        # Get total count
        total_count = await Analysis.find(query_filter).count()
        
        # Get analyses with pagination
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
                "tags": analysis.tags
            }
            formatted_analyses.append(formatted_analysis)
        
        return {
            "analyses": formatted_analyses,
            "totalCount": total_count,
            "page": page,
            "pageSize": pageSize
        }
        
    except Exception as e:
        logger.error(f"Failed to get analysis history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analysis history: {str(e)}"
        )


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

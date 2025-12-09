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
from typing import List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBearer
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from PIL import Image
from beanie import PydanticObjectId
from pydantic import BaseModel, Field
from gridfs.errors import NoFile

from app.core.database import db
from app.models.analysis import Analysis, PredictionResult, ImageInfo, MLResults, AnalysisMetadata, AnalysisType
from app.models.user import User
from app.ml.model_service import predict_breast_cancer, get_predictor
from app.utils.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()


def get_gridfs_bucket() -> AsyncIOMotorGridFSBucket:
    """Return GridFS bucket bound to current database."""
    if not db.database:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database is not initialized",
        )
    return AsyncIOMotorGridFSBucket(db.database)


@router.post("/images")
async def upload_image(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload an image to GridFS and return its id + URL."""
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )

    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size too large. Maximum 10MB allowed."
        )

    bucket = get_gridfs_bucket()
    filename = image.filename or "upload.jpg"
    try:
        file_id = await bucket.upload_from_stream(
            filename,
            image_bytes,
            metadata={
                "contentType": image.content_type,
                "userId": str(current_user.id),
                "originalName": filename,
                "uploadedAt": datetime.utcnow(),
            },
        )
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save image"
        )

    return {
        "id": str(file_id),
        "url": f"/analysis/images/{file_id}",
        "filename": filename,
        "contentType": image.content_type,
        "size": len(image_bytes),
    }


@router.get("/images/{file_id}")
async def download_image(file_id: str):
    """Download/stream image from GridFS by file id."""
    try:
        bucket = get_gridfs_bucket()
        grid_out = await bucket.open_download_stream(ObjectId(file_id))
        media_type = (grid_out.metadata or {}).get("contentType", "application/octet-stream")
        headers = {"Content-Disposition": f"inline; filename={grid_out.filename}"}
        data = await grid_out.read()
        return Response(content=data, media_type=media_type, headers=headers)
    except NoFile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve image"
        )


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

    try:
        # Read image data
        image_data = await image.read()

        # Check file size (max 10MB)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size too large. Maximum 10MB allowed."
            )
        
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

        # Determine filename and extension
        file_extension = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
        saved_filename = f"{analysis_id}{file_extension}"
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, saved_filename)

        # Save to GridFS (primary storage)
        gridfs_id = None
        try:
            bucket = get_gridfs_bucket()
            metadata = {
                "originalName": image.filename,
                "contentType": image.content_type,
                "userId": str(current_user.id),
                "analysisId": analysis_id,
                "uploadedAt": datetime.utcnow(),
            }
            gridfs_id = await bucket.upload_from_stream(
                saved_filename,
                image_data,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to save image to GridFS: {e}")

        # Save file locally for backward compatibility (best-effort)
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(image_data)
        except Exception as e:
            logger.warning(f"Failed to save local file: {e}")
            file_path = None
        
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
                "gridfsId": str(gridfs_id) if gridfs_id else None,
                "fileSize": file_size,
                "mimeType": image.content_type,
                "dimensions": {"width": width, "height": height}
            },
            "imageUrl": f"/analysis/images/{gridfs_id}" if gridfs_id else f"/analysis/image/{analysis_id}",
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
                    imageData=None,  # Use GridFS instead of base64
                    gridfsId=str(gridfs_id) if gridfs_id else None,
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

        # ✅ Priority 1: Try to stream from GridFS
        if analysis.imageInfo.gridfsId:
            try:
                bucket = get_gridfs_bucket()
                grid_out = await bucket.open_download_stream(ObjectId(analysis.imageInfo.gridfsId))
                media_type = (
                    (grid_out.metadata or {}).get("contentType")
                    or analysis.imageInfo.mimeType
                    or "application/octet-stream"
                )
                headers = {"Content-Disposition": f"inline; filename={analysis.imageInfo.originalName}"}
                data = await grid_out.read()
                return Response(content=data, media_type=media_type, headers=headers)
            except NoFile:
                logger.warning(f"GridFS file not found for analysis {analysis_id}")
            except Exception as e:
                logger.error(f"Failed to stream from GridFS: {e}")

        # ✅ Priority 2: Try to get image from base64 data (legacy)
        if analysis.imageInfo.imageData:
            try:
                image_bytes = base64.b64decode(analysis.imageInfo.imageData)
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
    request: Request,
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
            image_url = None
            if analysis.imageInfo.gridfsId:
                try:
                    image_url = str(request.url_for("download_image", file_id=analysis.imageInfo.gridfsId))
                except Exception:
                    image_url = f"/analysis/images/{analysis.imageInfo.gridfsId}"
            else:
                try:
                    image_url = str(request.url_for("get_analysis_image", analysis_id=str(analysis.id)))
                except Exception:
                    image_url = f"/analysis/image/{analysis.id}"

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
                "imageUrl": image_url,
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
    request: Request,
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
            image_url = None
            if analysis.imageInfo.gridfsId:
                try:
                    image_url = str(request.url_for("download_image", file_id=analysis.imageInfo.gridfsId))
                except Exception:
                    image_url = f"/analysis/images/{analysis.imageInfo.gridfsId}"
            else:
                try:
                    image_url = str(request.url_for("get_analysis_image", analysis_id=str(analysis.id)))
                except Exception:
                    image_url = f"/analysis/image/{analysis.id}"

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
                "imageUrl": image_url,
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

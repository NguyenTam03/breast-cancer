"""
Database configuration for BreastCare AI Backend
MongoDB connection using Beanie ODM
"""

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import logging

from app.core.config import settings
from app.models.user import User
from app.models.analysis import Analysis
from app.models.auth import RefreshToken

logger = logging.getLogger(__name__)


class Database:
    client: AsyncIOMotorClient = None
    database = None


db = Database()


async def get_database() -> AsyncIOMotorClient:
    """Get database instance"""
    return db.database


async def connect_to_mongo():
    """Create database connection"""
    logger.info("Connecting to MongoDB...")
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)
    db.database = db.client[settings.DATABASE_NAME]
    logger.info("Connected to MongoDB successfully")


async def close_mongo_connection():
    """Close database connection"""
    logger.info("Closing MongoDB connection...")
    if db.client:
        db.client.close()
    logger.info("MongoDB connection closed")


async def init_database():
    """Initialize database and collections"""
    try:
        await connect_to_mongo()
        
        # Initialize Beanie with document models
        await init_beanie(
            database=db.database,
            document_models=[
                User,
                Analysis,
                RefreshToken
            ]
        )
        
        logger.info("Database initialized successfully")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def create_indexes():
    """Create database indexes for performance"""
    try:
        # User indexes
        await User.find().create_index("email", unique=True)
        await User.find().create_index("createdAt")
        
        # Analysis indexes
        await Analysis.find().create_index("userId")
        await Analysis.find().create_index("createdAt")
        await Analysis.find().create_index([("userId", 1), ("createdAt", -1)])
        
        # RefreshToken indexes
        await RefreshToken.find().create_index("userId")
        await RefreshToken.find().create_index("expiresAt")
        await RefreshToken.find().create_index("refreshToken", unique=True)
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.warning(f"Failed to create some indexes: {e}")

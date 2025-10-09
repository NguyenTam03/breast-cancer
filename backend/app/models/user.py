"""
User model for BreastCare AI Backend
"""

from beanie import Document
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from enum import Enum


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class Language(str, Enum):
    VI = "vi"
    EN = "en"


class UserProfile(BaseModel):
    """User profile information"""
    firstName: str
    lastName: str
    dateOfBirth: Optional[datetime] = None
    phone: Optional[str] = None
    avatar: Optional[str] = None
    gender: Optional[str] = None


class UserPreferences(BaseModel):
    """User preferences"""
    theme: Theme = Theme.LIGHT
    language: Language = Language.VI
    notifications: bool = True


class User(Document):
    """User document model"""
    email: EmailStr
    password: str  # Hashed password
    profile: UserProfile
    preferences: UserPreferences = UserPreferences()
    createdAt: datetime = datetime.utcnow()
    updatedAt: datetime = datetime.utcnow()
    isActive: bool = True
    lastLogin: Optional[datetime] = None
    
    class Settings:
        name = "users"
        
    def __repr__(self) -> str:
        return f"<User {self.email}>"
    
    def full_name(self) -> str:
        """Get user's full name"""
        return f"{self.profile.firstName} {self.profile.lastName}"
    
    async def update_last_login(self):
        """Update last login timestamp"""
        self.lastLogin = datetime.utcnow()
        self.updatedAt = datetime.utcnow()
        await self.save()
    
    async def update_profile(self, profile_data: dict):
        """Update user profile"""
        for key, value in profile_data.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
        self.updatedAt = datetime.utcnow()
        await self.save()



from sqlalchemy import (
    Column, Integer, String,Boolean, Text, DateTime,ForeignKey,Numeric,ARRAY
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer,primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String,nullable=False)
    role = Column(String(20),default="mentee")
    phone = Column(String(20), nullable=True)
    location = Column(String(100), nullable=True)
    bio = Column(Text,nullable=True)
    profile_image = Column(String,nullable=True)
    linkedin_url= Column(String,nullable=True)
    twitter_url = Column(String, nullable=True)
    
    #email verification
    is_verified = Column(Boolean,default=False)
    verification_token  = Column(String(100), nullable=True)
    verification_token_expires = Column(DateTime(timezone=True), nullable=True)
    
    #password reset
    reset_token = Column(String(100), nullable=True)
    reset_token_expires = Column(DateTime(timezone=True),nullable=True)
    
    is_active = Column(Boolean,default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    #relationship
    mentee_profile = relationship("MenteeProfile", back_populates="user",uselist=False)
    mentor_profile = relationship("MentorProfile", back_populates="user", uselist=False)
    partner_profile = relationship("PartnerProfile", back_populates="user",uselist=False)
    
    def __repr__(self):
        return f"<User {self.email}>"
    
    
class MenteeProfile(Base):
    __tablename__= "mentee_profiles"
    
    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    current_level       = Column(String(20), default="beginner")   # beginner | intermediate | advanced
    areas_of_interest   = Column(ARRAY(String), default=list)
    goals               = Column(Text, nullable=True)
    learning_pace       = Column(String(20), default="medium")     # slow | medium | fast
    availability_hours  = Column(Integer, default=5)

    # Gamification stats
    total_points        = Column(Integer, default=0)
    learning_hours      = Column(Numeric(6, 2), default=0.00)
    completed_tasks     = Column(Integer, default=0)

    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="mentee_profile")


class MentorProfile(Base):
    __tablename__ = "mentor_profiles"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    current_position    = Column(String(255), nullable=True)
    company             = Column(String(255), nullable=True)
    years_of_experience = Column(Integer, default=0)
    expertise_areas     = Column(ARRAY(String), default=list)

    verification_status = Column(String(20), default="pending")   # pending | verified | rejected
    rating              = Column(Numeric(3, 2), default=0.00)
    total_mentees       = Column(Integer, default=0)
    total_sessions      = Column(Integer, default=0)
    acceptance_rate     = Column(Numeric(5, 2), default=0.00)

    availability        = Column(String(255), nullable=True)
    languages           = Column(ARRAY(String), default=list)
    timezone            = Column(String(50), nullable=True)

    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="mentor_profile")


class PartnerProfile(Base):
    __tablename__ = "partner_profiles"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    organization_name   = Column(String(255), nullable=False)
    partner_type        = Column(String(20), default="company")   # company | ngo | university | other
    industry            = Column(String(100), nullable=True)
    size                = Column(String(50), nullable=True)
    website             = Column(String, nullable=True)

    verification_status = Column(String(20), default="pending")

    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="partner_profile")

    
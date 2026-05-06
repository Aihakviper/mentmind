"""
Pydantic schemas - what goes IN and comes OUT of each endpoint
"""
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────────
#  Auth schemas
# ─────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    confirm_password: str
    role: str = "mentee"                        # mentee | mentor | partner
    phone: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    industry: Optional[str] = None
    areas_of_interest: Optional[List[str]] = []  # for mentee
    expertise_areas: Optional[List[str]] = []    # for mentor
    current_skills: Optional[List[str]] = []      # currently captured for matching/onboarding
    organization_name: Optional[str] = None      # for partner


    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ["mentee", "mentor", "partner"]:
            raise ValueError("Role must be mentee, mentor, or partner")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Full name must be at least 2 characters")
        return v.strip()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class VerifyEmailRequest(BaseModel):
    token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    password: str
    confirm_password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str
    confirm_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# ─────────────────────────────────────────────
#  Response schemas
# ─────────────────────────────────────────────

class MenteeProfileResponse(BaseModel):
    current_level: str
    areas_of_interest: List[str]
    goals: Optional[str]
    total_points: int
    learning_hours: float
    completed_tasks: int

    class Config:
        from_attributes = True


class MentorProfileResponse(BaseModel):
    current_position: Optional[str]
    company: Optional[str]
    years_of_experience: int
    expertise_areas: List[str]
    verification_status: str
    rating: float
    total_mentees: int
    total_sessions: int

    class Config:
        from_attributes = True


class PartnerProfileResponse(BaseModel):
    organization_name: str
    partner_type: str
    industry: Optional[str]
    verification_status: str

    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    phone: Optional[str]
    location: Optional[str]
    bio: Optional[str]
    profile_image: Optional[str]
    linkedin_url: Optional[str]
    twitter_url: Optional[str]
    is_verified: bool
    is_active: bool
    created_at: datetime
    mentee_profile: Optional[MenteeProfileResponse] = None
    mentor_profile: Optional[MentorProfileResponse] = None
    partner_profile: Optional[PartnerProfileResponse] = None

    class Config:
        from_attributes = True


class TokensResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    message: str
    tokens: TokensResponse
    user: UserResponse
    is_verified: bool


class RegisterResponse(BaseModel):
    message: str
    user: UserResponse


class MessageResponse(BaseModel):
    message: str


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None

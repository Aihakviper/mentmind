"""
Authentication routes
POST /auth/register
POST /auth/login
POST /auth/verify-email
POST /auth/resend-verification
POST /auth/forgot-password
POST /auth/reset-password
POST /auth/refresh
POST /auth/change-password
POST /auth/logout
GET  /auth/me
PUT  /auth/me
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone

from database import get_db
from models import User, MenteeProfile, MentorProfile, PartnerProfile
from schemas import (
    RegisterRequest, RegisterResponse,
    LoginRequest, LoginResponse,
    VerifyEmailRequest, MessageResponse,
    ForgotPasswordRequest, ResetPasswordRequest,
    ChangePasswordRequest, RefreshTokenRequest,
    UserResponse, UpdateProfileRequest, TokensResponse,
)
from security import (
    hash_password, verify_password,
    create_token_pair, create_access_token,
    decode_token, generate_secure_token,
)
from email_utils import send_verification_email, send_password_reset_email
from dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ─────────────────────────────────────────────────────────────────────────────
#  REGISTER
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user (mentee, mentor, or partner).
    Sends a verification email automatically.
    """
    # 1. Check email is not already taken
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists",
        )

    # 2. Create the User row
    verification_token   = generate_secure_token()
    token_expiry         = datetime.now(timezone.utc) + timedelta(hours=24)

    user = User(
        email                       = payload.email,
        full_name                   = payload.full_name,
        hashed_password             = hash_password(payload.password),
        role                        = payload.role,
        phone                       = payload.phone,
        location                    = payload.location,
        bio                         = payload.bio,
        is_verified                 = False,
        verification_token          = verification_token,
        verification_token_expires  = token_expiry,
    )
    db.add(user)
    db.flush()   # get user.id without committing

    # 3. Create the role-specific profile
    if payload.role == "mentee":
        profile = MenteeProfile(
            user_id           = user.id,
            areas_of_interest = payload.areas_of_interest or [],
        )
        db.add(profile)

    elif payload.role == "mentor":
        profile = MentorProfile(
            user_id         = user.id,
            expertise_areas = payload.expertise_areas or [],
        )
        db.add(profile)

    elif payload.role == "partner":
        if not payload.organization_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization name is required for partners",
            )
        profile = PartnerProfile(
            user_id           = user.id,
            organization_name = payload.organization_name,
        )
        db.add(profile)

    db.commit()
    db.refresh(user)

    # 4. Send verification email
    send_verification_email(user.email, user.full_name, verification_token)

    return RegisterResponse(
        message="Registration successful! Please check your email to verify your account.",
        user=UserResponse.model_validate(user),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  LOGIN
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """
    Login with email and password.
    Returns access + refresh JWT tokens.
    """
    # 1. Find user
    user = db.query(User).filter(User.email == payload.email).first()

    # 2. Verify credentials (same error for both cases to prevent user enumeration)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # 3. Check account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated. Please contact support.",
        )

    # 4. Create tokens
    tokens = create_token_pair(user.id, user.role)

    return LoginResponse(
        message     = "Login successful",
        tokens      = TokensResponse(**tokens),
        user        = UserResponse.model_validate(user),
        is_verified = user.is_verified,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  VERIFY EMAIL
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/verify-email", response_model=MessageResponse)
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    """Verify email address using the token sent to the user's inbox"""
    user = db.query(User).filter(User.verification_token == payload.token).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token",
        )

    if user.is_verified:
        return MessageResponse(message="Email is already verified. You can login.")

    # Check token expiry
    if user.verification_token_expires and \
       datetime.now(timezone.utc) > user.verification_token_expires:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired. Please request a new one.",
        )

    # Mark verified and clear token
    user.is_verified               = True
    user.verification_token        = None
    user.verification_token_expires = None
    db.commit()

    return MessageResponse(message="Email verified successfully! You can now login.")


# ─────────────────────────────────────────────────────────────────────────────
#  RESEND VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/resend-verification", response_model=MessageResponse)
def resend_verification(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Resend the verification email"""
    user = db.query(User).filter(User.email == payload.email).first()

    # Always return the same message (don't leak whether email exists)
    generic_msg = MessageResponse(
        message="If this email exists and is unverified, we've sent a new verification link."
    )

    if not user or user.is_verified:
        return generic_msg

    token   = generate_secure_token()
    expiry  = datetime.now(timezone.utc) + timedelta(hours=24)

    user.verification_token         = token
    user.verification_token_expires = expiry
    db.commit()

    send_verification_email(user.email, user.full_name, token)
    return generic_msg


# ─────────────────────────────────────────────────────────────────────────────
#  FORGOT PASSWORD
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Request a password reset link"""
    user = db.query(User).filter(User.email == payload.email).first()

    generic_msg = MessageResponse(
        message="If this email exists, a password reset link has been sent."
    )

    if not user or not user.is_active:
        return generic_msg

    token  = generate_secure_token()
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)

    user.reset_token         = token
    user.reset_token_expires = expiry
    db.commit()

    send_password_reset_email(user.email, user.full_name, token)
    return generic_msg


# ─────────────────────────────────────────────────────────────────────────────
#  RESET PASSWORD
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/reset-password", response_model=MessageResponse)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Set a new password using the reset token"""
    user = db.query(User).filter(User.reset_token == payload.token).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    if user.reset_token_expires and \
       datetime.now(timezone.utc) > user.reset_token_expires:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired. Please request a new one.",
        )

    user.hashed_password    = hash_password(payload.password)
    user.reset_token        = None
    user.reset_token_expires = None
    db.commit()

    return MessageResponse(message="Password reset successful! You can now login.")


# ─────────────────────────────────────────────────────────────────────────────
#  REFRESH TOKEN
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokensResponse)
def refresh_token(payload: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Exchange a refresh token for a new access token"""
    token_data = decode_token(payload.refresh_token)

    if not token_data or token_data.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user_id = token_data.get("sub")
    user    = db.query(User).filter(User.id == int(user_id), User.is_active == True).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    tokens = create_token_pair(user.id, user.role)
    return TokensResponse(**tokens)


# ─────────────────────────────────────────────────────────────────────────────
#  CHANGE PASSWORD  (authenticated)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/change-password", response_model=MessageResponse)
def change_password(
    payload:      ChangePasswordRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Change password for the logged-in user"""
    if not verify_password(payload.old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    current_user.hashed_password = hash_password(payload.new_password)
    db.commit()

    return MessageResponse(message="Password changed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
#  GET / UPDATE CURRENT USER  (authenticated)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile"""
    return UserResponse.model_validate(current_user)


@router.put("/me", response_model=UserResponse)
def update_me(
    payload:      UpdateProfileRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Update the currently authenticated user's basic info"""
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(current_user, field, value)
    db.commit()
    db.refresh(current_user)
    return UserResponse.model_validate(current_user)


# ─────────────────────────────────────────────────────────────────────────────
#  LOGOUT  (client-side - just a confirmation endpoint)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/logout", response_model=MessageResponse)
def logout(current_user: User = Depends(get_current_user)):
    """
    Logout endpoint.
    Since JWTs are stateless, the client simply discards the tokens.
    If you need server-side blacklisting, add a token blocklist table later.
    """
    return MessageResponse(message="Logged out successfully.")
0

"""
Mentor Discovery routes
GET /mentors                - browse & search with filters
GET /mentors/{mentor_id}    - single mentor detail
GET /mentors/recommended    - top AI matches for current mentee
"""
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from database import get_db
from models import User, MentorProfile, MenteeProfile
from dependencies import get_current_user

router = APIRouter(prefix="/mentors", tags=["Mentor Discovery"])


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class MentorCard(BaseModel):
    id: int
    name: str
    position: Optional[str]
    company: Optional[str]
    bio: Optional[str]
    expertise_areas: list[str]
    years_of_experience: int
    rating: float
    total_mentees: int
    total_sessions: int
    availability: Optional[str]
    match_score: int
    avatar_url: Optional[str]
    linkedin_url: Optional[str]
    verification_status: str


class MentorListResponse(BaseModel):
    mentors: list[MentorCard]
    total: int


class RecommendedResponse(BaseModel):
    recommendations: list[MentorCard]
    total: int
    based_on: dict


# ─────────────────────────────────────────────────────────────────────────────
#  Helper - match score
# ─────────────────────────────────────────────────────────────────────────────

def calc_score(mentee: Optional[MenteeProfile], mentor: MentorProfile) -> int:
    score = 70
    if mentee:
        overlap = set(mentee.areas_of_interest or []) & set(mentor.expertise_areas or [])
        score += len(overlap) * 5
        if mentor.rating and float(mentor.rating) >= 4.5:
            score += 3
        if mentor.years_of_experience:
            level_bonus = {
                "beginner":     mentor.years_of_experience >= 5,
                "intermediate": mentor.years_of_experience >= 8,
                "advanced":     mentor.years_of_experience >= 10,
            }
            if level_bonus.get(mentee.current_level, False):
                score += 2
    return min(score, 98)


def mentor_to_card(mentor: MentorProfile, score: int) -> MentorCard:
    return MentorCard(
        id                  = mentor.id,
        name                = mentor.user.full_name,
        position            = mentor.current_position,
        company             = mentor.company,
        bio                 = mentor.user.bio,
        expertise_areas     = mentor.expertise_areas or [],
        years_of_experience = mentor.years_of_experience or 0,
        rating              = float(mentor.rating or 0),
        total_mentees       = mentor.total_mentees or 0,
        total_sessions      = mentor.total_sessions or 0,
        availability        = mentor.availability,
        match_score         = score,
        avatar_url          = mentor.user.profile_image,
        linkedin_url        = mentor.user.linkedin_url,
        verification_status = mentor.verification_status,
    )


def get_mentee_profile(user: User, db: Session) -> Optional[MenteeProfile]:
    if user.role == "mentee":
        return db.query(MenteeProfile).filter(MenteeProfile.user_id == user.id).first()
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  GET /mentors   — browse, search, filter, sort
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=MentorListResponse)
def list_mentors(
    # Search & filters
    search:     Optional[str] = Query(None, description="Search by name, company or expertise"),
    category:   Optional[str] = Query(None, description="Comma-separated expertise areas e.g. Design,Engineering"),
    experience: Optional[str] = Query(None, description="junior | mid | senior"),
    min_rating: Optional[float] = Query(None, ge=0, le=5),
    min_match:  Optional[int]   = Query(None, ge=0, le=100),
    # Sort
    sort_by: str = Query("match", description="match | rating | experience | name"),
    # Pagination
    limit:  int = Query(20, ge=1, le=100),
    offset: int = Query(0,  ge=0),
    # Auth + DB
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Browse all verified mentors with optional search, filtering and sorting."""

    # Base query — verified mentors only
    q = db.query(MentorProfile).filter(
        MentorProfile.verification_status == "verified"
    )

    # ── Search ────────────────────────────────────────────────────────────────
    if search:
        term = f"%{search.lower()}%"
        q = q.join(MentorProfile.user).filter(
            User.full_name.ilike(term)
            | MentorProfile.current_position.ilike(term)
            | MentorProfile.company.ilike(term)
        )

    # ── Experience filter ─────────────────────────────────────────────────────
    if experience:
        ranges = {
            "junior": (1, 3),
            "mid":    (4, 7),
            "senior": (8, 99),
        }
        lo, hi = ranges.get(experience, (0, 99))
        q = q.filter(
            MentorProfile.years_of_experience >= lo,
            MentorProfile.years_of_experience <= hi,
        )

    # ── Rating filter ─────────────────────────────────────────────────────────
    if min_rating:
        q = q.filter(MentorProfile.rating >= min_rating)

    mentors = q.all()

    # ── Category filter (array field — easier in Python than SQL) ─────────────
    if category:
        cats = [c.strip().lower() for c in category.split(",")]
        mentors = [
            m for m in mentors
            if any(
                cat in (exp.lower() for exp in (m.expertise_areas or []))
                for cat in cats
            )
        ]

    # ── Compute match scores ──────────────────────────────────────────────────
    mentee_profile = get_mentee_profile(current_user, db)
    cards = [mentor_to_card(m, calc_score(mentee_profile, m)) for m in mentors]

    # ── Match score filter ────────────────────────────────────────────────────
    if min_match:
        cards = [c for c in cards if c.match_score >= min_match]

    # ── Sort ──────────────────────────────────────────────────────────────────
    sort_key = {
        "match":      lambda c: c.match_score,
        "rating":     lambda c: c.rating,
        "experience": lambda c: c.years_of_experience,
        "name":       lambda c: c.name.lower(),
    }.get(sort_by, lambda c: c.match_score)

    cards.sort(key=sort_key, reverse=(sort_by != "name"))

    total = len(cards)
    return MentorListResponse(mentors=cards[offset: offset + limit], total=total)


# ─────────────────────────────────────────────────────────────────────────────
#  GET /mentors/recommended  — must be BEFORE /{mentor_id} to avoid collision
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/recommended", response_model=RecommendedResponse)
def recommended_mentors(
    limit:        int     = Query(5, ge=1, le=20),
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Return top AI-matched mentors for the logged-in mentee."""
    mentee_profile = get_mentee_profile(current_user, db)

    mentors = db.query(MentorProfile).filter(
        MentorProfile.verification_status == "verified"
    ).all()

    cards = sorted(
        [mentor_to_card(m, calc_score(mentee_profile, m)) for m in mentors],
        key=lambda c: c.match_score,
        reverse=True,
    )

    return RecommendedResponse(
        recommendations = cards[:limit],
        total           = len(cards),
        based_on        = {
            "interests": mentee_profile.areas_of_interest if mentee_profile else [],
            "level":     mentee_profile.current_level if mentee_profile else "beginner",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /mentors/{mentor_id}  — single mentor detail
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{mentor_id}", response_model=MentorCard)
def get_mentor(
    mentor_id:    int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Get full details for a single mentor."""
    mentor = db.query(MentorProfile).filter(
        MentorProfile.id == mentor_id,
        MentorProfile.verification_status == "verified",
    ).first()

    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    mentee_profile = get_mentee_profile(current_user, db)
    return mentor_to_card(mentor, calc_score(mentee_profile, mentor))

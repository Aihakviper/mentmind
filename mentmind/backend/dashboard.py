"""
Dashboard routes
GET  /dashboard/mentee   - Full mentee dashboard data
GET  /dashboard/mentor   - Full mentor dashboard data
POST /dashboard/update-stats   - Add learning hours / tasks / points
GET  /dashboard/badges   - Earned badges
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

from database import get_db
from models import User, MenteeProfile, MentorProfile
from schemas import UserResponse
from dependencies import get_current_user
from ai_matching import get_match_score, recommend_for_mentee


router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


# ─────────────────────────────────────────────────────────────────────────────
#  Inline response schemas (simple, no separate file needed)
# ─────────────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    learning_hours: float
    completed_tasks: int
    total_points: int
    hours_change: str
    tasks_change: str
    points_change: str


class ProgressItem(BaseModel):
    id: int
    name: str
    icon: str
    progress: int
    status: str   # on_track | behind | completed


class WeeklyEngagement(BaseModel):
    labels: list[str]
    data: list[int]


class MentorMatch(BaseModel):
    id: int
    name: str
    position: Optional[str]
    company: Optional[str]
    match_score: float
    rating: float
    expertise_areas: list[str]
    avatar_url: Optional[str]


class Badge(BaseModel):
    id: int
    name: str
    icon: str
    color: str
    description: str


class MenteeDashboardResponse(BaseModel):
    user: UserResponse
    stats: StatsResponse
    learning_progress: list[ProgressItem]
    weekly_engagement: WeeklyEngagement
    mentor_matches: list[MentorMatch]
    badges: list[Badge]
    welcome_message: str
    progress_message: str


class MentorDashboardResponse(BaseModel):
    user: UserResponse
    stats: dict
    weekly_engagement: WeeklyEngagement
    welcome_message: str


class UpdateStatsRequest(BaseModel):
    learning_hours: Optional[float] = 0
    completed_tasks: Optional[int] = 0
    points: Optional[int] = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_match_score(mentee: MenteeProfile, mentor: MentorProfile) -> int:
    """
    Simple match scoring:
    - Base 70 points
    - +5 per overlapping interest / expertise area
    - +3 if mentor has rating >= 4.5
    - Cap at 98
    """
    score = 70

    mentee_interests = set(mentee.areas_of_interest or [])
    mentor_expertise = set(mentor.expertise_areas or [])
    overlap = mentee_interests & mentor_expertise
    score += len(overlap) * 5

    if mentor.rating and float(mentor.rating) >= 4.5:
        score += 3

    return min(score, 98)

def get_top_matches(db, mentee_profile, top_k=3):
    mentors = db.query(MentorProfile).filter(
        MentorProfile.verification_status == "verified"
    ).all()

    results = []

    for m in mentors:
        results.append(MentorMatch(
            id=m.id,
            name=m.user.full_name,
            position=m.current_position,
            company=m.company,
            match_score=get_match_score(m, mentee_profile),
            rating=float(m.rating or 0),
            expertise_areas=m.expertise_areas or [],
            avatar_url=m.user.profile_image,
        ))

    results.sort(key=lambda x: x.match_score, reverse=True)
    return results[:top_k]



def get_weekly_engagement_data(db: Session, user_id: int) -> WeeklyEngagement:
    """
    Returns last 7 days labels + engagement data based on recent activities.
    Engagement = number of lesson progress updates in the last 7 days.
    """
    from datetime import timedelta
    from sqlalchemy import func
    from classrooms import LessonProgress  # import here to avoid circular

    today = datetime.now(timezone.utc).date()
    labels = []
    data = []

    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        labels.append(day.strftime("%a"))

        # Count progress updates on this day
        start_of_day = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        end_of_day = start_of_day + timedelta(days=1)

        count = db.query(func.count(LessonProgress.id)).filter(
            LessonProgress.user_id == user_id,
            LessonProgress.updated_at >= start_of_day,
            LessonProgress.updated_at < end_of_day
        ).scalar()

        data.append(int(count))

    return WeeklyEngagement(labels=labels, data=data)


def get_default_badges() -> list[Badge]:
    """
    Hardcoded badge definitions.
    Move to a DB table when you build the full gamification system.
    """
    return [
        Badge(id=1, name="Fast Learner",  icon="🎯", color="gold",
              description="Complete 10 hours of learning in a week"),
        Badge(id=2, name="Deep Thinker",  icon="🧠", color="blue",
              description="Complete 5 advanced courses"),
        Badge(id=3, name="Team Player",   icon="👥", color="green",
              description="Participate in 10 group discussions"),
        Badge(id=4, name="Bookworm",      icon="📚", color="pink",
              description="Read 20 articles"),
    ]


def get_default_progress() -> list[ProgressItem]:
    """
    Placeholder learning progress.
    Replace with real Enrollment + Course tables later.
    """
    return [
        ProgressItem(id=1, name="Advanced Prototyping",
                     icon="🎨", progress=82, status="on_track"),
        ProgressItem(id=2, name="React for Designers",
                     icon="⚛️",  progress=45, status="behind"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  MENTEE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/mentee", response_model=MenteeDashboardResponse)
def mentee_dashboard(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """
    Return everything the mentee dashboard needs in one call:
    user info, stats, learning progress, weekly chart,
    top 3 AI mentor matches, and earned badges.
    """
    if current_user.role.lower() != "mentee":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This endpoint is only for mentees. Your role is: '{current_user.role}'",
        )

    # ── Mentee profile ───────────────────────────────────────────────────────
    profile = db.query(MenteeProfile).filter(
        MenteeProfile.user_id == current_user.id
    ).first()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mentee profile not found",
        )

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = StatsResponse(
        learning_hours  = float(profile.learning_hours or 0),
        completed_tasks = profile.completed_tasks or 0,
        total_points    = profile.total_points or 0,
        hours_change    = "+12%",   # TODO: compare with last week
        tasks_change    = "+5",
        points_change   = "+240",
    )

    # # ── AI Mentor matches ────────────────────────────────────────────────────
    # mentors = db.query(MentorProfile).filter(
    #     MentorProfile.verification_status == "verified"
    # ).all()

    # scored = sorted(
    #     [
    #         MentorMatch(
    #             id              = m.id,
    #             name            = m.user.full_name,
    #             position        = m.current_position,
    #             company         = m.company,
    #             match_score     = get_match_score(m, profile),
    #             rating          = float(m.rating or 0),
    #             expertise_areas = m.expertise_areas or [],
    #             avatar_url      = m.user.profile_image,
    #         )
    #         for m in mentors
    #     ],
    #     key=lambda x: x.match_score,
    #     reverse=True,
    # )
    # top_matches = scored[:3]
    top_matches = get_top_matches(db, profile)

    # ── Build first name for welcome ─────────────────────────────────────────
    first_name = current_user.full_name.split()[0]

    return MenteeDashboardResponse(
        user               = UserResponse.model_validate(current_user),
        stats              = stats,
        learning_progress  = get_default_progress(),
        weekly_engagement  = get_weekly_engagement_data(db, current_user.id),
        mentor_matches     = top_matches,
        badges             = get_default_badges(),
        welcome_message    = f"Welcome back, {first_name}! 👋",
        progress_message   = "You're making great progress on your learning journey!",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MENTOR DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/mentor", response_model=MentorDashboardResponse)
def mentor_dashboard(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Return everything the mentor dashboard needs."""
    if current_user.role.lower() != "mentor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This endpoint is only for mentors. Your role is: '{current_user.role}'",
        )

    profile = db.query(MentorProfile).filter(
        MentorProfile.user_id == current_user.id
    ).first()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mentor profile not found",
        )

    first_name = current_user.full_name.split()[0]

    stats = {
        "total_mentees":      profile.total_mentees or 0,
        "total_sessions":     profile.total_sessions or 0,
        "rating":             float(profile.rating or 0),
        "acceptance_rate":    float(profile.acceptance_rate or 0),
        "verification_status": profile.verification_status,
    }

    return MentorDashboardResponse(
        user              = UserResponse.model_validate(current_user),
        stats             = stats,
        weekly_engagement = get_weekly_engagement_data(db, current_user.id),
        welcome_message   = f"Welcome back, {first_name}! 👋",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  UPDATE STATS  (mentee only)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/update-stats")
def update_stats(
    payload:      UpdateStatsRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Add learning hours, tasks completed, and/or points to the mentee profile."""
    if current_user.role.lower() != "mentee":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Only mentees can update stats. Your role is: '{current_user.role}'",
        )

    profile = db.query(MenteeProfile).filter(
        MenteeProfile.user_id == current_user.id
    ).first()

    if not profile:
        raise HTTPException(status_code=404, detail="Mentee profile not found")

    if payload.learning_hours:
        profile.learning_hours  = float(profile.learning_hours or 0) + payload.learning_hours
    if payload.completed_tasks:
        profile.completed_tasks = (profile.completed_tasks or 0) + payload.completed_tasks
    if payload.points:
        profile.total_points    = (profile.total_points or 0) + payload.points

    db.commit()
    db.refresh(profile)

    return {
        "message": "Stats updated successfully",
        "stats": {
            "learning_hours":  float(profile.learning_hours),
            "completed_tasks": profile.completed_tasks,
            "total_points":    profile.total_points,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  BADGES
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/badges")
def get_badges(current_user: User = Depends(get_current_user)):
    """Return the user's earned badges."""
    return {
        "badges":       get_default_badges(),
        "total_badges": len(get_default_badges()),
    }

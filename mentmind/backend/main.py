"""
MentMinds FastAPI Application
Run with: uvicorn main:app --reload
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import engine, Base, SessionLocal
from models import User, MentorProfile, MenteeProfile
from dependencies import get_current_user
from security import hash_password
import auth, dashboard, mentors, classrooms, forum, opportunities, resources, ai_matching, ai_mentor


def seed_sample_profiles():
    from datetime import datetime, timedelta, timezone
    from classrooms import Module, LessonProgress, seed_courses_if_empty

    db = SessionLocal()
    try:
        mentor_count = db.query(MentorProfile).count()
        mentee_count = db.query(MenteeProfile).count()

        if mentor_count == 0:
            mentors = [
                {
                    "email": "david.okoro@mentminds.test",
                    "full_name": "David Okoro",
                    "password": "MentMind$123",
                    "role": "mentor",
                    "current_position": "Lead Product Designer",
                    "company": "NairaSoft",
                    "years_of_experience": 8,
                    "expertise_areas": ["Design", "UX", "Product"],
                    "rating": 4.7,
                    "total_mentees": 24,
                    "total_sessions": 56,
                    "acceptance_rate": 92.0,
                    "verification_status": "verified",
                    "availability": "Weeknights",
                    "languages": ["English"],
                },
                {
                    "email": "ngozi.eze@mentminds.test",
                    "full_name": "Ngozi Eze",
                    "password": "MentMind$123",
                    "role": "mentor",
                    "current_position": "Senior Data Engineer",
                    "company": "Paystack",
                    "years_of_experience": 10,
                    "expertise_areas": ["Engineering", "Python", "Data"],
                    "rating": 4.5,
                    "total_mentees": 18,
                    "total_sessions": 48,
                    "acceptance_rate": 88.0,
                    "verification_status": "verified",
                    "availability": "Flexible",
                    "languages": ["English"],
                },
                {
                    "email": "kemi.ade@mentminds.test",
                    "full_name": "Kemi Ade",
                    "password": "MentMind$123",
                    "role": "mentor",
                    "current_position": "Growth Marketing Lead",
                    "company": "Flutterwave",
                    "years_of_experience": 7,
                    "expertise_areas": ["Marketing", "Growth", "Strategy"],
                    "rating": 4.2,
                    "total_mentees": 14,
                    "total_sessions": 34,
                    "acceptance_rate": 85.0,
                    "verification_status": "verified",
                    "availability": "Weekends",
                    "languages": ["English"],
                },
            ]

            for mentor in mentors:
                user = User(
                    email=mentor["email"],
                    full_name=mentor["full_name"],
                    hashed_password=hash_password(mentor["password"]),
                    role=mentor["role"],
                    is_verified=True,
                    is_active=True,
                )
                db.add(user)
                db.flush()
                db.add(MentorProfile(
                    user_id=user.id,
                    current_position=mentor["current_position"],
                    company=mentor["company"],
                    years_of_experience=mentor["years_of_experience"],
                    expertise_areas=mentor["expertise_areas"],
                    rating=mentor["rating"],
                    total_mentees=mentor["total_mentees"],
                    total_sessions=mentor["total_sessions"],
                    acceptance_rate=mentor["acceptance_rate"],
                    verification_status=mentor["verification_status"],
                    availability=mentor["availability"],
                    languages=mentor["languages"],
                ))

        if mentee_count == 0:
            mentee_user = User(
                email="amina.yusuf@mentminds.test",
                full_name="Amina Yusuf",
                hashed_password=hash_password("MentMind$123"),
                role="mentee",
                is_verified=True,
                is_active=True,
            )
            db.add(mentee_user)
            db.flush()
            db.add(MenteeProfile(
                user_id=mentee_user.id,
                current_level="intermediate",
                areas_of_interest=["Design", "Product", "UX"],
                availability_hours=8,
                total_points=180,
                learning_hours=12.5,
                completed_tasks=9,
            ))

            # Seed sample classroom content and lesson progress for engagement tracking
            seed_courses_if_empty(db)
            modules = db.query(Module).limit(3).all()
            for i, module in enumerate(modules):
                db.add(LessonProgress(
                    user_id=mentee_user.id,
                    module_id=module.id,
                    completed=True,
                    watch_time=600 + i * 120,
                    updated_at=datetime.now(timezone.utc) - timedelta(days=i),
                ))

        if mentor_count == 0 or mentee_count == 0:
            db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(" Starting MentMinds API...")
    from classrooms    import Course, Module, Enrollment, LessonProgress, Assignment, Quiz
    from forum         import ForumTopic, ForumReply, ForumLike
    from opportunities import Opportunity, Application
    from resources     import Resource
    from ai_matching   import  MLMatchResult
    from ai_mentor     import ChatMessage
    Base.metadata.create_all(bind=engine)
    seed_sample_profiles()
    print(" Database tables ready")
    yield
    print(" Shutting down...")


app = FastAPI(
    title       = "MentMinds API",
    description = "AI-powered mentorship platform",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["https://mentmind.vercel.app",
                         "https://mentmind-44lv33f21-aihakvipers-projects.vercel.app/" ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(auth.router,           prefix="/api")
app.include_router(dashboard.router,      prefix="/api")
app.include_router(mentors.router,        prefix="/api")
app.include_router(classrooms.router,     prefix="/api")
app.include_router(forum.router,          prefix="/api")
app.include_router(opportunities.router,  prefix="/api")
app.include_router(resources.router,      prefix="/api")
app.include_router(ai_matching.router,     prefix="/api")
app.include_router(ai_mentor.router,      prefix="/api")


@app.get("/", tags=["Health"])
def root():
    return {
        "status":    "online",
        "message":   "MentMinds API ",
        "docs":      "/docs",
        "endpoints": {
            "auth":          "/api/auth",
            "dashboard":     "/api/dashboard",
            "mentors":       "/api/mentors",
            "classrooms":    "/api/classrooms",
            "forum":         "/api/forum",
            "opportunities": "/api/opportunities",
            "resources":     "/api/resources",
            "ai_mentor":     "/api/ai-mentor",
        }
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

@app.get("/api/debug/me", tags=["Debug"])
def debug_me(current_user: User = Depends(get_current_user)):
    return {
        "id":          current_user.id,
        "email":       current_user.email,
        "full_name":   current_user.full_name,
        "role":        current_user.role,
        "is_verified": current_user.is_verified,
        "is_active":   current_user.is_active,
    }

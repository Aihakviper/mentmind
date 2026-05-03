"""
Learning Classrooms routes
GET  /classrooms                  - list all courses
GET  /classrooms/{id}             - course detail + modules
POST /classrooms/{id}/enroll      - enroll in a course
POST /classrooms/{id}/progress    - update lesson progress
GET  /classrooms/my               - my enrolled courses
POST /classrooms                  - create a course (mentors/admins)
GET  /classrooms/assignments      - list assignments
POST /classrooms/assignments      - create assignment (mentors/admins)
GET  /classrooms/quizzes          - list quizzes
POST /classrooms/quizzes          - create quiz (mentors/admins)
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime, Numeric, JSON
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone

from database import Base, get_db
from models import User
from dependencies import get_current_user

router = APIRouter(prefix="/classrooms", tags=["Learning Classrooms"])


# ─────────────────────────────────────────────────────────────────────────────
#  DB Models (defined here to keep things simple)
# ─────────────────────────────────────────────────────────────────────────────

class Course(Base):
    __tablename__ = "courses"
    id              = Column(Integer, primary_key=True, index=True)
    title           = Column(String(255), nullable=False)
    description     = Column(Text)
    category        = Column(String(100))           # Design | Engineering | Marketing …
    level           = Column(String(20))            # beginner | intermediate | advanced
    thumbnail       = Column(String)
    duration_hours  = Column(Numeric(4, 1), default=0)
    total_modules   = Column(Integer, default=0)
    instructor_name = Column(String(255))
    instructor_bio  = Column(Text)
    is_published    = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    modules     = relationship("Module", back_populates="course", order_by="Module.order")
    enrollments = relationship("Enrollment", back_populates="course")


class Module(Base):
    __tablename__ = "modules"
    id          = Column(Integer, primary_key=True, index=True)
    course_id   = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title       = Column(String(255), nullable=False)
    description = Column(Text)
    video_url   = Column(String)
    duration_minutes = Column(Integer, default=0)
    order       = Column(Integer, default=0)

    course      = relationship("Course", back_populates="modules")
    progress    = relationship("LessonProgress", back_populates="module")


class Enrollment(Base):
    __tablename__ = "enrollments"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id   = Column(Integer, ForeignKey("courses.id"), nullable=False)
    progress    = Column(Integer, default=0)        # 0-100 %
    enrolled_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    course      = relationship("Course", back_populates="enrollments")
    user        = relationship("User")


class LessonProgress(Base):
    __tablename__ = "lesson_progress"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    module_id   = Column(Integer, ForeignKey("modules.id"), nullable=False)
    completed   = Column(Boolean, default=False)
    watch_time  = Column(Integer, default=0)        # seconds watched
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    module      = relationship("Module", back_populates="progress")


class Assignment(Base):
    __tablename__ = "assignments"

    id          = Column(Integer, primary_key=True, index=True)
    created_by  = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id   = Column(Integer, ForeignKey("courses.id"), nullable=True)
    title       = Column(String(255), nullable=False)
    instructions = Column(Text, nullable=True)
    due_at      = Column(DateTime(timezone=True), nullable=True)
    points      = Column(Integer, default=100)
    is_published = Column(Boolean, default=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    creator = relationship("User")
    course  = relationship("Course")


class Quiz(Base):
    __tablename__ = "quizzes"

    id          = Column(Integer, primary_key=True, index=True)
    created_by  = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id   = Column(Integer, ForeignKey("courses.id"), nullable=True)
    title       = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    questions   = Column(JSON, default=list)
    time_limit_minutes = Column(Integer, default=10)
    is_published = Column(Boolean, default=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    creator = relationship("User")
    course  = relationship("Course")


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class ModuleOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    video_url: Optional[str]
    duration_minutes: int
    order: int
    completed: bool = False

    class Config:
        from_attributes = True


class CourseCard(BaseModel):
    id: int
    title: str
    description: Optional[str]
    category: Optional[str]
    level: Optional[str]
    thumbnail: Optional[str]
    duration_hours: float
    total_modules: int
    instructor_name: Optional[str]
    is_enrolled: bool = False
    progress: int = 0

    class Config:
        from_attributes = True


class CourseDetail(CourseCard):
    instructor_bio: Optional[str]
    modules: list[ModuleOut] = []


class EnrollResponse(BaseModel):
    message: str
    course_id: int
    enrolled_at: datetime


class ProgressUpdate(BaseModel):
    module_id: int
    completed: bool
    watch_time: Optional[int] = 0


class ProgressResponse(BaseModel):
    message: str
    course_progress: int


class ModuleCreate(BaseModel):
    title: str
    description: Optional[str] = None
    video_url: Optional[str] = None
    duration_minutes: int = 0


class CourseCreate(BaseModel):
    title: str
    description: Optional[str] = None
    category: Optional[str] = "General"
    level: str = "beginner"
    thumbnail: Optional[str] = None
    duration_hours: Optional[float] = None
    instructor_bio: Optional[str] = None
    is_published: bool = True
    modules: list[ModuleCreate] = Field(default_factory=list)


class AssignmentCreate(BaseModel):
    title: str
    instructions: Optional[str] = None
    course_id: Optional[int] = None
    due_at: Optional[datetime] = None
    points: int = 100
    is_published: bool = True


class AssignmentOut(BaseModel):
    id: int
    title: str
    instructions: Optional[str]
    course_id: Optional[int]
    due_at: Optional[datetime]
    points: int
    is_published: bool
    created_at: datetime

    class Config:
        from_attributes = True


class QuizCreate(BaseModel):
    title: str
    description: Optional[str] = None
    course_id: Optional[int] = None
    questions: list[dict] = Field(default_factory=list)
    time_limit_minutes: int = 10
    is_published: bool = True


class QuizOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    course_id: Optional[int]
    questions: list[dict]
    time_limit_minutes: int
    is_published: bool
    created_at: datetime

    class Config:
        from_attributes = True


def require_mentor_or_admin(current_user: User):
    if current_user.role.lower() not in ["mentor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only mentors and admins can perform this action",
        )


def require_mentee(current_user: User):
    if current_user.role.lower() != "mentee":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Classrooms are only available to mentees",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Seed helper — creates sample courses if DB is empty
# ─────────────────────────────────────────────────────────────────────────────

def seed_courses_if_empty(db: Session):
    if db.query(Course).count() > 0:
        return

    sample = [
        {
            "title": "UI/UX Foundations",
            "description": "Master the fundamentals of user interface and experience design. Learn Figma, design systems, and user research.",
            "category": "Design", "level": "beginner",
            "duration_hours": 12.5, "total_modules": 8,
            "instructor_name": "Sarah Jenkins",
            "instructor_bio": "Senior Product Designer at Google with 10+ years experience.",
            "thumbnail": None,
        },
        {
            "title": "Advanced Prototyping",
            "description": "Build high-fidelity interactive prototypes using Figma and Framer. Learn animation principles and micro-interactions.",
            "category": "Design", "level": "intermediate",
            "duration_hours": 8.0, "total_modules": 6,
            "instructor_name": "Marcus Chen",
            "instructor_bio": "Engineering Manager at Google specializing in design tooling.",
            "thumbnail": None,
        },
        {
            "title": "React for Designers",
            "description": "Learn React from a designer's perspective. Build interactive components without deep JavaScript knowledge.",
            "category": "Engineering", "level": "intermediate",
            "duration_hours": 15.0, "total_modules": 10,
            "instructor_name": "David Park",
            "instructor_bio": "Full-stack developer and design systems advocate.",
            "thumbnail": None,
        },
        {
            "title": "Product Management Essentials",
            "description": "Everything you need to transition into product management. Roadmaps, PRDs, stakeholder management and more.",
            "category": "Leadership", "level": "beginner",
            "duration_hours": 10.0, "total_modules": 7,
            "instructor_name": "Elena Rodriguez",
            "instructor_bio": "Head of Marketing at Notion, former PM at Stripe.",
            "thumbnail": None,
        },
        {
            "title": "Data Analysis with Python",
            "description": "Go from zero to hero in data analysis. Pandas, NumPy, Matplotlib and real-world Nigerian business datasets.",
            "category": "Engineering", "level": "beginner",
            "duration_hours": 20.0, "total_modules": 12,
            "instructor_name": "Chidi Okafor",
            "instructor_bio": "Data Scientist at Flutterwave with 8 years experience.",
            "thumbnail": None,
        },
        {
            "title": "Digital Marketing Strategy",
            "description": "Build and execute digital marketing campaigns. SEO, social media, email marketing and analytics.",
            "category": "Marketing", "level": "beginner",
            "duration_hours": 9.0, "total_modules": 6,
            "instructor_name": "Amara Nwosu",
            "instructor_bio": "Marketing Director at a leading Lagos-based fintech.",
            "thumbnail": None,
        },
    ]

    for s in sample:
        course = Course(**s)
        db.add(course)

    db.flush()  # get course IDs

    # Add modules to first course (UI/UX Foundations) as example
    course = db.query(Course).filter(Course.title == "UI/UX Foundations").first()
    if course:
        for i, mod in enumerate([
            ("Introduction to UX Design", 25),
            ("User Research Methods", 40),
            ("Wireframing Basics", 35),
            ("Design Systems Overview", 30),
            ("Module 3: Design Systems & Components", 45),
            ("Prototyping in Figma", 50),
            ("Usability Testing", 35),
            ("Final Project", 60),
        ], start=1):
            db.add(Module(course_id=course.id, title=mod[0],
                          duration_minutes=mod[1], order=i))

    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  GET /classrooms/my  (must be before /{id})
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/my", response_model=list[CourseCard])
def my_courses(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Return all courses the current user is enrolled in."""
    require_mentee(current_user)
    seed_courses_if_empty(db)

    enrollments = db.query(Enrollment).filter(
        Enrollment.user_id == current_user.id
    ).all()

    result = []
    for e in enrollments:
        card = CourseCard(
            id              = e.course.id,
            title           = e.course.title,
            description     = e.course.description,
            category        = e.course.category,
            level           = e.course.level,
            thumbnail       = e.course.thumbnail,
            duration_hours  = float(e.course.duration_hours or 0),
            total_modules   = e.course.total_modules,
            instructor_name = e.course.instructor_name,
            is_enrolled     = True,
            progress        = e.progress,
        )
        result.append(card)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  GET /classrooms
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[CourseCard])
def list_courses(
    search:   Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    level:    Optional[str] = Query(None),
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """List all published courses with optional filtering."""
    seed_courses_if_empty(db)

    q = db.query(Course).filter(Course.is_published == True)

    if search:
        q = q.filter(
            Course.title.ilike(f"%{search}%") |
            Course.description.ilike(f"%{search}%") |
            Course.instructor_name.ilike(f"%{search}%")
        )
    if category:
        q = q.filter(Course.category.ilike(f"%{category}%"))
    if level:
        q = q.filter(Course.level == level.lower())

    courses = q.all()

    # Get user's enrollments to mark enrolled courses
    enrolled_ids = {
        e.course_id: e.progress
        for e in db.query(Enrollment).filter(Enrollment.user_id == current_user.id).all()
    }

    return [
        CourseCard(
            id              = c.id,
            title           = c.title,
            description     = c.description,
            category        = c.category,
            level           = c.level,
            thumbnail       = c.thumbnail,
            duration_hours  = float(c.duration_hours or 0),
            total_modules   = c.total_modules,
            instructor_name = c.instructor_name,
            is_enrolled     = c.id in enrolled_ids,
            progress        = enrolled_ids.get(c.id, 0),
        )
        for c in courses
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  GET /classrooms/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=CourseDetail, status_code=status.HTTP_201_CREATED)
def create_course(
    payload: CourseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a published course with optional modules. Mentors and admins only."""
    require_mentor_or_admin(current_user)

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Course title is required")

    modules = [m for m in payload.modules if m.title.strip()]
    duration_hours = payload.duration_hours
    if duration_hours is None:
        duration_hours = round(sum(max(m.duration_minutes, 0) for m in modules) / 60, 1)

    course = Course(
        title=title,
        description=payload.description,
        category=payload.category or "General",
        level=(payload.level or "beginner").lower(),
        thumbnail=payload.thumbnail,
        duration_hours=duration_hours or 0,
        total_modules=len(modules),
        instructor_name=current_user.full_name,
        instructor_bio=payload.instructor_bio,
        is_published=payload.is_published,
    )
    db.add(course)
    db.flush()

    for index, module in enumerate(modules, start=1):
        db.add(Module(
            course_id=course.id,
            title=module.title.strip(),
            description=module.description,
            video_url=module.video_url,
            duration_minutes=max(module.duration_minutes, 0),
            order=index,
        ))

    db.commit()
    db.refresh(course)

    return CourseDetail(
        id=course.id,
        title=course.title,
        description=course.description,
        category=course.category,
        level=course.level,
        thumbnail=course.thumbnail,
        duration_hours=float(course.duration_hours or 0),
        total_modules=course.total_modules,
        instructor_name=course.instructor_name,
        instructor_bio=course.instructor_bio,
        is_enrolled=False,
        progress=0,
        modules=[
            ModuleOut(
                id=m.id,
                title=m.title,
                description=m.description,
                video_url=m.video_url,
                duration_minutes=m.duration_minutes,
                order=m.order,
                completed=False,
            )
            for m in course.modules
        ],
    )


@router.get("/assignments", response_model=list[AssignmentOut])
def list_assignments(
    course_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List published assignments for mentees; mentors can see all."""
    q = db.query(Assignment)
    if current_user.role.lower() not in ["mentor", "admin"]:
        q = q.filter(Assignment.is_published == True)
    if course_id:
        q = q.filter(Assignment.course_id == course_id)
    return q.order_by(Assignment.created_at.desc()).all()


@router.post("/assignments", response_model=AssignmentOut, status_code=status.HTTP_201_CREATED)
def create_assignment(
    payload: AssignmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create an assignment. Mentors and admins only."""
    require_mentor_or_admin(current_user)
    if payload.course_id and not db.query(Course).filter(Course.id == payload.course_id).first():
        raise HTTPException(status_code=404, detail="Course not found")

    assignment = Assignment(
        created_by=current_user.id,
        title=payload.title.strip(),
        instructions=payload.instructions,
        course_id=payload.course_id,
        due_at=payload.due_at,
        points=payload.points,
        is_published=payload.is_published,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment


@router.get("/quizzes", response_model=list[QuizOut])
def list_quizzes(
    course_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List published quizzes for mentees; mentors can see all."""
    q = db.query(Quiz)
    if current_user.role.lower() not in ["mentor", "admin"]:
        q = q.filter(Quiz.is_published == True)
    if course_id:
        q = q.filter(Quiz.course_id == course_id)
    return q.order_by(Quiz.created_at.desc()).all()


@router.post("/quizzes", response_model=QuizOut, status_code=status.HTTP_201_CREATED)
def create_quiz(
    payload: QuizCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a quiz. Mentors and admins only."""
    require_mentor_or_admin(current_user)
    if payload.course_id and not db.query(Course).filter(Course.id == payload.course_id).first():
        raise HTTPException(status_code=404, detail="Course not found")

    quiz = Quiz(
        created_by=current_user.id,
        title=payload.title.strip(),
        description=payload.description,
        course_id=payload.course_id,
        questions=payload.questions,
        time_limit_minutes=payload.time_limit_minutes,
        is_published=payload.is_published,
    )
    db.add(quiz)
    db.commit()
    db.refresh(quiz)
    return quiz


@router.get("/{course_id}", response_model=CourseDetail)
def get_course(
    course_id:    int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Get full course detail including all modules and user progress."""
    require_mentee(current_user)
    seed_courses_if_empty(db)

    course = db.query(Course).filter(
        Course.id == course_id,
        Course.is_published == True,
    ).first()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Check enrollment
    enrollment = db.query(Enrollment).filter(
        Enrollment.user_id  == current_user.id,
        Enrollment.course_id == course_id,
    ).first()

    # Get completed modules for this user
    completed_ids = {
        lp.module_id
        for lp in db.query(LessonProgress).filter(
            LessonProgress.user_id   == current_user.id,
            LessonProgress.completed == True,
        ).all()
    }

    modules = [
        ModuleOut(
            id               = m.id,
            title            = m.title,
            description      = m.description,
            video_url        = m.video_url,
            duration_minutes = m.duration_minutes,
            order            = m.order,
            completed        = m.id in completed_ids,
        )
        for m in course.modules
    ]

    return CourseDetail(
        id              = course.id,
        title           = course.title,
        description     = course.description,
        category        = course.category,
        level           = course.level,
        thumbnail       = course.thumbnail,
        duration_hours  = float(course.duration_hours or 0),
        total_modules   = course.total_modules,
        instructor_name = course.instructor_name,
        instructor_bio  = course.instructor_bio,
        is_enrolled     = enrollment is not None,
        progress        = enrollment.progress if enrollment else 0,
        modules         = modules,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /classrooms/{id}/enroll
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{course_id}/enroll", response_model=EnrollResponse)
def enroll(
    course_id:    int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Enroll the current user in a course."""
    require_mentee(current_user)
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    existing = db.query(Enrollment).filter(
        Enrollment.user_id   == current_user.id,
        Enrollment.course_id == course_id,
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Already enrolled in this course")

    enrollment = Enrollment(user_id=current_user.id, course_id=course_id)
    db.add(enrollment)
    db.commit()
    db.refresh(enrollment)

    return EnrollResponse(
        message     = f"Successfully enrolled in '{course.title}'",
        course_id   = course_id,
        enrolled_at = enrollment.enrolled_at,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /classrooms/{id}/progress
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{course_id}/progress", response_model=ProgressResponse)
def update_progress(
    course_id:    int,
    payload:      ProgressUpdate,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Mark a lesson as complete and recalculate overall course progress."""
    require_mentee(current_user)
    # Must be enrolled
    enrollment = db.query(Enrollment).filter(
        Enrollment.user_id   == current_user.id,
        Enrollment.course_id == course_id,
    ).first()

    if not enrollment:
        raise HTTPException(status_code=400, detail="Not enrolled in this course")

    # Upsert lesson progress
    lp = db.query(LessonProgress).filter(
        LessonProgress.user_id   == current_user.id,
        LessonProgress.module_id == payload.module_id,
    ).first()

    if lp:
        lp.completed  = payload.completed
        lp.watch_time = payload.watch_time or lp.watch_time
    else:
        lp = LessonProgress(
            user_id   = current_user.id,
            module_id = payload.module_id,
            completed = payload.completed,
            watch_time = payload.watch_time or 0,
        )
        db.add(lp)

    db.flush()

    # Recalculate overall course progress
    total   = db.query(Module).filter(Module.course_id == course_id).count()
    done    = db.query(LessonProgress).join(Module).filter(
        Module.course_id           == course_id,
        LessonProgress.user_id     == current_user.id,
        LessonProgress.completed   == True,
    ).count()

    pct = int((done / total) * 100) if total > 0 else 0
    enrollment.progress = pct

    if pct == 100:
        enrollment.completed_at = datetime.now(timezone.utc)

    db.commit()

    return ProgressResponse(
        message         = "Progress updated",
        course_progress = pct,
    )

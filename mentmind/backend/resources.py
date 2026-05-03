"""
Resources routes
GET  /resources              - list all resources (filterable)
GET  /resources/{id}         - single resource detail
POST /resources              - upload resource (mentors only)
DELETE /resources/{id}       - delete resource (owner or admin)
GET  /resources/course/{id}  - resources for a specific course
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime, BigInteger
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

from database import Base, get_db
from models import User
from dependencies import get_current_user

router = APIRouter(prefix="/resources", tags=["Resources"])


# ─────────────────────────────────────────────────────────────────────────────
#  DB Model
# ─────────────────────────────────────────────────────────────────────────────

class Resource(Base):
    __tablename__ = "resources"

    id           = Column(Integer, primary_key=True, index=True)
    uploaded_by  = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id    = Column(Integer, nullable=True)          # optional link to a course
    title        = Column(String(255), nullable=False)
    description  = Column(Text, nullable=True)
    file_type    = Column(String(20))                      # pdf | video | xlsx | zip | doc | image
    file_size    = Column(String(30), nullable=True)       # "4.2 MB"
    file_url     = Column(String, nullable=True)           # actual download URL (future)
    category     = Column(String(100), nullable=True)      # Design | Engineering | General …
    is_public    = Column(Boolean, default=True)
    download_count = Column(Integer, default=0)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    uploader = relationship("User")


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class UploaderOut(BaseModel):
    id: int
    full_name: str
    role: str
    profile_image: Optional[str]

    class Config:
        from_attributes = True


class ResourceOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    file_type: str
    file_size: Optional[str]
    file_url: Optional[str]
    category: Optional[str]
    course_id: Optional[int]
    is_public: bool
    download_count: int
    uploader: UploaderOut
    created_at: datetime
    time_ago: str

    class Config:
        from_attributes = True


class CreateResourceRequest(BaseModel):
    title: str
    description: Optional[str] = None
    file_type: str = "pdf"
    file_size: Optional[str] = None
    file_url: Optional[str] = None
    category: Optional[str] = "General"
    course_id: Optional[int] = None
    is_public: bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

FILE_ICONS = {
    "pdf":   "📄", "video": "🎬", "xlsx": "📊",
    "zip":   "📦", "doc":   "📝", "image": "🖼️",
    "ppt":   "📋", "csv":   "📈", "other": "📁",
}

def time_ago(dt: datetime) -> str:
    now  = datetime.now(timezone.utc)
    diff = now - dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else now - dt
    s    = int(diff.total_seconds())
    if s < 60:      return "just now"
    if s < 3600:    return f"{s//60}m ago"
    if s < 86400:   return f"{s//3600}h ago"
    if s < 604800:  return f"{s//86400}d ago"
    return dt.strftime("%b %d")


def to_out(r: Resource) -> ResourceOut:
    return ResourceOut(
        id             = r.id,
        title          = r.title,
        description    = r.description,
        file_type      = r.file_type,
        file_size      = r.file_size,
        file_url       = r.file_url,
        category       = r.category,
        course_id      = r.course_id,
        is_public      = r.is_public,
        download_count = r.download_count,
        uploader       = UploaderOut(
            id            = r.uploader.id,
            full_name     = r.uploader.full_name,
            role          = r.uploader.role,
            profile_image = r.uploader.profile_image,
        ),
        created_at = r.created_at,
        time_ago   = time_ago(r.created_at),
    )


def seed_resources_if_empty(db: Session, user_id: int):
    if db.query(Resource).count() > 0:
        return

    samples = [
        {
            "title": "Design System Guide",
            "description": "Comprehensive guide to building scalable design systems with Figma.",
            "file_type": "pdf",
            "file_size": "4.2 MB",
            "category": "Design",
        },
        {
            "title": "Token Structure Template",
            "description": "Excel template for organizing design tokens across platforms.",
            "file_type": "xlsx",
            "file_size": "1.1 MB",
            "category": "Design",
        },
        {
            "title": "Asset Pack v1",
            "description": "Complete asset pack with icons, illustrations and UI components.",
            "file_type": "zip",
            "file_size": "25 MB",
            "category": "Design",
        },
        {
            "title": "Portfolio Review Session #4",
            "description": "Recorded session covering portfolio best practices for junior designers.",
            "file_type": "video",
            "file_size": "850 MB",
            "category": "Career",
        },
        {
            "title": "React Component Library",
            "description": "Pre-built React components following atomic design principles.",
            "file_type": "zip",
            "file_size": "12 MB",
            "category": "Engineering",
        },
        {
            "title": "Career Roadmap Template",
            "description": "Plan your career path with this structured roadmap template.",
            "file_type": "doc",
            "file_size": "380 KB",
            "category": "Career",
        },
        {
            "title": "Python Data Analysis Notebook",
            "description": "Jupyter notebook with real-world data analysis examples using Nigerian datasets.",
            "file_type": "other",
            "file_size": "2.3 MB",
            "category": "Engineering",
        },
        {
            "title": "Interview Preparation Guide",
            "description": "50+ common technical interview questions with detailed answers.",
            "file_type": "pdf",
            "file_size": "1.8 MB",
            "category": "Career",
        },
    ]

    for s in samples:
        db.add(Resource(uploaded_by=user_id, **s))

    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  GET /resources/course/{course_id}   (must be before /{id})
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/course/{course_id}", response_model=list[ResourceOut])
def resources_by_course(
    course_id:    int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Get all resources linked to a specific course."""
    seed_resources_if_empty(db, current_user.id)
    resources = db.query(Resource).filter(
        Resource.course_id == course_id,
        Resource.is_public == True,
    ).all()
    return [to_out(r) for r in resources]


# ─────────────────────────────────────────────────────────────────────────────
#  GET /resources
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ResourceOut])
def list_resources(
    search:    Optional[str] = Query(None),
    category:  Optional[str] = Query(None),
    file_type: Optional[str] = Query(None),
    sort_by:   str           = Query("latest", description="latest | popular"),
    limit:     int           = Query(20, ge=1, le=100),
    offset:    int           = Query(0, ge=0),
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """List all public resources with optional search and filter."""
    seed_resources_if_empty(db, current_user.id)

    q = db.query(Resource).filter(Resource.is_public == True)

    if search:
        q = q.filter(
            Resource.title.ilike(f"%{search}%") |
            Resource.description.ilike(f"%{search}%")
        )

    if category and category.lower() != "all":
        q = q.filter(Resource.category.ilike(f"%{category}%"))

    if file_type:
        q = q.filter(Resource.file_type == file_type.lower())

    resources = q.all()

    if sort_by == "popular":
        resources.sort(key=lambda r: r.download_count, reverse=True)
    else:
        resources.sort(key=lambda r: r.created_at, reverse=True)

    return [to_out(r) for r in resources[offset: offset + limit]]


# ─────────────────────────────────────────────────────────────────────────────
#  GET /resources/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{resource_id}", response_model=ResourceOut)
def get_resource(
    resource_id:  int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    r = db.query(Resource).filter(Resource.id == resource_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Resource not found")

    # Increment download count
    r.download_count += 1
    db.commit()

    return to_out(r)


# ─────────────────────────────────────────────────────────────────────────────
#  POST /resources   (mentors and admins only)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=ResourceOut, status_code=status.HTTP_201_CREATED)
def create_resource(
    payload:      CreateResourceRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Upload a new resource. Mentors and admins only."""
    if current_user.role.lower() not in ["mentor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only mentors and admins can upload resources",
        )

    r = Resource(
        uploaded_by = current_user.id,
        title       = payload.title.strip(),
        description = payload.description,
        file_type   = payload.file_type.lower(),
        file_size   = payload.file_size,
        file_url    = payload.file_url,
        category    = payload.category,
        course_id   = payload.course_id,
        is_public   = payload.is_public,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return to_out(r)


# ─────────────────────────────────────────────────────────────────────────────
#  DELETE /resources/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resource(
    resource_id:  int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    r = db.query(Resource).filter(Resource.id == resource_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Resource not found")

    if r.uploaded_by != current_user.id and current_user.role.lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own resources",
        )

    db.delete(r)
    db.commit()

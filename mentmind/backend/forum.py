"""
Community Forum routes
GET  /forum/topics              - list topics (with filters)
POST /forum/topics              - create a topic
GET  /forum/topics/{id}         - topic detail + replies
POST /forum/topics/{id}/reply   - add a reply
POST /forum/topics/{id}/like    - like / unlike a topic
GET  /forum/trending            - trending topics
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, timedelta

from database import Base, get_db
from models import User
from dependencies import get_current_user

router = APIRouter(prefix="/forum", tags=["Community Forum"])


# ─────────────────────────────────────────────────────────────────────────────
#  DB Models
# ─────────────────────────────────────────────────────────────────────────────

class ForumTopic(Base):
    __tablename__ = "forum_topics"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    title      = Column(String(255), nullable=False)
    body       = Column(Text, nullable=False)
    category   = Column(String(100), default="General")
    views      = Column(Integer, default=0)
    is_pinned  = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    author  = relationship("User")
    replies = relationship("ForumReply", back_populates="topic")
    likes   = relationship("ForumLike", back_populates="topic")


class ForumReply(Base):
    __tablename__ = "forum_replies"
    id         = Column(Integer, primary_key=True, index=True)
    topic_id   = Column(Integer, ForeignKey("forum_topics.id"), nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    body       = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    topic  = relationship("ForumTopic", back_populates="replies")
    author = relationship("User")


class ForumLike(Base):
    __tablename__ = "forum_likes"
    id       = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("forum_topics.id"), nullable=False)
    user_id  = Column(Integer, ForeignKey("users.id"), nullable=False)

    topic = relationship("ForumTopic", back_populates="likes")


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AuthorOut(BaseModel):
    id: int
    full_name: str
    profile_image: Optional[str]
    role: str

    class Config:
        from_attributes = True


class TopicCard(BaseModel):
    id: int
    title: str
    body: str
    category: str
    views: int
    reply_count: int
    like_count: int
    liked_by_me: bool
    author: AuthorOut
    created_at: datetime
    time_ago: str


class ReplyOut(BaseModel):
    id: int
    body: str
    author: AuthorOut
    created_at: datetime
    time_ago: str


class TopicDetail(TopicCard):
    replies: list[ReplyOut]


class CreateTopicRequest(BaseModel):
    title: str
    body: str
    category: str = "General"


class CreateReplyRequest(BaseModel):
    body: str


class TrendingTopic(BaseModel):
    id: int
    title: str
    reply_count: int


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES = ["Career Advice", "Technical Skills", "Soft Skills",
              "Scholarships", "General"]

def time_ago(dt: datetime) -> str:
    """Human-readable relative time."""
    now  = datetime.now(timezone.utc)
    diff = now - dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else now - dt
    s    = int(diff.total_seconds())
    if s < 60:        return "just now"
    if s < 3600:      return f"{s//60}m ago"
    if s < 86400:     return f"{s//3600}h ago"
    if s < 604800:    return f"{s//86400}d ago"
    return dt.strftime("%b %d")


def topic_to_card(topic: ForumTopic, user_id: int) -> TopicCard:
    liked = any(l.user_id == user_id for l in topic.likes)
    return TopicCard(
        id          = topic.id,
        title       = topic.title,
        body        = topic.body,
        category    = topic.category,
        views       = topic.views,
        reply_count = len(topic.replies),
        like_count  = len(topic.likes),
        liked_by_me = liked,
        author      = AuthorOut(
            id            = topic.author.id,
            full_name     = topic.author.full_name,
            profile_image = topic.author.profile_image,
            role          = topic.author.role,
        ),
        created_at  = topic.created_at,
        time_ago    = time_ago(topic.created_at),
    )


def seed_forum_if_empty(db: Session, user_id: int):
    """Seed sample forum topics if the table is empty."""
    if db.query(ForumTopic).count() > 0:
        return

    samples = [
        ("How to prepare for your first Junior Dev interview?",
         "I finally landed an interview at a tech startup! Does anyone have tips on what technical questions they might ask for a React role?",
         "Career Advice"),
        ("Global Empowerment Scholarship 2024 is now open",
         "Just saw that the GE scholarship applications are live. It covers full tuition and a monthly stipend for international students.",
         "Scholarships"),
        ("The importance of active listening in mentorship",
         "I've been meeting with my mentor for 3 months now, and I've realized that the most valuable skill I've learned isn't technical.",
         "Soft Skills"),
        ("Best resources for learning Python for Data Science?",
         "I'm looking to transition into data analytics. Which libraries should I focus on after mastering the basics of Python?",
         "Technical Skills"),
        ("AI in 2024: What mentees need to know",
         "Let's discuss how AI is changing the landscape for junior professionals and what skills are becoming critical.",
         "Technical Skills"),
    ]

    for title, body, category in samples:
        t = ForumTopic(user_id=user_id, title=title, body=body, category=category)
        db.add(t)

    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  GET /forum/trending  (before /{id})
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/trending", response_model=list[TrendingTopic])
def trending_topics(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Return the top 5 most-replied topics from the last 7 days."""
    seed_forum_if_empty(db, current_user.id)

    topics = db.query(ForumTopic).all()
    sorted_topics = sorted(topics, key=lambda t: len(t.replies), reverse=True)[:5]

    return [
        TrendingTopic(id=t.id, title=t.title, reply_count=len(t.replies))
        for t in sorted_topics
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  GET /forum/topics
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/topics", response_model=list[TopicCard])
def list_topics(
    search:   Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    sort_by:  str           = Query("latest", description="latest | popular | views"),
    limit:    int           = Query(20, ge=1, le=100),
    offset:   int           = Query(0, ge=0),
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """List forum topics with optional search, category filter, and sort."""
    seed_forum_if_empty(db, current_user.id)

    q = db.query(ForumTopic)

    if search:
        q = q.filter(
            ForumTopic.title.ilike(f"%{search}%") |
            ForumTopic.body.ilike(f"%{search}%")
        )

    if category and category.lower() != "all":
        q = q.filter(ForumTopic.category.ilike(f"%{category}%"))

    topics = q.all()

    # Sort in Python (simpler than SQL for computed fields)
    if sort_by == "popular":
        topics.sort(key=lambda t: len(t.likes), reverse=True)
    elif sort_by == "views":
        topics.sort(key=lambda t: t.views, reverse=True)
    else:
        topics.sort(key=lambda t: t.created_at, reverse=True)

    # Pinned topics always first
    pinned = [t for t in topics if t.is_pinned]
    rest   = [t for t in topics if not t.is_pinned]
    topics = pinned + rest

    return [topic_to_card(t, current_user.id) for t in topics[offset: offset + limit]]


# ─────────────────────────────────────────────────────────────────────────────
#  POST /forum/topics
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/topics", response_model=TopicCard, status_code=status.HTTP_201_CREATED)
def create_topic(
    payload:      CreateTopicRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Create a new forum topic."""
    if payload.category not in CATEGORIES:
        payload.category = "General"

    topic = ForumTopic(
        user_id  = current_user.id,
        title    = payload.title.strip(),
        body     = payload.body.strip(),
        category = payload.category,
    )
    db.add(topic)
    db.commit()
    db.refresh(topic)
    return topic_to_card(topic, current_user.id)


# ─────────────────────────────────────────────────────────────────────────────
#  GET /forum/topics/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/topics/{topic_id}", response_model=TopicDetail)
def get_topic(
    topic_id:     int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Get full topic detail including all replies. Also increments view count."""
    topic = db.query(ForumTopic).filter(ForumTopic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    # Increment views
    topic.views += 1
    db.commit()

    replies = [
        ReplyOut(
            id         = r.id,
            body       = r.body,
            author     = AuthorOut(
                id            = r.author.id,
                full_name     = r.author.full_name,
                profile_image = r.author.profile_image,
                role          = r.author.role,
            ),
            created_at = r.created_at,
            time_ago   = time_ago(r.created_at),
        )
        for r in sorted(topic.replies, key=lambda r: r.created_at)
    ]

    card = topic_to_card(topic, current_user.id)
    return TopicDetail(**card.model_dump(), replies=replies)


# ─────────────────────────────────────────────────────────────────────────────
#  POST /forum/topics/{id}/reply
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/topics/{topic_id}/reply", response_model=ReplyOut,
             status_code=status.HTTP_201_CREATED)
def add_reply(
    topic_id:     int,
    payload:      CreateReplyRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Add a reply to a topic."""
    topic = db.query(ForumTopic).filter(ForumTopic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    reply = ForumReply(
        topic_id = topic_id,
        user_id  = current_user.id,
        body     = payload.body.strip(),
    )
    db.add(reply)
    db.commit()
    db.refresh(reply)

    return ReplyOut(
        id         = reply.id,
        body       = reply.body,
        author     = AuthorOut(
            id            = current_user.id,
            full_name     = current_user.full_name,
            profile_image = current_user.profile_image,
            role          = current_user.role,
        ),
        created_at = reply.created_at,
        time_ago   = "just now",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /forum/topics/{id}/like
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/topics/{topic_id}/like")
def toggle_like(
    topic_id:     int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Toggle like on a topic. Returns new like count and liked status."""
    topic = db.query(ForumTopic).filter(ForumTopic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    existing = db.query(ForumLike).filter(
        ForumLike.topic_id == topic_id,
        ForumLike.user_id  == current_user.id,
    ).first()

    if existing:
        db.delete(existing)
        db.commit()
        liked = False
    else:
        db.add(ForumLike(topic_id=topic_id, user_id=current_user.id))
        db.commit()
        liked = True

    count = db.query(ForumLike).filter(ForumLike.topic_id == topic_id).count()
    return {"liked": liked, "like_count": count}

"""
Opportunities Hub routes
GET  /opportunities              - list all opportunities
GET  /opportunities/{id}         - opportunity detail
POST /opportunities/{id}/apply   - apply to opportunity
GET  /opportunities/my           - my applications
POST /opportunities              - post opportunity (partners only)
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, DateTime, Date
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date, timezone

from database import Base, get_db
from models import User
from dependencies import get_current_user

router = APIRouter(prefix="/opportunities", tags=["Opportunities Hub"])


# ─────────────────────────────────────────────────────────────────────────────
#  DB Models
# ─────────────────────────────────────────────────────────────────────────────

class Opportunity(Base):
    __tablename__ = "opportunities"
    id              = Column(Integer, primary_key=True, index=True)
    posted_by       = Column(Integer, ForeignKey("users.id"), nullable=True)
    title           = Column(String(255), nullable=False)
    organization    = Column(String(255), nullable=False)
    type            = Column(String(50))   # Internship | Scholarship | Workshop | Volunteering
    location        = Column(String(255))
    description     = Column(Text)
    requirements    = Column(Text)
    deadline        = Column(Date, nullable=True)
    apply_url       = Column(String)
    thumbnail       = Column(String)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    poster       = relationship("User")
    applications = relationship("Application", back_populates="opportunity")


class Application(Base):
    __tablename__ = "applications"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    opportunity_id  = Column(Integer, ForeignKey("opportunities.id"), nullable=False)
    cover_note      = Column(Text)
    status          = Column(String(20), default="pending")  # pending | reviewed | accepted | rejected
    applied_at      = Column(DateTime(timezone=True), server_default=func.now())

    applicant    = relationship("User")
    opportunity  = relationship("Opportunity", back_populates="applications")


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class OpportunityCard(BaseModel):
    id: int
    title: str
    organization: str
    type: Optional[str]
    location: Optional[str]
    description: Optional[str]
    deadline: Optional[date]
    apply_url: Optional[str]
    thumbnail: Optional[str]
    applicant_count: int
    has_applied: bool
    days_left: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class OpportunityDetail(OpportunityCard):
    requirements: Optional[str]
    posted_by_name: Optional[str]


class CreateOpportunityRequest(BaseModel):
    title: str
    organization: str
    type: str = "Internship"
    location: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None
    deadline: Optional[date] = None
    apply_url: Optional[str] = None


class ApplyRequest(BaseModel):
    cover_note: Optional[str] = None


class ApplicationOut(BaseModel):
    id: int
    status: str
    applied_at: datetime
    opportunity: OpportunityCard


# ─────────────────────────────────────────────────────────────────────────────
#  Seed helper
# ─────────────────────────────────────────────────────────────────────────────

def seed_opportunities_if_empty(db: Session):
    if db.query(Opportunity).count() > 0:
        return

    samples = [
        {
            "title": "UI/UX Design Summer '24",
            "organization": "FinTech Startup",
            "type": "Internship",
            "location": "Remote / Global",
            "description": "Join our product team to build the next generation of fintech tools for African markets. Work alongside senior designers.",
            "requirements": "Portfolio required. Figma experience preferred.",
            "deadline": date(2024, 10, 30),
        },
        {
            "title": "STEM Excellence Grant",
            "organization": "Global Education Fund",
            "type": "Scholarship",
            "location": "Academic",
            "description": "Full tuition coverage for undergraduate students pursuing degrees in STEM fields. Monthly stipend included.",
            "requirements": "Min GPA 3.5. Open to African students.",
            "deadline": date(2024, 11, 15),
        },
        {
            "title": "Leadership Bootcamp",
            "organization": "AfriLeaders Initiative",
            "type": "Workshop",
            "location": "Nairobi, Kenya",
            "description": "A 3-day intensive workshop focused on community organizing and public speaking for young African leaders.",
            "requirements": "18-30 years old. Letter of motivation required.",
            "deadline": date(2024, 10, 25),
        },
        {
            "title": "Green Earth Initiative",
            "organization": "EcoNigeria",
            "type": "Volunteering",
            "location": "Hybrid",
            "description": "Collaborate with local partners to implement sustainable waste management solutions in Lagos communities.",
            "requirements": "Passion for sustainability. 5 hours/week commitment.",
            "deadline": date(2024, 12, 1),
        },
        {
            "title": "Backend Engineering Internship",
            "organization": "TechCorp Lagos",
            "type": "Internship",
            "location": "Lagos, Nigeria",
            "description": "Work with Python and Go to scale our community-driven API services. Mentorship provided.",
            "requirements": "Python knowledge required. Django/FastAPI preferred.",
            "deadline": date(2024, 10, 28),
        },
        {
            "title": "Startup Mentorship Circle",
            "organization": "Venture Lagos",
            "type": "Workshop",
            "location": "Virtual",
            "description": "Exclusive access to venture capitalists and serial entrepreneurs. Build your network and learn to pitch.",
            "requirements": "Must have a startup idea or early-stage business.",
            "deadline": date(2025, 1, 10),
        },
    ]

    for s in samples:
        db.add(Opportunity(**s))

    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def days_until(d: date) -> Optional[int]:
    if not d:
        return None
    delta = d - date.today()
    return max(0, delta.days)


def opp_to_card(opp: Opportunity, user_id: int) -> OpportunityCard:
    applied = any(a.user_id == user_id for a in opp.applications)
    return OpportunityCard(
        id              = opp.id,
        title           = opp.title,
        organization    = opp.organization,
        type            = opp.type,
        location        = opp.location,
        description     = opp.description,
        deadline        = opp.deadline,
        apply_url       = opp.apply_url,
        thumbnail       = opp.thumbnail,
        applicant_count = len(opp.applications),
        has_applied     = applied,
        days_left       = days_until(opp.deadline),
        created_at      = opp.created_at,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /opportunities/my  (before /{id})
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/my", response_model=list[ApplicationOut])
def my_applications(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Return all opportunities the current user has applied to."""
    apps = db.query(Application).filter(Application.user_id == current_user.id).all()
    return [
        ApplicationOut(
            id          = a.id,
            status      = a.status,
            applied_at  = a.applied_at,
            opportunity = opp_to_card(a.opportunity, current_user.id),
        )
        for a in apps
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  GET /opportunities
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[OpportunityCard])
def list_opportunities(
    search:   Optional[str] = Query(None),
    type:     Optional[str] = Query(None, description="Internship | Scholarship | Workshop | Volunteering"),
    sort_by:  str           = Query("latest", description="latest | deadline"),
    limit:    int           = Query(20, ge=1, le=100),
    offset:   int           = Query(0, ge=0),
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """List all active opportunities."""
    seed_opportunities_if_empty(db)

    q = db.query(Opportunity).filter(Opportunity.is_active == True)

    if search:
        q = q.filter(
            Opportunity.title.ilike(f"%{search}%") |
            Opportunity.organization.ilike(f"%{search}%") |
            Opportunity.description.ilike(f"%{search}%")
        )

    if type and type.lower() != "all programs":
        q = q.filter(Opportunity.type.ilike(f"%{type}%"))

    opps = q.all()

    if sort_by == "deadline":
        opps.sort(key=lambda o: o.deadline or date.max)
    else:
        opps.sort(key=lambda o: o.created_at, reverse=True)

    return [opp_to_card(o, current_user.id) for o in opps[offset: offset + limit]]


# ─────────────────────────────────────────────────────────────────────────────
#  GET /opportunities/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{opp_id}", response_model=OpportunityDetail)
def get_opportunity(
    opp_id:       int,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    opp = db.query(Opportunity).filter(
        Opportunity.id == opp_id,
        Opportunity.is_active == True,
    ).first()

    if not opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")

    card = opp_to_card(opp, current_user.id)
    return OpportunityDetail(
        **card.model_dump(),
        requirements    = opp.requirements,
        posted_by_name  = opp.poster.full_name if opp.poster else "MentMinds Team",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /opportunities/{id}/apply
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{opp_id}/apply", status_code=status.HTTP_201_CREATED)
def apply(
    opp_id:       int,
    payload:      ApplyRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    opp = db.query(Opportunity).filter(Opportunity.id == opp_id).first()
    if not opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")

    existing = db.query(Application).filter(
        Application.user_id        == current_user.id,
        Application.opportunity_id == opp_id,
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Already applied to this opportunity")

    app = Application(
        user_id        = current_user.id,
        opportunity_id = opp_id,
        cover_note     = payload.cover_note,
    )
    db.add(app)
    db.commit()

    return {"message": f"Successfully applied to '{opp.title}'!", "status": "pending"}


# ─────────────────────────────────────────────────────────────────────────────
#  POST /opportunities  (partners only)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=OpportunityCard, status_code=status.HTTP_201_CREATED)
def post_opportunity(
    payload:      CreateOpportunityRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Post a new opportunity. Partners and admins only."""
    if current_user.role.lower() not in ["partner", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only partners and admins can post opportunities",
        )

    opp = Opportunity(
        posted_by    = current_user.id,
        title        = payload.title,
        organization = payload.organization,
        type         = payload.type,
        location     = payload.location,
        description  = payload.description,
        requirements = payload.requirements,
        deadline     = payload.deadline,
        apply_url    = payload.apply_url,
    )
    db.add(opp)
    db.commit()
    db.refresh(opp)

    return opp_to_card(opp, current_user.id)

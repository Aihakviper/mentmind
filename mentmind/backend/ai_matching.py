"""
ML Mentor Matching Router
Integrates the trained Random Forest model into FastAPI.

Endpoints:
  GET  /ml/status            - check if model is loaded
  POST /ml/match             - get match score for a single mentor-mentee pair
  GET  /ml/recommend/{id}    - get top-k ML-ranked mentors for a mentee
  POST /ml/recommend         - get ML recommendations (pass mentee data directly)
"""
import os
import numpy as np
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from models import User, MenteeProfile, MentorProfile
from dependencies import get_current_user

router = APIRouter(prefix="/ml", tags=["ML Mentor Matching"])

# ─────────────────────────────────────────────────────────────────────────────
#  Load model on startup (lazy — loads once, reuses)
# ─────────────────────────────────────────────────────────────────────────────

_model          = None
_feature_cols   = None
_scaler         = None
_model_type     = "rule_based"   # fallback label

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ML_MODELS_DIR = Path(os.environ.get("ML_MODELS_PATH", "ai_services/model"))
if not ML_MODELS_DIR.is_absolute():
    ML_MODELS_DIR = BASE_DIR / ML_MODELS_DIR


def load_model():
    """Try to load the saved ML model. Falls back to rule-based scoring."""
    global _model, _feature_cols, _scaler, _model_type
    if _model is not None:
        return True

    try:
        import joblib

        model_path   = os.path.join(ML_MODELS_DIR, "mentor_matching_modelv2.pkl")
        cols_path    = os.path.join(ML_MODELS_DIR, "feature_columnsv2.pkl")
        scaler_path  = os.path.join(ML_MODELS_DIR, "feature_scalerv2.pkl")

        # Try v2 first, then v1
        if not os.path.exists(model_path):
            model_path  = os.path.join(ML_MODELS_DIR, "mentor_matching_model.pkl")
            cols_path   = os.path.join(ML_MODELS_DIR, "feature_columns.pkl")
            scaler_path = os.path.join(ML_MODELS_DIR, "feature_scaler.pkl")

        if not os.path.exists(model_path):
            print("ML model files not found - using rule-based fallback")
            return False

        _model        = joblib.load(model_path)
        _feature_cols = joblib.load(cols_path)
        _model_type   = type(_model).__name__

        if os.path.exists(scaler_path):
            _scaler = joblib.load(scaler_path)

        print(f"ML model loaded: {_model_type}")
        return True

    except Exception as e:
        print(f"Could not load ML model: {e} - using rule-based fallback")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering — matches the 14 features from your training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _domain_overlap(mentor_domains: list, mentee_interests: list) -> float:
    if not mentor_domains or not mentee_interests:
        return 0.0
    overlap = len(set(mentor_domains) & set(mentee_interests))
    return overlap / len(mentee_interests)


def _skill_overlap(mentor_domains: list, mentee_interests: list) -> float:
    if not mentor_domains or not mentee_interests:
        return 0.0
    a, b   = set(mentor_domains), set(mentee_interests)
    total  = len(a | b)
    return len(a & b) / total if total > 0 else 0.0


def _availability_compat(mentor_hours: float, mentee_hours: int) -> float:
    diff = abs((mentor_hours or 10) - (mentee_hours or 10))
    return max(0.0, 1.0 - diff / 30)


def _experience_compat(years: int, level: str) -> float:
    level_map = {"beginner": 1, "intermediate": 2, "advanced": 3}
    n = level_map.get(level, 2)
    if n == 1 and years >= 10:
        return 1.0
    if n == 2 and years >= 7:
        return 0.8
    if n == 3 and years >= 5:
        return 0.6
    return 0.3


def build_feature_vector(mentor: MentorProfile, mentee: MenteeProfile) -> dict:
    """Build the same 14 features used during training."""
    m_domains   = mentor.expertise_areas  or []
    me_domains  = mentee.areas_of_interest or []
    m_years     = mentor.years_of_experience or 0
    m_rating    = float(mentor.rating or 0)
    m_rate      = float(mentor.acceptance_rate or 0)
    m_mentees   = mentor.total_mentees or 0
    me_level    = mentee.current_level or "beginner"
    me_hours    = mentee.availability_hours or 10
    m_hours     = 10   # default — MentorProfile doesn't store this yet

    level_map   = {"beginner": 1, "intermediate": 2, "advanced": 3}

    return {
        "domain_overlap":             _domain_overlap(m_domains, me_domains),
        "skill_overlap":              _skill_overlap(m_domains, me_domains),
        "availability_compatibility": _availability_compat(m_hours, me_hours),
        "style_match":                0,    # no style field yet
        "industry_match":             0,    # no industry field yet
        "mentor_experience_years":    m_years,
        "mentor_rating":              m_rating,
        "mentor_acceptance_rate":     m_rate,
        "mentor_total_mentees":       m_mentees,
        "mentee_level_numeric":       level_map.get(me_level, 2),
        "experience_compatibility":   _experience_compat(m_years, me_level),
        "mentor_domain_count":        len(m_domains),
        "mentee_domain_count":        len(me_domains),
        "mentor_skill_count":         len(m_domains),
    }


def rule_based_score(mentor: MentorProfile, mentee: MenteeProfile) -> float:
    """Fallback when ML model is not available."""
    fv    = build_feature_vector(mentor, mentee)
    score = (
        fv["domain_overlap"]             * 0.35 +
        fv["skill_overlap"]              * 0.15 +
        fv["availability_compatibility"] * 0.10 +
        fv["experience_compatibility"]   * 0.15 +
        min(fv["mentor_rating"] / 5, 1)  * 0.15 +
        min(fv["mentor_acceptance_rate"] / 100, 1) * 0.10
    )
    return round(float(score) * 100, 1)


def ml_score(mentor: MentorProfile, mentee: MenteeProfile) -> float:
    """Use the trained model to predict match probability."""
    global _model, _feature_cols, _scaler

    fv = build_feature_vector(mentor, mentee)

    # Build feature array in the right column order
    cols = _feature_cols if _feature_cols else list(fv.keys())
    X    = np.array([[fv.get(c, 0) for c in cols]])

    if _scaler:
        X = _scaler.transform(X)

    # predict_proba → probability of class 1 (successful match)
    proba = _model.predict_proba(X)[0][1]
    return round(float(proba) * 100, 1)


def get_match_score(mentor: MentorProfile, mentee: MenteeProfile) -> float:
    if load_model() and _model is not None:
        try:
            return ml_score(mentor, mentee)
        except Exception as e:
            print(f"ML scoring failed: {e} - using rule-based")
    return rule_based_score(mentor, mentee)


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class MLMatchResult(BaseModel):
    mentor_id:    int
    mentor_name:  str
    position:     Optional[str]
    company:      Optional[str]
    match_score:  float           # 0–100
    match_type:   str             # "ml_model" | "rule_based"
    features:     dict
    expertise_areas: list[str]
    rating:       float
    years_of_experience: int


class ManualMatchRequest(BaseModel):
    """Pass mentee data directly — useful for testing without a DB record."""
    areas_of_interest: list[str]
    current_level:     str = "beginner"
    availability_hours: int = 10
    top_k:             int = 5


# ─────────────────────────────────────────────────────────────────────────────
#  GET /ml/status
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status")
def model_status():
    """Check whether the ML model is loaded and ready."""
    loaded = load_model()
    return {
        "ml_model_loaded": loaded,
        "model_type":      _model_type,
        "feature_count":   len(_feature_cols) if _feature_cols else 14,
        "features":        list(_feature_cols) if _feature_cols else [
            "domain_overlap", "skill_overlap", "availability_compatibility",
            "style_match", "industry_match", "mentor_experience_years",
            "mentor_rating", "mentor_acceptance_rate", "mentor_total_mentees",
            "mentee_level_numeric", "experience_compatibility",
            "mentor_domain_count", "mentee_domain_count", "mentor_skill_count",
        ],
        "models_directory": ML_MODELS_DIR,
        "scoring_method":  "ML model" if loaded else "Rule-based fallback",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GET /ml/recommend/{mentee_id}  — top-k mentors for a registered mentee
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/recommend/{mentee_user_id}", response_model=list[MLMatchResult])
def recommend_for_mentee(
    mentee_user_id: int,
    top_k:          int           = 5,
    current_user:   User          = Depends(get_current_user),
    db:             Session       = Depends(get_db),
):
    """
    Get top-k mentor recommendations for a mentee, ranked by the ML model.
    Any authenticated user can query (admins can query for any mentee).
    """
    mentee_profile = db.query(MenteeProfile).filter(
        MenteeProfile.user_id == mentee_user_id
    ).first()

    if not mentee_profile:
        raise HTTPException(status_code=404, detail="Mentee profile not found")

    mentors = db.query(MentorProfile).filter(
        MentorProfile.verification_status == "verified"
    ).all()

    results = []
    model_loaded = load_model()

    for mentor in mentors:
        score = get_match_score(mentor, mentee_profile)
        fv    = build_feature_vector(mentor, mentee_profile)

        results.append(MLMatchResult(
            mentor_id           = mentor.id,
            mentor_name         = mentor.user.full_name,
            position            = mentor.current_position,
            company             = mentor.company,
            match_score         = score,
            match_type          = "ml_model" if (model_loaded and _model) else "rule_based",
            features            = {k: round(v, 4) for k, v in fv.items()},
            expertise_areas     = mentor.expertise_areas or [],
            rating              = float(mentor.rating or 0),
            years_of_experience = mentor.years_of_experience or 0,
        ))

    results.sort(key=lambda x: x.match_score, reverse=True)
    return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
#  POST /ml/recommend  — pass mentee data directly (no DB record needed)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/recommend", response_model=list[MLMatchResult])
def recommend_direct(
    payload:      ManualMatchRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """
    Score all mentors against a mentee profile provided directly in the request.
    Useful for the admin panel and for testing.
    """
    # Create a temporary MenteeProfile object (not saved to DB)
    temp_mentee = MenteeProfile(
        user_id            = 0,
        areas_of_interest  = payload.areas_of_interest,
        current_level      = payload.current_level,
        availability_hours = payload.availability_hours,
    )

    mentors = db.query(MentorProfile).filter(
        MentorProfile.verification_status == "verified"
    ).all()

    model_loaded = load_model()
    results      = []

    for mentor in mentors:
        score = get_match_score(mentor, temp_mentee)
        fv    = build_feature_vector(mentor, temp_mentee)

        results.append(MLMatchResult(
            mentor_id           = mentor.id,
            mentor_name         = mentor.user.full_name,
            position            = mentor.current_position,
            company             = mentor.company,
            match_score         = score,
            match_type          = "ml_model" if (model_loaded and _model) else "rule_based",
            features            = {k: round(v, 4) for k, v in fv.items()},
            expertise_areas     = mentor.expertise_areas or [],
            rating              = float(mentor.rating or 0),
            years_of_experience = mentor.years_of_experience or 0,
        ))

    results.sort(key=lambda x: x.match_score, reverse=True)
    return results[:payload.top_k]


# ─────────────────────────────────────────────────────────────────────────────
#  POST /ml/match  — single mentor-mentee pair score
# ─────────────────────────────────────────────────────────────────────────────

class PairMatchRequest(BaseModel):
    mentor_id:       int
    mentee_user_id:  int


@router.post("/match")
def match_pair(
    payload:      PairMatchRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Get match score for a specific mentor-mentee pair."""
    mentor  = db.query(MentorProfile).filter(MentorProfile.id == payload.mentor_id).first()
    mentee  = db.query(MenteeProfile).filter(MenteeProfile.user_id == payload.mentee_user_id).first()

    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")
    if not mentee:
        raise HTTPException(status_code=404, detail="Mentee profile not found")

    model_loaded = load_model()
    score        = get_match_score(mentor, mentee)
    fv           = build_feature_vector(mentor, mentee)

    return {
        "mentor_id":     mentor.id,
        "mentor_name":   mentor.user.full_name,
        "mentee_id":     mentee.user_id,
        "match_score":   score,
        "match_type":    "ml_model" if (model_loaded and _model) else "rule_based",
        "features":      {k: round(v, 4) for k, v in fv.items()},
        "interpretation": (
            "Excellent match" if score >= 80
            else "Good match" if score >= 60
            else "Moderate match" if score >= 40
            else "Low match"
        ),
    }

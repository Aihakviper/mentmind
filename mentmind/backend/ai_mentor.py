"""
AI Mentor Chat routes
POST /ai-mentor/chat          - send a message, get AI response
GET  /ai-mentor/history       - get conversation history
DELETE /ai-mentor/history     - clear conversation history
GET  /ai-mentor/suggestions   - get quick suggestion prompts
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from decouple import config
import asyncio
import json
import urllib.error
import urllib.request

from database import Base, get_db
from models import User, MenteeProfile, MentorProfile
from dependencies import get_current_user

router = APIRouter(prefix="/ai-mentor", tags=["AI Mentor Chat"])

# OpenAI API config
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_API_KEY = config("OPENAI_API_KEY", default="")
OPENAI_MODEL = config("OPENAI_MODEL", default="gpt-5.2")


# ─────────────────────────────────────────────────────────────────────────────
#  DB Model — store chat history per user
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    role       = Column(String(20), nullable=False)   # "user" | "assistant"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    time_ago: str

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    message: ChatMessageOut
    reply:   ChatMessageOut


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def time_ago(dt: datetime) -> str:
    now  = datetime.now(timezone.utc)
    diff = now - (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt)
    s    = int(diff.total_seconds())
    if s < 5:     return "just now"
    if s < 60:    return f"{s}s ago"
    if s < 3600:  return f"{s//60}m ago"
    return f"{s//3600}h ago"


def build_system_prompt(user: User, db: Session) -> str:
    """Build a personalised system prompt based on the user's profile."""
    name  = user.full_name.split()[0]
    role  = user.role
    extra = ""

    if role == "mentee":
        try:
            p = db.query(MenteeProfile).filter(MenteeProfile.user_id == user.id).first()
            if p:
                interests = ", ".join(p.areas_of_interest or []) or "general topics"
                level     = p.current_level or "beginner"
                extra     = f"The mentee's level is {level} and their interests are: {interests}."
        except Exception:
            pass

    elif role == "mentor":
        try:
            p = db.query(MentorProfile).filter(MentorProfile.user_id == user.id).first()
            if p:
                areas = ", ".join(p.expertise_areas or []) or "general mentorship"
                extra = f"The mentor's expertise areas are: {areas}."
        except Exception:
            pass

    return f"""You are an AI Mentor Assistant for MentMinds, an AI-powered mentorship platform
for Nigerian youth and professionals. You help users with career advice, skill development,
learning paths, and connecting with mentors.

The user's name is {name} and their role is {role}. {extra}

Key guidelines:
- Be warm, encouraging, and culturally aware (Nigerian / African context)
- Give concrete, actionable advice with numbered steps where helpful
- Reference MentMinds features: Classrooms, Resources, Mentor Discovery, Opportunities Hub, Forum
- Keep responses focused and practical — no more than 3–4 short paragraphs
- When recommending resources, suggest checking the Resources or Classrooms section
- End responses with a follow-up question to keep the conversation going"""


def get_recent_history(user_id: int, db: Session, limit: int = 10) -> list[dict]:
    """Get the last N messages for this user to send as context."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    # Return in chronological order
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


# ─────────────────────────────────────────────────────────────────────────────
#  POST /ai-mentor/chat
# ─────────────────────────────────────────────────────────────────────────────

def extract_openai_text(data: dict) -> Optional[str]:
    """Pull text from a Responses API payload."""
    if data.get("output_text"):
        return data["output_text"].strip()

    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text and content.get("type") in {"output_text", "text"}:
                chunks.append(text)
    return "\n".join(chunks).strip() or None


def _post_openai_response(system_prompt: str, history: list[dict]) -> Optional[str]:
    body = json.dumps({
        "model": OPENAI_MODEL,
        "instructions": system_prompt,
        "input": history,
        "max_output_tokens": 800,
    }).encode("utf-8")

    request = urllib.request.Request(
        OPENAI_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    return extract_openai_text(data)


async def call_openai_ai_mentor(system_prompt: str, history: list[dict]) -> Optional[str]:
    """Call OpenAI and return the assistant text, or None so callers can fallback."""
    if not OPENAI_API_KEY:
        return None

    return await asyncio.to_thread(_post_openai_response, system_prompt, history)


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def chat(
    payload:      ChatRequest,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """
    Send a message to the AI Mentor and get a response.
    Maintains conversation history for context.
    """
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 1. Save the user message
    user_msg = ChatMessage(
        user_id = current_user.id,
        role    = "user",
        content = payload.message.strip(),
    )
    db.add(user_msg)
    db.flush()

    # 2. Build conversation history for OpenAI
    history = get_recent_history(current_user.id, db, limit=12)
    # history already includes the message we just saved
    if not history or history[-1]["content"] != payload.message.strip():
        history.append({"role": "user", "content": payload.message.strip()})

    # 3. Call OpenAI API
    system_prompt = build_system_prompt(current_user, db)

    try:
        ai_text = await call_openai_ai_mentor(system_prompt, history)
        if not ai_text:
            ai_text = _fallback_response(payload.message, current_user)
    except Exception:
        # Network error — use fallback
        ai_text = _fallback_response(payload.message, current_user)

    # 4. Save AI response
    ai_msg = ChatMessage(
        user_id = current_user.id,
        role    = "assistant",
        content = ai_text,
    )
    db.add(ai_msg)
    db.commit()
    db.refresh(user_msg)
    db.refresh(ai_msg)

    return ChatResponse(
        message = ChatMessageOut(
            id         = user_msg.id,
            role       = user_msg.role,
            content    = user_msg.content,
            created_at = user_msg.created_at,
            time_ago   = time_ago(user_msg.created_at),
        ),
        reply = ChatMessageOut(
            id         = ai_msg.id,
            role       = ai_msg.role,
            content    = ai_msg.content,
            created_at = ai_msg.created_at,
            time_ago   = time_ago(ai_msg.created_at),
        ),
    )


def _fallback_response(message: str, user: User) -> str:
    """Smart fallback when OpenAI is unavailable."""
    name = user.full_name.split()[0]
    msg  = message.lower()

    if any(w in msg for w in ["career", "job", "transition", "switch"]):
        return (
            f"Great question, {name}! Career transitions take intentional planning. "
            "Here's what I recommend:\n\n"
            "1. **Identify your transferable skills** — list everything from your current role\n"
            "2. **Research your target role** — check job postings for required skills\n"
            "3. **Close the skill gap** — use MentMinds Classrooms for structured learning\n"
            "4. **Connect with a mentor** — find someone already in your target role in Mentor Discovery\n\n"
            "What specific career path are you targeting? That'll help me give more tailored advice."
        )
    elif any(w in msg for w in ["skill", "learn", "study", "course"]):
        return (
            f"Learning the right skills is key to growth, {name}! Based on current market trends:\n\n"
            "1. **Technical skills** — Python, data analysis, UI/UX design, and cloud are in high demand\n"
            "2. **Soft skills** — communication, stakeholder management, and leadership\n"
            "3. **Domain knowledge** — deep expertise in your chosen field\n\n"
            "Check out the **Classrooms** section — we have structured courses for all these. "
            "Which area interests you most?"
        )
    elif any(w in msg for w in ["mentor", "connect", "find"]):
        return (
            f"Finding the right mentor is a game-changer, {name}! Here's how to get the most from it:\n\n"
            "1. **Use Mentor Discovery** — our AI matches you based on your goals and interests\n"
            "2. **Be specific in your request** — mention your goals and what you need help with\n"
            "3. **Come prepared** — have questions ready for your first session\n\n"
            "Would you like me to suggest what to look for in a mentor based on your goals?"
        )
    elif any(w in msg for w in ["resource", "book", "material", "guide"]):
        return (
            f"Great initiative, {name}! The **Resources** section has materials shared by your mentors — "
            "PDFs, video recordings, templates and more.\n\n"
            "For self-study, I'd also recommend:\n"
            "1. The courses in **Classrooms** — structured and mentor-guided\n"
            "2. Discussions in the **Community Forum** — real questions and answers\n"
            "3. Opportunities in the **Opportunities Hub** — apply what you learn\n\n"
            "What specific topic are you looking to learn about?"
        )
    else:
        return (
            f"Thanks for reaching out, {name}! I'm your AI Mentor here on MentMinds. "
            "I can help you with:\n\n"
            "- 📈 **Career advice** and transition planning\n"
            "- 🎓 **Learning paths** and skill development\n"
            "- 👥 **Mentor matching** and connection tips\n"
            "- 🌟 **Opportunities** — internships, scholarships, workshops\n\n"
            "What would you like to work on today?"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /ai-mentor/history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/history", response_model=list[ChatMessageOut])
def get_history(
    limit:        int     = 50,
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Get the user's full chat history with the AI mentor."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return [
        ChatMessageOut(
            id         = m.id,
            role       = m.role,
            content    = m.content,
            created_at = m.created_at,
            time_ago   = time_ago(m.created_at),
        )
        for m in messages
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  DELETE /ai-mentor/history
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(
    current_user: User    = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Clear all chat history for the current user."""
    db.query(ChatMessage).filter(ChatMessage.user_id == current_user.id).delete()
    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  GET /ai-mentor/suggestions
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/suggestions")
def get_suggestions(
    current_user: User = Depends(get_current_user),
    db:           Session = Depends(get_db),
):
    """Return quick prompt suggestions based on user's role."""
    if current_user.role == "mentor":
        return {"suggestions": [
            "How can I be a more effective mentor?",
            "Tips for running productive 1-on-1 sessions",
            "How to track my mentees' progress",
            "Career Roadmap",
            "Resume Review",
        ]}

    return {"suggestions": [
        "Learning Resources",
        "Schedule with Mentor",
        "Career Roadmap",
        "Resume Review",
        "How to transition to mid-level?",
        "What skills should I focus on?",
    ]}

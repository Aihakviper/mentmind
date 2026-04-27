"""
MentMinds FastAPI Application
Run with: uvicorn main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import engine
from models import Base
import auth, dashboard, mentors


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting MentMinds API...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables ready")
    yield
    print("👋 Shutting down...")


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
    allow_origins     = ["http://localhost:5500", "http://127.0.0.1:5500",
                         "http://localhost:3000", "http://localhost:8000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(auth.router,      prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(mentors.router,   prefix="/api")


@app.get("/", tags=["Health"])
def root():
    return {"status": "online", "message": "MentMinds API 🎓", "docs": "/docs"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

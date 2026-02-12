"""
Auth endpoints: register (invite-gated), login, and token verification.
"""

import logging
import os
import smtplib
import threading
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .database import (
    create_user,
    consume_invite_code,
    get_user_by_email,
    validate_invite_code,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

SECRET_KEY = os.environ.get("BFT_JWT_SECRET", "bft-dev-secret-change-in-prod")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72


# ── Models ────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    invite_code: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    token: str
    email: str


class UserResponse(BaseModel):
    email: str
    created_at: str


# ── Token helpers ─────────────────────────────────────────────────

def create_token(email: str) -> str:
    payload = {
        "sub": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(request: Request) -> str:
    """Extract and verify JWT from Authorization header. Returns email."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = auth_header.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


# ── Welcome Email ─────────────────────────────────────────────────

SMTP_SERVER = os.environ.get("IBKR_SMTP_SERVER")
SMTP_PORT = int(os.environ.get("IBKR_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("IBKR_SMTP_USER")
SMTP_PASSWORD = os.environ.get("IBKR_SMTP_PASSWORD")


def _send_welcome_email(to_email: str):
    """Send welcome email in background thread (non-blocking)."""
    if not all([SMTP_SERVER, SMTP_USER, SMTP_PASSWORD]):
        logger.debug("SMTP not configured — skipping welcome email")
        return

    def _send():
        try:
            msg = MIMEMultipart()
            msg["From"] = SMTP_USER
            msg["To"] = to_email
            msg["Subject"] = "Welcome to Bar Fly Trading & Investing"
            msg.attach(MIMEText(
                f"Welcome to Bar Fly Trading & Investing!\n\n"
                f"Your account ({to_email}) is now active.\n\n"
                f"Visit https://www.barflytrading.com to explore S&P 500 sectors, "
                f"view technical analysis, and build your custom watchlist.\n\n"
                f"This is a beta — we'd love your feedback.\n\n"
                f"— The BFT Team",
                "plain",
            ))
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            logger.info(f"Welcome email sent to {to_email}")
        except Exception as e:
            logger.error(f"Failed to send welcome email to {to_email}: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    if not validate_invite_code(req.invite_code):
        raise HTTPException(400, "Invalid or expired invite code")

    if get_user_by_email(req.email):
        raise HTTPException(409, "Email already registered")

    pw_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    create_user(req.email, pw_hash)
    consume_invite_code(req.invite_code, req.email)

    token = create_token(req.email.lower())
    logger.info(f"New user registered: {req.email}")
    _send_welcome_email(req.email.lower())
    return TokenResponse(token=token, email=req.email.lower())


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(401, "Invalid credentials")

    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        raise HTTPException(401, "Invalid credentials")

    token = create_token(user["email"])
    return TokenResponse(token=token, email=user["email"])


@router.get("/me", response_model=UserResponse)
def get_me(request: Request):
    email = get_current_user(request)
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(401, "User not found")
    return UserResponse(email=user["email"], created_at=user["created_at"])

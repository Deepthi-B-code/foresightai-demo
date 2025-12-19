import hashlib
from fastapi import HTTPException
from sqlmodel import select
from .db import get_session
from .models import User

def hash_password(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

def verify_password(p: str, h: str) -> bool:
    return hash_password(p) == h

def authenticate_user(username: str, password: str) -> User:
    with get_session() as session:
        user = session.exec(select(User).where(User.username == username)).first()
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return user

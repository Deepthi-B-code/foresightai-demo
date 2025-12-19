from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .routes import router

# Initialize DB
init_db()

app = FastAPI(title="Foresight AI – Groq Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="backend/static", html=True), name="frontend")

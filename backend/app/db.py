from sqlmodel import SQLModel, create_engine, Session

# Local SQLite DB
engine = create_engine("sqlite:///insightforge_groq.db", echo=False)

def init_db():
    from . import models  # ensure models imported
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)

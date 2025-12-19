from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from sqlmodel import select
from .db import get_session
from .models import User
from .auth import hash_password, authenticate_user
from .rag_demo import (
    extract_text_from_upload,
    add_document_and_chunks,
    list_documents,
    delete_document,
    retrieve_relevant_chunks,
    build_context_with_citations,
    chat_answer,
    summarize_document,
    generate_quiz_for_document,
)

router = APIRouter(prefix="/api")

@router.post("/signup")
async def signup(request: Request):
    data = await request.json()
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    with get_session() as session:
        existing = session.exec(select(User).where(User.username == username)).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        role = "admin" if username.lower() == "admin" else "user"
        user = User(username=username, password_hash=hash_password(password), role=role)
        session.add(user)
        session.commit()
        session.refresh(user)
        return {"success": True, "role": user.role}

@router.post("/login")
async def login(request: Request):
    data = await request.json()
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    user = authenticate_user(username, password)
    return {"success": True, "role": user.role, "username": user.username}

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="File required")
    text = extract_text_from_upload(file)
    if not text or len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Could not extract readable text from this file.")
    chunks = add_document_and_chunks(file.filename, text)
    return {"success": True, "chunks": chunks}

@router.get("/docs")
async def get_docs():
    docs = list_documents()
    return {"documents": docs}

@router.delete("/docs/{doc_id}")
async def delete_doc(doc_id: int):
    delete_document(doc_id)
    return {"success": True}

@router.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    top_chunks = retrieve_relevant_chunks(question, top_k=5)
    context = build_context_with_citations(top_chunks)
    if not context:
        context = "(No document context available. Answer from general knowledge.)"
    prompt = f"""You are Foresight AI, an internal knowledge assistant.

Use the context below as your main source of truth. Cite evidence from context when possible.

Context:
{context}

Question:
{question}

Answer in 4-7 clear bullet points, professional but friendly tone:
"""
    answer = chat_answer(prompt)
    citations = [
        {"chunk_id": ch.id, "document_id": ch.document_id, "score": float(score), "preview": ch.content[:180]}
        for ch, score in top_chunks
    ]
    return {"answer": answer, "citations": citations}

@router.get("/docs/{doc_id}/summary")
async def doc_summary(doc_id: int):
    summary = summarize_document(doc_id)
    return {"summary": summary}

@router.get("/docs/{doc_id}/quiz")
async def doc_quiz(doc_id: int):
    quiz = generate_quiz_for_document(doc_id)
    return {"quiz": quiz}

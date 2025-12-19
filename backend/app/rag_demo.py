# rag_demo.py
# Demo-only implementations for freelancing / preview
# No API keys required

from typing import List, Tuple


# ---------- Upload & Document Handling (DEMO) ----------

def extract_text_from_upload(file):
    return "This is demo extracted text from the uploaded document."


def add_document_and_chunks(filename: str, text: str):
    return [
        {"chunk_id": 1, "content": "Demo chunk content from document."}
    ]


def list_documents():
    return [
        {"id": 1, "name": "Demo Document.pdf"}
    ]


def delete_document(doc_id: int):
    return True


# ---------- Retrieval & Context (DEMO) ----------

class DemoChunk:
    def __init__(self, id, document_id, content):
        self.id = id
        self.document_id = document_id
        self.content = content


def retrieve_relevant_chunks(question: str, top_k: int = 5) -> List[Tuple[DemoChunk, float]]:
    chunks = [
        DemoChunk(1, 1, "This is a demo knowledge chunk related to the question.")
    ]
    return [(chunks[0], 0.95)]


def build_context_with_citations(chunks_with_scores):
    return "This is demo contextual information built from uploaded documents."


# ---------- AI Responses (DEMO) ----------

def chat_answer(prompt: str):
    return (
        "• This is a demo answer.\n"
        "• It shows how the AI response will look.\n"
        "• No external API is used.\n"
        "• Full AI integration can be enabled later.\n"
    )


def summarize_document(doc_id: int):
    return "This is a demo summary of the selected document."


def generate_quiz_for_document(doc_id: int):
    return [
        {
            "question": "What is this project about?",
            "options": ["AI demo", "Game", "Music App", "Chat App"],
            "answer": "AI demo",
        }
    ]

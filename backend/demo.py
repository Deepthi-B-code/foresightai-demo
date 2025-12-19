import gradio as gr
import requests

# Your FastAPI backend running locally
BACKEND_URL = "http://127.0.0.1:8000"

def upload_doc(file):
    try:
        files = {"file": (file.name, file)}
        r = requests.post(f"{BACKEND_URL}/upload-document", files=files)
        return r.text
    except Exception as e:
        return f"Error: {e}"

def ask_question(query):
    try:
        r = requests.post(f"{BACKEND_URL}/ask", json={"query": query})
        if r.status_code == 200:
            return r.json().get("answer", "No answer found.")
        return f"Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="ForesightAI Demo") as demo:
    gr.Markdown("# 📘 ForesightAI – Knowledge Base Agent Demo")
    gr.Markdown("Upload documents and ask questions using your local FastAPI backend.")

    with gr.Tab("📄 Upload Document"):
        file_input = gr.File(label="Upload PDF, DOCX, or TXT")
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Status")
        upload_btn.click(upload_doc, inputs=file_input, outputs=upload_output)

    with gr.Tab("❓ Ask a Question"):
        question = gr.Textbox(label="Enter your question")
        ask_btn = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer")
        ask_btn.click(ask_question, inputs=question, outputs=answer_output)

demo.launch(share=True)

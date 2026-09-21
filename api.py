from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

app = FastAPI()

embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=0)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

class Question(BaseModel):
    question: str

@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name": "RAG Equipment Agent",
        "description": "Répond aux questions sur les pannes équipements",
        "endpoint": "http://localhost:8000/ask",
        "input_schema": {"question": "string"},
        "output_schema": {"reponse": "string"}
    }

@app.post("/ask")
def ask(body: Question):
    docs = vectorstore.similarity_search(body.question, k=3)
    contexte = "\n\n".join([d.page_content for d in docs])
    prompt = f"""Réponds uniquement à partir du contexte suivant.

Contexte :
{contexte}

Question : {body.question}
Réponse :"""
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"num_gpu": 0}
    )
    return {"reponse": response["message"]["content"]}
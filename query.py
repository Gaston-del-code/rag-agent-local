from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

# Connexion à la base vectorielle
embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=0)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Boucle de questions
print("Agent RAG prêt. Tape 'exit' pour quitter.\n")
while True:
    question = input("Ta question : ")
    if question.lower() == "exit":
        break

    # Retrieval : trouver les chunks pertinents
    docs = vectorstore.similarity_search(question, k=3)
    contexte = "\n\n".join([d.page_content for d in docs])

    # Génération : envoyer à Mistral
    prompt = f"""Tu es un assistant expert. Réponds uniquement à partir du contexte suivant.

Contexte :
{contexte}

Question : {question}
Réponse :"""

    response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": prompt}],
    options={"num_gpu": 0}
)
    print(f"\nRéponse : {response['message']['content']}\n")
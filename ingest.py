from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Chargement du PDF
print("Chargement du PDF...")
loader = PyPDFLoader("03_Equipment_Failure_Corrective_Action_Guide.pdf")
documents = loader.load()

# Découpage en chunks
print("Découpage en chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"{len(chunks)} chunks créés")

# Embeddings + stockage dans ChromaDB
print("Création des embeddings (peut prendre quelques minutes)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=0)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✓ Base vectorielle créée dans ./chroma_db")
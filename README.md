# Local RAG Agent — 100% On-Premise

Agent RAG entièrement local, sans dépendance cloud.

## Stack
- **LLM** : Mistral 7B via Ollama (localhost:11434)
- **Embeddings** : nomic-embed-text (Ollama)
- **Vector store** : ChromaDB (persisté sur disque)
- **Orchestration** : LangChain
- **REST API** : FastAPI (localhost:8000)

## Architecture
PDF → Chunking → Embeddings → ChromaDB
Question → Embedding → Similarité → Top-3 chunks → Prompt → Mistral → Réponse


## Cas d'usage
Idéal pour les secteurs réglementés (santé, finance, RH) où les données ne peuvent pas quitter le SI.
Appelable depuis n'importe quel orchestrateur métier (Pega, SAP, etc.) via REST.

## REST API

Lancer le serveur :
```bash
python -m uvicorn api:app --port 8000
```

Appel :
```bash
POST http://localhost:8000/ask
Content-Type: application/json

{"question": "What should I do if GC-204 has a bearing wear issue?"}
```

Réponse :
```json
{"reponse": "..."}
```

## ⚠️ Mode CPU (configuration actuelle)
Ce projet tourne sur **CPU** (RAM 16 Go). Mistral 7B requiert ~5-6 Go de VRAM — incompatible avec les GPUs < 6 Go (ex. GTX 1050 4 Go).

### Passer en mode GPU (carte ≥ 6 Go VRAM)

**Quand le recommander :**
- GPU avec ≥ 6 Go VRAM (ex. RTX 3060, RTX 4070)
- Volumes importants ou besoin de faible latence (2-5s vs 30-60s en CPU)

**Quand rester en CPU :**
- GPU < 6 Go VRAM → crash garanti
- Usage occasionnel où la latence n'est pas critique

Supprimer les paramètres `num_gpu=0` dans `ingest.py` et `query.py` — Ollama détecte automatiquement le GPU.

## Lancer le projet
```bash
# 1. Indexer le document (une seule fois)
python ingest.py

# 2. Mode CLI interactif
python query.py

# 3. Mode API REST
python -m uvicorn api:app --port 8000
```

## Prérequis
- Ollama : `ollama pull mistral && ollama pull nomic-embed-text`
- Python 3.10+ :
```bash
pip install langchain langchain-community langchain-ollama langchain-text-splitters chromadb pypdf ollama fastapi uvicorn
```
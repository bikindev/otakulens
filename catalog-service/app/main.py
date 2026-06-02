# otakulens catalog-service/app/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.vector_store import popular_banco_vetorial_com_json, buscar_animes_similares

app = FastAPI(
    title="OtakuLens - Catalog Service",
    description="Microsserviço de Catálogo de Animes - Lado de Leitura (CQRS / RAG)"
)

DATA_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_animes.json")

@app.on_event("startup")
def startup_event():
    print("Iniciando Catalog Service e sincronizando banco vetorial...")
    popular_banco_vetorial_com_json(DATA_JSON_PATH)

# Schemas do Pydantic para garantir a tipagem e validação de dados e evitar erros de comunicação
# Contrato para fazer a busca RAG
class SearchQuery(BaseModel):
    prompt: str
    limite: int = 2

# Contrato de reposta do serviço
class AnimeResponseItem(BaseModel):
    id_anime: int
    titulo: str
    ano: int
    generos: list[str] 
    plataformas: list[str]
    trecho_contexto: str

@app.post("/catalog/search-semantic", response_model=list[AnimeResponseItem])
def search_semantic(query_data: SearchQuery):
    if not query_data.prompt:
        raise HTTPException(status_code=400, detail="O prompt de busca não pode estar vazio.")
    
    docs_encontrados = buscar_animes_similares(query_data.prompt, k=query_data.limite)
    
    response = []
    for doc in docs_encontrados:
        # Trata as strings do ChromaDB de volta para listas do Python
        lista_plataformas = [p.strip() for p in doc.metadata["plataformas"].split(",")]
        lista_generos = [g.strip() for g in doc.metadata["generos"].split(",")] if doc.metadata.get("generos") else []
        
        response.append(AnimeResponseItem(
            id_anime=doc.metadata["id_anime"],
            titulo=doc.metadata["titulo"],
            ano=doc.metadata["ano"],
            generos=lista_generos, 
            plataformas=lista_plataformas,
            trecho_contexto=doc.page_content
        ))
        
    return response
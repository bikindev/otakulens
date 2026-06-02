# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.llm_chain import gerar_recomendacao_rag

# Importa o protótipo MCP
import os
import sys

# 1. Encontra o caminho absoluto da raiz do projeto (OtakuLens)
RAIZ_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Injeta a raiz temporariamente no sistema de busca de pacotes do Python
if RAIZ_PROJETO not in sys.path:
    sys.path.append(RAIZ_PROJETO)

# 3. Agora fazemos o import navegando a partir da raiz (com um truque para o hífen)
# Como a pasta tem hífen, o Python não aceita o import estático. 
# Usamos o import dinâmico nativo do Python:
import importlib
mcp_server_module = importlib.import_module("mcp-prototype.app.mcp_server")
OtakuLensMCPServer = mcp_server_module.OtakuLensMCPServer

# Inicializa o servidor MCP no escopo global do controlador de IA
mcp_server = OtakuLensMCPServer()

app = FastAPI(
    title="OtakuLens - AI Controller",
    description="Controlador Central de IA - Orquestrador do Pipeline RAG (Camada 2)"
)

# Contrato de entrada (vindo do Gateway / Frontend)
class RecommendationRequest(BaseModel):
    user_prompt: str

# Contrato de saída estruturado
class RecommendationResponse(BaseModel):
    recommendation: str

@app.post("/ai/recommend", response_model=RecommendationResponse)
def recommend_anime(request: RecommendationRequest):
    if not request.user_prompt.strip():
        raise HTTPException(status_code=400, detail="O prompt do usuário não pode estar vazio.")
    
    # Dispara a orquestração do RAG
    resposta_ia = gerar_recomendacao_rag(request.user_prompt, mcp_server=mcp_server)
    
    return RecommendationResponse(recommendation=resposta_ia)

if __name__ == "__main__":
    import uvicorn
    # Rodando na porta 8000 (o catalog service roda na 8001)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
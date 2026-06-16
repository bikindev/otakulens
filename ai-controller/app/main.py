# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.llm_chain import gerar_recomendacao_rag

app = FastAPI(
    title="OtakuLens - AI Controller",
    description="Controlador Central de IA - Orquestrador do Pipeline RAG + MCP (Camada 2)"
)

class RecommendationRequest(BaseModel):
    user_prompt: str

class RecommendationResponse(BaseModel):
    recommendation: str

@app.post("/ai/recommend", response_model=RecommendationResponse)
async def recommend_anime(request: RecommendationRequest):
    if not request.user_prompt.strip():
        raise HTTPException(status_code=400, detail="O prompt do usuário não pode estar vazio.")
    
    # Chama o pipeline que gerencia internamente as chamadas de rede para o RAG e MCP
    # e aguarda o pipeline assíncrono de ponta a ponta
    resposta_ia = await gerar_recomendacao_rag(request.user_prompt)
    return RecommendationResponse(recommendation=resposta_ia)

if __name__ == "__main__":
    import uvicorn
    # Rodando na porta 8000
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
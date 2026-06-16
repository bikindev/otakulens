# interface/backend.py
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OtakuLens - Backend Único (Síncrono)",
    description="Agregador estável e linear para a Interface Mobile/Web"
)

# Libera CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_CONTROLLER_URL = "http://127.0.0.1:8000/ai/recommend"

class UserPromptRequest(BaseModel):
    prompt: str

# Mudamos para 'def' comum (síncrono), eliminando o loop de Proactor assíncrono nesta rota
@app.post("/api/otakulens/recommend")
def processar_requisicao_usuario(request: UserPromptRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="O texto inserido não pode estar vazio.")
    
    payload = {"user_prompt": request.prompt}
    
    try:
        print(f"[BACKEND] Repassando prompt para o AI Controller na porta 8000...")
        
        # Fazemos uma chamada síncrona linear. O Windows manterá o socket aberto esperando.
        # Definimos um timeout de 5 minutos (300s) para o Llama3 processar com calma.
        response = requests.post(
            AI_CONTROLLER_URL,
            json=payload,
            timeout=300.0
        )
        
        if response.status_code == 200:
            print("[BACKEND] Resposta da IA recebida com sucesso!")
            return response.json()
        
        print(f"[BACKEND ERROR] AI Controller respondeu com status: {response.status_code}")
        raise HTTPException(
            status_code=response.status_code, 
            detail="Erro interno no processamento do controlador de IA."
        )
        
    except requests.exceptions.Timeout:
        print("[BACKEND ERROR] Timeout estourado aguardando o AI Controller (Ollama lento).")
        raise HTTPException(
            status_code=504,
            detail="A Inteligência Artificial demorou muito para responder. O modelo local pode estar sobrecarregado."
        )
    except Exception as e:
        print(f"[BACKEND ERROR] Falha crítica de conexão com o AI Controller: {e}")
        raise HTTPException(
            status_code=503, 
            detail="O serviço de Inteligência Artificial está temporariamente indisponível."
        )

if __name__ == "__main__":
    import uvicorn
    # Mantemos na porta 5000
    uvicorn.run("backend:app", host="127.0.0.1", port=5000, reload=True)
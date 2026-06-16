# app/llm_chain.py
import requests
import json
import asyncio
import anyio
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CATALOG_SERVICE_URL = "http://127.0.0.1:8001/catalog/search-semantic"
MCP_SERVER_SSE_URL = "http://127.0.0.1:8002/mcp/sse"
MCP_SERVER_MSG_URL = "http://127.0.0.1:8002/mcp/messages"

# Inicialização do LLM via Ollama
llm = Ollama(
    base_url="http://127.0.0.1:11434",
    model="llama3"
)

async def consultar_catalogo_vetorial(prompt_usuario: str, limite: int = 2) -> list:
    """Busca animes no catálogo."""
    payload = {"prompt": prompt_usuario, "limite": limite}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(CATALOG_SERVICE_URL, json=payload, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return []
    except Exception as e:
        print(f"[CATALOG ERROR] Erro ao conectar no Catalog Service: {e}")
        return []
    
async def buscar_links_mcp_via_http_direto(titulo_anime: str) -> str:
    """
    Simula nativamente o protocolo Model Context Protocol (MCP) via JSON-RPC
    enviando a requisição diretamente para o endpoint de mensagens HTTP do Starlette.
    """
    # Payload no formato exato que a especificação oficial do MCP / JSON-RPC espera
    mcp_payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_streaming_links",
            "arguments": {
                "anime_title": titulo_anime
            }
        },
        "id": 1 # Identificador padrão da requisição RPC
    }

    try:
        # Definimos um timeout confortável para a busca no ChromaDB local
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MCP_SERVER_MSG_URL, 
                json=mcp_payload, 
                headers={"Content-Type": "application/json"},
                timeout=120.0
            )
            
            if response.status_code == 200:
                mcp_response_json = response.json()
                
                # O protocolo MCP encapsula a resposta dentro de 'result' -> 'content'
                if "result" in mcp_response_json and "content" in mcp_response_json["result"]:
                    conteudo_texto = mcp_response_json["result"]["content"][0]["text"]
                    
                    if "Nenhum link direto" in conteudo_texto:
                        return "  - Links diretos de transmissão indisponíveis no momento.\n"
                    
                    # Faz o parse da string de links gerada pelo seu ChromaDB
                    links_lista = json.loads(conteudo_texto)
                    links_texto = "Links oficiais calculados pelo MCP:\n"
                    for item in links_lista:
                        links_texto += f"  - Na {item['plataforma']}: {item['url_direta']}\n"
                    return links_texto
                    
            return "  - Links detalhados indisponíveis no MCP.\n"
            
    except Exception as e:
        print(f"[MCP DIRECT HTTP ERROR] Falha ao invocar ferramenta via HTTP: {e}")
        return "  - Links temporariamente indisponíveis (Erro de Comunicação).\n"

async def pipeline_enriquecimento_mcp(animes_contexto: list) -> str:
    """Varre em lote todos os animes sugeridos montando o contexto."""
    contexto_formatado = ""
    
    for anime in animes_contexto:
        titulo = anime["titulo"]
        plataformas = ", ".join(anime["plataformas"])
        
        contexto_formatado += f"\n---\n"
        contexto_formatado += f"Título: {titulo} ({anime['ano']})\n"
        contexto_formatado += f"Disponibilidade Geral: {plataformas}\n"
        
        # Faz a chamada HTTP direta para o servidor MCP sem depender de SDK instável
        links_mcp = await buscar_links_mcp_via_http_direto(titulo)
        contexto_formatado += links_mcp
        
        contexto_formatado += f"Informações e Reviews: {anime['trecho_contexto']}\n"
        
    return contexto_formatado


def formatar_contexto_hibrido(animes_retornados: list) -> str:
    """Agrega os dados síncronos do catálogo com as buscas assíncronas do MCP."""
    contexto_formatado = ""
    for anime in animes_retornados:
        titulo = anime["titulo"]
        plataformas = ", ".join(anime["plataformas"])
        
        contexto_formatado += f"\n---\n"
        contexto_formatado += f"Título: {titulo} ({anime['ano']})\n"
        contexto_formatado += f"Disponibilidade Geral: {plataformas}\n"
        
        # --- ACIONAMENTO DO BARRAMENTO MCP VIA REDE (Roda o loop assíncrono para o cliente HTTP) ---
        links_mcp = asyncio.run(buscar_links_mcp_via_rede(titulo))
        contexto_formatado += links_mcp
        
        contexto_formatado += f"Informações e Reviews: {anime['trecho_contexto']}\n"
        
    return contexto_formatado

async def gerar_recomendacao_rag(prompt_usuario: str) -> str:
    """Pipeline Consolidado RAG + MCP 100% estável via requisições diretas."""
    
    # 1. RETRIEVAL
    animes_contexto = await consultar_catalogo_vetorial(prompt_usuario, limite=2)
    if not animes_contexto:
        return "Desculpe, estou com dificuldades para acessar meu catálogo de animes no momento."

    # 2. CONTEXT ENRICHMENT (Chamadas HTTP limpas)
    contexto_str = await pipeline_enriquecimento_mcp(animes_contexto)

    # 3. PROMPT ENGINEERING
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é o OtakuLens, um recomendador especialista em animes. "
            "Use estritamente o CONTEXTO fornecido abaixo para indicar animes ao usuário com base no que ele solicitou.\n"
            "O contexto contém URLs reais e diretas trazidas pelo protocolo distribuído MCP. "
            "É OBRIGATÓRIO incluir esses links textuais exatos na sua resposta para que o usuário possa clicar e assistir.\n\n"
            "Se o contexto fugir do escopo de sugestão de animes, como por exemplo o pedido de uma receita de hamburguer, não forneça a resposta, apenas diga que você é um modelo treinado apenas para sugerir animes.\n\n"
            "Responda de forma direta apenas o essencial, sem muita criatividade.\n\n"
            "Responda apenas com o nome do anime sugerido, com uma breve sinopse e os links correspondentes estruturados e nada além disso.\n\n"
            "Responda apenas em português. Se a sinopse existente for em inglês, traduza para o português.\n\n"
            "CONTEXTO DOS ANIMES E LINKS DE STREAMING (RAG + MCP):\n{contexto}"
        )),
        ("user", "{pergunta}")
    ])

    # 4. EXECUÇÃO DA CHAIN
    chain = prompt_template | llm | StrOutputParser()
    
    try:
        resultado = await chain.ainvoke({"contexto": contexto_str, "pergunta": prompt_usuario})
        return resultado
    except Exception as llm_err:
        print(f"[LLM GENERATION ERROR] Falha na resposta do Ollama: {llm_err}")
        return "O gerador de texto local demorou muito para responder. Por favor, tente novamente."
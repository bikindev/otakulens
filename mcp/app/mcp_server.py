# -*- coding: utf-8 -*-
"""
OtakuLens - Servidor MCP Oficial (Modo SSE / HTTP)
Componente: mcp_server.py
"""

import os
import json
import urllib.parse
import asyncio
import chromadb
from chromadb.utils import embedding_functions
from mcp.server import Server
import mcp.types as types
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# caminhos absolutos
BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_db"))
ANIMES_JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw_animes.json"))
MCP_CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "mcp_streaming.json"))

# Inicializa o objeto do Servidor MCP Oficial
server = Server("otakulens-mcp-server")

# Inicializa o ChromaDB Global
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

embedding_fn = embedding_functions.OllamaEmbeddingFunction(
    url="http://127.0.0.1:11434/api/embeddings",
    model_name="nomic-embed-text"
)

collection = chroma_client.get_or_create_collection(
    name="streaming_links_mcp_official_sse",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

def _seed_mock_data():
    """Popula o ChromaDB do MCP dinamicamente se estiver vazio."""
    if collection.count() > 0:
        return
        
    if not os.path.exists(ANIMES_JSON_PATH) or not os.path.exists(MCP_CONFIG_PATH):
        return

    with open(ANIMES_JSON_PATH, "r", encoding="utf-8") as f:
        catalog_animes = json.load(f)
    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        mcp_config = json.load(f)
        
    platforms_map = {p["name"]: p for p in mcp_config["platforms_config"]}
    ids, documents, metadatas = [], [], []
    
    for anime in catalog_animes:
        titulo = anime["titulo"]
        id_anime = anime["id_anime"]
        for plat_name in anime.get("plataformas", []):
            if plat_name in platforms_map:
                plat_info = platforms_map[plat_name]
                url_direta = f"{plat_info['base_url']}{urllib.parse.quote(titulo)}"
                
                ids.append(f"mcp_{id_anime}_{plat_name.lower()}")
                documents.append(f"Assistir {titulo} online na plataforma {plat_name}")
                metadatas.append({
                    "id_anime": id_anime,
                    "anime_title": titulo,
                    "platform": plat_name,
                    "url_direta": url_direta
                })
                
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

_seed_mock_data()

# --- RECURSOS (RESOURCES) ---
@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="anime://catalog/streamings",
            name="Plataformas de Streaming Suportadas",
            description="Configurações e URLs base das plataformas parceiras do OtakuLens.",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    if uri == "anime://catalog/streamings":
        if os.path.exists(MCP_CONFIG_PATH):
            with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
                conteudo_bruto = f.read()
            
            print(f"[MCP SERVER] Recurso lido com sucesso. Tamanho: {len(conteudo_bruto)} caracteres.")
            # Retorna a string bruta diretamente. Muitas versões do SDK preferem a string crua
            # e o próprio decorador se encarrega de envelopar no formato JSON-RPC.
            return conteudo_bruto
            
    raise ValueError(f"Recurso não encontrado: {uri}")

# --- FERRAMENTAS (TOOLS) ---
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_streaming_links",
            description="Busca links diretos de streaming (URL) para assistir a um anime específico com base no título.",
            inputSchema={
                "type": "object",
                "properties": {
                    "anime_title": { "type": "string", "description": "O nome ou título exato do anime." },
                    "preferred_platform": { "type": "string", "description": "Filtro opcional." }
                },
                "required": ["anime_title"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name == "search_streaming_links":
        if not arguments: 
            raise ValueError("Argumentos ausentes.")
        
        anime_title = arguments.get("anime_title")
        preferred_platform = arguments.get("preferred_platform")
        
        results = collection.query(query_texts=[f"Assistir {anime_title}"], n_results=3)
        output_links = []
        
        if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
            for meta in results['metadatas'][0]:
                if preferred_platform and preferred_platform.lower() not in meta['platform'].lower():
                    continue
                output_links.append({
                    "anime": meta["anime_title"],
                    "plataforma": meta["platform"],
                    "url_direta": meta["url_direta"]
                })
                
        if not output_links:
            # Envelopa usando a classe de conteúdo do tipo texto do SDK
            return [types.TextContent(type="text", text="Nenhum link direto encontrado no MCP.")]
            
        # Retorna o JSON serializado dentro do objeto de texto padrão do protocolo
        return [
            types.TextContent(
                type="text", 
                text=json.dumps(output_links, ensure_ascii=False, indent=2)
            )
        ]
        
    raise ValueError(f"Ferramenta desconhecida: {name}")

# --- INSTANCIAÇÃO DO SERVIDOR WEB SSE ---
sse = SseServerTransport("/mcp/messages")

async def handle_sse(request):
    """Endpoint onde o cliente se conecta para receber o stream de eventos SSE."""
    async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        # Inicialização simplificada e direta sem usar classes de configurações mutáveis
        await server.run(read_stream, write_stream, server.create_initialization_options())

async def handle_messages(request):
    """Endpoint post para onde o cliente envia os comandos/mensagens."""
    await sse.handle_post_message(request.scope, request.receive, request._send)

# Cria a aplicação web usando Starlette (nativa, leve e assíncrona)
starlette_app = Starlette(
    routes=[
        Route("/mcp/sse", endpoint=handle_sse, methods=["GET"]),
        Route("/mcp/messages", endpoint=handle_messages, methods=["POST"]),
    ],
    # Adicione propriedade CORS para liberar as requisições OPTIONS:
    middleware=[
        Middleware(
            CORSMiddleware, 
            allow_origins=["*"], 
            allow_methods=["*"], 
            allow_headers=["*"]
        )
    ]
)

if __name__ == "__main__":
    import uvicorn
    print("[MCP SSE SERVER] Inicializando servidor HTTP na porta 8002...")
    uvicorn.run(starlette_app, host="127.0.0.1", port=8002)
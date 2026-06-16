# -*- coding: utf-8 -*-
"""
OtakuLens - Servidor MCP (Modo SSE / HTTP)
Componente: mcp_server.py

CORREÇÃO PRINCIPAL:
  Adicionada rota HTTP direta /tools/search que não depende de sessão SSE.
  O llm_chain.py passa a chamar essa rota — resolvendo o problema de links
  não retornados quando o JSON-RPC era enviado ao /mcp/messages sem sessão ativa.

CORREÇÃO SECUNDÁRIA:
  A busca no ChromaDB agora usa filtro de metadados exato (where=) em vez de
  similaridade semântica, garantindo que cada anime retorne SEUS PRÓPRIOS links.
"""

import os
import json
import urllib.parse
import chromadb
from chromadb.utils import embedding_functions
from mcp.server import Server
import mcp.types as types
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Caminhos absolutos
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(__file__)
CHROMA_DIR      = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_db"))
ANIMES_JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw_animes.json"))
MCP_CONFIG_PATH  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "mcp_streaming.json"))

# ---------------------------------------------------------------------------
# Servidor MCP (SSE — mantido para conformidade com o protocolo)
# ---------------------------------------------------------------------------
server = Server("otakulens-mcp-server")

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Seed dos dados mockados
# ---------------------------------------------------------------------------
def _seed_mock_data():
    """Popula o ChromaDB do MCP se estiver vazio."""
    if collection.count() > 0:
        return

    if not os.path.exists(ANIMES_JSON_PATH) or not os.path.exists(MCP_CONFIG_PATH):
        print("[MCP SERVER] AVISO: Arquivos de dados não encontrados. ChromaDB ficará vazio.")
        return

    with open(ANIMES_JSON_PATH, "r", encoding="utf-8") as f:
        catalog_animes = json.load(f)
    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        mcp_config = json.load(f)

    platforms_map = {p["name"]: p for p in mcp_config["platforms_config"]}
    ids, documents, metadatas = [], [], []

    for anime in catalog_animes:
        titulo   = anime["titulo"]
        id_anime = anime["id_anime"]
        for plat_name in anime.get("plataformas", []):
            if plat_name not in platforms_map:
                continue
            plat_info  = platforms_map[plat_name]
            # quote_plus é mais compatível com parâmetros de busca (?q=)
            url_direta = f"{plat_info['base_url']}{urllib.parse.quote_plus(titulo)}"

            ids.append(f"mcp_{id_anime}_{plat_name.lower()}")
            documents.append(f"Assistir {titulo} online na plataforma {plat_name}")
            metadatas.append({
                "id_anime":    id_anime,
                "anime_title": titulo,
                "platform":    plat_name,
                "url_direta":  url_direta
            })

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[MCP SERVER] ChromaDB populado com {len(ids)} registros de streaming.")

_seed_mock_data()

# ---------------------------------------------------------------------------
# Handlers SSE (mantidos para conformidade com o protocolo MCP)
# ---------------------------------------------------------------------------
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
                return f.read()
    raise ValueError(f"Recurso não encontrado: {uri}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_streaming_links",
            description="Busca links diretos de streaming para um anime específico.",
            inputSchema={
                "type": "object",
                "properties": {
                    "anime_title": {
                        "type": "string",
                        "description": "Título exato do anime."
                    },
                    "preferred_platform": {
                        "type": "string",
                        "description": "Filtro opcional de plataforma."
                    }
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
        resultado = _buscar_links_no_chroma(
            anime_title=arguments.get("anime_title"),
            preferred_platform=arguments.get("preferred_platform")
        )
        return [types.TextContent(type="text", text=json.dumps(resultado, ensure_ascii=False))]
    raise ValueError(f"Ferramenta desconhecida: {name}")

# ---------------------------------------------------------------------------
# Função de busca centralizada — usada tanto pela rota SSE quanto pela direta
# ---------------------------------------------------------------------------
def _buscar_links_no_chroma(anime_title: str, preferred_platform: str = None) -> list:
    """
    Busca links no ChromaDB usando filtro de metadados exato pelo título.

    MOTIVO DA MUDANÇA:
      A busca semântica anterior (query_texts) podia retornar animes similares
      ao invés do anime solicitado — ex: buscar "Jujutsu Kaisen" e receber links
      de "Kimetsu no Yaiba" por proximidade semântica.
      Com where= garantimos que só retornam links do anime exato requisitado.
    """
    output_links = []
    try:
        results = collection.get(
            where={"anime_title": anime_title},
        )

        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                if preferred_platform:
                    if preferred_platform.lower() not in meta["platform"].lower():
                        continue
                output_links.append({
                    "anime":      meta["anime_title"],
                    "plataforma": meta["platform"],
                    "url_direta": meta["url_direta"]
                })

        # Fallback: se o título exato não foi encontrado, tenta busca semântica
        if not output_links:
            print(f"[MCP SERVER] Título exato '{anime_title}' não encontrado. Tentando busca semântica...")
            sem_results = collection.query(
                query_texts=[f"Assistir {anime_title}"],
                n_results=3
            )
            if sem_results and sem_results.get("metadatas") and sem_results["metadatas"][0]:
                for meta in sem_results["metadatas"][0]:
                    if preferred_platform:
                        if preferred_platform.lower() not in meta["platform"].lower():
                            continue
                    output_links.append({
                        "anime":      meta["anime_title"],
                        "plataforma": meta["platform"],
                        "url_direta": meta["url_direta"]
                    })

    except Exception as e:
        print(f"[MCP SERVER] Erro na busca ChromaDB: {e}")

    return output_links

# ---------------------------------------------------------------------------
# ROTA HTTP DIRETA — /tools/search
# ---------------------------------------------------------------------------
async def handle_direct_tool_search(request: Request) -> JSONResponse:
    """
    Rota HTTP direta que NÃO depende de sessão SSE.

    Recebe: { "anime_title": "...", "preferred_platform": "..." (opcional) }
    Retorna: { "links": [ { "anime": ..., "plataforma": ..., "url_direta": ... } ] }

    Esta é a rota que o llm_chain.py deve chamar — simples, estável e sem
    dependência de estado de sessão do protocolo SSE.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"erro": "Body JSON inválido."}, status_code=400)

    anime_title = body.get("anime_title", "").strip()
    if not anime_title:
        return JSONResponse({"erro": "Campo 'anime_title' é obrigatório."}, status_code=400)

    preferred_platform = body.get("preferred_platform")
    links = _buscar_links_no_chroma(anime_title, preferred_platform)

    print(f"[MCP SERVER] /tools/search → '{anime_title}' → {len(links)} link(s) encontrado(s).")
    return JSONResponse({"links": links})

# ---------------------------------------------------------------------------
# SSE Transport (mantido)
# ---------------------------------------------------------------------------
sse = SseServerTransport("/mcp/messages")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

async def handle_messages(request):
    await sse.handle_post_message(request.scope, request.receive, request._send)

# ---------------------------------------------------------------------------
# Aplicação Starlette
# ---------------------------------------------------------------------------
starlette_app = Starlette(
    routes=[
        # Rota SSE original (mantida)
        Route("/mcp/sse",      endpoint=handle_sse,               methods=["GET"]),
        Route("/mcp/messages", endpoint=handle_messages,           methods=["POST"]),
        # Nova rota HTTP direta — usada pelo llm_chain.py
        Route("/tools/search", endpoint=handle_direct_tool_search, methods=["POST"]),
    ],
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
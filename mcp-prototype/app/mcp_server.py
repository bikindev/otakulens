# -*- coding: utf-8 -*-
"""
OtakuLens - MCP Server Prototype
Componente: mcp_server.py
"""

import os
import json
import urllib.parse
import chromadb
# Voltando para a importação nativa oficial do Chroma + Ollama
from chromadb.utils import embedding_functions

# Estratégia de caminhos absolutos baseada no seu vector_store.py
BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_db"))
ANIMES_JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw_animes.json"))
MCP_CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "mcp_streaming.json"))

class OtakuLensMCPServer:
    def __init__(self):
        print("[MCP SERVER] Inicializando servidor de contexto do OtakuLens...")
        
        # Garante a criação da pasta física do banco do MCP
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Utilizando a função de embedding oficial do pacote 'ollama'
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text"
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="streaming_links_mcp_native",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Popula a coleção se ela estiver vazia
        if self.collection.count() == 0:
            self._seed_mock_data()
        else:
            print(f"[MCP SERVER] Base vetorial nativa já possui {self.collection.count()} registros. Carga pulada.")

    def _seed_mock_data(self):
        print("[MCP SERVER] Gerando base vetorial nativa do MCP a partir do raw_animes.json...")
        
        if not os.path.exists(ANIMES_JSON_PATH):
            print(f"[MCP SERVER] [ERRO] Arquivo de catálogo não encontrado em: {ANIMES_JSON_PATH}")
            return
            
        if not os.path.exists(MCP_CONFIG_PATH):
            print(f"[MCP SERVER] [ERRO] Arquivo de configuração não encontrado em: {MCP_CONFIG_PATH}")
            return

        with open(ANIMES_JSON_PATH, "r", encoding="utf-8") as f:
            catalog_animes = json.load(f)
            
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            mcp_config = json.load(f)
            
        platforms_map = {p["name"]: p for p in mcp_config["platforms_config"]}
        
        ids = []
        documents = []
        metadatas = []
        
        counter = 0
        for anime in catalog_animes:
            titulo = anime["titulo"]
            id_anime = anime["id_anime"]
            plataformas_disponiveis = anime.get("plataformas", [])
            
            for plat_name in plataformas_disponiveis:
                if plat_name in platforms_map:
                    counter += 1
                    plat_info = platforms_map[plat_name]
                    
                    # Gera uma URL de busca simulada baseada no título real
                    query_encoded = urllib.parse.quote(titulo)
                    url_direta = f"{plat_info['base_url']}{query_encoded}"
                    
                    text_to_vectorize = f"Assistir {titulo} online dublado e legendado oficial na plataforma {plat_name}"
                    
                    ids.append(f"mcp_{id_anime}_{plat_name.lower()}")
                    documents.append(text_to_vectorize)
                    metadatas.append({
                        "id_anime": id_anime,
                        "anime_title": titulo,
                        "platform": plat_name,
                        "url_direta": url_direta
                    })

        if ids:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"[MCP SERVER] Sucesso! {counter} links de streaming indexados via biblioteca nativa Ollama.")

    # --- DEFINIÇÃO DE RESOURCES ---
    def list_resources(self):
        return {
            "resources": [
                {
                    "uri": "anime://catalog/streamings",
                    "name": "Plataformas de Streaming Suportadas",
                    "description": "Lista de plataformas oficiais integradas com o ecossistema OtakuLens.",
                    "mimeType": "application/json"
                }
            ]
        }

    def read_resource(self, uri: str):
        if uri == "anime://catalog/streamings":
            if os.path.exists(MCP_CONFIG_PATH):
                with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        raise ValueError(f"Resource {uri} não encontrado.")

    # --- DEFINIÇÃO DE TOOLS ---
    def list_tools(self):
        return {
            "tools": [
                {
                    "name": "search_streaming_links",
                    "description": "Busca links diretos de streaming (URL) para assistir a um anime específico com base no título.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "anime_title": {
                                "type": "string",
                                "description": "O nome/título do anime retornado pelo RAG."
                            },
                            "preferred_platform": {
                                "type": "string",
                                "description": "Filtro opcional pelo nome da plataforma (ex: Netflix, Crunchyroll)."
                            }
                        },
                        "required": ["anime_title"]
                    }
                }
            ]
        }

    def call_tool(self, name: str, arguments: dict):
        if name == "search_streaming_links":
            anime_title = arguments.get("anime_title")
            preferred_platform = arguments.get("preferred_platform")
            
            print(f"[MCP SERVER] Tool nativa 'search_streaming_links' acionada para: '{anime_title}'")
            
            results = self.collection.query(
                query_texts=[f"Assistir {anime_title}"],
                n_results=3
            )
            
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
                return {"status": "success", "content": "Nenhum link direto exato encontrado no MCP."}
                
            return {"status": "success", "content": output_links}
            
        raise ValueError(f"Tool '{name}' não mapeada.")
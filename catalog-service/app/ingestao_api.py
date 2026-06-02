# catalog-service/app/ingestao_api.py
import os
import json
import requests

# Definição das rotas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON_PATH = os.path.join(BASE_DIR, "data", "raw_animes.json")

# Endpoint da Jikan API v4 para pegar os animes mais populares 
JIKAN_URL = "https://api.jikan.moe/v4/top/anime"

def extrair_dados_api(limite_animes: int = 10) -> list:
    """Faz a requisição HTTP para a API pública do Jikan."""
    print(f"Buscando os top {limite_animes} animes na Jikan API...")
    
    # Parâmetros da paginação e filtros da API
    params = {
        "limit": limite_animes,
        "filter": "bypopularity" # Filtra pelos mais populares do mundo
    }
    
    try:
        response = requests.get(JIKAN_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Erro na API externa: Status {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Falha de rede ao conectar na API: {e}")
        return []

def transformar_e_desnormalizar(lista_animes_raw: list) -> list:
    """Transforma o JSON complexo da API no formato simplificado do OtakuLens (CQRS)."""
    animes_formatados = []
    
    for item in lista_animes_raw:
        # 1. MAPEAMENTO DINÂMICO 
        # Captura gêneros, temas e demografias que a API separa, unificando tudo como tags de gênero
        generos = [g["name"] for g in item.get("genres", [])]
        temas = [t["name"] for t in item.get("themes", [])]
        demografias = [d["name"] for d in item.get("explicit_genres", [])]
        
        lista_generos_completa = list(set(generos + temas + demografias)) # set evita duplicatas
        
        # 2. Tratamento do Ano
        ano = item.get("year")
        if not ano:
            prop_date = item.get("aired", {}).get("prop", {}).get("from", {})
            ano = prop_date.get("year", 2000)
            
        # 3. Tratamento das Plataformas de Streaming
        streamings_raw = item.get("streaming", [])
        plataformas = [s["name"] for s in streamings_raw] if streamings_raw else ["Crunchyroll", "Netflix"]

        # 4. Construção do bloco de texto denso (Contexto RAG)
        titulo = item.get("title")
        sinopse = item.get("synopsis", "Sinopse não disponível.")
        score = item.get("score", "Sem nota")
        tipo = item.get("type", "TV")
        
        contexto_rag = (
            f"Anime: {titulo}. Ano de lançamento: {ano}. Tipo: {tipo}. Nota da crítica: {score}. "
            f"Gêneros e Categorias: {', '.join(lista_generos_completa)}. Sinopse: {sinopse}"
        )

        # 5. Montagem do objeto aderente ao novo formato
        anime_pronto = {
            "id_anime": item.get("mal_id"),
            "titulo": titulo,
            "ano": int(ano),
            "generos": lista_generos_completa, # Agora é uma lista dinâmica de strings
            "plataformas": plataformas,
            "contexto_rag": contexto_rag
        }
        
        animes_formatados.append(anime_pronto)
        
    return animes_formatados

def salvar_json_local(dados_formatados: list):
    """Salva os dados desnormalizados no arquivo local raw_animes.json (Lado de Escrita)."""
    # Garante que a pasta data existe
    os.makedirs(os.path.dirname(DATA_JSON_PATH), exist_ok=True)
    
    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dados_formatados, f, ensure_ascii=False, indent=4)
    
    print(f"Sucesso! {len(dados_formatados)} animes salvos localmente em: {DATA_JSON_PATH}")

def rodar_pipeline_ingestao():
    """Função principal que orquestra o processo de ingestão."""
    # Coleta (Extract)
    dados_api = extrair_dados_api(limite_animes=15) # Buscando os 15 animes mais populares do mundo
    
    if not dados_api:
        print("Não foi possível coletar dados da API externa. Abortando ingestão.")
        return
        
    # Processamento (Transform)
    dados_tratados = transformar_e_desnormalizar(dados_api)
    
    # Carga Local (Load)
    salvar_json_local(dados_tratados)

"""Garante que o script só execute caso seja chamado ativamente no terminal"""
if __name__ == "__main__":
    rodar_pipeline_ingestao()
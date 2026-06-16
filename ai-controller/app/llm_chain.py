# app/llm_chain.py
import asyncio
import httpx
from langchain_community.llms import Ollama

CATALOG_SERVICE_URL = "http://127.0.0.1:8001/catalog/search-semantic"
MCP_DIRECT_TOOL_URL = "http://127.0.0.1:8002/tools/search"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = Ollama(
    base_url="http://127.0.0.1:11434",
    model="phi3:mini",
    temperature=0.1,
    num_predict=400,
)

MAX_CONTEXTO_CHARS = 800

# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------
async def consultar_catalogo_vetorial(prompt_usuario: str, limite: int = 2) -> list:
    payload = {"prompt": prompt_usuario, "limite": limite}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(CATALOG_SERVICE_URL, json=payload, timeout=10.0)
            if response.status_code == 200:
                return response.json()
        return []
    except Exception as e:
        print(f"[CATALOG ERROR] {e}")
        return []

# ---------------------------------------------------------------------------
# MCP — busca links via rota HTTP direta
# ---------------------------------------------------------------------------
async def buscar_links_mcp(titulo_anime: str) -> list:
    payload = {"anime_title": titulo_anime}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                MCP_DIRECT_TOOL_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15.0,
            )
        if response.status_code == 200:
            return response.json().get("links", [])
        print(f"[MCP HTTP] Status inesperado: {response.status_code}")
        return []
    except Exception as e:
        print(f"[MCP HTTP ERROR] {e}")
        return []

# ---------------------------------------------------------------------------
# Geração de sinopse — 1 chamada por anime (resolve o bug de injeção cruzada)
# ---------------------------------------------------------------------------
async def gerar_sinopse_anime(titulo: str, ano: int, trecho_contexto: str, pergunta: str) -> str:
    """
    Pede ao LLM APENAS a sinopse de UM anime específico.
    Prompt minimalista para evitar vazamento de instruções pelo phi3:mini.
    """
    trecho = trecho_contexto
    if len(trecho) > MAX_CONTEXTO_CHARS:
        trecho = trecho[:MAX_CONTEXTO_CHARS] + "..."

    # Prompt simples em formato de instrução direta (sem ChatPromptTemplate)
    # O phi3:mini lida melhor com uma única string do que com roles system/user separados
    prompt_direto = (
        f"Resuma em português brasileiro, em 2 a 3 frases, o anime '{titulo}' "
        f"com base nestas informações: {trecho}\n\n"
        f"Escreva APENAS o resumo, sem título, sem listas, sem explicações extras."
    )

    try:
        sinopse = await llm.ainvoke(prompt_direto)
        # Garante que só o primeiro parágrafo seja usado (corta qualquer vazamento)
        linhas = [l for l in sinopse.strip().splitlines() if l.strip()]
        # Descarta linhas que parecem ser instruções vazadas (palavras-chave reveladoras)
        linhas_limpas = [
            l for l in linhas
            if not any(kw in l.lower() for kw in [
                "system:", "anime:", "informações:", "respon", "escreva",
                "sinopse:", "onde assistir", "pt-br", "português"
            ])
        ]
        return " ".join(linhas_limpas).strip() if linhas_limpas else " ".join(linhas[:3]).strip()
    except Exception as e:
        print(f"[LLM SINOPSE ERROR] {titulo}: {e}")
        return "Sinopse temporariamente indisponível."

# ---------------------------------------------------------------------------
# Montagem da seção de links
# ---------------------------------------------------------------------------
def _formatar_secao_links(links: list) -> str:
    if not links:
        return "**Onde assistir:** Links indisponíveis no momento.\n"
    linhas = "**Onde assistir:**\n"
    for item in links:
        linhas += f"- Na {item['plataforma']}: {item['url_direta']}\n"
    return linhas

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
async def gerar_recomendacao_rag(prompt_usuario: str) -> str:
    """
    Pipeline RAG + MCP com resposta montada inteiramente pelo Python.

    Fluxo por anime:
      1. Busca catálogo vetorial
      2. Busca links MCP (paralelo entre animes)
      3. Gera sinopse via LLM (paralelo entre animes, 1 chamada por anime)
      4. Python monta o bloco final: ### título + sinopse + links reais
    """

    # 1. RETRIEVAL
    animes_contexto = await consultar_catalogo_vetorial(prompt_usuario, limite=2)
    if not animes_contexto:
        return "Desculpe, estou com dificuldades para acessar meu catálogo de animes no momento."

    # 2 + 3. MCP e sinopses em paralelo para todos os animes
    tarefas = []
    for anime in animes_contexto:
        tarefas.append(buscar_links_mcp(anime["titulo"]))
        tarefas.append(gerar_sinopse_anime(
            titulo=anime["titulo"],
            ano=anime["ano"],
            trecho_contexto=anime["trecho_contexto"],
            pergunta=prompt_usuario,
        ))

    resultados = await asyncio.gather(*tarefas)

    # resultados vem intercalado: [links_0, sinopse_0, links_1, sinopse_1, ...]
    blocos = []
    for i, anime in enumerate(animes_contexto):
        links   = resultados[i * 2]
        sinopse = resultados[i * 2 + 1]

        secao_links = _formatar_secao_links(links)

        bloco = (
            f"### {anime['titulo']} ({anime['ano']})\n"
            f"**Sinopse:** {sinopse}\n\n"
            f"{secao_links}"
        )
        blocos.append(bloco)

    # 4. MONTAGEM FINAL — Python controla 100% da estrutura
    return "\n\n---\n\n".join(blocos)
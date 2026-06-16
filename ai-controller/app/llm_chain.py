# app/llm_chain.py
import re
import asyncio
import httpx
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CATALOG_SERVICE_URL = "http://127.0.0.1:8001/catalog/search-semantic"
MCP_DIRECT_TOOL_URL = "http://127.0.0.1:8002/tools/search"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = Ollama(
    base_url="http://127.0.0.1:11434",
    model="phi3:mini",
    temperature=0.1,
    num_predict=512,
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
# MCP — busca links via rota HTTP direta (sem SSE)
# Retorna lista de dicts: [{"anime":..., "plataforma":..., "url_direta":...}]
# ---------------------------------------------------------------------------
async def buscar_links_mcp(titulo_anime: str) -> list:
    """
    Chama /tools/search e retorna a lista de links brutos.
    O LLM NÃO verá essas URLs — elas são injetadas pelo Python depois da geração.
    """
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
# Pipeline de enriquecimento
# ---------------------------------------------------------------------------
async def pipeline_enriquecimento_mcp(animes_contexto: list):
    """
    Retorna:
      - contexto_str : texto injetado no prompt (SEM URLs — só título, sinopse, plataformas)
      - links_map    : dict { titulo -> lista de links } para injeção pós-geração
    """
    tarefas_mcp = [buscar_links_mcp(anime["titulo"]) for anime in animes_contexto]
    resultados_mcp = await asyncio.gather(*tarefas_mcp)

    contexto_str = ""
    links_map = {}

    for anime, links in zip(animes_contexto, resultados_mcp):
        titulo = anime["titulo"]
        plataformas = ", ".join(anime["plataformas"])
        trecho = anime["trecho_contexto"]
        if len(trecho) > MAX_CONTEXTO_CHARS:
            trecho = trecho[:MAX_CONTEXTO_CHARS] + "..."

        # Contexto para o LLM — sem nenhuma URL
        contexto_str += "\n---\n"
        contexto_str += f"Título: {titulo} ({anime['ano']})\n"
        contexto_str += f"Plataformas disponíveis: {plataformas}\n"
        contexto_str += f"Informações: {trecho}\n"

        # Links guardados separadamente para injeção posterior
        links_map[titulo] = links

    return contexto_str, links_map

# ---------------------------------------------------------------------------
# Injeção de links pós-geração
# ---------------------------------------------------------------------------
def _formatar_secao_links(links: list) -> str:
    """Monta a seção 'Onde assistir' com os links reais do MCP."""
    if not links:
        return "**Onde assistir:** Links indisponíveis no momento.\n"
    linhas = "**Onde assistir:**\n"
    for item in links:
        linhas += f"- Na {item['plataforma']}: {item['url_direta']}\n"
    return linhas

def injetar_links_na_resposta(texto_llm: str, links_map: dict) -> str:
    """
    Estratégia: remove qualquer bloco '**Onde assistir:**...' que o LLM tenha
    gerado (com ou sem URL real) e substitui pelo conteúdo exato do MCP.

    Para cada título em links_map, localiza o cabeçalho ### do anime na resposta
    e injeta a seção de links logo após o bloco de Sinopse.
    """
    resultado = texto_llm

    for titulo, links in links_map.items():
        secao_links = _formatar_secao_links(links)

        # Remove qualquer "**Onde assistir:**" existente (com conteúdo até o próximo ### ou fim)
        resultado = re.sub(
            r'\*\*Onde assistir:\*\*.*?(?=\n###|\Z)',
            '',
            resultado,
            flags=re.DOTALL
        )

        # Localiza o fim do bloco **Sinopse:** deste anime e injeta os links logo depois
        # Busca pelo padrão: **Sinopse:** [qualquer coisa] seguido de linha em branco
        padrao_sinopse = r'(\*\*Sinopse:\*\*[^\n]*(?:\n(?!\n###|\n\*\*)[^\n]*)*)'
        match = re.search(padrao_sinopse, resultado)
        if match:
            pos_fim_sinopse = match.end()
            resultado = (
                resultado[:pos_fim_sinopse]
                + "\n\n"
                + secao_links
                + resultado[pos_fim_sinopse:]
            )

    return resultado.strip()

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
async def gerar_recomendacao_rag(prompt_usuario: str) -> str:
    """
    Pipeline RAG + MCP com links injetados pelo Python (não pelo LLM).

    Fluxo:
      1. Busca animes no catálogo vetorial
      2. Busca links no MCP (paralelo) — armazena em links_map, NÃO passa ao LLM
      3. LLM gera apenas título + sinopse
      4. Python injeta os links reais na resposta final
    """

    # 1. RETRIEVAL
    animes_contexto = await consultar_catalogo_vetorial(prompt_usuario, limite=2)
    if not animes_contexto:
        return "Desculpe, estou com dificuldades para acessar meu catálogo de animes no momento."

    # 2. CONTEXT ENRICHMENT
    contexto_str, links_map = await pipeline_enriquecimento_mcp(animes_contexto)

    # 3. PROMPT — LLM gera apenas sinopse, sem ver URLs
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é o OtakuLens, recomendador especialista em animes. "
            "Responda EXCLUSIVAMENTE em Português do Brasil (pt-BR). "
            "Traduza qualquer texto em inglês ou espanhol.\n\n"

            "Para cada anime no CONTEXTO, escreva EXATAMENTE neste formato:\n\n"
            "### [Nome do Anime] ([Ano])\n"
            "**Sinopse:** [Escreva aqui um resumo em português, baseado nas Informações do contexto]\n\n"

            "NÃO escreva a seção 'Onde assistir'. Ela será adicionada automaticamente.\n"
            "NÃO invente informações. Use apenas o que está no CONTEXTO.\n\n"

            "CONTEXTO:\n{contexto}"
        )),
        ("user", "{pergunta}"),
    ])

    chain = prompt_template | llm | StrOutputParser()
    try:
        texto_llm = await chain.ainvoke(
            {"contexto": contexto_str, "pergunta": prompt_usuario}
        )
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "O modelo local demorou muito para responder. Tente novamente."

    # 4. INJEÇÃO DOS LINKS PELO PYTHON (100% confiável, sem depender do LLM)
    resposta_final = injetar_links_na_resposta(texto_llm, links_map)
    return resposta_final
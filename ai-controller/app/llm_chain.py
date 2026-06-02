# app/llm_chain.py
import requests
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuração do endpoint do Catalog Service (Microsserviço de Leitura CQRS)
CATALOG_SERVICE_URL = "http://localhost:8001/catalog/search-semantic"

# Inicialização do LLM via Ollama
llm = Ollama(
    base_url="http://localhost:11434",
    model="llama3" 
)

def consultar_catalogo_vetorial(prompt_usuario: str, limite: int = 2) -> list:
    """Consome o endpoint do Catalog Service para trazer o contexto desnormalizado."""
    payload = {"prompt": prompt_usuario, "limite": limite}
    try:
        response = requests.post(CATALOG_SERVICE_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar no Catalog Service: {e}")
        return []

def buscar_links_mcp(mcp_server, titulo_anime: str) -> str:
    """Invoca a Tool do MCP Server para buscar links diretos no banco vetorial de streaming."""
    if not mcp_server:
        return ""
    
    try:
        # Aciona a ferramenta do servidor MCP
        mcp_response = mcp_server.call_tool(
            name="search_streaming_links",
            arguments={"anime_title": titulo_anime}
        )
        
        # Converte o retorno estruturado do MCP em texto para o contexto da IA
        if mcp_response.get("status") == "success" and isinstance(mcp_response.get("content"), list):
            links_texto = "Links diretos para assistir:\n"
            for item in mcp_response["content"]:
                links_texto += f"  - Na {item['plataforma']}: {item['url_direta']}\n"
            return links_texto
    except Exception as e:
        print(f"[MCP INTEGRATION ERROR] Falha ao chamar a Tool do MCP: {e}")
    
    return "  - Links diretos indisponíveis no servidor de contexto MCP no momento.\n"

def formatar_contexto_hibrido(animes_retornados: list, mcp_server) -> str:
    """Formata os dados do RAG e agrega as informações de links em tempo real do MCP."""
    contexto_formatado = ""
    for anime in animes_retornados:
        titulo = anime["titulo"]
        plataformas = ", ".join(anime["plataformas"])
        
        contexto_formatado += f"\n---\n"
        contexto_formatado += f"Título: {titulo} ({anime['ano']})\n"
        contexto_formatado += f"Disponibilidade Geral: {plataformas}\n"
        
        # --- ENRIQUECIMENTO VIA PROTOCOLO MCP ---
        contexto_formatado += buscar_links_mcp(mcp_server, titulo)
        
        contexto_formatado += f"Informações e Reviews: {anime['trecho_contexto']}\n"
        
    return contexto_formatado

def gerar_recomendacao_rag(prompt_usuario: str, mcp_server=None) -> str:
    """Executa o pipeline completo integrado: Retrieval (RAG) + Contexto Dinâmico (MCP) + Generation"""
    
    # 1. RETRIEVAL: Busca animes relevantes no banco vetorial via microsserviço
    animes_contexto = consultar_catalogo_vetorial(prompt_usuario, limite=2)
    
    if not animes_contexto:
        return "Desculpe, estou com dificuldades para acessar meu catálogo de animes no momento."

    # 2. Formata o contexto híbrido (Dados do Catálogo + Links Vetoriais do MCP Server)
    contexto_str = formatar_contexto_hibrido(animes_contexto, mcp_server)

    # 3. PROMPT TEMPLATE: Adiciona as regras explícitas para incluir as URLs do MCP
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é o OtakuLens, um recomendador especialista em animes. "
            "Use o CONTEXTO fornecido abaixo para indicar um novo anime ao usuário com base no que ele pediu.\n"
            "O contexto contém links diretos de streaming fornecidos pelo protocolo MCP. "
            "É OBRIGATÓRIO incluir esses links textuais exatos na sua resposta para que o usuário possa clicar e assistir.\n\n"
            "Responda de forma direta em português (Brasil), trazendo as sugestões, uma breve sinopse e os links correspondentes.\n\n"
            "CONTEXTO DOS ANIMES E LINKS DE STREAMING (RAG + MCP):\n{contexto}"
        )),
        ("user", "{pergunta}")
    ])

    # 4. CHAIN: Conecta o Prompt -> LLM -> Parser de Saída
    chain = prompt_template | llm | StrOutputParser()

    # 5. GENERATION: Invoca o modelo gerando a resposta final consolidada
    resposta_final = chain.invoke({
        "contexto": contexto_str,
        "pergunta": prompt_usuario
    })

    return resposta_final
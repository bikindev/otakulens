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

def formatar_contexto(animes_retornados: list) -> str:
    """Formata os metadados e blocos de texto textuais para injetar no prompt."""
    contexto_formatado = ""
    for anime in animes_retornados:
        plataformas = ", ".join(anime["plataformas"])
        contexto_formatado += f"\n---\n"
        contexto_formatado += f"Título: {anime['titulo']} ({anime['ano']})\n"
        contexto_formatado += f"Disponível em: {plataformas}\n"
        contexto_formatado += f"Informações e Reviews: {anime['trecho_contexto']}\n"
    return contexto_formatado

def gerar_recomendacao_rag(prompt_usuario: str) -> str:
    """Executa o pipeline completo: Retrieval (Catalog) + Generation (Ollama)"""
    
    # 1. RETRIEVAL: Busca animes relevantes no banco vetorial via microsserviço
    animes_contexto = consultar_catalogo_vetorial(prompt_usuario, limite=2)
    
    if not animes_contexto:
        return "Desculpe, estou com dificuldades para acessar meu catálogo de animes no momento."

    # 2. Formata o contexto recuperado
    contexto_str = formatar_contexto(animes_contexto)

    # 3. PROMPT TEMPLATE: Estrutura a persona e as regras do assistente
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é o OtakuLens, um recomendador especialista em animes. "
            "Use estritamente o CONTEXTO fornecido abaixo para indicar um novo anime ao usuário, baseado no que ele quer assistir. "
            "Responda em português (Brasil)."
            "Seja direto em sua reposta: entregue apenas as sugestões de anime, uma breve sinopse e as plataformas onde eles podem ser assistidos.\n\n"
            "CONTEXTO DOS ANIMES:\n{contexto}"
        )),
        ("user", "{pergunta}")
    ])

    # 4. CHAIN: Conecta o Prompt -> LLM -> Parser de Saída
    chain = prompt_template | llm | StrOutputParser()

    # 5. GENERATION: Invoca o modelo gerando a resposta final
    resposta_final = chain.invoke({
        "contexto": contexto_str,
        "pergunta": prompt_usuario
    })

    return resposta_final
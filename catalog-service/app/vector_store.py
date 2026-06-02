import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# Configuração do Modelo de Embeddings do Ollama
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text"
)

# Caminho onde o ChromaDB vai salvar os dados em disco
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def inicializar_banco_vetorial():
    # Inicializa ou conecta ao banco vetorial ChromaDB
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="catalog_animes_rag" # nome da coleção ("tabela") dentro do banco vetorial
    )

def popular_banco_vetorial_com_json(json_path: str):
    """
    Lê o arquivo JSON (carga da API externa),
    cria os documentos com metadados estruturados e os salva no banco vetorial.
    """
    db = inicializar_banco_vetorial()
    
    # Verifica se o banco já possui registros para não popular novamente
    if len(db.get()["ids"]) > 0:
        print("Banco vetorial já populado. Pulando etapa de carga.")
        return db

    if not os.path.exists(json_path):
        print(f"Erro: Arquivo {json_path} não encontrado.")
        return db

    with open(json_path, "r", encoding="utf-8") as file:
        animes_data = json.load(file)

    documentos = []
    
    for item in animes_data:
        # Cria o objeto Document do LangChain.
        # O page_content é o texto denso que o RAG vai varrer (sinopse + reviews).
        # O metadata é o JSON estruturado (CQRS desnormalizado) que retorna na busca.
        doc = Document(
            page_content=item["contexto_rag"],
            metadata={
                "id_anime": item["id_anime"],
                "titulo": item["titulo"],
                "ano": item["ano"],
                "generos": ", ".join(item["generos"]), 
                "plataformas": ", ".join(item["plataformas"])
            }
        )
        documentos.append(doc)

    # O LangChain envia os textos para o Ollama gerar o embedding e salva tudo no ChromaDB
    db.add_documents(documentos)
    print(f"Sucesso: {len(documentos)} animes vetorizados e salvos no ChromaDB!")
    return db

def buscar_animes_similares(query: str, k: int = 2):
    """
    Realiza a busca por similaridade (Retrieval do RAG)
    Retorna os documentos mais próximos semanticamente do prompt do usuário.
    """
    db = inicializar_banco_vetorial()
    # Executa a busca vetorial por proximidade (Cosine Similarity interna do Chroma)
    resultados = db.similarity_search(query, k=k)
    return resultados
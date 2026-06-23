# 🎬 OtakuLens - Recomendador Inteligente de Animes

Trabalho prático final desenvolvido para a disciplina **GCC129 Sistemas Distribuídos** no curso de Sistemas de Informação da **Universidade Federal de Lavras (UFLA)**.

---

## 1. Introdução e Objetivos do Sistema

O projeto **OtakuLens** consiste em um sistema distribuído de recomendação de animes que utiliza Inteligência Artificial e processamento de linguagem natural para entregar sugestões personalizadas e altamente contextualizadas aos usuários. 

Diferente dos sistemas de recomendação tradicionais baseados apenas em filtros colaborativos ou tags estáticas, o OtakuLens adota uma abordagem semântica, sendo capaz de interpretar prompts subjetivos e correlacioná-los com sinopses, enredos e metadados densos de produções reais.

O objetivo do sistema é consolidar uma especificação arquitetural de microsserviços fracamente acoplados, implementando um pipeline de RAG (Retrieval-Augmented Generation) com armazenamento vetorial unificado no serviço de catálogo e utilizando o protocolo MCP (Model Context Protocol) para o enriquecimento dinâmico de links de streaming em tempo de execução de forma isolada.

---

## 2. Modelo Arquitetural e Visão Geral do Sistema

O sistema foi modelado seguindo uma arquitetura de microsserviços fracamente acoplados, distribuídos em camadas lógicas para garantir escalabilidade, isolamento de falhas e independência de implantação.

### Diagrama da Arquitetura
<img width="551" height="546" alt="616db8be-af13-4718-b0f9-29d44d380f07" src="https://github.com/user-attachments/assets/b27bec13-af21-424f-9871-f7c65a6452bf" />

### Descrição dos Componentes

1. **Interface do Usuário (Streamlit - Porta 8501):** Frontend responsivo (Web/Mobile) que interage com o usuário, captura os prompts subjetivos e renderiza dinamicamente as respostas estruturadas em Markdown e botões interativos de redirecionamento.
2. **Backend Único (FastAPI - Porta 5000):** Ponto de entrada central de requisições do cliente. Atua como um agregador síncrono linear estável para blindar a comunicação contra instabilidades de timeout de rede do sistema operacional.
3. **Controlador de IA (LangChain / FastAPI - Porta 8000):** Orquestrador do pipeline de RAG clássico acoplado ao modelo local `phi:mini` (via Ollama). Atua também como o cliente que consome as ferramentas do barramento MCP.
4. **Catalog Service (FastAPI / ChromaDB - Porta 8001):** Microsserviço de domínio responsável pela busca semântica por similaridade de cosseno em reviews, metadados e acervos conceituais de animes.
5. **MCP Server (Starlette / ChromaDB - Porta 8002):** Servidor independente que implementa a especificação oficial do **Model Context Protocol (MCP)** da Anthropic. Ele expõe a ferramenta vetorial `search_streaming_links` para calcular dinamicamente URLs legítimas de transmissão (Crunchyroll, Netflix, etc.) baseadas nos mapeamentos de `mcp_streaming.json`.

---

## 3. Detalhamento Técnico da Implementação

### 3.1. Catalog Service
* **Ingestão de Dados e Cache Local:** Sob demanda, o serviço consome dados brutos de catálogos públicos, realiza processos de limpeza, mapeamento de gêneros e plataformas disponíveis, salvando o estado atualizado em um arquivo JSON local desnormalizado (`data/raw_animes.json`).
* **Persistência Vetorial:** No evento de inicialização da aplicação, o serviço lê o JSON local e utiliza o framework LangChain com a classe Ollama Embeddings para enviar as sinopses e metadados textuais (`contexto_rag`) ao modelo local `nomic-embed-text` do Ollama. Os vetores resultantes são persistidos em disco em uma instância dedicada do ChromaDB, permitindo buscas semânticas locais imediatas por similaridade de cosseno.

### 3.2. Pipeline de RAG (Retrieval-Augmented Generation)
Para mitigar o problema de alucinação do modelo de linguagem de grande porte (LLM) e garantir recomendações ancoradas em dados fidedignos, o pipeline de RAG foi estruturado da seguinte forma:
1. O usuário submete um prompt subjetivo à interface, que repassa ao Backend Único até chegar ao endpoint de recomendação do AI Controller (`POST /ai/recommend`).
2. O AI Controller realiza uma chamada assíncrona de alto desempenho via cliente HTTP para o endpoint de busca semântica do Catalog Service (`POST /catalog/search-semantic`).
3. O Catalog Service realiza a varredura por proximidade vetorial no seu ChromaDB e devolve uma carga estruturada em JSON contendo o título, ano, gêneros e os trechos contextuais dos animes mais relevantes encontrados.

### 3.3. Integração MCP (Model Context Protocol) para Links de Streaming
Para evitar o acoplamento rígido de regras de negócios mutáveis (como links diretos e URLs de plataformas parceiras que expiram ou mudam frequentemente) dentro do core do orquestrador de IA, foi implementado o padrão MCP, funcionando como um barramento dinâmico de habilidades e dados contextuais para o LLM.

* **Estrutura de Dados Consistente:** O servidor consome o arquivo de configuração de plataformas (`data/mcp_streaming.json`) e cruza dinamicamente com o catálogo real de animes (`raw_animes.json`). Isso garante consistência absoluta: todo anime sugerido pelo RAG clássico possui indexação correspondente no ecossistema MCP.
* **O MCP Server (`mcp_server.py`):** Atua como o provedor de contexto, inicializando uma coleção isolada no ChromaDB (gerando embeddings com o modelo `nomic-embed-text`). O servidor expõe formalmente **Resources** (URIs estáticas como `anime://catalog/streamings` para mapear plataformas ativas) e **Tools** (a ferramenta funcional `search_streaming_links`, que recebe o parâmetro estruturado `anime_title` para retornar a URL exata calculada).
* **Integração no AI Controller (MCP Client por Emulação HTTP):** O AI Controller atua como o cliente das ferramentas. Ao receber os animes recomendados pelo RAG, em vez de depender de gerenciadores de contexto assíncronos instáveis do SDK em ambiente Windows, ele emite uma chamada HTTP direta contendo payloads padronizados na especificação JSON-RPC (`method: tools/call`) para o endpoint `/mcp/messages` do servidor MCP. O retorno contendo os links diretos e o contexto textual são unificados no template de prompt.

---

## 4. Decisões de Engenharia de Sistemas Distribuídos

* **Abandono do Transporte STDIO por SSE/HTTP:** Devido a restrições crônicas do ecossistema Windows ao lidar com a decodificação de caracteres de terminal (`sys.stdin`/`sys.stdout` com acentuações em caminhos de usuário de sistema), o transporte STDIO nativo do MCP foi descontinuado. O servidor e o cliente MCP foram implementados utilizando o transporte **SSE (Server-Sent Events) sobre HTTP**, dividindo-se nos endpoints de handshake (`GET /mcp/sse`) e envio assíncrono (`POST /mcp/messages`).
* **Desacoplamento e Comunicação Direta JSON-RPC:** Para mitigar conflitos severos de concorrência assíncrona e timeouts entre o loop de eventos e o SDK de cliente oficial do MCP da Anthropic, a camada de IA consome o servidor MCP simulando de forma enxuta o protocolo JSON-RPC via requisições HTTP POST puras e velozes, mantendo o servidor Starlette operando na especificação oficial de forma intocada.
* **Alinhamento de Idioma:** O prompt do controlador de IA força a tradução em tempo real para português do Brasil (pt-BR) de conteúdos do banco vetorial que originalmente estejam em inglês ou espanhol.
* **Otimização de Hardware para Demonstração:** Para evitar gargalos e travamentos de processamento local por CPU optamos por utilizar o modelo **`phi3:mini` (3.8B parâmetros)** da Microsoft, garantindo respostas rápidas, fluidas e com baixo consumo de recursos, mantendo a premissa de execução 100% local. Mas para aqueles que posssuem maior poder computacional é possível utilizar o modelo **`lamma3` ((8B parâmetros))**.

---

## 5. Tecnologias Utilizadas

* **Linguagem Principal:** Python 3.11+
* **Frameworks Web:** FastAPI, Starlette, Uvicorn
* **Framework de IA:** LangChain
* **Provedor de LLM Local:** Ollama (`phi3:mini`)
* **Banco Vetorial & Embeddings:** ChromaDB (`nomic-embed-text`)
* **Interface Gráfica:** Streamlit
* **Protocolo de Contexto:** Model Context Protocol (MCP)

---

## 6. Como Executar o Projeto

Certifique-se de ter o **Ollama** instalado na máquina e com os modelos baixados (`ollama pull phi:mini` e `ollama pull nomic-embed-text`).

Divida seu terminal em 6 janelas e execute os serviços na ordem abaixo:

### 1. Executar a IA local, buscar os animes na API do MyAnimeList e instalar as dependências
ollama pull nomic-embed-text\
ollama pull phi:mini\
pip install -r requirements.txt\
python \catalog-service\ingestao_api.py

### 2. Catalog Service
cd catalog-service\
uvicorn main:app --host 127.0.0.1 --port 8001 --reload

### 3. MCP Server
cd mcp-server\
python mcp_server.py

### 4. AI Controller
cd ai-controller\
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

### 5. Backend Único
cd interface\
python backend.py

### 6. Interface Gráfica (Streamlit)
cd interaface\
streamlit run interface.py\
\
O navegador abrirá automaticamente em http://localhost:8501.

## Exemplo de Uso e Respostas
Prompt Válido: "Quero um anime de ação com uma protagonista feminina forte."\
\
Comportamento: O catálogo busca os títulos correlacionados através de uma busca por similaridade no banco, o MCP calcula as URLs oficiais da Crunchyroll/Netflix, a LLM monta o texto com as recomendações e a interface exibe botões clicáveis para assistir.

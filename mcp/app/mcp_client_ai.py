# -*- coding: utf-8 -*-
"""
OtakuLens - AI MCP Client Simulator
Componente: mcp_client_ai.py
"""

import json
from mcp_server import OtakuLensMCPServer

class OtakuLensAIClient:
    def __init__(self, server_instance: OtakuLensMCPServer):
        self.mcp_server = server_instance
        print("[MCP CLIENT] Cliente de IA registrado e conectado ao MCP Server local.")

    def inspect_capabilities(self):
        print("\n=== [MCP HANDSHAKE] Descobrindo Recursos e Ferramentas do Servidor ===")
        resources = self.mcp_server.list_resources()
        tools = self.mcp_server.list_tools()
        print(f"Recursos:\n{json.dumps(resources, indent=2, ensure_ascii=False)}")
        print(f"Ferramentas:\n{json.dumps(tools, indent=2, ensure_ascii=False)}")
        print("===================================================================\n")

    def execution_pipeline_demo(self, anime_detectado: str, plataforma_desejada: str = None):
        print(f"[MCP CLIENT] [Orquestração] IA detectou intenção de reprodução para o anime '{anime_detectado}'.")
        print("[MCP CLIENT] [Orquestração] Invocando ferramenta MCP dinamicamente...")
        
        arguments = {"anime_title": anime_detectado}
        if plataforma_desejada:
            arguments["preferred_platform"] = plataforma_desejada
            
        response = self.mcp_server.call_tool(
            name="search_streaming_links",
            arguments=arguments
        )
        
        print("\n=== [MENSAGEM DE CONTEXTO RETORNADA PELO MCP SERVER] ===")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("======================================================\n")
        
        context_str = ""
        if isinstance(response.get("content"), list):
            context_str = "Links oficiais de streaming encontrados na base MCP:\n"
            for item in response["content"]:
                context_str += f"- Assistir no serviço {item['plataforma']}: {item['url_direta']}\n"
        else:
            context_str = f"Informação de streaming: {response.get('content')}\n"
            
        langchain_prompt_mock = (
            f"Você é o assistente virtual do OtakuLens.\n"
            f"Responda à solicitação do usuário utilizando o contexto do RAG clássico "
            f"e agregue estes dados em tempo real fornecidos pelo protocolo MCP:\n\n"
            f"--- CONTEXTO MCP ---\n{context_str}--------------------\n\n"
            f"Gere a resposta final incluindo as URLs textuais exatas para redirecionamento."
        )
        print("[MCP CLIENT] Simulação do Prompt formatado pronto para envio ao Ollama (llama3):\n")
        print(langchain_prompt_mock)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # Inicializa o Servidor (que buscará dinamicamente os caminhos relativos de data/)
    server_instance = OtakuLensMCPServer()
    
    # Conecta o Cliente
    ai_client = OtakuLensAIClient(server_instance=server_instance)
    
    # Realiza Handshake
    ai_client.inspect_capabilities()
    
    # Cenário 1: # Testando com um anime real do seu JSON
    print("--- TESTANDO CENÁRIO REAL (Dados extraídos do seu catálogo) ---")
    ai_client.execution_pipeline_demo(anime_detectado="Steins;Gate")
    
    # Cenário 2: Usuário pede especificamente na Netflix
    print("--- EXECUÇÃO DO CENÁRIO 2: Solicitação com Filtro de Plataforma ---")
    ai_client.execution_pipeline_demo(anime_detectado="Naruto Shippuden", plataforma_desejada="Netflix")
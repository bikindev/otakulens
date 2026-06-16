# interface/interface.py
import streamlit as st
import requests

# Configurações básicas da página
st.set_page_config(
    page_title="OtakuLens - Recomendador de Animes",
    page_icon="🎬",
    layout="centered"
)

BACKEND_URL = "http://127.0.0.1:5000/api/otakulens/recommend"

# Cabeçalho da Interface
st.title("🎬 OtakuLens")
st.subheader("Seu recomendador inteligente de animes")
st.write("Digite o que você está com vontade de assistir (ex: *'Quero um anime de luta com temática sombria e mistério'*):")

# Input de texto do usuário
user_input = st.text_area("O que você quer assistir hoje?", placeholder="Descreva seu gosto subjetivo aqui...", height=100)

# Botão de envio
if st.button("Buscar Recomendações ✨", use_container_width=True):
    if not user_input.strip():
        st.warning("Por favor, digite alguma descrição antes de buscar.")
    else:
        # Cria uma animação de "Carregando" enquanto o pipeline (RAG + MCP + Ollama) executa
        with st.spinner("Processando..."):
            try:
                payload = {"prompt": user_input}
                response = requests.post(BACKEND_URL, json=payload, timeout=190.0)
                
                if response.status_code == 200:
                    data = response.json()
                    recommendation_text = data.get("recommendation", "Nenhuma resposta gerada.")
                    
                    st.success("Aqui estão suas recomendações personalizadas!")
                    st.markdown("---")
                    # Renderiza o Markdown retornado pelo LLM (com os links calculados pelo MCP!)
                    st.markdown(recommendation_text)
                    st.markdown("---")
                else:
                    st.error(f"Erro no servidor backend: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                st.error("A inferência local do Ollama estourou o tempo limite. Tente simplificar seu prompt.")
            except Exception as e:
                st.error(f"Não foi possível conectar ao Backend único: {e}")
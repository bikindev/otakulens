# interface/interface.py
import re
import streamlit as st
import requests

st.set_page_config(
    page_title="OtakuLens - Recomendador de Animes",
    page_icon="🎬",
    layout="centered"
)

BACKEND_URL = "http://127.0.0.1:5000/api/otakulens/recommend"

# ---------------------------------------------------------------------------
# Parser de links — separa o texto limpo dos blocos "Onde assistir"
# ---------------------------------------------------------------------------
ICONES_PLATAFORMA = {
    "crunchyroll": "🟠",
    "netflix":     "🔴",
    "prime video": "🔵",
    "funimation":  "🟣",
}

def parse_resposta(texto: str):
    """
    Separa o texto em blocos de anime.
    Para cada bloco retorna:
      - texto_limpo : markdown sem as linhas de link
      - links       : lista de (plataforma, url)
    """
    # Divide nos separadores --- que o llm_chain.py usa entre animes
    blocos_raw = re.split(r'\n\s*---\s*\n', texto)
    blocos = []

    for bloco in blocos_raw:
        if not bloco.strip():
            continue

        links = []
        linhas_texto = []

        # Sinaliza quando estamos dentro da seção "Onde assistir"
        dentro_links = False

        for linha in bloco.splitlines():
            # Detecta o cabeçalho da seção de links
            if re.match(r'\*\*Onde assistir:\*\*', linha.strip()):
                dentro_links = True
                continue  # não adiciona o cabeçalho ao texto

            if dentro_links:
                # Captura linhas do tipo: - Na Crunchyroll: https://...
                match = re.match(r'-\s*Na\s+(.+?):\s*(https?://\S+)', linha.strip())
                if match:
                    links.append((match.group(1).strip(), match.group(2).strip()))
                    continue
                # Linha vazia ou outra coisa: sai do bloco de links
                if linha.strip():
                    dentro_links = False
                    linhas_texto.append(linha)
            else:
                linhas_texto.append(linha)

        blocos.append({
            "texto": "\n".join(linhas_texto).strip(),
            "links": links,
        })

    return blocos

def renderizar_botoes_links(links: list):
    """Renderiza cada link como st.link_button lado a lado."""
    if not links:
        return
    st.markdown("**Onde assistir:**")
    cols = st.columns(len(links))
    for col, (plataforma, url) in zip(cols, links):
        icone = ICONES_PLATAFORMA.get(plataforma.lower(), "▶️")
        col.link_button(f"{icone} {plataforma}", url, use_container_width=True)

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("🎬 OtakuLens")
st.subheader("Seu recomendador inteligente de animes")
st.write("Digite o que você está com vontade de assistir (ex: *'Quero um anime de luta com temática sombria e mistério'*):")

user_input = st.text_area(
    "O que você quer assistir hoje?",
    placeholder="Descreva seu gosto subjetivo aqui...",
    height=100
)

if st.button("Buscar Recomendações ✨", use_container_width=True):
    if not user_input.strip():
        st.warning("Por favor, digite alguma descrição antes de buscar.")
    else:
        with st.spinner("Consultando catálogo e buscando links de streaming..."):
            try:
                payload  = {"prompt": user_input}
                response = requests.post(BACKEND_URL, json=payload, timeout=310.0)

                if response.status_code == 200:
                    data = response.json()
                    recommendation_text = data.get("recommendation", "Nenhuma resposta gerada.")

                    st.success("Aqui estão suas recomendações personalizadas!")
                    st.markdown("---")

                    blocos = parse_resposta(recommendation_text)

                    for bloco in blocos:
                        # Renderiza o markdown do anime (título + sinopse)
                        st.markdown(bloco["texto"])
                        # Renderiza os botões de streaming logo abaixo
                        renderizar_botoes_links(bloco["links"])
                        st.markdown("---")

                else:
                    st.error(f"Erro no servidor backend: {response.status_code}")

            except requests.exceptions.Timeout:
                st.error("A inferência local do Ollama estourou o tempo limite. Tente simplificar seu prompt.")
            except Exception as e:
                st.error(f"Não foi possível conectar ao Backend único: {e}")
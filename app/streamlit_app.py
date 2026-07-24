"""
JobMatch AI - Interface Streamlit
Consome a API FastAPI local (app/main.py) para prever o fit_percentual de vagas.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/prever-fit"

st.set_page_config(
    page_title="JobMatch AI",
    page_icon="🎯",
    layout="centered",
)

st.title("🎯 JobMatch AI")
st.markdown(
    "Sistema de predição de compatibilidade entre vagas de emprego e o perfil "
    "profissional do autor, usando **BERTimbau fine-tuned**."
)

st.divider()

texto_vaga = st.text_area(
    "Cole aqui o texto completo da vaga:",
    height=250,
    placeholder="Ex: Vaga para Cientista de Dados Júnior, com foco em Python, SQL, "
                "Machine Learning e Power BI...",
)

analisar = st.button("Analisar vaga", type="primary", use_container_width=True)

if analisar:
    if len(texto_vaga.strip()) < 20:
        st.warning("Cole um texto de vaga com pelo menos 20 caracteres.")
    else:
        with st.spinner("Analisando com o modelo BERTimbau..."):
            try:
                resposta = requests.post(API_URL, json={"texto": texto_vaga}, timeout=30)
                resposta.raise_for_status()
                resultado = resposta.json()

                fit = resultado["fit_percentual"]
                interpretacao = resultado["interpretacao"]

                st.divider()
                st.subheader("Resultado")

                # Cor dinâmica conforme a faixa de fit
                if fit >= 70:
                    cor = "green"
                elif fit >= 45:
                    cor = "orange"
                elif fit >= 20:
                    cor = "orange"
                else:
                    cor = "red"

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Fit estimado", f"{fit}/100")
                with col2:
                    st.markdown(f"### :{cor}[{interpretacao}]")

                st.progress(int(fit) / 100)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Não foi possível conectar à API. Verifique se ela está rodando "
                    "(`uvicorn app.main:app --reload`) na porta 8000."
                )
            except Exception as e:
                st.error(f"Erro ao processar a predição: {e}")

st.divider()
st.caption(
    "Modelo: BERTimbau fine-tuned (MAE 12.36, 51% de melhoria sobre baseline) · "
    "Treinado em 162 vagas reais coletadas manualmente e via API Adzuna."
)

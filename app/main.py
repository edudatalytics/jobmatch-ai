"""
JobMatch AI - API de predição de fit_percentual
Endpoint principal: POST /prever-fit
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# --- Configuração ---
CAMINHO_MODELO = os.path.join(os.path.dirname(__file__), "..", "models", "bertimbau_fit_v1")
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256

app = FastAPI(
    title="JobMatch AI API",
    description="Sistema de predição de compatibilidade (fit) entre vagas de emprego e o perfil profissional do autor, usando BERTimbau fine-tuned.",
    version="1.0.0",
)

# --- Carregamento do modelo (executado uma única vez, na inicialização) ---
print("Carregando modelo BERTimbau fine-tuned...")
tokenizer = AutoTokenizer.from_pretrained(CAMINHO_MODELO)
modelo = AutoModelForSequenceClassification.from_pretrained(CAMINHO_MODELO)
modelo.to(DISPOSITIVO)
modelo.eval()
print(f"Modelo carregado com sucesso. Rodando em: {DISPOSITIVO}")


# --- Schemas de entrada e saída (validação automática via Pydantic) ---
class VagaInput(BaseModel):
    texto: str = Field(..., min_length=20, description="Texto completo da descrição da vaga")


class FitOutput(BaseModel):
    fit_percentual: float
    interpretacao: str


# --- Função de predição (mesma lógica do Notebook 05) ---
def prever_fit(texto_vaga: str) -> dict:
    inputs = tokenizer(
        texto_vaga,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DISPOSITIVO)

    with torch.no_grad():
        saida = modelo(**inputs)
        predicao_normalizada = saida.logits.item()

    fit_percentual = predicao_normalizada * 100
    fit_percentual = max(0, min(100, fit_percentual))

    if fit_percentual >= 70:
        interpretacao = "Fit alto — forte alinhamento com perfil de Ciência de Dados"
    elif fit_percentual >= 45:
        interpretacao = "Fit médio — área adjacente ou parcialmente alinhada"
    elif fit_percentual >= 20:
        interpretacao = "Fit baixo — pouca aderência ao perfil técnico"
    else:
        interpretacao = "Fit muito baixo — provavelmente fora da área de interesse"

    return {"fit_percentual": round(fit_percentual, 1), "interpretacao": interpretacao}


# --- Endpoints ---
@app.get("/")
def raiz():
    return {
        "projeto": "JobMatch AI",
        "status": "online",
        "modelo": "BERTimbau fine-tuned (MAE 12.36)",
        "endpoint_principal": "/prever-fit (POST)",
    }


@app.post("/prever-fit", response_model=FitOutput)
def prever_fit_endpoint(vaga: VagaInput):
    try:
        resultado = prever_fit(vaga.texto)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a predição: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "dispositivo": DISPOSITIVO}

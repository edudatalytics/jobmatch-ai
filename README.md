# JobMatch AI 🎯

Sistema de matching entre currículo e vagas de emprego usando NLP e Deep Learning
(fine-tuning de BERTimbau), aplicado ao meu próprio processo de busca por vaga
júnior em Data Science.

## Status do projeto: completo ✅

## Etapas concluídas
- [x] Definição da arquitetura e problema
- [x] Coleta de 162 vagas reais (15 manuais + 147 via API Adzuna)
- [x] Limpeza e padronização do dataset
- [x] Análise Exploratória (EDA) — N=162
- [x] Tokenização com BERTimbau — validação de limite de tokens
- [x] Embeddings semânticos e visualização (PCA + UMAP)
- [x] Fine-tuning supervisionado do BERTimbau
- [x] Sistema de recomendação (função prever_fit())
- [x] Deploy local (FastAPI + Streamlit)

## Resultado principal

Modelo BERTimbau fine-tuned para prever compatibilidade (fit_percentual, 0-100)
entre uma vaga e o perfil profissional do autor:

| Métrica | Baseline ingênuo | BERTimbau fine-tuned | Melhoria |
|---|---|---|---|
| MAE | 25.22 | **12.36** | 51.0% |
| RMSE | 27.62 | **15.61** | 43.5% |

## Demonstração

![Interface Streamlit](notebooks/streamlit_demo.png)
*(adicione um print da interface funcionando aqui)*

## Como rodar o projeto

### 1. Clonar o repositório
\`\`\`bash
git clone https://github.com/edudatalytics/jobmatch-ai.git
cd jobmatch-ai
\`\`\`

### 2. Reproduzir o modelo treinado
O modelo fine-tuned (~500MB) não está versionado neste repositório devido ao
tamanho. Para reproduzi-lo:
1. Abra os notebooks 01 a 04 no Google Colab, na ordem
2. Rode célula por célula (o Notebook 04 salva o modelo em `models/bertimbau_fit_v1/`)
3. Baixe a pasta do modelo do Google Drive para `models/` neste projeto local

### 3. Rodar a API e a interface
\`\`\`bash
pip install -r app/requirements.txt

# Terminal 1
uvicorn app.main:app --reload

# Terminal 2
streamlit run app/streamlit_app.py
\`\`\`

## Principais descobertas
- Dataset balanceado: 61 vagas de fit alto, 62 de fit baixo, 39 de fit médio
- Separação semântica clara entre vagas técnicas e não-técnicas, validada via
  embeddings e visualização UMAP
- O fine-tuning reduziu o erro médio em 51% comparado a um baseline ingênuo
- Limitação identificada: tanto os embeddings quanto o modelo fine-tuned têm
  dificuldade em capturar nuances de senioridade dentro da mesma área técnica
- Apenas 0.81% dos tokens gerados são [UNK], majoritariamente por truncamento
  na fonte de dados (API) — limitação conhecida e documentada

## Stack técnica
Python · Pandas · Scikit-learn · Hugging Face Transformers (BERTimbau) ·
Sentence-Transformers · PyTorch · FastAPI · Streamlit · Matplotlib · Seaborn ·
UMAP · Google Colab (GPU T4) · API Adzuna

## Estrutura do projeto
\`\`\`
jobmatch-ai/
├── app/
│   ├── main.py              # API FastAPI
│   └── streamlit_app.py      # Interface visual
├── data/
│   └── processed/
│       └── vagas_clean_v2.csv
├── notebooks/
│   ├── 01_EDA_JobMatchAI.ipynb
│   ├── 02_Tokenizacao_BERTimbau.ipynb
│   ├── 03_Embeddings_Similaridade.ipynb
│   ├── 04_Fine_tuning_do_BERTimbau.ipynb
│   └── 05_Sistema_de_Recomendação.ipynb
└── models/                   # (gerado localmente, não versionado)
\`\`\`

## Limitações conhecidas
- Dataset de 162 vagas é pequeno para deep learning — resultados são uma
  prova de conceito metodologicamente correta, não um modelo pronto para
  produção em larga escala
- O campo `remoto` não pôde ser extraído de forma confiável da API Adzuna
- Modelo apresenta regressão à média nos extremos da distribuição de fit
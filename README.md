# JobMatch AI 🎯

Sistema de matching entre currículo e vagas de emprego usando NLP e Deep Learning
(fine-tuning de BERTimbau), aplicado ao meu próprio processo de busca por vaga
júnior em Data Science.

## Status do projeto: em desenvolvimento 🚧

## Etapas concluídas
- [x] Definição da arquitetura e problema
- [x] Coleta manual de 15 vagas reais (LinkedIn/Gupy)
- [x] Limpeza e padronização do dataset
- [x] Análise Exploratória (EDA)
- [ ] Tokenização com BERTimbau
- [ ] Fine-tuning
- [ ] Embeddings e busca semântica
- [ ] Deploy (FastAPI + Streamlit)

## Principais descobertas da EDA
- Vagas variam de 31 a 1081 caracteres
- Vocabulário de vagas "bom fit" (dados, soluções, projetos) claramente
  diferente de vagas "fora de área" (implantação, sustentação, atendimento)

## Estrutura do projeto
\`\`\`
jobmatch-ai/
├── data/
├── notebooks/
├── models/
\`\`\`
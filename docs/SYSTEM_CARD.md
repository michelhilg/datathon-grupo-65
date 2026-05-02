# System Card — Churn Retention Assistant

**Versão**: 2.0.0  
**Data**: 2026-05-02  
**Referência**: Mitchell et al. (2019) — Model Cards for Model Reporting

---

## Descrição do Sistema

Assistente inteligente para análise de risco de churn em clientes de telecomunicações. Combina modelo ML de classificação com agente ReAct e recuperação de contexto (RAG) para gerar recomendações personalizadas de retenção.

---

## Arquitetura

```
Input (features do cliente)
  → InputGuardrail (injeção, tamanho, escopo)
  → ReAct Agent (gpt-4o-mini)
      ├── Tool: churn_predictor  (Random Forest → probabilidade + risk level)
      ├── Tool: feature_importance  (top 5 fatores de risco)
      └── Tool: retention_knowledge  (RAG ChromaDB → estratégias de retenção)
  → OutputGuardrail (remoção de PII via Presidio)
  → Resposta estruturada (análise + recomendação)
```

---

## Uso Pretendido

**Cenário de uso**: Analistas de retenção de operadoras de telefonia para priorizar clientes em risco e personalizar ofertas.

**Fora do escopo**:
- Decisões automatizadas de cancelamento sem revisão humana
- Análise de clientes de outros setores
- Substituição de avaliação jurídica em casos de disputa contratual

---

## Avaliação de Qualidade

| Métrica | Valor | Conjunto |
|---------|-------|----------|
| RAGAS Faithfulness | — | Golden set (20 pares) |
| RAGAS Answer Relevancy | — | Golden set |
| RAGAS Context Precision | — | Golden set |
| RAGAS Context Recall | — | Golden set |
| LLM-as-judge (relevância, precisão, acionabilidade) | — | 20 amostras |

*Valores preenchidos após execução de `evaluation/ragas_eval.py`.*

---

## Segurança

- Guardrails de input: detecção de prompt injection + limite de tamanho + filtragem de tópico
- Guardrails de output: remoção de PII via Presidio (PERSON, EMAIL, PHONE)
- OWASP Top 10 LLM: 6 ameaças mapeadas e mitigadas (ver `OWASP_MAPPING.md`)
- Red team: 6 cenários adversariais testados (ver `RED_TEAM_REPORT.md`)

---

## Limitações

- Dependência de API externa OpenAI — indisponibilidade afeta o endpoint `/analyze` mas não `/predict`
- Modelo ML treinado em dataset IBM Telco (EUA) — pode não generalizar para perfis de clientes brasileiros
- Drift de conceito não dispara retraining automático — threshold PSI > 0.2 gera alerta mas requer intervenção manual
- Respostas em linguagem natural podem conter imprecisões — não substituem análise humana

---

## Conformidade

- **LGPD**: Plano documentado em `LGPD_PLAN.md`
- **Fairness**: Análise por `SeniorCitizen` documentada em `MODEL_CARD.md`
- **Explicabilidade**: Feature importance disponível via endpoint `/predict`
- **Monitoramento**: Prometheus + Langfuse + Evidently (drift detection)

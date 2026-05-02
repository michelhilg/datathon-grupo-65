# Model Card — Churn Prediction Model

**Referência**: Mitchell et al. (2019) — Model Cards for Model Reporting  
**Data**: 2026-05-02

---

## Detalhes do Modelo

| Campo | Valor |
|-------|-------|
| Tipo | Random Forest Classifier |
| Framework | scikit-learn |
| Tarefa | Classificação binária (churn = 1 / não-churn = 0) |
| Tracking | MLflow (experimento registrado em `mlruns/`) |
| Versão selecionada | Maior AUC no Model Registry |

---

## Dados de Treinamento

| Campo | Valor |
|-------|-------|
| Dataset | IBM Telco Customer Churn |
| Tamanho | 7.043 clientes |
| Taxa de churn | ~26,5% |
| Features usadas | 20 (após feature engineering: tenure, charges, serviços, contrato, pagamento) |
| Split | 80% treino / 20% teste (stratified, random_state=42) |
| Versionamento | DVC (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc`) |

---

## Métricas de Desempenho

| Métrica | Valor (holdout 20%) |
|---------|-------------------|
| AUC-ROC | — |
| F1-Score | — |
| Precision | — |
| Recall | — |

*Valores preenchidos após execução de `src/models/train.py` e consulta ao MLflow UI.*

---

## Fairness — Análise por Subgrupos

| Subgrupo | AUC | Observação |
|----------|-----|-----------|
| SeniorCitizen = 1 | — | Grupo potencialmente vulnerável (LGPD) |
| SeniorCitizen = 0 | — | Grupo majoritário |
| Contract = Month-to-month | — | Maior taxa de churn histórica |
| Contract = Two year | — | Menor taxa de churn histórica |

*Disparidade AUC > 5pp entre subgrupos requer revisão antes de produção.*

---

## Explicabilidade

Feature importance nativa do Random Forest disponível via endpoint `/predict` (top 5 fatores). Interpretação por fator:

- **tenure**: clientes com menor tempo de contrato têm maior risco
- **MonthlyCharges / TotalCharges**: clientes com mensalidade alta relativa ao tempo de contrato
- **Contract_Month-to-month**: maior flexibilidade = maior propensão ao cancelamento
- **InternetService_Fiber optic**: associado a insatisfação por custo

---

## Limitações

- Treinado em dados de operadora norte-americana (IBM dataset) — possível viés de distribuição para mercado brasileiro
- Não captura eventos externos (promoções de concorrentes, crises econômicas)
- Predição pontual — não modela sequência temporal de comportamento do cliente
- Retraining manual — não há pipeline automático de retraining via drift trigger

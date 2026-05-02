# Plano de Conformidade LGPD

**Sistema**: Churn Retention Assistant — Telco  
**Lei**: Lei nº 13.709/2018 (LGPD)  
**Data**: 2026-05-02

---

## Dados Tratados

| Dado | Categoria | Base Legal | Retenção |
|------|-----------|-----------|----------|
| Tenure (meses) | Contratual | Execução de contrato (Art. 7º, V) | Enquanto vigente |
| MonthlyCharges / TotalCharges | Financeiro | Execução de contrato | Enquanto vigente |
| Tipo de contrato, serviços ativos | Contratual | Execução de contrato | Enquanto vigente |
| SeniorCitizen (0/1) | Sensível (age proxy) | Legítimo interesse (Art. 7º, IX) | Anonimizado após análise |
| Probabilidade de churn (output ML) | Inferido | Legítimo interesse | Não armazenado |

**Dados ausentes no dataset**: nome, CPF, endereço, e-mail — não são coletados nem processados.

---

## Medidas Técnicas Implementadas

**Minimização**: O pipeline de feature engineering remove `CustomerID` antes do treino. O modelo opera sobre features agregadas, não identidades individuais.

**Dados sintéticos em testes**: `tests/conftest.py` usa exclusivamente fixtures sintéticas geradas programaticamente, sem dados reais de clientes.

**Anonimização de output**: `OutputGuardrail` aplica Presidio para remover PII residual de respostas do LLM antes de expô-las ao cliente da API.

**Sem armazenamento de predições individuais**: O drift detector mantém uma janela deslizante de features agregadas (sem identificadores). Langfuse armazena traces de rastreabilidade operacional, sem dados pessoais nos campos de input.

---

## Direitos do Titular

| Direito (Art. 18) | Como atender |
|-------------------|-------------|
| Acesso | Exportar registros do MLflow associados ao CustomerID |
| Correção | Reprocessar análise com features corrigidas |
| Eliminação | Remover CustomerID do dataset e retreinar modelo |
| Explicação | Feature importance disponível via `/predict` (top 5 fatores) |
| Oposição ao tratamento automatizado | Human-in-the-loop recomendado para decisões de cancelamento |

---

## Encarregado de Dados (DPO)

Responsável indicado pelo grupo para fins de demonstração: michelhilg@gmail.com  
Canal de contato para exercício de direitos: disponibilizado no README do sistema.

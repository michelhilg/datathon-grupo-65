# Fatores de Risco de Churn — Guia de Avaliação por Perfil

## Como avaliar o risco de churn de um cliente

O risco de churn é determinado pela combinação de múltiplos fatores. Este guia descreve como identificar
o nível de risco a partir das features do cliente e qual intervenção aplicar.

## Fatores de alto risco (cada um eleva significativamente o risco)

| Fator | Impacto no risco | Observação |
|---|---|---|
| Contrato month-to-month | +++ | 42,7% de churn histórico |
| Tenure ≤ 1 mês (primeiro mês) | +++ | 65% de churn — período mais crítico |
| Tenure 2-6 meses | ++ | 57% de churn — onboarding crítico |
| Fibra óptica sem TechSupport | ++ | Insatisfação com custo/desempenho |
| Electronic check como pagamento | ++ | 45,3% de churn histórico |
| MonthlyCharges > R$70 sem serviços adicionais | ++ | Percepção de custo-benefício ruim |
| SeniorCitizen sem TechSupport | +++ | 52% de churn neste grupo |
| 0 serviços adicionais | ++ | 55% de churn histórico |
| Sem OnlineSecurity | ++ | 41,8% de churn histórico |
| Sem TechSupport | ++ | 41,6% de churn histórico |

## Fatores protetores (cada um reduz o risco)

| Fator | Impacto protetor | Observação |
|---|---|---|
| Contrato de 2 anos | --- | 2,8% de churn — 15x menor que mês-a-mês |
| Contrato anual | -- | 11,3% de churn |
| Tenure > 24 meses | -- | 15% de churn |
| Tenure > 48 meses | --- | < 5% de churn |
| Pagamento automático (cartão ou débito) | -- | 3x menor churn que electronic check |
| OnlineSecurity presente | -- | Reduz churn em 27 pp |
| TechSupport presente | -- | Reduz churn em 26 pp |
| Partner (parceiro) | - | Churn 20% vs 33% sem parceiro |
| Dependents (dependentes) | -- | Churn 15% vs 31% sem dependentes |
| 3+ serviços adicionais | -- | 12% de churn |

## Classificação de risco por perfil

### ALTO RISCO — churn acima de 70%

**Perfil A1: Cliente novo em risco crítico**
- Tenure: 0-2 meses
- Contrato: month-to-month
- Serviços adicionais: nenhum ou poucos
- Pagamento: qualquer (risco agravado com electronic check)
- Probabilidade de churn estimada: 65-80%
- Ação: intervenção imediata nas primeiras 48h, trial gratuito de OnlineSecurity, oferta de contrato anual

**Perfil A2: Cliente tenure baixo com fibra sem proteção**
- Tenure: 2-12 meses
- Contrato: month-to-month
- Internet: fiber optic
- Sem OnlineSecurity e sem TechSupport
- Pagamento: electronic check
- MonthlyCharges: > R$70
- Probabilidade de churn estimada: 70-85%
- Ação: bundle OnlineSecurity + TechSupport com 3 meses grátis, migração para contrato anual com 20% desconto

**Perfil A3: Idoso sem suporte**
- SeniorCitizen: 1
- Tenure: < 12 meses
- Contrato: month-to-month
- Sem TechSupport
- Probabilidade de churn estimada: 70-80%
- Ação: contato telefônico proativo, TechSupport com desconto especial para idosos (30%), desconto permanente de 10% na mensalidade

**Perfil A4: Alta cobrança sem valor percebido**
- MonthlyCharges: > R$80
- Contrato: month-to-month
- Sem TechSupport e sem OnlineSecurity
- Pagamento: electronic check
- Probabilidade de churn estimada: 72-85%
- Ação: oferecer OnlineSecurity + TechSupport para justificar o valor pago, desconto na mensalidade condicionado à migração para contrato anual

**Perfil A5: Aniversário de 1 ano em mês-a-mês**
- Tenure: 12 meses
- Contrato: month-to-month (não migrou para anual)
- Internet: fiber optic
- Sem serviços de suporte
- Probabilidade de churn estimada: 55-70%
- Ação: oferta de aniversário — desconto de 20-25% para migrar para contrato de 2 anos + bundle de serviços

### RISCO MÉDIO — churn entre 30-70%

**Perfil M1: Idoso com contrato anual sem suporte**
- SeniorCitizen: 1
- Contrato: one year (fator protetor moderado)
- Sem TechSupport
- Pagamento: automático (fator protetor)
- Probabilidade de churn estimada: 35-50%
- Ação: oferta de TechSupport antes da renovação anual, contato proativo 60 dias antes do vencimento

**Perfil M2: Tenure médio em mês-a-mês com alguma proteção**
- Tenure: 12-24 meses
- Contrato: month-to-month
- 1-2 serviços adicionais (ex: OnlineSecurity)
- Pagamento: automático
- Parceiro: sim (fator protetor)
- Probabilidade de churn estimada: 35-50%
- Ação: migração incentivada para contrato anual antes dos 24 meses, bundle com OnlineBackup + DeviceProtection

**Perfil M3: Contrato anual sem segurança digital**
- Contrato: one year
- Internet: fiber optic
- OnlineBackup ou DeviceProtection presentes, mas sem OnlineSecurity e sem TechSupport
- Pagamento: mailed check (fator de risco)
- Probabilidade de churn estimada: 30-45%
- Ação: adicionar TechSupport + OnlineSecurity, migrar pagamento para automático, propor renovação para 2 anos

**Perfil M4: DSL com tenure curto sem adicionais**
- Internet: DSL (menor risco que fibra)
- Tenure: 8-12 meses
- Contrato: month-to-month
- Sem serviços adicionais
- Pagamento: automático por cartão (fator protetor)
- Probabilidade de churn estimada: 30-45%
- Ação: oferta de primeiro serviço adicional com desconto, proposta de contrato anual com mensalidade congelada

**Perfil M5: Cliente só-telefone com mês-a-mês**
- Sem internet (apenas PhoneService/MultipleLines)
- Tenure: 18-24 meses
- Contrato: month-to-month
- Pagamento: electronic check (fator de risco)
- Probabilidade de churn estimada: 25-40%
- Ação: migração para contrato anual do serviço de telefone, oferta de internet DSL introdutória, migração de pagamento

### BAIXO RISCO — churn abaixo de 30%

**Perfil B1: Cliente premium multi-serviços**
- Tenure: > 48 meses
- Contrato: two year
- Todos ou maioria dos serviços adicionais
- Parceiro e dependentes
- Pagamento: automático
- Probabilidade de churn estimada: < 5%
- Ação: upsell de StreamingTV/Movies, programa VIP de fidelidade, oferta de upgrade de velocidade

**Perfil B2: Cliente estável com contrato anual**
- Tenure: 30-60 meses
- Contrato: one year
- 2-3 serviços adicionais (OnlineSecurity, OnlineBackup)
- Parceiro e/ou dependentes
- Pagamento: automático por cartão
- Probabilidade de churn estimada: 10-20%
- Ação: oferta de migração para contrato de 2 anos com congelamento, adicionar TechSupport ou DeviceProtection, programa de indicação

**Perfil B3: Veterano de longa data**
- Tenure: > 60 meses
- Contrato: two year
- Todos os serviços contratados
- Pagamento: automático
- Probabilidade de churn estimada: 2-5%
- Ação: benefícios de lealdade, upgrade de velocidade, programa de embaixador da marca

**Perfil B4: Cliente só-telefone fiel**
- Apenas PhoneService
- Tenure: > 36 meses
- Contrato: two year ou one year
- Parceiro e dependentes
- Pagamento: automático
- Probabilidade de churn estimada: < 5%
- Ação: oferta de internet DSL para aumentar LTV, programa de fidelidade, upsell de MultipleLines

## Regras de decisão rápida

**Se tenure ≤ 2 meses → ALTO RISCO, intervenção imediata obrigatória**

**Se SeniorCitizen = 1 e sem TechSupport → ALTO RISCO, prioridade TechSupport**

**Se contrato mês-a-mês e tenure = 12 meses → ALTO RISCO, oferta de aniversário urgente**

**Se MonthlyCharges > R$70 e 0 serviços adicionais → ALTO RISCO, bundle de proteção**

**Se contrato 2 anos e tenure > 24 meses → BAIXO RISCO, foco em upsell**

**Se pagamento automático e 3+ serviços → BAIXO RISCO, programa de indicação**

**Se electronic check e mês-a-mês → pelo menos MÉDIO RISCO, migrar pagamento é prioritário**

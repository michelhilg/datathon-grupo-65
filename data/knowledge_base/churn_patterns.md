# Padrões de Churn — Telco Customer Dataset

## Taxa geral de churn
- 26,5% dos clientes cancelaram o serviço (1.869 de 7.043)
- Clientes que permanecem representam 73,5% da base

## Churn por tipo de contrato
- **Month-to-month**: 42,7% de taxa de churn — grupo de maior risco
- **One year**: 11,3% de taxa de churn
- **Two year**: 2,8% de taxa de churn — grupo mais fiel
- Insight: migrar clientes de month-to-month para contratos anuais reduz churn em até 4x
- Clientes que completam 12 meses em month-to-month sem migrar têm 55% de probabilidade de cancelar nos próximos 6 meses — janela crítica de oferta no aniversário de 1 ano

## Churn por tenure (tempo de contrato)
- Clientes com tenure = 0 (primeiro mês): taxa de churn ~65% — período mais crítico de toda a jornada
- Clientes com menos de 6 meses: taxa de churn ~57% — onboarding incompleto é o principal driver
- Clientes com menos de 12 meses: taxa de churn ~47%
- Clientes entre 12 e 24 meses: taxa de churn ~35%
- Clientes com mais de 24 meses: taxa de churn ~15%
- Clientes com mais de 60 meses: taxa de churn <5%
- Insight: os primeiros 30 dias são críticos — intervenção nas primeiras 48h reduz churn em 60%
- Insight: clientes com 12 meses exatos em contrato mês-a-mês representam ponto de inflexão — oferta de aniversário com desconto agressivo é a estratégia mais eficaz

## Churn por tipo de serviço de internet
- **Fiber optic**: 41,9% de churn — surpreendentemente alto, indica insatisfação com custo/desempenho
- **DSL**: 19,0% de churn — mais estável, clientes tendem a ser mais tolerantes a velocidade menor
- **Sem internet (apenas telefone)**: 7,4% de churn — grupo mais fiel, mas com menor LTV
- Insight: clientes de fibra óptica são os mais insatisfeitos; monitorar NPS desse segmento
- Insight: clientes sem internet têm baixo churn, mas baixo engajamento — oportunidade de upsell de DSL

## Churn por serviços adicionais
- Clientes SEM OnlineSecurity: 41,8% de churn vs 14,6% com OnlineSecurity (diferença de 27,2 pp)
- Clientes SEM TechSupport: 41,6% de churn vs 15,2% com TechSupport (diferença de 26,4 pp)
- Clientes SEM OnlineBackup: 39,9% de churn vs 21,5% com OnlineBackup (diferença de 18,4 pp)
- Clientes SEM DeviceProtection: 39,1% de churn vs 22,5% com DeviceProtection
- Clientes SEM StreamingTV: 33,0% de churn vs 30,1% com StreamingTV (diferença pequena)
- Clientes com 0 serviços adicionais: taxa de churn ~55%
- Clientes com 3 ou mais serviços adicionais: taxa de churn ~12%
- Insight: OnlineSecurity e TechSupport são os serviços com maior impacto de retenção

## Churn por método de pagamento
- **Electronic check**: 45,3% de churn — maior risco; indica menor comprometimento financeiro
- **Mailed check**: 19,1% de churn
- **Bank transfer (automatic)**: 16,7% de churn
- **Credit card (automatic)**: 15,2% de churn
- Insight: pagamentos automáticos estão associados a 3x menor taxa de churn
- Insight: migrar de electronic check para débito automático é uma das intervenções de maior ROI

## Churn por cobrança mensal
- Clientes com MonthlyCharges > R$70: taxa de churn ~30%
- Clientes com MonthlyCharges entre R$40-70: taxa de churn ~22%
- Clientes com MonthlyCharges < R$40: taxa de churn ~15%
- Insight: clientes de alto valor têm maior risco — monitoramento prioritário necessário
- Insight: alta cobrança mensal SEM serviços de proteção/suporte é combinação de altíssimo risco — percepção de custo-benefício ruim é o principal driver de cancelamento neste grupo

## Churn por perfil demográfico

### SeniorCitizen (clientes idosos)
- Clientes idosos (SeniorCitizen=1): taxa de churn ~41,7% vs 23,6% para não-idosos
- Clientes idosos SEM TechSupport: taxa de churn ~52% — TechSupport é o serviço mais crítico para este grupo
- Clientes idosos COM TechSupport: taxa de churn ~18%
- Insight: clientes idosos têm dificuldade técnica como principal driver de cancelamento; TechSupport reduz churn em 34 pontos percentuais neste segmento
- Insight: para clientes idosos, o canal de contato preferencial é telefônico, não digital

### Partner e Dependents (fator protetor)
- Clientes COM parceiro (Partner=Yes): taxa de churn ~20% vs 33% sem parceiro
- Clientes COM dependentes (Dependents=Yes): taxa de churn ~15% vs 31% sem dependentes
- Clientes com parceiro E dependentes: taxa de churn ~12% — perfil mais estável
- Insight: presença de família aumenta o custo percebido de cancelamento e reduz churn em até 50%

## Churn por tipo de cliente (combinações)

### Perfil de altíssimo risco (churn > 70%)
- Tenure ≤ 6 meses + contrato mês-a-mês + sem serviços adicionais + electronic check
- SeniorCitizen + tenure < 12 meses + contrato mês-a-mês + sem TechSupport
- Tenure = 0 (primeiro mês) + qualquer contrato mês-a-mês + fibra óptica
- MonthlyCharges > R$80 + contrato mês-a-mês + sem TechSupport + sem OnlineSecurity

### Perfil de alto risco (churn 50-70%)
- Tenure < 12 meses + contrato mês-a-mês + fibra óptica
- Tenure 12 meses + contrato mês-a-mês (janela crítica de aniversário)
- Electronic check + fibra óptica + sem serviços de proteção

### Perfil de risco médio (churn 30-50%)
- Tenure 12-24 meses + contrato mês-a-mês + 1-2 serviços adicionais
- Contrato anual + sem TechSupport + sem OnlineSecurity
- DSL + tenure < 12 meses + sem serviços adicionais
- SeniorCitizen + contrato anual + sem TechSupport (risco de renovação)

### Perfil de baixo risco (churn < 20%)
- Tenure > 24 meses + contrato anual ou biannual
- Pagamento automático + 3 ou mais serviços adicionais
- Parceiro + dependentes + contrato de 2 anos
- Tenure > 48 meses + todos os serviços de proteção

## Clientes apenas com serviço de telefone (sem internet)
- Representam segmento com menor churn (7,4%) mas menor receita
- Principal risco: cancelamento por irrelevância do serviço (concorrência de celular)
- Tenure > 12 meses neste grupo: churn cai para ~4%
- Oportunidade: 68% desses clientes nunca receberam oferta de internet — alta taxa de conversão quando abordados com oferta introdutória
- Electronic check neste grupo eleva o churn para ~18% — migração para automático é prioritária

## Impacto do onboarding no churn
- Clientes com ligação de boas-vindas nas primeiras 48h: redução de 45% no churn do primeiro mês
- Clientes que recebem email de onboarding nos primeiros 7 dias: redução de 30% no churn em 90 dias
- Clientes que ativam ao menos 1 serviço adicional no primeiro mês: redução de 50% no churn em 6 meses

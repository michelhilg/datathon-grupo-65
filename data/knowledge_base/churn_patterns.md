# Padrões de Churn — Telco Customer Dataset

## Taxa geral de churn
- 26,5% dos clientes cancelaram o serviço (1.869 de 7.043)
- Clientes que permanecem representam 73,5% da base

## Churn por tipo de contrato
- **Month-to-month**: 42,7% de taxa de churn — grupo de maior risco
- **One year**: 11,3% de taxa de churn
- **Two year**: 2,8% de taxa de churn — grupo mais fiel
- Insight: migrar clientes de month-to-month para contratos anuais reduz churn em até 4x

## Churn por tenure (tempo de contrato)
- Clientes com menos de 12 meses: taxa de churn ~47%
- Clientes entre 12 e 24 meses: taxa de churn ~35%
- Clientes com mais de 24 meses: taxa de churn ~15%
- Clientes com mais de 60 meses: taxa de churn <5%
- Insight: os primeiros 12 meses são críticos — intervenção precoce é mais eficaz

## Churn por tipo de serviço de internet
- **Fiber optic**: 41,9% de churn — surpreendentemente alto, pode indicar insatisfação com custo/desempenho
- **DSL**: 19,0% de churn
- **Sem internet**: 7,4% de churn
- Insight: clientes de fibra óptica são os mais insatisfeitos; monitorar NPS desse segmento

## Churn por serviços adicionais
- Clientes SEM OnlineSecurity: 41,8% de churn vs 14,6% com OnlineSecurity
- Clientes SEM TechSupport: 41,6% de churn vs 15,2% com TechSupport
- Clientes SEM OnlineBackup: 39,9% de churn vs 21,5% com OnlineBackup
- Insight: oferecer serviços de proteção e suporte reduz churn significativamente

## Churn por método de pagamento
- **Electronic check**: 45,3% de churn — maior risco
- **Mailed check**: 19,1% de churn
- **Bank transfer (automatic)**: 16,7% de churn
- **Credit card (automatic)**: 15,2% de churn
- Insight: incentivar migração para pagamento automático reduz churn em ~3x

## Churn por cobrança mensal
- Clientes com MonthlyCharges > R$70: taxa de churn ~30%
- Clientes com MonthlyCharges entre R$40-70: taxa de churn ~22%
- Clientes com MonthlyCharges < R$40: taxa de churn ~15%
- Insight: clientes de alto valor têm maior risco — monitoramento prioritário necessário

## Perfil do cliente de alto risco
Combinação que maximiza risco de churn:
- Contrato: Month-to-month
- Tenure: < 12 meses
- Serviço: Fiber optic
- Sem OnlineSecurity e sem TechSupport
- Pagamento: Electronic check
- MonthlyCharges: > R$70

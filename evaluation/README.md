# 📊 Benchmark do Agente ReAct

Para cumprir a exigência da **Etapa 2**, foram avaliadas três configurações diferentes (temperatura e `top_k` do RAG) em três cenários de clientes (Risco Alto, Risco Médio e Risco Baixo).

O arquivo bruto dos testes foi gerado e salvo em `benchmark_results.json`. Abaixo está o resumo documentado do nosso benchmark:

## ⚙️ Configurações Avaliadas

* **`config_A`**: `top_k = 3`, `temperature = 0.0`
* **`config_B`**: `top_k = 5`, `temperature = 0.0`
* **`config_C`**: `top_k = 3`, `temperature = 0.3`

*Modelo utilizado em todas as configurações:* `gpt-4o-mini`

---

## 📈 Resumo de Performance

| Configuração | Taxa de Sucesso | Latência Média (ms) | Tamanho Médio da Resposta (chars) | Observações |
| :--- | :---: | :---: | :---: | :--- |
| **config_A** | 3/3 | **14.104 ms** | 1988 | **Mais Rápida.** Respostas mais objetivas e focadas, excelente consistência. |
| **config_B** | 3/3 | 15.408 ms | **2227** | Respostas mais longas, justificado pelo aumento do contexto recuperado (`top_k=5`). |
| **config_C** | 3/3 | 15.828 ms | 1898 | Mais variabilidade na estrutura da resposta devido ao aumento de temperatura (`temp=0.3`). |

---

## 🧑‍💻 Casos de Teste (Cenários)

A avaliação garantiu que a probabilidade matemática do modelo (`churn_predictor`) continuasse sendo interpretada corretamente e com sucesso em todos os casos:

1. **Cliente de Alto Risco (`high_risk_customer`)**
   * *Perfil*: Contrato mês-a-mês, Fibra óptica, tempo de casa baixo (2 meses).
   * *Resultado da ML*: ~83% de probabilidade de churn (Alto Risco).
   * *Comportamento do Agente*: Sugere fortemente descontos de retenção a curto prazo e prioridade máxima no atendimento.

2. **Cliente de Risco Médio (`medium_risk_customer`)**
   * *Perfil*: Contrato de um ano, Fibra, sem suporte técnico, idoso (`SeniorCitizen`).
   * *Resultado da ML*: ~66% de probabilidade de churn (Risco Médio).
   * *Comportamento do Agente*: Focou em melhorar a segurança online e sugerir suporte técnico, evitando churn no fim do contrato.

3. **Cliente de Baixo Risco (`low_risk_customer`)**
   * *Perfil*: Contrato de 2 anos, DSL, todos serviços de suporte, com dependentes.
   * *Resultado da ML*: ~9% de probabilidade de churn (Baixo Risco).
   * *Comportamento do Agente*: Recomendou manter bom relacionamento e focar em estratégias de *upsell*, em vez de medidas agressivas de retenção.

---

### 🏆 Conclusão

A **`config_A`** provou ser a mais estável e rápida para produção. Um `top_k` de 3 com temperatura 0.0 garante que o agente foque cirurgicamente nos dados que recebe e evite "alucinar" nas respostas de retenção, mantendo uma boa velocidade de API (~14s com o `gpt-4o-mini`).

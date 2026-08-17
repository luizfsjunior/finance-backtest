# Conclusões do backtester — teses testadas e veredictos

Histórico das estratégias testadas neste projeto, o que cada uma acredita, como
foi testada, e o que os resultados mostraram. Cada seção fecha uma tese; o que
sobra vira insumo para a próxima.

Universo padrão: 10 ativos líquidos da B3 (PETR4, VALE3, ITUB4, BBDC4, WEGE3,
ABEV3, B3SA3, RENT3, SUZB3, RADL3). Long-only. Capital inicial R$ 10.000,
risco de 1% por trade, custos e slippage padrão, benchmark buy-and-hold com os
mesmos custos.

---

## 1. Cruzamento de médias (MovingAverageCrossover 9/21) — REFUTADA

**Tese:** tendências continuam. Se a média rápida cruza a lenta para cima, o
movimento tem inércia e vai seguir; se cruza para baixo, saia.

**Testes:** 8 batches nos mesmos 10 tickers, dois regimes principais
(2018-2024 em alta ampla, 2022 em queda), variando stop percentual vs ATR e
seus parâmetros.

**Veredicto:**

- Não bate B&H em nenhum regime. 1/10 em alta (2018-2024), 2/10 em queda
  (2022).
- As poucas vitórias são sempre o mesmo mecanismo: a estratégia só ganha
  quando o B&H do ativo foi negativo. Ela perde menos por ficar fora do
  mercado — é ausência de participação, não proteção ativa.
- **Achado transferível:** stop ATR domina stop percentual em drawdown
  (10/10) e é robusto nos dois regimes. Ponto ótimo do multiplicador:
  2.0–2.5 (platô real). 3.0 reduz drawdown mais mas piora Sharpe.
  `AtrStop(period=14, multiplier=2.0)` vira o default para novas estratégias.

**Adendo (Componente 3 do laboratório, `experiments/mac_robustness.yaml`):** a
config 9/21 + ATR 2.0 foi submetida a 19 perturbações do ambiente (leave-one-out
dos 10 papéis, início ±1 e ±3 meses, custo ×0 e ×2, terços do período) e saiu
**ROBUSTA** — Sharpe entre 0.21 e 0.58, baseline 0.45, mediana retendo 103%,
nenhuma perturbação virando negativa. E **perde do buy-and-hold em 20 de 20
execuções.**

Isso não reabilita a tese; reforça o veredicto e o mecanismo já descrito. Um
Sharpe modesto, positivo e estabilíssimo é a assinatura de quem fica pouco tempo
exposto ao mercado — que é exatamente o diagnóstico do item anterior. O que o
adendo acrescenta é sobre a **bancada**: robustez mede dependência do ambiente,
não qualidade. As duas perguntas se respondem separadas, e um "ROBUSTA" isolado
nunca é aprovação de tese (ver D4.1 do `SPEC_ROBUSTNESS.md`).

O ponto mais frágil apareceu no sub-período 2/3 (2020-01 a 2021-12, o choque da
pandemia e a recuperação): Sharpe 0.21, 46% do baseline. Consistente com uma
estratégia de seguimento de tendência levando chicotada em reversão violenta.

---

## 2. Bollinger conservadora (BollingerReversion 20/2.0) — REFUTADA

**Tese:** contraponto filosófico do cruzamento — exageros voltam ao normal. O
preço tem gravidade em torno da média móvel de 20 dias; quando cai mais de
2 desvios abaixo, aposta que volta. Entrada na borda de cruzamento da banda
inferior, saída na borda de cruzamento da média central (versão conservadora
— não espera reversão até a banda superior).

**Escolha de projeto:** sem stop de verdade (`NoStop`, um "stop de fachada" a
50% ou 10% abaixo da entrada, quase inalcançável). A tese diz "quanto mais
cai, melhor a entrada"; um stop apertado corta o trade exatamente quando ele
ficaria mais válido. Para medir a tese pura, o stop precisa ficar fora do
caminho.

**Testes:**

| Cenário | Placar | Retorno médio estratégia | Retorno médio B&H |
|---|---:|---:|---:|
| 2018-2024, NoStop 50% (sizing anêmico — 2% do capital por trade) | 1/10 | 0.06% | 173.8% |
| 2018-2024, NoStop 10% (sizing comparável ao MAC) | 1/10 | -0.76% | 173.8% |
| 2015-2016, NoStop 10% (regime supostamente lateral) | 0/8 | 0.01% | 51.3% |

**Veredicto:**

- **A tese localmente funciona.** Win rate ~60% em todos os batches — o preço
  realmente reverte à média com boa frequência. O problema não é
  capacidade preditiva local.
- **Payoff assimétrico no lado errado.** Ganha pouco quando acerta, perde
  razoavelmente quando erra. O produto disso não cobre o custo de
  oportunidade de ficar fora do mercado.
- **Mesmo mecanismo do MAC nas vitórias.** ABEV3 em 2018-2024 (única vitória)
  foi o único ativo com B&H negativo. Estratégias que reduzem exposição só
  ganham onde o mercado perde. Isso é ausência de exposição, não proteção
  ativa — o mesmo padrão que refutou o MAC.
- **A hipótese "vai brilhar em regime lateral" foi refutada** em 2015-2016.
  Descoberta secundária importante: **"mercado lateral" no índice não
  implica ativos individuais laterais.** A maioria dos papéis líquidos teve
  alta ampla em 2015-2016 (RADL3 +151%, B3SA3 +91%, PETR4 +59%) apesar do
  Ibovespa reputadamente lateral. Estratégia opera papel-a-papel, então
  precisaria de ativos individuais laterais para brilhar.
- **Corrigir o sizing não mudou o veredicto** (1/10 nos dois batches de
  2018-2024) — o problema não era dimensionamento, era a tese não gerar
  retorno significativo mesmo quando acerta.

**Não voltar** para Bollinger em ações individuais. Se um dia testar reversão
de novo, considerar: ETFs/pares (instrumentos mais estacionários), filtro de
tendência (só entrar em ativo com MA 200 subindo), ou saída na banda oposta
(aumenta payoff, testa a assimetria).

---

## 3. Momentum time-series 12-1 (TimeSeriesMomentum) — REFUTADA

**Tese:** ativos com retorno positivo nos últimos 12 meses (excluindo o mês
mais recente, para não pegar reversão de curto prazo) tendem a continuar
subindo. Anomalia estruturalmente documentada — Jegadeesh & Titman (1993)
cross-sectional, Moskowitz-Ooi-Pedersen (2012) time-series (a versão usada
aqui). Escolhemos time-series (single-asset) primeiro porque cabe no motor
existente sem reescrever para portfolio.

**Formulação:** entra quando `preço(D-21) / preço(D-252) − 1` cruza para
positivo, sai quando cruza para negativo. Warm-up de 252 dias. Sem stop
(`NoStop` de fachada), pelo mesmo argumento da Bollinger — a tese diz "estou
comprado enquanto a tendência existir", o stop briga menos aqui mas não
ajuda a testar a tese pura.

**Testes:**

| Cenário | Placar | Retorno médio estratégia | Sharpe médio | Trades médios |
|---|---:|---:|---:|---:|
| 2017-2024, 12-1 puro, sem stop | 0/10 | +8.6% | 0.25 | 23.1 |
| 2017-2024, 12-1 + banda morta 2% + rebalance mensal | 0/10 | +8.3% | 0.22 | 4.5 |

**Veredicto:**

- **A tese funciona onde a tendência é evidente para o olho humano.** PETR4
  Sharpe 0.89 = B&H 0.86 (com metade do drawdown!); WEGE3, VALE3, SUZB3 com
  Sharpe positivo razoável. Momentum captura tendências limpas.
- **Falha onde a tendência precisa ser detectada.** RENT3 e ABEV3 com Sharpe
  negativo: sinal atrasa a virada e come todo o lucro.
- **Whipsaw diagnosticado e neutralizado, mas retorno não mudou.** A
  hipótese "sem whipsaw, tese anda" foi refutada: banda morta 2% +
  rebalance mensal reduziu trades de 23 para 4.5 sem mudar retorno médio
  (8.6% → 8.3%). O whipsaw não era o problema principal — era ruído em
  cima do fato de fundo.
- **Ninguém bate B&H, nem em Sharpe (0.22 vs 0.64).** Mesmo sendo a menos
  ruim das três famílias em retorno absoluto (única com retorno positivo
  médio), momentum single-asset ainda perde estruturalmente para "comprar e
  segurar" em ativos brasileiros líquidos no período testado.
- **Detalhe elegante:** PETR4 fez zero trades no batch refinado — o
  momentum ficou o tempo todo acima de +2%. Estratégia = buy-and-hold com
  atraso de warm-up. Ilustra literalmente o limite do que single-asset
  long/flat pode ser em ativo com tendência clara.

---

## 4. Cross-sectional momentum 12-1 (top-K por ranking mensal) — REFUTADA

**Tese:** mesma anomalia do momentum time-series, mas aplicada
transversalmente — rankeia todos os ativos do universo por retorno 12-1 a
cada mês e mantém os top-K sempre comprados. Elimina o "custo de
oportunidade estrutural" das três primeiras famílias por construção: nunca
está fora do mercado, só escolhe em quais estar. Se o padrão macro
consolidado nas três anteriores fosse *exclusivo* de single-asset long/flat,
essa família deveria virar o jogo.

**Escolha de projeto:** motor novo (`portfolio_backtest.py`), Protocol
próprio (`PortfolioStrategy` devolve scores contínuos, não sinais discretos),
rebalanceamento mensal, K fixo com resto em caixa se universo elegível <K,
sem stops (rebalance substitui), custos por perna em cada rotação, benchmark
equal-weight buy-and-hold do próprio universo (comparação apples-to-apples)
+ IBOV.

**Testes:**

| Cenário | Estratégia (ret aa / Sharpe) | B&H EW (ret aa / Sharpe) | IBOV (ret aa / Sharpe) |
|---|---:|---:|---:|
| 2017-2024, universo 10, top-3 mensal | 10.81% / 0.52 | **20.39% / 0.91** | 12.54% / 0.60 |
| 2017-2024, universo 25 (expandido), top-3 mensal | 21.03% / 0.73 | **26.45% / 0.96** | 12.54% / 0.60 |

**Veredicto:**

- **A tese localmente funciona muito melhor que single-asset.** Win rate
  61.5% no universo maior, PnL médio positivo de R$412/trade, retorno
  absoluto de 21% aa, holding médio ~5.5 meses. Não é uma estratégia ruim
  no vácuo — apenas insuficiente contra o baseline correto.
- **Escala do universo importa mais do que a tese em si.** Aumentar de 10
  para 25 tickers multiplicou o retorno da estratégia por 2.6x sem
  qualquer outra mudança. Confirma que cross-sectional precisa de universo
  grande para diferenciar (top-3 de 10 = 30% de concentração, quase
  "quem entrou no ranking"; top-3 de 25 = 12%, seleção real).
- **Mas o benchmark também escala — e mais.** B&H EW foi de 20.4% para
  26.5% aa; expandir o universo ajuda mais quem "não faz nada" do que
  quem seleciona. Universo maior = mais chances de pegar um foguete (PRIO3
  ~50x no período, capturado automaticamente pelo B&H com peso 1/29).
- **Concentração intensifica o drawdown.** Ao concentrar em 3 ativos que
  já subiram, o portfolio ficou mais frágil a reversões: drawdown 60% vs
  49% do B&H EW no universo expandido — 11 pontos percentuais a mais de
  fragilidade sem retorno adicional que compense.
- **Bate o IBOV, mas isso não salva a tese.** 21% aa vs 12.5% do IBOV
  parece bom até você comparar com o B&H EW do mesmo universo (26.5% aa).
  O ganho sobre o IBOV vem de estar em ações líquidas selecionadas em vez
  do índice inteiro — não da rotação por momentum.

**Não voltar** para cross-sectional momentum puro em universo restrito de
ações brasileiras líquidas. Se um dia testar de novo, considerar: universo
de 100+ ativos (Ibovespa+ small caps), filtro de qualidade (momentum +
volatilidade baixa, "Sharpe do lookback"), ou setor concentrado (uma cesta
homogênea onde a dispersão intra-setor é real).

---

## Padrão macro consolidado — beco reafirmado por quatro caminhos independentes

**Estratégias de seleção baseadas em sinal técnico simples — sejam elas
long/flat single-asset ou top-K cross-sectional — perdem estruturalmente
para "comprar e segurar" o próprio universo em ações brasileiras líquidas
no período 2015-2024.** As quatro famílias testadas — tendência local
(MAC), reversão (Bollinger), tendência de médio prazo single-asset
(momentum time-series), tendência de médio prazo cross-sectional (momentum
top-K) — chegaram ao mesmo veredicto por caminhos filosoficamente e
estruturalmente diferentes.

O padrão é robusto porque cada família ataca o problema por um ângulo
distinto:

- **MAC vs Bollinger** = tendência vs reversão (filosofias opostas)
- **Time-series vs cross-sectional** = decisão isolada por ativo vs
  seleção transversal no universo (arquiteturas opostas de portfolio)
- **Single-asset long/flat vs top-K portfolio** = "carregar ou não" vs
  "quais carregar" (elimina o custo de oportunidade estrutural, ainda
  perde)

Que os quatro cheguem ao mesmo lugar por caminhos ortogonais é evidência
que o problema **não é escolha de indicador nem de arquitetura** — é o
próprio **drift positivo do universo no período**. Ações líquidas
brasileiras 2015-2024 subiram forte e amplo o bastante para que qualquer
regra de seleção pague um custo (de oportunidade, de rotação, de
concentração) que o retorno adicional não cobre.

**A única forma dessas estratégias "vencerem"** foi por ausência de
exposição em ativos que caíram (single-asset long/flat) ou por bater um
benchmark que não é o correto (IBOV em vez do B&H EW do próprio universo,
no caso cross-sectional). Nenhuma delas entregou valor genuíno versus o
baseline honesto.

**Decisão de curso:** parar de testar variações incrementais dessa família
ampla. Duas direções qualitativamente diferentes que valeria a pena
explorar em algum momento:

1. **Regime que efetivamente puna o buy-and-hold** — bear market
   prolongado real do Ibovespa (2011-2015), ou período de correlação
   descolada (crise específica). Se as mesmas estratégias vencerem lá, o
   veredicto vira "essas famílias são condicionais ao regime bull", não
   "não funcionam nunca".

2. **Instrumentos ou universos estruturalmente diferentes** — pares
   estatísticos (long-short em ativos cointegrados), commodities/futuros
   (mean-reversion funciona melhor em séries com âncora fundamental), ou
   universo internacional (dispersão real entre países/setores). Todos
   exigem infra nova e mudam qualitativamente a tese testada.

Nenhum dos dois é "próximo passo natural" — são reinícios em outro nicho.
O aprendizado central desta rodada é o veredicto negativo consolidado
sobre uma família ampla de teses, que é resultado científico legítimo:
sabemos agora que uma classe grande de abordagens não funciona nesse
contexto específico, e por quê.

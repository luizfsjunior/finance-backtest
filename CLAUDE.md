# CLAUDE.md — instruções para o assistente neste projeto

Este projeto é um **backtester de estratégias quantitativas** para ações da B3,
usado como bancada de pesquisa: testar teses, refutá-las quando não sobrevivem
a custos/regimes, registrar o veredicto. O tom da documentação e dos comentários
reflete essa postura — cético, focado em separar sinal de sorte.

## Fluxo de trabalho neste repositório

Este projeto segue **SDD + TDD** (ver instruções globais). Especificamente:

1. Specs entram em `SPEC_*.md` (`SPEC_LAB.md`, `SPEC_ROBUSTNESS.md`). Quando uma
   spec existente deixa decisões em aberto, o detalhamento vira um `SPEC_*.md`
   novo com uma seção por decisão e o motivo escrito, e a spec original ganha um
   ponteiro — não se decide em silêncio dentro do código.
2. Testes em `tests/` são fornecidos ANTES da implementação.
3. Só implementar após specs + testes explícitos. Se algo estiver ambíguo, parar
   e perguntar — nunca assumir default silencioso.

## Arquitetura (mapa mental)

Duas trilhas paralelas, mesma disciplina anti-look-ahead:

**Single-asset** (`main.py` → `backtest.py`):
- `strategy.py`: contrato `Strategy` (Protocol). Cada implementação recebe OHLCV
  e devolve `Series[Signal]` — eventos de borda, não estado contínuo.
  Estratégias: `MovingAverageCrossover`, `BollingerReversion`, `TimeSeriesMomentum`.
- `stops.py`: contrato `StopLoss`. Implementações: `FixedPctStop`, `AtrStop`,
  `NoStop`. Separado da estratégia de propósito — permite trocar regra de risco
  sem tocar no setup.
- `backtest.py`: motor. Consome sinais + stop + custos, mantém estado (cash,
  posição), aplica risk-per-trade sizing. Execução na abertura de D+1 quando
  o sinal vem em D (nunca lê o futuro).
- `costs.py`, `metrics.py`: custos (corretagem + slippage bps) e relatório
  (retorno, Sharpe, Sortino, drawdown, payoff, expectância).
- `data.py`: fetch com cache em parquet (`data_cache/`), filtro de liquidez.

**Portfolio cross-sectional** (`portfolio_backtest.py`):
- `portfolio_strategy.py`: contrato baseado em `rank()` — score por ticker em D.
- Motor mantém `dict[ticker, quantity]`, rebalanceia em cadência declarada
  (mensal/semanal/trimestral), top-K fixo (sobra vai pra caixa). Sem stops.

**Batches** (`batch.py`, `portfolio_batch.py`): varrem os 10 tickers default e
gravam uma linha por execução em `runs.csv` / `portfolio_runs.csv`. Cada batch
exige uma `--hypothesis` — o CSV sem isso é inútil em 3 meses.

**Plot** (`plot_runs.py`): lê o CSV e desenha comparações. Cada gráfico responde
a UMA pergunta.

**Laboratório** (`sweep.py`): Componente 1 do `SPEC_LAB.md`, camada por cima do
pipeline. Varre grade declarada (produto cartesiano determinístico), filtra
combinações inválidas ANTES de rodar (tentando construir estratégia/stop —
`ValueError` filtra, `TypeError` sobe), tem teto de combinações e grava
proveniência (`sweep_id`, `combo_id`, `n_combos`, `train_test`) em
`sweep_runs.csv` — separado do `runs.csv` para não afogar o acervo curado dos
batches.

**Experimento como arquivo** (`lab.py` + `experiments/*.yaml`): Componente 5 da
spec. YAML (não JSON) por causa dos comentários — o arquivo registra por que
aquela grade. Registro FECHADO de classes (`STRATEGIES`/`STOPS`), nunca
`getattr`. Campo desconhecido, classe inexistente ou hipótese vazia = erro
explícito, nunca default silencioso. Seções `walk_forward` e
`perturbations` são mutuamente exclusivas no mesmo arquivo (escolher a config e
estressá-la são etapas diferentes). O Componente 6 (visualizações) ainda não
existe; o 4 é disciplina de leitura, com o suporte de dados (`n_combos`,
`n_perturbations`, distribuição impressa) já no lugar — falta só a contagem de
"espiadas" de um mesmo período.

**Walk-forward** (`walkforward.py`): Componente 2, esquema `expanding` só
(`rolling` levanta erro em vez de rodar expanding com nome errado). Defesa
anti-vazamento é ESTRUTURAL: o sweep de treino roda com `end = train_end`, então
os dados de teste nem são lidos. Qualquer código novo que faça a seleção enxergar
o bloco de teste é bug crítico — `tests/test_walkforward.py` cobre isso por dois
ângulos. WFE = razão das MÉDIAS (não média das razões) e é `None` com in-sample
≤ 0. Limitação assumida: o bloco de teste roda sozinho, então o warm-up da
estratégia consome o início dele — subestima a tese, nunca superestima.

**Robustez** (`robustness.py` + `SPEC_ROBUSTNESS.md`): Componente 3. A spec do
`SPEC_LAB.md` deixava decisões em aberto; elas estão FECHADAS no
`SPEC_ROBUSTNESS.md`, uma seção por decisão (D1..D6) — discordar exige mudar lá,
não adivinhar aqui. Os dois invariantes que não se negociam: a config perturbada
é DECLARADA (grade de valor único; 2+ combinações = erro) e **não existe função
que escolha a melhor perturbação** — a saída é a distribuição e um veredicto
severo (ROBUSTA exige survival 100% e retenção mediana ≥ 50%). Qualquer código
novo que perturbe a estratégia/stop em vez do ambiente é bug crítico:
`tests/test_robustness.py::TestConfiguracaoNaoMuda` é o equivalente aqui ao
teste anti-vazamento do walk-forward. Números são in-sample e não substituem o
WFE. **ROBUSTA significa ESTÁVEL, nunca SUPERIOR** (D4.1): o MAC já refutado sai
ROBUSTA aqui e perde do buy-and-hold em 20 de 20 execuções — Sharpe baixo,
positivo e estável passa nos limiares. Por isso o relatório imprime o placar
contra o benchmark ao lado do veredicto. Nunca ler ROBUSTA como aprovação de tese.

## Convenções não óbvias

- **Nunca olhar o futuro.** Sinal em D → execução na abertura de D+1. Qualquer
  código novo que quebre isso é bug crítico. O motor materializa passando
  `prices.iloc[:i+1]` para a estratégia no passo `i`.
- **Custos aplicados no benchmark também.** O buy-and-hold do relatório usa a
  MESMA tabela de custos da estratégia — comparar sob custos assimétricos é
  vazamento sutil de vantagem.
- **Batch nunca aborta em ticker ruim.** Falhas viram entrada em `failures[]`;
  o batch continua.
- **Encoding no Windows.** `main.py` chama `sys.stdout.reconfigure(encoding="utf-8")`
  porque o console default do Windows (cp1252/cp850) corrompe os acentos do
  relatório.
- **Determinismo é obrigatório.** Mesma configuração rodada duas vezes tem que
  gerar exatamente os mesmos números. O cache de dados ajuda; qualquer código
  novo que introduza aleatoriedade sem seed é bug.
- **Furo de determinismo conhecido e ABERTO no cache.** `data._covers_range`
  compara a primeira barra do cache com a data PEDIDA: `start` em feriado/fim de
  semana nunca "cobre", o parquet é reescrito, e como o preço ajustado depende
  do range baixado a mesma data volta com diferença na 7ª casa. Afeta quem varre
  vários períodos (sweep, walk-forward, robustez). `robustness._warm_cache`
  mitiga; a correção é em `data.py` e ainda não foi feita — ver os gotchas do
  README. Antes de investigar "resultado mudou sozinho", checar isto primeiro.

## Sobre commits

- **NÃO incluir `Co-Authored-By: Claude`** em mensagens de commit (regra global
  do usuário).
- No fluxo "commit+push", fazer passada dedicada de revisão de README/CLAUDE.md
  imediatamente antes do commit — atualizar o que a mudança tornou enganoso.

## O que NÃO fazer

- Não reescrever o motor para acomodar uma feature nova. Se parece necessário,
  parar e reconsiderar — quase sempre há caminho aditivo (o laboratório do
  `SPEC_LAB.md` é explicitamente uma camada POR CIMA, não substituição).
- Não construir otimizadores de parâmetros (gradiente, bayesiano). Sweep
  exaustivo declarado é mais honesto e mais legível.
- Não paralelizar antes de precisar. Sequencial e lento > paralelo com bug de
  estado compartilhado.
- Não celebrar teses aprovadas antes dos 4 obstáculos do `SPEC_LAB.md` (vizinhança
  de parâmetros, walk-forward, perturbação, correção por número de tentativas).
  Bancada honesta rejeita mais do que aprova. E um veredicto ROBUSTA sozinho não
  é aprovação: ele responde "depende do ambiente exato?", nunca "a tese presta?".

## Testes

```
pytest
```
Rodar antes de considerar qualquer implementação terminada. Casos em `tests/`
cobrem cada módulo isoladamente.

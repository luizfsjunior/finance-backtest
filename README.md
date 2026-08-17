# finance-backtest

Bancada de backtest para estratégias quantitativas em ações da B3. Escrita
menos como "framework" e mais como um instrumento de laboratório: submete teses
a custos, regimes e obstáculos, e reporta o veredicto. O objetivo declarado é
**derrubar teses**, não aprová-las. Se muitas teses "vencerem", a primeira
hipótese é vazamento no motor — não que você ficou bom em achar alpha.

Long-only, capital inicial R$ 10.000, risco de 1% por trade, custos + slippage
configuráveis. Universo default: 10 ativos líquidos da B3 (PETR4, VALE3, ITUB4,
BBDC4, WEGE3, ABEV3, B3SA3, RENT3, SUZB3, RADL3). Benchmark: buy-and-hold sob
a **mesma** tabela de custos.

> Documento de projeto. Não é recomendação de investimento.

---

## O que resolve

Testar uma hipótese de mercado (ex.: "tendências continuam", "exageros voltam")
sob condições realistas — custos, slippage, filtro de liquidez, sem look-ahead —
e produzir um relatório comparável entre execuções. Ao longo do tempo, o acervo
em `CONCLUSOES.md` vira histórico das teses testadas e refutadas.

Já refutadas neste projeto (ver `CONCLUSOES.md` para o detalhe): cruzamento de
médias 9/21, Bollinger conservadora 20/2.0, Bollinger simétrica, momentum
time-series 12-1.

---

## Arquitetura

Duas trilhas com a mesma disciplina anti-look-ahead. Sinal em D → execução na
abertura de D+1. Sempre.

### Single-asset (`main.py` como entrada)

```
data.py  ──►  strategy.py  ──►  backtest.py  ──►  metrics.py  ──►  relatório
                    │              │       ▲
                    │              │       │
                  (sinais)     (custos, stop)
                                stops.py  costs.py
```

- **`strategy.py`** — Protocol `Strategy`. Recebe OHLCV, devolve `Series[Signal]`
  (eventos de borda, não estado). Implementações: `MovingAverageCrossover`,
  `BollingerReversion`, `TimeSeriesMomentum`.
- **`stops.py`** — Protocol `StopLoss`. Implementações: `FixedPctStop`, `AtrStop`,
  `NoStop`. Separado da estratégia de propósito — trocar regra de risco sem
  tocar no setup e vice-versa é o que permite atribuir performance corretamente.
- **`backtest.py`** — o motor. Estado interno (cash + posição única), sizing por
  risco (`risk_pct` × capital / distância ao stop), custos aplicados em cada
  fill. Rastreia `skipped_signals` (motivos: capital insuficiente, sinal
  duplicado, etc.).
- **`costs.py`** — `Costs(brokerage, slippage_bps)`.
- **`metrics.py`** — retorno total/anualizado, drawdown máx., Sharpe, Sortino,
  número de trades, taxa de acerto, payoff médio, expectância, tempo médio em
  posição. Buy-and-hold sob os mesmos custos.
- **`data.py`** — fetch via yfinance, cache em parquet (`data_cache/`), filtro
  de liquidez (`min_median_turnover`, default R$ 1MM/dia).

### Portfolio cross-sectional (`portfolio_backtest.py` como entrada)

Contrapartida multi-asset. Diferenças estruturais:
- Estratégia expõe `rank()` — score por ticker em D. Não é evento discreto.
- Estado interno: `cash` + `dict[ticker, quantity]`.
- Rebalance em cadência declarada (mensal/semanal/trimestral). Top-K fixo —
  se o universo elegível tem menos que K, aloca 1/K nos elegíveis e o resto
  vai pra caixa. Preserva o significado de "K posições" em warm-up.
- **Sem stops.** Saída é decidida no próximo rebalance.

### Batches e log

- **`batch.py`** — varre os 10 tickers default, uma execução por ticker, grava
  linha em `runs.csv`. Achata o `MetricsReport` por introspecção — o CSV
  acompanha mudanças no schema sem precisar editar o script. Se o schema mudar
  entre execuções, escreve novo cabeçalho em vez de desalinhar silenciosamente.
  Requer `--hypothesis` (o CSV sem isso é inútil em 3 meses).
- **`portfolio_batch.py`** — equivalente para o motor multi-asset.
  Grava em `portfolio_runs.csv`.
- **`plot_runs.py`** — lê o CSV e desenha comparações. Regra: cada gráfico
  responde a UMA pergunta. Serve exclusivamente o `runs.csv`: os gráficos do
  laboratório (sweep, walk-forward, robustez) vivem separados, em `plot_lab.py`
  — spec fechada em `SPEC_PLOTS.md`, ainda não implementado.

### Laboratório — sweep de parâmetros (`sweep.py`)

Primeira peça do `SPEC_LAB.md`, camada POR CIMA do que já existe: varre uma
grade declarada de parâmetros, uma execução do pipeline normal por combinação
por ticker. Responde "o bom resultado é um platô ou um pico?".

- `expand_grid` faz o produto cartesiano em ordem determinística.
- `valid_combos` descarta combinações inválidas **antes** de rodar. A regra de
  validade não é reimplementada: ele tenta construir a estratégia/stop e
  descarta o que levantar `ValueError` (`MovingAverageCrossover` já rejeita
  `fast >= slow`). Um `TypeError` (nome de parâmetro errado na grade) sobe —
  senão um typo viraria "todas inválidas" em silêncio.
- Teto de combinações (`max_combos`, default 256): sweep grande demais aborta
  antes de escrever qualquer linha.
- `aggregate_by_combo` resume cada combinação na média da métrica no universo;
  `print_distribution` imprime o topo do ranking **e** a distribuição inteira
  (suporte do obstáculo 4).

Colunas de proveniência gravadas em toda linha: `sweep_id`, `combo_id`,
`n_combos`, `train_test`. `n_combos` é o que sustenta o obstáculo 4 — o mesmo
Sharpe vale menos como melhor de 200 tentativas do que como resultado único.
`train_test` é preenchido por quem chama, para o walk-forward (Componente 2)
reusar este sweep marcando cada janela.

**Desvio consciente da spec:** o sweep grava em `sweep_runs.csv`, não em
`runs.csv`. O `runs.csv` é o acervo curado dos batches que sustentam o
`CONCLUSOES.md` — uma linha por decisão de pesquisa. Um único sweep despeja
centenas de linhas e afogaria esse histórico.

### Laboratório — experimento como arquivo (`lab.py`, `experiments/*.yaml`)

Componente 5 do `SPEC_LAB.md`. Um experimento é universo + período +
estratégia + grade + métrica de seleção — ou seja, um documento. Documento se
versiona: commitado junto do resultado, seis meses depois você recria a
varredura exata. É por isso que a entrada é arquivo e não formulário web.

YAML e não JSON por causa dos **comentários**: o arquivo é onde se registra por
que aquela grade e não outra, e essa justificativa é metade do valor dele. Ver
`experiments/mac_plateau.yaml`, que serve de template.

Postura de erro: **nada de default silencioso.** Campo com nome errado, classe
inexistente, seção obrigatória faltando e hipótese vazia levantam erro nomeando
o problema. Num laboratório cujo produto é o veredicto, um typo que vira default
é pior que um crash — o experimento roda, o resultado sai, e responde a uma
pergunta diferente da que você fez.

Duas decisões que valem explicação:

- **Registro fechado de classes** (`STRATEGIES`/`STOPS` em `lab.py`): o YAML
  escolhe por nome dentro de um dicionário e nada mais. Resolver por `getattr`
  num módulo deixaria um arquivo de experimento instanciar qualquer coisa
  importável, e um nome errado passaria a depender do acaso.
- **`walk_forward` declarado é erro, não campo ignorado**, enquanto o Componente
  2 não existir. Aceitar o campo e rodar um sweep simples devolveria resultado
  in-sample com aparência de out-of-sample — o pior desfecho possível aqui.

O `sweep_id` de cada linha carrega o nome do experimento
(`mac_plateau-2026-08-17T13:22:01`), então a linha do CSV diz de qual arquivo
ela saiu.

### Laboratório — walk-forward e WFE (`walkforward.py`)

Componente 2 do `SPEC_LAB.md`, o obstáculo que de fato separa robusto de
sortudo: **os parâmetros escolhidos no passado funcionam no futuro que não foi
usado para escolhê-los?** Esquema `expanding` — origem do treino fixa, treino
cresce absorvendo o bloco de teste anterior, cada período é out-of-sample
exatamente uma vez.

**WFE (Walk-Forward Efficiency)** = retorno out-of-sample / retorno in-sample,
anualizados. Acima de ~50-60% sugere robustez real; consistentemente baixa é
overfitting. É a saída principal do laboratório.

Decisões que valem explicação:

- **A defesa anti-vazamento é estrutural, não disciplinar.** A fase de treino
  roda um sweep cujo `SweepSpec` tem `end = train_end`: os dados do teste nem
  chegam a ser lidos, não existe caminho de código em que a seleção veja o
  futuro. `tests/test_walkforward.py` verifica por dois ângulos independentes —
  as datas efetivamente pedidas em cada fase, e a invariância da escolha quando
  só o futuro é perturbado.
- **WFE agregada usa a razão das médias, não a média das razões.** Uma janela
  com in-sample perto de zero dominaria a média de razões e viraria o veredicto
  por artefato de divisão.
- **WFE é `None` quando o in-sample é ≤ 0.** "Manteve 200% da performance" de um
  treino que perdeu dinheiro não significa nada. Já um out-of-sample negativo é
  informação legítima e vira WFE negativa.
- **Janela de teste incompleta é descartada** — anualizar meio bloco como se
  fosse um bloco inteiro inventa performance.
- **Seleção e WFE usam métricas diferentes de propósito:** a combinação é
  escolhida por `select_by` (Sharpe, tipicamente) e o WFE é sempre sobre retorno
  anualizado. São perguntas distintas ("qual escolher" vs "quanto sobreviveu").

**Limitação conhecida — warm-up dentro do teste.** O bloco de teste roda
sozinho, começando em `test_start`, então uma estratégia com lookback longo
(momentum 12-1 precisa de 252 barras) passa o início do bloco sem poder operar.
Carregar histórico anterior só para aquecer exigiria o motor devolver equity de
um sub-trecho — reescrever o núcleo por causa de uma feature. A escolha é
conservadora de propósito: **subestima** a estratégia, nunca a superestima, e
nada do teste vaza para a decisão. Use `test_years` grande o bastante.

**O que não está coberto:** o "fitting implícito" (você olha o out-of-sample,
ajusta a grade e roda de novo — contaminando o teste com a própria memória).
Nenhum código detecta isso; `n_combos` cobre só a parte mecânica do obstáculo 4.

### Laboratório — robustez a perturbação (`robustness.py`)

Componente 3 do `SPEC_LAB.md`, detalhado em `SPEC_ROBUSTNESS.md` (cada decisão
com o motivo escrito). Pergunta: **a tese aguenta mexer no setup, ou só funciona
na configuração exata em que foi achada?** Pega uma config única já escolhida e
roda sob quatro eixos de perturbação:

| eixo | pergunta | default |
|---|---|---|
| `leave_one_out` | um papel sortudo segura o resultado? | ligado (exige ≥ 3 tickers) |
| `start_shift_months` | depende do ponto exato de partida? | `[-3, -1, 1, 3]` |
| `cost_multipliers` | a margem sobrevive a execução pessimista? | `[0.0, 2.0]` |
| `subperiods` | funciona ao longo do tempo ou só num pedaço? | `3` |

Decisões que valem explicação:

- **A config é declarada, nunca inferida.** A grade tem que produzir exatamente
  uma combinação válida; duas ou mais é erro. Ler automaticamente a vencedora do
  walk-forward esconderia a instabilidade da escolha entre janelas atrás de um
  "a da última janela" arbitrário. Pelo mesmo motivo, `walk_forward` e
  `perturbations` no mesmo arquivo é erro — são etapas diferentes.
- **Não existe função que escolha a melhor perturbação.** A saída é a
  distribuição inteira e um veredicto conservador. Perturbar até achar a
  variação que sobrevive é overfitting disfarçado de teste de robustez, e a
  defesa aqui é estrutural: a porta não existe. É o que o teste-armadilha
  `TestConfiguracaoNaoMuda` trava — nenhum eixo pode alterar estratégia ou stop.
- **Veredicto deliberadamente severo:** ROBUSTA exige que *todas* as
  perturbações mantenham a métrica > 0 **e** que a mediana retenha ≥ 50% do
  baseline. Qualquer outra coisa é FRÁGIL; baseline ≤ 0 é N/A. Um limiar
  tolerante deixaria passar exatamente o caso que o componente existe para pegar.
- **Percentis por rank mais próximo, sem interpolar.** Com 12 ou 19 pontos,
  interpolar inventa um valor que nenhuma perturbação produziu; aqui todo número
  reportado aconteceu de verdade.
- **Os números são in-sample** (a config já estava escolhida) e não substituem o
  WFE. O relatório imprime esse aviso.

Grava em `robustness_runs.csv` com `run_id`, `perturbation_kind`,
`perturbation_value` e `n_perturbations` (contagem separada de `n_combos`: uma
conta tentativas de tuning, a outra de estresse). `train_test` vale
`"perturbation"` — não é `full` (sub-período não é o período inteiro) nem
`train`/`test` (nada aqui é out-of-sample).

---

## Setup

Python 3.11+ recomendado.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependências: `pandas`, `numpy`, `pyarrow` (cache parquet), `yfinance`,
`matplotlib`, `pyyaml` (config de experimento), `pytest`.

## Uso

### Execução única (single-asset)

```powershell
python main.py PETR4.SA 2018-01-01 2024-01-01 --stop atr --atr-multiplier 2.0 --plot
```

Principais flags (ver `python main.py -h` para lista completa):
- `--fast-window` / `--slow-window` — janelas do cruzamento de médias (default 9/21).
- `--stop {pct,atr}` + `--stop-pct` / `--atr-period` / `--atr-multiplier`.
- `--initial-capital`, `--risk-pct`, `--brokerage`, `--slippage-bps`.
- `--min-turnover` — R$/dia mediano mínimo (0 desliga).
- `--plot` — mostra a equity curve × buy-and-hold.

### Batch (10 tickers, uma execução por ticker)

```powershell
python batch.py --strategy mac --stop atr --atr-multiplier 2.0 --hypothesis "ATR 2.0 domina stop percentual em drawdown"
```

Estratégias: `mac` (MovingAverageCrossover), `bollinger` (BollingerReversion),
`momentum` (TimeSeriesMomentum). Cada uma tem seus próprios parâmetros — ver
`python batch.py -h`.

Cada linha do `runs.csv` fica com `batch_id`, `hypothesis`, ticker, parâmetros
usados e todas as métricas do relatório.

### Portfolio

```powershell
python portfolio_batch.py --top-k 3 --rebalance monthly --hypothesis "..."
```

### Sweep de parâmetros

```powershell
python sweep.py --strategy mac --fast-window 5,9,20 --slow-window 21,50 `
  --stop atr --atr-multiplier 2.0,2.5 --select-by sharpe `
  --hypothesis "o 9/21 do MAC era pico ou platô?"
```

Listas são separadas por vírgula. Grava em `sweep_runs.csv` e imprime o topo do
ranking mais a distribuição de todas as combinações. `--max-combos` sobe o teto
(default 256), `--log-path` muda o destino, `--select-by` escolhe a métrica do
ranking.

### Experimento declarado em arquivo

```powershell
python lab.py run experiments/mac_plateau.yaml --dry-run   # mostra o plano
python lab.py run experiments/mac_plateau.yaml             # executa
```

`--dry-run` lista as combinações válidas e o total de backtests antes de gastar
os minutos. `--max-combos` sobrescreve o teto do arquivo. Template comentado:
`experiments/mac_plateau.yaml`.

### Walk-forward

Basta o experimento declarar a seção `walk_forward` — aí `lab.py run` já executa
walk-forward em vez de sweep simples, e grava em `walkforward_runs.csv`:

```yaml
walk_forward:
  scheme: expanding
  train_years: 3
  test_years: 1
```

```powershell
python walkforward.py experiments/mac_walkforward.yaml   # atalho equivalente
```

O relatório sai por janela (combinação escolhida, in-sample, out-of-sample, WFE)
e fecha com a WFE agregada.

### Robustez a perturbação

Mesma ideia: o experimento declara a seção `perturbations` sobre uma grade de
**valor único**, e `lab.py run` executa robustez, gravando em
`robustness_runs.csv`:

```yaml
strategy:
  class: MovingAverageCrossover
  grid: {fast_window: 9, slow_window: 21} # config única — nada de varrer aqui

perturbations:
  leave_one_out: true
  start_shift_months: [-3, -1, 1, 3]
  cost_multipliers: [0.0, 2.0]
  subperiods: 3
```

```powershell
python lab.py run experiments/mac_robustness.yaml --dry-run  # lista as perturbações
python robustness.py experiments/mac_robustness.yaml         # atalho equivalente
```

O relatório lista as perturbações **da pior para a melhor** (é a pior que carrega
a informação), imprime a distribuição (pior / p25 / mediana / p75 / melhor) e
fecha com ROBUSTA, FRÁGIL ou N/A. Template comentado:
`experiments/mac_robustness.yaml`.

### Plot do log

```powershell
python plot_runs.py
```

### Testes

```powershell
pytest
```

Cobrem cada módulo isoladamente (`tests/test_backtest.py`, `test_strategy.py`,
`test_stops.py`, `test_costs.py`, `test_metrics.py`, `test_data.py`,
`test_main.py`, `test_sweep.py`, `test_lab.py`, `test_walkforward.py`,
`test_robustness.py`).

---

## Decisões não óbvias (gotchas)

- **Sinal em D → fill em D+1 na abertura.** Escolha do motor, não da estratégia.
  A estratégia é chamada em cada passo `i` recebendo `prices.iloc[:i+1]` — nunca
  vê barras futuras. Qualquer código novo que quebre isso é bug crítico.
- **Buy-and-hold do relatório usa a mesma tabela de custos.** Comparar sob custos
  assimétricos é vazamento sutil de vantagem para a estratégia. A linha do
  gráfico é a mesma que gerou os números impressos.
- **Encoding no Windows.** `main.py` chama `sys.stdout.reconfigure(encoding="utf-8")`
  porque o console default (cp1252/cp850) corrompe os acentos do relatório.
- **`batch.py` não aborta em ticker ruim.** Uma falha vira entrada em `failures[]`
  e o batch continua para os demais tickers.
- **Sizing.** Quantidade = `risk_pct × capital / distância_ao_stop`, arredondada
  para o lote. Se o capital não cobre 1 lote, o sinal vai para `skipped_signals`
  com motivo — não silenciosamente ignorado.
- **`--hypothesis` obrigatória em batches.** Roda mesmo sem, mas com aviso.
  Preencher é convenção do projeto.
- **Filtro de liquidez.** Default R$ 1MM/dia (mediana). `--min-turnover 0`
  desliga. Aplicado ANTES do backtest — ticker ilíquido gera exceção clara,
  não resultado enganoso.
- **Determinismo.** Mesma configuração → mesmos números. O cache parquet ajuda.
  Nada aleatório sem seed. **Com uma exceção conhecida, ver abaixo.**
- **Furo de determinismo no cache (aberto).** `data._covers_range` compara a
  primeira barra do cache com a data **pedida**, não com o primeiro pregão
  existente. Um `start` que caia em fim de semana ou feriado (`2018-01-01`, por
  exemplo) nunca é considerado coberto, e o cache daquele ticker é **reescrito**
  a cada chamada. Como o preço ajustado do yfinance depende do range baixado, a
  mesma data volta com diferença na sétima casa decimal, e quem varre vários
  ranges (sweep com períodos diferentes, walk-forward, robustez) pode ver o
  resultado mudar entre execuções idênticas. Medido: rodando
  `experiments/mac_robustness.yaml` duas vezes, o Sharpe do pior sub-período foi
  de `0.20635` para `0.20633` — irrelevante para o veredicto, fatal para a
  regra. `robustness._warm_cache` mitiga (busca o range mais amplo antes de
  começar), mas a correção real é em `data.py`, que o Componente 0 do
  `SPEC_LAB.md` declara fora do escopo do laboratório. **Ainda não corrigido.**

---

## Roadmap — Laboratório de pesquisa

Ver `SPEC_LAB.md`. Camada aditiva por cima do motor, adicionando os 4 obstáculos
que separam robusto de sortudo:

1. **Sweep de parâmetros** — platô vs pico. ✅ implementado (`sweep.py`).
2. **Walk-forward + WFE** — a tese funciona em dados que ela não escolheu?
   ✅ implementado (`walkforward.py`), esquema `expanding` só.
3. **Robustez a perturbação** — leave-one-out de ativo, deslocamento de datas,
   custos dobrados, sub-períodos. ✅ implementado (`robustness.py`).
4. **Correção por número de tentativas** — a melhor de 200 combinações não vale
   o que valeria um resultado único. Parcial: `n_combos` e `n_perturbations`
   estão gravados e a distribuição é impressa; falta a contagem de "espiadas".

Config declarativo (YAML): ✅ implementado (`lab.py` + `experiments/*.yaml`).
Web app só depois de o acervo justificar navegar.

Visualizações (6): spec fechada em `SPEC_PLOTS.md` — módulo próprio
(`plot_lab.py`), um subcomando por gráfico (`heatmap`, `dist`, `walkforward`,
`robustness`), uma rodada por gráfico, mediana entre tickers, e a regra de que
o plot LÊ o veredicto em vez de recalculá-lo. Ainda não implementado.

Estado: componentes 1, 2, 3 e 5 prontos. Faltam as visualizações (6) e a
contagem de "espiadas" que o obstáculo 4 pede.

## Histórico de teses

`CONCLUSOES.md` — cada seção fecha uma tese com veredicto e o achado
transferível. Estado atual: 4 famílias refutadas, nenhuma sobrevivente.

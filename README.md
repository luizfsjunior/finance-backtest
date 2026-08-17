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
  responde a UMA pergunta.

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
`test_main.py`, `test_sweep.py`, `test_lab.py`).

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
  Nada aleatório sem seed.

---

## Roadmap — Laboratório de pesquisa

Ver `SPEC_LAB.md`. Camada aditiva por cima do motor, adicionando os 4 obstáculos
que separam robusto de sortudo:

1. **Sweep de parâmetros** — platô vs pico. ✅ implementado (`sweep.py`).
2. **Walk-forward + WFE** — a tese funciona em dados que ela não escolheu?
3. **Robustez a perturbação** — leave-one-out de ativo, deslocamento de datas,
   custos dobrados.
4. **Correção por número de tentativas** — a melhor de 200 combinações não vale
   o que valeria um resultado único.

Config declarativo (YAML): ✅ implementado (`lab.py` + `experiments/*.yaml`).
Web app só depois de o acervo justificar navegar.

Estado: componentes 1 e 5 prontos; o próximo é o walk-forward (2), o obstáculo
que de fato separa robusto de sortudo.

## Histórico de teses

`CONCLUSOES.md` — cada seção fecha uma tese com veredicto e o achado
transferível. Estado atual: 4 famílias refutadas, nenhuma sobrevivente.

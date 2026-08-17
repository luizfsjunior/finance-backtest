# Spec — Componente 3: robustez a perturbação

Detalhamento do Componente 3 do `SPEC_LAB.md`. Aquela seção define a pergunta e
os eixos de perturbação; esta fecha as decisões que ela deixou em aberto, com o
motivo de cada uma. Quem for mexer no `robustness.py` depois precisa discordar
daqui, não adivinhar.

**Pergunta que o componente responde:** a tese aguenta mexer no setup, ou só
funciona na configuração exata em que foi achada?

---

## D1 — Qual configuração é perturbada

**Decisão: uma configuração única, declarada explicitamente no arquivo de
experimento** (a grade tem que produzir exatamente uma combinação válida).

Por quê: a spec diz "a config sobrevivente do Componente 2", mas o walk-forward
pode eleger uma combinação diferente por janela — o próprio
`print_walk_forward` avisa que isso já é sinal ruim. Ler a vencedora
automaticamente do último `WalkForwardResult` esconderia essa instabilidade
atrás de uma escolha arbitrária ("a da última janela"). Obrigando a declarar,
o arquivo de robustez registra qual config você decidiu levar adiante, e o
commit guarda essa decisão.

Consequência: grade com 2+ combinações válidas é `RobustnessError`, não
"perturba a melhor". **Este módulo não escolhe nada** — ver D6.

`walk_forward` e `perturbations` no mesmo arquivo é erro: são etapas
diferentes da mesma investigação, e misturá-las produziria um relatório em que
não dá para dizer qual número veio de onde.

---

## D2 — O que roda em cada perturbação

**Decisão: um sweep de combinação única sobre o período, não um walk-forward
completo.**

Por quê: walk-forward sob cada perturbação seria o ideal teórico, mas o custo é
`perturbações × janelas × combinações × tickers` — uma varredura de horas para
responder a uma pergunta de dispersão. O Componente 2 já respondeu "sobrevive
fora da amostra?"; o Componente 3 pergunta outra coisa: "sobrevive a mexer no
ambiente?".

Consequência assumida e impressa no relatório: **os números do Componente 3 são
in-sample.** Eles não substituem o WFE, e nenhum deles deve ser lido como
performance esperada.

---

## D3 — Eixos de perturbação

Ordem de geração fixa e determinística: baseline → leave-one-out → deslocamento
de início → custo → sub-períodos.

| eixo | o que faz | default |
|---|---|---|
| `baseline` | a config como declarada; a referência, sempre presente | sempre |
| `leave_one_out` | uma execução por ticker, com aquele ticker fora do universo | ligado |
| `start_shift_months` | desloca só a data de início, mantendo o fim | `[-3, -1, 1, 3]` |
| `cost_multipliers` | multiplica corretagem **e** slippage pelo mesmo fator | `[0.0, 2.0]` |
| `subperiods` | divide o período em N blocos contíguos e roda cada um | `3` |

Detalhes que não são óbvios:

- **Leave-one-out exige pelo menos 3 tickers.** Tirar um de dois deixa um
  universo de um papel — a perturbação viraria um backtest single-asset, que
  responde a outra pergunta.
- **Custo multiplica os dois componentes juntos.** Separá-los dobraria o número
  de execuções para distinguir corretagem de slippage, e essa distinção não é a
  pergunta aqui (a pergunta é "a margem sobrevive a execução mais pessimista").
  Fator `0.0` é o teto otimista, `2.0` o pessimista.
- **Deslocamento negativo alonga o período** (pede dados anteriores ao início
  declarado); positivo encurta. Deslocamento de `0` é erro: é o baseline com
  outro nome, e ocuparia uma linha do relatório dizendo nada.
- **Sub-períodos dividem por dias corridos**, em blocos contíguos que cobrem
  exatamente `[start, end]` sem sobreposição. O último bloco absorve o resto da
  divisão inteira.

**Teto:** `max_perturbations`, default 64, contando sem o baseline. Mesmo espírito
do teto do sweep: não é proteção da máquina, é obrigar você a declarar
conscientemente quando está subindo a aposta do obstáculo 4.

---

## D4 — Métrica e leitura

A métrica é a mesma `select_by` do experimento (default `sharpe`). Cada
execução vira um número: a média da métrica sobre os tickers daquela execução —
a mesma agregação que o sweep já usa.

A spec manda **olhar a dispersão, não a média**. O relatório imprime a
distribuição inteira (pior, p25, mediana, p75, melhor) e a lista completa por
perturbação, ordenada da pior para a melhor — a pior primeiro de propósito, é
ela que carrega a informação.

Dois números resumem:

- `survival_rate` — fração das perturbações com métrica **> 0**.
- `median_retention` — `mediana(perturbações) / baseline`, quanto da performance
  sobra na perturbação típica.

**Veredicto (deliberadamente severo):**

```
baseline ausente ou <= 0            → N/A   (não há performance a preservar)
survival_rate == 1.0 e retention >= 0.5 → ROBUSTA
caso contrário                      → FRÁGIL
```

Por que `survival_rate == 1.0` e não 80%: a spec diz que a tese frágil "desaba
com uma única perturbação". Um limiar tolerante deixaria passar exatamente o
caso que o componente existe para pegar — a tese que só funciona porque um
papel sortudo estava no universo. Uma bancada honesta rejeita mais do que
aprova; se este veredicto parecer duro demais na prática, o lugar de mudar é
aqui na spec, com o motivo escrito.

### D4.1 — ROBUSTA mede estabilidade, não superioridade

Descoberto na primeira execução real (`experiments/mac_robustness.yaml`, MAC
9/21 + ATR 2.0): o veredicto saiu **ROBUSTA** para uma tese já refutada. Não é
vazamento — é o critério fazendo exatamente o que foi especificado. Sharpe ~0.45
com dispersão pequena passa em `survival_rate == 1.0` e `median_retention >= 0.5`,
e a estratégia continua perdendo do buy-and-hold em quase toda execução.

Uma estratégia que fica pouco tempo exposta ao mercado tem justamente esse
perfil: retorno modesto, positivo e estabilíssimo.

**Decisão: o veredicto NÃO passa a comparar com o benchmark.** Robustez e
superioridade são perguntas diferentes, e embutir a segunda na primeira faria o
componente responder duas coisas com um rótulo só. Em vez disso, o relatório
imprime ao lado do veredicto **quantas execuções superam o buy-and-hold sob os
mesmos custos** e, quando ROBUSTA convive com derrota na maioria delas, um
aviso explícito de que estável ≠ superior.

Regra de leitura que vale para quem for usar isso: `ROBUSTA` responde "a tese
depende do ambiente exato?". Ela nunca responde "a tese presta?" — para isso
existem o benchmark no relatório e o `CONCLUSOES.md`.

---

## D5 — Saída e proveniência

CSV próprio, `robustness_runs.csv` — mesmo motivo de `sweep_runs.csv` e
`walkforward_runs.csv` serem separados: as linhas têm significados diferentes, e
misturá-las faria a leitura do acervo depender de lembrar qual coluna estava
preenchida.

Colunas acrescentadas às que o sweep já grava:

- `run_id` — identifica a rodada de robustez inteira.
- `perturbation_kind` — `baseline`, `leave_one_out`, `start_shift`,
  `cost_multiplier` ou `subperiod`.
- `perturbation_value` — o valor concreto (`-PETR4.SA`, `+3m`, `0x`, `2/3`).
- `n_perturbations` — quantas perturbações competiram, sem contar o baseline.

`train_test` vale `"perturbation"`: não é `full` (sub-período não é o período
inteiro) e não é `train`/`test` (nada aqui é out-of-sample).

---

## D6 — Relação com o obstáculo 4

`n_perturbations` fica em coluna própria, separada de `n_combos` (que vale 1
aqui, por D1). São contagens de naturezas diferentes: `n_combos` conta as
tentativas de *tuning*, `n_perturbations` conta as tentativas de *estresse*.

O relatório imprime o aviso que a spec pede: perturbar até achar a variação que
sobrevive é overfitting disfarçado de teste de robustez. A defesa estrutural é
que **o módulo não tem função "escolher a melhor perturbação"** — a saída é a
distribuição e um veredicto conservador. Se alguém precisar dessa função um dia,
ela é a porta de entrada do exato viés que o componente existe para medir.

---

## Contrato de testes

`tests/test_robustness.py`. O teste-armadilha central, equivalente ao
anti-vazamento do Componente 2, é **a configuração não pode mudar entre
perturbações**: se um eixo alterar os parâmetros da estratégia ou do stop, o
módulo estaria comparando setups diferentes e chamando isso de robustez.

*Documento de projeto. Não é recomendação de investimento.*

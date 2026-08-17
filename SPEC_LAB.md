# Spec — Laboratório de pesquisa quantitativa

Objetivo: uma bancada para submeter teses a obstáculos crescentes e chamar de
robusta apenas a que sobreviver a todos. O laboratório não encontra teses boas —
ele derruba teses e reporta qual resistiu, e com quanta força.

Princípio de construção: cada peça nasce de uma pergunta que a peça anterior não
respondeu. Nada de framework especulativo. A ordem das seções é a ordem de
construção.

---

## Filosofia central (ler antes de codar qualquer coisa)

- O laboratório mede **taxa de rejeição**, não de aprovação. Se ele aprova teses
  com frequência, procure o vazamento (look-ahead, baseline errado, teste
  contaminado) antes de comemorar.
- "Robusta" = sobreviveu a: (1) vizinhança de parâmetros, (2) dados não usados
  na escolha, (3) perturbação do setup, (4) desconto pela quantidade de tentativas.
- Uma tese declarada boa antes de completar os quatro é uma tese que você gostou,
  não que você testou.

---

## Componente 0 — o que já existe e NÃO muda

Motor (`backtest.py`), custos (`core.py`/`costs.py`), estratégias
(`strategy.py`), stops (`stops.py`), métricas (`metrics.py`), batch (`batch.py`),
CSV (`runs.csv`), plot (`plot_runs.py`). O laboratório é uma camada POR CIMA
disso. Se alguma peça nova exigir reescrever o motor, pare e reconsidere — quase
certamente há um caminho aditivo.

---

## Componente 1 — Sweep de parâmetros

**Pergunta que responde:** o bom resultado é um platô (região contígua que
funciona) ou um pico (ponto isolado cercado de fracasso = sorte de tuning)?

**Comportamento:**
- Recebe uma grade: cada parâmetro mapeado para uma lista de valores.
  Ex: `{fast_window: [5,9,20], slow_window: [21,50], atr_multiplier: [1.5,2,2.5,3]}`.
- Roda o produto cartesiano. Cada combinação = uma execução do batch existente =
  uma ou mais linhas no CSV (uma por ticker).
- Combinações inválidas (ex: fast >= slow) são filtradas antes de rodar, não
  executadas e descartadas depois.

**Colunas novas obrigatórias no CSV (proveniência):**
- `sweep_id` — identifica o sweep inteiro (todas as combinações de uma varredura).
- `combo_id` — identifica a combinação de parâmetros dentro do sweep.
- `n_combos` — quantas combinações o sweep testou. **Esta é a defesa do
  obstáculo 4** — ver Componente 4.
- `train_test` — marca se a linha é de período de treino ou teste (Componente 2).

**Pontos de atenção:**
- O produto cartesiano cresce rápido. 4 params × 4 valores = 256 combinações ×
  10 tickers = 2560 linhas. Coloque um teto e avise antes de rodar algo enorme.
- Determinismo: o mesmo sweep rodado duas vezes deve gerar exatamente os mesmos
  números (o cache de dados já ajuda). Sem isso, nada abaixo é confiável.
- Não paralelize antes de precisar. Rodar sequencial e lento é melhor que
  paralelo e com bug de estado compartilhado.

**Não construir ainda:** otimizador que "busca" o melhor (gradiente, bayesiano).
Sweep exaustivo numa grade declarada é mais honesto e mais legível — você vê o
espaço inteiro, não só o ponto que um otimizador achou.

---

## Componente 2 — Walk-forward (o obstáculo que importa)

**Pergunta que responde:** os parâmetros escolhidos no passado funcionam no
futuro que não foi usado para escolhê-los?

**Por que não basta um corte único treino/teste:** uma única janela de validação
pode ser sorte. A forma madura desliza a janela: otimiza num período, testa no
seguinte intocado, desliza, repete. Cada período serve como teste out-of-sample
uma vez e depois entra na janela de otimização seguinte.

**Comportamento:**
- Divide o histórico em janelas. Duas variantes:
  - **Rolling** (janela de treino de tamanho fixo que desliza).
  - **Expanding** (treino começa no início e cresce; teste sempre o próximo bloco).
  - Começar com uma só; expanding é mais simples de raciocinar.
- Para cada janela: roda o sweep (Componente 1) SÓ no treino, escolhe a melhor
  combinação por uma métrica declarada (ex: Sharpe), aplica ESSA combinação
  fixa no bloco de teste seguinte, registra o resultado out-of-sample.
- Concatena os resultados out-of-sample de todas as janelas = a curva "honesta"
  da tese.

**Métrica-chave — Walk-Forward Efficiency (WFE):**
- `WFE = retorno_out_of_sample / retorno_in_sample` (anualizados).
- Leitura: acima de ~50-60% = a tese mantém pelo menos metade da performance em
  dados não vistos, sugere robustez real. Consistentemente baixa = overfitting.
- **Este é o número que transforma "é robusta?" de opinião em medida.** É a saída
  principal do laboratório.

**Pontos de atenção:**
- **Nunca deixar o teste vazar para o treino.** A combinação aplicada no bloco de
  teste é escolhida usando SÓ dados do treino. Se em algum ponto a escolha olha o
  teste, o WFE fica lindo e mentiroso. Este é o bug mais perigoso de todo o
  laboratório — vale um teste automatizado que o detecte.
- **Fitting implícito.** Mesmo com walk-forward correto, se VOCÊ olha o resultado
  out-of-sample e ajusta a grade e roda de novo, você contaminou o teste com sua
  própria memória. O out-of-sample só é limpo na PRIMEIRA vez que aquele período
  é usado como teste. Registrar isso (quantas vezes o período já foi "espiado")
  é parte do obstáculo 4.
- Poucas janelas = WFE ruidoso. Precisa de histórico suficiente para várias.

---

## Componente 3 — Robustez a perturbação

> Implementado em `robustness.py`. As decisões que esta seção deixou em aberto
> (qual config é perturbada, o que roda em cada perturbação, limiares do
> veredicto, proveniência) estão fechadas em **`SPEC_ROBUSTNESS.md`**, uma
> seção por decisão com o motivo escrito.

**Pergunta que responde:** a tese aguenta mexer no setup, ou só funciona na
configuração exata em que foi achada?

**Comportamento:** pega a config sobrevivente do Componente 2 e roda sob N
variações do ambiente:
- Tirar 1 ativo do universo (leave-one-out) — o resultado depende de um único
  papel sortudo?
- Deslocar a data de início em ±1, ±3 meses — depende do ponto exato de partida?
- Dobrar / zerar o custo — a margem sobrevive a execução mais pessimista?
- Sub-períodos (dividir em terços) — funciona ao longo do tempo ou só num pedaço?

**Leitura:** olhar a DISPERSÃO dos resultados, não a média. Tese robusta degrada
suavemente; tese frágil desaba com uma única perturbação. O gráfico certo aqui é
uma distribuição (boxplot / faixa), não um número.

**Ponto de atenção:** cada perturbação é mais uma "tentativa" no sentido do
obstáculo 4. Robustez e multiple-testing se cruzam — perturbar demais e ficar com
a variação que sobrevive é overfitting disfarçado de teste de robustez.

---

## Componente 4 — Correção por número de tentativas

**Pergunta que responde:** esse resultado bom é sinal, ou é a melhor de muitas
tentativas (cauda da sorte)?

**Não é um script, é uma disciplina com suporte de dados:**
- A coluna `n_combos` (Componente 1) já registra quantas combinações competiram.
- Regra de leitura: um Sharpe X como resultado único vale muito mais que o mesmo
  X como melhor de 200 combinações. Quanto maior `n_combos`, mais alto a barra
  para acreditar no melhor.
- Suporte visual: no gráfico de sweep, mostrar a DISTRIBUIÇÃO de todas as
  combinações, não só a melhor. Se a melhor é um outlier solitário acima de uma
  nuvem medíocre, desconfie. Se ela está no topo de um platô povoado, confie mais.

**Referência conceitual (não precisa implementar agora):** existe literatura
formal disso — "Probability of Backtest Overfitting", "Deflated Sharpe Ratio" —
que ajusta o Sharpe pela quantidade de tentativas. Vale conhecer o nome; a versão
prática pra você agora é a disciplina de contar e a distribuição visual.

---

## Componente 5 — Interface para mudar parâmetros

**A pergunta real:** qual a forma de menor atrito para declarar um experimento?

**Recomendação: arquivo de config declarativo (YAML/JSON), NÃO app web — ainda.**

Por quê:
- Um experimento é: universo + período + estratégia + grade de parâmetros +
  esquema de walk-forward + métrica de seleção. Isso é um documento, e um
  documento é melhor versionado como arquivo do que digitado num formulário.
- Config em arquivo é **reproduzível e versionável** — commita junto do resultado,
  e seis meses depois você recria o experimento exato. Um formulário web perde isso.
- Custo de construção de um app web (backend, estado, deploy) compete diretamente
  com o tempo de fazer perguntas. É o canto de sereia clássico.

**Formato sugerido (uma tese = um arquivo):**
```yaml
experiment: bollinger_regime_check
universe: [PETR4.SA, VALE3.SA, ...]
period: {start: 2015-01-01, end: 2024-01-01}
strategy:
  class: BollingerReversion
  grid:
    window: [10, 20, 50]
    k: [1.5, 2.0, 2.5]
stop:
  class: AtrStop
  grid: {multiplier: [2.0, 2.5]}
walk_forward: {scheme: expanding, train_years: 3, test_years: 1}
select_by: sharpe
```
Um comando: `python lab.py run experiment.yaml`. Gera CSV com toda a proveniência
e um relatório.

**Quando um app web passa a valer a pena:** só quando VISUALIZAR o acervo de
experimentos passados virar o gargalo — comparar dezenas de sweeps, navegar
heatmaps interativamente. Isso é um problema de LEITURA, e aí uma UI ganha. Mas é
um visualizador do CSV/resultados, não um substituto do config de entrada. E é
depois de os componentes 1-4 existirem e você ter acervo que justifique navegar.

---

## Componente 6 — Visualização

> Detalhado em **`SPEC_PLOTS.md`**. As decisões que esta seção deixou em aberto
> (módulo separado em vez de estender o `plot_runs.py`, um subcomando por
> gráfico, qual rodada é plotada, agregação entre tickers, o que fazer quando a
> grade não é 2D) estão fechadas lá, uma seção por decisão com o motivo escrito.
> Correção ao texto abaixo: o `plot_runs.py` **não** é estendido — ele continua
> servindo só o `runs.csv`; os gráficos do laboratório vivem em `plot_lab.py`.

Gráficos, conforme cada componente gera dados novos:

- **Sweep 1D** (já existe, forma de linha) — métrica vs um parâmetro. Platô vs pico.
- **Sweep 2D** (heatmap) — dois parâmetros nos eixos, métrica na cor. A visão que
  revela platô/pico em duas dimensões de uma vez. Construir quando o sweep gerar
  grade 2D.
- **Distribuição do sweep** (histograma) — todas as combinações, com a melhor
  marcada. Suporte visual do obstáculo 4.
- **Curva walk-forward** — in-sample vs out-of-sample lado a lado, WFE anotado.
  A visão que fecha o veredicto de robustez.
- **Faixa de robustez** (boxplot) — dispersão sob perturbação (Componente 3).

Regra herdada: cada gráfico responde a UMA pergunta e é lido em segundos. Se
precisa de legenda longa pra entender, está fazendo demais.

---

## Ordem de construção (resumo)

1. Sweep + colunas de proveniência (1 e 4-dados). ✅ `sweep.py`
2. Config declarativo (5). ✅ `lab.py` + `experiments/*.yaml`
3. Walk-forward + WFE (2). ✅ `walkforward.py`
4. Perturbação (3). ✅ `robustness.py` (ver `SPEC_ROBUSTNESS.md`)
5. Visualizações (6), cada uma quando seu componente gerar dados.
   📝 spec fechada em `SPEC_PLOTS.md`; `plot_lab.py` ainda não implementado.
6. App web — só se e quando a LEITURA do acervo virar o gargalo.

Cada passo é aditivo sobre o motor existente. Nenhum exige reescrever o núcleo.

---

## Aviso final

Este laboratório vai passar a maior parte do tempo confirmando que teses não
funcionam. Isso é o funcionamento correto, não a falha. As quatro famílias já
refutadas neste projeto são o histórico saudável de uma bancada honesta. Se de
repente muitas teses começarem a "vencer", a primeira hipótese é vazamento no
laboratório — não que você ficou bom em achar alpha.

*Documento de projeto. Não é recomendação de investimento.*

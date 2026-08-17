# Spec — Componente 6: visualização do laboratório

Detalhamento do Componente 6 do `SPEC_LAB.md`. Aquela seção lista os gráficos e
a regra herdada ("cada gráfico responde a UMA pergunta"); esta fecha as decisões
que ela deixou em aberto — de onde vêm os dados, como se agregam, o que acontece
quando não dão para desenhar. Quem for mexer no `plot_lab.py` depois precisa
discordar daqui, não adivinhar.

**Pergunta que o componente responde:** os componentes 1–3 produzem tabelas
grandes demais para ler linha a linha. Qual é a forma de olhar cada uma delas em
segundos e sair com o veredicto certo — inclusive o veredicto negativo?

**O que este componente NÃO é:** ele não calcula veredicto novo. WFE, survival
rate, retenção mediana e o placar contra o benchmark já são calculados em
`walkforward.py` e `robustness.py`. Aqui eles são LIDOS do CSV e desenhados.
Qualquer fórmula de julgamento reimplementada no módulo de plot é bug — duas
fontes de verdade divergem em silêncio.

---

## D1 — Módulo separado, não extensão do `plot_runs.py`

**Decisão: arquivo novo `plot_lab.py`.** O `plot_runs.py` continua servindo
exclusivamente o `runs.csv` (acervo curado dos batches) e não muda.

Por quê: são quatro esquemas de CSV diferentes, com públicos diferentes. O
`plot_runs.py` responde "como as variantes se comportaram nos 10 tickers"; o
`plot_lab.py` responde "a tese sobrevive aos obstáculos". Fundir os dois faria a
flag `--csv` significar coisas distintas dependendo do painel, que é exatamente
o tipo de ambiguidade que o resto do projeto evita.

Consequência: nenhuma linha do `plot_runs.py` é editada por este componente. Se
a implementação precisar mexer lá, parar e reconsiderar.

---

## D2 — Um subcomando por gráfico

**Decisão: `python plot_lab.py <grafico> [opções]`,** com quatro subcomandos:

| subcomando | pergunta que responde | CSV lido |
|---|---|---|
| `heatmap` | platô ou pico, em duas dimensões de parâmetro? | `sweep_runs.csv` |
| `dist` | a melhor combinação é outlier solitário ou topo de platô povoado? | `sweep_runs.csv` |
| `walkforward` | quanto do in-sample sobrevive out-of-sample? | `walkforward_runs.csv` |
| `robustness` | a tese degrada suave ou desaba com uma perturbação? | `robustness_runs.csv` |

Por quê: o painel 2×2 do `plot_runs.py` funciona porque os quatro painéis leem o
mesmo CSV. Aqui cada gráfico depende de um arquivo diferente — quem rodou só o
sweep não tem `walkforward_runs.csv`, e um painel único ou quebraria ou
mostraria três quartos de placeholder. Subcomando também é a forma que respeita
a regra herdada: um gráfico, uma pergunta, uma janela.

Flags comuns a todos:

- `--csv PATH` — sobrescreve o CSV default do subcomando.
- `--run ID` — filtra uma rodada específica (`sweep_id` no sweep, `run_id` no
  walk-forward e na robustez). **Default: a rodada mais recente do arquivo**, e o
  título do gráfico imprime qual foi. Ver D3.
- `--metric NOME` — métrica plotada. Defaults em D4.
- `--save PATH` — salva PNG em vez de abrir janela (mesmo comportamento do
  `plot_runs.py`).

---

## D3 — Uma rodada por gráfico, nunca a mistura

**Decisão: todo gráfico plota exatamente uma rodada.** Sem `--run`, usa a mais
recente presente no arquivo; com `--run ID` inexistente, erro explícito listando
os IDs disponíveis.

Por quê: os CSVs do laboratório são acumulativos. Duas rodadas do mesmo
experimento com grades diferentes, sobrepostas no mesmo heatmap, produziriam
células com número de amostras diferente sem nenhuma marca visual disso. Pior:
misturar rodadas de períodos diferentes num histograma inflaria artificialmente
a nuvem contra a qual a melhor combinação é comparada — o obstáculo 4 lido ao
contrário.

"Mais recente" é definido pelo ID, não pela ordem das linhas: os IDs carregam
timestamp ISO (`mac_plateau-2026-08-17T14:03:40`), então o máximo lexicográfico
é o mais recente e a escolha é determinística mesmo se as linhas forem
reordenadas.

---

## D4 — Métrica: `sharpe` por default, exceto no walk-forward

**Decisão:**

- `heatmap`, `dist`, `robustness` → default `sharpe`. É a métrica de seleção
  default do `SweepSpec`/`RobustnessSpec`, então o gráfico mostra por default a
  dimensão em que a decisão foi tomada.
- `walkforward` → **fixo em `annualized_return`, sem `--metric`.** O WFE é
  definido sobre retorno anualizado (`walkforward.WFE_METRIC`); desenhar o par
  in/out-of-sample numa métrica e anotar um WFE calculado em outra seria um
  gráfico que mente pela legenda.

Métrica inexistente nas colunas do CSV = erro listando as disponíveis, igual ao
`plot_runs.py` já faz.

---

## D5 — Agregação entre tickers: mediana

**Decisão: mediana**, em todos os gráficos que precisam colapsar as linhas por
ticker num número por combinação/perturbação. Sem flag para trocar.

Por quê: o sweep grava uma linha por ticker por combinação, e a pergunta do
platô é sobre o parâmetro, não sobre o papel. A média deixa um único ticker
extremo deslocar a célula inteira — precisamente o "papel sortudo" que o
Componente 3 existe para caçar. A mediana é a leitura que a robustez já usa
(`median_retention`), então os dois componentes falam a mesma língua.

Sem flag `--agg` porque a escolha da agregação muda o veredicto visual, e uma
flag convida a rodar as duas e ficar com a que parece melhor — a versão gráfica
do viés do obstáculo 4.

**Exceção:** a linha `baseline` da robustez e os valores por janela do
walk-forward já vêm agregados na origem (média sobre tickers, calculada em
`robustness.py`/`walkforward.py`). O plot NÃO reagrega esses — ver D1 do
princípio "não recalcular veredicto". Onde o CSV tem várias linhas por
perturbação (uma por ticker), a mediana vale.

---

## D6 — `heatmap`: eixos declarados, erro quando ambíguo

**Decisão: `--x COLUNA --y COLUNA`.** Se ambos forem omitidos e a rodada tiver
exatamente dois parâmetros com 2+ valores distintos, usa esses dois. Em qualquer
outro caso — um só parâmetro variando, ou três ou mais — **erro explícito**
listando os parâmetros encontrados e quantos valores cada um tem.

Por quê: coerente com o `lab.py`, que nunca aceita default silencioso. Inferir
"os dois com mais valores" escolheria em silêncio o que o gráfico mostra, e
marginalizar os parâmetros excedentes produziria uma cor que é média de coisas
diferentes — um heatmap bonito e ininterpretável.

Detalhes de desenho:

- Colunas de parâmetro são as prefixadas `param_` no CSV. Combinações inválidas
  filtradas pelo sweep simplesmente não existem no CSV; a célula correspondente
  fica **vazia e visualmente distinta** (não zero, que se confundiria com
  resultado ruim).
- Escala de cor divergente centrada em zero quando a métrica tem sinal (Sharpe,
  retorno): o olho precisa separar "positivo fraco" de "negativo" sem consultar
  a barra de cor.
- Valor numérico impresso dentro de cada célula enquanto a grade for pequena
  (≤ 8×8); acima disso só a cor, senão vira ruído.

---

## D7 — `dist`: a nuvem é o ponto, a melhor é só uma marca

**Decisão: histograma de TODAS as combinações da rodada**, com a melhor marcada
por uma linha vertical anotada, e `n_combos` impresso no título.

Por quê: é o suporte visual do obstáculo 4, e o obstáculo 4 é uma disciplina de
leitura — o gráfico existe para que a melhor combinação seja vista NO CONTEXTO
da distribuição que a produziu. Um gráfico que destacasse a melhor e escondesse
a nuvem seria o oposto do componente.

O título traz `n_combos` porque é o número que calibra a leitura: um Sharpe de
1,2 como melhor de 4 tentativas e como melhor de 200 são afirmações
diferentes. **Nenhum ajuste formal de Sharpe (Deflated Sharpe, PBO) é
implementado** — a spec do Componente 4 é explícita que a versão prática agora é
contar e mostrar.

---

## D8 — `walkforward`: barras pareadas por janela, WFE anotado

**Decisão:** eixo x = janelas em ordem; por janela, duas barras — in-sample
(treino, combinação escolhida) e out-of-sample (teste). WFE agregada anotada no
canto.

- O valor in-sample de cada janela é o da combinação **escolhida** naquela
  janela (`chosen_combo_id`), não a melhor do CSV inteiro nem a média das
  combinações. Plotar outra coisa contaria uma história que o walk-forward não
  viveu.
- A combinação escolhida é impressa como rótulo da janela. Escolha diferente a
  cada janela já é veredicto negativo — a spec do Componente 2 diz isso, e o
  gráfico tem que deixar visível sem precisar abrir o CSV.
- WFE vem recalculado a partir das mesmas linhas com a fórmula do
  `walkforward.py` — **razão das MÉDIAS, não média das razões** — ou é lido
  quando disponível. `None` (in-sample ≤ 0) imprime `N/A`, nunca 0% nem célula
  em branco: a distinção entre "não se aplica" e "zero" é o ponto.

---

## D9 — `robustness`: boxplot, e o placar do benchmark junto

**Decisão: boxplot da distribuição das perturbações**, com o baseline marcado
como linha de referência horizontal, e os pontos individuais sobrepostos ao box
quando forem poucos (≤ 30).

- Pontos sobrepostos porque com 19 perturbações o box sozinho esconde se a cauda
  ruim é uma perturbação ou seis — e "qual eixo quebrou" é a informação
  acionável.
- Cor/marca distinta por `perturbation_kind`, para que "desabou só no
  sub-período 3/3" seja legível sem legenda longa.
- O veredicto (`ROBUSTA`/`FRÁGIL`) e o **placar contra o benchmark** aparecem
  juntos no título. Isso não é enfeite: por D4.1 do `SPEC_ROBUSTNESS.md`,
  ROBUSTA significa ESTÁVEL e nunca SUPERIOR, e o MAC refutado sai ROBUSTA aqui
  perdendo do buy-and-hold em 20 de 20. Um gráfico que mostrasse o veredicto sem
  o placar seria lido como aprovação da tese — o erro exato que a bancada existe
  para não cometer.

---

## D10 — CSV ausente, rodada vazia, coluna faltando

**Decisão: erro explícito com mensagem acionável, nunca gráfico vazio.**
Mesmo padrão do `plot_runs.py` (`SystemExit(f"Não achei {path}. Rode ... primeiro.")`).

Por quê: um gráfico em branco é ambíguo entre "não rodei o experimento" e "o
experimento não produziu resultado" — e a segunda leitura é um veredicto. O
único placeholder tolerado no projeto é o do `panel_plateau`, que existe porque
lá o painel é parte de uma grade fixa 2×2; aqui cada subcomando é uma janela
inteira, e uma janela inteira em branco não tem desculpa.

Falhas parciais (um ticker que quebrou e virou `failures[]` no batch) não são
erro: o gráfico desenha o que existe. O que é erro é a rodada inteira ausente.

---

## D11 — Determinismo do desenho

Mesmo CSV + mesmos argumentos = mesmo PNG, byte a byte não é exigido, mas
**ordem de séries, cores e posições têm que ser estáveis**: ordenação explícita
em toda iteração sobre tickers, combinações, janelas e perturbações; nunca
iteração sobre `set` ou sobre ordem de inserção de dicionário vinda de I/O.

Por quê: a mesma disciplina do resto do projeto. Um gráfico cuja legenda troca
de cor entre execuções faz o leitor comparar duas fotos e ver diferença onde não
há.

---

## Contrato de testes

`tests/test_plot_lab.py`, sem abrir janela (backend `Agg`).

O teste-armadilha central deste componente, equivalente ao anti-vazamento do
Componente 2 e ao "configuração não muda" do Componente 3, é **o plot não
inventa veredicto**: os números anotados no gráfico (WFE, veredicto de robustez,
melhor combinação) têm que bater com o que `walkforward.py`/`robustness.py`
produzem para as mesmas linhas. Um plot que recalcula por conta própria diverge
da fonte de verdade sem ninguém perceber, e o gráfico é justamente a peça em que
o erro passa despercebido — ninguém confere um pixel contra um CSV.

Os demais alvos, em ordem de importância:

1. Seleção de rodada: sem `--run` pega a mais recente; `--run` inexistente é erro.
2. `heatmap` com grade ambígua (1 ou 3+ parâmetros variando) é erro, não palpite.
3. Agregação entre tickers é mediana (D5), verificável com valores assimétricos.
4. CSV ausente/vazio é erro com mensagem, não figura em branco (D10).
5. `walkforward` usa o valor da combinação escolhida por janela, não a melhor
   global (D8).
6. WFE `None` imprime `N/A`, não `0%` (D8).

*Documento de projeto. Não é recomendação de investimento.*

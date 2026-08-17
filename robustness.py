"""Componente 3 do laboratório (SPEC_LAB.md, detalhado em SPEC_ROBUSTNESS.md):
robustez a perturbação.

Pergunta que este módulo responde: **a tese aguenta mexer no setup, ou só
funciona na configuração exata em que foi achada?**

O Componente 2 perguntou se a tese sobrevive a dados que não foram usados para
escolhê-la. Este pergunta outra coisa: se ela sobrevive a mexer no ambiente —
tirar um papel do universo, começar três meses antes, dobrar o custo, olhar só
um terço do período. Tese robusta degrada suavemente; tese frágil desaba com uma
única perturbação.

## O que este módulo NÃO faz — e por quê isso é o ponto

**Não existe função que escolha a melhor perturbação.** A saída é a distribuição
inteira e um veredicto conservador. Perturbar até achar a variação que sobrevive
é overfitting disfarçado de teste de robustez, e a defesa contra isso aqui é
estrutural: a porta não existe.

Pela mesma razão, a configuração perturbada é **declarada**, nunca inferida: a
grade tem que produzir exatamente uma combinação válida. Ler automaticamente a
vencedora do walk-forward esconderia a instabilidade da escolha entre janelas
atrás de um "a da última janela" arbitrário.

## Limitação assumida

Os números daqui são **in-sample**: a config já foi escolhida, e cada
perturbação roda o período inteiro (ou um pedaço dele) sem separar treino de
teste. Eles não substituem o WFE do Componente 2 e nenhum deles é performance
esperada. Rodar walk-forward sob cada perturbação seria o ideal teórico e custa
`perturbações × janelas × tickers` — ver D2 do `SPEC_ROBUSTNESS.md`.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from batch import append_row
from data import FetchFn, get_data
from sweep import SweepError, SweepSpec, run_sweep, valid_combos

DEFAULT_LOG_PATH = Path("robustness_runs.csv")

# Teto de perturbações, sem contar o baseline. Mesmo espírito do teto do sweep:
# não protege a máquina, obriga a declarar conscientemente quando a aposta do
# obstáculo 4 sobe.
DEFAULT_MAX_PERTURBATIONS = 64

# Limiares do veredicto (D4 do SPEC_ROBUSTNESS.md). Severos de propósito: a
# spec diz que a tese frágil "desaba com uma única perturbação", então qualquer
# perturbação que vire a métrica para <= 0 já reprova.
MIN_SURVIVAL_RATE = 1.0
MIN_MEDIAN_RETENTION = 0.5

DEFAULT_START_SHIFT_MONTHS = (-3, -1, 1, 3)
DEFAULT_COST_MULTIPLIERS = (0.0, 2.0)
DEFAULT_SUBPERIODS = 3


class RobustnessError(Exception):
    """Configuração de teste de robustez inviável."""


@dataclass(frozen=True)
class Perturbation:
    """Uma variação do ambiente: mesma config, `SweepSpec` alterado num eixo."""

    kind: str
    value: str  # rótulo legível e determinístico: '-PETR4.SA', '+3m', '2x', '2/3'
    spec: SweepSpec

    @property
    def label(self) -> str:
        return self.kind if not self.value else f"{self.kind} {self.value}"


@dataclass(frozen=True)
class RobustnessSpec:
    """A declaração de um teste de robustez.

    `base` precisa produzir exatamente UMA combinação válida — a config já
    escolhida. Cada eixo pode ser desligado (lista vazia, `False`, `0`).
    """

    base: SweepSpec
    select_by: str = "sharpe"
    leave_one_out: bool = True
    start_shift_months: Sequence[int] = DEFAULT_START_SHIFT_MONTHS
    cost_multipliers: Sequence[float] = DEFAULT_COST_MULTIPLIERS
    subperiods: int = DEFAULT_SUBPERIODS
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS


@dataclass(frozen=True)
class PerturbationResult:
    perturbation: Perturbation
    rows: list[dict[str, Any]]
    value: float | None  # média da métrica de seleção sobre os tickers
    benchmark: float | None = None  # a mesma métrica no buy-and-hold, se existir


@dataclass(frozen=True)
class RobustnessSummary:
    """A distribuição, não a média. A pior perturbação é o número que decide."""

    baseline: float | None
    n_perturbations: int
    n_valid: int
    worst: float | None
    p25: float | None
    median: float | None
    p75: float | None
    best: float | None
    survival_rate: float | None
    median_retention: float | None
    verdict: str


@dataclass(frozen=True)
class RobustnessResult:
    run_id: str
    baseline: PerturbationResult
    perturbations: list[PerturbationResult]  # sem o baseline
    rows: list[dict[str, Any]]
    failures: list[tuple[str, str, str]]  # (perturbação, ticker, mensagem)
    select_by: str

    @property
    def all_perturbations(self) -> list[PerturbationResult]:
        return [self.baseline, *self.perturbations]

    @property
    def summary(self) -> RobustnessSummary:
        return summarize(
            baseline=self.baseline.value, values=[p.value for p in self.perturbations]
        )

    @property
    def beats_benchmark(self) -> tuple[int, int]:
        """(quantas perturbações superam o buy-and-hold, quantas dá para comparar).

        Existe porque o veredicto deste componente mede ESTABILIDADE, não
        superioridade: uma estratégia que fica pouco tempo exposta pode ter
        Sharpe baixo, positivo e estabilíssimo, e sair daqui como ROBUSTA sem
        nunca ter batido o buy-and-hold. Sem este contador ao lado, o veredicto
        seria lido como aprovação da tese — que é exatamente o erro que a
        bancada existe para não cometer.
        """
        comparaveis = [
            p for p in self.all_perturbations if p.value is not None and p.benchmark is not None
        ]
        return sum(1 for p in comparaveis if p.value > p.benchmark), len(comparaveis)


def build_perturbations(spec: RobustnessSpec) -> list[Perturbation]:
    """Materializa o baseline e cada perturbação, em ordem determinística.

    Ordem fixa: baseline → leave-one-out → deslocamento de início → custo →
    sub-períodos. A ordem importa porque o CSV é lido depois por gente.

    Nenhum eixo toca em `strategy_class`, `stop_class` ou nas grades: perturbar
    o setup em vez do ambiente compararia coisas diferentes e chamaria isso de
    robustez. É o que `tests/test_robustness.py::TestConfiguracaoNaoMuda` trava.
    """
    base = spec.base
    _require_single_combo(base)

    perturbations = [Perturbation("baseline", "", base)]
    perturbations += _leave_one_out(spec)
    perturbations += _start_shifts(spec)
    perturbations += _cost_multipliers(spec)
    perturbations += _subperiods(spec)

    n = len(perturbations) - 1  # o baseline é referência, não tentativa
    if n > spec.max_perturbations:
        raise RobustnessError(
            f"{n} perturbações ultrapassam o teto de {spec.max_perturbations}. "
            f"Cada perturbação é mais uma tentativa a descontar no obstáculo 4 — "
            f"reduza os eixos ou suba max_perturbations conscientemente."
        )
    return perturbations


def _require_single_combo(base: SweepSpec) -> None:
    try:
        combos = valid_combos(base)
    except SweepError as exc:
        raise RobustnessError(f"configuração inválida para perturbar: {exc}") from exc

    if len(combos) != 1:
        raise RobustnessError(
            f"a robustez perturba UMA configuração já escolhida, mas a grade "
            f"declarada produz {len(combos)} combinações válidas. Fixe cada "
            f"parâmetro num único valor — escolher a melhor aqui dentro seria "
            f"tuning disfarçado de teste de robustez."
        )


def _leave_one_out(spec: RobustnessSpec) -> list[Perturbation]:
    """O resultado depende de um único papel sortudo?"""
    if not spec.leave_one_out:
        return []

    tickers = list(spec.base.tickers)
    if len(tickers) < 3:
        raise RobustnessError(
            f"leave-one-out precisa de pelo menos 3 tickers, o universo tem "
            f"{len(tickers)}. Tirar um de dois deixa um backtest single-asset, "
            f"que responde a outra pergunta."
        )
    return [
        Perturbation(
            "leave_one_out",
            f"-{fora}",
            replace(spec.base, tickers=[t for t in tickers if t != fora]),
        )
        for fora in tickers
    ]


def _start_shifts(spec: RobustnessSpec) -> list[Perturbation]:
    """Depende do ponto exato de partida?

    Deslocamento negativo alonga o período (pede dados anteriores ao início
    declarado); positivo encurta. O fim nunca se move — mover os dois seria
    outro eixo (janela deslizante), não "começar em outro ponto".
    """
    perturbations = []
    for meses in spec.start_shift_months:
        if meses == 0:
            raise RobustnessError(
                "deslocamento de 0 meses é o próprio baseline com outro nome — "
                "ocuparia uma linha do relatório dizendo nada"
            )
        novo_inicio = _shift_months(spec.base.start, meses)
        if novo_inicio >= spec.base.end:
            raise RobustnessError(
                f"deslocar o início em {meses:+d} meses o joga para {novo_inicio}, "
                f"depois do fim ({spec.base.end})"
            )
        perturbations.append(
            Perturbation("start_shift", f"{meses:+d}m", replace(spec.base, start=novo_inicio))
        )
    return perturbations


def _cost_multipliers(spec: RobustnessSpec) -> list[Perturbation]:
    """A margem sobrevive a uma execução mais pessimista?

    Corretagem e slippage são multiplicados pelo MESMO fator: separá-los
    dobraria as execuções para distinguir duas fontes de custo, e a pergunta
    aqui não é qual delas machuca mais.
    """
    perturbations = []
    for fator in spec.cost_multipliers:
        if fator < 0:
            raise RobustnessError(f"multiplicador de custo negativo: {fator}")
        perturbations.append(
            Perturbation(
                "cost_multiplier",
                f"{fator:g}x",
                replace(
                    spec.base,
                    brokerage=spec.base.brokerage * fator,
                    slippage_bps=spec.base.slippage_bps * fator,
                ),
            )
        )
    return perturbations


def _subperiods(spec: RobustnessSpec) -> list[Perturbation]:
    """Funciona ao longo do tempo, ou só num pedaço?"""
    if not spec.subperiods:
        return []
    if spec.subperiods < 2:
        raise RobustnessError("subperiods precisa ser 0 (desligado) ou >= 2")

    blocos = _split_period(spec.base.start, spec.base.end, spec.subperiods)
    return [
        Perturbation(
            "subperiod",
            f"{i + 1}/{spec.subperiods}",
            replace(spec.base, start=inicio, end=fim),
        )
        for i, (inicio, fim) in enumerate(blocos)
    ]


def _shift_months(d: date, months: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(months=months)).date()


def _split_period(start: date, end: date, k: int) -> list[tuple[date, date]]:
    """Divide [start, end] em k blocos contíguos que cobrem o período inteiro."""
    total = (end - start).days
    if total < k:
        raise RobustnessError(
            f"período de {total} dia(s) não comporta {k} sub-períodos"
        )

    passo = total // k
    blocos: list[tuple[date, date]] = []
    inicio = start
    for i in range(k):
        # o último bloco absorve o resto da divisão inteira, para a cobertura
        # terminar exatamente em `end`
        fim = end if i == k - 1 else inicio + timedelta(days=passo) - timedelta(days=1)
        blocos.append((inicio, fim))
        inicio = fim + timedelta(days=1)
    return blocos


def run_robustness(
    spec: RobustnessSpec,
    *,
    run_id: str | None = None,
    log_path: Path | None = DEFAULT_LOG_PATH,
    cache_dir: Path = Path("data_cache"),
    fetch_fn: FetchFn | None = None,
    verbose: bool = True,
) -> RobustnessResult:
    """Roda o baseline e cada perturbação, e devolve a distribuição."""
    perturbations = build_perturbations(spec)
    run_id = run_id or f"rb-{pd.Timestamp.now().isoformat(timespec='seconds')}"
    n_perturbations = len(perturbations) - 1

    _warm_cache(perturbations, cache_dir=cache_dir, fetch_fn=fetch_fn)

    results: list[PerturbationResult] = []
    all_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []

    for perturbation in perturbations:
        sweep = run_sweep(
            perturbation.spec,
            sweep_id=f"{run_id}-{_slug(perturbation)}",
            train_test="perturbation",  # nada aqui é out-of-sample, e o período
            log_path=None,              # pode ser um pedaço: 'full' mentiria
            cache_dir=cache_dir,
            fetch_fn=fetch_fn,
            verbose=False,
        )
        failures.extend(
            (perturbation.label, ticker, msg) for _combo, ticker, msg in sweep.failures
        )

        rows = [
            {
                "run_id": run_id,
                "perturbation_kind": perturbation.kind,
                "perturbation_value": perturbation.value,
                "n_perturbations": n_perturbations,
                **row,
            }
            for row in sweep.rows
        ]
        value = _mean([row.get(spec.select_by) for row in rows])
        benchmark = _mean([row.get(f"benchmark_{spec.select_by}") for row in rows])
        results.append(PerturbationResult(perturbation, rows, value, benchmark))

        for row in rows:
            all_rows.append(row)
            if log_path is not None:
                append_row(row, path=log_path)

        if verbose:
            print(f"  {perturbation.label:<28}{_fmt(value)}")

    result = RobustnessResult(
        run_id=run_id,
        baseline=results[0],
        perturbations=results[1:],
        rows=all_rows,
        failures=failures,
        select_by=spec.select_by,
    )

    if verbose:
        print_robustness(result)
        if log_path is not None:
            print(f"\n{len(all_rows)} linha(s) em {log_path.resolve()}")

    return result


def _warm_cache(
    perturbations: Sequence[Perturbation],
    *,
    cache_dir: Path,
    fetch_fn: FetchFn | None,
) -> None:
    """Preenche o cache com o range mais amplo da rodada, antes da primeira execução.

    Sem isso, a rodada não é determinística — e a causa não é óbvia. O cache de
    `data.py` é por ticker e é REESCRITO sempre que o range pedido não está
    coberto. As perturbações pedem ranges diferentes de propósito
    (`start_shift -3m` alonga, sub-períodos cortam), então o cache é refeito no
    meio da rodada; como o preço ajustado do provedor depende do range baixado,
    a mesma data volta com diferença na sétima casa decimal e a ordem das
    perturbações passa a influenciar o resultado.

    Baixando uma vez o intervalo que cobre TODAS as perturbações, todas leem a
    mesma série. Liquidez fica em 0 aqui de propósito: a decisão de rejeitar
    papel ilíquido é da execução real, sobre a fatia que ela usa. Falha de um
    ticker é ignorada — o loop principal a reporta em `failures` com contexto.

    ATENÇÃO — isto mitiga, não elimina. `data._covers_range` compara a primeira
    barra do cache com a data PEDIDA, então um `start` que caia em fim de semana
    ou feriado (2018-01-01, por exemplo) nunca é considerado coberto e o cache é
    reescrito mesmo assim. A correção mora em `data.py`, que o Componente 0 da
    spec declara fora do escopo do laboratório — ver a seção de gotchas do
    README. Enquanto ela não vier, execuções repetidas podem divergir na sétima
    casa decimal quando as bordas do período não são dias úteis.
    """
    inicio = min(p.spec.start for p in perturbations)
    fim = max(p.spec.end for p in perturbations)
    tickers = {t for p in perturbations for t in p.spec.tickers}

    for ticker in sorted(tickers):
        try:
            get_data(
                ticker,
                inicio,
                fim,
                cache_dir=cache_dir,
                fetch_fn=fetch_fn,
                min_median_turnover=0.0,
            )
        except Exception:
            continue


def _slug(perturbation: Perturbation) -> str:
    limpo = perturbation.value.replace(" ", "").replace("/", "-")
    return f"{perturbation.kind}{'-' + limpo if limpo else ''}"


def summarize(*, baseline: float | None, values: Sequence[float | None]) -> RobustnessSummary:
    """Resume a rodada pela DISPERSÃO das perturbações, não pela média.

    Perturbações sem métrica válida (execução sem trades, ticker sem dados) são
    ignoradas na distribuição e contadas em `n_valid` — "não deu para medir" é
    diferente de "mediu zero".

    O veredicto é deliberadamente severo: basta uma perturbação virar a métrica
    para <= 0 e a tese é FRÁGIL. Um limiar tolerante deixaria passar exatamente
    o caso que este componente existe para pegar — a tese que só funcionava
    porque um papel sortudo estava no universo.
    """
    validos = sorted(v for v in values if v is not None and v == v)
    n = len(values)

    if not validos or baseline is None or baseline != baseline or baseline <= 0:
        return RobustnessSummary(
            baseline=baseline,
            n_perturbations=n,
            n_valid=len(validos),
            worst=validos[0] if validos else None,
            p25=_percentile(validos, 0.25),
            median=_percentile(validos, 0.50),
            p75=_percentile(validos, 0.75),
            best=validos[-1] if validos else None,
            survival_rate=None,
            median_retention=None,
            verdict="N/A",
        )

    survival_rate = sum(1 for v in validos if v > 0) / len(validos)
    median = _percentile(validos, 0.50)
    median_retention = median / baseline

    robusta = survival_rate >= MIN_SURVIVAL_RATE and median_retention >= MIN_MEDIAN_RETENTION
    return RobustnessSummary(
        baseline=baseline,
        n_perturbations=n,
        n_valid=len(validos),
        worst=validos[0],
        p25=_percentile(validos, 0.25),
        median=median,
        p75=_percentile(validos, 0.75),
        best=validos[-1],
        survival_rate=survival_rate,
        median_retention=median_retention,
        verdict="ROBUSTA" if robusta else "FRÁGIL",
    )


def _percentile(ordenados: Sequence[float], q: float) -> float | None:
    """Percentil por rank mais próximo — sem interpolar.

    Com 4 ou 12 pontos, interpolar inventa um valor que nenhuma perturbação
    produziu. Aqui todo número reportado é um resultado que realmente aconteceu.
    """
    if not ordenados:
        return None
    rank = max(1, math.ceil(q * len(ordenados)))
    return ordenados[rank - 1]


def _mean(values: Sequence[float | None]) -> float | None:
    validos = [v for v in values if v is not None and v == v]
    return sum(validos) / len(validos) if validos else None


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def print_robustness(result: RobustnessResult) -> None:
    """Relatório: a lista da PIOR para a melhor, a distribuição e o veredicto."""
    resumo = result.summary
    metrica = result.select_by

    print(f"\n=== Robustez: {resumo.n_perturbations} perturbação(ões), métrica '{metrica}' ===")
    print(f"baseline (config declarada): {_fmt(resumo.baseline)}")

    # da pior para a melhor: é a pior que carrega a informação
    ordenadas = sorted(
        result.perturbations,
        key=lambda p: (p.value is None, p.value if p.value is not None else 0.0),
    )
    print(f"\n{'perturbação':<30}{metrica:>10}{'vs baseline':>14}")
    for p in ordenadas:
        if p.value is None or resumo.baseline is None or resumo.baseline <= 0:
            razao = "N/A"
        else:
            razao = f"{p.value / resumo.baseline:.0%}"
        print(f"{p.perturbation.label:<30}{_fmt(p.value):>10}{razao:>14}")

    print(
        f"\nDistribuição — pior {_fmt(resumo.worst)} | p25 {_fmt(resumo.p25)} | "
        f"mediana {_fmt(resumo.median)} | p75 {_fmt(resumo.p75)} | melhor {_fmt(resumo.best)}"
    )

    if resumo.verdict == "N/A":
        print(
            "\nVeredicto: N/A — o baseline não é positivo, então não há "
            "performance a preservar e a razão não significaria nada."
        )
    else:
        print(
            f"\nSobrevivência: {resumo.survival_rate:.0%} das perturbações mantêm "
            f"{metrica} > 0 | retenção na mediana: {resumo.median_retention:.0%}"
        )
        print(f"Veredicto: {resumo.verdict}")
        if resumo.verdict == "FRÁGIL":
            print(
                "Leitura: a tese depende do ambiente exato em que foi achada. "
                "Olhe a pior linha da tabela — é ela que diz de que ela depende."
            )

    batem, comparaveis = result.beats_benchmark
    if comparaveis:
        print(
            f"\nSuperam o buy-and-hold (mesmos custos): {batem}/{comparaveis} execuções"
        )
        if resumo.verdict == "ROBUSTA" and batem * 2 <= comparaveis:
            print(
                "ATENÇÃO: ROBUSTA aqui significa ESTÁVEL, não superior. Esta "
                "config é estável e perde do buy-and-hold na maioria das "
                "execuções — robustez de um resultado ruim continua sendo um "
                "resultado ruim."
            )

    print(
        "\nAviso: estes números são in-sample (a config já estava escolhida) e "
        "não substituem o WFE do walk-forward."
    )
    print(
        "Cada perturbação é mais uma tentativa (obstáculo 4). Ficar com a "
        "variação que sobreviveu é overfitting disfarçado de teste de robustez."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustez a perturbação (Componente 3 do laboratório). "
        "A entrada é um experimento YAML com a seção perturbations — ver lab.py."
    )
    parser.add_argument("experiment", type=Path, help="Caminho do .yaml do experimento")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser.parse_args()


def main() -> None:
    # mesma razão de main.py: o console default do Windows corrompe acentos
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # a construção do experimento mora em lab.py; aqui só o atalho de execução
    from lab import ExperimentError, load_experiment, run_experiment

    args = _parse_args()
    try:
        experiment = load_experiment(args.experiment)
        if experiment.robustness is None:
            raise ExperimentError(
                f"{args.experiment} não declara a seção 'perturbations'. "
                f"Sem ela, use `python lab.py run {args.experiment}`."
            )
        run_experiment(experiment, log_path=args.log_path)
    except (ExperimentError, RobustnessError) as exc:
        raise SystemExit(f"Robustez abortada: {exc}")


if __name__ == "__main__":
    main()

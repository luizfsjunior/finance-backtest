"""Componente 2 do laboratório (SPEC_LAB.md): walk-forward e WFE.

Pergunta que este módulo responde: **os parâmetros escolhidos no passado
funcionam no futuro que não foi usado para escolhê-los?**

Um corte único treino/teste pode ser sorte. A forma madura desliza a janela:
otimiza num período, testa no bloco seguinte intocado, desliza, repete. Cada
período serve como teste out-of-sample exatamente uma vez e só depois entra na
janela de otimização seguinte. Concatenado, isso é a curva honesta da tese.

**Walk-Forward Efficiency (WFE)** = retorno out-of-sample / retorno in-sample,
ambos anualizados. Acima de ~50-60% sugere robustez real; consistentemente
baixa é overfitting. É o número que transforma "é robusta?" de opinião em
medida — a saída principal do laboratório.

## O bug mais perigoso do laboratório

Se a escolha da combinação enxergar o bloco de teste em qualquer ponto, o WFE
fica lindo e mentiroso. A defesa aqui é estrutural, não de disciplina: a fase
de treino roda um sweep cujo `SweepSpec` tem `end = train_end`, então os dados
do teste nem chegam a ser lidos — não existe caminho de código em que a
seleção veja o futuro. `tests/test_walkforward.py` verifica isso por dois
ângulos independentes (datas pedidas e invariância da escolha à perturbação do
futuro).

## Limitação conhecida: warm-up dentro do bloco de teste

O bloco de teste é rodado sozinho, começando em `test_start`. Uma estratégia
com lookback longo (momentum 12-1 precisa de 252 barras) passa o início do
bloco em warm-up, sem poder operar. A alternativa — carregar histórico anterior
só para aquecer e medir apenas o trecho de teste — exigiria o motor devolver
equity de um sub-trecho, isto é, reescrever o núcleo por causa de uma feature.
A escolha aqui é conservadora de propósito: ela SUBESTIMA a estratégia (menos
tempo operando), nunca a superestima, e nenhum dado de teste vaza para a
decisão. Use `test_years` grande o bastante para o warm-up ser uma fração
pequena do bloco.

## O que este módulo NÃO faz

Não conta quantas vezes você já espiou um período (o "fitting implícito" do
Componente 2 da spec). Se você olha o out-of-sample, muda a grade e roda de
novo, contaminou o teste com a sua própria memória — e isso nenhum código
detecta. `n_combos` no CSV cobre só a parte mecânica do obstáculo 4.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from batch import append_row
from data import FetchFn
from sweep import Combo, ComboSummary, SweepSpec, aggregate_by_combo, run_sweep, valid_combos

DEFAULT_LOG_PATH = Path("walkforward_runs.csv")

# A métrica do WFE é sempre retorno anualizado, por definição da spec, mesmo
# quando a SELEÇÃO da combinação usa outra métrica (Sharpe, tipicamente). São
# perguntas diferentes: "qual combinação escolher" e "quanto da performance
# sobreviveu fora da amostra".
WFE_METRIC = "annualized_return"


class WalkForwardError(Exception):
    """Configuração ou execução de walk-forward inviável."""


@dataclass(frozen=True)
class Window:
    """Uma janela: um bloco de treino e o bloco de teste imediatamente seguinte."""

    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class WalkForwardSpec:
    sweep: SweepSpec  # universo, estratégia, grade, custos — o período é fatiado
    train_years: int
    test_years: int
    scheme: str = "expanding"
    select_by: str = "sharpe"


@dataclass(frozen=True)
class WindowResult:
    window: Window
    chosen: ComboSummary
    train_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    in_sample: float | None
    out_of_sample: float | None

    @property
    def wfe(self) -> float | None:
        return walk_forward_efficiency(self.in_sample, self.out_of_sample)


@dataclass(frozen=True)
class WalkForwardResult:
    run_id: str
    windows: list[WindowResult]
    rows: list[dict[str, Any]]
    wfe: float | None
    failures: list[tuple[str, str, str]]


def make_windows(
    start: date,
    end: date,
    train_years: int,
    test_years: int,
    scheme: str = "expanding",
) -> list[Window]:
    """Fatia [start, end] em janelas treino→teste contíguas e sem sobreposição.

    Esquema `expanding`: a origem do treino fica fixa e o treino cresce
    absorvendo o bloco de teste da janela anterior. É o mais simples de
    raciocinar e o que a spec manda construir primeiro.

    Janela de teste que não cabe inteira no histórico é DESCARTADA: anualizar
    meio bloco como se fosse um bloco inteiro inventa performance.
    """
    if scheme != "expanding":
        raise WalkForwardError(
            f"scheme '{scheme}' ainda não implementado — só 'expanding' existe. "
            f"Rodar expanding sob o nome 'rolling' devolveria um resultado com o "
            f"nome errado, que é pior do que não rodar."
        )
    if train_years < 1 or test_years < 1:
        raise WalkForwardError("train_years e test_years precisam ser >= 1")
    if start >= end:
        raise WalkForwardError(f"start ({start}) precisa ser anterior a end ({end})")

    windows: list[Window] = []
    train_end = _shift_years(start, train_years) - pd.Timedelta(days=1)

    while True:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = _shift_years(test_start, test_years) - pd.Timedelta(days=1)
        if test_end > end:
            break
        windows.append(
            Window(
                index=len(windows),
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_end = test_end  # expanding: o teste de agora é treino da próxima

    if not windows:
        raise WalkForwardError(
            f"nenhuma janela cabe entre {start} e {end} com treino de {train_years} "
            f"ano(s) e teste de {test_years} ano(s). Poucas janelas = WFE ruidoso; "
            f"zero janelas = nada a medir."
        )
    return windows


def _shift_years(d: date, years: int) -> date:
    return (pd.Timestamp(d) + pd.DateOffset(years=years)).date()


def walk_forward_efficiency(
    in_sample: float | None, out_of_sample: float | None
) -> float | None:
    """WFE = retorno out-of-sample / retorno in-sample (anualizados).

    Devolve None quando o in-sample é <= 0: a razão até existe aritmeticamente,
    mas "manteve 200% da performance" de um treino que perdeu dinheiro não
    significa nada. Um out-of-sample negativo, ao contrário, é informação
    legítima (a tese virou prejuízo fora da amostra) e vira WFE negativa.
    """
    if in_sample is None or out_of_sample is None or in_sample <= 0:
        return None
    return out_of_sample / in_sample


def run_walk_forward(
    spec: WalkForwardSpec,
    *,
    run_id: str | None = None,
    log_path: Path | None = DEFAULT_LOG_PATH,
    cache_dir: Path = Path("data_cache"),
    fetch_fn: FetchFn | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """Roda o walk-forward completo e devolve os resultados por janela + WFE."""
    windows = make_windows(
        spec.sweep.start, spec.sweep.end, spec.train_years, spec.test_years, spec.scheme
    )
    combos_por_id = {c.combo_id: c for c in valid_combos(spec.sweep)}
    run_id = run_id or f"wf-{pd.Timestamp.now().isoformat(timespec='seconds')}"

    results: list[WindowResult] = []
    all_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []

    for window in windows:
        if verbose:
            print(
                f"\n[janela {window.index}] treino {window.train_start}→{window.train_end} "
                f"| teste {window.test_start}→{window.test_end}"
            )

        # --- treino: a grade inteira, e SÓ com dados até train_end ---------
        train = run_sweep(
            replace(spec.sweep, start=window.train_start, end=window.train_end),
            sweep_id=f"{run_id}-w{window.index}-train",
            train_test="train",
            log_path=None,  # gravação é deste módulo, que enriquece as linhas
            cache_dir=cache_dir,
            fetch_fn=fetch_fn,
            verbose=False,
        )
        failures.extend(train.failures)

        ranking = aggregate_by_combo(train.rows, spec.select_by)
        chosen = next((s for s in ranking if s.value is not None), None)
        if chosen is None:
            raise WalkForwardError(
                f"janela {window.index}: nenhuma combinação produziu '{spec.select_by}' "
                f"válido no treino — não há como escolher uma para o teste."
            )

        # --- teste: só a combinação escolhida, no bloco intocado -----------
        combo = combos_por_id[chosen.combo_id]
        test = run_sweep(
            _fix_to_combo(spec.sweep, combo, window.test_start, window.test_end),
            sweep_id=f"{run_id}-w{window.index}-test",
            train_test="test",
            log_path=None,
            cache_dir=cache_dir,
            fetch_fn=fetch_fn,
            verbose=False,
        )
        failures.extend(test.failures)

        train_rows = _enrich(train.rows, run_id, window.index, chosen.combo_id)
        test_rows = _enrich(test.rows, run_id, window.index, chosen.combo_id)

        in_sample = _mean_metric(
            [r for r in train_rows if r["combo_id"] == chosen.combo_id], WFE_METRIC
        )
        out_of_sample = _mean_metric(test_rows, WFE_METRIC)

        window_result = WindowResult(
            window=window,
            chosen=chosen,
            train_rows=train_rows,
            test_rows=test_rows,
            in_sample=in_sample,
            out_of_sample=out_of_sample,
        )
        results.append(window_result)

        for row in train_rows + test_rows:
            all_rows.append(row)
            if log_path is not None:
                append_row(row, path=log_path)

        if verbose:
            print(
                f"  escolhida {chosen.combo_id} {chosen.params} "
                f"({spec.select_by}={chosen.value:.4f})"
            )
            print(f"  in-sample {_fmt(in_sample)} → out-of-sample {_fmt(out_of_sample)}")

    # WFE agregada sobre as MÉDIAS, não média das razões: uma janela com
    # in-sample perto de zero dominaria a média de razões e viraria o veredicto
    # inteiro por um artefato de divisão.
    wfe = walk_forward_efficiency(
        _mean([w.in_sample for w in results]), _mean([w.out_of_sample for w in results])
    )

    if verbose:
        print_walk_forward(
            WalkForwardResult(run_id, results, all_rows, wfe, failures), spec.select_by
        )
        if log_path is not None:
            print(f"\n{len(all_rows)} linha(s) em {log_path.resolve()}")

    return WalkForwardResult(
        run_id=run_id, windows=results, rows=all_rows, wfe=wfe, failures=failures
    )


def _fix_to_combo(spec: SweepSpec, combo: Combo, start: date, end: date) -> SweepSpec:
    """SweepSpec reduzido a uma única combinação, no período dado."""
    return replace(
        spec,
        start=start,
        end=end,
        strategy_grid={k: [v] for k, v in combo.strategy_params.items()},
        stop_grid={k: [v] for k, v in combo.stop_params.items()},
    )


def _enrich(
    rows: Sequence[dict[str, Any]], run_id: str, window: int, chosen_combo_id: str
) -> list[dict[str, Any]]:
    """Prefixa a proveniência do walk-forward nas linhas do sweep.

    `chosen_combo_id` vai também nas linhas de treino: é o que permite, lendo o
    CSV, separar "a combinação que venceu" do resto da nuvem daquela janela —
    e a nuvem é o suporte do obstáculo 4.
    """
    return [
        {"run_id": run_id, "window": window, "chosen_combo_id": chosen_combo_id, **row}
        for row in rows
    ]


def _mean_metric(rows: Sequence[dict[str, Any]], metric: str) -> float | None:
    return _mean([row.get(metric) for row in rows])


def _mean(values: Sequence[float | None]) -> float | None:
    validos = [v for v in values if v is not None and v == v]
    return sum(validos) / len(validos) if validos else None


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def print_walk_forward(result: WalkForwardResult, select_by: str) -> None:
    """Relatório de fechamento: uma linha por janela e o veredicto do WFE."""
    print(f"\n=== Walk-forward: {len(result.windows)} janela(s) ===")
    print(f"{'jan':<5}{'teste':<26}{'combo':<8}{'in-sample':>12}{'out-sample':>12}{'WFE':>9}")
    for w in result.windows:
        wfe = "N/A" if w.wfe is None else f"{w.wfe:.0%}"
        periodo = f"{w.window.test_start}→{w.window.test_end}"
        print(
            f"{w.window.index:<5}{periodo:<26}{w.chosen.combo_id:<8}"
            f"{_fmt(w.in_sample):>12}{_fmt(w.out_of_sample):>12}{wfe:>9}"
        )

    escolhas = {w.chosen.combo_id for w in result.windows}
    print(f"\nCombinação escolhida por {select_by}; {len(escolhas)} combinação(ões) distinta(s)")
    print(
        "Instabilidade da escolha entre janelas é sinal ruim por si só: se cada "
        "janela elege um parâmetro diferente, não há regime estável para explorar."
    )

    if result.wfe is None:
        print("\nWFE: N/A (in-sample médio <= 0 — a razão não significaria nada)")
    else:
        veredicto = (
            "sugere robustez real" if result.wfe >= 0.5 else "cheira a overfitting"
        )
        print(f"\nWFE agregada: {result.wfe:.0%} — {veredicto}")
        print("Leitura: acima de ~50-60% a tese mantém metade da performance fora da amostra.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward (Componente 2 do laboratório). "
        "A entrada é um experimento YAML com a seção walk_forward — ver lab.py."
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
        if experiment.walk_forward is None:
            raise ExperimentError(
                f"{args.experiment} não declara a seção 'walk_forward'. "
                f"Sem ela, use `python lab.py run {args.experiment}` (sweep simples)."
            )
        run_experiment(experiment, log_path=args.log_path)
    except (ExperimentError, WalkForwardError) as exc:
        raise SystemExit(f"Walk-forward abortado: {exc}")


if __name__ == "__main__":
    main()

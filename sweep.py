"""Componente 1 do laboratório (SPEC_LAB.md): sweep de parâmetros.

Pergunta que este módulo responde: **o bom resultado é um platô ou um pico?**
Um Sharpe alto num ponto isolado, cercado de fracasso em toda a vizinhança, é
sorte de tuning. Um Sharpe alto no meio de uma região contígua que também
funciona é candidato a sinal. Só dá para distinguir os dois vendo o espaço
inteiro — por isso aqui se varre uma grade declarada, não se "busca o melhor"
com otimizador.

Camada POR CIMA do que já existe: cada combinação da grade é uma execução do
mesmo pipeline de `main.run` sobre os mesmos tickers, exatamente como o
`batch.py` faz. O motor, as estratégias e os stops não sabem que existe um
sweep.

O que este módulo acrescenta ao que o batch já gravava é **proveniência**:

- `sweep_id`  — identifica a varredura inteira.
- `combo_id`  — identifica a combinação dentro da varredura.
- `n_combos`  — quantas combinações competiram. Esta é a defesa do obstáculo 4
  (correção por número de tentativas): um Sharpe X como resultado único vale
  muito mais que o mesmo X como melhor de 200 tentativas. Sem esta coluna
  gravada junto do resultado, daqui a três meses ninguém sabe qual era o caso.
- `train_test` — rótulo de treino/teste, preenchido por quem chama (o
  walk-forward do Componente 2 reusa este sweep marcando cada janela).

O CSV é separado do `runs.csv` de propósito: aquele é o acervo curado dos
batches que sustentam o `CONCLUSOES.md`, com uma linha por decisão de pesquisa.
Um sweep despeja centenas de linhas de uma vez e afogaria esse histórico.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from batch import TICKERS, append_row, flatten
from data import DEFAULT_MIN_MEDIAN_TURNOVER, FetchFn
from main import run
from stops import AtrStop, FixedPctStop, NoStop
from strategy import BollingerReversion, MovingAverageCrossover, TimeSeriesMomentum

DEFAULT_LOG_PATH = Path("sweep_runs.csv")

# Teto default de combinações. 4 parâmetros × 4 valores = 256 combinações × 10
# tickers = 2560 backtests: já é uma varredura de minutos. O teto não existe
# para proteger a máquina — existe para você declarar conscientemente quando
# está subindo a aposta do obstáculo 4.
DEFAULT_MAX_COMBOS = 256


class SweepError(Exception):
    """Erro base do sweep."""


class SweepTooLargeError(SweepError):
    """A grade produz mais combinações do que o teto declarado."""


class EmptySweepError(SweepError):
    """Nenhuma combinação da grade é válida — o sweep testaria nada."""


@dataclass(frozen=True)
class Combo:
    """Uma combinação concreta da grade, já validada."""

    combo_id: str
    strategy_params: dict[str, Any]
    stop_params: dict[str, Any]

    @property
    def all_params(self) -> dict[str, Any]:
        return {**self.strategy_params, **self.stop_params}


@dataclass(frozen=True)
class SweepSpec:
    """A declaração de um experimento de varredura.

    Deliberadamente um objeto de dados e não um monte de argumentos soltos: é
    ele que o Componente 5 (config declarativo em arquivo) vai materializar, e
    é ele que o walk-forward vai reusar trocando só `start`/`end`.
    """

    strategy_class: type
    strategy_grid: Mapping[str, Sequence[Any]]
    tickers: Sequence[str]
    start: date
    end: date
    stop_class: type = NoStop
    stop_grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    initial_capital: float = 10_000.0
    risk_pct: float = 0.01
    brokerage: float = 0.0
    slippage_bps: float = 5.0
    min_median_turnover: float = DEFAULT_MIN_MEDIAN_TURNOVER
    max_combos: int = DEFAULT_MAX_COMBOS
    hypothesis: str = ""


@dataclass(frozen=True)
class SweepResult:
    sweep_id: str
    combos: list[Combo]
    rows: list[dict[str, Any]]
    failures: list[tuple[str, str, str]]  # (combo_id, ticker, mensagem)


@dataclass(frozen=True)
class ComboSummary:
    """Uma combinação resumida em um número: a média da métrica no universo."""

    combo_id: str
    params: dict[str, Any]
    value: float | None
    n_tickers: int


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Produto cartesiano da grade, em ordem determinística.

    O primeiro parâmetro varia mais devagar (ordem de `itertools.product`), o
    que deixa o CSV legível: as linhas de um mesmo valor do primeiro parâmetro
    ficam contíguas.

    Grade vazia devolve `[{}]` — "nada para varrer" é uma execução com os
    defaults, não zero execuções. Já uma lista vazia em algum parâmetro é erro:
    o produto cartesiano com fator vazio é vazio, e o sweep sumiria em
    silêncio reportando sucesso.
    """
    for name, values in grid.items():
        if len(values) == 0:
            raise ValueError(
                f"grade do parâmetro '{name}' está vazia — o produto cartesiano "
                f"seria vazio e o sweep não rodaria nada"
            )

    names = list(grid.keys())
    return [dict(zip(names, values)) for values in itertools.product(*grid.values())]


def valid_combos(spec: SweepSpec) -> list[Combo]:
    """Expande a grade e descarta as combinações inválidas ANTES de rodar.

    A regra de validade não é reimplementada aqui: ela é consultada tentando
    construir a estratégia e o stop. `MovingAverageCrossover` já rejeita
    `fast >= slow` no `__post_init__`, então o filtro acompanha automaticamente
    qualquer regra nova que uma estratégia futura declare.

    Só `ValueError` conta como "combinação inválida". `TypeError` (parâmetro
    inexistente na classe) é erro de programação na grade e sobe — senão um
    nome errado viraria "todas inválidas" e o sweep reportaria ter testado
    algo que nunca existiu.
    """
    strategy_combos = expand_grid(spec.strategy_grid)
    stop_combos = expand_grid(spec.stop_grid)

    combos: list[Combo] = []
    for strategy_params, stop_params in itertools.product(strategy_combos, stop_combos):
        try:
            spec.strategy_class(**strategy_params)
            spec.stop_class(**stop_params)
        except ValueError:
            continue
        combos.append(
            Combo(
                combo_id=f"c{len(combos):03d}",
                strategy_params=strategy_params,
                stop_params=stop_params,
            )
        )

    if not combos:
        raise EmptySweepError(
            f"nenhuma das {len(strategy_combos) * len(stop_combos)} combinações da "
            f"grade é válida para {spec.strategy_class.__name__}/{spec.stop_class.__name__}"
        )

    if len(combos) > spec.max_combos:
        raise SweepTooLargeError(
            f"{len(combos)} combinações válidas ultrapassam o teto de "
            f"{spec.max_combos}. Serão {len(combos) * len(spec.tickers)} backtests. "
            f"Reduza a grade ou suba max_combos conscientemente — cada combinação "
            f"é mais uma tentativa a descontar no obstáculo 4."
        )

    return combos


def run_sweep(
    spec: SweepSpec,
    *,
    sweep_id: str | None = None,
    train_test: str = "full",
    log_path: Path | None = DEFAULT_LOG_PATH,
    cache_dir: Path = Path("data_cache"),
    fetch_fn: FetchFn | None = None,
    verbose: bool = True,
) -> SweepResult:
    """Roda a grade inteira e devolve uma linha por (combinação, ticker).

    - A validação da grade acontece toda antes do primeiro backtest: um sweep
      grande demais aborta sem ter escrito nada.
    - Um ticker que falha (sem dados, ilíquido) não derruba a varredura — vira
      entrada em `failures` e o loop segue. Mesma convenção do `batch.py`.
    - `log_path=None` não grava nada (usado pelo walk-forward, que decide
      sozinho o que persistir).
    """
    combos = valid_combos(spec)
    sweep_id = sweep_id or f"sweep-{datetime.now().isoformat(timespec='seconds')}"

    if not spec.hypothesis and verbose:
        print("AVISO: sweep sem hipótese registrada.\n")

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []

    for combo in combos:
        stop_loss = spec.stop_class(**combo.stop_params)

        for ticker in spec.tickers:
            try:
                report, _result, _prices = run(
                    ticker=ticker,
                    start=spec.start,
                    end=spec.end,
                    # instância nova por execução: estratégias são dataclasses
                    # sem estado hoje, mas depender disso é convite a bug
                    strategy=spec.strategy_class(**combo.strategy_params),
                    stop_loss=stop_loss,
                    initial_capital=spec.initial_capital,
                    risk_pct=spec.risk_pct,
                    brokerage=spec.brokerage,
                    slippage_bps=spec.slippage_bps,
                    cache_dir=cache_dir,
                    fetch_fn=fetch_fn,
                    min_median_turnover=spec.min_median_turnover,
                )
            except Exception as exc:
                failures.append((combo.combo_id, ticker, f"{type(exc).__name__}: {exc}"))
                if verbose:
                    print(f"  {combo.combo_id} {ticker:>10}  FALHOU ({type(exc).__name__})")
                continue

            row: dict[str, Any] = {
                "sweep_id": sweep_id,
                "combo_id": combo.combo_id,
                "n_combos": len(combos),
                "train_test": train_test,
                "hypothesis": spec.hypothesis,
                "ticker": ticker,
                "start": spec.start.isoformat(),
                "end": spec.end.isoformat(),
                "strategy": spec.strategy_class.__name__,
                "stop": spec.stop_class.__name__,
                **{f"param_{k}": v for k, v in combo.all_params.items()},
                "risk_pct": spec.risk_pct,
                "brokerage": spec.brokerage,
                "slippage_bps": spec.slippage_bps,
                **flatten(report),
            }
            rows.append(row)
            if log_path is not None:
                append_row(row, path=log_path)

        if verbose:
            print(f"{combo.combo_id}  {combo.all_params}  ok")

    if verbose and log_path is not None:
        print(f"\n{len(rows)} linha(s) em {log_path.resolve()}")

    return SweepResult(sweep_id=sweep_id, combos=combos, rows=rows, failures=failures)


def aggregate_by_combo(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> list[ComboSummary]:
    """Resume cada combinação na média da métrica sobre os tickers.

    Métricas ausentes (`None`/`NaN` — ex.: Sharpe de uma execução sem trades)
    são ignoradas na média e não contam em `n_tickers`, para a média não ser
    puxada por um zero que não existiu. Combinação sem nenhuma métrica válida
    fica com `value=None` em vez de sumir da lista: "não deu para medir" é
    informação diferente de "não foi testada".

    Ordena do melhor para o pior; combinações sem valor vão para o fim.
    """
    ordered_ids: list[str] = []
    params_by_id: dict[str, dict[str, Any]] = {}
    values_by_id: dict[str, list[float]] = {}

    for row in rows:
        combo_id = row["combo_id"]
        if combo_id not in values_by_id:
            ordered_ids.append(combo_id)
            values_by_id[combo_id] = []
            params_by_id[combo_id] = {
                key[len("param_") :]: value
                for key, value in row.items()
                if key.startswith("param_")
            }
        value = row.get(metric)
        if value is not None and value == value:  # descarta None e NaN
            values_by_id[combo_id].append(float(value))

    summaries = [
        ComboSummary(
            combo_id=combo_id,
            params=params_by_id[combo_id],
            value=(sum(v) / len(v)) if v else None,
            n_tickers=len(v),
        )
        for combo_id in ordered_ids
        for v in [values_by_id[combo_id]]
    ]
    return sorted(
        summaries,
        key=lambda s: (s.value is not None, s.value if s.value is not None else 0.0),
        reverse=True,
    )


def print_distribution(summaries: Sequence[ComboSummary], metric: str, top: int = 10) -> None:
    """Imprime o topo do ranking E a distribuição inteira.

    A distribuição não é decoração: é o suporte visual do obstáculo 4. Se a
    melhor combinação é um outlier solitário muito acima da nuvem, desconfie;
    se ela está no topo de um platô povoado, confie mais.
    """
    values = sorted(s.value for s in summaries if s.value is not None)
    if not values:
        print("Nenhuma combinação com métrica válida.")
        return

    print(f"\n=== Sweep: {len(summaries)} combinações, métrica '{metric}' ===")
    print(f"{'combo':<8}{'valor':>10}  parâmetros")
    for summary in summaries[:top]:
        valor = "N/A" if summary.value is None else f"{summary.value:.4f}"
        print(f"{summary.combo_id:<8}{valor:>10}  {summary.params}")

    mediana = values[len(values) // 2]
    print(
        f"\nDistribuição — pior {values[0]:.4f} | mediana {mediana:.4f} | "
        f"melhor {values[-1]:.4f}"
    )
    print(
        f"A melhor de {len(summaries)} tentativas. Quanto maior esse número, "
        f"mais alta a barra para acreditar no topo (obstáculo 4)."
    )


def _int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep de parâmetros (Componente 1 do laboratório)",
        epilog="Listas são separadas por vírgula: --fast-window 5,9,20",
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--strategy", choices=["mac", "bollinger", "momentum"], default="mac")
    parser.add_argument("--fast-window", type=_int_list, default=[9])
    parser.add_argument("--slow-window", type=_int_list, default=[21])
    parser.add_argument("--boll-window", type=_int_list, default=[20])
    parser.add_argument("--boll-k", type=_float_list, default=[2.0])
    parser.add_argument("--mom-lookback", type=_int_list, default=[252])
    parser.add_argument("--mom-skip", type=_int_list, default=[21])
    parser.add_argument("--mom-threshold", type=_float_list, default=[0.0])
    parser.add_argument("--stop", choices=["pct", "atr", "none"], default="atr")
    parser.add_argument("--stop-pct", type=_float_list, default=[0.05])
    parser.add_argument("--atr-period", type=_int_list, default=[14])
    parser.add_argument("--atr-multiplier", type=_float_list, default=[2.0])
    parser.add_argument("--no-stop-distance", type=_float_list, default=[0.50])
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--brokerage", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_MEDIAN_TURNOVER)
    parser.add_argument("--max-combos", type=int, default=DEFAULT_MAX_COMBOS)
    parser.add_argument("--select-by", default="sharpe", help="Métrica do ranking impresso")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--hypothesis",
        default="",
        help="Por que este sweep existe. Preencha — o CSV sem isso é inútil em 3 meses.",
    )
    return parser.parse_args()


def _spec_from_args(args: argparse.Namespace) -> SweepSpec:
    if args.strategy == "mac":
        strategy_class: type = MovingAverageCrossover
        strategy_grid = {"fast_window": args.fast_window, "slow_window": args.slow_window}
    elif args.strategy == "bollinger":
        strategy_class = BollingerReversion
        strategy_grid = {"window": args.boll_window, "k": args.boll_k}
    else:
        strategy_class = TimeSeriesMomentum
        strategy_grid = {
            "lookback": args.mom_lookback,
            "skip": args.mom_skip,
            "threshold": args.mom_threshold,
        }

    if args.stop == "pct":
        stop_class: type = FixedPctStop
        stop_grid = {"pct": args.stop_pct}
    elif args.stop == "atr":
        stop_class = AtrStop
        stop_grid = {"period": args.atr_period, "multiplier": args.atr_multiplier}
    else:
        stop_class = NoStop
        stop_grid = {"distance_pct": args.no_stop_distance}

    return SweepSpec(
        strategy_class=strategy_class,
        strategy_grid=strategy_grid,
        tickers=TICKERS,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        stop_class=stop_class,
        stop_grid=stop_grid,
        risk_pct=args.risk_pct,
        brokerage=args.brokerage,
        slippage_bps=args.slippage_bps,
        min_median_turnover=args.min_turnover,
        max_combos=args.max_combos,
        hypothesis=args.hypothesis,
    )


def main() -> None:
    # mesma razão de main.py: o console default do Windows corrompe acentos
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    spec = _spec_from_args(args)
    result = run_sweep(spec, log_path=args.log_path)
    print_distribution(aggregate_by_combo(result.rows, args.select_by), args.select_by)

    if result.failures:
        print(f"\n{len(result.failures)} falha(s):")
        for combo_id, ticker, msg in result.failures[:20]:
            print(f"  {combo_id} {ticker}: {msg}")


if __name__ == "__main__":
    main()

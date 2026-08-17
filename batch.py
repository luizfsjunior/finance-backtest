"""
Fase 1 — varredura de tickers com a configuração default.

Pergunta que este script responde:
    Em quantos ativos o cruzamento de médias bate o buy & hold?

Uso:
    python batch.py
    python batch.py --stop atr --hypothesis "ATR vence o stop percentual"

Saída: runs.csv (append). Cada execução vira uma linha.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from datetime import date, datetime
from pathlib import Path
from typing import Any

from main import run
from strategy import BollingerReversion, MovingAverageCrossover, TimeSeriesMomentum
from stops import AtrStop, FixedPctStop, NoStop

LOG_PATH = Path("runs.csv")

TICKERS = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "WEGE3.SA",
    "ABEV3.SA",
    "B3SA3.SA",
    "RENT3.SA",
    "SUZB3.SA",
    "RADL3.SA",
]


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Achata um dataclass (possivelmente aninhado) em dict plano.

    Feito por introspecção de propósito: o CSV acompanha qualquer mudança
    no MetricsReport sem precisar editar este script.
    """
    if not dataclasses.is_dataclass(obj):
        return {prefix.rstrip("_"): obj}

    out: dict[str, Any] = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        key = f"{prefix}{field.name}"
        if dataclasses.is_dataclass(value):
            out.update(flatten(value, prefix=f"{key}_"))
        else:
            out[key] = value
    return out


def append_row(row: dict[str, Any], path: Path = LOG_PATH) -> None:
    """Grava uma linha, criando o cabeçalho na primeira vez.

    Se o schema mudar entre execuções, escreve um novo cabeçalho em vez de
    silenciosamente desalinhar as colunas.

    `path` é parâmetro (e não a constante direta) porque o sweep do laboratório
    escreve no seu próprio CSV — ver `sweep.py`.
    """
    existing_header: list[str] | None = None
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            first = f.readline().strip()
            existing_header = first.split(",") if first else None

    header = list(row.keys())
    write_header = existing_header != header

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument(
        "--strategy",
        choices=["mac", "bollinger", "momentum"],
        default="mac",
        help="mac = MovingAverageCrossover; bollinger = BollingerReversion; momentum = TimeSeriesMomentum",
    )
    parser.add_argument("--fast-window", type=int, default=9)
    parser.add_argument("--slow-window", type=int, default=21)
    parser.add_argument("--boll-window", type=int, default=20)
    parser.add_argument("--boll-k", type=float, default=2.0)
    parser.add_argument("--mom-lookback", type=int, default=252, help="Dias úteis; 252 ≈ 12 meses")
    parser.add_argument("--mom-skip", type=int, default=21, help="Dias úteis pulados; 21 ≈ 1 mês")
    parser.add_argument(
        "--mom-threshold", type=float, default=0.0,
        help="Banda morta: |momentum| deve ser > threshold para trocar de estado (ex.: 0.02 = ±2%%)",
    )
    parser.add_argument(
        "--mom-monthly", action="store_true",
        help="Rebalancear só no primeiro pregão de cada mês (evita whipsaw diário)",
    )
    parser.add_argument("--stop", choices=["pct", "atr", "none"], default="pct")
    parser.add_argument("--stop-pct", type=float, default=0.05)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--atr-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--no-stop-distance",
        type=float,
        default=0.50,
        help="Usado com --stop none: distância do stop de fachada (fração da entrada)",
    )
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--brokerage", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--min-turnover", type=float, default=1_000_000.0)
    parser.add_argument(
        "--hypothesis",
        default="",
        help="Por que este batch existe. Preencha — o CSV sem isso é inútil em 3 meses.",
    )
    args = parser.parse_args()

    if not args.hypothesis:
        print("AVISO: rodando sem hipótese registrada.\n")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    batch_id = datetime.now().isoformat(timespec="seconds")

    if args.stop == "pct":
        stop_loss = FixedPctStop(pct=args.stop_pct)
    elif args.stop == "atr":
        stop_loss = AtrStop(period=args.atr_period, multiplier=args.atr_multiplier)
    else:
        stop_loss = NoStop(distance_pct=args.no_stop_distance)

    if args.strategy == "mac":
        strategy_factory = lambda: MovingAverageCrossover(
            fast_window=args.fast_window, slow_window=args.slow_window
        )
    elif args.strategy == "bollinger":
        strategy_factory = lambda: BollingerReversion(
            window=args.boll_window, k=args.boll_k
        )
    else:
        strategy_factory = lambda: TimeSeriesMomentum(
            lookback=args.mom_lookback,
            skip=args.mom_skip,
            threshold=args.mom_threshold,
            monthly_rebalance=args.mom_monthly,
        )

    params = {
        "strategy": args.strategy,
        "fast_window": args.fast_window if args.strategy == "mac" else "",
        "slow_window": args.slow_window if args.strategy == "mac" else "",
        "boll_window": args.boll_window if args.strategy == "bollinger" else "",
        "boll_k": args.boll_k if args.strategy == "bollinger" else "",
        "mom_lookback": args.mom_lookback if args.strategy == "momentum" else "",
        "mom_skip": args.mom_skip if args.strategy == "momentum" else "",
        "mom_threshold": args.mom_threshold if args.strategy == "momentum" else "",
        "mom_monthly": args.mom_monthly if args.strategy == "momentum" else "",
        "stop": args.stop,
        "stop_pct": args.stop_pct if args.stop == "pct" else "",
        "atr_period": args.atr_period if args.stop == "atr" else "",
        "atr_multiplier": args.atr_multiplier if args.stop == "atr" else "",
        "no_stop_distance": args.no_stop_distance if args.stop == "none" else "",
        "risk_pct": args.risk_pct,
        "brokerage": args.brokerage,
        "slippage_bps": args.slippage_bps,
    }

    failures: list[tuple[str, str]] = []

    for ticker in TICKERS:
        try:
            report, _result, _prices = run(
                ticker=ticker,
                start=start,
                end=end,
                strategy=strategy_factory(),
                stop_loss=stop_loss,
                initial_capital=args.initial_capital,
                risk_pct=args.risk_pct,
                brokerage=args.brokerage,
                slippage_bps=args.slippage_bps,
                min_median_turnover=args.min_turnover,
            )
        except Exception as exc:  # um ticker ruim não pode derrubar o batch
            failures.append((ticker, f"{type(exc).__name__}: {exc}"))
            print(f"{ticker:>10}  FALHOU  ({type(exc).__name__})")
            continue

        row: dict[str, Any] = {
            "batch_id": batch_id,
            "hypothesis": args.hypothesis,
            "ticker": ticker,
            "start": args.start,
            "end": args.end,
            **params,
            **flatten(report),
        }
        append_row(row)
        print(f"{ticker:>10}  ok")

    print(f"\nGravado em {LOG_PATH.resolve()}")
    if failures:
        print("\nFalhas:")
        for ticker, msg in failures:
            print(f"  {ticker}: {msg}")


if __name__ == "__main__":
    main()
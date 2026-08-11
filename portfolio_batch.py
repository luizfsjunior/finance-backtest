"""Runner de batch para portfolios cross-sectional.

Uma linha do `portfolio_runs.csv` = UM backtest de portfolio inteiro (não
um por ticker como no `batch.py` single-asset). Cada configuração de
(universo, estratégia, K, cadência de rebalance, período) vira uma linha.

Benchmarks embutidos:
  - Equal-weight buy-and-hold do universo (primário — comparação
    apples-to-apples: mesmo universo, mesma máscara de elegibilidade, sem
    seleção).
  - IBOV via BOVA11.SA (secundário — contexto de mercado).

Uso:
    python portfolio_batch.py --hypothesis "cross-sectional 12-1 top-3 vence single-asset"
    python portfolio_batch.py --top-k 5 --rebalance monthly
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from costs import Costs
from data import DEFAULT_MIN_MEDIAN_TURNOVER
from metrics import (
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from portfolio_backtest import (
    portfolio_buy_and_hold_equity_curve,
    run_portfolio_backtest,
)
from portfolio_data import get_portfolio_data
from portfolio_strategy import CrossSectionalMomentum

LOG_PATH = Path("portfolio_runs.csv")

TICKERS = [
    # Núcleo original (10) — mantido para comparação direta com batch anterior
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA",
    "ABEV3.SA", "B3SA3.SA", "RENT3.SA", "SUZB3.SA", "RADL3.SA",
    # Bancos e financeiro (+2)
    "BBAS3.SA", "SANB11.SA",
    # Siderurgia/metalurgia (+3)
    "CSNA3.SA", "GGBR4.SA", "USIM5.SA",
    # Energia elétrica (+3)
    "ELET3.SA", "EGIE3.SA", "CMIG4.SA",
    # Varejo/consumo (+4)
    "LREN3.SA", "MGLU3.SA", "NTCO3.SA", "JBSS3.SA",
    # Utilities/infra (+3)
    "SBSP3.SA", "CCRO3.SA", "RAIL3.SA",
    # Papel e celulose (+1)
    "KLBN11.SA",
    # Educação/saúde (+2)
    "COGN3.SA", "HAPV3.SA",
    # Óleo e gás (+1)
    "PRIO3.SA",
]

IBOV_TICKER = "BOVA11.SA"


def _metrics(equity_curve: pd.Series) -> dict[str, float | None]:
    return {
        "total_return": total_return(equity_curve),
        "annualized_return": annualized_return(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe": sharpe_ratio(equity_curve),
        "sortino": sortino_ratio(equity_curve),
    }


def _trade_stats(trades: list) -> dict[str, float | int | None]:
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": None,
            "avg_pnl": None,
            "avg_holding_days": None,
        }
    wins = sum(1 for t in trades if t.is_win)
    return {
        "num_trades": len(trades),
        "win_rate": wins / len(trades),
        "avg_pnl": sum(t.pnl for t in trades) / len(trades),
        "avg_holding_days": sum(t.holding_days for t in trades) / len(trades),
    }


def append_row(row: dict[str, Any]) -> None:
    """Grava uma linha em `portfolio_runs.csv`, escrevendo novo cabeçalho se o
    schema mudou (mesmo padrão do `batch.py` single-asset)."""
    existing_header: list[str] | None = None
    if LOG_PATH.exists():
        with LOG_PATH.open(newline="", encoding="utf-8") as f:
            first = f.readline().strip()
            existing_header = first.split(",") if first else None

    header = list(row.keys())
    write_header = existing_header != header

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rebalance", choices=["monthly", "weekly", "quarterly"], default="monthly")
    parser.add_argument("--mom-lookback", type=int, default=252)
    parser.add_argument("--mom-skip", type=int, default=21)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--brokerage", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_MEDIAN_TURNOVER)
    parser.add_argument("--hypothesis", default="")
    args = parser.parse_args()

    if not args.hypothesis:
        print("AVISO: rodando sem hipótese registrada.\n")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    batch_id = datetime.now().isoformat(timespec="seconds")

    costs = Costs(brokerage=args.brokerage, slippage_bps=args.slippage_bps)

    print(f"Carregando universo ({len(TICKERS)} tickers) de {start} a {end}...")
    prices, eligibility, failures = get_portfolio_data(
        TICKERS,
        start,
        end,
        min_median_turnover=args.min_turnover,
    )

    for ticker, msg in failures:
        print(f"  {ticker}: FALHOU — {msg}")
    print(f"Universo efetivo: {len(prices.columns.get_level_values(0).unique())} tickers")

    strategy = CrossSectionalMomentum(lookback=args.mom_lookback, skip=args.mom_skip)

    print("Rodando estratégia cross-sectional momentum...")
    result = run_portfolio_backtest(
        prices=prices,
        eligibility=eligibility,
        strategy=strategy,
        top_k=args.top_k,
        rebalance=args.rebalance,
        initial_capital=args.initial_capital,
        costs=costs,
    )

    print("Rodando benchmark equal-weight buy-and-hold do universo...")
    bh_equal_weight = portfolio_buy_and_hold_equity_curve(
        prices=prices,
        eligibility=eligibility,
        initial_capital=args.initial_capital,
        costs=costs,
    )

    print(f"Carregando benchmark IBOV ({IBOV_TICKER})...")
    ibov_equity: pd.Series | None = None
    try:
        ibov_prices, ibov_elig, ibov_fail = get_portfolio_data(
            [IBOV_TICKER],
            start,
            end,
            min_median_turnover=0.0,
        )
        if not ibov_fail:
            ibov_equity = portfolio_buy_and_hold_equity_curve(
                prices=ibov_prices,
                eligibility=ibov_elig,
                initial_capital=args.initial_capital,
                costs=costs,
            )
    except Exception as exc:
        print(f"  IBOV falhou: {exc}")

    strat_metrics = _metrics(result.equity_curve)
    bh_metrics = _metrics(bh_equal_weight)
    ibov_metrics = _metrics(ibov_equity) if ibov_equity is not None else {
        "total_return": None,
        "annualized_return": None,
        "max_drawdown": None,
        "sharpe": None,
        "sortino": None,
    }
    tstats = _trade_stats(result.trades)

    row: dict[str, Any] = {
        "batch_id": batch_id,
        "hypothesis": args.hypothesis,
        "start": args.start,
        "end": args.end,
        "n_tickers_universe": len(TICKERS),
        "n_tickers_loaded": len(prices.columns.get_level_values(0).unique()),
        "top_k": args.top_k,
        "rebalance": args.rebalance,
        "mom_lookback": args.mom_lookback,
        "mom_skip": args.mom_skip,
        "brokerage": args.brokerage,
        "slippage_bps": args.slippage_bps,
        "min_turnover": args.min_turnover,
        # métricas da estratégia
        **{f"strat_{k}": v for k, v in strat_metrics.items()},
        **{f"strat_{k}": v for k, v in tstats.items()},
        "strat_open_positions_at_end": len(result.open_positions_at_end),
        "strat_num_rebalances": len(result.rebalance_dates),
        # benchmark equal-weight universo
        **{f"bh_ew_{k}": v for k, v in bh_metrics.items()},
        # benchmark IBOV
        **{f"ibov_{k}": v for k, v in ibov_metrics.items()},
    }
    append_row(row)

    print(f"\n=== Resultado ===")
    print(f"{'Métrica':<25}{'Estratégia':>15}{'B&H EW':>15}{'IBOV':>15}")

    def fmt_pct(x):
        return "N/A" if x is None else f"{x * 100:.2f}%"

    def fmt(x):
        return "N/A" if x is None else f"{x:.2f}"

    print(
        f"{'Retorno total':<25}{fmt_pct(strat_metrics['total_return']):>15}"
        f"{fmt_pct(bh_metrics['total_return']):>15}{fmt_pct(ibov_metrics['total_return']):>15}"
    )
    print(
        f"{'Retorno anualizado':<25}{fmt_pct(strat_metrics['annualized_return']):>15}"
        f"{fmt_pct(bh_metrics['annualized_return']):>15}{fmt_pct(ibov_metrics['annualized_return']):>15}"
    )
    print(
        f"{'Drawdown máximo':<25}{fmt_pct(strat_metrics['max_drawdown']):>15}"
        f"{fmt_pct(bh_metrics['max_drawdown']):>15}{fmt_pct(ibov_metrics['max_drawdown']):>15}"
    )
    print(
        f"{'Sharpe':<25}{fmt(strat_metrics['sharpe']):>15}"
        f"{fmt(bh_metrics['sharpe']):>15}{fmt(ibov_metrics['sharpe']):>15}"
    )
    print(
        f"{'Sortino':<25}{fmt(strat_metrics['sortino']):>15}"
        f"{fmt(bh_metrics['sortino']):>15}{fmt(ibov_metrics['sortino']):>15}"
    )
    print(f"\nTrades (períodos com peso > 0): {tstats['num_trades']}")
    if tstats["num_trades"]:
        print(f"Win rate: {tstats['win_rate']:.1%}")
        print(f"PnL médio: R$ {tstats['avg_pnl']:.2f}")
        print(f"Holding médio: {tstats['avg_holding_days']:.1f} dias")
    print(f"Rebalances: {len(result.rebalance_dates)}")
    print(f"Posições abertas no fim: {len(result.open_positions_at_end)} — {list(result.open_positions_at_end.keys())}")
    print(f"\nGravado em {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()

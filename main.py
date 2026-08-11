"""Orquestração: busca dados, roda a estratégia, executa o backtest e
imprime/plota o relatório de métricas comparado ao buy-and-hold.

Sem framework de config — só uma função `run()` com parâmetros explícitos e
um `argparse` simples para uso via linha de comando.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtest import BacktestResult, run_backtest
from costs import Costs
from data import DEFAULT_MIN_MEDIAN_TURNOVER, FetchFn, get_data
from metrics import MetricsReport, buy_and_hold_equity_curve, metrics_report
from stops import AtrStop, FixedPctStop, StopLoss
from strategy import MovingAverageCrossover, Strategy


def run(
    ticker: str,
    start: date,
    end: date,
    strategy: Strategy | None = None,
    stop_loss: StopLoss | None = None,
    initial_capital: float = 10_000.0,
    risk_pct: float = 0.01,
    brokerage: float = 0.0,
    slippage_bps: float = 5.0,
    cache_dir: Path = Path("data_cache"),
    fetch_fn: FetchFn | None = None,
    min_median_turnover: float = DEFAULT_MIN_MEDIAN_TURNOVER,
) -> tuple[MetricsReport, BacktestResult, pd.DataFrame]:
    """Roda o pipeline completo para um ticker e devolve o relatório de
    métricas, o resultado bruto do backtest e os preços usados (para plot).

    `strategy` e `stop_loss` são injetados como os contratos que são (não
    como parâmetros de uma implementação específica): é o que permite trocar
    o setup sem tocar no critério de risco e vice-versa. Os defaults abaixo
    existem só para o CLI ter o que rodar sem flags.

    `fetch_fn` existe só para permitir testar `run()` sem rede — mesmo
    propósito do parâmetro homônimo em `data.get_data`.
    """
    strategy = strategy if strategy is not None else MovingAverageCrossover(9, 21)
    stop_loss = stop_loss if stop_loss is not None else FixedPctStop(0.05)

    prices = get_data(
        ticker,
        start,
        end,
        cache_dir=cache_dir,
        fetch_fn=fetch_fn,
        min_median_turnover=min_median_turnover,
    )

    signals = strategy.generate_signals(prices)
    costs = Costs(brokerage=brokerage, slippage_bps=slippage_bps)

    result = run_backtest(
        prices,
        signals,
        initial_capital=initial_capital,
        risk_pct=risk_pct,
        stop_loss_fn=stop_loss,
        costs=costs,
    )

    # os mesmos `costs` do motor vão para o relatório: é o que faz o
    # buy-and-hold ser comparado sob a mesma tabela de custos
    report = metrics_report(
        result.equity_curve,
        result.trades,
        prices["Close"],
        initial_capital=initial_capital,
        costs=costs,
    )
    return report, result, prices


def print_report(ticker: str, report: MetricsReport, result: BacktestResult) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    def num(x: float | None, fmt: str = "{:.2f}") -> str:
        return "N/A" if x is None else fmt.format(x)

    print(f"\n=== Backtest: {ticker} ===")
    print(f"{'Métrica':<28}{'Estratégia':>15}{'Buy & Hold':>15}")
    print(f"{'Retorno total':<28}{pct(report.total_return):>15}{pct(report.benchmark_total_return):>15}")
    print(
        f"{'Retorno anualizado':<28}{pct(report.annualized_return):>15}"
        f"{pct(report.benchmark_annualized_return):>15}"
    )
    print(f"{'Drawdown máximo':<28}{pct(report.max_drawdown):>15}{pct(report.benchmark_max_drawdown):>15}")
    print(f"{'Sharpe':<28}{num(report.sharpe):>15}{num(report.benchmark_sharpe):>15}")
    print(f"{'Sortino':<28}{num(report.sortino):>15}{num(report.benchmark_sortino):>15}")
    print(f"\n{'Número de trades':<28}{report.num_trades:>15}")
    print(f"{'Taxa de acerto':<28}{num(report.win_rate, '{:.1%}'):>15}")
    print(f"{'Payoff médio':<28}{num(report.avg_payoff):>15}")
    print(f"{'Expectativa (R$/trade)':<28}{num(report.expectancy):>15}")
    print(f"{'Tempo médio em posição (d)':<28}{num(report.avg_holding_days):>15}")

    if result.skipped_signals:
        print(f"\n{len(result.skipped_signals)} sinal(is) de entrada ignorado(s):")
        for signal_date, reason in result.skipped_signals:
            print(f"  {signal_date.date()}: {reason}")

    if result.open_position_at_end:
        print("\nAviso: havia posição aberta no fim do período (marcada a mercado, não liquidada).")


def plot_equity_curve(
    result: BacktestResult,
    prices: pd.DataFrame,
    initial_capital: float,
    ticker: str,
    costs: Costs = Costs(),
) -> plt.Figure:
    # mesmos custos do relatório, senão a linha do buy-and-hold no gráfico
    # não é a mesma que gerou os números impressos
    benchmark = buy_and_hold_equity_curve(prices["Close"], initial_capital, costs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.equity_curve.index, result.equity_curve.values, label="Estratégia")
    ax.plot(benchmark.index, benchmark.values, label="Buy & Hold", linestyle="--")
    ax.set_title(f"Equity curve — {ticker}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Patrimônio (R$)")
    ax.legend()
    fig.tight_layout()
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest de cruzamento de médias móveis (B3)")
    parser.add_argument("ticker", help="Ticker com sufixo .SA, ex.: PETR4.SA")
    parser.add_argument("start", help="Data inicial YYYY-MM-DD")
    parser.add_argument("end", help="Data final YYYY-MM-DD")
    parser.add_argument("--fast-window", type=int, default=9)
    parser.add_argument("--slow-window", type=int, default=21)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument(
        "--stop", choices=["pct", "atr"], default="pct", help="Critério de stop loss"
    )
    parser.add_argument("--stop-pct", type=float, default=0.05, help="Usado com --stop pct")
    parser.add_argument("--atr-period", type=int, default=14, help="Usado com --stop atr")
    parser.add_argument("--atr-multiplier", type=float, default=2.0, help="Usado com --stop atr")
    parser.add_argument("--brokerage", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument(
        "--min-turnover",
        type=float,
        default=DEFAULT_MIN_MEDIAN_TURNOVER,
        help="Volume financeiro diário mediano mínimo em R$ (0 desliga o filtro de liquidez)",
    )
    parser.add_argument("--plot", action="store_true", help="Exibe o gráfico da equity curve")
    return parser.parse_args()


def _build_stop(args: argparse.Namespace) -> StopLoss:
    if args.stop == "atr":
        return AtrStop(period=args.atr_period, multiplier=args.atr_multiplier)
    return FixedPctStop(pct=args.stop_pct)


def main() -> None:
    # console do Windows costuma usar um codepage legado (cp1252/cp850) em
    # vez de UTF-8 por padrão, corrompendo os acentos do relatório.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    report, result, prices = run(
        ticker=args.ticker,
        start=datetime.strptime(args.start, "%Y-%m-%d").date(),
        end=datetime.strptime(args.end, "%Y-%m-%d").date(),
        strategy=MovingAverageCrossover(
            fast_window=args.fast_window, slow_window=args.slow_window
        ),
        stop_loss=_build_stop(args),
        initial_capital=args.initial_capital,
        risk_pct=args.risk_pct,
        brokerage=args.brokerage,
        slippage_bps=args.slippage_bps,
        min_median_turnover=args.min_turnover,
    )
    print_report(args.ticker, report, result)
    if args.plot:
        plot_equity_curve(
            result,
            prices,
            args.initial_capital,
            args.ticker,
            costs=Costs(brokerage=args.brokerage, slippage_bps=args.slippage_bps),
        )
        plt.show()


if __name__ == "__main__":
    main()

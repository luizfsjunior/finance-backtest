from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from main import plot_equity_curve, print_report, run
from stops import AtrStop, FixedPctStop
from strategy import MovingAverageCrossover


def _synthetic_ticker_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)
    # tendência com ruído leve, o bastante para gerar cruzamentos de médias
    trend = [10.0 + 0.05 * i for i in range(n)]
    wiggle = [0.3 * ((i % 7) - 3) for i in range(n)]
    close = pd.Series([t + w for t, w in zip(trend, wiggle)], index=idx)

    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], [ticker]])
    df = pd.DataFrame(index=idx, columns=columns, dtype=float)
    df[("Open", ticker)] = close
    df[("High", ticker)] = close + 0.5
    df[("Low", ticker)] = close - 0.5
    df[("Close", ticker)] = close
    # volume alto o bastante para o papel sintético passar no filtro de
    # liquidez com a folga que o preço variável do período exige
    df[("Volume", ticker)] = 1_000_000
    return df


def _oscillating_ticker_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Ciclos longos de alta e baixa, para janelas de média mais lentas
    (as defaults, 9/21) também produzirem cruzamentos."""
    import math

    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)
    period = 80  # ~4 meses de pregão por ciclo completo
    close = pd.Series(
        [20.0 + 6.0 * math.sin(2 * math.pi * i / period) for i in range(n)], index=idx
    )

    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], [ticker]])
    df = pd.DataFrame(index=idx, columns=columns, dtype=float)
    df[("Open", ticker)] = close
    df[("High", ticker)] = close + 0.5
    df[("Low", ticker)] = close - 0.5
    df[("Close", ticker)] = close
    df[("Volume", ticker)] = 1_000_000
    return df


class TestRun:
    def test_pipeline_completo_produz_relatorio_valido(self, tmp_path: Path):
        report, result, prices = run(
            "TEST4.SA",
            date(2024, 1, 2),
            date(2024, 6, 28),
            strategy=MovingAverageCrossover(fast_window=5, slow_window=15),
            cache_dir=tmp_path,
            fetch_fn=_synthetic_ticker_data,
        )

        assert len(result.equity_curve) == len(prices)
        assert report.num_trades == len(result.trades)
        # buy-and-hold do mesmo ativo/período sempre calculado, é o baseline obrigatório
        assert report.benchmark_total_return != 0.0 or report.total_return == 0.0

    def test_print_report_nao_levanta_erro(self, tmp_path: Path, capsys):
        report, result, prices = run(
            "TEST4.SA",
            date(2024, 1, 2),
            date(2024, 6, 28),
            strategy=MovingAverageCrossover(fast_window=5, slow_window=15),
            cache_dir=tmp_path,
            fetch_fn=_synthetic_ticker_data,
        )
        print_report("TEST4.SA", report, result)
        captured = capsys.readouterr()
        assert "TEST4.SA" in captured.out
        assert "Retorno total" in captured.out

    def test_plot_equity_curve_gera_figura(self, tmp_path: Path):
        report, result, prices = run(
            "TEST4.SA",
            date(2024, 1, 2),
            date(2024, 6, 28),
            strategy=MovingAverageCrossover(fast_window=5, slow_window=15),
            cache_dir=tmp_path,
            fetch_fn=_synthetic_ticker_data,
        )
        fig = plot_equity_curve(result, prices, initial_capital=10_000.0, ticker="TEST4.SA")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestInjecaoDeStopEEstrategia:
    def _run_with(self, stop, tmp_path: Path):
        return run(
            "TEST4.SA",
            date(2024, 1, 2),
            date(2024, 6, 28),
            strategy=MovingAverageCrossover(fast_window=5, slow_window=15),
            stop_loss=stop,
            cache_dir=tmp_path,
            fetch_fn=_synthetic_ticker_data,
        )

    def test_trocar_o_stop_muda_o_resultado_sem_tocar_na_estrategia(self, tmp_path: Path):
        # mesma estratégia, mesmo dado: só o critério de risco muda. É a
        # combinação que a separação stop/estratégia existe para permitir.
        report_pct, result_pct, _ = self._run_with(FixedPctStop(0.05), tmp_path)
        report_atr, result_atr, _ = self._run_with(AtrStop(period=14, multiplier=2.0), tmp_path)

        assert result_pct.trades != result_atr.trades
        assert report_pct.total_return != report_atr.total_return

    def test_defaults_rodam_sem_estrategia_nem_stop_explicitos(self, tmp_path: Path):
        # dados com oscilação ampla o bastante para as janelas default (9/21)
        # realmente cruzarem: os dados de tendência suave usados nos outros
        # testes só produzem cruzamento com janelas mais curtas.
        report, result, prices = run(
            "TEST4.SA",
            date(2024, 1, 2),
            date(2025, 6, 30),
            cache_dir=tmp_path,
            fetch_fn=_oscillating_ticker_data,
        )
        assert len(result.equity_curve) == len(prices)
        assert report.num_trades == len(result.trades)
        assert result.trades, "defaults deveriam gerar ao menos um trade nesses dados"

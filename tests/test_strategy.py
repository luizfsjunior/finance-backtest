from __future__ import annotations

import pandas as pd
import pytest

from backtest import run_backtest
from core import Signal
from stops import FixedPctStop
from strategy import MovingAverageCrossover, Strategy


def _prices_from_closes(closes: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(closes))
    close = pd.Series(closes, index=idx)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.1,
            "Low": close - 0.1,
            "Close": close,
            "Volume": 1000,
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


class TestProtocolo:
    def test_moving_average_crossover_satisfaz_o_protocol(self):
        strat = MovingAverageCrossover(fast_window=2, slow_window=4)
        assert isinstance(strat, Strategy)

    def test_fast_maior_ou_igual_a_slow_levanta_erro(self):
        with pytest.raises(ValueError):
            MovingAverageCrossover(fast_window=10, slow_window=10)
        with pytest.raises(ValueError):
            MovingAverageCrossover(fast_window=10, slow_window=5)

    def test_janela_nao_positiva_levanta_erro(self):
        with pytest.raises(ValueError):
            MovingAverageCrossover(fast_window=0, slow_window=5)

    def test_sem_coluna_close_levanta_erro(self):
        strat = MovingAverageCrossover(fast_window=2, slow_window=3)
        prices = pd.DataFrame({"Open": [1, 2, 3]})
        with pytest.raises(ValueError):
            strat.generate_signals(prices)


class TestCruzamento:
    def test_sinais_no_mesmo_indice_de_prices(self):
        prices = _prices_from_closes([10, 11, 12, 13, 12, 11, 10, 9])
        strat = MovingAverageCrossover(fast_window=2, slow_window=4)
        signals = strat.generate_signals(prices)
        assert signals.index.equals(prices.index)
        assert signals.isin([s.value for s in Signal]).all()

    def test_warmup_nao_gera_sinal_espurio(self):
        # série constante: fast_ma == slow_ma sempre que ambas ficam válidas,
        # então "acima" nunca vira True — não deve haver NENHUM ENTER_LONG,
        # nem no instante em que a média lenta passa a existir.
        prices = _prices_from_closes([10.0] * 20)
        strat = MovingAverageCrossover(fast_window=3, slow_window=8)
        signals = strat.generate_signals(prices)
        assert (signals == Signal.ENTER_LONG).sum() == 0
        assert (signals == Signal.EXIT_LONG).sum() == 0

    def test_cruzamento_para_cima_gera_enter_long_uma_unica_vez(self):
        # sobe de forma sustentada: fast cruza slow para cima uma vez e
        # permanece acima -> exatamente 1 ENTER_LONG, sem repetição por barra
        closes = [10.0] * 5 + [10 + i for i in range(1, 15)]
        prices = _prices_from_closes(closes)
        strat = MovingAverageCrossover(fast_window=2, slow_window=5)
        signals = strat.generate_signals(prices)
        assert (signals == Signal.ENTER_LONG).sum() == 1

    def test_cruzamento_para_baixo_gera_exit_long(self):
        closes = [10.0] * 5 + [10 + i for i in range(1, 10)] + [19 - i for i in range(1, 10)]
        prices = _prices_from_closes(closes)
        strat = MovingAverageCrossover(fast_window=2, slow_window=5)
        signals = strat.generate_signals(prices)
        assert (signals == Signal.ENTER_LONG).sum() >= 1
        assert (signals == Signal.EXIT_LONG).sum() >= 1
        # a primeira saída deve vir depois da primeira entrada
        entry_idx = signals[signals == Signal.ENTER_LONG].index[0]
        exit_idx = signals[signals == Signal.EXIT_LONG].index[0]
        assert exit_idx > entry_idx


class TestAntiLookAhead:
    def test_sinais_ate_o_corte_nao_mudam_com_futuro_diferente(self):
        base = [10.0] * 5 + [10 + i * 0.5 for i in range(1, 20)]
        prices_a = _prices_from_closes(base)
        prices_b = _prices_from_closes(base)

        cutoff = 15
        prices_b.iloc[cutoff:, prices_b.columns.get_indexer(["Open", "High", "Low", "Close"])] *= 10

        strat = MovingAverageCrossover(fast_window=3, slow_window=7)
        signals_a = strat.generate_signals(prices_a)
        signals_b = strat.generate_signals(prices_b)

        pd.testing.assert_series_equal(signals_a.iloc[:cutoff], signals_b.iloc[:cutoff])


class TestIntegracaoComBacktest:
    def test_pipeline_completo_estrategia_mais_motor_roda_sem_erro(self):
        # preço fica flat por tempo suficiente para as duas médias (janela 8)
        # ficarem válidas e iguais ANTES da subida começar, garantindo que o
        # cruzamento de entrada aconteça depois do warm-up, não durante ele.
        closes = [10.0] * 10 + [10 + i * 0.5 for i in range(1, 20)] + [19.5 - i * 0.5 for i in range(1, 15)]
        prices = _prices_from_closes(closes)
        strat = MovingAverageCrossover(fast_window=3, slow_window=8)
        signals = strat.generate_signals(prices)

        result = run_backtest(
            prices, signals, initial_capital=10_000, risk_pct=0.05, stop_loss_fn=FixedPctStop(0.08)
        )

        assert len(result.equity_curve) == len(prices)
        assert len(result.trades) >= 1

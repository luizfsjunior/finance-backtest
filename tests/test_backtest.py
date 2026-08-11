from __future__ import annotations

import pandas as pd
import pytest

from backtest import run_backtest
from core import Signal
from costs import Costs
from stops import AtrStop, FixedPctStop


def _prices(rows: list[dict], start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(rows))
    df = pd.DataFrame(rows, index=idx)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _flat_bar(price: float, volume: int = 1000) -> dict:
    return {"Open": price, "High": price + 0.5, "Low": price - 0.5, "Close": price, "Volume": volume}


def _signals(values: list[int], index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index, dtype=int)


class TestTimingBasico:
    def test_sinal_em_d_executa_na_abertura_de_d_mais_1(self):
        prices = _prices([_flat_bar(p) for p in [10, 10, 10, 15, 15]])
        # sinal de entrada gerado no fechamento da barra 0 -> deve executar
        # na abertura da barra 1 (preço 10), nunca na barra 0.
        sig = _signals([Signal.ENTER_LONG, 0, 0, Signal.EXIT_LONG, 0], prices.index)

        result = run_backtest(
            prices, sig, initial_capital=10_000, risk_pct=0.5, stop_loss_fn=FixedPctStop(0.5)
        )

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_date == prices.index[1]
        assert trade.entry_price == pytest.approx(10.0)
        # saída sinalizada no fechamento da barra 3 -> executa na abertura da barra 4
        assert trade.exit_date == prices.index[4]
        assert trade.exit_price == pytest.approx(15.0)

    def test_sem_sinal_nenhum_trade_e_equity_curve_completa(self):
        prices = _prices([_flat_bar(p) for p in [10, 11, 12, 13]])
        sig = _signals([0, 0, 0, 0], prices.index)

        result = run_backtest(prices, sig, 10_000, 0.1, FixedPctStop(0.05))

        assert result.trades == []
        assert len(result.equity_curve) == len(prices)
        assert (result.equity_curve == 10_000).all()


class TestAntiLookAhead:
    def test_decisoes_ate_o_corte_nao_mudam_com_dados_futuros_diferentes(self):
        base = [_flat_bar(10 + i * 0.1) for i in range(30)]
        full_prices = _prices(base)
        sig = _signals(
            [Signal.ENTER_LONG if i == 2 else (Signal.EXIT_LONG if i == 20 else 0) for i in range(30)],
            full_prices.index,
        )

        mutated = _prices(base)
        cutoff = 25
        # muda drasticamente os preços FUTUROS (depois do corte) mantendo o
        # histórico até o corte idêntico.
        mutated.iloc[cutoff:, mutated.columns.get_indexer(["Open", "High", "Low", "Close"])] *= 5

        result_full = run_backtest(full_prices, sig, 10_000, 0.1, FixedPctStop(0.05))
        result_mutated = run_backtest(mutated, sig, 10_000, 0.1, FixedPctStop(0.05))

        pd.testing.assert_series_equal(
            result_full.equity_curve.iloc[:cutoff], result_mutated.equity_curve.iloc[:cutoff]
        )

        trades_full_before_cutoff = [t for t in result_full.trades if t.exit_date < full_prices.index[cutoff]]
        trades_mutated_before_cutoff = [
            t for t in result_mutated.trades if t.exit_date < mutated.index[cutoff]
        ]
        assert trades_full_before_cutoff == trades_mutated_before_cutoff

    def test_truncar_dataset_no_corte_produz_mesmas_decisoes_ate_la(self):
        base = [_flat_bar(10 + i * 0.1) for i in range(30)]
        full_prices = _prices(base)
        sig_full = _signals(
            [Signal.ENTER_LONG if i == 2 else (Signal.EXIT_LONG if i == 20 else 0) for i in range(30)],
            full_prices.index,
        )

        cutoff = 22
        truncated_prices = full_prices.iloc[:cutoff]
        sig_truncated = sig_full.iloc[:cutoff]

        result_full = run_backtest(full_prices, sig_full, 10_000, 0.1, FixedPctStop(0.05))
        result_truncated = run_backtest(truncated_prices, sig_truncated, 10_000, 0.1, FixedPctStop(0.05))

        pd.testing.assert_series_equal(
            result_full.equity_curve.iloc[:cutoff], result_truncated.equity_curve
        )


class TestStop:
    def test_gap_abaixo_do_stop_preenche_na_abertura_nao_no_stop_teorico(self):
        # Low da barra de entrada fica acima do stop teórico (9.5) de propósito,
        # para isolar o cenário de gap na barra seguinte (sem stop disparar antes).
        rows = [_flat_bar(10), {"Open": 10.0, "High": 10.5, "Low": 9.8, "Close": 10.0, "Volume": 1000}]
        # barra 2: abre bem abaixo do stop teórico (10 * 0.95 = 9.5)
        rows.append({"Open": 8.0, "High": 8.2, "Low": 7.8, "Close": 8.0, "Volume": 1000})
        prices = _prices(rows)
        sig = _signals([Signal.ENTER_LONG, 0, 0], prices.index)

        result = run_backtest(prices, sig, 10_000, 0.5, FixedPctStop(0.05))

        assert len(result.trades) == 1
        assert result.trades[0].exit_price == pytest.approx(8.0)  # não 9.5

    def test_stop_intrabarra_sem_gap_preenche_no_preco_teorico_do_stop(self):
        rows = [_flat_bar(10), _flat_bar(10)]
        # barra 2: abre em 10, mas o Low fura o stop teórico (9.5) sem gap na abertura
        rows.append({"Open": 10.0, "High": 10.2, "Low": 9.0, "Close": 9.8, "Volume": 1000})
        prices = _prices(rows)
        sig = _signals([Signal.ENTER_LONG, 0, 0], prices.index)

        result = run_backtest(prices, sig, 10_000, 0.5, FixedPctStop(0.05))

        assert len(result.trades) == 1
        assert result.trades[0].exit_price == pytest.approx(9.5)

    def test_stop_atingido_na_propria_barra_de_entrada(self):
        rows = [_flat_bar(10)]
        # barra 1: abre em 10 (entrada) e o Low já fura o stop teórico (9.5) no mesmo candle
        rows.append({"Open": 10.0, "High": 10.1, "Low": 9.0, "Close": 9.7, "Volume": 1000})
        prices = _prices(rows)
        sig = _signals([Signal.ENTER_LONG, 0], prices.index)

        result = run_backtest(prices, sig, 10_000, 0.5, FixedPctStop(0.05))

        assert len(result.trades) == 1
        assert result.trades[0].entry_date == prices.index[1]
        assert result.trades[0].exit_date == prices.index[1]
        assert result.trades[0].exit_price == pytest.approx(9.5)

    def test_atr_stop_sem_historico_suficiente_pula_o_sinal(self):
        prices = _prices([_flat_bar(10 + i * 0.1) for i in range(5)])
        sig = _signals([Signal.ENTER_LONG, 0, 0, 0, 0], prices.index)

        result = run_backtest(prices, sig, 10_000, 0.1, AtrStop(period=14))

        assert result.trades == []
        assert len(result.skipped_signals) == 1
        assert result.skipped_signals[0][1] == "stop_indisponivel_ou_invalido"


class TestSizingECustos:
    def test_sinal_ignorado_por_caixa_insuficiente(self):
        prices = _prices([_flat_bar(100), _flat_bar(100), _flat_bar(105)])
        sig = _signals([Signal.ENTER_LONG, 0, 0], prices.index)

        # capital pequeno, stop bem distante -> risco permitiria muita
        # quantidade, mas o caixa não cobre nem 1 ação a 100.
        result = run_backtest(prices, sig, initial_capital=50, risk_pct=1.0, stop_loss_fn=FixedPctStop(0.5))

        assert result.trades == []
        assert result.skipped_signals[0][1] == "caixa_insuficiente"

    def test_sinal_ignorado_por_risco_arredondar_para_zero(self):
        prices = _prices([_flat_bar(100), _flat_bar(100), _flat_bar(105)])
        sig = _signals([Signal.ENTER_LONG, 0, 0], prices.index)

        # risco de 0,001% do capital (R$1) não cobre nem 1 ação com stop de
        # 5% sobre um papel a 100 (distância de R$5 por ação)
        result = run_backtest(
            prices, sig, initial_capital=100_000, risk_pct=0.00001, stop_loss_fn=FixedPctStop(0.05)
        )

        assert result.trades == []
        assert result.skipped_signals[0][1] == "risco_arredondou_para_zero"

    def test_pnl_bate_com_calculo_manual_incluindo_custos(self):
        prices = _prices([_flat_bar(100), _flat_bar(100), _flat_bar(110), _flat_bar(110)])
        sig = _signals([Signal.ENTER_LONG, 0, Signal.EXIT_LONG, 0], prices.index)
        costs = Costs(brokerage=5.0, slippage_bps=10.0)  # 0,10% de slippage

        result = run_backtest(
            prices, sig, initial_capital=100_000, risk_pct=1.0, stop_loss_fn=FixedPctStop(0.5), costs=costs
        )

        assert len(result.trades) == 1
        trade = result.trades[0]

        entry_fill = 100 * 1.001  # slippage de compra
        exit_fill = 110 * 0.999  # slippage de venda
        cost_rate = 0.000325 * 2
        quantity = trade.quantity

        entry_cost = entry_fill * quantity * cost_rate + 5.0
        exit_cost = exit_fill * quantity * cost_rate + 5.0
        expected_pnl = (exit_fill - entry_fill) * quantity - entry_cost - exit_cost

        assert trade.entry_price == pytest.approx(entry_fill)
        assert trade.exit_price == pytest.approx(exit_fill)
        assert trade.pnl == pytest.approx(expected_pnl)

    def test_risk_base_equity_vs_initial_capital_dao_tamanhos_diferentes_apos_lucro(self):
        # primeira operação lucrativa aumenta o equity; a segunda entrada,
        # com risk_base="equity", deve usar um capital-base maior que a
        # primeira, e portanto comprar mais ações que com "initial_capital".
        rows = [
            _flat_bar(100),
            _flat_bar(100),
            _flat_bar(150),  # saída lucrativa
            _flat_bar(150),
            _flat_bar(150),
            _flat_bar(150),
        ]
        prices = _prices(rows)
        sig = _signals(
            [Signal.ENTER_LONG, 0, Signal.EXIT_LONG, Signal.ENTER_LONG, Signal.EXIT_LONG, 0],
            prices.index,
        )

        result_equity = run_backtest(
            prices, sig, 10_000, 0.5, FixedPctStop(0.5), risk_base="equity"
        )
        result_fixed = run_backtest(
            prices, sig, 10_000, 0.5, FixedPctStop(0.5), risk_base="initial_capital"
        )

        qty_equity_second_trade = result_equity.trades[1].quantity
        qty_fixed_second_trade = result_fixed.trades[1].quantity
        assert qty_equity_second_trade > qty_fixed_second_trade


class TestSemPiramidacao:
    def test_sinal_de_entrada_repetido_enquanto_em_posicao_e_ignorado(self):
        prices = _prices([_flat_bar(p) for p in [10, 10, 10, 10, 10]])
        sig = _signals(
            [Signal.ENTER_LONG, Signal.ENTER_LONG, Signal.ENTER_LONG, 0, 0], prices.index
        )

        result = run_backtest(prices, sig, 10_000, 0.5, FixedPctStop(0.5))

        assert len(result.trades) == 0  # nunca fechou
        assert result.open_position_at_end is True


class TestValidacaoDeEntrada:
    def test_indices_diferentes_levanta_erro(self):
        prices = _prices([_flat_bar(10), _flat_bar(10)])
        sig = _signals([0, 0], pd.bdate_range("2020-01-01", periods=2))
        with pytest.raises(ValueError):
            run_backtest(prices, sig, 10_000, 0.1, FixedPctStop(0.05))

    def test_valores_invalidos_em_signals_levanta_erro(self):
        prices = _prices([_flat_bar(10), _flat_bar(10)])
        sig = _signals([0, 99], prices.index)
        with pytest.raises(ValueError):
            run_backtest(prices, sig, 10_000, 0.1, FixedPctStop(0.05))

    def test_risk_pct_fora_do_intervalo_levanta_erro(self):
        prices = _prices([_flat_bar(10)])
        sig = _signals([0], prices.index)
        with pytest.raises(ValueError):
            run_backtest(prices, sig, 10_000, 0.0, FixedPctStop(0.05))


class TestPosicaoAbertaNoFim:
    def test_posicao_aberta_no_fim_nao_gera_trade_mas_e_sinalizada(self):
        prices = _prices([_flat_bar(10), _flat_bar(10), _flat_bar(12)])
        sig = _signals([Signal.ENTER_LONG, 0, 0], prices.index)

        result = run_backtest(prices, sig, 10_000, 0.5, FixedPctStop(0.5))

        assert result.trades == []
        assert result.open_position_at_end is True
        # equity final deve refletir a posição marcada a mercado no último Close
        assert result.equity_curve.iloc[-1] > 10_000

from __future__ import annotations

import pandas as pd
import pytest

from stops import AtrStop, FixedPctStop, StopLoss


def _prices(n: int, start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    base = pd.Series(range(n), index=idx, dtype=float) + 10.0
    df = pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base + 0.5,
            "Volume": 200_000,
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


class TestProtocolo:
    def test_ambas_implementacoes_satisfazem_o_protocol(self):
        assert isinstance(FixedPctStop(0.05), StopLoss)
        assert isinstance(AtrStop(period=14), StopLoss)


class TestFixedPctStop:
    def test_stop_fica_pct_abaixo_da_entrada(self):
        stop = FixedPctStop(0.05)
        assert stop(_prices(10), 3, 100.0) == pytest.approx(95.0)

    def test_pct_fora_do_intervalo_levanta_erro(self):
        with pytest.raises(ValueError):
            FixedPctStop(0.0)
        with pytest.raises(ValueError):
            FixedPctStop(1.0)


class TestAtrStop:
    def test_sem_historico_suficiente_devolve_none(self):
        stop = AtrStop(period=14)
        assert stop(_prices(30), 10, 100.0) is None

    def test_com_historico_suficiente_devolve_stop_abaixo_da_entrada(self):
        stop = AtrStop(period=5, multiplier=2.0)
        value = stop(_prices(30), 20, 100.0)
        assert value is not None
        assert value < 100.0

    def test_period_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            AtrStop(period=0)


class TestContratoAntiLookAhead:
    def test_stop_nao_muda_quando_a_barra_de_entrada_e_as_futuras_mudam(self):
        # o stop calculado para uma entrada em `bar_index` só pode depender de
        # prices.iloc[:bar_index]; alterar a própria barra de entrada (e todas
        # as seguintes) não pode mexer no resultado.
        original = _prices(40)
        mutated = original.copy()
        bar_index = 25
        cols = mutated.columns.get_indexer(["Open", "High", "Low", "Close"])
        mutated.iloc[bar_index:, cols] *= 7

        for stop in (FixedPctStop(0.05), AtrStop(period=14)):
            assert stop(original, bar_index, 100.0) == stop(mutated, bar_index, 100.0)

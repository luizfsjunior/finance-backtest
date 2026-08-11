from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import data as data_mod
from data import (
    DEFAULT_MIN_MEDIAN_TURNOVER,
    DataFetchError,
    DataValidationError,
    IlliquidTickerError,
    get_data,
)


def _dates(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    # dias úteis, imitando calendário de pregão (sem fins de semana)
    return pd.bdate_range(start=start, periods=n)


def _valid_ohlcv(
    n: int = 40, start: str = "2024-01-02", volume: int = 200_000
) -> pd.DataFrame:
    """OHLCV sadio e LÍQUIDO por default: preço ~R$10-50 e 200 mil ações/dia
    dão giro financeiro bem acima do piso de `DEFAULT_MIN_MEDIAN_TURNOVER`,
    para os testes que não são sobre liquidez não esbarrarem no filtro."""
    idx = _dates(n, start)
    base = pd.Series(range(n), index=idx, dtype=float) + 10.0
    df = pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base + 0.5,
            "Volume": pd.Series([volume] * n, index=idx),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def _yfinance_style(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Simula o formato bruto devolvido por yf.download: colunas MultiIndex
    (Price, Ticker), mesmo para um único papel."""
    out = df.copy()
    out.columns = pd.MultiIndex.from_product([out.columns, [ticker]], names=["Price", "Ticker"])
    return out


def make_fetch_fn(df: pd.DataFrame, ticker: str, calls: list[tuple]):
    def _fetch(t: str, start: date, end: date) -> pd.DataFrame:
        calls.append((t, start, end))
        sliced = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        return _yfinance_style(sliced, ticker)

    return _fetch


class TestNormalizacaoEBusca:
    def test_retorna_colunas_e_indice_esperados(self, tmp_path: Path):
        raw = _valid_ohlcv(40)
        calls: list[tuple] = []
        fetch_fn = make_fetch_fn(raw, "PETR4.SA", calls)

        result = get_data(
            "PETR4.SA",
            date(2024, 1, 2),
            date(2024, 2, 28),
            cache_dir=tmp_path,
            fetch_fn=fetch_fn,
        )

        assert list(result.columns) == data_mod.OHLC_COLUMNS
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "Date"
        assert result.index.is_monotonic_increasing

    def test_fetch_fn_com_multiindex_estilo_yfinance_e_normalizado(self, tmp_path: Path):
        raw = _valid_ohlcv(35)
        calls: list[tuple] = []
        fetch_fn = make_fetch_fn(raw, "VALE3.SA", calls)

        result = get_data(
            "VALE3.SA",
            date(2024, 1, 2),
            date(2024, 2, 20),
            cache_dir=tmp_path,
            fetch_fn=fetch_fn,
        )

        assert not isinstance(result.columns, pd.MultiIndex)
        assert set(result.columns) == set(data_mod.OHLC_COLUMNS)

    def test_start_maior_ou_igual_a_end_levanta_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError):
            get_data(
                "PETR4.SA",
                date(2024, 2, 1),
                date(2024, 1, 1),
                cache_dir=tmp_path,
                fetch_fn=make_fetch_fn(_valid_ohlcv(40), "PETR4.SA", []),
            )

    def test_fetch_fn_retornando_vazio_levanta_data_fetch_error(self, tmp_path: Path):
        def _empty_fetch(t: str, start: date, end: date) -> pd.DataFrame:
            raise DataFetchError("sem dados")

        with pytest.raises(DataFetchError):
            get_data(
                "PETR4.SA",
                date(2024, 1, 2),
                date(2024, 2, 1),
                cache_dir=tmp_path,
                fetch_fn=_empty_fetch,
            )


class TestCache:
    def test_segunda_chamada_com_mesmo_intervalo_usa_cache_sem_rebuscar(self, tmp_path: Path):
        raw = _valid_ohlcv(40)
        calls: list[tuple] = []
        fetch_fn = make_fetch_fn(raw, "PETR4.SA", calls)

        get_data("PETR4.SA", date(2024, 1, 2), date(2024, 2, 28), cache_dir=tmp_path, fetch_fn=fetch_fn)
        get_data("PETR4.SA", date(2024, 1, 10), date(2024, 2, 20), cache_dir=tmp_path, fetch_fn=fetch_fn)

        assert len(calls) == 1

    def test_intervalo_fora_do_cache_dispara_nova_busca(self, tmp_path: Path):
        raw = _valid_ohlcv(60)
        calls: list[tuple] = []
        fetch_fn = make_fetch_fn(raw, "PETR4.SA", calls)

        get_data(
            "PETR4.SA", date(2024, 1, 2), date(2024, 2, 10), cache_dir=tmp_path, fetch_fn=fetch_fn, min_rows=5
        )
        get_data(
            "PETR4.SA", date(2024, 1, 2), date(2024, 3, 20), cache_dir=tmp_path, fetch_fn=fetch_fn, min_rows=5
        )

        assert len(calls) == 2

    def test_force_refresh_ignora_cache_existente(self, tmp_path: Path):
        raw = _valid_ohlcv(40)
        calls: list[tuple] = []
        fetch_fn = make_fetch_fn(raw, "PETR4.SA", calls)

        get_data("PETR4.SA", date(2024, 1, 2), date(2024, 2, 28), cache_dir=tmp_path, fetch_fn=fetch_fn)
        get_data(
            "PETR4.SA",
            date(2024, 1, 2),
            date(2024, 2, 28),
            cache_dir=tmp_path,
            fetch_fn=fetch_fn,
            force_refresh=True,
        )

        assert len(calls) == 2

    def test_cache_persiste_em_parquet_no_disco(self, tmp_path: Path):
        raw = _valid_ohlcv(40)
        fetch_fn = make_fetch_fn(raw, "PETR4.SA", [])

        get_data("PETR4.SA", date(2024, 1, 2), date(2024, 2, 28), cache_dir=tmp_path, fetch_fn=fetch_fn)

        assert (tmp_path / "PETR4.SA.parquet").exists()

    def test_cache_corrompido_com_duplicatas_e_detectado_na_leitura(self, tmp_path: Path):
        raw = _valid_ohlcv(40)
        corrupted = pd.concat([raw.iloc[[0]], raw])  # duplica a primeira data
        corrupted = corrupted.sort_index()
        path = tmp_path / "PETR4.SA.parquet"
        corrupted.to_parquet(path)

        with pytest.raises(DataValidationError, match="duplicad"):
            get_data(
                "PETR4.SA",
                date(2024, 1, 2),
                date(2024, 2, 20),
                cache_dir=tmp_path,
                fetch_fn=make_fetch_fn(raw, "PETR4.SA", []),
                min_rows=5,
            )


class TestValidacao:
    def _get(self, df: pd.DataFrame, tmp_path: Path, ticker: str = "PETR4.SA", min_rows: int = 5):
        fetch_fn = make_fetch_fn(df, ticker, [])
        return get_data(
            ticker,
            df.index.min().date(),
            df.index.max().date(),
            cache_dir=tmp_path,
            fetch_fn=fetch_fn,
            min_rows=min_rows,
        )

    def test_preco_zero_levanta_erro(self, tmp_path: Path):
        df = _valid_ohlcv(10)
        df.iloc[3, df.columns.get_loc("Close")] = 0.0
        with pytest.raises(DataValidationError, match="zerado"):
            self._get(df, tmp_path)

    def test_preco_negativo_levanta_erro(self, tmp_path: Path):
        df = _valid_ohlcv(10)
        df.iloc[2, df.columns.get_loc("Low")] = -1.0
        with pytest.raises(DataValidationError, match="zerado"):
            self._get(df, tmp_path)

    def test_nan_em_preco_levanta_erro(self, tmp_path: Path):
        df = _valid_ohlcv(10)
        df.iloc[5, df.columns.get_loc("Open")] = float("nan")
        with pytest.raises(DataValidationError, match="NaN"):
            self._get(df, tmp_path)

    def test_high_menor_que_low_levanta_erro(self, tmp_path: Path):
        df = _valid_ohlcv(10)
        df.iloc[4, df.columns.get_loc("High")] = df.iloc[4]["Low"] - 5.0
        with pytest.raises(DataValidationError, match="High menor que Low"):
            self._get(df, tmp_path)

    def test_close_fora_do_intervalo_high_low_levanta_erro(self, tmp_path: Path):
        df = _valid_ohlcv(10)
        df.iloc[6, df.columns.get_loc("Close")] = df.iloc[6]["High"] + 10.0
        with pytest.raises(DataValidationError, match="fora do intervalo"):
            self._get(df, tmp_path)

    def test_menos_linhas_que_min_rows_levanta_erro(self, tmp_path: Path):
        df = _valid_ohlcv(4)
        with pytest.raises(DataValidationError, match="mínimo exigido"):
            self._get(df, tmp_path, min_rows=10)

    def test_dataframe_valido_nao_levanta_erro(self, tmp_path: Path):
        df = _valid_ohlcv(10)
        result = self._get(df, tmp_path)
        assert len(result) == 10


class TestLiquidez:
    def _get(self, df: pd.DataFrame, tmp_path: Path, ticker: str = "MICO3.SA", **kwargs):
        fetch_fn = make_fetch_fn(df, ticker, [])
        return get_data(
            ticker,
            df.index.min().date(),
            df.index.max().date(),
            cache_dir=tmp_path,
            fetch_fn=fetch_fn,
            min_rows=5,
            **kwargs,
        )

    def test_papel_iliquido_e_rejeitado(self, tmp_path: Path):
        # 100 ações/dia a ~R$10 => giro de ~R$1.000/dia, muito abaixo do piso
        df = _valid_ohlcv(30, volume=100)
        with pytest.raises(IlliquidTickerError, match="ilíquido"):
            self._get(df, tmp_path)

    def test_papel_liquido_passa(self, tmp_path: Path):
        df = _valid_ohlcv(30, volume=200_000)
        result = self._get(df, tmp_path)
        assert len(result) == 30

    def test_min_median_turnover_zero_desliga_o_filtro(self, tmp_path: Path):
        df = _valid_ohlcv(30, volume=100)
        result = self._get(df, tmp_path, min_median_turnover=0.0)
        assert len(result) == 30

    def test_usa_mediana_nao_media_um_pregao_atipico_nao_salva_o_papel(self, tmp_path: Path):
        # papel ilíquido com UM pregão de giro gigantesco (leilão/evento):
        # a média passaria folgada no piso, a mediana não — e é a mediana
        # que decide, justamente para esse caso não promover um papel morto.
        df = _valid_ohlcv(30, volume=100)
        df.iloc[15, df.columns.get_loc("Volume")] = 500_000_000
        mean_turnover = (df["Close"] * df["Volume"]).mean()
        assert mean_turnover > DEFAULT_MIN_MEDIAN_TURNOVER  # a média enganaria

        with pytest.raises(IlliquidTickerError):
            self._get(df, tmp_path)

    def test_liquidez_e_avaliada_na_fatia_pedida_nao_no_cache_inteiro(self, tmp_path: Path):
        # cache cobre 60 pregões, mas só a segunda metade é líquida; pedir
        # apenas a primeira metade deve reprovar, mesmo com o dataset inteiro
        # em cache passando na média.
        df = _valid_ohlcv(60, volume=100)
        df.iloc[30:, df.columns.get_loc("Volume")] = 500_000
        ticker = "MICO3.SA"
        fetch_fn = make_fetch_fn(df, ticker, [])

        with pytest.raises(IlliquidTickerError):
            get_data(
                ticker,
                df.index[0].date(),
                df.index[25].date(),
                cache_dir=tmp_path,
                fetch_fn=fetch_fn,
                min_rows=5,
            )

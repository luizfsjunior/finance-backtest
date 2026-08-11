"""Busca, cache local e validação de dados históricos de ações (B3).

A fonte da verdade do projeto. Existe separado por dois motivos práticos: o
backtest roda centenas de vezes e não pode bater na API toda vez (lento e
sujeito a rate limit), e todo teste precisa usar exatamente o mesmo dataset —
backtest que muda de resultado porque a fonte mudou é impossível de depurar.

Regras não negociáveis deste módulo:
- O preço usado como "Close" é sempre o fechamento AJUSTADO (dividendos/splits).
- Papéis ilíquidos são REJEITADOS, não apenas sinalizados (ver
  `min_median_turnover`). É em candle sem liquidez que o backtest fabrica
  retorno que não existiria na vida real: a ordem simulada preenche inteira,
  no preço teórico, contra um book que não tinha contraparte.
- Dados inválidos (preços zerados/negativos, NaN, OHLC inconsistente) fazem o
  módulo levantar exceção — nunca são corrigidos silenciosamente. Um backtest
  rodando sobre dado ruim sem avisar é pior do que um backtest que não roda.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf

OHLC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Tolerância para inconsistências de arredondamento entre Open/Close e High/Low
# introduzidas pelo processo de ajuste por dividendos/splits do provedor de dados.
_OHLC_TOLERANCE = 1e-2

# Volume financeiro diário mediano mínimo (R$) para o papel ser considerado
# negociável. Mediana, não média: um único pregão atípico de altíssimo giro
# (leilão, evento corporativo) não deve promover um papel ilíquido a líquido.
# R$ 1 milhão/dia é o piso abaixo do qual uma ordem de porte modesto já
# move o preço o bastante para invalidar a premissa de fill no preço teórico.
DEFAULT_MIN_MEDIAN_TURNOVER = 1_000_000.0


class DataError(Exception):
    """Erro base para problemas de dados."""


class DataFetchError(DataError):
    """Falha ao buscar dados na fonte externa."""


class DataValidationError(DataError):
    """Dados obtidos (fonte ou cache) não passaram na validação de sanidade."""


class IlliquidTickerError(DataValidationError):
    """Papel não atinge o volume financeiro mínimo exigido no período."""


FetchFn = Callable[[str, date, date], pd.DataFrame]


def get_data(
    ticker: str,
    start: date,
    end: date,
    cache_dir: Path = Path("data_cache"),
    min_rows: int = 30,
    force_refresh: bool = False,
    fetch_fn: FetchFn | None = None,
    min_median_turnover: float = DEFAULT_MIN_MEDIAN_TURNOVER,
) -> pd.DataFrame:
    """Retorna OHLCV diário de `ticker` entre `start` e `end` (inclusive).

    - `Close` é sempre o fechamento ajustado.
    - Índice é DatetimeIndex diário, ordenado, sem duplicatas.
    - Usa cache em parquet quando o intervalo pedido já está coberto;
      caso contrário busca na fonte e sobrescreve o cache desse ticker.
    - Levanta `DataValidationError` se os dados não passarem na checagem de
      sanidade (nunca corrige silenciosamente).
    - Levanta `IlliquidTickerError` se o volume financeiro diário mediano do
      período ficar abaixo de `min_median_turnover`. Passe 0 para desligar a
      checagem (útil em testes com dados sintéticos).

    `fetch_fn` existe para permitir injeção de uma função de busca alternativa
    (principalmente em testes, para não depender de rede).
    """
    if start >= end:
        raise ValueError(f"start ({start}) deve ser anterior a end ({end})")

    fetch = fetch_fn or _fetch_from_yfinance
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, cache_dir)

    cached = None if force_refresh else _read_cache(path)
    if cached is not None and _covers_range(cached, start, end):
        df = _slice_range(cached, start, end)
    else:
        df = fetch(ticker, start, end)
        df = _normalize(df, ticker)
        # O cache guarda tudo que a fonte devolveu, inclusive um período que
        # sozinho reprovaria na liquidez; a decisão de rejeitar é tomada sobre
        # a fatia efetivamente usada, logo abaixo.
        _validate(df, ticker, min_rows, min_median_turnover=0.0)
        _write_cache(df, path)
        df = _slice_range(df, start, end)

    _validate(df, ticker, min_rows, min_median_turnover)
    return df


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"{ticker}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


def _covers_range(df: pd.DataFrame, start: date, end: date) -> bool:
    if df.empty:
        return False
    return df.index.min().date() <= start and df.index.max().date() >= end


def _slice_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def _fetch_from_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    # yfinance trata `end` como exclusivo; +1 dia para incluir a data pedida.
    raw = yf.download(
        ticker,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataFetchError(f"Nenhum dado retornado para {ticker} entre {start} e {end}")
    return raw


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=-1)
    df = df[OHLC_COLUMNS].copy()
    df.columns.name = None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def _validate(
    df: pd.DataFrame,
    ticker: str,
    min_rows: int,
    min_median_turnover: float = DEFAULT_MIN_MEDIAN_TURNOVER,
) -> None:
    if df.empty:
        raise DataValidationError(f"{ticker}: dataframe vazio")

    if len(df) < min_rows:
        raise DataValidationError(
            f"{ticker}: apenas {len(df)} linhas, mínimo exigido é {min_rows}"
        )

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].tolist()
        raise DataValidationError(f"{ticker}: datas duplicadas no índice: {dupes}")

    if not df.index.is_monotonic_increasing:
        raise DataValidationError(f"{ticker}: índice de datas não está ordenado")

    price_cols = ["Open", "High", "Low", "Close"]
    if df[price_cols].isna().any().any():
        raise DataValidationError(f"{ticker}: valores NaN em colunas de preço")

    if df["Volume"].isna().any():
        raise DataValidationError(f"{ticker}: valores NaN em Volume")

    if (df[price_cols] <= 0).any().any():
        raise DataValidationError(f"{ticker}: preço zerado ou negativo encontrado")

    if (df["Volume"] < 0).any():
        raise DataValidationError(f"{ticker}: volume negativo encontrado")

    if (df["High"] + _OHLC_TOLERANCE < df["Low"]).any():
        raise DataValidationError(f"{ticker}: High menor que Low em pelo menos um candle")

    out_of_range_open = (df["Open"] < df["Low"] - _OHLC_TOLERANCE) | (
        df["Open"] > df["High"] + _OHLC_TOLERANCE
    )
    out_of_range_close = (df["Close"] < df["Low"] - _OHLC_TOLERANCE) | (
        df["Close"] > df["High"] + _OHLC_TOLERANCE
    )
    if out_of_range_open.any() or out_of_range_close.any():
        raise DataValidationError(
            f"{ticker}: Open ou Close fora do intervalo [Low, High] em pelo menos um candle"
        )

    # Liquidez por último: só faz sentido perguntar se o papel é negociável
    # depois de saber que os preços e volumes são sãos.
    if min_median_turnover > 0:
        median_turnover = float((df["Close"] * df["Volume"]).median())
        if median_turnover < min_median_turnover:
            raise IlliquidTickerError(
                f"{ticker}: volume financeiro diário mediano de R$ {median_turnover:,.2f} "
                f"abaixo do mínimo de R$ {min_median_turnover:,.2f} — papel ilíquido demais "
                f"para o backtest assumir fill no preço teórico"
            )

"""Carga multi-ticker alinhada e máscara de elegibilidade por data.

Complementa `data.py` (que serve um único ticker por vez) montando o "cubo"
de preços que o motor de portfolio precisa: mesmo eixo de datas, N tickers
lado a lado, e uma máscara booleana [data × ticker] dizendo quem pode ser
rankeado/comprado em cada dia.

Regras não negociáveis:
- Nunca preencher NaN em preços. Se o ticker não teve pregão numa data, fica
  NaN — a máscara de elegibilidade é quem carrega essa informação, não o
  preço. Ffill em preço aqui inventaria candles e enganaria o motor sobre a
  possibilidade real de fill.
- Filtro de liquidez é por janela móvel, não estático. Um papel pode virar
  líquido no meio do período (ex.: SUZB3 pós-fusão) e a máscara precisa
  refletir isso — descartar o ticker inteiro porque a mediana do período todo
  ficou baixa desperdiça o histórico bom.
- `get_portfolio_data` sempre reusa `data.get_data` para cada ticker
  individualmente. Isso garante que as mesmas checagens de sanidade (OHLC
  consistente, sem NaN em preço, sem preço zerado/negativo) valem — só depois
  os cubos são montados. Nada de baixar N tickers de uma vez e pular
  validação: o custo é uma chamada por ticker, o benefício é rigor uniforme.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data import (
    DEFAULT_MIN_MEDIAN_TURNOVER,
    DataError,
    FetchFn,
    get_data,
)

# Janela para a mediana de turnover que decide elegibilidade dia a dia. 63
# dias úteis ≈ 3 meses corridos: curto o suficiente para captar a transição
# de "ilíquido → líquido" (SUZB3 pós-fusão em 2018, por exemplo) em poucos
# meses, longo o suficiente para não deixar um pregão atípico (leilão, evento
# corporativo) sozinho promover ou demitir um papel.
LIQUIDITY_WINDOW_DAYS = 63


def get_portfolio_data(
    tickers: list[str],
    start: date,
    end: date,
    cache_dir: Path = Path("data_cache"),
    fetch_fn: FetchFn | None = None,
    min_median_turnover: float = DEFAULT_MIN_MEDIAN_TURNOVER,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """Carrega OHLCV dos `tickers` no intervalo e devolve o cubo alinhado + máscara.

    Retorna:
      - `prices`: DataFrame com MultiIndex de colunas (ticker, field), onde
        field ∈ {Open, High, Low, Close, Volume}. Index = união ordenada das
        datas de pregão de todos os tickers. NaN nas células em que o ticker
        não tinha pregão.
      - `eligibility`: DataFrame [data × ticker] → bool. True se, naquela
        data, o ticker (a) tem preço não-NaN e (b) tem mediana de turnover
        dos últimos `LIQUIDITY_WINDOW_DAYS` acima de `min_median_turnover`.
      - `failures`: lista de (ticker, motivo) para tickers que falharam
        completamente na carga (não entram no cubo). Um ticker que falha aqui
        não trava o batch — o motor lida com um universo menor.

    A checagem de liquidez estática (`data.get_data(min_median_turnover=...)`)
    é DESLIGADA aqui — quem decide elegibilidade é a máscara rolling,
    justamente para captar transição de regime. Se o papel foi ilíquido a
    vida inteira do período, isso vai aparecer: a coluna dele fica False
    inteira na máscara.
    """
    if not tickers:
        raise ValueError("tickers vazio")
    if start >= end:
        raise ValueError(f"start ({start}) deve ser anterior a end ({end})")

    per_ticker: dict[str, pd.DataFrame] = {}
    failures: list[tuple[str, str]] = []

    for ticker in tickers:
        try:
            df = get_data(
                ticker,
                start,
                end,
                cache_dir=cache_dir,
                fetch_fn=fetch_fn,
                # Liquidez estática desligada; a máscara rolling decide dia a dia.
                min_median_turnover=0.0,
            )
        except DataError as exc:
            failures.append((ticker, f"{type(exc).__name__}: {exc}"))
            continue
        per_ticker[ticker] = df

    if not per_ticker:
        raise ValueError(
            f"nenhum ticker carregado com sucesso — todas as {len(tickers)} tentativas falharam"
        )

    # `pd.concat(axis=1, keys=...)` monta MultiIndex de colunas (ticker, field)
    # e alinha o eixo de datas pela união dos índices — NaN onde faltar.
    prices = pd.concat(per_ticker.values(), axis=1, keys=per_ticker.keys())
    prices = prices.sort_index()

    eligibility = _build_eligibility(prices, min_median_turnover)
    return prices, eligibility, failures


def _build_eligibility(prices: pd.DataFrame, min_median_turnover: float) -> pd.DataFrame:
    """Máscara [data × ticker] a partir do cubo de preços.

    Duas condições combinadas por AND:
      1. `Close` da célula não é NaN (o ticker teve pregão)
      2. Mediana móvel de `Close * Volume` dos últimos LIQUIDITY_WINDOW_DAYS
         pregões (contando só os pregões do próprio ticker, não o calendário
         corrido) é >= `min_median_turnover`.

    O ponto (2) é rolling sobre o histórico do próprio ticker, então a
    máscara é causal — cada célula depende só de dados até aquela data,
    nunca do futuro. E como usa `.rolling().median()` com min_periods igual
    à janela, os primeiros ~63 pregões do ticker são False mesmo se o preço
    existir: sem histórico suficiente para atestar liquidez, não elegível.
    """
    tickers = prices.columns.get_level_values(0).unique()

    close = prices.xs("Close", axis=1, level=1)
    volume = prices.xs("Volume", axis=1, level=1)

    has_price = close.notna()

    if min_median_turnover <= 0:
        return has_price

    # NaN em Close ou Volume → turnover NaN → rolling().median() NaN.
    # Uma NaN dentro da janela também zera a mediana daquela linha pra
    # NaN por default; queremos que a janela ignore os NaNs mas ainda
    # exija LIQUIDITY_WINDOW_DAYS observações válidas no histórico do
    # próprio ticker antes de considerar liquidez estabelecida.
    turnover = close * volume
    rolling_median = turnover.rolling(
        window=LIQUIDITY_WINDOW_DAYS,
        min_periods=LIQUIDITY_WINDOW_DAYS,
    ).median()

    liquid_enough = rolling_median >= min_median_turnover
    eligibility = has_price & liquid_enough.reindex_like(has_price).fillna(False)
    # Garante colunas na mesma ordem do cubo de preços
    return eligibility[list(tickers)]

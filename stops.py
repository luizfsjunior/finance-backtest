"""Critérios de stop loss: quando você admite estar errado.

Separado da estratégia porque "quando entro" e "quanto tolero de ruído" são
perguntas independentes — você quer rodar o mesmo setup com stops diferentes
e isolar o efeito de cada um. Separado do motor porque escrever um stop novo
não deveria exigir abrir o arquivo mais delicado do projeto.

Um `StopLoss` recebe (prices, bar_index, entry_price) e devolve o preço de
stop para uma entrada que vai executar na ABERTURA de `bar_index`.

Contrato crítico: a implementação só pode olhar para `prices.iloc[:bar_index]`
(barras estritamente anteriores). A barra `bar_index` ainda não "aconteceu" no
momento em que a decisão de risco é tomada — ela só existe no dataset porque o
backtest inteiro já é histórico. Usar High/Low/Close da própria barra de
entrada para calcular o stop seria look-ahead bias na sizing, não só na
execução.

Retornar None sinaliza "sem dado suficiente para calcular o stop" (ex.: ATR
pedindo mais histórico do que existe ainda) — o motor pula o sinal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StopLoss(Protocol):
    def __call__(
        self, prices: pd.DataFrame, bar_index: int, entry_price: float
    ) -> float | None:
        """Preço de stop para uma entrada na abertura de `bar_index`, ou None
        se não houver histórico suficiente para calculá-lo. Ver o contrato
        anti-look-ahead no docstring do módulo."""
        ...


@dataclass(frozen=True)
class NoStop:
    """Stop "de fachada" para estratégias em que o stop briga com a tese.

    A reversão à média diz "quanto mais cai depois da entrada, mais barato e
    mais atraente"; um stop apertado corta o trade exatamente quando a tese
    ficaria mais válida. Para medir a tese em estado puro, esta classe põe o
    stop tão longe (default 50% abaixo da entrada) que ele quase nunca
    dispara nos ativos líquidos da B3 — mas ainda existe como piso de
    catástrofe absoluta e, mais importante, dá ao motor uma distância finita
    para dimensionar a posição por risco.

    Não é "sem stop": um stop de verdade em 0 zeraria a distância de risco de
    forma degenerada. `distance_pct` é a fração abaixo da entrada onde o stop
    fica — quanto maior, mais capital vai para cada trade (mais afrouxa o
    sizing por risco).
    """

    distance_pct: float = 0.50

    def __post_init__(self) -> None:
        if not 0 < self.distance_pct < 1:
            raise ValueError(f"distance_pct deve estar entre 0 e 1, recebeu {self.distance_pct}")

    def __call__(
        self, prices: pd.DataFrame, bar_index: int, entry_price: float
    ) -> float | None:
        return entry_price * (1 - self.distance_pct)


@dataclass(frozen=True)
class FixedPctStop:
    """Stop a `pct` abaixo do preço de entrada (ex.: pct=0.05 -> -5%)."""

    pct: float

    def __post_init__(self) -> None:
        if not 0 < self.pct < 1:
            raise ValueError(f"pct deve estar entre 0 e 1, recebeu {self.pct}")

    def __call__(
        self, prices: pd.DataFrame, bar_index: int, entry_price: float
    ) -> float | None:
        return entry_price * (1 - self.pct)


@dataclass(frozen=True)
class AtrStop:
    """Stop a `multiplier` ATRs abaixo da entrada.

    ATR aqui é suavização de Wilder (via `ewm(alpha=1/period)`), calculado
    usando SOMENTE barras anteriores à entrada (ver contrato no docstring do
    módulo). Nas primeiras `period` barras do dataset não há histórico
    suficiente para uma ATR confiável — devolve None e o motor pula o sinal,
    em vez de calcular uma ATR curta e silenciosamente enviesada.

    Nota de performance: recalcula a série de ATR inteira a cada chamada
    (O(n) por entrada). Para o tamanho de dataset deste projeto (anos de
    pregão diário) é irrelevante; se o motor crescer para milhares de
    entradas por backtest, vale pré-computar a série uma vez fora do loop.
    """

    period: int = 14
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError(f"period deve ser >= 1, recebeu {self.period}")

    def __call__(
        self, prices: pd.DataFrame, bar_index: int, entry_price: float
    ) -> float | None:
        if bar_index < self.period:
            return None
        window = prices.iloc[:bar_index]
        high, low, close = window["High"], window["Low"], window["Close"]
        prev_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr_value = true_range.ewm(alpha=1 / self.period, adjust=False).mean().iloc[-1]
        if pd.isna(atr_value):
            return None
        return entry_price - self.multiplier * atr_value

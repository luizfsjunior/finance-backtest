"""Protocol de estratégia e a estratégia de validação (cruzamento de médias).

Uma estratégia é a hipótese: recebe o DataFrame de preços e devolve uma Series
de `core.Signal` no MESMO índice, já como eventos de borda (o instante do
cruzamento) — não estado contínuo. Só isso.

Não sabe quanto comprar, não sabe onde fica o stop, não sabe quanto há em
caixa — nem importa de `backtest.py` ou `stops.py`. Essa ignorância é
proposital: é o que permite rodar a mesma estratégia com regras de risco
diferentes e isolar o efeito de cada uma. Se estratégia e execução ficam
misturadas, você nunca sabe se o resultado mudou por causa do setup ou do
sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from core import Signal


@runtime_checkable
class Strategy(Protocol):
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Recebe OHLCV (ver `data.get_data`) e devolve uma Series de
        `Signal`, mesmo índice de `prices`, ordenado, sem olhar para barras
        futuras em relação a cada posição do índice."""
        ...


@dataclass
class MovingAverageCrossover:
    """Entra no cruzamento da média rápida para cima da lenta, sai no
    cruzamento para baixo (o stop é decidido em `stops.py` e aplicado por
    `backtest.py`, não aqui — a estratégia só sabe de médias móveis)."""

    fast_window: int
    slow_window: int

    def __post_init__(self) -> None:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("fast_window e slow_window devem ser positivos")
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) deve ser menor que "
                f"slow_window ({self.slow_window})"
            )

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if "Close" not in prices.columns:
            raise ValueError("prices precisa ter a coluna 'Close'")

        close = prices["Close"]
        # rolling().mean() é causal por natureza (só olha pra trás), então o
        # cálculo em si não vaza futuro. O cuidado real é no que vem depois.
        fast_ma = close.rolling(self.fast_window).mean()
        slow_ma = close.rolling(self.slow_window).mean()

        # Enquanto a média lenta não tem `slow_window` pontos, fast_ma/slow_ma
        # são NaN. Comparar NaN > NaN direto com `.gt()` do pandas devolve
        # False (não NaN) — se eu não isolar esse trecho, o período de warm-up
        # inteiro entraria como "fast <= slow" e o instante em que a média
        # lenta finalmente fica válida apareceria como um cruzamento real,
        # mesmo sem ter havido cruzamento nenhum. Por isso mantenho NaN
        # explícito em `above` enquanto qualquer uma das médias for NaN.
        valid = fast_ma.notna() & slow_ma.notna()
        above = pd.Series(np.nan, index=close.index)
        above[valid] = (fast_ma[valid] > slow_ma[valid]).astype(float)

        # diff() de NaN para um número é NaN (não 1 nem -1), então a barra em
        # que o warm-up termina nunca é lida como cruzamento — só barras onde
        # o estado anterior E o atual são válidos podem gerar sinal.
        transition = above.diff()

        signals = pd.Series(int(Signal.HOLD), index=close.index, dtype=int)
        signals[transition == 1] = int(Signal.ENTER_LONG)
        signals[transition == -1] = int(Signal.EXIT_LONG)
        return signals


@dataclass
class BollingerReversion:
    """Reversão à média por bandas de Bollinger (long-only, conservadora).

    Tese: o preço tem gravidade em torno de uma média móvel; quando cai muito
    abaixo dela (medido em desvios-padrão da própria janela), a aposta é que
    volta. Contraponto filosófico do cruzamento de médias: aquela compra força
    que continua, esta compra fraqueza que reverte.

    - Entrada (ENTER_LONG): o fechamento cruzou de cima para baixo da banda
      inferior (`média − k·desvio`) — borda, não estado. Se o preço fica N dias
      abaixo da banda, sinaliza só no primeiro.
    - Saída (EXIT_LONG): o fechamento voltou e cruzou a média central de baixo
      para cima. Versão conservadora: fecha o trade quando a "gravidade"
      cumpriu o mínimo, sem esperar reversão até a banda superior (que prende
      posição perdedora por mais tempo em tendências de queda).

    A estratégia não sabe se há posição aberta — o motor ignora ENTER_LONG
    quando já está comprado e EXIT_LONG quando está zerado, então emitir por
    borda evita reentradas espúrias enquanto a condição persiste.
    """

    window: int = 20
    k: float = 2.0

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window deve ser >= 2 (desvio-padrão precisa de >=2 pontos), recebeu {self.window}")
        if self.k <= 0:
            raise ValueError(f"k deve ser positivo, recebeu {self.k}")

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if "Close" not in prices.columns:
            raise ValueError("prices precisa ter a coluna 'Close'")

        close = prices["Close"]
        # rolling().mean() e .std() são causais (só olham para trás na janela),
        # então o cálculo em si não vaza futuro. O cuidado real é comparar a
        # banda de D com o fechamento de D e agir em D+1 — o motor já garante
        # o D+1 (ver backtest.py).
        mean = close.rolling(self.window).mean()
        std = close.rolling(self.window).std(ddof=0)
        lower = mean - self.k * std

        # Bandas são NaN nas primeiras `window - 1` barras. Comparar NaN com
        # número devolve False no pandas: sem isolar, o warm-up inteiro
        # contaria como "acima da banda" e a primeira barra válida em que o
        # preço estivesse abaixo apareceria como cruzamento falso.
        valid = mean.notna() & lower.notna()

        below_lower = pd.Series(float("nan"), index=close.index)
        below_lower[valid] = (close[valid] < lower[valid]).astype(float)
        # diff() de NaN para número é NaN, então a barra em que o warm-up
        # termina nunca é lida como transição.
        cross_below_lower = below_lower.diff()  # 1 quando cruzou para dentro

        above_mean = pd.Series(float("nan"), index=close.index)
        above_mean[valid] = (close[valid] > mean[valid]).astype(float)
        cross_above_mean = above_mean.diff()  # 1 quando cruzou a média para cima

        signals = pd.Series(int(Signal.HOLD), index=close.index, dtype=int)
        signals[cross_below_lower == 1] = int(Signal.ENTER_LONG)
        signals[cross_above_mean == 1] = int(Signal.EXIT_LONG)
        return signals


@dataclass
class TimeSeriesMomentum:
    """Momentum de médio prazo, versão time-series (single-asset).

    Tese: ativos com retorno positivo nos últimos ~12 meses (menos o mês mais
    recente) tendem a continuar subindo nos próximos meses. Anomalia
    estruturalmente documentada — Jegadeesh & Titman (1993) cross-sectional,
    Moskowitz-Ooi-Pedersen (2012) na versão time-series usada aqui.

    Por que pular o último mês: no curtíssimo prazo (dias/semanas) o efeito
    dominante é o oposto — reversão. Sem pular, o ranking de "momentum" fica
    contaminado por reversão de curto prazo e os dois efeitos se anulam.
    Estrutura padrão: reversão curta (< 1 mês) → momentum médio (2-12 meses)
    → reversão longa (3-5 anos).

    Não é sinal de timing — é filtro de regime. Diz "vale a pena estar
    comprado neste ativo agora?", não "compre exatamente aqui". Consequência:
    trades raros e longos (poucas transições, cada uma durando meses), ao
    contrário de MAC/Bollinger que geram dezenas de trades curtos.

    - Entrada (ENTER_LONG): o retorno lookback-skip cruza de ≤ 0 para > 0.
    - Saída (EXIT_LONG): cruza de > 0 para ≤ 0.
    - Warm-up: as primeiras `lookback` barras são HOLD (não há histórico
      suficiente para o retorno de referência).

    Parâmetros em dias úteis, não em meses corridos: 252 ≈ 12 meses, 21 ≈ 1
    mês. Mantém a semântica da literatura sem depender de calendário.
    """

    lookback: int = 252
    skip: int = 21
    threshold: float = 0.0
    monthly_rebalance: bool = False

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError(f"lookback deve ser >= 2, recebeu {self.lookback}")
        if self.skip < 0:
            raise ValueError(f"skip deve ser >= 0, recebeu {self.skip}")
        if self.skip >= self.lookback:
            raise ValueError(
                f"skip ({self.skip}) deve ser menor que lookback ({self.lookback}), "
                f"senão a janela de momentum vira negativa"
            )
        if self.threshold < 0:
            raise ValueError(f"threshold deve ser >= 0, recebeu {self.threshold}")

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if "Close" not in prices.columns:
            raise ValueError("prices precisa ter a coluna 'Close'")

        close = prices["Close"]
        # Retorno de D-lookback até D-skip: `preço agora pulando o mês recente`
        # dividido por `preço lookback dias atrás` menos 1. shift(skip) desloca
        # a série para frente por skip barras — no índice D, close.shift(skip)
        # traz o valor de D-skip. É causal por construção (só olha para trás).
        numerator = close.shift(self.skip)
        denominator = close.shift(self.lookback)
        momentum = numerator / denominator - 1.0

        # Estado desejado por barra: 1 (long) acima de +threshold, 0 (flat)
        # abaixo de -threshold, e "mantém o anterior" entre os dois. Isso é
        # uma máquina de três zonas, não um comparador binário — a banda
        # morta existe justamente para o estado NÃO oscilar quando o momentum
        # roça o zero. Nas primeiras `lookback` barras (momentum NaN), estado
        # também é NaN — nada de sinal antes do warm-up terminar.
        state = pd.Series(float("nan"), index=close.index)
        current = float("nan")
        thr = self.threshold
        for i, m in enumerate(momentum.values):
            if pd.notna(m):
                if m > thr:
                    current = 1.0
                elif m < -thr:
                    current = 0.0
                # else: mantém `current` (banda morta ou ainda NaN antes do 1º toque)
            state.iloc[i] = current

        # Rebalanceamento mensal: só aja no primeiro pregão de cada (ano, mês).
        # Nos outros dias, o estado congela no valor do último dia de decisão.
        # Faz o "grosseamento" temporal sem tocar no motor — o Protocol de
        # Strategy continua devolvendo Series diária de sinais; o que muda é
        # que a maioria dos dias é forçosamente HOLD.
        if self.monthly_rebalance:
            idx = close.index
            month_key = pd.Series(
                list(zip(idx.year, idx.month)), index=idx
            )
            is_first_of_month = month_key != month_key.shift(1)
            # Congela o estado: no dia da decisão usa o valor calculado;
            # nos demais dias, propaga o último valor de decisão.
            frozen = state.where(is_first_of_month)
            state = frozen.ffill()

        transition = state.diff()

        signals = pd.Series(int(Signal.HOLD), index=close.index, dtype=int)
        signals[transition == 1] = int(Signal.ENTER_LONG)
        signals[transition == -1] = int(Signal.EXIT_LONG)
        return signals

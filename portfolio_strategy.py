"""Protocol de estratégia de portfolio e primeira implementação (cross-sectional momentum).

Diferença central em relação ao `strategy.py` single-asset: aqui a estratégia
devolve um SCORE CONTÍNUO por (data, ticker), não sinais discretos por barra.
O motor (`portfolio_backtest.py`) é quem decide, a cada dia de
rebalanceamento, os top-K por score da linha e distribui capital.

Essa separação preserva a mesma filosofia do projeto: estratégia é hipótese
("qual é a força da tese para cada ativo hoje?"), motor é execução ("quem
levo, com que peso, quando ajusto"). Estratégia continua ignorando caixa,
custos, cadência de rebalance — só sabe do universo e dos preços.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PortfolioStrategy(Protocol):
    def rank(
        self,
        prices: pd.DataFrame,
        eligibility: pd.DataFrame,
    ) -> pd.DataFrame:
        """Recebe cubo de preços (MultiIndex de colunas ticker×field) + máscara
        de elegibilidade [data × ticker] e devolve um DataFrame de scores
        [data × ticker].

        Convenções:
        - Score maior = melhor. Motor pega os top-K por linha.
        - NaN = ativo não rankeável naquela data (não elegível, sem histórico
          suficiente para o cálculo, ou explicitamente descartado). NaN
          NUNCA entra no top-K.
        - Não precisa estar normalizado. O motor só usa a ordem dos scores,
          não seus valores absolutos.
        - Causal: o score da linha D só depende de preços até D. É
          responsabilidade da estratégia respeitar isso (o motor não filtra
          o input por data — passa tudo).
        """
        ...


@dataclass
class CrossSectionalMomentum:
    """Momentum 12-1 aplicado transversalmente ao universo.

    Tese: mesma anomalia do TimeSeriesMomentum single-asset — retornos dos
    últimos ~12 meses (pulando o mês mais recente) predizem retornos futuros.
    A diferença é o USO: em vez de cada ativo decidir sozinho "long/flat"
    olhando só para si, todos são rankeados juntos e o motor pega os
    top-K. Sempre carrega os melhores; não há "estar fora do mercado" por
    ausência de sinal — só há "esse ativo específico caiu do ranking".

    Por que essa mudança pode virar o jogo: a hipótese descartada em
    momentum single-asset foi "single-asset long/flat gera custo de
    oportunidade demais em ações que sobem no longo prazo". Cross-sectional
    troca a decisão "carregar ou não carregar" pela decisão "qual carregar
    entre os melhores" — nunca fica fora do mercado (a menos que haja menos
    elegíveis que K), então o custo de oportunidade estrutural some.

    Parâmetros em dias úteis: 252 ≈ 12 meses, 21 ≈ 1 mês. Mesma convenção
    do TimeSeriesMomentum para permitir comparação direta.
    """

    lookback: int = 252
    skip: int = 21

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

    def rank(self, prices: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
        """Score de cada ativo em cada data = retorno de D-lookback até D-skip.

        Fórmula: `close.shift(skip) / close.shift(lookback) - 1`.
        - `shift(skip)` em D traz o valor de D-skip (pula o último mês).
        - `shift(lookback)` traz o valor de D-lookback (12 meses atrás).
        - Ambos são causais (só olham para trás).

        Retornos NaN nos primeiros `lookback` pregões de cada ticker (sem
        histórico suficiente) e onde a máscara de elegibilidade for False.
        """
        close = prices.xs("Close", axis=1, level=1)
        numerator = close.shift(self.skip)
        denominator = close.shift(self.lookback)
        # NaN / NaN e num / NaN dão NaN — não precisa isolar warm-up
        # explicitamente, pandas já garante.
        momentum = numerator / denominator - 1.0

        # Aplica máscara de elegibilidade: onde não elegível, score = NaN
        # (motor não pega NaN no top-K). Reindex por segurança caso as
        # colunas venham em ordens diferentes.
        eligibility_aligned = eligibility.reindex_like(momentum).fillna(False)
        momentum = momentum.where(eligibility_aligned)

        return momentum

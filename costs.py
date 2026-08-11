"""Modelo de custos de transação: quanto cada ordem realmente custa.

Extraído de `core.py` para o núcleo não virar gaveta: `Signal` e `Trade` são
estrutura de dados pura, sem comportamento; a aritmética de custo (slippage,
taxas, quantidade que cabe no caixa) é uma peça de domínio própria, usada por
`backtest.py` (o motor) e `metrics.py` (o baseline de buy-and-hold) — os dois
precisam arredondar e cobrar exatamente do mesmo jeito, senão a comparação
estratégia-vs-baseline não é honesta (ver `metrics.buy_and_hold_equity_curve`).

Este módulo não importa nada do projeto: fica no mesmo nível de `core.py` no
grafo de dependências, não abaixo dele.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Costs:
    brokerage: float = 0.0  # taxa fixa em R$ por ordem (compra ou venda)
    slippage_bps: float = 0.0  # derrapagem em pontos-base sobre o preço de fill

    # Emolumentos e taxa de liquidação da B3, cada um 0,0325% do valor
    # negociado, incidindo tanto na compra quanto na venda — a tabela vigente
    # no momento em que este projeto foi escrito. São DEFAULTS de campo, não
    # constante de módulo: continuam cobrados em `Costs()` sem argumentos (não
    # existe buy-and-hold nem backtest de graça), mas passam a ser dado de
    # instância, não verdade fixa do código-fonte — a tabela de taxas muda ao
    # longo do tempo, e aplicar a tabela vigente hoje a um backtest de 2015
    # seria anacronismo. Quem precisar simular uma tabela diferente (histórica
    # ou hipotética) sobrescreve os campos; quem não precisar, não pensa nisso.
    emolumentos_rate: float = 0.000325
    liquidacao_rate: float = 0.000325

    @property
    def b3_fee_rate(self) -> float:
        return self.emolumentos_rate + self.liquidacao_rate


def apply_slippage(price: float, slippage_bps: float, side: Literal["buy", "sell"]) -> float:
    factor = slippage_bps / 10_000
    return price * (1 + factor) if side == "buy" else price * (1 - factor)


def transaction_cost(notional: float, costs: Costs) -> float:
    return notional * costs.b3_fee_rate + costs.brokerage


def affordable_quantity(cash: float, entry_fill: float, costs: Costs) -> int:
    """Maior quantidade de ações INTEIRAS que `cash` compra a `entry_fill`, já
    reservando o custo de transação. Mora aqui porque o motor e o baseline de
    buy-and-hold precisam arredondar do mesmo jeito — um baseline que compra
    fração de ação compara contra uma carteira que não existe.
    """
    affordable_cash = cash - costs.brokerage
    if affordable_cash <= 0:
        return 0
    return int(affordable_cash // (entry_fill * (1 + costs.b3_fee_rate)))

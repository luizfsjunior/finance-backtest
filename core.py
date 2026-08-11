"""Tipos compartilhados por todos os módulos — a base do grafo de dependências.

Este módulo não importa nada do projeto, propositalmente. `Signal` e `Trade`
são o vocabulário comum entre estratégia, stops, motor e métricas; se eles
morassem dentro de um desses módulos, os outros passariam a depender dele por
um motivo que não é o dele (ex.: a estratégia importando do motor só para
saber dizer "entrada"). Aqui, cada módulo depende deste núcleo e de mais nada
entre si.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import pandas as pd

Side = Literal["long"]


class Signal(IntEnum):
    """Eventos discretos que uma estratégia pode emitir por barra.

    Sinal é EVENTO DE BORDA (o instante do cruzamento), não estado contínuo.
    Se uma estratégia emitir "1 enquanto média rápida > lenta", o motor
    reentraria em toda barra em que a condição seguir verdadeira — a
    responsabilidade de detectar a borda é da estratégia, não do motor.
    """

    HOLD = 0
    ENTER_LONG = 1
    EXIT_LONG = -1


@dataclass(frozen=True)
class Trade:
    """Uma operação fechada (entrada + saída), com custos já aplicados."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    side: Side
    pnl: float  # resultado líquido em R$, já descontados os custos

    def __post_init__(self) -> None:
        # exit_date == entry_date é válido: um stop pode ser atingido na
        # própria barra em que a posição foi aberta (ver backtest.py).
        if self.exit_date < self.entry_date:
            raise ValueError(
                f"exit_date ({self.exit_date}) não pode ser anterior a entry_date ({self.entry_date})"
            )
        if self.quantity <= 0:
            raise ValueError(f"quantity deve ser positivo, recebeu {self.quantity}")

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

"""Motor de execução do backtest.

O único módulo que sabe do tempo passando. Percorre as barras e, para cada
uma: recebeu sinal na barra anterior? então executa na abertura desta. Tem
posição aberta? checa se bateu o stop. Aplica custo em toda ordem. Atualiza
o capital.

Não conhece a lógica da estratégia nem o critério de stop — só reage a
eventos discretos de `core.Signal` e chama o `stops.StopLoss` que recebeu.
Toda decisão de risco (tamanho de posição, onde fica o stop) é aplicada aqui,
propositalmente: assim dá para trocar o critério de stop sem tocar em
`strategy.py`, e trocar a estratégia sem tocar em `stops.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from core import Signal, Trade
from costs import Costs, affordable_quantity, apply_slippage, transaction_cost
from stops import StopLoss

RiskBase = Literal["equity", "initial_capital"]


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]
    skipped_signals: list[tuple[pd.Timestamp, str]]  # (data do sinal, motivo do skip)
    open_position_at_end: bool
    # Posição em aberto no fim do dataset NÃO vira Trade nem é liquidada à
    # força: forçar uma venda no último candle inventaria uma ordem que nunca
    # existiu (e um custo de transação que nunca foi pago). A equity_curve
    # continua marcada a mercado normalmente até o último candle; essa flag
    # só avisa que o resultado final inclui capital "preso" numa posição
    # aberta, para quem for interpretar os números não ser enganado.


@dataclass
class _OpenPosition:
    entry_date: pd.Timestamp
    entry_price: float  # preço de fill já com slippage
    quantity: int
    stop_price: float
    entry_cost: float  # custo de transação pago na abertura, para calcular pnl no fechamento


def _close_position(
    position: _OpenPosition,
    exit_price_raw: float,
    exit_date: pd.Timestamp,
    costs: Costs,
) -> tuple[Trade, float]:
    """Liquida `position` a `exit_price_raw` e devolve (trade, caixa recebido).

    Existe como função única porque há dois caminhos de saída no loop (a saída
    normal e o stop que dispara na própria barra de entrada) e a aritmética de
    slippage, custo e PnL precisa ser idêntica nos dois. Duplicar isso é o tipo
    de erro que não quebra teste nenhum — só muda o número no fim.
    """
    fill = apply_slippage(exit_price_raw, costs.slippage_bps, "sell")
    notional = fill * position.quantity
    exit_cost = transaction_cost(notional, costs)
    pnl = (fill - position.entry_price) * position.quantity - position.entry_cost - exit_cost
    trade = Trade(
        entry_date=position.entry_date,
        exit_date=exit_date,
        entry_price=position.entry_price,
        exit_price=fill,
        quantity=position.quantity,
        side="long",
        pnl=pnl,
    )
    return trade, notional - exit_cost


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float,
    risk_pct: float,
    stop_loss_fn: StopLoss,
    risk_base: RiskBase = "equity",
    costs: Costs = Costs(),
) -> BacktestResult:
    """Simula a execução de `signals` sobre `prices`.

    - `prices`: OHLCV diário (ver `data.get_data`), Close já ajustado.
    - `signals`: Series de `core.Signal`, MESMO índice de `prices`. O sinal em
      signals[D] só é executado na abertura de D+1 (a barra seguinte na
      sequência, não no calendário — feriados não viram gaps de 1 dia aqui
      porque iteramos por posição, não por data).
    - `risk_pct`: fração do capital-base arriscada por trade (distância até
      o stop define o tamanho da posição).
    - `stop_loss_fn`: qualquer `stops.StopLoss` (percentual, ATR, ...).
    - `risk_base`: "equity" (patrimônio no fechamento da barra anterior —
      risco com compounding) ou "initial_capital" (base fixa do início ao
      fim). Default "equity" por ser o que mais se aproxima de uma conta real.
    """
    if not prices.index.equals(signals.index):
        raise ValueError("prices e signals precisam ter o mesmo índice, na mesma ordem")
    if not 0 < risk_pct <= 1:
        raise ValueError(f"risk_pct deve estar entre 0 e 1, recebeu {risk_pct}")
    if initial_capital <= 0:
        raise ValueError(f"initial_capital deve ser positivo, recebeu {initial_capital}")
    if not signals.isin([s.value for s in Signal]).all():
        raise ValueError("signals contém valores fora de {HOLD, ENTER_LONG, EXIT_LONG}")

    cash = initial_capital
    position: _OpenPosition | None = None
    trades: list[Trade] = []
    skipped: list[tuple[pd.Timestamp, str]] = []
    equity_values: list[float] = []

    for i in range(len(prices)):
        date = prices.index[i]
        row = prices.iloc[i]

        # Sinal decidido no fechamento da barra ANTERIOR, executado na
        # abertura desta barra — é aqui que a regra "D fecha, D+1 abre e
        # executa" é imposta estruturalmente. Não existe caminho de código
        # que leia signals.iloc[i] para agir sobre row (a barra atual).
        prev_signal = Signal(signals.iloc[i - 1]) if i > 0 else Signal.HOLD

        was_flat_at_start = position is None

        if position is not None:
            exit_price_raw: float | None = None

            if row["Open"] <= position.stop_price:
                # Gap para baixo através do stop: o preço já abriu pior que o
                # stop teórico. Preencher no stop mesmo assim superestimaria
                # o resultado — o fill real é na abertura (pior preço).
                exit_price_raw = row["Open"]
            elif prev_signal == Signal.EXIT_LONG:
                # Empate "sinal de saída vs stop, ambos na abertura": stop
                # (gestão de risco) já foi checado acima e tem prioridade.
                # Chegando aqui, não houve gap through stop, então a saída
                # por sinal é o próximo evento cronológico possível.
                exit_price_raw = row["Open"]
            elif row["Low"] <= position.stop_price:
                # Stop atingido intrabarra, sem gap: preenche exatamente no
                # preço de stop (não temos dado intrabarra para saber se
                # existia um preço pior entre Open e o Low, então assumimos
                # o cenário não-adverso quando não há evidência de gap).
                exit_price_raw = position.stop_price

            if exit_price_raw is not None:
                trade, proceeds = _close_position(position, exit_price_raw, date, costs)
                cash += proceeds
                trades.append(trade)
                position = None

        if was_flat_at_start and position is None and prev_signal == Signal.ENTER_LONG:
            equity_before = equity_values[-1] if equity_values else initial_capital
            entry_fill = apply_slippage(row["Open"], costs.slippage_bps, "buy")

            stop_price = stop_loss_fn(prices, i, entry_fill)
            if stop_price is None or stop_price >= entry_fill:
                skipped.append((date, "stop_indisponivel_ou_invalido"))
            else:
                stop_distance = entry_fill - stop_price
                risk_capital_base = equity_before if risk_base == "equity" else initial_capital
                risked_capital = risk_pct * risk_capital_base
                qty_by_risk = int(risked_capital // stop_distance)
                qty_by_cash = affordable_quantity(cash, entry_fill, costs)
                quantity = min(qty_by_risk, qty_by_cash)

                if quantity < 1:
                    reason = "risco_arredondou_para_zero" if qty_by_risk < 1 else "caixa_insuficiente"
                    skipped.append((date, reason))
                else:
                    notional = entry_fill * quantity
                    entry_cost = transaction_cost(notional, costs)
                    cash -= notional + entry_cost
                    position = _OpenPosition(
                        entry_date=date,
                        entry_price=entry_fill,
                        quantity=quantity,
                        stop_price=stop_price,
                        entry_cost=entry_cost,
                    )
                    # Mesma barra em que a posição abriu já pode acionar o
                    # stop (ex.: entra na abertura e o papel despenca no
                    # mesmo pregão). Ignorar isso e só checar o stop a partir
                    # da barra seguinte deixaria a posição "sobreviver"
                    # artificialmente a um candle que deveria tê-la stopado.
                    if row["Low"] <= position.stop_price:
                        trade, proceeds = _close_position(
                            position, position.stop_price, date, costs
                        )
                        cash += proceeds
                        trades.append(trade)
                        position = None

        mark_to_market = cash + (position.quantity * row["Close"] if position is not None else 0.0)
        equity_values.append(mark_to_market)

    equity_curve = pd.Series(equity_values, index=prices.index, name="equity")
    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        skipped_signals=skipped,
        open_position_at_end=position is not None,
    )

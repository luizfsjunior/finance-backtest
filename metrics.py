"""Métricas de performance de backtest: o juiz.

Transforma a curva de capital em números comparáveis. Não conhece
`backtest.py`, `strategy.py` nem `stops.py` — só o vocabulário de `core.py`
(`Trade`) e a curva de patrimônio. As métricas não dependem de como você
chegou na curva, e é justamente isso que permite rodá-las também sobre o
buy-and-hold para ter baseline.

Convenções:
- `equity_curve`: pd.Series indexada por DatetimeIndex diário e ordenado,
  com o valor total da carteira (posição + caixa) ao final de cada pregão.
  Deve começar no capital inicial (primeiro valor = capital investido em t0).
- Métricas de retorno/risco (Sharpe, Sortino, retorno anualizado) operam sobre
  retornos percentuais diários derivados dessa curva.
- Métricas de trade (taxa de acerto, payoff, expectativa) operam sobre uma
  lista de `Trade` já com custos de transação aplicados no `pnl`.
- A única exceção à regra "nenhuma métrica aplica custos" é
  `buy_and_hold_equity_curve`: o baseline precisa pagar a MESMA tabela de
  custos que a estratégia paga em `backtest.py`, senão a comparação favorece
  o benchmark por construção. Sizing de estratégia continua exclusivo do
  motor — isso aqui é só sizing do baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import Trade
from costs import Costs, affordable_quantity, apply_slippage, transaction_cost

# Reexportados por conveniência: quem calcula métricas quase sempre importa
# `Trade` junto, e o baseline precisa dos mesmos `Costs` do motor. `Trade`
# mora em `core.py`; `Costs` mora em `costs.py`.
__all__ = [
    "Trade",
    "Costs",
    "MetricsReport",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate",
    "avg_payoff",
    "expectancy",
    "avg_holding_days",
    "buy_and_hold_equity_curve",
    "metrics_report",
]


@dataclass(frozen=True)
class MetricsReport:
    """Resultado consolidado: métricas da estratégia lado a lado com o
    baseline de buy-and-hold do mesmo ativo no mesmo período."""

    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe: float | None
    sortino: float | None
    num_trades: int
    win_rate: float | None
    avg_payoff: float | None
    expectancy: float | None
    avg_holding_days: float | None

    benchmark_total_return: float
    benchmark_annualized_return: float
    benchmark_max_drawdown: float
    benchmark_sharpe: float | None
    benchmark_sortino: float | None


# Abaixo de quanto um desvio-padrão conta como zero. Comparar `std == 0` exato
# é frágil: retornos matematicamente constantes deixam resíduo de ponto
# flutuante (uma série de -0,01 dá std ~1e-16, uma de +0,01 dá exatamente 0),
# e aí duas curvas idênticas em espírito tomariam caminhos diferentes — uma
# devolvendo ±infinito, a outra um número astronômico que só parece métrica.
#
# Tolerância PURAMENTE relativa à média (abs(mean) * rel_tol) não basta: numa
# série de juros compostos por muitos períodos, a média por período é pequena
# (~1e-4) e o resíduo de ponto flutuante no desvio (~1e-16) fica na MESMA
# ordem de abs(mean) * 1e-12 — a tolerância relativa degenera de volta para
# "std == 0" bit a bit, exatamente o problema que existia antes. Por isso o
# piso é o maior entre a tolerância relativa e um piso absoluto: abaixo de
# 1e-9 de desvio-padrão por período, nenhum retorno diário de ativo real tem
# volatilidade distinguível de zero — é sempre ruído de representação.
_ZERO_VOL_REL_TOLERANCE = 1e-12
_ZERO_VOL_ABS_TOLERANCE = 1e-9


def _is_zero_volatility(std: float, mean: float) -> bool:
    threshold = max(abs(mean) * _ZERO_VOL_REL_TOLERANCE, _ZERO_VOL_ABS_TOLERANCE)
    return std <= threshold


# Sharpe e Sortino são medidos sobre a série de RETORNOS POR PERÍODO
# (equity_curve.pct_change()), não sobre o número de trades. Uma estratégia
# com 1 trade em 2 anos ainda produz ~500 observações diárias — a maioria
# zero, mas presentes. O piso abaixo é sobre essa contagem: com menos de 2
# retornos, `std(ddof=1)` não tem grau de liberdade nenhum (é NaN, não zero),
# e o índice é matematicamente indefinido, não "um valor real com muita
# incerteza". Por isso é constante de módulo, não parâmetro: confiabilidade
# estatística (n=8 vs n=30) é uma leitura sobre o número já calculado, não uma
# segunda linha de corte — inventar um segundo piso arbitrário (10? 30?) só
# convida a ajustar o número até o resultado ficar bonito. Quem quiser badge
# de amostra pequena, anota ao lado do valor no relatório; não apaga o valor.
_MIN_PERIODS_FOR_RATIO = 2


def _degenerate_ratio(mean_excess: float) -> float | None:
    """Veredito quando a dispersão (total ou de downside) é efetivamente zero.

    `mean_excess` é o retorno médio EXCEDENTE (`rets - risk_free_rate`
    periodizado, ver `sharpe_ratio`/`sortino_ratio`), não o retorno bruto da
    curva. Com `risk_free_rate=0.0` (o default) os dois coincidem; com
    `risk_free_rate>0`, uma curva de capital parado (retorno bruto zero) tem
    excesso NEGATIVO real — capital parado perde contra a renda fixa — e cai
    no mesmo `None` de qualquer outro excesso não-nulo, pelo mesmo motivo.

    Não é "sem risco e sem prêmio -> zero, sem risco e com prêmio -> infinito".
    Um índice risco-ajustado é uma razão entre prêmio e dispersão; dispersão
    zero não é "performance infinita", é a métrica saindo do domínio em que
    ela significa algo — uma curva perfeitamente reta subindo é capital
    parado em renda fixa ou bug no motor, nunca uma leitura de risco. Por
    isso o veredito é `None` (indefinido) quando há excesso sem dispersão
    nenhuma para dividi-lo, e só `0.0` no caso realmente degenerado de "nem
    excesso nem dispersão" — sem risco e sem prêmio é defensável como
    ausência de performance, não como indefinição.

    A decisão depende só da MAGNITUDE de `mean_excess`, nunca do sinal: um
    excesso negativo constante (ex.: capital parado abaixo do risk-free) é
    tão indefinido quanto um positivo constante — não vira `-infinito`. Isso
    é o que garante que a semântica não flipe de jeito nenhum quando
    `risk_free_rate` deixa de ser zero.

    Reaproveita `_ZERO_VOL_ABS_TOLERANCE` para decidir se `mean_excess` em si
    é zero de verdade ou resíduo de ponto flutuante: comparar `mean == 0`
    exato aqui reintroduziria, na média, a mesma fragilidade que a tolerância
    combinada corrigiu no desvio-padrão.
    """
    if abs(mean_excess) <= _ZERO_VOL_ABS_TOLERANCE:
        return 0.0
    return None


def _returns(equity_curve: pd.Series) -> pd.Series:
    if len(equity_curve) < 2:
        raise ValueError("equity_curve precisa de ao menos 2 pontos para calcular retornos")
    return equity_curve.pct_change().dropna()


def total_return(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        raise ValueError("equity_curve vazia")
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        raise ValueError("equity_curve precisa de ao menos 2 pontos")
    growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return float(growth ** (periods_per_year / n_periods) - 1)


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        raise ValueError("equity_curve vazia")
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    # max(0.0, ...) em vez de só -drawdown.min(): drawdown é por definição
    # não-negativo, e -0.0 escapa da subtração em pontos onde a curva está
    # no próprio pico (equity_curve == running_max).
    return max(0.0, float(-drawdown.min()))


def sharpe_ratio(
    equity_curve: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float | None:
    """`None` quando o índice é matematicamente indefinido: menos de
    `_MIN_PERIODS_FOR_RATIO` observações, ou dispersão zero com EXCESSO de
    retorno (`retorno bruto - risk_free_rate` periodizado, não o retorno
    bruto) diferente de zero (ver `_degenerate_ratio`).

    `risk_free_rate=0.0` (default) trata retorno bruto e excesso como a mesma
    coisa. Com `risk_free_rate>0`, uma curva de capital parado tem excesso
    NEGATIVO (perde contra a renda fixa) mas segue caindo em `None`, não em
    `-infinito` — o veredito depende só da magnitude do excesso, nunca do
    sinal, então subir `risk_free_rate` não faz o resultado "virar" de um
    jeito descontínuo.
    """
    rets = _returns(equity_curve)
    if len(rets) < _MIN_PERIODS_FOR_RATIO:
        return None
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = rets - rf_per_period
    std = excess.std(ddof=1)
    mean = float(excess.mean())
    if _is_zero_volatility(std, mean):
        return _degenerate_ratio(mean)
    return float(mean / std * np.sqrt(periods_per_year))


def sortino_ratio(
    equity_curve: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float | None:
    """`None` quando o índice é matematicamente indefinido: menos de
    `_MIN_PERIODS_FOR_RATIO` observações, ou downside zero com EXCESSO de
    retorno (mesma definição de `sharpe_ratio`) diferente de zero (ver
    `_degenerate_ratio`).

    O piso de amostra usa `len(rets)` — o total de retornos, não o tamanho do
    subconjunto de downside — para que Sharpe e Sortino cheguem ao MESMO
    veredito de "amostra insuficiente" sobre a mesma série de entrada, mesmo
    calculando a dispersão de formas diferentes (`std(ddof=1)` sobre a série
    inteira aqui vs. só o subconjunto de perdas ali). Os dois estimadores não
    precisam compartilhar `ddof` — medem coisas diferentes por definição —,
    mas o corte de "não há dado o suficiente" não pode divergir entre eles.

    CUIDADO com `risk_free_rate>0`: `downside_std` aqui é semi-desvio em
    relação ao ALVO (zero de excesso), não à própria média do downside — se
    o excesso for constante e inteiramente negativo (ex.: capital parado
    rendendo menos que o risk-free todo período), o downside_std é real e
    igual à magnitude do excesso, não zero. Nesse caso o Sortino dá um número
    finito bem definido, enquanto o Sharpe da MESMA curva dá `None` (porque
    Sharpe mede dispersão em torno da média, e uma série constante não tem
    nenhuma — mesmo sendo constante e diferente de zero). As duas métricas
    divergirem nesse cenário não é inconsistência: refletem definições de
    dispersão estruturalmente diferentes.
    """
    rets = _returns(equity_curve)
    if len(rets) < _MIN_PERIODS_FOR_RATIO:
        return None
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = rets - rf_per_period
    mean = float(excess.mean())
    downside = excess[excess < 0]
    downside_std = float(np.sqrt((downside**2).mean())) if not downside.empty else 0.0
    if _is_zero_volatility(downside_std, mean):
        return _degenerate_ratio(mean)
    return float(mean / downside_std * np.sqrt(periods_per_year))


def win_rate(trades: list[Trade]) -> float:
    if not trades:
        raise ValueError("lista de trades vazia; win_rate indefinida")
    wins = sum(1 for t in trades if t.is_win)
    return wins / len(trades)


def avg_payoff(trades: list[Trade]) -> float:
    """Razão entre ganho médio e perda média (em módulo). >1 significa que
    ganhos médios superam perdas médias."""
    if not trades:
        raise ValueError("lista de trades vazia; avg_payoff indefinido")
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    if not wins:
        return 0.0
    if not losses:
        return float("inf")
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    return avg_win / avg_loss


def expectancy(trades: list[Trade]) -> float:
    """Resultado médio esperado por trade, em R$ (equivalente a
    taxa_acerto * ganho_médio - taxa_perda * perda_média)."""
    if not trades:
        raise ValueError("lista de trades vazia; expectancy indefinida")
    return sum(t.pnl for t in trades) / len(trades)


def avg_holding_days(trades: list[Trade]) -> float:
    if not trades:
        raise ValueError("lista de trades vazia; avg_holding_days indefinido")
    return sum(t.holding_days for t in trades) / len(trades)


def buy_and_hold_equity_curve(
    close_prices: pd.Series, initial_capital: float, costs: Costs = Costs()
) -> pd.Series:
    """Curva de patrimônio de comprar o máximo de ações inteiras possível no
    primeiro candle e segurar até o fim, sem fracionamento.

    O baseline paga os MESMOS custos de entrada que a estratégia (slippage,
    emolumentos/liquidação e corretagem, via `costs.py`): um buy-and-hold que
    compra de graça não é o concorrente honesto da estratégia, é um
    concorrente com desconto — e a diferença aparece inteira no veredito.

    Só o custo de ENTRADA é cobrado, porque a posição nunca é vendida: é
    marcada a mercado até o último candle, exatamente como o motor trata uma
    posição ainda aberta no fim do período.
    """
    if close_prices.empty:
        raise ValueError("close_prices vazia")

    entry_fill = apply_slippage(float(close_prices.iloc[0]), costs.slippage_bps, "buy")
    quantity = affordable_quantity(initial_capital, entry_fill, costs)
    if quantity < 1:
        raise ValueError(
            f"capital inicial ({initial_capital}) insuficiente para comprar 1 ação a "
            f"{entry_fill} já contando os custos de entrada"
        )
    notional = entry_fill * quantity
    entry_cost = transaction_cost(notional, costs)
    leftover_cash = initial_capital - notional - entry_cost
    return close_prices * quantity + leftover_cash


def metrics_report(
    equity_curve: pd.Series,
    trades: list[Trade],
    benchmark_prices: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    costs: Costs = Costs(),
) -> MetricsReport:
    """`costs` deve ser o MESMO objeto passado ao motor: é o que garante que
    estratégia e buy-and-hold sejam comparados sob a mesma tabela de custos."""
    benchmark_curve = buy_and_hold_equity_curve(benchmark_prices, initial_capital, costs)

    has_trades = len(trades) > 0
    return MetricsReport(
        total_return=total_return(equity_curve),
        annualized_return=annualized_return(equity_curve, periods_per_year),
        max_drawdown=max_drawdown(equity_curve),
        sharpe=sharpe_ratio(equity_curve, risk_free_rate, periods_per_year),
        sortino=sortino_ratio(equity_curve, risk_free_rate, periods_per_year),
        num_trades=len(trades),
        win_rate=win_rate(trades) if has_trades else None,
        avg_payoff=avg_payoff(trades) if has_trades else None,
        expectancy=expectancy(trades) if has_trades else None,
        avg_holding_days=avg_holding_days(trades) if has_trades else None,
        benchmark_total_return=total_return(benchmark_curve),
        benchmark_annualized_return=annualized_return(benchmark_curve, periods_per_year),
        benchmark_max_drawdown=max_drawdown(benchmark_curve),
        benchmark_sharpe=sharpe_ratio(benchmark_curve, risk_free_rate, periods_per_year),
        benchmark_sortino=sortino_ratio(benchmark_curve, risk_free_rate, periods_per_year),
    )

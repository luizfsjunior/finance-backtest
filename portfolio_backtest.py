"""Motor de execução do backtest multi-asset (portfolio cross-sectional).

Contraparte do `backtest.py` single-asset. As diferenças estruturais:

- Estado interno é `cash` + `dict[ticker, quantity]`, não uma posição única.
- Sinal deixa de ser evento discreto: em cada dia de rebalanceamento, o
  motor pede um SCORE por ticker (via `strategy.rank`), escolhe os top-K,
  calcula pesos alvo e emite as ordens que zeram a diferença para os pesos
  atuais.
- Não usa stops. A saída de um ativo é decidida no próximo rebalanceamento,
  se ele cair fora do top-K — não por trigger de preço intrabarra.
- Cadência de rebalance é parametrizável (mensal, semanal, trimestral).
  Fora dos dias de rebalance, o motor só marca a mercado — nenhuma ordem.
- Top-K FIXO: se o universo elegível numa data tem menos que K ativos, o
  motor aloca 1/K em cada um dos elegíveis e o RESTO FICA EM CAIXA. Isso
  é a escolha explícita do usuário para preservar o significado de "K
  posições" mesmo em warm-up.

Mesma disciplina anti-look-ahead do motor single-asset: score/decisão em
D usa dados até D (inclusive close); execução acontece na abertura de D+1.
O motor materializa isso passando `prices.iloc[:i+1]` para a estratégia no
passo `i` — a estratégia nunca vê barras à frente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from costs import Costs, apply_slippage, transaction_cost

RebalanceFrequency = Literal["monthly", "weekly", "quarterly"]


@dataclass(frozen=True)
class PortfolioTrade:
    """Um "trade" no portfolio = período contíguo em que o ativo teve peso > 0.

    Diferente do `core.Trade` single-asset por dois motivos:
      1. Não há stop nem sinal de saída explícito — a "saída" é o
         rebalanceamento que zerou a posição.
      2. `pnl` já considera os fills de todas as compras/vendas parciais que
         ocorreram durante o período, incluindo custos e slippage. Se o
         motor comprou mais no meio (rebalance aumentou o peso), isso está
         embutido no cálculo.
    """

    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    total_bought: float  # R$ pagos em compras (líquido de slippage, bruto de custo)
    total_sold: float  # R$ recebidos em vendas
    buy_costs: float  # soma dos custos de transação nas compras
    sell_costs: float  # soma dos custos de transação nas vendas
    pnl: float  # (total_sold - total_bought) - buy_costs - sell_costs

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class _PositionState:
    """Estado agregado de UMA posição enquanto ela está aberta.

    Fica de lado do `positions: dict[ticker, quantity]` (que só guarda
    quantidade viva) para acumular a métrica de trade. Quando a posição
    zera (rebalanceamento a vendeu completamente), essa entrada é
    convertida em `PortfolioTrade` e removida.
    """

    ticker: str
    entry_date: pd.Timestamp
    total_bought: float = 0.0
    total_sold: float = 0.0
    buy_costs: float = 0.0
    sell_costs: float = 0.0


@dataclass(frozen=True)
class PortfolioResult:
    equity_curve: pd.Series
    trades: list[PortfolioTrade]
    holdings: pd.DataFrame  # [data × ticker] com quantidades (ou zero)
    rebalance_dates: list[pd.Timestamp]
    open_positions_at_end: dict[str, int] = field(default_factory=dict)


# --- helpers de calendário ------------------------------------------------


def _rebalance_calendar(index: pd.DatetimeIndex, frequency: RebalanceFrequency) -> pd.DatetimeIndex:
    """Datas em que o motor RECALCULA o portfolio (decisão em D fecha, execução D+1).

    - "monthly": primeiro pregão de cada (ano, mês) do índice
    - "weekly": primeiro pregão de cada semana ISO
    - "quarterly": primeiro pregão de cada trimestre

    Como sempre pega o primeiro pregão do intervalo, feriados no dia 1 do
    mês não bagunçam nada — cai automaticamente no próximo pregão.
    """
    if frequency == "monthly":
        key = pd.Series(list(zip(index.year, index.month)), index=index)
    elif frequency == "weekly":
        iso = index.isocalendar()
        key = pd.Series(list(zip(iso.year, iso.week)), index=index)
    elif frequency == "quarterly":
        quarter = (index.month - 1) // 3 + 1
        key = pd.Series(list(zip(index.year, quarter)), index=index)
    else:
        raise ValueError(f"frequency desconhecida: {frequency}")

    is_first = key != key.shift(1)
    return index[is_first.values]


# --- núcleo do motor ------------------------------------------------------


def run_portfolio_backtest(
    prices: pd.DataFrame,  # MultiIndex de colunas (ticker, field)
    eligibility: pd.DataFrame,  # [data × ticker] bool
    strategy,  # PortfolioStrategy — Protocol, não importado para evitar ciclo
    top_k: int = 3,
    rebalance: RebalanceFrequency = "monthly",
    initial_capital: float = 10_000.0,
    costs: Costs = Costs(),
) -> PortfolioResult:
    """Simula portfolio top-K com rebalanceamento periódico.

    Loop por barra (i = 0..N-1):
      1. Se a barra ANTERIOR (i-1) era dia de rebalanceamento, executa as
         ordens decididas lá na ABERTURA desta barra (mesma semântica D+1
         do motor single-asset).
      2. Marca a mercado no CLOSE desta barra.
      3. Se ESTA barra é dia de rebalanceamento, calcula os alvos usando
         `strategy.rank(prices.iloc[:i+1], eligibility.iloc[:i+1])` — a
         estratégia enxerga até o close de hoje, decide, ordens ficam
         pendentes para execução amanhã na abertura.

    Isso separa DECISÃO (fim do dia D) de EXECUÇÃO (abertura de D+1) e
    garante que nada na barra i seja lido antes de decidir a barra i.
    """
    if len(prices) < 2:
        raise ValueError("prices precisa de ao menos 2 barras")
    if not prices.index.equals(eligibility.index):
        raise ValueError("prices e eligibility precisam ter o mesmo índice")
    if top_k < 1:
        raise ValueError(f"top_k deve ser >= 1, recebeu {top_k}")
    if initial_capital <= 0:
        raise ValueError(f"initial_capital deve ser positivo, recebeu {initial_capital}")

    tickers = list(prices.columns.get_level_values(0).unique())
    index = prices.index

    close = prices.xs("Close", axis=1, level=1)
    open_ = prices.xs("Open", axis=1, level=1)

    rebalance_dates = _rebalance_calendar(index, rebalance)
    rebalance_set = set(rebalance_dates)

    cash = float(initial_capital)
    positions: dict[str, int] = {t: 0 for t in tickers}
    position_states: dict[str, _PositionState] = {}
    trades: list[PortfolioTrade] = []
    equity_values: list[float] = []
    holdings_rows: list[dict[str, int]] = []

    # Ordens pendentes decididas no fim do dia anterior, executam hoje na abertura.
    # {ticker: quantidade_alvo} — positivo compra a diferença, negativo vende.
    pending_target_qty: dict[str, int] | None = None

    for i in range(len(index)):
        today = index[i]

        # --- 1. EXECUÇÃO DAS ORDENS PENDENTES (na abertura de hoje) ---
        if pending_target_qty is not None:
            today_open = open_.iloc[i]

            # Fazemos VENDAS primeiro — liberam caixa que pode ser reinvestido
            # em compras nesta mesma barra. Sem isso, poderia faltar caixa
            # para um rebalance que só rotaciona posições (não injeta capital).
            for ticker, target_qty in pending_target_qty.items():
                current_qty = positions[ticker]
                delta = target_qty - current_qty
                if delta >= 0:
                    continue
                price_raw = today_open.get(ticker)
                if pd.isna(price_raw):
                    # Ativo sem pregão hoje: não executa. Estado fica como
                    # está; próximo rebalance decide de novo.
                    continue
                sell_qty = -delta
                fill = apply_slippage(float(price_raw), costs.slippage_bps, "sell")
                notional = fill * sell_qty
                cost = transaction_cost(notional, costs)
                cash += notional - cost
                positions[ticker] = current_qty + delta  # = target_qty

                st = position_states.get(ticker)
                if st is not None:
                    st.total_sold += notional
                    st.sell_costs += cost
                    if positions[ticker] == 0:
                        pnl = (
                            (st.total_sold - st.total_bought)
                            - st.buy_costs
                            - st.sell_costs
                        )
                        trades.append(
                            PortfolioTrade(
                                ticker=ticker,
                                entry_date=st.entry_date,
                                exit_date=today,
                                total_bought=st.total_bought,
                                total_sold=st.total_sold,
                                buy_costs=st.buy_costs,
                                sell_costs=st.sell_costs,
                                pnl=pnl,
                            )
                        )
                        del position_states[ticker]

            for ticker, target_qty in pending_target_qty.items():
                current_qty = positions[ticker]
                delta = target_qty - current_qty
                if delta <= 0:
                    continue
                price_raw = today_open.get(ticker)
                if pd.isna(price_raw):
                    continue
                fill = apply_slippage(float(price_raw), costs.slippage_bps, "buy")
                buy_qty = delta
                notional = fill * buy_qty
                cost = transaction_cost(notional, costs)
                # Se cash insuficiente, reduz quantidade para o que cabe.
                if notional + cost > cash:
                    max_buyable = _max_affordable(cash, fill, costs)
                    if max_buyable < 1:
                        continue
                    buy_qty = max_buyable
                    notional = fill * buy_qty
                    cost = transaction_cost(notional, costs)
                cash -= notional + cost
                positions[ticker] = current_qty + buy_qty

                st = position_states.get(ticker)
                if st is None:
                    st = _PositionState(ticker=ticker, entry_date=today)
                    position_states[ticker] = st
                st.total_bought += notional
                st.buy_costs += cost

            pending_target_qty = None

        # --- 2. MARK-TO-MARKET NO CLOSE ---
        today_close = close.iloc[i]
        equity = cash
        for ticker, qty in positions.items():
            if qty > 0:
                px = today_close.get(ticker)
                if pd.notna(px):
                    equity += qty * float(px)
                else:
                    # Sem preço hoje — marca pelo último close conhecido.
                    # Preserva capital em vez de fingir que a posição sumiu.
                    last_valid = close[ticker].iloc[: i + 1].last_valid_index()
                    if last_valid is not None:
                        equity += qty * float(close[ticker].loc[last_valid])
        equity_values.append(equity)
        holdings_rows.append(dict(positions))

        # --- 3. DECISÃO DE REBALANCEAMENTO (usa dados até hoje inclusive) ---
        if today in rebalance_set:
            scores_today = strategy.rank(
                prices.iloc[: i + 1], eligibility.iloc[: i + 1]
            ).iloc[-1]
            target_qty = _compute_target_quantities(
                scores=scores_today,
                today_close=today_close,
                total_equity=equity,
                top_k=top_k,
                costs=costs,
            )
            pending_target_qty = target_qty

    equity_curve = pd.Series(equity_values, index=index, name="equity")
    holdings = pd.DataFrame(holdings_rows, index=index, columns=tickers).fillna(0).astype(int)

    # Fecha qualquer trade em aberto no fim do período SEM emitir ordem: a
    # posição não foi liquidada de verdade (nenhuma venda), então o
    # PortfolioTrade fica com sold=0 e o PnL é o "não realizado" atual.
    # Mesma filosofia do motor single-asset — não inventamos ordens que
    # nunca aconteceram. Reportamos como open_positions_at_end.
    open_at_end = {t: q for t, q in positions.items() if q > 0}

    return PortfolioResult(
        equity_curve=equity_curve,
        trades=trades,
        holdings=holdings,
        rebalance_dates=list(rebalance_dates),
        open_positions_at_end=open_at_end,
    )


def _max_affordable(cash: float, fill: float, costs: Costs) -> int:
    """Maior quantidade inteira comprável com `cash` a `fill`, já reservando custo.

    Duplica a lógica de `costs.affordable_quantity` mas trabalha em cima do
    caixa CORRENTE do motor (não do inicial), então mora aqui em vez de
    depender de `Cash` do outro módulo. A fórmula é a mesma: `(cash - brokerage) //
    (fill * (1 + b3_fee_rate))`.
    """
    affordable_cash = cash - costs.brokerage
    if affordable_cash <= 0:
        return 0
    return int(affordable_cash // (fill * (1 + costs.b3_fee_rate)))


def _compute_target_quantities(
    scores: pd.Series,
    today_close: pd.Series,
    total_equity: float,
    top_k: int,
    costs: Costs,
) -> dict[str, int]:
    """Dado o vetor de scores de hoje, decide `{ticker: qty_alvo}` para amanhã.

    Regra dos pesos (escolha explícita do usuário): K FIXO com resto em caixa.
    Cada ativo selecionado recebe `1/top_k` do equity atual; se menos que K
    ativos têm score válido, os que não existem viram "posição zero" —
    diferença sobra em caixa.

    Aproximação de sizing: usa close de hoje para calcular quantidade alvo.
    A execução amanhã na abertura pode dar preço diferente, o que introduz
    erro de tracking pequeno em relação aos pesos exatos 1/K. É aceitável —
    a alternativa (recalcular no fill de cada perna, sequencial) exigiria
    saber o preço de execução antes de decidir a ordem, o que quebra a
    separação decisão/execução.
    """
    valid_scores = scores.dropna()
    if valid_scores.empty:
        # Ninguém elegível — alvo é zerar tudo, ficar 100% em caixa.
        return {t: 0 for t in scores.index}

    top = valid_scores.nlargest(top_k).index
    capital_per_slot = total_equity / top_k

    target: dict[str, int] = {t: 0 for t in scores.index}
    for ticker in top:
        px = today_close.get(ticker)
        if pd.isna(px) or px <= 0:
            continue
        # Considera custo de compra ao calcular quantidade, senão fica sempre
        # 1 ação abaixo do que caberia.
        px_with_slip_est = float(px) * (1 + costs.slippage_bps / 10_000)
        cost_multiplier = 1 + costs.b3_fee_rate
        qty = int(capital_per_slot // (px_with_slip_est * cost_multiplier))
        target[ticker] = max(qty, 0)
    return target


def portfolio_buy_and_hold_equity_curve(
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    initial_capital: float = 10_000.0,
    costs: Costs = Costs(),
) -> pd.Series:
    """Benchmark equal-weight: compra 1/N em cada ticker no dia em que ele fica elegível PELA PRIMEIRA VEZ, nunca vende.

    N = número de tickers no universo (não o número elegível na 1ª data). Se
    um ativo só ficar elegível no meio do período, o capital reservado para
    ele fica em caixa até lá. Isso torna a comparação com a estratégia
    apples-to-apples: os dois enfrentam a mesma máscara de elegibilidade e o
    mesmo warm-up.

    Nunca rebalanceia (é buy-and-hold). Vai ficando "des-equal-weighted"
    conforme os ativos performam de forma diferente — é a mesma dinâmica de
    um B&H tradicional.
    """
    tickers = list(prices.columns.get_level_values(0).unique())
    n = len(tickers)
    if n == 0:
        raise ValueError("universo vazio")

    close = prices.xs("Close", axis=1, level=1)
    open_ = prices.xs("Open", axis=1, level=1)

    capital_per_ticker = initial_capital / n
    quantities: dict[str, int] = {t: 0 for t in tickers}
    cash_by_ticker: dict[str, float] = {t: capital_per_ticker for t in tickers}
    bought: dict[str, bool] = {t: False for t in tickers}

    equity_values: list[float] = []
    index = prices.index

    for i in range(len(index)):
        today_open = open_.iloc[i]
        today_close = close.iloc[i]
        today_elig = eligibility.iloc[i]

        # Executa compras no OPEN do dia (mesma semântica D+0 do motor —
        # aqui não há sinal D+1, o "sinal" é a máscara ficar True).
        for ticker in tickers:
            if bought[ticker]:
                continue
            if not bool(today_elig.get(ticker, False)):
                continue
            px_raw = today_open.get(ticker)
            if pd.isna(px_raw):
                continue
            fill = apply_slippage(float(px_raw), costs.slippage_bps, "buy")
            qty = _max_affordable(cash_by_ticker[ticker], fill, costs)
            if qty < 1:
                bought[ticker] = True  # não tem grana suficiente, desiste
                continue
            notional = fill * qty
            cost = transaction_cost(notional, costs)
            cash_by_ticker[ticker] -= notional + cost
            quantities[ticker] = qty
            bought[ticker] = True

        equity = sum(cash_by_ticker.values())
        for ticker, qty in quantities.items():
            if qty > 0:
                px = today_close.get(ticker)
                if pd.notna(px):
                    equity += qty * float(px)
                else:
                    last_valid = close[ticker].iloc[: i + 1].last_valid_index()
                    if last_valid is not None:
                        equity += qty * float(close[ticker].loc[last_valid])
        equity_values.append(equity)

    return pd.Series(equity_values, index=index, name="equity_bh")

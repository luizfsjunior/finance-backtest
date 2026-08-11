from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from costs import Costs
from metrics import (
    MetricsReport,
    Trade,
    annualized_return,
    avg_holding_days,
    avg_payoff,
    buy_and_hold_equity_curve,
    expectancy,
    max_drawdown,
    metrics_report,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)


def _curve(values: list[float], start: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx)


def _trade(entry: str, exit_: str, entry_price: float, exit_price: float, qty: int = 100) -> Trade:
    return Trade(
        entry_date=pd.Timestamp(entry),
        exit_date=pd.Timestamp(exit_),
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=qty,
        side="long",
        pnl=(exit_price - entry_price) * qty,
    )


class TestTrade:
    def test_exit_antes_de_entry_levanta_erro(self):
        with pytest.raises(ValueError):
            Trade(
                entry_date=pd.Timestamp("2024-01-10"),
                exit_date=pd.Timestamp("2024-01-05"),
                entry_price=10.0,
                exit_price=11.0,
                quantity=100,
                side="long",
                pnl=100.0,
            )

    def test_exit_igual_a_entry_e_valido_stop_na_barra_de_entrada(self):
        # cenário real em backtest.py: stop atingido na própria barra de entrada
        t = Trade(
            entry_date=pd.Timestamp("2024-01-10"),
            exit_date=pd.Timestamp("2024-01-10"),
            entry_price=10.0,
            exit_price=9.5,
            quantity=100,
            side="long",
            pnl=-50.0,
        )
        assert t.holding_days == 0

    def test_quantity_nao_positiva_levanta_erro(self):
        with pytest.raises(ValueError):
            Trade(
                entry_date=pd.Timestamp("2024-01-05"),
                exit_date=pd.Timestamp("2024-01-10"),
                entry_price=10.0,
                exit_price=11.0,
                quantity=0,
                side="long",
                pnl=0.0,
            )

    def test_holding_days_e_is_win(self):
        t = _trade("2024-01-02", "2024-01-12", 10.0, 12.0)
        assert t.holding_days == 10
        assert t.is_win is True

        loser = _trade("2024-01-02", "2024-01-05", 10.0, 9.0)
        assert loser.is_win is False


class TestRetornoEDrawdown:
    def test_total_return_simples(self):
        curve = _curve([100, 110, 121])
        assert total_return(curve) == pytest.approx(0.21)

    def test_total_return_curva_vazia_levanta_erro(self):
        with pytest.raises(ValueError):
            total_return(pd.Series(dtype=float))

    def test_annualized_return_um_ano_bate_com_total_return(self):
        curve = _curve([100.0] + [100.0] * 251 + [121.0])
        # 252 períodos de negociação ~ 1 ano
        ann = annualized_return(curve, periods_per_year=252)
        assert ann == pytest.approx(0.21, abs=1e-6)

    def test_max_drawdown_simples(self):
        curve = _curve([100, 120, 90, 95, 130])
        # pico 120 -> vale 90 => dd de 25%
        assert max_drawdown(curve) == pytest.approx(0.25)

    def test_max_drawdown_serie_sempre_subindo_e_zero(self):
        curve = _curve([100, 105, 110, 120])
        assert max_drawdown(curve) == pytest.approx(0.0)

    def test_max_drawdown_curva_reta_nao_devolve_zero_negativo(self):
        # equity_curve == running_max em todo ponto -> drawdown exatamente 0.0,
        # mas -0.0 (zero negativo de ponto flutuante) não é o mesmo valor
        # para efeitos de repr/formatação, e drawdown é por definição >= 0.
        curve = _curve([100.0, 100.0, 100.0, 100.0])
        result = max_drawdown(curve)
        assert result == 0.0
        assert math.copysign(1.0, result) == 1.0


class TestSharpeSortino:
    def test_sharpe_com_retornos_constantes_positivos_e_indefinido(self):
        # retorno diário constante e positivo: excesso constante, desvio zero,
        # prêmio real e não-nulo. Dispersão zero não é "performance infinita"
        # — é a métrica saindo do domínio em que ela significa algo.
        curve = _curve([100, 101, 102.01, 103.0301])
        assert sharpe_ratio(curve) is None

    def test_curva_reta_da_sharpe_e_sortino_zero_em_vez_de_indefinido(self):
        # estratégia que nunca operou: capital parado do início ao fim.
        # Sem risco E sem prêmio é o único caso degenerado com valor definido
        # (ausência de performance); com prêmio, vira indefinido (None).
        curve = _curve([10_000.0] * 30)
        assert sharpe_ratio(curve) == 0.0
        assert sortino_ratio(curve) == 0.0

    def test_crescimento_composto_quase_constante_e_indefinido_nao_espurio(self):
        # juros compostos ao longo de muitos períodos: o retorno percentual
        # teórico é idêntico a cada barra, mas a representação em ponto
        # flutuante deixa um resíduo de desvio-padrão da ordem de 1e-16
        # enquanto a média (~1e-4) é real. Tolerância PURAMENTE relativa à
        # média (abs(mean) * rel_tol) fica da mesma ordem desse resíduo — a
        # comparação some no ruído de ponto flutuante em vez de reconhecer que
        # a série é, na prática, constante. O motor de fato não teve
        # nenhum trade aqui; é só uma curva de referência sintética para
        # expor o problema de tolerância. O veredito correto para "prêmio
        # real sem dispersão nenhuma" é indefinido, não um número astronômico
        # nem infinito (que contaminaria qualquer ranking por Sharpe).
        values = [10_000.0 * (1.0001**i) for i in range(200)]
        curve = _curve(values)
        assert sharpe_ratio(curve) is None
        assert sortino_ratio(curve) is None

    def test_curva_reta_nao_derruba_o_relatorio_inteiro(self):
        equity = _curve([10_000.0] * 30)
        prices = _curve([10.0 + i * 0.1 for i in range(30)])
        report = metrics_report(equity, [], prices, initial_capital=10_000.0)
        assert report.num_trades == 0
        assert report.total_return == 0.0
        assert report.sharpe == 0.0
        # o baseline segue calculado normalmente: é o ponto de comparação
        assert report.benchmark_total_return != 0.0

    def test_sharpe_negativo_constante_e_indefinido(self):
        curve = _curve([100, 99, 98.01, 97.0299])
        assert sharpe_ratio(curve) is None

    def test_sharpe_maior_quando_retorno_maior_com_mesma_volatilidade(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.5, 100)
        base_low = 100 + np.cumsum(noise + 0.05)
        base_high = 100 + np.cumsum(noise + 0.5)
        sharpe_low = sharpe_ratio(_curve(list(base_low)))
        sharpe_high = sharpe_ratio(_curve(list(base_high)))
        assert sharpe_low is not None and sharpe_high is not None
        assert sharpe_high > sharpe_low

    def test_sortino_sem_retornos_negativos_e_indefinido_com_premio(self):
        # sem nenhum retorno abaixo do risk-free, downside_std é zero — mas o
        # excesso médio da série é positivo e real (~0,0099), não ruído.
        # Ausência de downside com prêmio real também é "sem dispersão pela
        # qual dividir", logo indefinido, não infinito.
        curve = _curve([100, 101, 102, 103, 104])
        assert sortino_ratio(curve) is None

    def test_sortino_com_perdas_e_finito(self):
        curve = _curve([100, 105, 98, 110, 90, 115])
        result = sortino_ratio(curve)
        assert result is not None and np.isfinite(result)

    def test_sortino_com_exatamente_uma_observacao_negativa_e_finito_nao_zero(self):
        # downside_std aqui é sqrt(mean(x**2)) sobre o subconjunto de perdas —
        # RMS em relação ao alvo (zero), não desvio em torno da própria média
        # do subconjunto. Com 1 único valor negativo, isso dá |x| > 0 sempre,
        # nunca 0.0 (só daria 0 com uma fórmula tipo std(ddof=0) em torno da
        # própria média, que não é a usada aqui).
        curve = _curve([100, 105, 110, 108, 115])
        result = sortino_ratio(curve)
        assert result is not None and np.isfinite(result) and result != 0.0

    def test_sortino_downside_vazio_nao_propaga_nan(self):
        # array de downside vazio cai no `else 0.0` explícito do ternário —
        # nunca chama np.std/mean sobre array vazio, então nunca produz NaN
        # silencioso. O resultado deve ser None (indefinido, ver item 2),
        # nunca NaN.
        curve = _curve([100, 101, 102, 103, 104])
        result = sortino_ratio(curve)
        assert result is None

    def test_amostra_menor_que_o_minimo_e_indefinida_nas_duas_metricas(self):
        # uma única barra de retorno (2 pontos de equity) não dá grau de
        # liberdade nenhum para std(ddof=1) existir — é indefinido, não erro
        # nem zero. O piso é o MESMO para as duas funções: precisam concordar
        # sobre "amostra insuficiente" para a mesma série de entrada.
        curve = _curve([10_000.0, 10_100.0])
        assert sharpe_ratio(curve) is None
        assert sortino_ratio(curve) is None

    def test_curva_reta_com_risk_free_positivo_nao_vira_menos_infinito(self):
        # capital parado (retorno bruto zero) avaliado contra um risk-free
        # real (~13% a.a., CDI): o excesso é NEGATIVO e real, não ruído de
        # ponto flutuante. Antes do item 4 essa combinação (dispersão zero +
        # excesso negativo) devolvia -infinito; agora o veredito depende só
        # da magnitude do excesso, nunca do sinal, e continua None.
        curve = _curve([10_000.0] * 60)
        result = sharpe_ratio(curve, risk_free_rate=0.13)
        assert result is None
        assert result != float("-inf")

    def test_sharpe_e_sortino_podem_divergir_com_risk_free_positivo(self):
        # mesma curva reta contra risk-free positivo: Sharpe mede dispersão
        # EM TORNO DA MÉDIA (zero numa série constante) -> indefinido.
        # Sortino mede semi-desvio em relação ao ALVO zero, não à própria
        # média do downside -> com excesso constante e integralmente
        # negativo, o "downside" é a série inteira e o semi-desvio é real
        # (igual à magnitude do excesso) -> resultado finito e definido.
        # Isso é esperado pela definição de cada métrica, não inconsistência.
        curve = _curve([10_000.0] * 60)
        sharpe = sharpe_ratio(curve, risk_free_rate=0.13)
        sortino = sortino_ratio(curve, risk_free_rate=0.13)
        assert sharpe is None
        assert sortino is not None and np.isfinite(sortino) and sortino < 0


class TestMetricasDeTrade:
    def test_win_rate(self):
        trades = [
            _trade("2024-01-02", "2024-01-05", 10, 12),
            _trade("2024-01-06", "2024-01-09", 10, 9),
            _trade("2024-01-10", "2024-01-15", 10, 11),
        ]
        assert win_rate(trades) == pytest.approx(2 / 3)

    def test_win_rate_lista_vazia_levanta_erro(self):
        with pytest.raises(ValueError):
            win_rate([])

    def test_avg_payoff(self):
        trades = [
            _trade("2024-01-02", "2024-01-05", 10, 12, qty=100),  # +200
            _trade("2024-01-06", "2024-01-09", 10, 8, qty=100),  # -200
        ]
        assert avg_payoff(trades) == pytest.approx(1.0)

    def test_avg_payoff_sem_perdas_e_infinito(self):
        trades = [_trade("2024-01-02", "2024-01-05", 10, 12)]
        assert avg_payoff(trades) == float("inf")

    def test_expectancy(self):
        trades = [
            _trade("2024-01-02", "2024-01-05", 10, 12, qty=100),  # +200
            _trade("2024-01-06", "2024-01-09", 10, 9, qty=100),  # -100
        ]
        assert expectancy(trades) == pytest.approx(50.0)

    def test_avg_holding_days(self):
        trades = [
            _trade("2024-01-02", "2024-01-12", 10, 11),  # 10 dias
            _trade("2024-01-15", "2024-01-20", 10, 11),  # 5 dias
        ]
        assert avg_holding_days(trades) == pytest.approx(7.5)


class TestBuyAndHold:
    def test_quantidade_inteira_sem_fracionamento(self):
        prices = _curve([10.0, 11.0, 12.0])
        curve = buy_and_hold_equity_curve(prices, initial_capital=1005.0)
        # 100 ações a 10 (o custo B3 de ~R$0,65 não tira a centésima ação),
        # sobra o caixa restante já descontado o custo de entrada
        entry_cost = 100 * 10.0 * Costs().b3_fee_rate
        assert curve.iloc[0] == pytest.approx(1005.0 - entry_cost)
        assert curve.iloc[-1] == pytest.approx(100 * 12.0 + 5.0 - entry_cost)

    def test_baseline_paga_os_mesmos_custos_da_estrategia(self):
        prices = _curve([10.0, 11.0, 12.0])
        sem_custo = buy_and_hold_equity_curve(prices, 100_000.0, Costs())
        com_custo = buy_and_hold_equity_curve(
            prices, 100_000.0, Costs(brokerage=20.0, slippage_bps=10.0)
        )
        # corretagem e slippage encarecem a entrada -> patrimônio final menor.
        # Um buy-and-hold que compra de graça é um concorrente com desconto.
        assert com_custo.iloc[-1] < sem_custo.iloc[-1]

    def test_custo_b3_e_sempre_cobrado_mesmo_sem_corretagem_nem_slippage(self):
        prices = _curve([10.0, 12.0])
        curve = buy_and_hold_equity_curve(prices, initial_capital=1000.0, costs=Costs())
        assert curve.iloc[0] < 1000.0  # emolumentos + liquidação não são opcionais

    def test_taxa_b3_e_parametrizavel_por_instancia_nao_constante_fixa(self):
        # a tabela de emolumentos/liquidação é DEFAULT de campo de `Costs`,
        # não constante de módulo — simular uma tabela histórica diferente
        # (ou uma hipótese de mudança futura) não deveria exigir editar
        # código-fonte, só instanciar `Costs` com outros valores.
        prices = _curve([10.0, 12.0])
        tabela_atual = Costs()
        tabela_hipotetica = Costs(emolumentos_rate=0.001, liquidacao_rate=0.001)
        curva_atual = buy_and_hold_equity_curve(prices, 100_000.0, tabela_atual)
        curva_hipotetica = buy_and_hold_equity_curve(prices, 100_000.0, tabela_hipotetica)
        assert curva_hipotetica.iloc[0] < curva_atual.iloc[0]

    def test_capital_insuficiente_levanta_erro(self):
        prices = _curve([100.0, 110.0])
        with pytest.raises(ValueError):
            buy_and_hold_equity_curve(prices, initial_capital=50.0)


class TestMetricsReport:
    def test_report_com_trades(self):
        equity = _curve([1000, 1050, 1100, 1080, 1150])
        prices = _curve([10, 10.5, 11, 10.8, 11.5])
        trades = [
            _trade("2024-01-02", "2024-01-04", 10, 11, qty=50),
            _trade("2024-01-05", "2024-01-08", 11, 10.5, qty=50),
        ]
        report = metrics_report(equity, trades, prices, initial_capital=1000.0)
        assert isinstance(report, MetricsReport)
        assert report.num_trades == 2
        assert report.win_rate == pytest.approx(0.5)
        assert report.total_return == pytest.approx(0.15)
        # o ativo subiu 15%, mas o baseline paga o custo de entrada como a
        # estratégia paga: o retorno do buy-and-hold fica logo abaixo disso
        assert report.benchmark_total_return < 0.15
        assert report.benchmark_total_return == pytest.approx(0.15, abs=0.01)

    def test_report_sem_trades_metricas_de_trade_sao_none(self):
        equity = _curve([1000, 1010, 1020])
        prices = _curve([10, 10.1, 10.2])
        report = metrics_report(equity, [], prices, initial_capital=1000.0)
        assert report.num_trades == 0
        assert report.win_rate is None
        assert report.avg_payoff is None
        assert report.expectancy is None
        assert report.avg_holding_days is None

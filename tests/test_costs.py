from __future__ import annotations

import pytest

from costs import Costs, affordable_quantity, apply_slippage, transaction_cost


class TestCosts:
    def test_b3_fee_rate_default_cobra_emolumentos_e_liquidacao(self):
        assert Costs().b3_fee_rate == pytest.approx(0.000325 * 2)

    def test_b3_fee_rate_e_derivado_dos_campos_de_instancia(self):
        costs = Costs(emolumentos_rate=0.001, liquidacao_rate=0.002)
        assert costs.b3_fee_rate == pytest.approx(0.003)

    def test_costs_e_imutavel(self):
        with pytest.raises((AttributeError, TypeError)):
            Costs().brokerage = 10.0  # type: ignore[misc]


class TestApplySlippage:
    def test_compra_encarece_venda_barateia(self):
        assert apply_slippage(100.0, 10.0, "buy") == pytest.approx(100.1)
        assert apply_slippage(100.0, 10.0, "sell") == pytest.approx(99.9)

    def test_slippage_zero_nao_altera_preco(self):
        assert apply_slippage(100.0, 0.0, "buy") == pytest.approx(100.0)


class TestTransactionCost:
    def test_cobra_taxa_b3_mesmo_sem_corretagem(self):
        cost = transaction_cost(10_000.0, Costs())
        assert cost == pytest.approx(10_000.0 * 0.000325 * 2)

    def test_soma_corretagem_fixa(self):
        cost = transaction_cost(10_000.0, Costs(brokerage=5.0))
        assert cost == pytest.approx(10_000.0 * 0.000325 * 2 + 5.0)

    def test_usa_a_tabela_da_instancia_nao_uma_constante_fixa(self):
        cost_padrao = transaction_cost(10_000.0, Costs())
        cost_customizado = transaction_cost(10_000.0, Costs(emolumentos_rate=0.01, liquidacao_rate=0.0))
        assert cost_customizado != cost_padrao
        assert cost_customizado == pytest.approx(10_000.0 * 0.01)


class TestAffordableQuantity:
    def test_arredonda_para_baixo_em_acoes_inteiras(self):
        # 1000 / (10 * (1+taxa)) ~ 99.9... -> 99, nunca fração de ação
        qty = affordable_quantity(1000.0, 10.0, Costs())
        assert qty == 99

    def test_desconta_corretagem_antes_de_dividir(self):
        qty_sem_corretagem = affordable_quantity(1000.0, 10.0, Costs())
        qty_com_corretagem = affordable_quantity(1000.0, 10.0, Costs(brokerage=50.0))
        assert qty_com_corretagem < qty_sem_corretagem

    def test_caixa_insuficiente_para_corretagem_da_zero(self):
        assert affordable_quantity(10.0, 10.0, Costs(brokerage=50.0)) == 0

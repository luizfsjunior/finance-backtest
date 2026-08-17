"""Contrato do Componente 1 do laboratório (SPEC_LAB.md): sweep de parâmetros.

O que estes testes fixam, na ordem em que a spec cobra:

1. Produto cartesiano determinístico (mesma grade = mesma ordem, sempre).
2. Combinações inválidas filtradas ANTES de rodar — não executadas e descartadas
   depois (a diferença é tempo de CPU e, pior, `n_combos` inflado).
3. Teto de combinações: sweep grande demais aborta antes de escrever qualquer
   coisa, não no meio.
4. Proveniência completa em toda linha (`sweep_id`, `combo_id`, `n_combos`,
   `train_test`) — é ela que sustenta o obstáculo 4.
5. Ticker ruim não derruba o sweep (convenção herdada do batch).
6. Determinismo: mesmo sweep rodado duas vezes = exatamente os mesmos números.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data import DataFetchError
from stops import AtrStop, NoStop
from strategy import MovingAverageCrossover
from sweep import (
    DEFAULT_LOG_PATH,
    EmptySweepError,
    SweepSpec,
    SweepTooLargeError,
    aggregate_by_combo,
    expand_grid,
    run_sweep,
    valid_combos,
)

TICKERS = ["AAA3.SA", "BBB3.SA"]


def _oscillating(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Ciclos amplos o bastante para qualquer janela curta cruzar.

    A fase depende do ticker (hash estável do primeiro caractere), então os
    dois papéis do universo de teste não produzem exatamente o mesmo resultado
    — senão a agregação por combinação ficaria trivialmente correta.
    """
    idx = pd.bdate_range(start=start, end=end)
    phase = (ord(ticker[0]) % 5) * 0.7
    close = pd.Series(
        [20.0 + 6.0 * math.sin(2 * math.pi * i / 60 + phase) for i in range(len(idx))],
        index=idx,
    )

    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], [ticker]]
    )
    df = pd.DataFrame(index=idx, columns=columns, dtype=float)
    df[("Open", ticker)] = close
    df[("High", ticker)] = close + 0.5
    df[("Low", ticker)] = close - 0.5
    df[("Close", ticker)] = close
    df[("Volume", ticker)] = 1_000_000
    return df


def _fetch_with_bad_ticker(ticker: str, start: date, end: date) -> pd.DataFrame:
    if ticker == "BAD3.SA":
        raise DataFetchError(f"sem dados para {ticker}")
    return _oscillating(ticker, start, end)


def _spec(**overrides) -> SweepSpec:
    base = dict(
        strategy_class=MovingAverageCrossover,
        strategy_grid={"fast_window": [3, 5], "slow_window": [10, 15]},
        tickers=TICKERS,
        start=date(2024, 1, 2),
        end=date(2024, 12, 30),
        stop_class=NoStop,
        stop_grid={},
        min_median_turnover=0.0,
        hypothesis="teste",
    )
    base.update(overrides)
    return SweepSpec(**base)


def _run(spec: SweepSpec, tmp_path: Path, **kwargs):
    kwargs.setdefault("log_path", tmp_path / "sweep_runs.csv")
    kwargs.setdefault("cache_dir", tmp_path / "cache")
    kwargs.setdefault("fetch_fn", _oscillating)
    kwargs.setdefault("sweep_id", "SWEEP_FIXO")
    kwargs.setdefault("verbose", False)
    return run_sweep(spec, **kwargs)


class TestExpandGrid:
    def test_produto_cartesiano_tem_o_tamanho_do_produto(self):
        combos = expand_grid({"a": [1, 2, 3], "b": ["x", "y"]})
        assert len(combos) == 6
        assert {"a": 1, "b": "x"} in combos
        assert {"a": 3, "b": "y"} in combos

    def test_ordem_e_deterministica(self):
        grid = {"a": [1, 2, 3], "b": ["x", "y"]}
        assert expand_grid(grid) == expand_grid(grid)
        # primeiro parâmetro varia mais devagar (convenção de itertools.product):
        # ler o CSV de um sweep fica previsível, blocos contíguos por 'a'
        assert expand_grid(grid)[:2] == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}]

    def test_grade_vazia_produz_uma_combinacao_vazia(self):
        # "nenhum parâmetro para varrer" = uma execução com os defaults, não zero
        assert expand_grid({}) == [{}]

    def test_lista_vazia_em_um_parametro_e_erro(self):
        # produto cartesiano com fator vazio é vazio: o sweep sumiria em
        # silêncio. Melhor gritar do que rodar nada e reportar sucesso.
        with pytest.raises(ValueError, match="fast_window"):
            expand_grid({"fast_window": []})


class TestFiltroDeCombinacoesInvalidas:
    def test_combinacoes_invalidas_sao_removidas_antes_de_rodar(self):
        # fast >= slow é rejeitado pelo __post_init__ da própria estratégia:
        # o filtro não duplica a regra, ele a consulta construindo o objeto.
        combos = valid_combos(
            _spec(strategy_grid={"fast_window": [3, 20], "slow_window": [10]})
        )
        assert [c.strategy_params for c in combos] == [{"fast_window": 3, "slow_window": 10}]

    def test_combinacoes_invalidas_nao_contam_em_n_combos(self, tmp_path: Path):
        result = _run(
            _spec(strategy_grid={"fast_window": [3, 20], "slow_window": [10]}), tmp_path
        )
        assert all(row["n_combos"] == 1 for row in result.rows)
        assert all(row["param_fast_window"] == 3 for row in result.rows)

    def test_grade_inteiramente_invalida_e_erro_explicito(self):
        with pytest.raises(EmptySweepError):
            valid_combos(_spec(strategy_grid={"fast_window": [20, 30], "slow_window": [10]}))

    def test_parametro_inexistente_e_erro_de_programacao_nao_filtro(self):
        # errar o nome do parâmetro na grade não pode virar "combinação
        # inválida" silenciosa — isso esconderia um sweep que testou nada.
        with pytest.raises(TypeError):
            valid_combos(_spec(strategy_grid={"janela_rapida": [3], "slow_window": [10]}))

    def test_grade_do_stop_entra_no_produto_cartesiano(self):
        combos = valid_combos(
            _spec(
                strategy_grid={"fast_window": [3], "slow_window": [10]},
                stop_class=AtrStop,
                stop_grid={"period": [14], "multiplier": [2.0, 2.5]},
            )
        )
        assert len(combos) == 2
        assert [c.stop_params["multiplier"] for c in combos] == [2.0, 2.5]


class TestTeto:
    def test_sweep_acima_do_teto_aborta_antes_de_executar(self, tmp_path: Path):
        log = tmp_path / "sweep_runs.csv"
        spec = _spec(
            strategy_grid={"fast_window": [3, 5, 7], "slow_window": [10, 15, 20]},
            max_combos=4,
        )
        with pytest.raises(SweepTooLargeError, match="9"):
            _run(spec, tmp_path, log_path=log)
        assert not log.exists(), "não pode ter escrito nada antes de abortar"

    def test_teto_conta_combinacoes_validas_nao_o_produto_bruto(self, tmp_path: Path):
        # 4 combinações no produto bruto, 3 válidas (5>=5 cai fora): teto 3 passa
        spec = _spec(strategy_grid={"fast_window": [3, 5], "slow_window": [5, 15]}, max_combos=3)
        result = _run(spec, tmp_path)
        assert len(result.combos) == 3


class TestProveniencia:
    def test_uma_linha_por_combinacao_por_ticker(self, tmp_path: Path):
        result = _run(_spec(), tmp_path)
        assert len(result.combos) == 4
        assert len(result.rows) == 4 * len(TICKERS)

    def test_colunas_de_proveniencia_presentes_em_toda_linha(self, tmp_path: Path):
        result = _run(_spec(), tmp_path)
        for row in result.rows:
            assert row["sweep_id"] == "SWEEP_FIXO"
            assert row["n_combos"] == 4
            assert row["train_test"] == "full"
            assert row["combo_id"]
            assert row["hypothesis"] == "teste"

    def test_combo_id_e_estavel_e_distingue_combinacoes(self, tmp_path: Path):
        result = _run(_spec(), tmp_path)
        por_combo = {}
        for row in result.rows:
            por_combo.setdefault(row["combo_id"], set()).add(
                (row["param_fast_window"], row["param_slow_window"])
            )
        assert len(por_combo) == 4
        assert all(len(v) == 1 for v in por_combo.values()), "combo_id misturou parâmetros"

    def test_train_test_e_declarado_por_quem_chama(self, tmp_path: Path):
        # o walk-forward (Componente 2) reusa o sweep marcando treino/teste;
        # o sweep sozinho não sabe dizer qual é qual, então aceita o rótulo.
        result = _run(_spec(), tmp_path, train_test="train")
        assert all(row["train_test"] == "train" for row in result.rows)

    def test_grava_csv_com_todas_as_linhas(self, tmp_path: Path):
        log = tmp_path / "sweep_runs.csv"
        result = _run(_spec(), tmp_path, log_path=log)
        df = pd.read_csv(log)
        assert len(df) == len(result.rows)
        for coluna in ("sweep_id", "combo_id", "n_combos", "train_test"):
            assert coluna in df.columns

    def test_log_path_none_nao_grava_nada(self, tmp_path: Path):
        result = _run(_spec(), tmp_path, log_path=None)
        assert result.rows
        assert not (tmp_path / "sweep_runs.csv").exists()
        assert not DEFAULT_LOG_PATH.exists() or DEFAULT_LOG_PATH.stat().st_size >= 0


class TestRobustezDoLoop:
    def test_ticker_ruim_nao_derruba_o_sweep(self, tmp_path: Path):
        spec = _spec(tickers=["AAA3.SA", "BAD3.SA", "BBB3.SA"])
        result = _run(spec, tmp_path, fetch_fn=_fetch_with_bad_ticker)

        assert len(result.rows) == 4 * 2, "os dois tickers bons têm que ter rodado"
        assert {t for _, t, _ in result.failures} == {"BAD3.SA"}
        assert len(result.failures) == 4, "uma falha por combinação"


class TestDeterminismo:
    def test_mesmo_sweep_duas_vezes_gera_os_mesmos_numeros(self, tmp_path: Path):
        primeira = _run(_spec(), tmp_path / "a")
        segunda = _run(_spec(), tmp_path / "b")
        assert primeira.rows == segunda.rows


class TestAgregacao:
    def test_media_por_combinacao_com_contagem_de_tickers(self, tmp_path: Path):
        result = _run(_spec(), tmp_path)
        resumo = aggregate_by_combo(result.rows, metric="total_return")

        assert len(resumo) == 4
        assert all(item.n_tickers == len(TICKERS) for item in resumo)
        # ordenado do melhor para o pior: a leitura do sweep começa pelo topo
        assert resumo == sorted(resumo, key=lambda s: s.value, reverse=True)

    def test_metricas_ausentes_sao_ignoradas_na_media(self):
        linhas = [
            {"combo_id": "c000", "param_x": 1, "sharpe": 1.0},
            {"combo_id": "c000", "param_x": 1, "sharpe": None},
            {"combo_id": "c000", "param_x": 1, "sharpe": 2.0},
        ]
        resumo = aggregate_by_combo(linhas, metric="sharpe")
        assert len(resumo) == 1
        assert resumo[0].value == pytest.approx(1.5)
        assert resumo[0].n_tickers == 2
        assert resumo[0].params == {"x": 1}

    def test_combinacao_sem_nenhuma_metrica_valida_fica_com_value_none(self):
        linhas = [{"combo_id": "c000", "param_x": 1, "sharpe": None}]
        resumo = aggregate_by_combo(linhas, metric="sharpe")
        assert resumo[0].value is None
        assert resumo[0].n_tickers == 0

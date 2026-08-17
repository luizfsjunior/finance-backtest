"""Contrato do Componente 3 do laboratório (SPEC_LAB.md + SPEC_ROBUSTNESS.md):
robustez a perturbação.

A pergunta é se a tese aguenta mexer no setup ou só funciona na configuração
exata em que foi achada. O teste-armadilha central deste módulo é o
`TestConfiguracaoNaoMuda`: se alguma perturbação alterar os parâmetros da
estratégia ou do stop, o módulo estaria comparando setups diferentes e chamando
isso de robustez — o equivalente aqui ao vazamento treino/teste do Componente 2.

O segundo contrato importante é o veredicto (`TestVeredicto`): ele é severo de
propósito e olha a PIOR perturbação, não a média. Uma tese cuja média continua
boa porque uma única perturbação desabou é exatamente o que o componente existe
para pegar.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import sweep as sweep_module
from metrics import MetricsReport
from robustness import (
    RobustnessError,
    RobustnessSpec,
    build_perturbations,
    run_robustness,
    summarize,
)
from stops import AtrStop
from strategy import MovingAverageCrossover
from sweep import SweepSpec

TICKERS = ["AAA3.SA", "BBB3.SA", "CCC3.SA"]


def _sweep_spec(**overrides) -> SweepSpec:
    """Config ÚNICA: grade de um valor por parâmetro, como a spec exige (D1)."""
    base = dict(
        strategy_class=MovingAverageCrossover,
        strategy_grid={"fast_window": [5], "slow_window": [15]},
        tickers=TICKERS,
        start=date(2018, 1, 1),
        end=date(2021, 1, 1),
        stop_class=AtrStop,
        stop_grid={"period": [14], "multiplier": [2.0]},
        brokerage=5.0,
        slippage_bps=10.0,
        min_median_turnover=0.0,
        hypothesis="teste",
    )
    base.update(overrides)
    return SweepSpec(**base)


def _spec(**overrides) -> RobustnessSpec:
    base = dict(base=_sweep_spec(), select_by="sharpe")
    base.update(overrides)
    return RobustnessSpec(**base)


def _prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end)
    fase = (ord(ticker[0]) % 5) * 0.7
    close = pd.Series(
        [20.0 + 6.0 * math.sin(2 * math.pi * i / 60 + fase) for i in range(len(idx))],
        index=idx,
    )
    df = pd.DataFrame(index=idx, dtype=float)
    df["Open"] = close
    df["High"] = close + 0.5
    df["Low"] = close - 0.5
    df["Close"] = close
    df["Volume"] = 1_000_000.0
    return df


def _fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return _prices(ticker, start, end)


def _report(sharpe: float) -> MetricsReport:
    return MetricsReport(
        total_return=0.1,
        annualized_return=0.1,
        max_drawdown=-0.1,
        sharpe=sharpe,
        sortino=sharpe,
        num_trades=1,
        win_rate=1.0,
        avg_payoff=1.0,
        expectancy=1.0,
        avg_holding_days=1.0,
        benchmark_total_return=0.0,
        benchmark_annualized_return=0.0,
        benchmark_max_drawdown=-0.1,
        benchmark_sharpe=0.0,
        benchmark_sortino=0.0,
    )


def _run(tmp_path: Path, **kwargs):
    kwargs.setdefault("spec", _spec())
    kwargs.setdefault("log_path", None)
    kwargs.setdefault("cache_dir", tmp_path / "cache")
    kwargs.setdefault("fetch_fn", _fetch)
    kwargs.setdefault("run_id", "RB_FIXO")
    kwargs.setdefault("verbose", False)
    return run_robustness(kwargs.pop("spec"), **kwargs)


class TestConfiguracaoUnica:
    def test_grade_com_mais_de_uma_combinacao_e_erro(self):
        # perturbar uma grade seria escolher uma vencedora escondida; a config
        # tem que estar declarada (D1)
        spec = _spec(base=_sweep_spec(strategy_grid={"fast_window": [5, 9], "slow_window": [15]}))
        with pytest.raises(RobustnessError, match="UMA configuração"):
            build_perturbations(spec)

    def test_grade_sem_combinacao_valida_e_erro(self):
        spec = _spec(base=_sweep_spec(strategy_grid={"fast_window": [20], "slow_window": [10]}))
        with pytest.raises(RobustnessError):
            build_perturbations(spec)


class TestEixos:
    def test_baseline_e_o_primeiro_e_nao_altera_nada(self):
        spec = _spec()
        perturbacoes = build_perturbations(spec)

        assert perturbacoes[0].kind == "baseline"
        base = perturbacoes[0].spec
        assert list(base.tickers) == TICKERS
        assert (base.start, base.end) == (spec.base.start, spec.base.end)
        assert (base.brokerage, base.slippage_bps) == (5.0, 10.0)

    def test_leave_one_out_remove_um_ticker_por_execucao(self):
        perturbacoes = [p for p in build_perturbations(_spec()) if p.kind == "leave_one_out"]

        assert len(perturbacoes) == len(TICKERS)
        removidos = []
        for p in perturbacoes:
            faltando = set(TICKERS) - set(p.spec.tickers)
            assert len(faltando) == 1, "leave-one-out tem que remover exatamente um"
            removido = faltando.pop()
            assert removido in p.value
            removidos.append(removido)
        assert sorted(removidos) == sorted(TICKERS)  # cada papel sai uma vez

    def test_leave_one_out_exige_pelo_menos_tres_tickers(self):
        # tirar um de dois deixa um universo de um papel: outra pergunta
        spec = _spec(base=_sweep_spec(tickers=["AAA3.SA", "BBB3.SA"]))
        with pytest.raises(RobustnessError, match="3"):
            build_perturbations(spec)

    def test_deslocamento_mexe_so_no_inicio(self):
        spec = _spec(start_shift_months=(-3, 1))
        perturbacoes = [p for p in build_perturbations(spec) if p.kind == "start_shift"]

        assert [p.value for p in perturbacoes] == ["-3m", "+1m"]
        assert [p.spec.start for p in perturbacoes] == [date(2017, 10, 1), date(2018, 2, 1)]
        assert {p.spec.end for p in perturbacoes} == {spec.base.end}

    def test_deslocamento_de_zero_meses_e_erro(self):
        with pytest.raises(RobustnessError, match="0"):
            build_perturbations(_spec(start_shift_months=(0,)))

    def test_custo_multiplica_corretagem_e_slippage_juntos(self):
        spec = _spec(cost_multipliers=(0.0, 2.0))
        perturbacoes = [p for p in build_perturbations(spec) if p.kind == "cost_multiplier"]

        assert [(p.spec.brokerage, p.spec.slippage_bps) for p in perturbacoes] == [
            (0.0, 0.0),
            (10.0, 20.0),
        ]

    def test_subperiodos_sao_contiguos_e_cobrem_o_periodo(self):
        spec = _spec(subperiods=3)
        blocos = [p for p in build_perturbations(spec) if p.kind == "subperiod"]

        assert len(blocos) == 3
        assert blocos[0].spec.start == spec.base.start
        assert blocos[-1].spec.end == spec.base.end
        for anterior, seguinte in zip(blocos, blocos[1:]):
            assert seguinte.spec.start == anterior.spec.end + timedelta(days=1)
        assert [p.value for p in blocos] == ["1/3", "2/3", "3/3"]

    def test_eixo_desligado_nao_gera_perturbacao(self):
        spec = _spec(
            leave_one_out=False, start_shift_months=(), cost_multipliers=(), subperiods=0
        )
        perturbacoes = build_perturbations(spec)
        assert [p.kind for p in perturbacoes] == ["baseline"]

    def test_ordem_e_deterministica(self):
        primeira = [(p.kind, p.value) for p in build_perturbations(_spec())]
        segunda = [(p.kind, p.value) for p in build_perturbations(_spec())]
        assert primeira == segunda
        assert [k for k, _ in primeira[:2]] == ["baseline", "leave_one_out"]

    def test_teto_de_perturbacoes(self):
        with pytest.raises(RobustnessError, match="teto"):
            build_perturbations(_spec(max_perturbations=3))


class TestConfiguracaoNaoMuda:
    """O teste-armadilha: nenhuma perturbação pode mexer no setup em si."""

    def test_nenhum_eixo_altera_estrategia_ou_stop(self):
        for p in build_perturbations(_spec()):
            assert p.spec.strategy_class is MovingAverageCrossover
            assert p.spec.stop_class is AtrStop
            assert dict(p.spec.strategy_grid) == {"fast_window": [5], "slow_window": [15]}
            assert dict(p.spec.stop_grid) == {"period": [14], "multiplier": [2.0]}

    def test_todas_as_execucoes_usam_os_mesmos_parametros(self, tmp_path, monkeypatch):
        """Ângulo de comportamento: observa o que chega ao motor, não à spec."""
        vistos: set[tuple] = set()
        real_run = sweep_module.run

        def _spiao(**kwargs):
            estrategia, stop = kwargs["strategy"], kwargs["stop_loss"]
            vistos.add(
                (
                    estrategia.fast_window,
                    estrategia.slow_window,
                    stop.period,
                    stop.multiplier,
                )
            )
            return real_run(**kwargs)

        monkeypatch.setattr(sweep_module, "run", _spiao)
        _run(tmp_path)

        assert vistos == {(5, 15, 14, 2.0)}, "a perturbação mudou o setup, não o ambiente"

    def test_todas_as_linhas_do_csv_carregam_a_mesma_config(self, tmp_path):
        log = tmp_path / "rb.csv"
        _run(tmp_path, log_path=log)
        df = pd.read_csv(log)

        for coluna, esperado in [
            ("param_fast_window", 5),
            ("param_slow_window", 15),
            ("param_multiplier", 2.0),
        ]:
            assert set(df[coluna]) == {esperado}
        assert set(df["n_combos"]) == {1}


class TestExecucao:
    def test_uma_execucao_por_perturbacao_com_o_universo_certo(self, tmp_path: Path):
        resultado = _run(tmp_path)

        assert resultado.baseline.perturbation.kind == "baseline"
        assert len(resultado.baseline.rows) == len(TICKERS)
        loo = [p for p in resultado.perturbations if p.perturbation.kind == "leave_one_out"]
        for p in loo:
            assert len(p.rows) == len(TICKERS) - 1

    def test_proveniencia_nas_linhas(self, tmp_path: Path):
        resultado = _run(tmp_path)
        n = len(resultado.perturbations)

        for row in resultado.rows:
            assert row["run_id"] == "RB_FIXO"
            assert row["n_perturbations"] == n
            assert row["train_test"] == "perturbation"
            assert row["perturbation_kind"]

        tipos = {row["perturbation_kind"] for row in resultado.rows}
        assert tipos == {
            "baseline",
            "leave_one_out",
            "start_shift",
            "cost_multiplier",
            "subperiod",
        }

    def test_n_perturbations_nao_conta_o_baseline(self, tmp_path: Path):
        resultado = _run(tmp_path)
        assert len(resultado.perturbations) == len(resultado.all_perturbations) - 1

    def test_grava_csv(self, tmp_path: Path):
        log = tmp_path / "rb.csv"
        resultado = _run(tmp_path, log_path=log)
        df = pd.read_csv(log)

        assert len(df) == len(resultado.rows)
        for coluna in ("run_id", "perturbation_kind", "perturbation_value", "n_perturbations"):
            assert coluna in df.columns

    def test_ticker_que_falha_nao_derruba_a_rodada(self, tmp_path: Path):
        def _fetch_parcial(ticker: str, start: date, end: date) -> pd.DataFrame:
            if ticker == "CCC3.SA":
                raise RuntimeError("sem dados")
            return _prices(ticker, start, end)

        resultado = _run(tmp_path, fetch_fn=_fetch_parcial)
        assert resultado.failures
        assert resultado.rows


class TestVeredicto:
    def test_baseline_nao_positivo_e_na(self):
        assert summarize(baseline=0.0, values=[1.0, 1.0]).verdict == "N/A"
        assert summarize(baseline=-0.5, values=[1.0]).verdict == "N/A"
        assert summarize(baseline=None, values=[1.0]).verdict == "N/A"

    def test_degradacao_suave_e_robusta(self):
        resumo = summarize(baseline=1.0, values=[0.9, 0.8, 1.1, 0.7, 0.95])
        assert resumo.survival_rate == 1.0
        assert resumo.median_retention == pytest.approx(0.9)
        assert resumo.verdict == "ROBUSTA"

    def test_uma_unica_perturbacao_que_desaba_derruba_o_veredicto(self):
        # a média continuaria boa; é justamente o caso que o componente existe
        # para pegar (o papel sortudo no universo)
        resumo = summarize(baseline=1.0, values=[1.0, 1.1, 0.9, 1.0, -0.4])
        assert resumo.worst == pytest.approx(-0.4)
        assert resumo.survival_rate == pytest.approx(0.8)
        assert resumo.verdict == "FRÁGIL"

    def test_todas_positivas_mas_muito_abaixo_do_baseline_e_fragil(self):
        resumo = summarize(baseline=1.0, values=[0.2, 0.3, 0.25])
        assert resumo.survival_rate == 1.0
        assert resumo.median_retention == pytest.approx(0.25)
        assert resumo.verdict == "FRÁGIL"

    def test_dispersao_reportada_e_nao_so_a_media(self):
        resumo = summarize(baseline=1.0, values=[0.2, 0.5, 0.8, 1.0])
        assert resumo.worst == pytest.approx(0.2)
        assert resumo.best == pytest.approx(1.0)
        assert resumo.p25 is not None and resumo.p75 is not None
        assert resumo.worst <= resumo.p25 <= resumo.median <= resumo.p75 <= resumo.best

    def test_perturbacao_sem_metrica_valida_e_ignorada_na_distribuicao(self):
        resumo = summarize(baseline=1.0, values=[0.9, None, 0.8])
        assert resumo.n_valid == 2
        assert resumo.worst == pytest.approx(0.8)

    def test_resumo_do_resultado_usa_a_metrica_declarada(self, tmp_path, monkeypatch):
        # Sharpe alto e retorno baixo: se o resumo usasse outra métrica, o
        # veredicto sairia diferente
        monkeypatch.setattr(
            sweep_module, "run", lambda **kwargs: (_report(sharpe=2.0), None, None)
        )
        resultado = _run(tmp_path)
        resumo = resultado.summary

        assert resumo.baseline == pytest.approx(2.0)
        assert resumo.verdict == "ROBUSTA"


class TestEstavelNaoESuperior:
    """ROBUSTA responde 'depende do ambiente exato?', nunca 'a tese presta?'.

    A primeira execução real deste componente devolveu ROBUSTA para uma tese já
    refutada — Sharpe baixo, positivo e estável passa nos dois limiares. O
    contador de vitórias sobre o buy-and-hold existe para essa leitura não se
    perder (D4.1 do SPEC_ROBUSTNESS.md).
    """

    def test_conta_execucoes_que_superam_o_buy_and_hold(self, tmp_path, monkeypatch):
        def _fake_run(**kwargs):
            report = _report(sharpe=0.4)
            return replace(report, benchmark_sharpe=0.9), None, None

        monkeypatch.setattr(sweep_module, "run", _fake_run)
        resultado = _run(tmp_path)

        batem, comparaveis = resultado.beats_benchmark
        assert comparaveis == len(resultado.all_perturbations)
        assert batem == 0, "estratégia abaixo do benchmark não pode contar vitória"
        # e o veredicto continua ROBUSTA: estabilidade e superioridade são
        # perguntas diferentes, e o componente só responde a primeira
        assert resultado.summary.verdict == "ROBUSTA"

    def test_metrica_sem_benchmark_correspondente_nao_e_comparada(self, tmp_path):
        resultado = _run(tmp_path, spec=_spec(select_by="num_trades"))
        assert resultado.beats_benchmark == (0, 0)


class TestDeterminismo:
    def test_mesma_rodada_duas_vezes_gera_os_mesmos_numeros(self, tmp_path: Path):
        primeira = _run(tmp_path, cache_dir=tmp_path / "a")
        segunda = _run(tmp_path, cache_dir=tmp_path / "b")
        assert primeira.rows == segunda.rows
        assert primeira.summary == segunda.summary

    def test_cache_e_aquecido_com_o_range_mais_amplo_antes_de_executar(self, tmp_path: Path):
        """Sem isso a rodada não é determinística — ver `_warm_cache`.

        O cache de data.py é reescrito quando o range pedido não está coberto,
        e as perturbações pedem ranges diferentes de propósito. Se a primeira
        busca não cobrir todas elas, o cache é refeito no meio da rodada e a
        ordem das perturbações passa a influenciar os números.
        """
        pedidos: list[tuple[str, date, date]] = []

        def _fetch_espiao(ticker: str, start: date, end: date) -> pd.DataFrame:
            pedidos.append((ticker, start, end))
            return _prices(ticker, start, end)

        spec = _spec(start_shift_months=(-3, 3), subperiods=3)
        _run(tmp_path, spec=spec, fetch_fn=_fetch_espiao)

        # as primeiras buscas são o aquecimento: um range por ticker, cobrindo
        # o deslocamento mais para trás e o fim mais adiante
        aquecimento = pedidos[: len(TICKERS)]
        assert sorted(t for t, _s, _e in aquecimento) == sorted(TICKERS)
        for _ticker, start, end in aquecimento:
            assert start == date(2017, 10, 1)  # base 2018-01-01 menos 3 meses
            assert end == spec.base.end

"""Contrato do Componente 2 do laboratório (SPEC_LAB.md): walk-forward + WFE.

Este é o obstáculo que separa "gostei do resultado" de "testei o resultado", e
o módulo com o bug mais perigoso do laboratório inteiro: se a escolha da
combinação em algum ponto enxergar o bloco de teste, o WFE fica lindo e
mentiroso. A spec pede um teste automatizado que detecte isso — são os dois de
`TestAntiVazamento`, por ângulos independentes:

1. Nenhuma execução da fase de treino recebe dados além do fim do treino
   (observação direta das datas pedidas).
2. Trocar o futuro não muda a escolha feita no passado (observação do
   comportamento: se a escolha mudar, ela dependia do futuro).

O resto fixa a geometria das janelas (expanding), a regra de leitura do WFE e
a proveniência das linhas gravadas.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import sweep as sweep_module
from metrics import MetricsReport
from stops import NoStop
from strategy import MovingAverageCrossover
from sweep import SweepSpec
from walkforward import (
    WalkForwardError,
    WalkForwardSpec,
    make_windows,
    run_walk_forward,
    walk_forward_efficiency,
)

TICKERS = ["AAA3.SA", "BBB3.SA"]


def _sweep_spec(**overrides) -> SweepSpec:
    base = dict(
        strategy_class=MovingAverageCrossover,
        strategy_grid={"fast_window": [3, 5], "slow_window": [10, 15]},
        tickers=TICKERS,
        start=date(2016, 1, 1),
        end=date(2021, 1, 1),
        stop_class=NoStop,
        stop_grid={},
        min_median_turnover=0.0,
        hypothesis="teste",
    )
    base.update(overrides)
    return SweepSpec(**base)


def _wf_spec(**overrides) -> WalkForwardSpec:
    base = dict(sweep=_sweep_spec(), train_years=2, test_years=1, select_by="sharpe")
    base.update(overrides)
    return WalkForwardSpec(**base)


def _prices(ticker: str, start: date, end: date, virada: date | None = None) -> pd.DataFrame:
    """Onda determinística; opcionalmente com o comportamento alterado a partir
    de `virada` (usado para perturbar só o futuro)."""
    idx = pd.bdate_range(start=start, end=end)
    fase = (ord(ticker[0]) % 5) * 0.7
    valores = []
    for i, dia in enumerate(idx):
        base = 20.0 + 6.0 * math.sin(2 * math.pi * i / 60 + fase)
        if virada is not None and dia.date() >= virada:
            base = 20.0 + 6.0 * math.sin(2 * math.pi * i / 13 + fase)  # outro regime
        valores.append(base)
    close = pd.Series(valores, index=idx)

    df = pd.DataFrame(index=idx, dtype=float)
    df["Open"] = close
    df["High"] = close + 0.5
    df["Low"] = close - 0.5
    df["Close"] = close
    df["Volume"] = 1_000_000.0
    return df


def _fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return _prices(ticker, start, end)


class TestJanelas:
    def test_expanding_gera_janelas_contiguas_sem_sobreposicao(self):
        janelas = make_windows(date(2015, 1, 1), date(2024, 1, 1), train_years=3, test_years=1)

        assert len(janelas) == 6
        primeira = janelas[0]
        assert primeira.train_start == date(2015, 1, 1)
        assert primeira.train_end == date(2017, 12, 31)
        assert primeira.test_start == date(2018, 1, 1)
        assert primeira.test_end == date(2018, 12, 31)

        for anterior, seguinte in zip(janelas, janelas[1:]):
            # bloco de teste vira treino da janela seguinte, sem buraco nem
            # sobreposição: cada período é out-of-sample exatamente uma vez
            assert seguinte.test_start == anterior.test_end + pd.Timedelta(days=1)
            assert seguinte.train_end == anterior.test_end

    def test_treino_cresce_e_a_origem_fica_fixa(self):
        janelas = make_windows(date(2015, 1, 1), date(2024, 1, 1), train_years=3, test_years=1)
        assert {j.train_start for j in janelas} == {date(2015, 1, 1)}
        duracoes = [(j.train_end - j.train_start).days for j in janelas]
        assert duracoes == sorted(duracoes) and duracoes[0] < duracoes[-1]

    def test_janela_de_teste_incompleta_e_descartada(self):
        # 2015→2019-06 com treino 3 + teste 1: o último bloco de teste (2019)
        # não cabe inteiro. Incluí-lo anualizaria meio ano como se fosse um.
        janelas = make_windows(date(2015, 1, 1), date(2019, 6, 30), train_years=3, test_years=1)
        assert len(janelas) == 1
        assert janelas[-1].test_end == date(2018, 12, 31)

    def test_historico_curto_demais_e_erro_explicito(self):
        with pytest.raises(WalkForwardError, match="nenhuma janela"):
            make_windows(date(2015, 1, 1), date(2017, 1, 1), train_years=3, test_years=1)

    def test_rolling_ainda_nao_implementado(self):
        # a spec manda começar por um esquema só; aceitar 'rolling' e rodar
        # expanding devolveria um resultado com o nome errado
        with pytest.raises(WalkForwardError, match="rolling"):
            make_windows(
                date(2015, 1, 1), date(2024, 1, 1), train_years=3, test_years=1, scheme="rolling"
            )


class TestWFE:
    def test_razao_out_of_sample_sobre_in_sample(self):
        assert walk_forward_efficiency(in_sample=0.20, out_of_sample=0.10) == pytest.approx(0.5)

    def test_in_sample_nao_positivo_devolve_none(self):
        # WFE é razão: com denominador <= 0 o número existe mas não significa
        # nada ("manteve 200% da performance" de um treino que perdeu dinheiro)
        assert walk_forward_efficiency(in_sample=0.0, out_of_sample=0.1) is None
        assert walk_forward_efficiency(in_sample=-0.1, out_of_sample=-0.2) is None

    def test_out_of_sample_negativo_e_wfe_negativa_nao_none(self):
        # perdeu dinheiro fora da amostra é informação legítima, não ausência
        assert walk_forward_efficiency(in_sample=0.2, out_of_sample=-0.1) == pytest.approx(-0.5)


class TestExecucao:
    def _run(self, tmp_path: Path, **kwargs):
        kwargs.setdefault("spec", _wf_spec())
        kwargs.setdefault("log_path", tmp_path / "wf.csv")
        kwargs.setdefault("cache_dir", tmp_path / "cache")
        kwargs.setdefault("fetch_fn", _fetch)
        kwargs.setdefault("run_id", "WF_FIXO")
        kwargs.setdefault("verbose", False)
        spec = kwargs.pop("spec")
        return run_walk_forward(spec, **kwargs)

    def test_uma_combinacao_escolhida_por_janela(self, tmp_path: Path):
        resultado = self._run(tmp_path)

        assert len(resultado.windows) == 3
        for janela in resultado.windows:
            assert janela.chosen.combo_id
            # o teste roda a combinação escolhida e só ela, em todos os tickers
            assert len(janela.test_rows) == len(TICKERS)
            assert {row["train_test"] for row in janela.test_rows} == {"test"}
            assert {row["train_test"] for row in janela.train_rows} == {"train"}

    def test_linhas_de_treino_cobrem_a_grade_inteira(self, tmp_path: Path):
        resultado = self._run(tmp_path)
        for janela in resultado.windows:
            assert len(janela.train_rows) == 4 * len(TICKERS)

    def test_proveniencia_de_janela_e_combinacao_escolhida(self, tmp_path: Path):
        resultado = self._run(tmp_path)
        for i, janela in enumerate(resultado.windows):
            for row in janela.train_rows + janela.test_rows:
                assert row["window"] == i
                assert row["chosen_combo_id"] == janela.chosen.combo_id
                assert row["run_id"] == "WF_FIXO"

    def test_grava_csv_com_treino_e_teste(self, tmp_path: Path):
        log = tmp_path / "wf.csv"
        resultado = self._run(tmp_path, log_path=log)
        df = pd.read_csv(log)
        assert len(df) == len(resultado.rows)
        assert set(df["train_test"]) == {"train", "test"}
        for coluna in ("run_id", "window", "chosen_combo_id"):
            assert coluna in df.columns

    def test_wfe_agregada_e_por_janela(self, tmp_path: Path):
        resultado = self._run(tmp_path)
        assert len(resultado.windows) == 3
        for janela in resultado.windows:
            assert janela.in_sample is not None
            assert janela.out_of_sample is not None
        # a agregada usa as médias, não a média das razões (denominador perto
        # de zero numa janela dominaria a média de razões)
        assert resultado.wfe == walk_forward_efficiency(
            in_sample=sum(j.in_sample for j in resultado.windows) / 3,
            out_of_sample=sum(j.out_of_sample for j in resultado.windows) / 3,
        )

    def test_treino_sem_metrica_valida_e_erro(self, tmp_path: Path):
        def _tudo_falha(ticker: str, start: date, end: date) -> pd.DataFrame:
            raise RuntimeError("sem dados")

        with pytest.raises(WalkForwardError, match="escolher"):
            self._run(tmp_path, fetch_fn=_tudo_falha)


class TestAntiVazamento:
    def test_nenhuma_execucao_de_treino_recebe_dados_do_teste(self, tmp_path, monkeypatch):
        """Ângulo 1: observa diretamente as datas pedidas em cada fase."""
        chamadas: list[tuple[str, date, date]] = []
        real_run = sweep_module.run

        def _spiao(**kwargs):
            chamadas.append((kwargs["ticker"], kwargs["start"], kwargs["end"]))
            return real_run(**kwargs)

        monkeypatch.setattr(sweep_module, "run", _spiao)

        spec = _wf_spec()
        resultado = run_walk_forward(
            spec,
            log_path=None,
            cache_dir=tmp_path / "cache",
            fetch_fn=_fetch,
            run_id="WF",
            verbose=False,
        )

        janelas = {j.window.index: j.window for j in resultado.windows}
        # reconstrói a fase de cada chamada pelas datas e confere os limites
        for _ticker, start, end in chamadas:
            fases = [
                w
                for w in janelas.values()
                if (start, end) in {(w.train_start, w.train_end), (w.test_start, w.test_end)}
            ]
            assert fases, f"execução fora de qualquer janela declarada: {start}..{end}"
            janela = fases[0]
            if (start, end) == (janela.train_start, janela.train_end):
                assert end < janela.test_start, "treino leu além do fim do treino"

    def test_mudar_o_futuro_nao_muda_a_escolha_feita_no_passado(self, tmp_path: Path):
        """Ângulo 2: se a escolha muda quando só o futuro muda, ela via o futuro.

        A perturbação começa no início do ÚLTIMO bloco de teste — perturbar
        antes disso mudaria legitimamente as escolhas seguintes, porque no
        esquema expanding o teste de uma janela vira treino da próxima.
        """
        spec = _wf_spec()
        janelas = make_windows(
            spec.sweep.start, spec.sweep.end, spec.train_years, spec.test_years
        )
        ultimo_teste = janelas[-1].test_start

        def _fetch_perturbado(ticker: str, start: date, end: date) -> pd.DataFrame:
            return _prices(ticker, start, end, virada=ultimo_teste)

        normal = run_walk_forward(
            spec, log_path=None, cache_dir=tmp_path / "a", fetch_fn=_fetch, verbose=False
        )
        perturbado = run_walk_forward(
            spec,
            log_path=None,
            cache_dir=tmp_path / "b",
            fetch_fn=_fetch_perturbado,
            verbose=False,
        )

        assert [j.chosen.combo_id for j in normal.windows] == [
            j.chosen.combo_id for j in perturbado.windows
        ], "a combinação escolhida mudou por causa de dados que ela não deveria ter visto"
        # sanidade da própria perturbação: o resultado out-of-sample mudou,
        # senão o teste passaria por não estar testando nada
        assert normal.windows[-1].out_of_sample != perturbado.windows[-1].out_of_sample


class TestDeterminismo:
    def test_mesmo_walk_forward_duas_vezes_gera_os_mesmos_numeros(self, tmp_path: Path):
        kwargs = dict(log_path=None, fetch_fn=_fetch, run_id="WF", verbose=False)
        primeira = run_walk_forward(_wf_spec(), cache_dir=tmp_path / "a", **kwargs)
        segunda = run_walk_forward(_wf_spec(), cache_dir=tmp_path / "b", **kwargs)
        assert primeira.rows == segunda.rows
        assert primeira.wfe == segunda.wfe


def _report(annualized: float, sharpe: float) -> MetricsReport:
    """MetricsReport mínimo para testes que não dependem de números reais."""
    return MetricsReport(
        total_return=annualized,
        annualized_return=annualized,
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


class TestSelecao:
    def test_escolhe_a_melhor_combinacao_pela_metrica_declarada(self, tmp_path, monkeypatch):
        # cada combinação recebe um Sharpe fixo em função de fast_window, para
        # a escolha ser verificável sem depender do backtest real
        def _fake_run(**kwargs):
            fast = kwargs["strategy"].fast_window
            return _report(annualized=0.1 * fast, sharpe=float(fast)), None, None

        monkeypatch.setattr(sweep_module, "run", _fake_run)

        resultado = run_walk_forward(
            _wf_spec(select_by="sharpe"),
            log_path=None,
            cache_dir=tmp_path,
            fetch_fn=_fetch,
            verbose=False,
        )
        for janela in resultado.windows:
            assert janela.chosen.params["fast_window"] == 5  # o maior Sharpe
            assert janela.chosen.value == pytest.approx(5.0)

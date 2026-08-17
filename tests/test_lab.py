"""Contrato do Componente 5 do laboratório (SPEC_LAB.md): config declarativo.

Um experimento é um documento — universo, período, estratégia, grade, métrica
de seleção — e documento se versiona como arquivo, não se digita em formulário.
O que estes testes fixam:

1. O YAML vira um `SweepSpec` fielmente (incluindo escalar onde cabe lista).
2. Erro de digitação NUNCA vira default silencioso: campo desconhecido, classe
   desconhecida e seção faltando levantam erro nomeando o problema.
3. Hipótese é obrigatória no arquivo (mais rígido que o CLI do batch, que só
   avisa) — o arquivo é o documento do experimento, e sem o "por quê" ele não
   documenta nada.
4. `walk_forward` declarado num laboratório que ainda não tem walk-forward é
   erro, não campo ignorado. Rodar um sweep simples e devolver resultado
   parecendo out-of-sample é o pior desfecho possível.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from batch import TICKERS
from lab import (
    Experiment,
    ExperimentError,
    load_experiment,
    make_sweep_id,
    run_experiment,
)
from stops import AtrStop, NoStop
from strategy import BollingerReversion, MovingAverageCrossover

YAML_MINIMO = """
experiment: mac_plateau
hypothesis: "o 9/21 era pico ou platô?"
universe: [AAA3.SA, BBB3.SA]
period: {start: 2024-01-02, end: 2024-12-30}
strategy:
  class: MovingAverageCrossover
  grid:
    fast_window: [3, 5]
    slow_window: [10, 15]
"""


# Mesmo experimento, período longo o bastante para caber janelas de
# walk-forward (o mínimo acima tem só um ano de histórico).
YAML_LONGO = YAML_MINIMO.replace(
    "period: {start: 2024-01-02, end: 2024-12-30}",
    "period: {start: 2016-01-01, end: 2021-01-01}",
)


# Mesmo experimento com a grade FIXA numa combinação só e universo de três
# papéis: é o que a seção `perturbations` exige (a robustez estressa uma config
# já escolhida, não uma grade).
YAML_FIXO = (
    YAML_LONGO.replace("fast_window: [3, 5]", "fast_window: 3")
    .replace("slow_window: [10, 15]", "slow_window: 10")
    .replace("universe: [AAA3.SA, BBB3.SA]", "universe: [AAA3.SA, BBB3.SA, CCC3.SA]")
)


def _write(tmp_path: Path, conteudo: str, nome: str = "exp.yaml") -> Path:
    caminho = tmp_path / nome
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def _oscillating(ticker: str, start: date, end: date) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end)
    close = pd.Series(
        [20.0 + 6.0 * math.sin(2 * math.pi * i / 60) for i in range(len(idx))], index=idx
    )
    df = pd.DataFrame(index=idx, dtype=float)
    df["Open"] = close
    df["High"] = close + 0.5
    df["Low"] = close - 0.5
    df["Close"] = close
    df["Volume"] = 1_000_000.0
    return df


class TestCarregarExperimento:
    def test_yaml_minimo_vira_sweep_spec(self, tmp_path: Path):
        exp = load_experiment(_write(tmp_path, YAML_MINIMO))

        assert isinstance(exp, Experiment)
        assert exp.name == "mac_plateau"
        assert exp.spec.strategy_class is MovingAverageCrossover
        assert exp.spec.strategy_grid == {"fast_window": [3, 5], "slow_window": [10, 15]}
        assert exp.spec.tickers == ["AAA3.SA", "BBB3.SA"]
        assert exp.spec.start == date(2024, 1, 2)
        assert exp.spec.end == date(2024, 12, 30)
        assert exp.spec.hypothesis == "o 9/21 era pico ou platô?"

    def test_sem_secao_stop_o_default_e_nostop(self, tmp_path: Path):
        exp = load_experiment(_write(tmp_path, YAML_MINIMO))
        assert exp.spec.stop_class is NoStop
        assert exp.spec.stop_grid == {}

    def test_secao_stop_com_grade_propria(self, tmp_path: Path):
        yaml = YAML_MINIMO + """
stop:
  class: AtrStop
  grid:
    period: [14]
    multiplier: [2.0, 2.5]
"""
        exp = load_experiment(_write(tmp_path, yaml))
        assert exp.spec.stop_class is AtrStop
        assert exp.spec.stop_grid == {"period": [14], "multiplier": [2.0, 2.5]}

    def test_escalar_na_grade_vira_lista_de_um(self, tmp_path: Path):
        # fixar um parâmetro e varrer outro é o caso mais comum de todos;
        # exigir `[14]` só para dizer "14" é atrito sem ganho
        yaml = YAML_MINIMO.replace("slow_window: [10, 15]", "slow_window: 10")
        exp = load_experiment(_write(tmp_path, yaml))
        assert exp.spec.strategy_grid["slow_window"] == [10]

    def test_universo_default_resolve_para_os_dez_tickers(self, tmp_path: Path):
        yaml = YAML_MINIMO.replace("universe: [AAA3.SA, BBB3.SA]", "universe: default")
        exp = load_experiment(_write(tmp_path, yaml))
        assert exp.spec.tickers == TICKERS

    def test_execution_sobrescreve_defaults(self, tmp_path: Path):
        yaml = YAML_MINIMO + """
execution:
  initial_capital: 50000
  risk_pct: 0.02
  slippage_bps: 10
  max_combos: 8
select_by: sortino
"""
        exp = load_experiment(_write(tmp_path, yaml))
        assert exp.spec.initial_capital == 50_000
        assert exp.spec.risk_pct == 0.02
        assert exp.spec.slippage_bps == 10
        assert exp.spec.max_combos == 8
        assert exp.select_by == "sortino"

    def test_select_by_default_e_sharpe(self, tmp_path: Path):
        assert load_experiment(_write(tmp_path, YAML_MINIMO)).select_by == "sharpe"

    def test_outra_estrategia_do_registro(self, tmp_path: Path):
        yaml = """
experiment: boll
hypothesis: "reversão em janela curta"
universe: [AAA3.SA]
period: {start: 2024-01-02, end: 2024-12-30}
strategy:
  class: BollingerReversion
  grid: {window: [10, 20], k: [1.5, 2.0]}
"""
        exp = load_experiment(_write(tmp_path, yaml))
        assert exp.spec.strategy_class is BollingerReversion


class TestErroNuncaViraDefaultSilencioso:
    def test_campo_desconhecido_no_topo(self, tmp_path: Path):
        yaml = YAML_MINIMO + "\nstratgy: {class: MovingAverageCrossover}\n"
        with pytest.raises(ExperimentError, match="stratgy"):
            load_experiment(_write(tmp_path, yaml))

    def test_campo_desconhecido_dentro_de_secao(self, tmp_path: Path):
        yaml = YAML_MINIMO + "\nexecution: {risk_percent: 0.02}\n"
        with pytest.raises(ExperimentError, match="risk_percent"):
            load_experiment(_write(tmp_path, yaml))

    def test_classe_desconhecida_lista_as_disponiveis(self, tmp_path: Path):
        yaml = YAML_MINIMO.replace("MovingAverageCrossover", "MinhaEstrategiaMagica")
        with pytest.raises(ExperimentError) as exc:
            load_experiment(_write(tmp_path, yaml))
        assert "MinhaEstrategiaMagica" in str(exc.value)
        assert "MovingAverageCrossover" in str(exc.value)

    @pytest.mark.parametrize("secao", ["experiment", "universe", "period", "strategy"])
    def test_secao_obrigatoria_ausente(self, tmp_path: Path, secao: str):
        linhas = [l for l in YAML_MINIMO.splitlines() if not l.startswith(secao)]
        if secao == "strategy":  # remove também o bloco indentado da estratégia
            linhas = [l for l in linhas if not l.startswith((" ", "-"))]
        with pytest.raises(ExperimentError, match=secao):
            load_experiment(_write(tmp_path, "\n".join(linhas)))

    def test_hipotese_e_obrigatoria_no_arquivo(self, tmp_path: Path):
        yaml = "\n".join(l for l in YAML_MINIMO.splitlines() if not l.startswith("hypothesis"))
        with pytest.raises(ExperimentError, match="hypothesis"):
            load_experiment(_write(tmp_path, yaml))

    def test_periodo_invertido_e_erro(self, tmp_path: Path):
        yaml = YAML_MINIMO.replace("end: 2024-12-30", "end: 2023-12-30")
        with pytest.raises(ExperimentError, match="start"):
            load_experiment(_write(tmp_path, yaml))

    def test_walk_forward_sem_test_years(self, tmp_path: Path):
        yaml = YAML_LONGO + "\nwalk_forward: {scheme: expanding, train_years: 3}\n"
        with pytest.raises(ExperimentError, match="test_years"):
            load_experiment(_write(tmp_path, yaml))

    def test_walk_forward_com_campo_desconhecido(self, tmp_path: Path):
        yaml = YAML_LONGO + "\nwalk_forward: {train_years: 3, test_years: 1, janelas: 4}\n"
        with pytest.raises(ExperimentError, match="janelas"):
            load_experiment(_write(tmp_path, yaml))

    def test_walk_forward_com_scheme_nao_implementado(self, tmp_path: Path):
        yaml = YAML_LONGO + "\nwalk_forward: {scheme: rolling, train_years: 3, test_years: 1}\n"
        with pytest.raises(ExperimentError, match="rolling"):
            load_experiment(_write(tmp_path, yaml))

    def test_walk_forward_que_nao_cabe_no_historico_falha_no_carregamento(self, tmp_path: Path):
        # descobrir isso depois de metade do sweep rodado é desperdício puro
        yaml = YAML_MINIMO + "\nwalk_forward: {train_years: 3, test_years: 1}\n"
        with pytest.raises(ExperimentError, match="nenhuma janela"):
            load_experiment(_write(tmp_path, yaml))

    def test_perturbations_com_campo_desconhecido(self, tmp_path: Path):
        yaml = YAML_FIXO + "\nperturbations: {leave_one_out: false, terços: 3}\n"
        with pytest.raises(ExperimentError, match="terços"):
            load_experiment(_write(tmp_path, yaml))

    def test_perturbations_sobre_grade_com_varias_combinacoes_falha_no_carregamento(
        self, tmp_path: Path
    ):
        # a robustez estressa UMA config declarada; perturbar uma grade seria
        # escolher uma vencedora escondida (D1 do SPEC_ROBUSTNESS.md)
        yaml = YAML_MINIMO + "\nperturbations: {leave_one_out: false}\n"
        with pytest.raises(ExperimentError, match="UMA configuração"):
            load_experiment(_write(tmp_path, yaml))

    def test_walk_forward_e_perturbations_juntos_e_erro(self, tmp_path: Path):
        # etapas diferentes: escolher a config vs estressá-la. Misturadas, o
        # relatório não deixa dizer qual número veio de onde.
        yaml = (
            YAML_FIXO
            + "\nwalk_forward: {train_years: 2, test_years: 1}"
            + "\nperturbations: {leave_one_out: false}\n"
        )
        with pytest.raises(ExperimentError, match="perturbations"):
            load_experiment(_write(tmp_path, yaml))

    def test_arquivo_vazio(self, tmp_path: Path):
        with pytest.raises(ExperimentError):
            load_experiment(_write(tmp_path, ""))

    def test_arquivo_inexistente(self, tmp_path: Path):
        with pytest.raises(ExperimentError, match="não encontrado"):
            load_experiment(tmp_path / "nao_existe.yaml")


class TestProveniencia:
    def test_sweep_id_carrega_o_nome_do_experimento(self):
        # o CSV de resultados tem que dizer de qual arquivo de experimento
        # aquela linha saiu; sem isso, reproduzir dali a três meses é chute
        sweep_id = make_sweep_id("mac_plateau", when="2026-08-17T10:00:00")
        assert sweep_id.startswith("mac_plateau-")
        assert "2026-08-17T10:00:00" in sweep_id


class TestRodarExperimento:
    def test_ponta_a_ponta_grava_linhas_com_proveniencia(self, tmp_path: Path):
        exp = load_experiment(_write(tmp_path, YAML_MINIMO))
        log = tmp_path / "sweep_runs.csv"

        result = run_experiment(
            exp,
            log_path=log,
            cache_dir=tmp_path / "cache",
            fetch_fn=_oscillating,
            verbose=False,
        )

        assert len(result.combos) == 4
        assert len(result.rows) == 4 * 2
        assert result.sweep_id.startswith("mac_plateau-")
        assert all(row["hypothesis"] == "o 9/21 era pico ou platô?" for row in result.rows)
        assert log.exists()

    def test_walk_forward_declarado_vira_walk_forward_de_verdade(self, tmp_path: Path):
        yaml = YAML_LONGO + "\nwalk_forward: {train_years: 2, test_years: 1}\n"
        exp = load_experiment(_write(tmp_path, yaml))

        assert exp.walk_forward is not None
        assert exp.walk_forward.train_years == 2
        assert exp.walk_forward.select_by == exp.select_by  # herdado do topo

        result = run_experiment(
            exp,
            log_path=tmp_path / "wf.csv",
            cache_dir=tmp_path / "cache",
            fetch_fn=_oscillating,
            verbose=False,
        )
        # é um WalkForwardResult, não um SweepResult: tem janelas e WFE
        assert len(result.windows) == 3
        assert result.run_id.startswith("mac_plateau-")

    def test_sem_walk_forward_o_experimento_continua_sweep_simples(self, tmp_path: Path):
        exp = load_experiment(_write(tmp_path, YAML_MINIMO))
        assert exp.walk_forward is None

    def test_experimento_de_exemplo_do_repo_carrega(self):
        # o arquivo de exemplo é documentação executável: se ele quebra, a
        # primeira coisa que alguém tenta rodar no laboratório não funciona
        raiz = Path(__file__).resolve().parent.parent
        exp = load_experiment(raiz / "experiments" / "mac_plateau.yaml")
        assert exp.spec.strategy_class is MovingAverageCrossover
        assert exp.spec.hypothesis

    def test_experimento_de_walk_forward_do_repo_carrega(self):
        raiz = Path(__file__).resolve().parent.parent
        exp = load_experiment(raiz / "experiments" / "mac_walkforward.yaml")
        assert exp.walk_forward is not None
        assert exp.walk_forward.scheme == "expanding"

    def test_perturbations_declarado_vira_robustez_de_verdade(self, tmp_path: Path):
        yaml = YAML_FIXO + "\nperturbations: {start_shift_months: [-1], subperiods: 0}\n"
        exp = load_experiment(_write(tmp_path, yaml))

        assert exp.robustness is not None
        assert exp.robustness.select_by == exp.select_by  # herdado do topo

        result = run_experiment(
            exp,
            log_path=tmp_path / "rb.csv",
            cache_dir=tmp_path / "cache",
            fetch_fn=_oscillating,
            verbose=False,
        )
        # é um RobustnessResult: tem baseline, perturbações e veredicto
        assert result.baseline.perturbation.kind == "baseline"
        # 3 leave-one-out + 1 deslocamento + 2 custos (defaults), sem sub-períodos
        assert len(result.perturbations) == 6
        assert result.summary.verdict in {"ROBUSTA", "FRÁGIL", "N/A"}
        assert result.run_id.startswith("mac_plateau-")

    def test_experimento_de_robustez_do_repo_carrega(self):
        raiz = Path(__file__).resolve().parent.parent
        exp = load_experiment(raiz / "experiments" / "mac_robustness.yaml")
        assert exp.robustness is not None
        assert exp.walk_forward is None

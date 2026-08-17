"""Componente 5 do laboratório (SPEC_LAB.md): experimento como arquivo.

Pergunta que este módulo responde: **qual a forma de menor atrito para declarar
um experimento?** Resposta da spec: um arquivo de config declarativo, não um
app web.

O motivo é reprodutibilidade. Um experimento é universo + período + estratégia +
grade + métrica de seleção — isto é, um documento. Documento se versiona:
commitado junto do resultado, seis meses depois você recria a varredura exata.
Um formulário web perde isso, e o custo de construí-lo (backend, estado, deploy)
compete diretamente com o tempo de fazer perguntas.

YAML e não JSON por um motivo específico desta bancada: **comentários**. O
arquivo de experimento é onde se registra por que aquela grade e não outra, e
essa justificativa é metade do valor do documento.

Postura de erro deste módulo: **nada de default silencioso.** Campo com nome
errado, classe inexistente, seção faltando e hipótese ausente levantam erro
nomeando o problema. Num laboratório cujo produto é o veredicto, um typo que
vira default é pior do que um crash — o experimento roda, o resultado sai, e
ele responde a uma pergunta diferente da que você fez.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from batch import TICKERS
from data import DEFAULT_MIN_MEDIAN_TURNOVER, FetchFn
from stops import AtrStop, FixedPctStop, NoStop
from strategy import BollingerReversion, MovingAverageCrossover, TimeSeriesMomentum
from sweep import (
    DEFAULT_LOG_PATH,
    SweepResult,
    SweepSpec,
    aggregate_by_combo,
    print_distribution,
    run_sweep,
    valid_combos,
)

# Registro fechado: o YAML escolhe por nome dentro deste dicionário e nada mais.
# Resolver classe por `getattr` num módulo deixaria um arquivo de experimento
# instanciar qualquer coisa importável — e um nome errado passaria a depender do
# acaso de existir um atributo com aquele nome.
STRATEGIES: dict[str, type] = {
    "MovingAverageCrossover": MovingAverageCrossover,
    "BollingerReversion": BollingerReversion,
    "TimeSeriesMomentum": TimeSeriesMomentum,
}

STOPS: dict[str, type] = {
    "AtrStop": AtrStop,
    "FixedPctStop": FixedPctStop,
    "NoStop": NoStop,
}

TOP_LEVEL_FIELDS = {
    "experiment",
    "hypothesis",
    "universe",
    "period",
    "strategy",
    "stop",
    "execution",
    "select_by",
}

PERIOD_FIELDS = {"start", "end"}
COMPONENT_FIELDS = {"class", "grid"}
EXECUTION_FIELDS = {
    "initial_capital",
    "risk_pct",
    "brokerage",
    "slippage_bps",
    "min_turnover",
    "max_combos",
}


class ExperimentError(Exception):
    """Arquivo de experimento inválido."""


@dataclass(frozen=True)
class Experiment:
    """Um experimento declarado em arquivo, pronto para virar sweep."""

    name: str
    spec: SweepSpec
    select_by: str
    source: Path | None = None


def load_experiment(path: Path | str) -> Experiment:
    """Lê um arquivo YAML de experimento e devolve o `Experiment` equivalente."""
    path = Path(path)
    if not path.exists():
        raise ExperimentError(f"arquivo de experimento não encontrado: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExperimentError(f"{path}: YAML inválido — {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ExperimentError(f"{path}: arquivo vazio ou não é um mapeamento YAML")

    if "walk_forward" in raw:
        raise ExperimentError(
            "'walk_forward' ainda não é suportado (Componente 2 do SPEC_LAB não "
            "existe). Aceitar o campo e rodar um sweep simples devolveria um "
            "resultado in-sample com aparência de out-of-sample — o pior desfecho "
            "possível para esta bancada. Remova o campo ou implemente o Componente 2."
        )

    _reject_unknown(raw, TOP_LEVEL_FIELDS, secao="topo do arquivo")

    name = _require(raw, "experiment", str)
    hypothesis = _require(raw, "hypothesis", str)
    if not hypothesis.strip():
        raise ExperimentError(
            "'hypothesis' está vazia. O arquivo de experimento é o documento que "
            "explica por que esta varredura existe; sem isso o CSV é inútil em 3 meses."
        )

    start, end = _parse_period(raw)
    tickers = _parse_universe(raw)
    strategy_class, strategy_grid = _parse_component(raw, "strategy", STRATEGIES, required=True)
    stop_class, stop_grid = _parse_component(raw, "stop", STOPS, required=False)

    execution = raw.get("execution") or {}
    if not isinstance(execution, Mapping):
        raise ExperimentError("'execution' precisa ser um mapeamento de parâmetros")
    _reject_unknown(execution, EXECUTION_FIELDS, secao="execution")

    spec = SweepSpec(
        strategy_class=strategy_class,
        strategy_grid=strategy_grid,
        tickers=tickers,
        start=start,
        end=end,
        stop_class=stop_class or NoStop,
        stop_grid=stop_grid,
        hypothesis=hypothesis,
        **{
            campo: execution[campo]
            for campo in EXECUTION_FIELDS & set(execution)
            if campo != "min_turnover"
        },
        min_median_turnover=execution.get("min_turnover", DEFAULT_MIN_MEDIAN_TURNOVER),
    )

    return Experiment(
        name=name,
        spec=spec,
        select_by=str(raw.get("select_by", "sharpe")),
        source=path,
    )


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], secao: str) -> None:
    desconhecidos = sorted(set(mapping) - allowed)
    if desconhecidos:
        raise ExperimentError(
            f"campo(s) desconhecido(s) em {secao}: {', '.join(desconhecidos)}. "
            f"Aceitos: {', '.join(sorted(allowed))}."
        )


def _require(raw: Mapping[str, Any], campo: str, tipo: type) -> Any:
    if campo not in raw:
        raise ExperimentError(f"campo obrigatório ausente: '{campo}'")
    valor = raw[campo]
    if not isinstance(valor, tipo):
        raise ExperimentError(
            f"'{campo}' deveria ser {tipo.__name__}, veio {type(valor).__name__}"
        )
    return valor


def _parse_period(raw: Mapping[str, Any]) -> tuple[date, date]:
    period = _require(raw, "period", Mapping)
    _reject_unknown(period, PERIOD_FIELDS, secao="period")
    for campo in PERIOD_FIELDS:
        if campo not in period:
            raise ExperimentError(f"'period' precisa de '{campo}'")

    start, end = (_as_date(period[c], c) for c in ("start", "end"))
    if start >= end:
        raise ExperimentError(f"period.start ({start}) precisa ser anterior a period.end ({end})")
    return start, end


def _as_date(valor: Any, campo: str) -> date:
    # o YAML já converte `2024-01-02` em date; string continua aceita para
    # quem escreveu a data entre aspas
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    if isinstance(valor, str):
        try:
            return date.fromisoformat(valor)
        except ValueError as exc:
            raise ExperimentError(f"period.{campo}: data inválida '{valor}'") from exc
    raise ExperimentError(f"period.{campo}: esperava data, veio {type(valor).__name__}")


def _parse_universe(raw: Mapping[str, Any]) -> list[str]:
    if "universe" not in raw:
        raise ExperimentError("campo obrigatório ausente: 'universe'")

    universe = raw["universe"]
    if universe == "default":
        return list(TICKERS)
    if isinstance(universe, str):
        raise ExperimentError(
            f"'universe' deve ser uma lista de tickers ou a palavra 'default', veio '{universe}'"
        )
    if not universe:
        raise ExperimentError("'universe' está vazio")
    return [str(t) for t in universe]


def _parse_component(
    raw: Mapping[str, Any],
    secao: str,
    registro: dict[str, type],
    required: bool,
) -> tuple[type | None, dict[str, list[Any]]]:
    if secao not in raw:
        if required:
            raise ExperimentError(f"seção obrigatória ausente: '{secao}'")
        return None, {}

    bloco = raw[secao]
    if not isinstance(bloco, Mapping):
        raise ExperimentError(f"'{secao}' precisa ser um mapeamento com 'class' e 'grid'")
    _reject_unknown(bloco, COMPONENT_FIELDS, secao=secao)

    nome = bloco.get("class")
    if nome is None:
        raise ExperimentError(f"'{secao}' precisa declarar 'class'")
    if nome not in registro:
        raise ExperimentError(
            f"{secao}.class '{nome}' não existe. Disponíveis: {', '.join(sorted(registro))}."
        )

    grid = bloco.get("grid") or {}
    if not isinstance(grid, Mapping):
        raise ExperimentError(f"'{secao}.grid' precisa ser um mapeamento parâmetro -> valores")

    # escalar vira lista de um: fixar um parâmetro e varrer outro é o caso mais
    # comum, e exigir `[14]` para dizer "14" é atrito sem ganho nenhum
    return registro[nome], {
        str(param): list(valores) if isinstance(valores, (list, tuple)) else [valores]
        for param, valores in grid.items()
    }


def make_sweep_id(name: str, when: str | None = None) -> str:
    """`sweep_id` que carrega o nome do experimento.

    A linha do CSV precisa dizer de qual arquivo de experimento ela saiu — sem
    isso, reproduzir um resultado antigo vira arqueologia. O timestamp separa
    execuções repetidas do mesmo arquivo.
    """
    return f"{name}-{when or datetime.now().isoformat(timespec='seconds')}"


def run_experiment(
    experiment: Experiment,
    *,
    log_path: Path | None = DEFAULT_LOG_PATH,
    cache_dir: Path = Path("data_cache"),
    fetch_fn: FetchFn | None = None,
    verbose: bool = True,
    train_test: str = "full",
) -> SweepResult:
    """Roda o experimento carregado. É o sweep do Componente 1, sem novidade
    de execução — o que este módulo agrega é a declaração reproduzível."""
    return run_sweep(
        experiment.spec,
        sweep_id=make_sweep_id(experiment.name),
        train_test=train_test,
        log_path=log_path,
        cache_dir=cache_dir,
        fetch_fn=fetch_fn,
        verbose=verbose,
    )


def _print_plan(experiment: Experiment) -> None:
    """Mostra o que o experimento VAI rodar, sem rodar.

    Existe porque o custo de um sweep é linear em combinações × tickers e o
    arrependimento acontece depois de 20 minutos de execução, não antes.
    """
    combos = valid_combos(experiment.spec)
    spec = experiment.spec
    print(f"\n=== {experiment.name} ({experiment.source}) ===")
    print(f"hipótese: {spec.hypothesis}")
    print(f"período:  {spec.start} → {spec.end}")
    print(f"universo: {len(spec.tickers)} ticker(s)")
    print(f"setup:    {spec.strategy_class.__name__} + {spec.stop_class.__name__}")
    print(f"grade:    {len(combos)} combinação(ões) válida(s)")
    print(f"total:    {len(combos) * len(spec.tickers)} backtests")
    for combo in combos:
        print(f"  {combo.combo_id}  {combo.all_params}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laboratório: roda experimentos declarados em YAML",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Roda um experimento")
    run_cmd.add_argument("experiment", type=Path, help="Caminho do .yaml do experimento")
    run_cmd.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    run_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o plano (combinações e nº de backtests) sem executar",
    )
    run_cmd.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Sobrescreve o teto do arquivo — cada combinação é mais uma tentativa (obstáculo 4)",
    )
    return parser.parse_args()


def main() -> None:
    # mesma razão de main.py: o console default do Windows corrompe acentos
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    try:
        experiment = load_experiment(args.experiment)
    except ExperimentError as exc:
        raise SystemExit(f"Experimento inválido: {exc}")

    if args.max_combos is not None:
        experiment = replace(
            experiment, spec=replace(experiment.spec, max_combos=args.max_combos)
        )

    try:
        _print_plan(experiment)
        if args.dry_run:
            print("\n(dry-run: nada executado)")
            return
        result = run_experiment(experiment, log_path=args.log_path)
    except Exception as exc:
        raise SystemExit(f"Experimento abortado: {exc}")

    print_distribution(
        aggregate_by_combo(result.rows, experiment.select_by), experiment.select_by
    )
    if result.failures:
        print(f"\n{len(result.failures)} falha(s):")
        for combo_id, ticker, msg in result.failures[:20]:
            print(f"  {combo_id} {ticker}: {msg}")


if __name__ == "__main__":
    main()

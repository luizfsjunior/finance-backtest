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
from robustness import DEFAULT_LOG_PATH as RB_DEFAULT_LOG_PATH
from robustness import (
    DEFAULT_COST_MULTIPLIERS,
    DEFAULT_MAX_PERTURBATIONS,
    DEFAULT_START_SHIFT_MONTHS,
    DEFAULT_SUBPERIODS,
    RobustnessError,
    RobustnessResult,
    RobustnessSpec,
    build_perturbations,
    run_robustness,
)
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
from walkforward import DEFAULT_LOG_PATH as WF_DEFAULT_LOG_PATH
from walkforward import (
    WalkForwardError,
    WalkForwardResult,
    WalkForwardSpec,
    make_windows,
    run_walk_forward,
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
    "walk_forward",
    "perturbations",
}

PERIOD_FIELDS = {"start", "end"}
WALK_FORWARD_FIELDS = {"scheme", "train_years", "test_years"}
PERTURBATION_FIELDS = {
    "leave_one_out",
    "start_shift_months",
    "cost_multipliers",
    "subperiods",
    "max_perturbations",
}
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
    """Um experimento declarado em arquivo, pronto para virar sweep.

    Com a seção `walk_forward` declarada, o experimento vira walk-forward: o
    mesmo universo/grade, fatiado em janelas treino→teste. Com `perturbations`,
    vira teste de robustez sobre uma config única. Sem nenhuma das duas, é um
    sweep simples sobre o período inteiro (in-sample, e o relatório não finge o
    contrário).
    """

    name: str
    spec: SweepSpec
    select_by: str
    source: Path | None = None
    walk_forward: WalkForwardSpec | None = None
    robustness: RobustnessSpec | None = None


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

    select_by = str(raw.get("select_by", "sharpe"))
    if "walk_forward" in raw and "perturbations" in raw:
        raise ExperimentError(
            "'walk_forward' e 'perturbations' no mesmo arquivo: são etapas "
            "diferentes da mesma investigação (escolher a config vs estressá-la), "
            "e um relatório misturando as duas não deixa dizer qual número veio "
            "de onde. Rode o walk-forward, decida a config, declare-a num "
            "segundo arquivo com 'perturbations'."
        )

    return Experiment(
        name=name,
        spec=spec,
        select_by=select_by,
        source=path,
        walk_forward=_parse_walk_forward(raw, spec, select_by),
        robustness=_parse_perturbations(raw, spec, select_by),
    )


def _parse_walk_forward(
    raw: Mapping[str, Any], spec: SweepSpec, select_by: str
) -> WalkForwardSpec | None:
    if "walk_forward" not in raw:
        return None

    bloco = raw["walk_forward"]
    if not isinstance(bloco, Mapping):
        raise ExperimentError("'walk_forward' precisa ser um mapeamento")
    _reject_unknown(bloco, WALK_FORWARD_FIELDS, secao="walk_forward")

    for campo in ("train_years", "test_years"):
        if campo not in bloco:
            raise ExperimentError(f"'walk_forward' precisa de '{campo}'")
        if not isinstance(bloco[campo], int) or bloco[campo] < 1:
            raise ExperimentError(f"walk_forward.{campo} precisa ser inteiro >= 1")

    wf = WalkForwardSpec(
        sweep=spec,
        train_years=bloco["train_years"],
        test_years=bloco["test_years"],
        scheme=str(bloco.get("scheme", "expanding")),
        select_by=select_by,
    )
    # valida a geometria já no carregamento: descobrir que o histórico não
    # comporta nenhuma janela depois de metade do sweep rodado é desperdício
    try:
        make_windows(spec.start, spec.end, wf.train_years, wf.test_years, wf.scheme)
    except WalkForwardError as exc:
        raise ExperimentError(f"walk_forward inválido: {exc}") from exc
    return wf


def _parse_perturbations(
    raw: Mapping[str, Any], spec: SweepSpec, select_by: str
) -> RobustnessSpec | None:
    if "perturbations" not in raw:
        return None

    bloco = raw["perturbations"]
    if bloco is None:  # `perturbations:` sem corpo = todos os eixos default
        bloco = {}
    if not isinstance(bloco, Mapping):
        raise ExperimentError("'perturbations' precisa ser um mapeamento")
    _reject_unknown(bloco, PERTURBATION_FIELDS, secao="perturbations")

    rb = RobustnessSpec(
        base=spec,
        select_by=select_by,
        leave_one_out=bool(bloco.get("leave_one_out", True)),
        start_shift_months=tuple(
            int(m) for m in bloco.get("start_shift_months", DEFAULT_START_SHIFT_MONTHS)
        ),
        cost_multipliers=tuple(
            float(f) for f in bloco.get("cost_multipliers", DEFAULT_COST_MULTIPLIERS)
        ),
        subperiods=int(bloco.get("subperiods", DEFAULT_SUBPERIODS)),
        max_perturbations=int(bloco.get("max_perturbations", DEFAULT_MAX_PERTURBATIONS)),
    )
    # valida os eixos já no carregamento, pelo mesmo motivo do walk_forward:
    # descobrir que a grade tem duas combinações depois de meia hora de
    # execução é desperdício
    try:
        build_perturbations(rb)
    except RobustnessError as exc:
        raise ExperimentError(f"perturbations inválido: {exc}") from exc
    return rb


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


# Sentinela para "escolha o CSV default do tipo de execução". Não dá para usar
# None: ali None já significa "não grave nada".
_AUTO = object()


def run_experiment(
    experiment: Experiment,
    *,
    log_path: Path | None | Any = _AUTO,
    cache_dir: Path = Path("data_cache"),
    fetch_fn: FetchFn | None = None,
    verbose: bool = True,
    train_test: str = "full",
) -> SweepResult | WalkForwardResult | RobustnessResult:
    """Roda o experimento carregado — sweep, walk-forward ou robustez.

    Nenhuma novidade de execução mora aqui — o que este módulo agrega é a
    declaração reproduzível. Cada tipo grava no seu CSV (`sweep_runs.csv`,
    `walkforward_runs.csv`, `robustness_runs.csv`): as linhas têm proveniência e
    significados diferentes, e misturá-las faria a leitura do acervo depender de
    lembrar qual coluna estava preenchida.
    """
    if experiment.robustness is not None:
        return run_robustness(
            experiment.robustness,
            run_id=make_sweep_id(experiment.name),
            log_path=RB_DEFAULT_LOG_PATH if log_path is _AUTO else log_path,
            cache_dir=cache_dir,
            fetch_fn=fetch_fn,
            verbose=verbose,
        )

    if experiment.walk_forward is not None:
        return run_walk_forward(
            experiment.walk_forward,
            run_id=make_sweep_id(experiment.name),
            log_path=WF_DEFAULT_LOG_PATH if log_path is _AUTO else log_path,
            cache_dir=cache_dir,
            fetch_fn=fetch_fn,
            verbose=verbose,
        )

    return run_sweep(
        experiment.spec,
        sweep_id=make_sweep_id(experiment.name),
        train_test=train_test,
        log_path=DEFAULT_LOG_PATH if log_path is _AUTO else log_path,
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

    if experiment.robustness is not None:
        perturbacoes = build_perturbations(experiment.robustness)
        print(f"modo:     robustez, {len(perturbacoes) - 1} perturbação(ões) + baseline")
        print(f"total:    {sum(len(p.spec.tickers) for p in perturbacoes)} backtests")
        for p in perturbacoes:
            print(f"  {p.label}")
        return

    if experiment.walk_forward is None:
        print(f"modo:     sweep simples (tudo in-sample)")
        print(f"total:    {len(combos) * len(spec.tickers)} backtests")
    else:
        wf = experiment.walk_forward
        janelas = make_windows(spec.start, spec.end, wf.train_years, wf.test_years, wf.scheme)
        # por janela: a grade inteira no treino + a combinação escolhida no teste
        por_janela = (len(combos) + 1) * len(spec.tickers)
        print(f"modo:     walk-forward {wf.scheme}, {len(janelas)} janela(s)")
        print(f"          treino {wf.train_years}a → teste {wf.test_years}a, seleção por {wf.select_by}")
        print(f"total:    {por_janela * len(janelas)} backtests")
        for janela in janelas:
            print(
                f"  w{janela.index}  treino {janela.train_start}→{janela.train_end}"
                f"  | teste {janela.test_start}→{janela.test_end}"
            )

    for combo in combos:
        print(f"  {combo.combo_id}  {combo.all_params}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laboratório: roda experimentos declarados em YAML",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Roda um experimento")
    run_cmd.add_argument("experiment", type=Path, help="Caminho do .yaml do experimento")
    run_cmd.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Default: sweep_runs.csv, ou walkforward_runs.csv se o experimento "
        "declarar walk_forward",
    )
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

    log_path = args.log_path if args.log_path is not None else _AUTO
    try:
        _print_plan(experiment)
        if args.dry_run:
            print("\n(dry-run: nada executado)")
            return
        result = run_experiment(experiment, log_path=log_path)
    except Exception as exc:
        raise SystemExit(f"Experimento abortado: {exc}")

    if experiment.walk_forward is None and experiment.robustness is None:
        # os relatórios do walk-forward e da robustez já saem durante a
        # execução; só o sweep simples precisa do fechamento aqui
        print_distribution(
            aggregate_by_combo(result.rows, experiment.select_by), experiment.select_by
        )

    if result.failures:
        print(f"\n{len(result.failures)} falha(s):")
        for combo_id, ticker, msg in result.failures[:20]:
            print(f"  {combo_id} {ticker}: {msg}")


if __name__ == "__main__":
    main()

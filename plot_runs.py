"""
Visualiza os resultados acumulados em runs.csv.

Uso:
    python plot_runs.py                      # todos os batches do CSV
    python plot_runs.py --batch <batch_id>   # filtra um batch
    python plot_runs.py --metric sharpe      # muda a métrica das barras
    python plot_runs.py --save graficos.png  # salva em vez de abrir janela

A ideia é ser um passo fixo do fluxo: rodou batch.py, roda plot_runs.py.
Funciona com qualquer combinação de stop/multiplier presente no CSV — os
painéis se adaptam ao que os dados contêm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path("runs.csv")


def load(path: Path, batch: str | None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Não achei {path}. Rode batch.py primeiro.")

    df = pd.read_csv(path)
    df = df.dropna(how="all")  # a linha fantasma de vírgulas no fim do arquivo

    if df.empty:
        raise SystemExit("CSV sem linhas de dados.")

    if batch:
        df = df[df["batch_id"] == batch]
        if df.empty:
            raise SystemExit(f"Nenhuma linha com batch_id == {batch}")

    return df


def variant_label(row: pd.Series) -> str:
    """Rótulo curto que identifica a configuração de uma linha.

    'pct' vira 'pct'; atr vira 'atr 2.0' — o multiplicador importa porque é
    a dimensão do teste de platô.
    """
    if row["stop"] == "atr":
        mult = row.get("atr_multiplier")
        return f"atr {mult:g}" if pd.notna(mult) else "atr"
    if row["stop"] == "none":
        dist = row.get("no_stop_distance")
        return f"none {dist:g}" if pd.notna(dist) else "none"
    return "pct"


def panel_return_vs_benchmark(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Retorno da estratégia vs buy-and-hold, por ticker.

    A escala log no eixo importa: buy-and-hold de 500% e estratégia de 15%
    no mesmo eixo linear achatam tudo. O ponto do painel é 'ficou acima ou
    abaixo da linha do baseline', e isso a escala log preserva.
    """
    # Usa a primeira variante de cada ticker para não sobrepor barras.
    first_variant = df["_variant"].iloc[0]
    sub = df[df["_variant"] == first_variant].copy()
    sub = sub.sort_values("ticker")

    x = np.arange(len(sub))
    width = 0.38

    strat = sub["total_return"] * 100
    bench = sub["benchmark_total_return"] * 100

    ax.bar(x - width / 2, strat, width, label="Estratégia", color="#3a7d5d")
    ax.bar(x + width / 2, bench, width, label="Buy & Hold", color="#b8b0a0")

    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(".SA", "") for t in sub["ticker"]], rotation=45, ha="right")
    ax.set_ylabel("Retorno total (%)")
    ax.set_title(f"Estratégia vs Buy & Hold  ({first_variant})", loc="left")
    ax.legend(frameon=False)

    # Conta e anota a vitória — o número que responde a Fase 1.
    wins = int((sub["total_return"] > sub["benchmark_total_return"]).sum())
    ax.text(
        0.99, 0.97, f"estratégia vence em {wins}/{len(sub)}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#666",
    )


def panel_drawdown(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Drawdown por ticker, uma barra por variante de stop.

    Este é o painel que revela o resultado forte da Fase 1: o ATR teve
    drawdown menor em 10/10. Barras lado a lado por ticker deixam isso óbvio.
    """
    variants = sorted(df["_variant"].unique())
    tickers = sorted(df["ticker"].unique())

    x = np.arange(len(tickers))
    width = 0.8 / max(len(variants), 1)
    palette = ["#c25a3c", "#3a7d5d", "#4a6d8c", "#a88b3a", "#7a5a8c"]

    for i, variant in enumerate(variants):
        vals = []
        for ticker in tickers:
            cell = df[(df["ticker"] == ticker) & (df["_variant"] == variant)]
            vals.append(cell["max_drawdown"].iloc[0] * 100 if not cell.empty else np.nan)
        offset = (i - (len(variants) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=variant, color=palette[i % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(".SA", "") for t in tickers], rotation=45, ha="right")
    ax.set_ylabel("Drawdown máximo (%)")
    ax.set_title("Drawdown por variante  (menor = melhor)", loc="left")
    ax.legend(frameon=False)


def panel_plateau(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Curva do multiplicador ATR — o painel do teste de platô da Fase 2.

    Só aparece se houver 2+ multiplicadores ATR no CSV. Cada linha é um
    ticker; o eixo x é o multiplicador. Linhas achatadas = platô (efeito
    estrutural). Bicos pronunciados = pico (sorte de parâmetro).
    """
    atr = df[df["stop"] == "atr"].copy()
    atr = atr.dropna(subset=["atr_multiplier"])
    mults = sorted(atr["atr_multiplier"].unique())

    if len(mults) < 2:
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Teste de platô aparece aqui\nquando houver ≥2 multiplicadores ATR\n"
            "no CSV (Fase 2).",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#999",
        )
        return

    for ticker in sorted(atr["ticker"].unique()):
        sub = atr[atr["ticker"] == ticker].sort_values("atr_multiplier")
        ax.plot(
            sub["atr_multiplier"], sub["max_drawdown"] * 100,
            marker="o", markersize=4, linewidth=1.2,
            label=ticker.replace(".SA", ""),
        )

    ax.set_xlabel("Multiplicador ATR")
    ax.set_ylabel("Drawdown máximo (%)")
    ax.set_title("Teste de platô: drawdown vs multiplicador", loc="left")
    ax.set_xticks(mults)
    ax.legend(frameon=False, fontsize=8, ncol=2)


def panel_metric_bar(ax: plt.Axes, df: pd.DataFrame, metric: str) -> None:
    """Painel genérico: uma métrica escolhida, estratégia por ticker/variante."""
    variants = sorted(df["_variant"].unique())
    tickers = sorted(df["ticker"].unique())

    x = np.arange(len(tickers))
    width = 0.8 / max(len(variants), 1)
    palette = ["#c25a3c", "#3a7d5d", "#4a6d8c", "#a88b3a", "#7a5a8c"]

    for i, variant in enumerate(variants):
        vals = []
        for ticker in tickers:
            cell = df[(df["ticker"] == ticker) & (df["_variant"] == variant)]
            vals.append(cell[metric].iloc[0] if not cell.empty else np.nan)
        offset = (i - (len(variants) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=variant, color=palette[i % len(palette)])

    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(".SA", "") for t in tickers], rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} por variante", loc="left")
    ax.legend(frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--batch", default=None, help="filtra por batch_id")
    parser.add_argument(
        "--metric", default="sharpe",
        help="métrica do 4º painel (ex: sharpe, sortino, expectancy, total_return)",
    )
    parser.add_argument("--save", type=Path, default=None, help="salva PNG em vez de abrir janela")
    args = parser.parse_args()

    df = load(args.csv, args.batch)
    df["_variant"] = df.apply(variant_label, axis=1)

    if args.metric not in df.columns:
        raise SystemExit(
            f"Métrica '{args.metric}' não existe no CSV. "
            f"Disponíveis: {', '.join(c for c in df.columns if df[c].dtype != object)}"
        )

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.axisbelow"] = True

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Análise de batches — backtester", fontsize=14, x=0.01, ha="left")

    panel_return_vs_benchmark(axes[0, 0], df)
    panel_drawdown(axes[0, 1], df)
    panel_plateau(axes[1, 0], df)
    panel_metric_bar(axes[1, 1], df, args.metric)

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.save:
        fig.savefig(args.save, dpi=130, bbox_inches="tight")
        print(f"Salvo em {args.save.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
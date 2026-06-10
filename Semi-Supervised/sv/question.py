import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import colorsys
import glob
from typing import Optional, List, Tuple, Optional
import numpy as np


def configure_chinese_font() -> str:
    """Configure a CJK-capable font for matplotlib.

    Searches for available CJK fonts in priority order and sets
    ``rcParams`` for correct Chinese character rendering.

    Returns
    -------
    str
        Name of the selected font, or empty string if no CJK font
        was found (falls back to DejaVu Sans).
    """
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Heiti SC",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    selected = ""
    for font_name in candidates:
        if font_name in available_fonts:
            selected = font_name
            break

    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans", "Arial Unicode MS"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS"]

    plt.rcParams["axes.unicode_minus"] = False
    return selected


def prepare_data(
    infile: str,
    sv_frequency_cols: List[str],
    mutation_col: str = "mutation",
    sv_frequency_ddPCR_col: str = "ddPCR_AF",
) -> Tuple[List[str], pd.DataFrame]:
    """Compute delta frequency (detected - ddPCR) for each SV frequency column.

    Parameters
    ----------
    infile : str
        Path to the input TSV file containing SV detection results
        and ddPCR ground truth.
    sv_frequency_cols : list of str
        Column names with detected SV frequencies to compare against ddPCR.
    mutation_col : str, optional
        Column name for mutation identifier. Default is ``"mutation"``.
    sv_frequency_ddPCR_col : str, optional
        Column name for ddPCR ground truth frequency. Default is
        ``"ddPCR_AF"``.

    Returns
    -------
    delta_frequency_cols : list of str
        Names of the newly created delta frequency columns
        (``"delta_frequency_{col}"`` for each input column).
    df : pandas.DataFrame
        Subset DataFrame containing only the mutation column and
        delta frequency columns.
    """
    df = pd.read_csv(infile, sep="\t")
    for col in sv_frequency_cols:
        df[f"delta_frequency_{col}"] = df[col] - df[sv_frequency_ddPCR_col]
    delta_frequency_cols = [f"delta_frequency_{col}" for col in sv_frequency_cols]
    return delta_frequency_cols, df.loc[:, [mutation_col, *delta_frequency_cols]]

def paired_violin_plot(
    df: pd.DataFrame,
    key: str,
    values: List[str],
    outfile: str,
    xlabel: str = "Group",
    ylabel: str = "Value",
    group_names: Optional[List[str]] = None,
    palette: Tuple[str, str] = ("#2166AC", "#B2182B"),
    figsize: tuple = (7, 8),
    violin_alpha: float = 0.25,
    violin_width: float = 0.7,
    point_size: float = 28,
    line_alpha: float = 0.35,
    line_width: float = 0.8,
    jitter: float = 0.04,
):
    """Draw a paired violin plot connecting matched samples across two groups.

    Each sample is represented by a point on each side with a gray line
    connecting the paired observations.  Median lines are drawn for each
    group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    key : str
        Column used as sample identifier (not plotted on x-axis;
        used only for pairing).
    values : list of str
        Exactly two column names to compare.  Each becomes one side
        of the paired violin.
    outfile : str
        Output image path (PNG/PDF).
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    group_names : list of str or None, optional
        Display names for the two groups.  Defaults to ``values``.
    palette : tuple of str, optional
        Two hex colors for the left and right violins.
    figsize : tuple, optional
        Figure size in inches.
    violin_alpha : float, optional
        Transparency of violin bodies.
    violin_width : float, optional
        Width of each violin.
    point_size : float, optional
        Scatter point size.
    line_alpha : float, optional
        Transparency of paired connecting lines.
    line_width : float, optional
        Width of paired connecting lines.
    jitter : float, optional
        Horizontal jitter range for scatter points.

    Raises
    ------
    ValueError
        If ``key`` is not in ``df.columns``, ``values`` does not
        contain exactly two entries, or any value column is missing.
    """
    if key not in df.columns:
        raise ValueError(
            f"sample id column not found: {key}"
        )

    if len(values) != 2:
        raise ValueError(
            "`values` must contain exactly two columns"
        )

    missing_cols = [
        v for v in values
        if v not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"columns not found: {missing_cols}"
        )
    if group_names is None:
        group_names = values

    if len(group_names) != 2:
        raise ValueError(
            "`group_names` must contain two names"
        )
    df_plot = (
        df[[key] + values]
        .dropna(subset=values)
        .copy()
    )
    long_parts = []

    for value_col, group_name in zip(
        values,
        group_names,
    ):

        sub_df = pd.DataFrame(
            {
                "sample_id": df_plot[key],
                "group": group_name,
                "value": df_plot[value_col],
            }
        )

        long_parts.append(sub_df)

    long_df = pd.concat(
        long_parts,
        ignore_index=True,
    )
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(
        figsize=figsize
    )
    violin = sns.violinplot(
        data=long_df,
        x="group",
        y="value",
        order=group_names,
        palette=list(palette),
        cut=1,
        inner=None,
        linewidth=1.5,
        width=violin_width,
        bw_adjust=1.2,
        density_norm="area",
        ax=ax,
    )
    for poly in violin.collections:

        try:
            poly.set_alpha(violin_alpha)
            poly.set_edgecolor("black")
            poly.set_linewidth(1.2)
        except Exception:
            pass
    x_positions = {
            group_names[0]: 0,
            group_names[1]: 1,
        }
    rng = np.random.default_rng(123)

    for _, row in df_plot.iterrows():


        x1 = (
            x_positions[group_names[0]]
            + rng.uniform(-jitter, jitter)
        )

        x2 = (
            x_positions[group_names[1]]
            + rng.uniform(-jitter, jitter)
        )

        y1 = row[values[0]]
        y2 = row[values[1]]

        # paired line
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="gray",
            linewidth=line_width,
            alpha=line_alpha,
            zorder=1,
        )

        # left point
        ax.scatter(
            x1,
            y1,
            s=point_size,
            color="black",
            alpha=0.9,
            zorder=3,
        )

        # right point
        ax.scatter(
            x2,
            y2,
            s=point_size,
            color="black",
            alpha=0.9,
            zorder=3,
        )
    medians = [
        np.median(df_plot[v])
        for v in values
    ]

    for i, median in enumerate(medians):

        ax.hlines(
            y=median,
            xmin=i - 0.18,
            xmax=i + 0.18,
            color="black",
            linewidth=2.5,
            zorder=5,
        )
    ax.set_xlabel(
        xlabel,
        fontsize=14,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=14,
    )
    ax.tick_params(
        axis="x",
        labelsize=12,
    )

    ax.tick_params(
        axis="y",
        labelsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )
    fig.tight_layout()
    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

def single_violin_plot(
    df: pd.DataFrame,
    value: str,
    key: Optional[str] = None,
    outfile: str = "violin.png",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Value",
    order: Optional[List[str]] = None,
    sort_keys: bool = True,
    palette: str = "Set2",
    figsize: tuple = (6, 6),
    violin_alpha: float = 0.85,
    point_size: float = 2,
    show_points: bool = False,
    show_median: bool = False,
    threshold: Optional[float] = None,
    xtick_fontsize: int = 10,
):
    """Draw a single (or per-category) violin + strip plot.

    When ``key`` is ``None`` a single violin is drawn for the entire
    ``value`` column.  When ``key`` is provided, one violin is drawn
    per unique category in that column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    value : str
        Column name for the numeric values (y-axis).
    key : str or None, optional
        Column name for x-axis categories.  ``None`` draws one
        overall violin.
    outfile : str, optional
        Output image path (PNG/PDF).
    title : str, optional
        Plot title.  Empty string (default) means no title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    order : list of str or None, optional
        Explicit category ordering on the x-axis.  Ignored when
        ``key`` is ``None``.
    sort_keys : bool, optional
        Sort categories alphabetically when ``order`` is ``None``.
    palette : str, optional
        Seaborn / matplotlib colormap name.
    figsize : tuple, optional
        Figure size in inches.
    violin_alpha : float, optional
        Fill transparency of violin bodies.
    point_size : float, optional
        Strip-plot dot size.
    show_points : bool, optional
        Whether to overlay individual data points as a strip plot.
        Default is ``False``.
    show_median : bool, optional
        Draw a horizontal median line inside each violin.
    threshold : float or None, optional
        When set, draws a red dashed horizontal line at this value
        and displays in the legend what percentage of data points
        fall below it (overall when ``key`` is ``None``, per
        category otherwise).

    Raises
    ------
    ValueError
        If ``value`` (or ``key``, when provided) is not in
        ``df.columns``.
    """
    if value not in df.columns:
        raise ValueError(f"Value column not found: {value}")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    sns.set_style("whitegrid")

    if key is not None:
        if key not in df.columns:
            raise ValueError(f"Category column not found: {key}")
        plot_df = df[[key, value]].dropna().copy()
        if order is None:
            observed = plot_df[key].unique().tolist()
            order = sorted(observed) if sort_keys else observed
    else:
        plot_df = df[[value]].dropna().copy()
        plot_df["_cat"] = ""
        key = "_cat"
        order = [""]

    n_cats = len(order)
    base_colors = sns.color_palette(palette, n_colors=max(n_cats, 1))

    fig, ax = plt.subplots(figsize=figsize)

    # ── violin ──────────────────────────────────────────────
    vp = sns.violinplot(
        data=plot_df,
        x=key,
        y=value,
        cut=0,
        order=order,
        palette=base_colors,
        inner="box",
        linewidth=1.2,
        width=0.7,
        bw_adjust=1.2,
        density_norm="area",
        ax=ax,
    )
    for poly in ax.collections:
        try:
            poly.set_alpha(violin_alpha)
            poly.set_edgecolor("black")
            poly.set_linewidth(1.0)
        except Exception:
            pass

    # ── strip ───────────────────────────────────────────────
    if show_points:
        sns.stripplot(
            data=plot_df,
            x=key,
            y=value,
            order=order,
            color="black",
            size=point_size,
            alpha=0.9,
            jitter=0.12,
            linewidth=0,
            ax=ax,
        )

    # ── median line ─────────────────────────────────────────
    if show_median:
        for i, cat in enumerate(order):
            vals = plot_df.loc[plot_df[key] == cat, value]
            if len(vals) > 0:
                med = float(np.median(vals))
                ax.hlines(
                    y=med,
                    xmin=i - 0.18,
                    xmax=i + 0.18,
                    color="black",
                    linewidth=2.5,
                    zorder=5,
                )

    # ── threshold line ──────────────────────────────────────
    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=4,
        )
        # Compute proportion below threshold
        below = plot_df[value] < threshold
        pct_below = 100.0 * below.sum() / len(plot_df)
        if key == "_cat":
            # Single violin: one overall percentage
            label = f"< {threshold}: {pct_below:.1f}%"
        else:
            # Per-category percentages
            parts = []
            for cat in order:
                cat_vals = plot_df.loc[plot_df[key] == cat, value]
                cat_pct = 100.0 * (cat_vals < threshold).sum() / len(cat_vals) if len(cat_vals) > 0 else 0.0
                parts.append(f"{cat} {cat_pct:.1f}%")
            label = f"< {threshold}: " + ", ".join(parts)
        ax.legend(
            [plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5)],
            [label],
            loc="upper right",
            fontsize=10,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
        )

    # ── axes ────────────────────────────────────────────────
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    if key == "_cat":
        ax.set_xticks([])

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)

def single_bar_plot(
    df: pd.DataFrame,
    value: str,
    key: Optional[str] = None,
    outfile: str = "bar.png",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Value",
    order: Optional[List[str]] = None,
    sort_keys: bool = True,
    palette: str = "Set2",
    figsize: tuple = (6, 6),
    bar_alpha: float = 0.85,
    point_size: float = 20,
    show_bar: bool = True,
    show_points: bool = False,
    show_mean: bool = True,
    errorbar: str = "se",
    threshold: Optional[float] = None,
    xtick_fontsize: int = 10,
):
    """Draw a bar chart with error bars and individual data points.

    Each category is represented by a bar showing the mean (or
    median) with error bars (SEM or SD), overlaid with a strip plot
    of individual observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    value : str
        Column name for the numeric values (y-axis).
    key : str or None, optional
        Column name for x-axis categories.  ``None`` draws one
        overall bar.
    outfile : str, optional
        Output image path (PNG/PDF).
    title : str, optional
        Plot title.  Empty string (default) means no title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    order : list of str or None, optional
        Explicit category ordering on the x-axis.  Ignored when
        ``key`` is ``None``.
    sort_keys : bool, optional
        Sort categories alphabetically when ``order`` is ``None``.
    palette : str, optional
        Seaborn / matplotlib colormap name.
    figsize : tuple, optional
        Figure size in inches.
    bar_alpha : float, optional
        Fill transparency of bars.
    point_size : float, optional
        Strip-plot dot size.
    show_bar : bool, optional
        Whether to draw the bars.  Default is ``True``.
    show_points : bool, optional
        Whether to overlay individual data points as a strip plot.
        Default is ``False``.
    show_mean : bool, optional
        If ``True`` bars show the mean; otherwise the median.
    errorbar : str, optional
        Type of error bar: ``"se"`` for standard error of the mean,
        ``"sd"`` for standard deviation, ``"ci95"`` for 95%
        confidence interval.
    threshold : float or None, optional
        When set, draws a red dashed horizontal line at this value
        and displays in the legend what percentage of data points
        fall below it.

    Raises
    ------
    ValueError
        If ``value`` (or ``key``, when provided) is not in
        ``df.columns``.
    """
    if value not in df.columns:
        raise ValueError(f"Value column not found: {value}")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    sns.set_style("whitegrid")

    if key is not None:
        if key not in df.columns:
            raise ValueError(f"Category column not found: {key}")
        plot_df = df[[key, value]].dropna().copy()
        if order is None:
            # Default: sort by value descending (largest first)
            group_means = plot_df.groupby(key)[value].mean()
            order = group_means.sort_values(ascending=False).index.tolist()
            if not sort_keys:
                order = plot_df[key].unique().tolist()
    else:
        plot_df = df[[value]].dropna().copy()
        plot_df["_cat"] = ""
        key = "_cat"
        order = [""]

    n_cats = len(order)
    base_colors = sns.color_palette(palette, n_colors=max(n_cats, 1))

    # ── compute summary statistics ──────────────────────────
    stats = []
    for i, cat in enumerate(order):
        vals = plot_df.loc[plot_df[key] == cat, value].to_numpy()
        n = len(vals)
        if n == 0:
            stats.append({"cat": cat, "mean": 0, "err": 0, "n": 0})
            continue
        mean = float(np.mean(vals)) if show_mean else float(np.median(vals))
        if errorbar == "se":
            err = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        elif errorbar == "sd":
            err = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        elif errorbar == "ci95":
            err = float(1.96 * np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        else:
            err = 0.0
        stats.append({"cat": cat, "mean": mean, "err": err, "n": n})
    stats_df = pd.DataFrame(stats)

    fig, ax = plt.subplots(figsize=figsize)

    # ── bars ────────────────────────────────────────────────
    x_pos = np.arange(n_cats)
    if show_bar:
        bars = ax.bar(
            x_pos,
            stats_df["mean"],
            yerr=stats_df["err"],
            width=0.6,
            color=base_colors,
            alpha=bar_alpha,
            edgecolor="black",
            linewidth=1.0,
            capsize=4,
            error_kw={"linewidth": 1.2, "elinewidth": 1.2},
            zorder=3,
        )

    # ── individual points ───────────────────────────────────
    if show_points:
        rng = np.random.default_rng(42)
        for i, cat in enumerate(order):
            vals = plot_df.loc[plot_df[key] == cat, value].to_numpy()
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                x_pos[i] + jitter,
                vals,
                s=point_size,
                color="black",
                alpha=0.9,
                zorder=4,
                linewidth=0,
            )

    # ── threshold line ──────────────────────────────────────
    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )
        below = plot_df[value] < threshold
        pct_below = 100.0 * below.sum() / len(plot_df)
        if key == "_cat":
            label = f"< {threshold}: {pct_below:.1f}%"
        else:
            parts = []
            for cat in order:
                cat_vals = plot_df.loc[plot_df[key] == cat, value]
                cat_pct = 100.0 * (cat_vals < threshold).sum() / len(cat_vals) if len(cat_vals) > 0 else 0.0
                parts.append(f"{cat} {cat_pct:.1f}%")
            label = f"< {threshold}: " + ", ".join(parts)
        ax.legend(
            [plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5)],
            [label],
            loc="upper right",
            fontsize=10,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
        )

    # ── axes ────────────────────────────────────────────────
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order, fontsize=xtick_fontsize, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    if key == "_cat":
        ax.set_xticks([])

    # ── y-axis from 0 ────────────────────────────────────────
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0, ymin), ymax)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)

def adjust_group_color(
    color,
    group_idx,
    n_groups,
    lightness_strength: float = 0.25,
    saturation_strength: float = 0.35,
    max_lightness: float = 0.75,
):
    """
    Adjust group color while keeping
    mutation hue stable.

    Parameters
    ----------
    color :
        Base RGB tuple.

    group_idx :
        Group index.

    n_groups :
        Total group count.

    lightness_strength :
        How much brighter later groups become.

        Larger value:
            stronger lightness contrast.

    saturation_strength :
        How much saturation decreases.

        Larger value:
            stronger pale effect.

    max_lightness :
        Upper limit of brightness.

        Prevents colors becoming nearly white.
    """

    r, g, b = color

    h, l, s = colorsys.rgb_to_hls(
        r,
        g,
        b,
    )

    if n_groups == 1:
        return color

    ratio = group_idx / (n_groups - 1)

    # increase lightness
    l = min(
        max_lightness,
        l + ratio * lightness_strength,
    )

    # reduce saturation
    s = max(
        0.05,
        s - ratio * saturation_strength,
    )

    return colorsys.hls_to_rgb(
        h,
        l,
        s,
    )


def violin_plot(
    df: pd.DataFrame,
    key: str,
    values: List[str],
    outfile: str,
    xlabel: str = "SV Type",
    ylabel: str = "Delta Frequency",
    order: Optional[List[str]] = None,
    sort_keys: bool = True,
    group_names: Optional[List[str]] = None,
    group_order: Optional[List[str]] = None,
    base_palette: str = "Set2",
    figsize: tuple = (16, 8),
):
    """Draw grouped violin + strip plot with per-mutation coloring.

    Each mutation (x-axis category) gets a distinct base color from
    ``base_palette``.  Groups within the same mutation are rendered
    with progressively lighter / less saturated shades of that base
    color (via :func:`adjust_group_color`).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    key : str
        Column name for the x-axis categories (e.g. mutation type).
    values : list of str
        Column names for y-axis values.  Each column becomes one
        group (dodged side-by-side within each x category).
    outfile : str
        Output image path (PNG/PDF).
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    order : list of str or None, optional
        Explicit x-axis category ordering.  If ``None``, inferred
        from data (optionally sorted via ``sort_keys``).
    sort_keys : bool, optional
        Whether to sort x-axis categories alphabetically when
        ``order`` is ``None``.
    group_names : list of str or None, optional
        Display names for each group (must match length of
        ``values``).  Defaults to column names.
    group_order : list of str or None, optional
        Explicit group ordering for the hue dimension.
    base_palette : str, optional
        Name of a seaborn / matplotlib colormap for base mutation
        colors.
    figsize : tuple, optional
        Figure size in inches.

    Raises
    ------
    ValueError
        If ``key`` or any entry in ``values`` is missing from
        ``df``, or ``group_names`` length does not match ``values``.
    """

    # ==========================================================
    # matplotlib settings
    # ==========================================================

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    sns.set_style("whitegrid")

    # ==========================================================
    # validation
    # ==========================================================

    if key not in df.columns:
        raise ValueError(
            f"x-axis column not found: {key}"
        )

    if not values:
        raise ValueError(
            "`values` cannot be empty"
        )

    missing_cols = [
        v for v in values
        if v not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"y-axis columns not found: {missing_cols}"
        )

    if (
        group_names is not None
        and len(group_names) != len(values)
    ):
        raise ValueError(
            "`group_names` length must equal "
            "`values` length"
        )

    # ==========================================================
    # build long dataframe
    # ==========================================================

    long_parts = []

    for i, v_col in enumerate(values):

        group_name = (
            group_names[i]
            if group_names is not None
            else v_col
        )

        sub_df = (
            df[[key, v_col]]
            .dropna(subset=[key, v_col])
            .copy()
        )

        sub_df = sub_df.rename(
            columns={v_col: "_y"}
        )

        sub_df["group"] = group_name

        long_parts.append(sub_df)

    if not long_parts:
        raise ValueError(
            "No valid plotting data"
        )

    df_plot = pd.concat(
        long_parts,
        ignore_index=True,
    )

    # ==========================================================
    # x-axis order
    # ==========================================================

    if order is None:

        observed_order = (
            df_plot[key]
            .dropna()
            .unique()
            .tolist()
        )

        if sort_keys:
            order = sorted(observed_order)
        else:
            order = observed_order

    # ==========================================================
    # group order
    # ==========================================================

    if group_order is None:

        group_order = (
            df_plot["group"]
            .dropna()
            .unique()
            .tolist()
        )

    # ==========================================================
    # base colors
    # ==========================================================

    base_colors = sns.color_palette(
        base_palette,
        n_colors=len(order),
    )

    mutation_color_map = {
        mutation: color
        for mutation, color in zip(
            order,
            base_colors,
        )
    }

    # ==========================================================
    # color adjustment
    # ==========================================================

    

    # ==========================================================
    # plotting
    # ==========================================================

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ----------------------------------------------------------
    # violin plot
    # ----------------------------------------------------------

    sns.violinplot(
        data=df_plot,
        x=key,
        y="_y",
        hue="group",
        order=order,
        hue_order=group_order,
        dodge=True,
        inner="quartile",
        linewidth=1.2,
        saturation=1,
        bw_adjust=1.2,
        density_norm="area",
        width=0.95,
        ax=ax,
    )

    # ==========================================================
    # recolor violins
    # ==========================================================

    violin_bodies = [
        c for c in ax.collections
        if isinstance(
            c,
            mcollections.PolyCollection
        )
    ]

    violin_idx = 0

    for mutation in order:

        base_color = mutation_color_map[
            mutation
        ]

        for group_idx, group in enumerate(
            group_order
        ):

            if violin_idx >= len(
                violin_bodies
            ):
                break

            color = adjust_group_color(
                base_color,
                group_idx,
                len(group_order),
            )

            violin = violin_bodies[
                violin_idx
            ]

            violin.set_facecolor(color)

            violin.set_edgecolor(
                "black"
            )

            violin.set_alpha(0.95)

            violin.set_linewidth(1)

            violin_idx += 1

    # ==========================================================
    # black solid strip points
    # ==========================================================

    sns.stripplot(
        data=df_plot,
        x=key,
        y="_y",
        hue="group",
        order=order,
        hue_order=group_order,
        dodge=True,
        jitter=0.12,
        color="black",
        size=2.5,
        alpha=0.9,
        linewidth=0,
        ax=ax,
    )

    # ==========================================================
    # remove duplicated legends
    # ==========================================================

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # ==========================================================
    # explanatory legend
    # ==========================================================

    example_mutation = order[0]

    example_base_color = mutation_color_map[
        example_mutation
    ]

    legend_handles = []

    n_groups = len(group_order)

    if n_groups == 2:

        shade_words = [
            "Dark",
            "Light",
        ]

    else:

        shade_words = [
            f"Shade {i + 1}"
            for i in range(n_groups)
        ]

    for group_idx, group in enumerate(
        group_order
    ):

        legend_color = adjust_group_color(
            example_base_color,
            group_idx,
            n_groups,
        )

        label = (
            f"{shade_words[group_idx]} "
            f"{group}"
        )

        legend_handles.append(
            mpatches.Patch(
                facecolor=legend_color,
                edgecolor="black",
                label=label,
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Color meaning",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )

    # ==========================================================
    # labels
    # ==========================================================

    ax.set_xlabel(
        xlabel,
        fontsize=14,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=14,
    )

    # ==========================================================
    # ticks
    # ==========================================================

    ax.tick_params(
        axis="x",
        rotation=45,
        labelsize=11,
    )

    ax.tick_params(
        axis="y",
        labelsize=11,
    )

    # ==========================================================
    # style
    # ==========================================================

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    # ==========================================================
    # layout
    # ==========================================================

    fig.tight_layout()

    # ==========================================================
    # save
    # ==========================================================

    fig.savefig(
        outfile,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

def format_excel(
    infile:str,
    outfile:str
):
    """Format an Excel table for downstream analysis.

    Reads all sheets from ``infile``, forward-fills merged cells,
    strips newlines and tabs, creates a unified ``mutation`` column
    from ``cHGVS`` and ``pHGVS``, and writes the result as TSV.

    Parameters
    ----------
    infile : str
        Path to the input Excel file (.xlsx).
    outfile : str
        Path to the output TSV file.
    """
    df_dict = pd.read_excel(infile, sheet_name=None)
    for sheet, df in df_dict.items():
        df = df.ffill()  # 处理合并单元格
        df = df.replace(r"[\n\t]", " ", regex=True)  # 去掉破坏结构的字符
        df.columns = df.columns.str.strip()
        df["mutation"] = df["cHGVS"] + "(" + df["pHGVS"] + ")"
        df.to_csv(outfile, sep="\t", index=False)

def combine_sv_origin(
    sv_file: str,
    map_file:str,
    outfile:str
):
    """Merge SV detection results with sample origin mapping and plot.

    Joins the SV table with a sample-ID mapping table on ``sampleID``,
    creates a combined ``SV_Type`` label (``FusionGene(FusionExon)``),
    saves the merged table as TSV, and generates a violin plot of
    raw frequencies grouped by SV type.

    Parameters
    ----------
    sv_file : str
        Path to the SV detection TSV (must contain ``sampleID``,
        ``FusionGene``, ``FusionExon``, ``Freq``).
    map_file : str
        Path to the sample-ID mapping TSV (must contain
        ``sampleID``).
    outfile : str
        Output path for the combined TSV.  The violin plot is saved
        alongside it with a ``_sv_freq_comparison_violin_plot.png``
        suffix.
    """
    df_sv = pd.read_csv(sv_file, sep="\t")
    df_map = pd.read_csv(map_file, sep="\t", dtype=str)
    df_combined = df_sv.merge(
        df_map,
        on="sampleID",
        how="left",
    )
    df_combined.to_csv(outfile, sep="\t", index=False)
    df_combined["SV_Type"] = df_combined["FusionGene"] + "(" + df_combined["FusionExon"] + ")"
    return df_combined

def run_new_cd6():
    """Run SV–origin merge and violin plot for the New_cd6 dataset.

    Hardcoded paths for the New_cd6 validation cohort.  Calls
    :func:`combine_sv_origin` to merge SV detections with sample
    mapping and produce a frequency comparison violin plot.
    """
    sv_file = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/New_cd6/new_cd6_SV.tsv"
    map_file = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/New_cd6/match.tsv"
    outfile = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/New_cd6/combined_data.tsv"
    df_combined = combine_sv_origin(sv_file, map_file, outfile)
    # single_violin_plot(
    #     df_combined,
    #     value="Freq",
    #     outfile=outfile.replace("combined_data.tsv", "sv_freq_comparison_violin_plot.png"),
    #     xlabel="SV",
    #     ylabel="Frequency",
    #     threshold=0.01,
    # )
    df_combined = df_combined[df_combined["Freq"] < 0.01]
    df_sv_count = df_combined["SV_Type"].value_counts()
    df_sv_count = df_sv_count.reset_index()
    df_sv_count.columns = ["SV_Type", "count"]
    single_bar_plot(
        df_sv_count,
        value="count",
        key="SV_Type",
        outfile=outfile.replace("combined_data.tsv", "sv_freq_comparison_bar_plot.png"),
        figsize=(12, 6),
        xlabel="SV Type",
        ylabel="Count",
        xtick_fontsize=8,
        title="SV Count Distribution (Freq < 0.01)",
    )


def main():
    """Run delta-frequency violin plot for the main validation dataset.

    Configures CJK font, computes delta frequency (detected − ddPCR)
    for each mutation, and generates a grouped violin plot comparing
    frequencies before correction.
    """
    infile = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/SV_jiaozheng_yanzheng.xlsx"
    outfile = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/SV_jiaozheng_yanzheng.tsv"
    # format_excel(infile, outfile)
    selected_font = configure_chinese_font()
    if selected_font:
        print(f"Using Chinese font: {selected_font}")
    else:
        print("No preferred Chinese font found. Using matplotlib fallback fonts.")
    
    infile = "/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/data/combined_data.tsv"
    delta_frequency_cols, df = prepare_data(infile, mutation_col="mutation", sv_frequency_cols=["Freq"], sv_frequency_ddPCR_col="ddPCR_AF")
    violin_plot(df, key="mutation", values= delta_frequency_cols, outfile=infile.replace("combined_data.tsv", "delta_frequency_violin_plot.png"), group_names=["before correction"])
    # for infile in glob.glob("/mnt/GenePlus002/genecloud/Org_terminal/org_52/terminal/luoshg_15179660974/Data/sta/20260508_SV_freq_correction/output/ML/predict/validation/model_*/predictions.tsv"):
    #     print(f"Processing {infile}...")
    #     delta_frequency_cols, df = prepare_data(infile, mutation_col="mutation", sv_frequency_cols=["Freq", "predicted_AF"], sv_frequency_ddPCR_col="ddPCR_AF")
    #     outfile = infile.replace("predictions.tsv", "delta_frequency_violin_plot.png")
    #     group_names = ["before correction", "after correction"] if len(delta_frequency_cols) == 2 else None
    #     violin_plot(
    #         df,
    #         key="mutation",
    #         values= delta_frequency_cols,
    #         outfile=outfile,
    #         group_names=group_names,
    #     )
    #     out_file_paired = infile.replace("predictions.tsv", "delta_frequency_paired_violin_plot.png")
        # paired_violin_plot(
        #     df,
        #     key="mutation",
        #     values= delta_frequency_cols,
        #     outfile=out_file_paired,
        #     group_names=group_names,
        # )

if __name__ == "__main__":
    run_new_cd6()
    # main()

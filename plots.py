#!/usr/bin/env python3

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HOUR_LABELS = [f"{hour:02d}" for hour in range(24)]


def set_plot_style() -> None:
    plt.style.use("default")
    sns.set_palette("colorblind")


def slugify(value: str) -> str:
    return str(value).strip().replace(" ", "-").replace("_", "-")


def get_series_spec(data_config: dict[str, Any], series_id: str) -> dict[str, Any]:
    if series_id not in data_config:
        raise ValueError(f"Unknown series '{series_id}'")
    series_config = data_config[series_id]
    label = series_config["label"] if isinstance(series_config, dict) else series_config.label
    color = series_config["color"] if isinstance(series_config, dict) else series_config.color
    return {
        "id": series_id,
        "label": label,
        "color": color,
    }


def select_series_data(dataframe: pd.DataFrame, series_id: str) -> pd.DataFrame:
    return dataframe[dataframe["series"] == series_id]


def get_hour_order(day_origin_hour: int) -> list[int]:
    return list(range(day_origin_hour, 24)) + list(range(0, day_origin_hour))


def parse_day_origin_hour(figure: dict[str, Any], plot_config: dict[str, Any] | None = None) -> int:
    day_origin_hour = int(
        figure.get("day_origin_hour", (plot_config or {}).get("day_origin_hour", 0))
    )
    if day_origin_hour < 0 or day_origin_hour > 23:
        raise ValueError("day_origin_hour must be between 0 and 23")
    return day_origin_hour


def bucket_metadata(mode: str, day_origin_hour: int) -> tuple[str, list[int], list[str], str]:
    if mode == "hour":
        hour_order = get_hour_order(day_origin_hour)
        return "hour", hour_order, [f"{hour:02d}" for hour in hour_order], "Hour of Day"
    if mode == "day":
        return "day_of_week", list(range(7)), WEEKDAY_LABELS, "Day of Week"
    if mode == "month":
        return "month", list(range(1, 13)), MONTH_LABELS, "Month"
    raise ValueError(f"Unsupported histogram mode: {mode}")


def plot_histogram(
    dataframe: pd.DataFrame,
    figure: dict[str, Any],
    plot_config: dict[str, Any],
    data_config: dict[str, Any],
    output_dir: Path,
) -> None:
    mode = figure["mode"]
    day_origin_hour = parse_day_origin_hour(figure, plot_config)
    column, buckets, tick_labels, x_label = bucket_metadata(mode, day_origin_hour)
    stacked_data: dict[str, pd.Series] = {}
    colors: list[str] = []
    for series_id in figure["series"]:
        series_spec = get_series_spec(data_config, series_id)
        series_data = select_series_data(dataframe, series_id)
        counts = series_data.groupby(column).size().reindex(buckets, fill_value=0)
        stacked_data[series_spec["label"]] = counts
        colors.append(series_spec["color"])

    histogram_data = pd.DataFrame(stacked_data, index=buckets)
    max_total = histogram_data.sum(axis=1).max()
    if max_total > 0:
        histogram_data = histogram_data * (100.0 / max_total)

    plt.figure(figsize=(12, 6))
    histogram_data.plot(kind="bar", stacked=True, alpha=0.8, color=colors)
    plt.title(figure["title"])
    plt.xlabel(x_label)
    plt.ylabel(figure["y_label"])
    plt.xticks(range(len(tick_labels)), tick_labels, rotation=45 if mode == "month" else 0)
    plt.ylim(0, 100)
    plt.legend(title=None)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / figure["filename"], dpi=300, bbox_inches="tight")
    plt.close()


def build_daily_series(series_data: pd.DataFrame) -> pd.DataFrame:
    daily_counts = series_data.groupby("date").size().reset_index(name="count")
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])
    return daily_counts


def parse_x_start(figure: dict[str, Any]) -> pd.Timestamp | None:
    x_start = figure.get("x_start")
    if x_start is None:
        return None
    return pd.to_datetime(x_start)


def plot_curves(
    dataframe: pd.DataFrame,
    figure: dict[str, Any],
    data_config: dict[str, Any],
    output_dir: Path,
    event_items: list[dict],
) -> None:
    rolling_window = int(figure.get("rolling_window", 14))
    y_scale = figure.get("y_scale", "linear")
    show_raw = bool(figure.get("show_raw", True))
    x_start = parse_x_start(figure)
    daily_series: list[tuple[dict[str, Any], pd.DataFrame]] = []
    max_count = 0
    for series_id in figure["series"]:
        series_spec = get_series_spec(data_config, series_id)
        daily_counts = build_daily_series(select_series_data(dataframe, series_id))
        daily_series.append((series_spec, daily_counts))
        if not daily_counts.empty:
            max_count = max(max_count, int(daily_counts["count"].max()))

    plt.figure(figsize=(15, 6))
    for series_spec, daily_counts in daily_series:
        if max_count > 0:
            normalized_count = (daily_counts["count"] * (100.0 / max_count)).clip(upper=100)
        else:
            normalized_count = daily_counts["count"]
        if show_raw:
            plt.plot(
                daily_counts["date"],
                normalized_count,
                alpha=0.2,
                linewidth=0.7,
                color=series_spec["color"],
            )
        plt.plot(
            daily_counts["date"],
            normalized_count.rolling(window=rolling_window, center=True).mean(),
            linewidth=2,
            color=series_spec["color"],
            label=f"{series_spec['label']} ({rolling_window}-day avg)",
        )

    plt.title(figure["title"])
    plt.xlabel("Date")
    plt.ylabel(figure["y_label"])
    if y_scale == "log":
        plt.yscale("log")
        plt.ylim(1, 100)
    else:
        plt.ylim(0, 100)
    add_events(plt.gca(), event_items)
    if x_start is not None:
        plt.xlim(left=x_start)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / figure["filename"], dpi=300, bbox_inches="tight")
    plt.close()


def plot_panel_curves(
    dataframe: pd.DataFrame,
    figure: dict[str, Any],
    data_config: dict[str, Any],
    output_dir: Path,
    event_items: list[dict],
) -> None:
    panels = figure["panels"]
    rolling_window = int(figure.get("rolling_window", 14))
    y_scale = figure.get("y_scale", "log")
    show_raw = bool(figure.get("show_raw", False))
    x_start = parse_x_start(figure)
    figure_obj, axes = plt.subplots(len(panels), 1, figsize=(15, 4 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    for axis, panel in zip(axes, panels):
        panel_series = []
        panel_max = 0
        active_start = None
        active_end = None
        for series_id in panel["series"]:
            series_spec = get_series_spec(data_config, series_id)
            daily_counts = build_daily_series(select_series_data(dataframe, series_id))
            panel_series.append((series_spec, daily_counts))
            if not daily_counts.empty:
                panel_max = max(panel_max, int(daily_counts["count"].max()))
                date_start = daily_counts["date"].min()
                date_end = daily_counts["date"].max()
                active_start = date_start if active_start is None else min(active_start, date_start)
                active_end = date_end if active_end is None else max(active_end, date_end)

        for series_spec, daily_counts in panel_series:
            if panel_max > 0:
                normalized_count = (daily_counts["count"] * (100.0 / panel_max)).clip(upper=100)
            else:
                normalized_count = daily_counts["count"]
            smoothed = normalized_count.rolling(window=rolling_window, center=True).mean()
            smoothed = smoothed.where(smoothed > 0)
            if show_raw:
                axis.plot(
                    daily_counts["date"],
                    normalized_count.where(normalized_count > 0),
                    alpha=0.2,
                    linewidth=0.7,
                    color=series_spec["color"],
                )
            axis.plot(
                daily_counts["date"],
                smoothed,
                linewidth=2,
                color=series_spec["color"],
                label=series_spec["label"],
            )

        add_events(axis, event_items)
        axis.set_title(panel["title"])
        axis.set_ylabel(panel.get("y_label", "Images (local max-normalized to 100)"))
        if y_scale == "log":
            axis.set_yscale("log")
            axis.set_ylim(1, 100)
        else:
            axis.set_ylim(0, 100)
        if active_start is not None and active_end is not None:
            padding_days = int(panel.get("padding_days", 30))
            padding = pd.Timedelta(days=padding_days)
            left_bound = active_start - padding
            if x_start is not None:
                left_bound = max(left_bound, x_start)
            axis.set_xlim(left_bound, active_end + padding)
        elif x_start is not None:
            axis.set_xlim(left=x_start)
        axis.grid(alpha=0.3)
        axis.legend(title=None)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(output_dir / figure["filename"], dpi=300, bbox_inches="tight")
    plt.close()


def render_figures(
    dataframe: pd.DataFrame,
    output_dir: Path,
    plot_config: dict[str, Any],
    data_config: dict[str, Any],
    event_items: list[dict],
) -> None:
    for figure in plot_config["figures"]:
        kind = figure["kind"]
        if kind == "histogram":
            plot_histogram(dataframe, figure, plot_config, data_config, output_dir)
            continue
        if kind == "curves":
            plot_curves(dataframe, figure, data_config, output_dir, event_items)
            continue
        if kind == "total_curve":
            total_rows = dataframe[dataframe["series"].isin(figure["series"])].copy()
            total_rows["series"] = "__sum__"
            total_figure = dict(figure)
            total_figure["series"] = ["__sum__"]
            total_data = dict(data_config)
            total_data["__sum__"] = {
                "label": figure.get("label", "summed sources"),
                "color": figure.get("color", "#444444"),
            }
            plot_curves(total_rows, total_figure, total_data, output_dir, event_items)
            continue
        if kind == "panel_curves":
            plot_panel_curves(dataframe, figure, data_config, output_dir, event_items)
            continue
        if kind == "heatmap_per_source":
            series_key = figure.get("series_key", "source")
            plot_series_heatmaps(
                dataframe, output_dir, plot_config.get("value_label", "Images"), series_key
            )
            continue
        raise ValueError(f"Unsupported figure kind: {kind}")


def add_events(axis, event_items: list[dict]) -> None:
    for event in event_items:
        label = event.get("label", "")
        color = event.get("color", "gray")
        alpha = float(event.get("alpha", 0.15))
        if event["type"] == "band":
            start = pd.to_datetime(event["after"])
            end = pd.to_datetime(event["before"])
            axis.axvspan(start, end, alpha=alpha, color=color)
            if label:
                label_rotation = int(event.get("label_rotation", 90))
                midpoint = start + (end - start) / 2
                axis.text(
                    midpoint,
                    0.98,
                    label,
                    ha="center",
                    va="top",
                    rotation=label_rotation,
                    transform=axis.get_xaxis_transform(),
                )
            continue
        if event["type"] == "marker":
            moment = pd.to_datetime(event["date"])
            axis.axvline(moment, color=color, alpha=max(alpha, 0.4), linestyle="--")
            if label:
                axis.text(
                    moment,
                    0.98,
                    label,
                    ha="left",
                    va="top",
                    rotation=90,
                    transform=axis.get_xaxis_transform(),
                )


def plot_hourly_stacked(
    dataframe: pd.DataFrame,
    output_path: Path,
    series_key: str,
    title: str,
    y_label: str,
    day_origin_hour: int,
    colors_by_series: dict[str, str] | None = None,
) -> None:
    plt.figure(figsize=(12, 6))
    hourly_data = dataframe.groupby(["hour", series_key]).size().unstack(fill_value=0)
    hour_order = get_hour_order(day_origin_hour)
    hourly_data = hourly_data.reindex(index=hour_order, fill_value=0)
    colors = None
    if colors_by_series is not None:
        colors = [colors_by_series.get(str(series_value)) for series_value in hourly_data.columns]
    max_total = hourly_data.sum(axis=1).max()
    if max_total > 0:
        normalized_hourly_data = hourly_data * (100.0 / max_total)
    else:
        normalized_hourly_data = hourly_data
    normalized_hourly_data.plot(kind="bar", stacked=True, alpha=0.8, color=colors)
    plt.title(title)
    plt.xlabel("Hour of Day")
    plt.ylabel(y_label)
    plt.ylim(0, 100)
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_series_heatmaps(
    dataframe: pd.DataFrame,
    output_dir: Path,
    value_label: str,
    series_key: str,
) -> None:
    for series_value in dataframe[series_key].dropna().unique():
        source = dataframe[dataframe[series_key] == series_value]
        if source.empty:
            continue

        data = source.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
        data = data.reindex(index=range(7), columns=range(24), fill_value=0)
        log = pd.DataFrame(
            np.log1p(data.to_numpy()),
            index=data.index,
            columns=data.columns,
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            log,
            cmap="YlOrRd",
            xticklabels=HOUR_LABELS,
            yticklabels=WEEKDAY_LABELS,
            cbar_kws={"label": f"{value_label} (log scale)"},
        )
        colorbar = plt.gca().collections[0].colorbar
        max_tick = int(np.ceil(log.to_numpy().max()))
        colorbar.set_ticks(list(range(0, max_tick + 1)))
        colorbar.set_ticklabels([str(tick) for tick in range(0, max_tick + 1)])
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        plt.title(f"{value_label} Heatmap - {str(series_value).title()}")
        plt.tight_layout()
        series_slug = slugify(series_value)
        plt.savefig(output_dir / f"heatmap-{series_slug}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot(
    dataframe: pd.DataFrame,
    output_dir: str,
    key: str,
    plot_config: dict | None = None,
    data_config: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_dir)
    clean_data = dataframe.dropna(subset=["timestamp"]).copy()
    if clean_data.empty:
        print("No timestamp data available for plotting")
        return

    config = plot_config or {}
    title = config.get("title", key.replace("_", " ").title())
    dir_name = slugify(config.get("output_dirname", key))
    output_dir = output_path / dir_name
    value_label = config.get("value_label", "Images")
    plots = config.get("plots", [])
    figures = config.get("figures", [])
    if not plots and not figures:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    event_items = config.get("event_items", [])
    set_plot_style()
    if figures:
        if data_config is None:
            raise ValueError("data configuration is required for figure rendering")
        render_figures(clean_data, output_dir, config, data_config, event_items)
        return
    series_key = config.get("series_key", "source")
    if series_key not in clean_data.columns:
        raise ValueError(f"series_key '{series_key}' not found in data")
    if "hourly" in plots:
        day_origin_hour = parse_day_origin_hour(config)
        color_map = None
        if data_config is not None:
            color_map = {
                series_id: (item["color"] if isinstance(item, dict) else item.color)
                for series_id, item in data_config.items()
            }
        plot_hourly_stacked(
            clean_data,
            output_dir / "hour.png",
            series_key,
            f"{title} by Hour of Day",
            f"{value_label} (max-normalized to 100)",
            day_origin_hour,
            color_map,
        )
    if "heatmap" in plots:
        plot_series_heatmaps(clean_data, output_dir, value_label, series_key)

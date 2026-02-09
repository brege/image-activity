#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

WEEKDAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
HOUR_LABELS = [f'{hour:02d}' for hour in range(24)]


def set_plot_style() -> None:
    plt.style.use('default')
    sns.set_palette('husl')


def plot_hourly_stacked(
    dataframe: pd.DataFrame,
    output_path: Path,
    series_key: str,
    title: str,
    y_label: str,
    legend_title: str,
) -> None:
    plt.figure(figsize=(12, 6))
    hourly_data = dataframe.groupby(['hour', series_key]).size().unstack(fill_value=0)
    hourly_data.plot(kind='bar', stacked=True, alpha=0.8)
    plt.title(title)
    plt.xlabel('Hour (24-hour format)')
    plt.ylabel(y_label)
    plt.legend(title=legend_title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_series_heatmaps(
    dataframe: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    value_label: str,
    series_key: str,
) -> None:
    for series_value in dataframe[series_key].dropna().unique():
        source_data = dataframe[dataframe[series_key] == series_value]
        if source_data.empty:
            continue

        heatmap_data = source_data.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.reindex(index=range(7), columns=range(24), fill_value=0)
        log_heatmap_data = pd.DataFrame(
            np.log1p(heatmap_data.to_numpy()),
            index=heatmap_data.index,
            columns=heatmap_data.columns,
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            log_heatmap_data,
            cmap='YlOrRd',
            xticklabels=HOUR_LABELS,
            yticklabels=WEEKDAY_LABELS,
            cbar_kws={'label': f'{value_label} (log scale)'},
        )
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.title(f'{value_label} Heatmap - {str(series_value).title()}')
        plt.tight_layout()
        plt.savefig(output_dir / f'{prefix}_heatmap_{series_value}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_daily_timeseries_by_series(
    dataframe: pd.DataFrame,
    output_path: Path,
    series_key: str,
    title: str,
    y_label: str,
    rolling_window: int,
) -> None:
    plt.figure(figsize=(15, 6))
    for series_value in dataframe[series_key].dropna().unique():
        source_data = dataframe[dataframe[series_key] == series_value]
        daily_counts = source_data.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        plt.plot(daily_counts['date'], daily_counts['count'], alpha=0.25, linewidth=0.7)
        plt.plot(
            daily_counts['date'],
            daily_counts['count'].rolling(window=rolling_window, center=True).mean(),
            linewidth=2,
            label=f'{series_value} ({rolling_window}-day avg)',
        )
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_daily_timeseries_total(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
    y_label: str,
    rolling_window: int,
) -> None:
    plt.figure(figsize=(15, 6))
    daily_counts = dataframe.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    plt.plot(daily_counts['date'], daily_counts['count'], alpha=0.3, linewidth=0.7)
    plt.plot(
        daily_counts['date'],
        daily_counts['count'].rolling(window=rolling_window, center=True).mean(),
        linewidth=2,
        label=f'{rolling_window}-day avg',
    )
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trend_month_day_panels(
    dataframe: pd.DataFrame,
    output_path: Path,
    trend_y_label: str,
    month_y_label: str,
    day_y_label: str,
    trend_title: str,
    month_title: str,
    day_title: str,
) -> None:
    plt.figure(figsize=(15, 10))
    month_year_data = dataframe.assign(month_year=dataframe['timestamp'].dt.to_period('M'))
    monthly_counts = month_year_data.groupby('month_year').size().reset_index(name='count')
    monthly_counts['month_year'] = monthly_counts['month_year'].dt.to_timestamp()

    plt.subplot(3, 1, 1)
    plt.plot(monthly_counts['month_year'], monthly_counts['count'], alpha=0.6, linewidth=1)
    plt.plot(
        monthly_counts['month_year'],
        monthly_counts['count'].rolling(window=3, center=True).mean(),
        linewidth=2,
        label='3-month moving average',
    )
    plt.xlabel('Date')
    plt.ylabel(trend_y_label)
    plt.title(trend_title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    month_totals = dataframe.groupby('month').size()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(range(1, 13), [month_totals.get(month, 0) for month in range(1, 13)])
    plt.xlabel('Month')
    plt.ylabel(month_y_label)
    plt.title(month_title)
    plt.xticks(range(1, 13), month_names, rotation=45)
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(3, 1, 3)
    day_totals = dataframe.groupby('day_of_week').size()
    plt.bar(range(7), [day_totals.get(day, 0) for day in range(7)])
    plt.xlabel('Day of Week')
    plt.ylabel(day_y_label)
    plt.title(day_title)
    plt.xticks(range(7), WEEKDAY_LABELS)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_analysis(
    dataframe: pd.DataFrame,
    output_dir: str,
    analysis_key: str,
    plot_config: dict | None = None,
) -> None:
    output_path = Path(output_dir)
    clean_data = dataframe.dropna(subset=['timestamp']).copy()
    if clean_data.empty:
        print("No timestamp data available for plotting")
        return

    config = plot_config or {}
    title = config.get('title', analysis_key.replace('_', ' ').title())
    value_label = config.get('value_label', 'Files')
    selected_plots = config.get('plots', ['hourly', 'timeseries', 'heatmap'])
    timeseries_mode = config.get('timeseries_mode', 'by_source')
    rolling_window = int(config.get('rolling_window', 7))
    series_key = config.get('series_key', 'source')
    if series_key not in clean_data.columns:
        raise ValueError(f"series_key '{series_key}' not found in data")
    legend_title = config.get('legend_title', series_key.title())

    set_plot_style()
    if 'hourly' in selected_plots:
        plot_hourly_stacked(
            clean_data,
            output_path / f'{analysis_key}_hourly.png',
            series_key,
            f'{title} by Hour of Day',
            f'Number of {value_label}',
            legend_title,
        )
    if 'timeseries' in selected_plots:
        timeseries_title = f'{title} Over Time (Smoothed)'
        if timeseries_mode == 'total':
            plot_daily_timeseries_total(
                clean_data,
                output_path / f'{analysis_key}_timeseries.png',
                timeseries_title,
                f'Number of {value_label}',
                rolling_window,
            )
        else:
            plot_daily_timeseries_by_series(
                clean_data,
                output_path / f'{analysis_key}_timeseries.png',
                series_key,
                timeseries_title,
                f'Number of {value_label}',
                rolling_window,
            )
    if 'panel' in selected_plots:
        plot_trend_month_day_panels(
            clean_data,
            output_path / f'{analysis_key}_temporal.png',
            value_label,
            f'Total {value_label}',
            f'Total {value_label}',
            f'{title} Over Time',
            f'{title} by Month',
            f'{title} by Day of Week',
        )
    if 'heatmap' in selected_plots:
        plot_series_heatmaps(clean_data, output_path, analysis_key, value_label, series_key)

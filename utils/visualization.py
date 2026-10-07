import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

def plot_time_series(df, show_rolling_avg=True, rolling_window=7):
    """
    Plot time series of daily patient arrivals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data with datetime and arrivals
    show_rolling_avg : bool
        Whether to show rolling average
    rolling_window : int
        Rolling average window in days
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    # Aggregate to daily
    daily_df = df.groupby('date')['arrivals'].sum().reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    fig = go.Figure()
    
    # Add daily arrivals
    fig.add_trace(go.Scatter(
        x=daily_df['date'],
        y=daily_df['arrivals'],
        mode='lines',
        name='Daily Arrivals',
        line=dict(color='lightblue', width=1),
        opacity=0.7
    ))
    
    # Add rolling average
    if show_rolling_avg:
        daily_df['rolling_avg'] = daily_df['arrivals'].rolling(window=rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['rolling_avg'],
            mode='lines',
            name=f'{rolling_window}-Day Moving Average',
            line=dict(color='darkblue', width=3)
        ))
    
    fig.update_layout(
        title='Patient Arrivals Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Patients',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_heatmap(df):
    """
    Plot heatmap of arrivals by day of week and hour of day.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data with time features
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    # Create pivot table
    heatmap_data = df.groupby(['day_of_week', 'hour'])['arrivals'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='arrivals')
    
    # Reorder days (Monday first)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot.index = day_names
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='YlOrRd',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Avg Arrivals: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Patient Arrivals by Day and Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_day_of_week_pattern(df):
    """
    Plot average arrivals by day of week.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data with time features
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    day_pattern = df.groupby('day_name')['arrivals'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ).reset_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=day_pattern['day_name'],
            y=day_pattern['arrivals'],
            marker_color=['#1f77b4' if day not in ['Saturday', 'Sunday'] else '#ff7f0e' 
                         for day in day_pattern['day_name']],
            text=day_pattern['arrivals'].round(1),
            textposition='outside',
            hovertemplate='%{x}<br>Avg Arrivals: %{y:.1f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Average Arrivals by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Average Arrivals per Hour',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_hourly_pattern(df):
    """
    Plot average arrivals by hour of day.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data with time features
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    hourly_pattern = df.groupby('hour')['arrivals'].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=hourly_pattern['hour'],
        y=hourly_pattern['mean'],
        mode='lines+markers',
        name='Average',
        line=dict(color='darkblue', width=3),
        marker=dict(size=8)
    ))
    
    # Add confidence band (mean ± std)
    fig.add_trace(go.Scatter(
        x=hourly_pattern['hour'].tolist() + hourly_pattern['hour'].tolist()[::-1],
        y=(hourly_pattern['mean'] + hourly_pattern['std']).tolist() + 
          (hourly_pattern['mean'] - hourly_pattern['std']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Std Dev',
        showlegend=True
    ))
    
    fig.update_layout(
        title='Average Arrivals by Hour of Day',
        xaxis_title='Hour of Day',
        yaxis_title='Average Arrivals',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_monthly_pattern(df):
    """
    Plot average arrivals by month.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data with time features
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    monthly_pattern = df.groupby('month')['arrivals'].mean().reset_index()
    monthly_pattern['month_name'] = monthly_pattern['month'].apply(
        lambda x: datetime(2000, x, 1).strftime('%B')
    )
    
    fig = go.Figure(data=[
        go.Bar(
            x=monthly_pattern['month_name'],
            y=monthly_pattern['arrivals'],
            marker_color='#2ecc71',
            text=monthly_pattern['arrivals'].round(1),
            textposition='outside',
            hovertemplate='%{x}<br>Avg Arrivals: %{y:.1f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Average Arrivals by Month',
        xaxis_title='Month',
        yaxis_title='Average Arrivals per Hour',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_forecast_comparison(historical_df, forecast_df, sample_days=30):
    """
    Plot comparison between historical and forecasted data.
    
    Parameters:
    -----------
    historical_df : pd.DataFrame
        Historical data
    forecast_df : pd.DataFrame
        Forecasted data
    sample_days : int
        Number of days to show from forecast
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    # Aggregate to daily
    hist_daily = historical_df.groupby('date')['arrivals'].sum().reset_index()
    hist_daily['date'] = pd.to_datetime(hist_daily['date'])
    hist_daily = hist_daily.tail(sample_days)  # Last N days of historical
    
    forecast_daily = forecast_df.groupby(forecast_df['datetime'].dt.date)['forecasted_arrivals'].sum().reset_index()
    forecast_daily.columns = ['date', 'arrivals']
    forecast_daily['date'] = pd.to_datetime(forecast_daily['date'])
    forecast_daily = forecast_daily.head(sample_days)  # First N days of forecast
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_daily['date'],
        y=hist_daily['arrivals'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Forecasted data
    fig.add_trace(go.Scatter(
        x=forecast_daily['date'],
        y=forecast_daily['arrivals'],
        mode='lines+markers',
        name='Forecasted',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'Historical vs Forecasted Daily Arrivals (Sample: {sample_days} days)',
        xaxis_title='Date',
        yaxis_title='Daily Patient Arrivals',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_staffing_vs_demand(optimization_results):
    """
    Plot staffing capacity vs demand with rolling demand visualization.
    
    Parameters:
    -----------
    optimization_results : dict
        Results from optimization containing schedule and metrics
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    schedule = optimization_results['schedule'].copy()
    
    fig = go.Figure()
    
    # Add stacked bar for patients seen (green)
    fig.add_trace(go.Bar(
        x=schedule['datetime'],
        y=schedule['patients_seen'],
        name='Patients Seen',
        marker_color='green',
        hovertemplate='%{x}<br>Patients Seen: %{y:.0f}<extra></extra>'
    ))
    
    # Add stacked bar for unmet demand (red)
    fig.add_trace(go.Bar(
        x=schedule['datetime'],
        y=schedule['unmet_demand'],
        name='Unmet Demand',
        marker_color='red',
        hovertemplate='%{x}<br>Unmet Demand: %{y:.0f}<extra></extra>'
    ))
    
    # Add line for rolling demand
    fig.add_trace(go.Scatter(
        x=schedule['datetime'],
        y=schedule['rolling_demand'],
        name='Rolling Demand',
        mode='lines',
        line=dict(color='orange', width=3),
        yaxis='y2',
        hovertemplate='%{x}<br>Rolling Demand: %{y:.1f}<extra></extra>'
    ))
    
    # Add line for total capacity
    fig.add_trace(go.Scatter(
        x=schedule['datetime'],
        y=schedule['total_capacity'],
        name='Total Capacity',
        mode='lines',
        line=dict(color='blue', width=3),
        yaxis='y2',
        hovertemplate='%{x}<br>Total Capacity: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Staffing Capacity vs Patient Demand',
        xaxis_title='Date & Time',
        yaxis_title='Patients (Stacked)',
        yaxis2=dict(
            title='Capacity / Demand (Lines)',
            overlaying='y',
            side='right'
        ),
        barmode='stack',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig


def plot_provider_schedule(schedule_df, provider_type='MD'):
    """
    Plot schedule for a specific provider type.
    
    Parameters:
    -----------
    schedule_df : pd.DataFrame
        Schedule dataframe
    provider_type : str
        Provider type to plot (e.g., 'MD', 'APP')
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=schedule_df['datetime'],
        y=schedule_df[provider_type],
        mode='lines+markers',
        name=provider_type,
        line=dict(width=2),
        marker=dict(size=4),
        fill='tozeroy',
        hovertemplate=f'%{{x}}<br>{provider_type}s on Duty: %{{y}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{provider_type} Staffing Schedule',
        xaxis_title='Date & Time',
        yaxis_title=f'Number of {provider_type}s',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig

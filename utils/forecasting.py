import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def forecast_demand(historical_data, forecast_year=2026, annual_growth_rate=0.05, 
                   percentile=0.75, include_seasonality=True):
    """
    Forecast patient demand for a future year based on historical patterns.
    
    Parameters:
    -----------
    historical_data : pd.DataFrame
        Historical patient arrival data with datetime and arrivals columns
    forecast_year : int
        Year to forecast
    annual_growth_rate : float
        Annual growth rate (e.g., 0.05 for 5%)
    percentile : float
        Percentile of historical demand to use (e.g., 0.75 for 75th percentile)
    include_seasonality : bool
        Whether to include monthly seasonality factors
    
    Returns:
    --------
    forecast_df : pd.DataFrame
        Forecasted demand with hourly granularity for the entire year
    """
    df = historical_data.copy()
    
    # Calculate years between historical data and forecast
    historical_year = df['datetime'].dt.year.mode()[0]
    years_ahead = forecast_year - historical_year
    
    # Calculate growth multiplier
    growth_multiplier = (1 + annual_growth_rate) ** years_ahead
    
    # Calculate baseline hourly patterns (by hour of day and day of week)
    hourly_patterns = df.groupby(['day_of_week', 'hour'])['arrivals'].quantile(percentile).reset_index()
    hourly_patterns.columns = ['day_of_week', 'hour', 'baseline_arrivals']
    
    # Calculate monthly seasonality factors if requested
    if include_seasonality:
        monthly_avg = df.groupby('month')['arrivals'].mean()
        overall_avg = df['arrivals'].mean()
        seasonality_factors = (monthly_avg / overall_avg).to_dict()
    else:
        seasonality_factors = {i: 1.0 for i in range(1, 13)}
    
    # Create forecast dataframe for entire year
    start_date = datetime(forecast_year, 1, 1, 0, 0, 0)
    end_date = datetime(forecast_year, 12, 31, 23, 0, 0)
    
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    forecast_df = pd.DataFrame({'datetime': forecast_dates})
    
    # Add time features
    forecast_df['month'] = forecast_df['datetime'].dt.month
    forecast_df['day_of_week'] = forecast_df['datetime'].dt.dayofweek
    forecast_df['hour'] = forecast_df['datetime'].dt.hour
    forecast_df['day_name'] = forecast_df['datetime'].dt.day_name()
    
    # Merge with baseline patterns
    forecast_df = forecast_df.merge(hourly_patterns, on=['day_of_week', 'hour'], how='left')
    
    # Apply seasonality
    forecast_df['seasonality_factor'] = forecast_df['month'].map(seasonality_factors)
    
    # Calculate forecasted arrivals
    forecast_df['forecasted_arrivals'] = (
        forecast_df['baseline_arrivals'] * 
        forecast_df['seasonality_factor'] * 
        growth_multiplier
    )
    
    # Round to reasonable values
    forecast_df['forecasted_arrivals'] = forecast_df['forecasted_arrivals'].round(2)
    
    # Fill any missing values with 0
    forecast_df['forecasted_arrivals'] = forecast_df['forecasted_arrivals'].fillna(0)
    
    # Add metadata columns
    forecast_df['growth_multiplier'] = growth_multiplier
    forecast_df['percentile_used'] = percentile
    
    return forecast_df


def calculate_rolling_demand(demand_series, window=24):
    """
    Calculate rolling average demand.
    
    Parameters:
    -----------
    demand_series : pd.Series
        Series of demand values
    window : int
        Rolling window size in hours
    
    Returns:
    --------
    rolling_demand : pd.Series
        Rolling average demand
    """
    return demand_series.rolling(window=window, min_periods=1).mean()


def get_peak_hours(forecast_df, top_n=10):
    """
    Identify peak demand hours across the forecast period.
    
    Parameters:
    -----------
    forecast_df : pd.DataFrame
        Forecasted demand data
    top_n : int
        Number of top peak hours to return
    
    Returns:
    --------
    peak_hours : pd.DataFrame
        Top peak demand periods
    """
    peak_hours = forecast_df.nlargest(top_n, 'forecasted_arrivals')[
        ['datetime', 'forecasted_arrivals', 'day_name', 'hour']
    ].copy()
    
    return peak_hours


def compare_forecast_to_historical(historical_data, forecast_data):
    """
    Compare forecasted demand to historical patterns.
    
    Parameters:
    -----------
    historical_data : pd.DataFrame
        Historical data
    forecast_data : pd.DataFrame
        Forecasted data
    
    Returns:
    --------
    comparison : dict
        Dictionary with comparison metrics
    """
    historical_daily_avg = historical_data.groupby(historical_data['datetime'].dt.date)['arrivals'].sum().mean()
    forecast_daily_avg = forecast_data.groupby(forecast_data['datetime'].dt.date)['forecasted_arrivals'].sum().mean()
    
    historical_total = historical_data['arrivals'].sum()
    forecast_total = forecast_data['forecasted_arrivals'].sum()
    
    # Normalize to annual if historical data is not a full year
    days_in_historical = (historical_data['datetime'].max() - historical_data['datetime'].min()).days
    if days_in_historical < 365:
        historical_annual = historical_total * (365 / days_in_historical)
    else:
        historical_annual = historical_total
    
    comparison = {
        'historical_daily_avg': historical_daily_avg,
        'forecast_daily_avg': forecast_daily_avg,
        'daily_change': forecast_daily_avg - historical_daily_avg,
        'daily_change_pct': ((forecast_daily_avg / historical_daily_avg) - 1) * 100,
        'historical_annual_estimate': historical_annual,
        'forecast_annual_total': forecast_total,
        'annual_change': forecast_total - historical_annual,
        'annual_change_pct': ((forecast_total / historical_annual) - 1) * 100
    }
    
    return comparison


def get_monthly_forecast_summary(forecast_df):
    """
    Summarize forecast by month.
    
    Parameters:
    -----------
    forecast_df : pd.DataFrame
        Forecasted demand data
    
    Returns:
    --------
    monthly_summary : pd.DataFrame
        Monthly aggregated forecast
    """
    monthly_summary = forecast_df.groupby('month').agg({
        'forecasted_arrivals': ['sum', 'mean', 'std', 'min', 'max']
    }).round(2)
    
    monthly_summary.columns = ['Total', 'Hourly Mean', 'Hourly Std', 'Hourly Min', 'Hourly Max']
    monthly_summary['Month'] = [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]
    monthly_summary = monthly_summary[['Month', 'Total', 'Hourly Mean', 'Hourly Std', 'Hourly Min', 'Hourly Max']]
    
    return monthly_summary

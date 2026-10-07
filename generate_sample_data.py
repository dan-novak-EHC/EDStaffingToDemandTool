"""
Generate sample patient arrival data for testing the ED Staffing Optimizer.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(start_date='2025-01-01', end_date='2025-12-31', output_file='sample_ed_data.xlsx'):
    """
    Generate realistic sample ED patient arrival data.
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    output_file : str
        Output filename
    """
    # Create hourly datetime range
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Initialize dataframe
    df = pd.DataFrame({'datetime': date_range})
    
    # Extract time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Base arrival rate (average patients per hour)
    base_rate = 8
    
    # Hourly pattern (multiplier by hour of day)
    hourly_pattern = {
        0: 0.4, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.5,
        6: 0.7, 7: 0.9, 8: 1.1, 9: 1.3, 10: 1.4, 11: 1.4,
        12: 1.3, 13: 1.2, 14: 1.2, 15: 1.3, 16: 1.4, 17: 1.5,
        18: 1.4, 19: 1.3, 20: 1.2, 21: 1.0, 22: 0.8, 23: 0.6
    }
    df['hourly_multiplier'] = df['hour'].map(hourly_pattern)
    
    # Day of week pattern (multiplier)
    dow_pattern = {
        0: 1.0,  # Monday
        1: 0.95,  # Tuesday
        2: 0.95,  # Wednesday
        3: 1.0,   # Thursday
        4: 1.05,  # Friday
        5: 1.1,   # Saturday
        6: 1.15   # Sunday
    }
    df['dow_multiplier'] = df['day_of_week'].map(dow_pattern)
    
    # Monthly seasonality (winter months busier)
    monthly_pattern = {
        1: 1.15,  # January
        2: 1.12,  # February
        3: 1.05,  # March
        4: 0.98,  # April
        5: 0.95,  # May
        6: 0.93,  # June
        7: 0.95,  # July
        8: 0.97,  # August
        9: 1.00,  # September
        10: 1.05, # October
        11: 1.10, # November
        12: 1.18  # December
    }
    df['monthly_multiplier'] = df['month'].map(monthly_pattern)
    
    # Calculate expected arrivals
    df['expected_arrivals'] = (
        base_rate * 
        df['hourly_multiplier'] * 
        df['dow_multiplier'] * 
        df['monthly_multiplier']
    )
    
    # Add random variation (Poisson distribution)
    np.random.seed(42)
    df['arrivals'] = np.random.poisson(df['expected_arrivals'])
    
    # Ensure non-negative
    df['arrivals'] = df['arrivals'].clip(lower=0)
    
    # Select final columns
    output_df = df[['datetime', 'arrivals']].copy()
    
    # Save to Excel
    output_df.to_excel(output_file, index=False)
    
    print(f"✅ Sample data generated successfully!")
    print(f"📁 File: {output_file}")
    print(f"📊 Records: {len(output_df):,}")
    print(f"📅 Date range: {output_df['datetime'].min()} to {output_df['datetime'].max()}")
    print(f"👥 Total arrivals: {output_df['arrivals'].sum():,}")
    print(f"📈 Average hourly arrivals: {output_df['arrivals'].mean():.2f}")
    
    return output_df

if __name__ == "__main__":
    generate_sample_data()

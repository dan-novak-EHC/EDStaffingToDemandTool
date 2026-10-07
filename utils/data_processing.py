import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_validate_data(file):
    """
    Load and validate patient arrival data from Excel file.
    Handles both pre-aggregated hourly data and patient-level data.
    
    Parameters:
    -----------
    file : UploadedFile
        Streamlit uploaded file object
    
    Returns:
    --------
    df : pd.DataFrame
        Validated dataframe with standardized columns
    validation_results : dict
        Dictionary containing validation results and issues
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'is_patient_level': False,
        'detected_datetime_columns': []
    }
    
    try:
        # Read the Excel file
        df = pd.read_excel(file, engine='openpyxl')
        
        validation_results['original_rows'] = len(df)
        validation_results['original_columns'] = len(df.columns)
        
        # Detect all potential datetime columns
        datetime_candidates = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'instant', 'timestamp', 'arrival']):
                datetime_candidates.append(col)
        
        validation_results['detected_datetime_columns'] = datetime_candidates
        
        if len(datetime_candidates) == 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append("No datetime column found. Please include a column with 'date', 'time', or 'arrival' in the name.")
            return None, validation_results
        
        # Check if this is patient-level data (many columns, likely one row per patient)
        if len(df.columns) > 5:
            validation_results['is_patient_level'] = True
            validation_results['warnings'].append(f"Detected patient-level data with {len(df):,} records and {len(df.columns)} columns.")
            
            # Use the first datetime candidate (or most likely one)
            datetime_col = None
            for candidate in ['Arrival Instant', 'Arrival Date', 'Arrival Time', 'Arrival DateTime']:
                if candidate in datetime_candidates:
                    datetime_col = candidate
                    break
            
            if datetime_col is None:
                datetime_col = datetime_candidates[0]
            
            validation_results['warnings'].append(f"Using '{datetime_col}' as the arrival datetime column.")
            
            # Only keep the datetime column to reduce memory
            df_small = df[[datetime_col]].copy()
            del df  # Free memory
            
            # Convert to datetime
            df_small['datetime'] = pd.to_datetime(df_small[datetime_col], errors='coerce')
            
            # Remove rows with invalid datetimes
            invalid_dates = df_small['datetime'].isna().sum()
            if invalid_dates > 0:
                validation_results['warnings'].append(f"Removed {invalid_dates} records with invalid datetime values.")
                df_small = df_small.dropna(subset=['datetime'])
            
            # Round to hour and count arrivals
            df_small['hour'] = df_small['datetime'].dt.floor('H')
            hourly_df = df_small.groupby('hour').size().reset_index()
            hourly_df.columns = ['datetime', 'arrivals']
            
            validation_results['warnings'].append(f"Aggregated {len(df_small):,} patient records into {len(hourly_df):,} hourly records.")
            
            df = hourly_df
        
        else:
            # Pre-aggregated data - find datetime and arrivals columns
            datetime_col = datetime_candidates[0] if datetime_candidates else None
            
            if datetime_col is None:
                validation_results['is_valid'] = False
                validation_results['issues'].append("No datetime column found.")
                return None, validation_results
            
            # Detect arrivals column
            arrivals_col = None
            for col in df.columns:
                if col != datetime_col and any(keyword in str(col).lower() for keyword in ['arrival', 'patient', 'count', 'volume', 'visits']):
                    arrivals_col = col
                    break
            
            if arrivals_col is None:
                # Try to find any numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    arrivals_col = numeric_cols[0]
                    validation_results['warnings'].append(f"Using '{arrivals_col}' as arrivals column. Please verify this is correct.")
                else:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append("No arrivals/patient count column found.")
                    return None, validation_results
            
            # Standardize column names
            df = df[[datetime_col, arrivals_col]].copy()
            df.columns = ['datetime', 'arrivals']
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Convert arrivals to numeric
            df['arrivals'] = pd.to_numeric(df['arrivals'], errors='coerce')
        
        # Common validation for both types
        
        # Check for missing values
        missing_datetime = df['datetime'].isna().sum()
        missing_arrivals = df['arrivals'].isna().sum()
        
        if missing_datetime > 0:
            validation_results['warnings'].append(f"Found {missing_datetime} missing datetime values. These rows will be removed.")
            df = df.dropna(subset=['datetime'])
        
        if missing_arrivals > 0:
            validation_results['warnings'].append(f"Found {missing_arrivals} missing arrival values. These will be filled with 0.")
            df['arrivals'] = df['arrivals'].fillna(0)
        
        # Remove duplicates
        duplicates = df.duplicated(subset=['datetime']).sum()
        if duplicates > 0:
            validation_results['warnings'].append(f"Found {duplicates} duplicate datetime entries. Keeping first occurrence.")
            df = df.drop_duplicates(subset=['datetime'], keep='first')
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Check data range
        date_range = (df['datetime'].max() - df['datetime'].min()).days
        if date_range < 30:
            validation_results['warnings'].append(f"Data range is only {date_range} days. Recommend at least 365 days for accurate forecasting.")
        
        # Add time-based features
        df = prepare_hourly_data(df)
        
        validation_results['total_records'] = len(df)
        validation_results['date_range_days'] = date_range
        validation_results['start_date'] = df['datetime'].min().strftime('%Y-%m-%d')
        validation_results['end_date'] = df['datetime'].max().strftime('%Y-%m-%d')
        validation_results['total_arrivals'] = int(df['arrivals'].sum())
        
        return df, validation_results
    
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Error reading file: {str(e)}")
        import traceback
        validation_results['issues'].append(f"Details: {traceback.format_exc()}")
        return None, validation_results
def prepare_hourly_data(df):
    """
    Add time-based features to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'datetime' and 'arrivals' columns
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with additional time-based features
    """
    df = df.copy()
    
    # Extract time components
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['datetime'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Week of year
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    
    return df

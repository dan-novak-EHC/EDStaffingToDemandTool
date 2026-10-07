import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from datetime import datetime, timedelta

def calculate_provider_capacity(start_hour, shift_length, productivity_pattern):
    """
    Calculate hourly capacity for a provider starting at a given hour.
    
    Parameters:
    -----------
    start_hour : int
        Hour when provider starts (0-23)
    shift_length : int
        Length of shift in hours
    productivity_pattern : list
        List of patients per hour for each hour of the shift
    
    Returns:
    --------
    capacity_dict : dict
        Dictionary mapping hour to capacity contribution
    """
    capacity_dict = {}
    for i in range(shift_length):
        hour = (start_hour + i) % 24
        capacity_dict[hour] = productivity_pattern[i]
    return capacity_dict


def calculate_rolling_demand(demand_array, capacity_array):
    """
    Calculate rolling demand accounting for carryover of unmet demand.
    
    Parameters:
    -----------
    demand_array : np.array
        Array of new patient arrivals per hour
    capacity_array : np.array
        Array of provider capacity per hour
    
    Returns:
    --------
    rolling_demand : np.array
        Rolling demand including carryover
    patients_seen : np.array
        Patients actually seen each hour
    unmet_demand : np.array
        Unmet demand each hour
    """
    n_hours = len(demand_array)
    rolling_demand = np.zeros(n_hours)
    patients_seen = np.zeros(n_hours)
    unmet_demand = np.zeros(n_hours)
    carryover = 0
    
    for i in range(n_hours):
        # Total demand = new arrivals + carryover from previous hour
        rolling_demand[i] = demand_array[i] + carryover
        
        # Patients seen = min(rolling demand, capacity)
        patients_seen[i] = min(rolling_demand[i], capacity_array[i])
        
        # Unmet demand
        unmet_demand[i] = rolling_demand[i] - patients_seen[i]
        
        # Carryover to next hour
        carryover = unmet_demand[i]
    
    return rolling_demand, patients_seen, unmet_demand


def optimize_staffing(demand_data, provider_types, optimization_goal='minimize_rmse'):
    """
    Optimize staffing schedule to meet demand.
    
    Parameters:
    -----------
    demand_data : pd.DataFrame
        Forecasted demand data with datetime and forecasted_arrivals
    provider_types : dict
        Dictionary of provider configurations
        Example: {
            'MD': {
                'shift_length': 8,
                'productivity': [3, 2, 2, 2, 2, 2, 2, 1],
                'cost_per_hour': 200,
                'min_per_hour': 1,
                'max_per_hour': 10
            },
            'APP': {...}
        }
    optimization_goal : str
        Goal to optimize: 'minimize_rmse', 'minimize_total_hours', 
        'minimize_cost', 'minimize_understaffing'
    
    Returns:
    --------
    results : dict
        Dictionary containing optimized schedule and metrics
    """
    # Prepare demand array
    demand_array = demand_data['forecasted_arrivals'].values
    n_hours = len(demand_array)
    
    # Get provider type names
    provider_names = list(provider_types.keys())
    n_provider_types = len(provider_names)
    
    # Calculate number of decision variables
    # For each hour and each provider type, we decide how many providers to start
    n_vars = n_hours * n_provider_types
    
    # Set bounds for decision variables (number of providers starting each hour)
    bounds = []
    for _ in range(n_hours):
        for provider_name in provider_names:
            max_starts = provider_types[provider_name]['max_per_hour']
            bounds.append((0, max_starts))
    
    def objective_function(x):
        """Calculate objective based on staffing schedule."""
        # Reshape x to (n_hours, n_provider_types)
        schedule = x.reshape(n_hours, n_provider_types)
        
        # Calculate total capacity for each hour
        total_capacity = np.zeros(n_hours)
        total_cost = 0
        total_hours = 0
        
        for hour in range(n_hours):
            for p_idx, provider_name in enumerate(provider_names):
                provider_config = provider_types[provider_name]
                shift_length = provider_config['shift_length']
                productivity = provider_config['productivity']
                cost_per_hour = provider_config['cost_per_hour']
                
                # Number of providers starting at this hour
                n_starting = schedule[hour, p_idx]
                
                # Add their capacity to relevant hours
                for shift_hour in range(shift_length):
                    target_hour = (hour + shift_hour) % n_hours
                    if target_hour < n_hours:  # Within our planning horizon
                        total_capacity[target_hour] += n_starting * productivity[shift_hour]
                
                # Add to cost and hours
                total_cost += n_starting * shift_length * cost_per_hour
                total_hours += n_starting * shift_length
        
        # Calculate rolling demand and metrics
        rolling_demand, patients_seen, unmet_demand = calculate_rolling_demand(
            demand_array, total_capacity
        )
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((total_capacity - rolling_demand) ** 2))
        
        # Calculate total understaffing
        total_understaffing = np.sum(unmet_demand)
        
        # Return objective based on goal
        if optimization_goal == 'minimize_rmse':
            return rmse
        elif optimization_goal == 'minimize_total_hours':
            # Penalize understaffing heavily
            penalty = 1000 * total_understaffing
            return total_hours + penalty
        elif optimization_goal == 'minimize_cost':
            penalty = 1000 * total_understaffing
            return total_cost + penalty
        elif optimization_goal == 'minimize_understaffing':
            return total_understaffing
        else:
            return rmse
    
    def constraint_min_providers(x):
        """Ensure minimum providers per hour."""
        schedule = x.reshape(n_hours, n_provider_types)
        violations = []
        
        for hour in range(n_hours):
            for p_idx, provider_name in enumerate(provider_names):
                provider_config = provider_types[provider_name]
                min_required = provider_config['min_per_hour']
                
                # Count how many of this provider type are working at this hour
                n_working = 0
                for start_hour in range(n_hours):
                    shift_length = provider_config['shift_length']
                    if start_hour <= hour < start_hour + shift_length:
                        n_working += schedule[start_hour, p_idx]
                    # Handle wrap-around for multi-day schedules
                    elif hour < (start_hour + shift_length) % n_hours and start_hour + shift_length > n_hours:
                        n_working += schedule[start_hour, p_idx]
                
                # Violation if below minimum
                violations.append(n_working - min_required)
        
        return min(violations)  # All must be >= 0
    
    # Initial guess: simple heuristic based on demand
    x0 = np.zeros(n_vars)
    for hour in range(min(n_hours, 168)):  # First week or less
        for p_idx, provider_name in enumerate(provider_names):
            # Start with minimum required
            x0[hour * n_provider_types + p_idx] = provider_types[provider_name]['min_per_hour']
    
    # Run optimization
    print("Starting optimization...")
    
    # Use differential evolution for global optimization (better for complex problems)
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        maxiter=100,
        popsize=15,
        seed=42,
        workers=1,
        updating='deferred',
        polish=True
    )
    
    optimal_schedule = result.x.reshape(n_hours, n_provider_types)
    
    # Round to whole numbers
    optimal_schedule = np.round(optimal_schedule).astype(int)
    
    # Build detailed schedule dataframe
    schedule_df = demand_data[['datetime', 'forecasted_arrivals']].copy()
    schedule_df.columns = ['datetime', 'demand']
    
    # Add provider counts
    for p_idx, provider_name in enumerate(provider_names):
        schedule_df[provider_name] = 0
    
    # Calculate providers working each hour
    for hour in range(n_hours):
        for p_idx, provider_name in enumerate(provider_names):
            provider_config = provider_types[provider_name]
            shift_length = provider_config['shift_length']
            
            # Count providers working at this hour
            n_working = 0
            for start_hour in range(max(0, hour - shift_length + 1), hour + 1):
                if start_hour >= 0 and start_hour < n_hours:
                    n_working += optimal_schedule[start_hour, p_idx]
            
            schedule_df.loc[hour, provider_name] = n_working
    
    # Calculate total capacity
    total_capacity = np.zeros(n_hours)
    for hour in range(n_hours):
        for p_idx, provider_name in enumerate(provider_names):
            provider_config = provider_types[provider_name]
            shift_length = provider_config['shift_length']
            productivity = provider_config['productivity']
            
            # For each provider working at this hour, determine their productivity
            for start_hour in range(max(0, hour - shift_length + 1), hour + 1):
                if start_hour >= 0 and start_hour < n_hours:
                    n_providers = optimal_schedule[start_hour, p_idx]
                    hours_into_shift = hour - start_hour
                    if hours_into_shift < len(productivity):
                        total_capacity[hour] += n_providers * productivity[hours_into_shift]
    
    schedule_df['total_capacity'] = total_capacity
    
    # Calculate rolling demand and outcomes
    rolling_demand, patients_seen, unmet_demand = calculate_rolling_demand(
        demand_array, total_capacity
    )
    
    schedule_df['rolling_demand'] = rolling_demand
    schedule_df['patients_seen'] = patients_seen
    schedule_df['unmet_demand'] = unmet_demand
    
    # Calculate metrics
    metrics = calculate_staffing_metrics(schedule_df, provider_types, optimal_schedule)
    
    results = {
        'schedule': schedule_df,
        'optimal_staffing': optimal_schedule,
        'metrics': metrics,
        'provider_types': provider_types,
        'optimization_goal': optimization_goal
    }
    
    return results


def calculate_staffing_metrics(schedule_df, provider_types, optimal_schedule):
    """
    Calculate performance metrics for staffing schedule.
    
    Parameters:
    -----------
    schedule_df : pd.DataFrame
        Schedule with demand, capacity, and outcomes
    provider_types : dict
        Provider type configurations
    optimal_schedule : np.array
        Optimal staffing decisions
    
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # RMSE
    rmse = np.sqrt(np.mean((schedule_df['total_capacity'] - schedule_df['rolling_demand']) ** 2))
    
    # Total staff hours
    total_hours = 0
    for p_idx, provider_name in enumerate(provider_types.keys()):
        shift_length = provider_types[provider_name]['shift_length']
        total_starts = np.sum(optimal_schedule[:, p_idx])
        total_hours += total_starts * shift_length
    
    # Total cost
    total_cost = 0
    for p_idx, provider_name in enumerate(provider_types.keys()):
        shift_length = provider_types[provider_name]['shift_length']
        cost_per_hour = provider_types[provider_name]['cost_per_hour']
        total_starts = np.sum(optimal_schedule[:, p_idx])
        total_cost += total_starts * shift_length * cost_per_hour
    
    # Coverage metrics
    total_demand = schedule_df['rolling_demand'].sum()
    total_seen = schedule_df['patients_seen'].sum()
    coverage_rate = (total_seen / total_demand * 100) if total_demand > 0 else 0
    
    # Understaffing
    avg_understaffing = schedule_df['unmet_demand'].mean()
    max_understaffing = schedule_df['unmet_demand'].max()
    
    # Overstaffing
    overstaffing = schedule_df['total_capacity'] - schedule_df['rolling_demand']
    overstaffing[overstaffing < 0] = 0
    avg_overstaffing = overstaffing.mean()
    
    metrics = {
        'rmse': rmse,
        'total_staff_hours': total_hours,
        'total_cost': total_cost,
        'coverage_rate': coverage_rate,
        'total_demand': total_demand,
        'total_patients_seen': total_seen,
        'total_unmet_demand': schedule_df['unmet_demand'].sum(),
        'avg_understaffing': avg_understaffing,
        'max_understaffing': max_understaffing,
        'avg_overstaffing': avg_overstaffing
    }
    
    return metrics

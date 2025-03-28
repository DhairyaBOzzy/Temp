import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

@st.cache_data
def calculate_driver_consistency(_laps_df, driver_code):
    """
    Calculate a driver's consistency metric based on lap time variations
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    driver_code : str
        Driver code (abbreviation)
        
    Returns:
    --------
    float
        Consistency score (lower is more consistent)
    """
    # Filter valid lap times for the specific driver
    driver_laps = _laps_df[(_laps_df['Driver'] == driver_code) & 
                          (_laps_df['IsPersonalBest'] == False) & 
                          (_laps_df['PitOutTime'].isnull()) &
                          (_laps_df['PitInTime'].isnull())]
    
    if len(driver_laps) < 5:
        return None, None
    
    # Calculate coefficient of variation (normalized standard deviation)
    lap_times_sec = driver_laps['LapTime'].dt.total_seconds()
    
    # Remove outliers (laps that are more than 2 standard deviations from the mean)
    mean_lap = np.mean(lap_times_sec)
    std_lap = np.std(lap_times_sec)
    filtered_times = lap_times_sec[(lap_times_sec > mean_lap - 2*std_lap) & 
                                   (lap_times_sec < mean_lap + 2*std_lap)]
    
    if len(filtered_times) < 5:
        return None, None
    
    consistency = np.std(filtered_times) / np.mean(filtered_times) * 100
    
    # Also calculate pace (average lap time)
    avg_pace = np.mean(filtered_times)
    
    return consistency, avg_pace

@st.cache_data
def calculate_team_performance_metrics(_laps_df, team_name):
    """
    Calculate various performance metrics for a team
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    team_name : str
        Team name
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    # Filter laps for the specific team
    team_laps = _laps_df[_laps_df['Team'] == team_name]
    
    if team_laps.empty:
        return None
    
    metrics = {}
    
    # Best lap time
    if not team_laps['LapTime'].isnull().all():
        best_lap_idx = team_laps['LapTime'].dropna().idxmin()
        if pd.notna(best_lap_idx):
            best_lap = team_laps.loc[best_lap_idx]
            metrics['best_lap_time'] = best_lap['LapTime']
            metrics['best_lap_driver'] = best_lap['Driver']
    
    # Average lap time (excluding pit stops and outliers)
    valid_laps = team_laps[(team_laps['PitOutTime'].isnull()) & 
                           (team_laps['PitInTime'].isnull())]
    
    if not valid_laps.empty and not valid_laps['LapTime'].isnull().all():
        lap_times_sec = valid_laps['LapTime'].dropna().dt.total_seconds()
        
        # Remove outliers
        if len(lap_times_sec) > 5:
            mean_lap = np.mean(lap_times_sec)
            std_lap = np.std(lap_times_sec)
            filtered_times = lap_times_sec[(lap_times_sec > mean_lap - 2*std_lap) & 
                                          (lap_times_sec < mean_lap + 2*std_lap)]
            metrics['avg_lap_time'] = pd.Timedelta(seconds=np.mean(filtered_times))
            metrics['lap_time_std'] = pd.Timedelta(seconds=np.std(filtered_times))
    
    # Pit stop metrics
    pit_laps = team_laps[team_laps['PitInTime'].notnull()]
    if not pit_laps.empty:
        metrics['total_pit_stops'] = len(pit_laps)
        if 'PitOutTime' in pit_laps.columns and 'PitInTime' in pit_laps.columns:
            pit_times = []
            for _, lap in pit_laps.iterrows():
                if pd.notnull(lap['PitOutTime']) and pd.notnull(lap['PitInTime']):
                    pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
                    if pit_duration > 0:
                        pit_times.append(pit_duration)
            
            if pit_times:
                metrics['avg_pit_time'] = pd.Timedelta(seconds=np.mean(pit_times))
                metrics['min_pit_time'] = pd.Timedelta(seconds=np.min(pit_times))
    
    return metrics

@st.cache_data
def analyze_race_pace_degradation(_laps_df, driver_code):
    """
    Analyze how a driver's pace degrades throughout a race
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    driver_code : str
        Driver code (abbreviation)
        
    Returns:
    --------
    tuple
        (slope, pace_degradation_percentage, r_value, lap_number_array, lap_times_array)
    """
    # Filter valid lap times for the specific driver
    driver_laps = _laps_df[(_laps_df['Driver'] == driver_code) & 
                          (_laps_df['PitOutTime'].isnull()) & 
                          (_laps_df['PitInTime'].isnull())]
    
    if len(driver_laps) < 10:
        return None, None, None, None, None
    
    # Convert lap times to seconds
    lap_times_sec = driver_laps['LapTime'].dt.total_seconds()
    lap_numbers = driver_laps['LapNumber'].values
    
    # Remove outliers (laps that are more than 2 standard deviations from the mean)
    mean_lap = np.mean(lap_times_sec)
    std_lap = np.std(lap_times_sec)
    
    # Create mask for valid laps
    valid_mask = (lap_times_sec > mean_lap - 2*std_lap) & (lap_times_sec < mean_lap + 2*std_lap)
    
    filtered_times = lap_times_sec[valid_mask]
    filtered_lap_numbers = lap_numbers[valid_mask]
    
    if len(filtered_times) < 10:
        return None, None, None, None, None
    
    # Perform linear regression to find the slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_lap_numbers, filtered_times)
    
    # Calculate degradation as percentage
    pace_degradation_percentage = (slope / intercept) * 100 * 10  # Per 10 laps
    
    return slope, pace_degradation_percentage, r_value, filtered_lap_numbers, filtered_times

@st.cache_data
def compare_stint_performance(_laps_df, driver_code):
    """
    Compare a driver's performance across different stints in a race
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    driver_code : str
        Driver code (abbreviation)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with stint performance metrics
    """
    # Filter laps for the driver
    driver_laps = _laps_df[_laps_df['Driver'] == driver_code].copy()
    
    if driver_laps.empty:
        return pd.DataFrame()
    
    # Sort by lap number
    driver_laps = driver_laps.sort_values('LapNumber')
    
    # Identify pit stops and stints
    stint_number = 1
    driver_laps['Stint'] = 0
    
    for i, lap in driver_laps.iterrows():
        driver_laps.at[i, 'Stint'] = stint_number
        if pd.notnull(lap['PitInTime']):
            stint_number += 1
    
    # Calculate performance metrics for each stint
    stint_metrics = []
    
    for stint, stint_laps in driver_laps.groupby('Stint'):
        # Filter valid lap times (no pit in/out)
        valid_laps = stint_laps[(stint_laps['PitOutTime'].isnull()) & 
                               (stint_laps['PitInTime'].isnull())]
        
        if len(valid_laps) < 3:
            continue
            
        lap_times_sec = valid_laps['LapTime'].dt.total_seconds()
        
        # Calculate metrics
        metrics = {
            'Stint': stint,
            'Compound': valid_laps.iloc[0]['Compound'] if 'Compound' in valid_laps.columns else 'Unknown',
            'NumLaps': len(valid_laps),
            'AvgLapTime': lap_times_sec.mean(),
            'BestLapTime': lap_times_sec.min(),
            'LapTimeStd': lap_times_sec.std(),
            'FirstLap': valid_laps['LapNumber'].min(),
            'LastLap': valid_laps['LapNumber'].max()
        }
        
        # Calculate degradation within stint
        if len(valid_laps) >= 5:
            lap_numbers = valid_laps['LapNumber'].values
            slope, _, _, _, _ = stats.linregress(lap_numbers, lap_times_sec)
            metrics['DegradationPerLap'] = slope
        else:
            metrics['DegradationPerLap'] = np.nan
            
        stint_metrics.append(metrics)
    
    return pd.DataFrame(stint_metrics)

@st.cache_data
def calculate_multivariate_driver_rating(_laps_df, results_df=None):
    """
    Calculate a comprehensive driver rating based on multiple performance factors
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    results_df : pandas.DataFrame, optional
        DataFrame containing race results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with driver ratings
    """
    if _laps_df.empty:
        return pd.DataFrame()
    
    driver_metrics = []
    
    # Get unique drivers
    drivers = _laps_df['Driver'].unique()
    
    for driver in drivers:
        # Get driver laps
        driver_laps = _laps_df[_laps_df['Driver'] == driver]
        
        if len(driver_laps) < 5:
            continue
            
        # Basic metrics
        metrics = {'Driver': driver}
        
        if 'Team' in driver_laps.columns:
            metrics['Team'] = driver_laps['Team'].iloc[0]
        
        # Calculate consistency
        consistency, avg_pace = calculate_driver_consistency(_laps_df, driver)
        metrics['Consistency'] = consistency
        metrics['AvgPace'] = avg_pace
        
        # Best lap time
        valid_laps = driver_laps[(driver_laps['PitOutTime'].isnull()) & 
                               (driver_laps['PitInTime'].isnull())]
        
        if not valid_laps.empty and not valid_laps['LapTime'].isnull().all():
            metrics['BestLapTime'] = valid_laps['LapTime'].min()
        
        # Calculate pace degradation
        slope, degr_pct, r_value, _, _ = analyze_race_pace_degradation(_laps_df, driver)
        metrics['PaceDegradation'] = degr_pct
        metrics['DegradationFit'] = r_value
        
        # Add race position if results available
        if results_df is not None and not results_df.empty:
            driver_result = results_df[results_df['Abbreviation'] == driver]
            if not driver_result.empty:
                metrics['Position'] = driver_result['Position'].iloc[0]
        
        driver_metrics.append(metrics)
    
    # Create DataFrame
    df_metrics = pd.DataFrame(driver_metrics)
    
    if df_metrics.empty:
        return df_metrics
    
    # Normalize metrics for rating calculation
    rating_columns = ['Consistency', 'AvgPace', 'PaceDegradation']
    df_normalized = df_metrics.copy()
    
    for col in rating_columns:
        if col in df_normalized.columns and not df_normalized[col].isnull().all():
            # Higher consistency (lower value) is better
            if col == 'Consistency' or col == 'AvgPace' or col == 'PaceDegradation':
                max_val = df_normalized[col].max()
                min_val = df_normalized[col].min()
                if max_val != min_val:
                    df_normalized[f'{col}Norm'] = 1 - ((df_normalized[col] - min_val) / (max_val - min_val))
                else:
                    df_normalized[f'{col}Norm'] = 1.0
            else:
                max_val = df_normalized[col].max()
                min_val = df_normalized[col].min()
                if max_val != min_val:
                    df_normalized[f'{col}Norm'] = (df_normalized[col] - min_val) / (max_val - min_val)
                else:
                    df_normalized[f'{col}Norm'] = 1.0
    
    # Calculate overall rating (customize weights as needed)
    if 'ConsistencyNorm' in df_normalized.columns and 'AvgPaceNorm' in df_normalized.columns:
        weights = {
            'ConsistencyNorm': 0.4,
            'AvgPaceNorm': 0.5,
            'PaceDegradationNorm': 0.1 if 'PaceDegradationNorm' in df_normalized.columns else 0
        }
        
        df_normalized['OverallRating'] = 0
        for col, weight in weights.items():
            if col in df_normalized.columns:
                df_normalized['OverallRating'] += df_normalized[col] * weight
        
        # Scale to 0-10 rating
        df_normalized['DriverRating'] = df_normalized['OverallRating'] * 10
    
    return df_normalized

@st.cache_data
def calculate_team_strategy_metrics(_laps_df):
    """
    Calculate metrics related to team strategy
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with team strategy metrics
    """
    if _laps_df.empty or 'Team' not in _laps_df.columns:
        return pd.DataFrame()
    
    team_metrics = []
    
    # Get unique teams
    teams = _laps_df['Team'].unique()
    
    for team in teams:
        # Get team laps
        team_laps = _laps_df[_laps_df['Team'] == team]
        
        # Basic metrics
        metrics = {'Team': team}
        
        # Calculate pit stop metrics
        pit_laps = team_laps[team_laps['PitInTime'].notnull()]
        metrics['TotalPitStops'] = len(pit_laps)
        
        if not pit_laps.empty and 'PitOutTime' in pit_laps.columns and 'PitInTime' in pit_laps.columns:
            # Calculate pit stop durations
            pit_times = []
            for _, lap in pit_laps.iterrows():
                if pd.notnull(lap['PitOutTime']) and pd.notnull(lap['PitInTime']):
                    pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
                    if pit_duration > 0:
                        pit_times.append(pit_duration)
            
            if pit_times:
                metrics['AvgPitTime'] = np.mean(pit_times)
                metrics['MinPitTime'] = np.min(pit_times)
                metrics['MaxPitTime'] = np.max(pit_times)
                metrics['PitTimeStd'] = np.std(pit_times)
        
        # Analyze tire strategy if compound information is available
        if 'Compound' in team_laps.columns:
            # Count laps by compound
            compound_counts = team_laps.groupby('Compound').size()
            for compound, count in compound_counts.items():
                if pd.notna(compound):
                    metrics[f'Laps{compound}'] = count
            
            # Identify the most used compound
            if not compound_counts.empty:
                metrics['MostUsedCompound'] = compound_counts.idxmax()
        
        team_metrics.append(metrics)
    
    return pd.DataFrame(team_metrics)

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

@st.cache_resource
def create_driver_clusters(_laps_df):
    """
    Apply clustering to identify groups of drivers with similar performance characteristics
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
        
    Returns:
    --------
    tuple
        (cluster_df, scaler, kmeans_model, pca_model)
    """
    if _laps_df.empty:
        return None, None, None, None
    
    # Aggregate data by driver
    driver_stats = []
    
    for driver, driver_laps in _laps_df.groupby('Driver'):
        # Skip if too few laps
        if len(driver_laps) < 5:
            continue
            
        # Calculate various performance metrics
        valid_laps = driver_laps[(driver_laps['PitOutTime'].isnull()) & 
                               (driver_laps['PitInTime'].isnull())]
        
        if len(valid_laps) < 5:
            continue
            
        lap_times_sec = valid_laps['LapTime'].dt.total_seconds()
        
        stats = {
            'Driver': driver,
            'MeanLapTime': lap_times_sec.mean(),
            'MinLapTime': lap_times_sec.min(),
            'StdLapTime': lap_times_sec.std(),
            'LapCount': len(valid_laps)
        }
        
        # Add team if available
        if 'Team' in driver_laps.columns:
            stats['Team'] = driver_laps['Team'].iloc[0]
        
        # Add car data if available
        car_data_cols = ['Speed', 'Throttle', 'Brake', 'RPM']
        for col in car_data_cols:
            if col in valid_laps.columns and not valid_laps[col].isnull().all():
                stats[f'Mean{col}'] = valid_laps[col].mean()
        
        driver_stats.append(stats)
    
    # Create DataFrame
    df_stats = pd.DataFrame(driver_stats)
    
    if len(df_stats) < 3:
        return None, None, None, None
    
    # Select features for clustering
    feature_cols = ['MeanLapTime', 'MinLapTime', 'StdLapTime']
    
    # Add additional features if available
    for col in df_stats.columns:
        if col.startswith('Mean') and col != 'MeanLapTime':
            feature_cols.append(col)
    
    # Ensure all selected features exist and have data
    feature_cols = [col for col in feature_cols if col in df_stats.columns and not df_stats[col].isnull().all()]
    
    if len(feature_cols) < 2:
        return None, None, None, None
    
    # Standardize features
    X = df_stats[feature_cols].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction and visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Determine optimal number of clusters (between 2 and 5)
    inertias = []
    for k in range(2, 6):
        if len(df_stats) >= k:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
    
    # Choose number of clusters
    if len(inertias) > 1:
        k_diff = np.diff(inertias)
        optimal_k = np.argmin(np.abs(k_diff / inertias[:-1] - 0.2)) + 2
    else:
        optimal_k = 2
    
    # Ensure we don't have more clusters than data points
    optimal_k = min(optimal_k, len(df_stats) - 1)
    optimal_k = max(optimal_k, 2)  # At least 2 clusters
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster info and PCA components to DataFrame
    df_stats['Cluster'] = clusters
    df_stats['PCA1'] = X_pca[:, 0]
    df_stats['PCA2'] = X_pca[:, 1]
    
    return df_stats, scaler, kmeans, pca

@st.cache_resource
def predict_lap_time(_laps_df, driver_code=None, team_name=None):
    """
    Build a model to predict lap times based on available features
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    driver_code : str, optional
        Driver code to filter for (if None, use all drivers)
    team_name : str, optional
        Team name to filter for (if None, use all teams)
        
    Returns:
    --------
    tuple
        (model, feature_importance_df, X_columns, r2_score, rmse)
    """
    if _laps_df.empty:
        return None, None, None, None, None
    
    # Filter data if needed
    if driver_code:
        laps_df = laps_df[laps_df['Driver'] == driver_code]
    elif team_name:
        laps_df = laps_df[laps_df['Team'] == team_name]
    
    # Valid laps only
    valid_laps = laps_df[(laps_df['PitOutTime'].isnull()) & 
                        (laps_df['PitInTime'].isnull())]
    
    if len(valid_laps) < 30:  # Need sufficient data for modeling
        return None, None, None, None, None
    
    # Convert lap times to seconds as target variable
    y = valid_laps['LapTime'].dt.total_seconds()
    
    # Prepare features
    feature_cols = []
    
    # Add lap number
    if 'LapNumber' in valid_laps.columns:
        feature_cols.append('LapNumber')
    
    # Add stint information if available
    if 'Stint' in valid_laps.columns:
        valid_laps['StintLap'] = valid_laps.groupby('Stint').cumcount() + 1
        feature_cols.append('StintLap')
    
    # Add tyre compound if available
    if 'Compound' in valid_laps.columns and valid_laps['Compound'].nunique() > 1:
        # One-hot encode compound
        compound_dummies = pd.get_dummies(valid_laps['Compound'], prefix='Compound')
        valid_laps = pd.concat([valid_laps, compound_dummies], axis=1)
        feature_cols.extend(compound_dummies.columns)
    
    # Add tyre age if available
    if 'TyreLife' in valid_laps.columns:
        feature_cols.append('TyreLife')
    
    # Add fuel load proxy (higher at start, lower at end)
    if 'LapNumber' in valid_laps.columns:
        max_lap = valid_laps['LapNumber'].max()
        valid_laps['FuelLoad'] = 1 - (valid_laps['LapNumber'] / max_lap)
        feature_cols.append('FuelLoad')
    
    # Add track position if available
    if 'Position' in valid_laps.columns:
        feature_cols.append('Position')
    
    # Check if we have sufficient features
    if len(feature_cols) < 2:
        return None, None, None, None, None
    
    # Create feature matrix
    X = valid_laps[feature_cols].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train different models and select the best one
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_score = -np.inf
    best_model_name = None
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        mean_score = cv_scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
    
    # Train the best model on the full training set
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Extract feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
    else:
        if hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_)
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': [1/len(X.columns)] * len(X.columns)
            })
    
    return best_model, feature_importance, X.columns.tolist(), test_r2, test_rmse

@st.cache_resource
def detect_anomalies(_laps_df, driver_code=None):
    """
    Detect anomalous laps using DBSCAN clustering
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    driver_code : str, optional
        Driver code to filter for (if None, analyze all drivers separately)
        
    Returns:
    --------
    dict
        Dictionary with driver codes as keys and anomalous laps as values
    """
    if _laps_df.empty:
        return {}
    
    # If driver code provided, filter data
    if driver_code:
        drivers_to_analyze = [driver_code]
    else:
        # Analyze top drivers by lap count
        driver_lap_counts = _laps_df.groupby('Driver').size()
        drivers_to_analyze = driver_lap_counts[driver_lap_counts >= 20].index.tolist()
    
    # No drivers with sufficient data
    if not drivers_to_analyze:
        return {}
    
    anomaly_results = {}
    
    for driver in drivers_to_analyze:
        # Get driver laps
        driver_laps = _laps_df[laps_df['Driver'] == driver].copy()
        
        # Skip if too few laps
        if len(driver_laps) < 20:
            continue
            
        # Valid laps only
        valid_laps = driver_laps[(driver_laps['PitOutTime'].isnull()) & 
                               (driver_laps['PitInTime'].isnull())]
        
        if len(valid_laps) < 20:
            continue
            
        # Features for anomaly detection
        feature_cols = []
        
        # Primary feature: lap time
        valid_laps['LapTimeSeconds'] = valid_laps['LapTime'].dt.total_seconds()
        feature_cols.append('LapTimeSeconds')
        
        # Add sectors if available
        for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
            if sector in valid_laps.columns and not valid_laps[sector].isnull().all():
                valid_laps[f'{sector}Seconds'] = valid_laps[sector].dt.total_seconds()
                feature_cols.append(f'{sector}Seconds')
        
        # Add speed features if available
        speed_cols = [col for col in valid_laps.columns if 'Speed' in col and col != 'SpeedI1' and col != 'SpeedI2']
        feature_cols.extend([col for col in speed_cols if not valid_laps[col].isnull().all()])
        
        # Check if we have sufficient features
        if len(feature_cols) < 2:
            continue
            
        # Prepare features
        X = valid_laps[feature_cols].copy()
        
        # Handle missing values
        X.fillna(X.mean(), inplace=True)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply DBSCAN for anomaly detection
        dbscan = DBSCAN(eps=1.5, min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Anomalies are labeled as -1
        anomaly_mask = clusters == -1
        anomalous_laps = valid_laps[anomaly_mask].copy()
        
        if not anomalous_laps.empty:
            # Calculate degree of anomaly (distance from mean)
            mean_lap_time = valid_laps['LapTimeSeconds'].mean()
            std_lap_time = valid_laps['LapTimeSeconds'].std()
            
            anomalous_laps['AnomalyScore'] = np.abs(anomalous_laps['LapTimeSeconds'] - mean_lap_time) / std_lap_time
            
            # Store result
            anomaly_results[driver] = anomalous_laps.sort_values('AnomalyScore', ascending=False)
    
    return anomaly_results

@st.cache_resource
def create_race_strategy_model(_laps_df, session_results=None):
    """
    Build a model to analyze race strategy effectiveness
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    session_results : pandas.DataFrame, optional
        DataFrame with session results
        
    Returns:
    --------
    tuple
        (strategy_df, importance_df)
    """
    if _laps_df.empty:
        return None, None
    
    # Check if we have compound information
    if 'Compound' not in laps_df.columns:
        return None, None
    
    # Get unique drivers
    drivers = _laps_df['Driver'].unique()
    
    strategy_data = []
    
    for driver in drivers:
        # Get driver laps
        driver_laps = _laps_df[laps_df['Driver'] == driver].copy()
        
        # Skip if too few laps
        if len(driver_laps) < 10:
            continue
            
        # Get driver result if available
        final_position = None
        if session_results is not None and not session_results.empty:
            driver_result = session_results[session_results['Abbreviation'] == driver]
            if not driver_result.empty:
                final_position = driver_result['Position'].iloc[0]
        
        # Identify stints based on pit stops
        stint_number = 1
        driver_laps['Stint'] = 0
        
        for i, lap in driver_laps.iterrows():
            driver_laps.at[i, 'Stint'] = stint_number
            if pd.notnull(lap['PitInTime']):
                stint_number += 1
        
        # Calculate stats for each stint
        for stint, stint_laps in driver_laps.groupby('Stint'):
            if len(stint_laps) < 3:
                continue
                
            # Get stint details
            compound = stint_laps['Compound'].iloc[0] if pd.notna(stint_laps['Compound'].iloc[0]) else 'Unknown'
            stint_length = len(stint_laps)
            start_lap = stint_laps['LapNumber'].min()
            end_lap = stint_laps['LapNumber'].max()
            
            # Calculate average pace
            valid_laps = stint_laps[(stint_laps['PitOutTime'].isnull()) & 
                                  (stint_laps['PitInTime'].isnull())]
            
            if valid_laps.empty:
                continue
                
            lap_times_sec = valid_laps['LapTime'].dt.total_seconds()
            avg_pace = lap_times_sec.mean()
            
            # Calculate pace degradation
            if len(valid_laps) >= 5:
                lap_numbers = valid_laps['LapNumber'].values
                rel_lap_numbers = lap_numbers - lap_numbers.min()  # Relative lap numbers within stint
                
                try:
                    slope, _, _, _, _ = stats.linregress(rel_lap_numbers, lap_times_sec)
                    pace_degradation = slope
                except:
                    pace_degradation = 0
            else:
                pace_degradation = 0
            
            # Collect strategy data
            strategy = {
                'Driver': driver,
                'Team': driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else 'Unknown',
                'Stint': stint,
                'Compound': compound,
                'StintLength': stint_length,
                'StartLap': start_lap,
                'EndLap': end_lap,
                'AvgPace': avg_pace,
                'PaceDegradation': pace_degradation,
                'FinalPosition': final_position
            }
            
            strategy_data.append(strategy)
    
    # Create DataFrame
    strategy_df = pd.DataFrame(strategy_data)
    
    if strategy_df.empty or 'FinalPosition' not in strategy_df.columns or strategy_df['FinalPosition'].isnull().all():
        return strategy_df, None
    
    # Analyze impact of strategy on race outcome
    # Prepare features for modeling
    strategy_features = []
    
    for driver, driver_stints in strategy_df.groupby('Driver'):
        # Skip if no final position
        if driver_stints['FinalPosition'].isnull().all():
            continue
            
        # Calculate strategy metrics
        num_stints = len(driver_stints)
        avg_stint_length = driver_stints['StintLength'].mean()
        
        # Compound usage
        compound_counts = driver_stints['Compound'].value_counts()
        compounds_used = len(compound_counts)
        
        # Most used compound
        most_used = compound_counts.index[0] if not compound_counts.empty else 'Unknown'
        
        # Average pace across all stints
        avg_race_pace = driver_stints['AvgPace'].mean()
        
        # Average degradation
        avg_degradation = driver_stints['PaceDegradation'].mean()
        
        # Collect features
        features = {
            'Driver': driver,
            'Team': driver_stints['Team'].iloc[0],
            'NumStints': num_stints,
            'AvgStintLength': avg_stint_length,
            'CompoundsUsed': compounds_used,
            'MostUsedCompound': most_used,
            'AvgRacePace': avg_race_pace,
            'AvgDegradation': avg_degradation,
            'FinalPosition': driver_stints['FinalPosition'].iloc[0]
        }
        
        # Add specific compound usage
        for compound in strategy_df['Compound'].unique():
            if pd.notna(compound):
                compound_laps = driver_stints[driver_stints['Compound'] == compound]['StintLength'].sum()
                features[f'{compound}Laps'] = compound_laps
        
        strategy_features.append(features)
    
    # Create features DataFrame
    features_df = pd.DataFrame(strategy_features)
    
    if len(features_df) < 10:
        return strategy_df, None
    
    # Prepare features for modeling
    feature_cols = [col for col in features_df.columns if col not in ['Driver', 'Team', 'FinalPosition', 'MostUsedCompound']]
    
    # One-hot encode categorical features
    if 'MostUsedCompound' in features_df.columns:
        compound_dummies = pd.get_dummies(features_df['MostUsedCompound'], prefix='MostUsed')
        features_df = pd.concat([features_df, compound_dummies], axis=1)
        feature_cols.extend(compound_dummies.columns)
    
    # Create feature matrix
    X = features_df[feature_cols].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Target variable: race position
    y = features_df['FinalPosition']
    
    # Train a model to predict race outcome
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Extract feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return strategy_df, importance_df

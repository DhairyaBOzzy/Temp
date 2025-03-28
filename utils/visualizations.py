import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from fastf1 import plotting
from fastf1.plotting import team_color

# Load the FastF1 color scheme for consistent visualizations
plotting.setup_mpl(mpl_timedelta_support=True, color_scheme=None, misc_mpl_mods=True)

@st.cache_data
def plot_lap_time_comparison(_laps_df, drivers=None, teams=None):
    """
    Create a box plot comparing lap time distributions for drivers or teams
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    drivers : list, optional
        List of driver codes to include (if None, top 10 by lap count)
    teams : list, optional
        List of team names to include (if None and drivers is None, all teams)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if _laps_df.empty:
        return go.Figure()
    
    # Filter valid laps
    valid_laps = _laps_df[(_laps_df['PitOutTime'].isnull()) & 
                        (_laps_df['PitInTime'].isnull())]
    
    if valid_laps.empty:
        return go.Figure()
    
    # Convert lap times to seconds
    valid_laps['LapTimeSeconds'] = valid_laps['LapTime'].dt.total_seconds()
    
    # Filter by drivers or teams
    if drivers:
        plot_data = valid_laps[valid_laps['Driver'].isin(drivers)]
        group_col = 'Driver'
    elif teams:
        plot_data = valid_laps[valid_laps['Team'].isin(teams)]
        group_col = 'Team'
    else:
        # Default: use drivers with most laps
        driver_lap_counts = valid_laps.groupby('Driver').size().sort_values(ascending=False)
        top_drivers = driver_lap_counts.head(10).index.tolist()
        plot_data = valid_laps[valid_laps['Driver'].isin(top_drivers)]
        group_col = 'Driver'
    
    if plot_data.empty:
        return go.Figure()
    
    # Create box plot
    fig = px.box(plot_data, 
                x=group_col, 
                y='LapTimeSeconds',
                title=f'Lap Time Distribution by {group_col}',
                labels={'LapTimeSeconds': 'Lap Time (seconds)', group_col: group_col})
    
    # Add median lap time as text
    medians = plot_data.groupby(group_col)['LapTimeSeconds'].median()
    
    annotations = []
    for i, (group, median) in enumerate(medians.items()):
        annotations.append(dict(
            x=group,
            y=median,
            text=f"{median:.2f}s",
            showarrow=False,
            font=dict(size=10),
            xanchor='center',
            yanchor='bottom',
            yshift=10
        ))
    
    fig.update_layout(annotations=annotations)
    
    # Update layout
    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title='Lap Time (seconds)',
        height=500
    )
    
    return fig

@st.cache_data
def plot_driver_pace_comparison(_laps_df, drivers=None, session_info=None):
    """
    Create a line plot showing lap time progression for selected drivers
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    drivers : list, optional
        List of driver codes to include (if None, top 5 by final position)
    session_info : dict, optional
        Session information for plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if _laps_df.empty:
        return go.Figure()
    
    # If no drivers specified, use top 5 from results if available
    if drivers is None:
        if 'Position' in _laps_df.columns:
            # Get top 5 drivers by best finishing position
            best_positions = _laps_df.groupby('Driver')['Position'].min()
            drivers = best_positions.sort_values().head(5).index.tolist()
        else:
            # Or just take 5 drivers with most laps
            lap_counts = _laps_df.groupby('Driver').size().sort_values(ascending=False)
            drivers = lap_counts.head(5).index.tolist()
    
    # Filter for selected drivers
    driver_laps = _laps_df[_laps_df['Driver'].isin(drivers)].copy()
    
    if driver_laps.empty:
        return go.Figure()
    
    # Convert lap times to seconds
    driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()
    
    # Filter out invalid laps
    valid_laps = driver_laps[(driver_laps['PitOutTime'].isnull()) & 
                            (driver_laps['PitInTime'].isnull())]
    
    if valid_laps.empty:
        return go.Figure()
    
    # Sort by lap number
    valid_laps = valid_laps.sort_values(['Driver', 'LapNumber'])
    
    # Create figure
    fig = go.Figure()
    
    # Try to use team colors if available
    if 'Team' in valid_laps.columns:
        driver_teams = valid_laps.groupby('Driver')['Team'].first()
        
        for driver in drivers:
            if driver in driver_teams:
                try:
                    team_name = driver_teams[driver]
                    driver_color = team_color(team_name)
                except:
                    driver_color = None
                
                driver_data = valid_laps[valid_laps['Driver'] == driver]
                
                fig.add_trace(go.Scatter(
                    x=driver_data['LapNumber'],
                    y=driver_data['LapTimeSeconds'],
                    mode='lines+markers',
                    name=driver,
                    line=dict(color=driver_color) if driver_color else None
                ))
    else:
        # Default plotting without team colors
        for driver in drivers:
            driver_data = valid_laps[valid_laps['Driver'] == driver]
            
            fig.add_trace(go.Scatter(
                x=driver_data['LapNumber'],
                y=driver_data['LapTimeSeconds'],
                mode='lines+markers',
                name=driver
            ))
    
    # Determine title based on session_info
    title = "Driver Pace Comparison"
    if session_info:
        if 'EventName' in session_info and 'Year' in session_info:
            title = f"Driver Pace Comparison - {session_info['EventName']} {session_info['Year']}"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (seconds)",
        legend_title="Driver",
        height=500
    )
    
    return fig

@st.cache_data
def plot_team_performance_comparison(_laps_df, teams=None):
    """
    Create a bar chart comparing team performance metrics
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data
    teams : list, optional
        List of team names to include (if None, all teams)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if _laps_df.empty or 'Team' not in _laps_df.columns:
        return go.Figure()
    
    # Filter for selected teams
    if teams:
        team_laps = _laps_df[_laps_df['Team'].isin(teams)]
    else:
        team_laps = _laps_df
    
    if team_laps.empty:
        return go.Figure()
    
    # Convert lap times to seconds
    team_laps['LapTimeSeconds'] = team_laps['LapTime'].dt.total_seconds()
    
    # Calculate team metrics
    team_metrics = []
    
    for team, data in team_laps.groupby('Team'):
        # Valid laps only
        valid_laps = data[(data['PitOutTime'].isnull()) & 
                         (data['PitInTime'].isnull())]
        
        if valid_laps.empty:
            continue
            
        lap_times = valid_laps['LapTimeSeconds']
        
        metrics = {
            'Team': team,
            'MedianLapTime': lap_times.median(),
            'FastestLapTime': lap_times.min(),
            'LapCount': len(valid_laps)
        }
        
        # Calculate standard deviation if enough laps
        if len(lap_times) >= 5:
            metrics['Consistency'] = lap_times.std()
        
        team_metrics.append(metrics)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(team_metrics)
    
    if metrics_df.empty:
        return go.Figure()
    
    # Create subplots for different metrics
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=("Median Lap Time", "Fastest Lap", "Consistency (Std Dev)"),
                       shared_yaxes=True)
    
    # Sort by median lap time
    metrics_df = metrics_df.sort_values('MedianLapTime')
    
    # Try to use team colors
    for team in metrics_df['Team']:
        try:
            team_col = team_color(team)
        except:
            team_col = None
        
        team_data = metrics_df[metrics_df['Team'] == team]
        
        # Median lap time
        fig.add_trace(
            go.Bar(
                x=[team],
                y=team_data['MedianLapTime'],
                name=team,
                marker_color=team_col
            ),
            row=1, col=1
        )
        
        # Fastest lap
        fig.add_trace(
            go.Bar(
                x=[team],
                y=team_data['FastestLapTime'],
                name=team,
                marker_color=team_col,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Consistency (if available)
        if 'Consistency' in team_data.columns and not team_data['Consistency'].isnull().all():
            fig.add_trace(
                go.Bar(
                    x=[team],
                    y=team_data['Consistency'],
                    name=team,
                    marker_color=team_col,
                    showlegend=False
                ),
                row=1, col=3
            )
    
    # Update layout
    fig.update_layout(
        title="Team Performance Comparison",
        height=500,
        yaxis_title="Seconds",
        yaxis2_title="Seconds",
        yaxis3_title="Seconds (lower is better)"
    )
    
    return fig

@st.cache_data
def plot_driver_performance_radar(driver_metrics_df, drivers=None):
    """
    Create a radar chart comparing driver performance across multiple metrics
    
    Parameters:
    -----------
    driver_metrics_df : pandas.DataFrame
        DataFrame containing driver performance metrics
    drivers : list, optional
        List of driver codes to include (if None, top 5 by overall rating)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if driver_metrics_df.empty:
        return go.Figure()
    
    # Select metrics to include in radar
    radar_metrics = [col for col in driver_metrics_df.columns if col.endswith('Norm')]
    
    if not radar_metrics:
        return go.Figure()
    
    # If no drivers specified, use top 5 by overall rating if available
    if drivers is None:
        if 'DriverRating' in driver_metrics_df.columns:
            drivers = driver_metrics_df.sort_values('DriverRating', ascending=False).head(5)['Driver'].tolist()
        else:
            drivers = driver_metrics_df.head(5)['Driver'].tolist()
    else:
        # Filter for specified drivers
        driver_metrics_df = driver_metrics_df[driver_metrics_df['Driver'].isin(drivers)]
    
    if driver_metrics_df.empty:
        return go.Figure()
    
    # Create radar chart
    fig = go.Figure()
    
    # Get human-readable metric names for display
    metric_labels = [m.replace('Norm', '') for m in radar_metrics]
    
    # Try to use team colors if available
    if 'Team' in driver_metrics_df.columns:
        for _, row in driver_metrics_df.iterrows():
            driver = row['Driver']
            team = row['Team']
            
            try:
                driver_color = team_color(team)
            except:
                driver_color = None
            
            fig.add_trace(go.Scatterpolar(
                r=[row[m] * 10 for m in radar_metrics],  # Scale to 0-10
                theta=metric_labels,
                fill='toself',
                name=driver,
                line=dict(color=driver_color) if driver_color else None
            ))
    else:
        # Default plotting without team colors
        for _, row in driver_metrics_df.iterrows():
            driver = row['Driver']
            
            fig.add_trace(go.Scatterpolar(
                r=[row[m] * 10 for m in radar_metrics],  # Scale to 0-10
                theta=metric_labels,
                fill='toself',
                name=driver
            ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title="Driver Performance Comparison",
        height=600
    )
    
    return fig

@st.cache_data
def plot_stint_strategy_comparison(stint_df, drivers=None):
    """
    Create a horizontal bar chart showing stint strategies for selected drivers
    
    Parameters:
    -----------
    stint_df : pandas.DataFrame
        DataFrame containing stint information
    drivers : list, optional
        List of driver codes to include (if None, top 10 by final position)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if stint_df.empty:
        return go.Figure()
    
    # If no drivers specified, use top 10 from results if available
    if drivers is None:
        if 'FinalPosition' in stint_df.columns:
            # Get top 10 drivers by finishing position
            best_positions = stint_df.groupby('Driver')['FinalPosition'].first()
            drivers = best_positions.sort_values().head(10).index.tolist()
        else:
            # Or just take top 10 drivers alphabetically
            drivers = sorted(stint_df['Driver'].unique())[:10]
    
    # Filter for selected drivers
    driver_stints = stint_df[stint_df['Driver'].isin(drivers)].copy()
    
    if driver_stints.empty:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Sort drivers by position if available, otherwise alphabetically
    if 'FinalPosition' in driver_stints.columns:
        driver_order = driver_stints.groupby('Driver')['FinalPosition'].first().sort_values().index.tolist()
    else:
        driver_order = sorted(driver_stints['Driver'].unique())
    
    # Compound colors
    compound_colors = {
        'SOFT': 'red',
        'MEDIUM': 'yellow',
        'HARD': 'white',
        'INTERMEDIATE': 'green',
        'WET': 'blue',
        'Unknown': 'grey'
    }
    
    # Add stints for each driver
    for driver in driver_order:
        driver_data = driver_stints[driver_stints['Driver'] == driver].sort_values('StartLap')
        
        for _, stint in driver_data.iterrows():
            compound = stint['Compound'] if pd.notna(stint['Compound']) else 'Unknown'
            start_lap = stint['StartLap']
            end_lap = stint['EndLap']
            
            # Try to get team color
            if 'Team' in stint and pd.notna(stint['Team']):
                try:
                    border_color = team_color(stint['Team'])
                except:
                    border_color = 'black'
            else:
                border_color = 'black'
            
            fig.add_trace(go.Bar(
                y=[driver],
                x=[end_lap - start_lap + 1],  # Stint length
                name=f"{driver} - {compound}",
                marker_color=compound_colors.get(compound, 'grey'),
                marker_line_color=border_color,
                marker_line_width=1.5,
                orientation='h',
                showlegend=False,
                base=start_lap - 1,  # Start position
                customdata=[[compound, start_lap, end_lap]],
                hovertemplate='<b>%{y}</b><br>' +
                              'Compound: %{customdata[0]}<br>' +
                              'Laps: %{customdata[1]}-%{customdata[2]}<br>' +
                              'Stint Length: %{x} laps<extra></extra>'
            ))
    
    # Add a legend for tire compounds
    for compound, color in compound_colors.items():
        fig.add_trace(go.Bar(
            y=[driver_order[-1] if driver_order else ''],
            x=[0],  # Zero width, won't show in chart
            name=compound,
            marker_color=color,
            orientation='h',
            visible='legendonly',  # Only show in legend
        ))
    
    # Update layout
    fig.update_layout(
        title="Race Strategy Comparison",
        xaxis_title="Lap Number",
        yaxis_title="Driver",
        barmode='overlay',
        bargap=0.2,
        height=600,
        xaxis=dict(
            title="Lap Number",
            zeroline=False
        ),
        yaxis=dict(
            title="Driver",
            categoryorder='array',
            categoryarray=driver_order,
            autorange="reversed"  # Reverse to match leaderboard order
        )
    )
    
    return fig

@st.cache_data
def plot_race_position_changes(_laps_df, drivers=None):
    """
    Create a line plot showing position changes throughout the race
    
    Parameters:
    -----------
    laps_df : pandas.DataFrame
        DataFrame containing lap data with position information
    drivers : list, optional
        List of driver codes to include (if None, all drivers)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if _laps_df.empty or 'Position' not in _laps_df.columns:
        return go.Figure()
    
    # Filter for selected drivers
    if drivers:
        position_data = _laps_df[_laps_df['Driver'].isin(drivers)]
    else:
        position_data = _laps_df
    
    if position_data.empty:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Try to use team colors if available
    if 'Team' in position_data.columns:
        driver_teams = position_data.groupby('Driver')['Team'].first()
        
        for driver, team in driver_teams.items():
            driver_data = position_data[position_data['Driver'] == driver].sort_values('LapNumber')
            
            try:
                driver_color = team_color(team)
            except:
                driver_color = None
            
            fig.add_trace(go.Scatter(
                x=driver_data['LapNumber'],
                y=driver_data['Position'],
                mode='lines+markers',
                name=driver,
                line=dict(color=driver_color) if driver_color else None
            ))
    else:
        # Default plotting without team colors
        for driver in position_data['Driver'].unique():
            driver_data = position_data[position_data['Driver'] == driver].sort_values('LapNumber')
            
            fig.add_trace(go.Scatter(
                x=driver_data['LapNumber'],
                y=driver_data['Position'],
                mode='lines+markers',
                name=driver
            ))
    
    # Update layout
    fig.update_layout(
        title="Race Position Changes",
        xaxis_title="Lap Number",
        yaxis_title="Position",
        yaxis=dict(
            autorange="reversed",  # Reverse to show P1 at the top
            dtick=1
        ),
        height=600
    )
    
    return fig

@st.cache_data
def plot_driver_clusters(cluster_df):
    """
    Create a scatter plot showing driver clusters from machine learning analysis
    
    Parameters:
    -----------
    cluster_df : pandas.DataFrame
        DataFrame containing driver cluster information
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if cluster_df is None or cluster_df.empty or 'Cluster' not in cluster_df.columns:
        return go.Figure()
    
    if 'PCA1' not in cluster_df.columns or 'PCA2' not in cluster_df.columns:
        return go.Figure()
    
    # Create figure
    fig = px.scatter(
        cluster_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        text='Driver',
        hover_data=['MeanLapTime', 'MinLapTime', 'StdLapTime', 'Team'] if 'Team' in cluster_df.columns else ['MeanLapTime', 'MinLapTime', 'StdLapTime'],
        color_continuous_scale=px.colors.qualitative.G10
    )
    
    # Update layout
    fig.update_layout(
        title="Driver Performance Clusters",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        height=600
    )
    
    return fig

@st.cache_data
def plot_feature_importance(importance_df, title="Feature Importance"):
    """
    Create a horizontal bar chart showing feature importance from ML models
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame containing feature importance values
    title : str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if importance_df is None or importance_df.empty or 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
        return go.Figure()
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    # Create bar chart
    fig = px.bar(
        importance_df.iloc[-15:] if len(importance_df) > 15 else importance_df,  # Show top 15 if more
        y='Feature',
        x='Importance',
        orientation='h',
        title=title
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=500
    )
    
    return fig

@st.cache_data
def plot_lap_time_prediction(actual, predicted, driver_name=None):
    """
    Create a scatter plot comparing actual vs predicted lap times
    
    Parameters:
    -----------
    actual : array-like
        Actual lap times
    predicted : array-like
        Predicted lap times
    driver_name : str, optional
        Driver name for title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if actual is None or predicted is None or len(actual) != len(predicted):
        return go.Figure()
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted,
        'Error': np.abs(actual - predicted)
    })
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mean_error = np.mean(np.abs(actual - predicted))
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='Actual',
        y='Predicted',
        color='Error',
        color_continuous_scale='Viridis'
    )
    
    # Add diagonal line (perfect prediction)
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    # Add statistics
    title = "Lap Time Prediction Results"
    if driver_name:
        title = f"Lap Time Prediction Results - {driver_name}"
    
    fig.update_layout(
        title=f"{title}<br><sup>RMSE: {rmse:.3f}s, Mean Error: {mean_error:.3f}s, R²: {r2:.3f}</sup>",
        xaxis_title="Actual Lap Time (seconds)",
        yaxis_title="Predicted Lap Time (seconds)",
        height=500
    )
    
    return fig

@st.cache_data
def plot_historical_performance_trend(_multi_season_df, metric='MedianLapTime', teams=None):
    """
    Create a line plot showing performance trends over multiple seasons
    
    Parameters:
    -----------
    multi_season_df : pandas.DataFrame
        DataFrame containing data from multiple seasons
    metric : str
        Performance metric to plot
    teams : list, optional
        List of team names to include (if None, all teams)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if _multi_season_df.empty or 'Season' not in _multi_season_df.columns:
        return go.Figure()
    
    # Convert lap times to seconds if not already
    if 'LapTime' in _multi_season_df.columns and not 'LapTimeSeconds' in _multi_season_df.columns:
        _multi_season_df['LapTimeSeconds'] = _multi_season_df['LapTime'].dt.total_seconds()
    
    # Calculate metrics by team and season
    if 'Team' in _multi_season_df.columns:
        # Filter teams if specified
        if teams:
            team_data = _multi_season_df[_multi_season_df['Team'].isin(teams)]
        else:
            team_data = _multi_season_df
        
        if team_data.empty:
            return go.Figure()
        
        # Calculate performance metrics
        if metric == 'MedianLapTime':
            perf_data = team_data.groupby(['Season', 'Team', 'Event'])['LapTimeSeconds'].median().reset_index()
        elif metric == 'FastestLapTime':
            perf_data = team_data.groupby(['Season', 'Team', 'Event'])['LapTimeSeconds'].min().reset_index()
        elif metric == 'Consistency':
            # Calculate within-event consistency
            perf_data = team_data.groupby(['Season', 'Team', 'Event'])['LapTimeSeconds'].std().reset_index()
        else:
            # Default to median lap time
            perf_data = team_data.groupby(['Season', 'Team', 'Event'])['LapTimeSeconds'].median().reset_index()
        
        # Aggregate across events for each season
        agg_data = perf_data.groupby(['Season', 'Team'])[metric if metric != 'MedianLapTime' and metric != 'FastestLapTime' and metric != 'Consistency' else 'LapTimeSeconds'].mean().reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add a line for each team
        for team in agg_data['Team'].unique():
            team_perf = agg_data[agg_data['Team'] == team].sort_values('Season')
            
            try:
                team_col = team_color(team)
            except:
                team_col = None
            
            fig.add_trace(go.Scatter(
                x=team_perf['Season'],
                y=team_perf['LapTimeSeconds'],
                mode='lines+markers',
                name=team,
                line=dict(color=team_col) if team_col else None
            ))
        
        # Update layout
        metric_name = {
            'MedianLapTime': 'Median Lap Time',
            'FastestLapTime': 'Fastest Lap Time',
            'Consistency': 'Consistency (Std Dev)'
        }.get(metric, metric)
        
        fig.update_layout(
            title=f"Historical {metric_name} Trends",
            xaxis_title="Season",
            yaxis_title=f"{metric_name} (seconds)",
            height=500,
            xaxis=dict(
                tickmode='array',
                tickvals=sorted(agg_data['Season'].unique())
            ),
            yaxis=dict(
                autorange=True if metric != 'Consistency' else False,
                range=[0, agg_data['LapTimeSeconds'].max() * 1.1] if metric == 'Consistency' else None
            )
        )
        
        return fig
    else:
        # If team information not available, display warning
        fig = go.Figure()
        fig.update_layout(
            title="Historical Performance Trends",
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="Team information not available for historical analysis",
                    showarrow=False,
                    font=dict(size=16)
                )
            ],
            height=500
        )
        return fig

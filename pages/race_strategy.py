import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from utils.data_loader import (
    get_available_seasons, get_available_events, load_session, 
    load_laps_data, get_all_drivers
)
from utils.statistics import (
    calculate_team_strategy_metrics, calculate_multivariate_driver_rating
)
from utils.visualizations import (
    plot_stint_strategy_comparison, plot_race_position_changes
)

def show_race_strategy():
    """Display race strategy analysis page"""
    st.title("Race Strategy Analysis")
    
    # Session selection and data loading
    with st.sidebar:
        st.header("Data Selection")
        
        # Season selector
        seasons = get_available_seasons()
        season = st.selectbox("Season", options=seasons, index=len(seasons)-1, key="strategy_season")
        
        # Event selector
        events_df = get_available_events(season)
        
        if not events_df.empty:
            event_options = events_df['EventName'].tolist()
            
            # Find the last completed event
            event_index = 0
            for i, completed in enumerate(events_df['EventDate'] < pd.Timestamp.now()):
                if completed:
                    event_index = i
            
            selected_event = st.selectbox("Event", 
                                        options=event_options,
                                        index=min(event_index, len(event_options)-1),
                                        key="strategy_event")
            
            event_round = events_df[events_df['EventName'] == selected_event]['RoundNumber'].iloc[0]
        else:
            st.warning(f"No events available for season {season}")
            selected_event = None
            event_round = None
        
        # Only race sessions are relevant for strategy analysis
        if selected_event is not None:
            session_types = ['R', 'S']  # Race or Sprint
            session_labels = ['Race', 'Sprint']
            
            session_type = st.selectbox("Session", 
                                     options=session_types,
                                     format_func=lambda x: session_labels[session_types.index(x)],
                                     key="strategy_session_type")
        else:
            session_type = None
        
        # Load data button
        if selected_event is not None and session_type is not None:
            if st.button("Load Race Data", key="strategy_load_button"):
                with st.spinner(f"Loading {selected_event} {session_type} data..."):
                    session = load_session(season, event_round, session_type)
                    
                    if session is not None:
                        # Store session in session state
                        st.session_state.strategy_session = session
                        st.session_state.strategy_laps = session.laps
                        
                        # Get race results
                        st.session_state.race_results = session.results
                        
                        # Store session info for reference
                        st.session_state.strategy_session_info = {
                            'Season': season,
                            'EventName': selected_event,
                            'RoundNumber': event_round,
                            'SessionType': session_type,
                            'Year': session.event.year
                        }
                        
                        # Display success message
                        st.success(f"Successfully loaded {selected_event} {session_type} data!")
                    else:
                        st.error("Failed to load session data. Please try another session.")
    
    # Main content area
    if 'strategy_session' not in st.session_state or st.session_state.strategy_session is None:
        st.info("Please select and load a race session from the sidebar to begin strategy analysis.")
        return
    
    # Get loaded data
    session = st.session_state.strategy_session
    laps_df = st.session_state.strategy_laps
    results_df = st.session_state.race_results if 'race_results' in st.session_state else None
    session_info = st.session_state.strategy_session_info
    
    # Check if this is actually a race session
    if session.name != 'Race' and session.name != 'Sprint':
        st.error("The loaded session is not a race. Please load a race session for strategy analysis.")
        return
    
    # Show session information
    st.markdown(f"### {session_info['EventName']} {session_info['Year']} - {session.name}")
    
    # Tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Race Overview", 
        "Pit Stop Analysis", 
        "Tire Strategy",
        "Position Battles"
    ])
    
    # Get list of drivers
    drivers = sorted(laps_df['Driver'].unique())
    
    # Tab 1: Race Overview
    with tab1:
        st.markdown("### Race Strategy Overview")
        
        # Show race results if available
        if results_df is not None and not results_df.empty:
            st.markdown("#### Race Results")
            
            # Format results for display
            display_results = results_df[['Position', 'Abbreviation', 'TeamName', 'GridPosition', 'Status', 'Points']].copy()
            display_results.columns = ['Position', 'Driver', 'Team', 'Grid', 'Status', 'Points']
            
            # Add position change
            display_results['Grid'] = pd.to_numeric(display_results['Grid'], errors='coerce')
            display_results['Position'] = pd.to_numeric(display_results['Position'], errors='coerce')
            display_results['Positions Gained/Lost'] = display_results['Grid'] - display_results['Position']
            
            # Display table
            st.dataframe(display_results, use_container_width=True)
            
            # Create position gain/loss chart
            position_change = display_results[['Driver', 'Positions Gained/Lost']].sort_values('Positions Gained/Lost', ascending=False)
            
            fig = px.bar(
                position_change,
                x='Driver',
                y='Positions Gained/Lost',
                title="Positions Gained/Lost During Race",
                color='Positions Gained/Lost',
                color_continuous_scale=px.colors.diverging.RdBu,
                labels={'Driver': 'Driver', 'Positions Gained/Lost': 'Positions Gained/Lost'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show overall race strategy
        st.markdown("#### Race Strategy Overview")
        
        # Create stint dataframe
        stint_data = []
        
        for driver in drivers:
            driver_laps = laps_df[laps_df['Driver'] == driver].copy()
            
            # Skip if too few laps
            if len(driver_laps) < 5:
                continue
            
            # Get team if available
            team = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else "Unknown"
            
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
                compound = stint_laps['Compound'].iloc[0] if 'Compound' in stint_laps.columns and pd.notna(stint_laps['Compound'].iloc[0]) else 'Unknown'
                stint_length = len(stint_laps)
                start_lap = stint_laps['LapNumber'].min()
                end_lap = stint_laps['LapNumber'].max()
                
                # Get results position if available
                final_position = None
                if results_df is not None and not results_df.empty:
                    driver_result = results_df[results_df['Abbreviation'] == driver]
                    if not driver_result.empty:
                        final_position = driver_result['Position'].iloc[0]
                
                # Collect stint data
                stint_info = {
                    'Driver': driver,
                    'Team': team,
                    'Stint': stint,
                    'Compound': compound,
                    'StintLength': stint_length,
                    'StartLap': start_lap,
                    'EndLap': end_lap,
                    'FinalPosition': final_position
                }
                
                stint_data.append(stint_info)
        
        if stint_data:
            stint_df = pd.DataFrame(stint_data)
            
            # Allow user to select drivers to display
            selected_drivers = st.multiselect(
                "Select drivers to view",
                options=drivers,
                default=drivers[:8] if len(drivers) >= 8 else drivers,
                key="overview_drivers"
            )
            
            if selected_drivers:
                # Filter for selected drivers
                selected_stint_df = stint_df[stint_df['Driver'].isin(selected_drivers)]
                
                # Show stint strategy comparison
                stint_fig = plot_stint_strategy_comparison(selected_stint_df, selected_drivers)
                st.plotly_chart(stint_fig, use_container_width=True)
                
                # Strategy summary statistics
                st.markdown("#### Strategy Summary")
                
                # Number of pit stops by driver
                pit_stops = selected_stint_df.groupby('Driver').size() - 1  # Subtract 1 to get pit stops (not stints)
                pit_stops = pit_stops.reset_index()
                pit_stops.columns = ['Driver', 'Pit Stops']
                
                # Add team if available
                if 'Team' in selected_stint_df.columns:
                    driver_teams = selected_stint_df.groupby('Driver')['Team'].first()
                    pit_stops = pit_stops.merge(driver_teams.reset_index(), on='Driver')
                
                # Add final position if available
                if 'FinalPosition' in selected_stint_df.columns and not selected_stint_df['FinalPosition'].isnull().all():
                    final_positions = selected_stint_df.groupby('Driver')['FinalPosition'].first()
                    pit_stops = pit_stops.merge(final_positions.reset_index(), on='Driver')
                
                # Add compound usage if available
                if 'Compound' in selected_stint_df.columns:
                    for driver in selected_drivers:
                        driver_stints = selected_stint_df[selected_stint_df['Driver'] == driver]
                        compounds = driver_stints['Compound'].tolist()
                        pit_stops.loc[pit_stops['Driver'] == driver, 'Strategy'] = ' → '.join(compounds)
                
                st.dataframe(pit_stops, use_container_width=True)
            else:
                st.warning("Please select at least one driver to view strategy.")
        else:
            st.warning("No stint data available for strategy analysis.")
    
    # Tab 2: Pit Stop Analysis
    with tab2:
        st.markdown("### Pit Stop Analysis")
        
        # Get pit stop data
        pit_laps = laps_df[laps_df['PitInTime'].notnull()].copy()
        
        if not pit_laps.empty:
            # Calculate pit stop durations
            pit_data = []
            
            for _, lap in pit_laps.iterrows():
                driver = lap['Driver']
                
                # Get team if available
                team = lap['Team'] if 'Team' in lap and pd.notna(lap['Team']) else "Unknown"
                
                # Calculate pit duration
                if pd.notnull(lap['PitOutTime']) and pd.notnull(lap['PitInTime']):
                    pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
                    
                    # Only include reasonable durations (> 0)
                    if pit_duration > 0:
                        pit_info = {
                            'Driver': driver,
                            'Team': team,
                            'Lap': lap['LapNumber'],
                            'PitDuration': pit_duration
                        }
                        
                        # Add compound change if available
                        if 'Compound' in lap:
                            next_lap_idx = laps_df[(laps_df['Driver'] == driver) & 
                                                  (laps_df['LapNumber'] > lap['LapNumber'])].index.min()
                            
                            if pd.notna(next_lap_idx):
                                next_lap = laps_df.loc[next_lap_idx]
                                if 'Compound' in next_lap and pd.notna(next_lap['Compound']):
                                    pit_info['CompoundBefore'] = lap['Compound']
                                    pit_info['CompoundAfter'] = next_lap['Compound']
                        
                        pit_data.append(pit_info)
            
            # Create DataFrame
            if pit_data:
                pit_df = pd.DataFrame(pit_data)
                
                # Show pit stop durations
                st.markdown("#### Pit Stop Durations")
                
                # Allow user to select teams for comparison
                if 'Team' in pit_df.columns:
                    teams = sorted(pit_df['Team'].unique())
                    selected_teams = st.multiselect(
                        "Select teams to compare",
                        options=teams,
                        default=teams,
                        key="pit_teams"
                    )
                    
                    if selected_teams:
                        # Filter for selected teams
                        team_pit_df = pit_df[pit_df['Team'].isin(selected_teams)]
                    else:
                        team_pit_df = pit_df
                else:
                    team_pit_df = pit_df
                
                # Show pit stop duration box plot
                fig = px.box(
                    team_pit_df,
                    x='Team' if 'Team' in team_pit_df.columns else 'Driver',
                    y='PitDuration',
                    points='all',
                    hover_data=['Driver', 'Lap'],
                    title="Pit Stop Duration Comparison",
                    labels={
                        'Team': 'Team',
                        'Driver': 'Driver',
                        'PitDuration': 'Pit Stop Duration (seconds)'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show pit stop summary statistics
                st.markdown("#### Pit Stop Statistics")
                
                # Calculate statistics by team
                if 'Team' in pit_df.columns:
                    team_stats = []
                    
                    for team in pit_df['Team'].unique():
                        team_pits = pit_df[pit_df['Team'] == team]
                        
                        stats = {
                            'Team': team,
                            'Total Pit Stops': len(team_pits),
                            'Avg Duration': team_pits['PitDuration'].mean(),
                            'Min Duration': team_pits['PitDuration'].min(),
                            'Max Duration': team_pits['PitDuration'].max(),
                            'Std Dev': team_pits['PitDuration'].std()
                        }
                        
                        team_stats.append(stats)
                    
                    team_stats_df = pd.DataFrame(team_stats)
                    st.dataframe(team_stats_df, use_container_width=True)
                
                # Calculate statistics by driver
                driver_stats = []
                
                for driver in pit_df['Driver'].unique():
                    driver_pits = pit_df[pit_df['Driver'] == driver]
                    
                    stats = {
                        'Driver': driver,
                        'Team': driver_pits['Team'].iloc[0] if 'Team' in driver_pits.columns else "Unknown",
                        'Pit Stops': len(driver_pits),
                        'Avg Duration': round(driver_pits['PitDuration'].mean(), 2),
                        'Min Duration': round(driver_pits['PitDuration'].min(), 2),
                        'Max Duration': round(driver_pits['PitDuration'].max(), 2)
                    }
                    
                    driver_stats.append(stats)
                
                driver_stats_df = pd.DataFrame(driver_stats).sort_values('Avg Duration')
                st.dataframe(driver_stats_df, use_container_width=True)
                
                # Show pit timing analysis
                st.markdown("#### Pit Stop Timing")
                
                # Create scatter plot of pit stop timing
                fig = px.scatter(
                    pit_df,
                    x='Lap',
                    y='Driver',
                    size='PitDuration',
                    color='Team' if 'Team' in pit_df.columns else 'Driver',
                    hover_data=['PitDuration', 'CompoundBefore', 'CompoundAfter'] if 'CompoundBefore' in pit_df.columns else ['PitDuration'],
                    title="Pit Stop Timing by Driver",
                    labels={
                        'Lap': 'Lap Number',
                        'Driver': 'Driver',
                        'PitDuration': 'Pit Duration (s)'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate undercut/overcut patterns if compound data available
                if 'CompoundBefore' in pit_df.columns and 'CompoundAfter' in pit_df.columns:
                    st.markdown("#### Tire Compound Changes")
                    
                    # Count compound transitions
                    transitions = pit_df.groupby(['CompoundBefore', 'CompoundAfter']).size().reset_index()
                    transitions.columns = ['From', 'To', 'Count']
                    
                    # Sort by count
                    transitions = transitions.sort_values('Count', ascending=False)
                    
                    # Show transition table
                    st.dataframe(transitions, use_container_width=True)
                    
                    # Show heatmap of transitions
                    fig = px.density_heatmap(
                        pit_df,
                        x='CompoundBefore',
                        y='CompoundAfter',
                        title="Tire Compound Transitions",
                        labels={
                            'CompoundBefore': 'From Compound',
                            'CompoundAfter': 'To Compound'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No pit stop data available for analysis.")
        else:
            st.warning("No pit stop data available in the race.")
    
    # Tab 3: Tire Strategy
    with tab3:
        st.markdown("### Tire Strategy Analysis")
        
        # Check if compound data is available
        if 'Compound' not in laps_df.columns or laps_df['Compound'].isnull().all():
            st.warning("Tire compound data is not available for this race.")
            return
        
        # Allow selection of drivers for comparison
        selected_drivers = st.multiselect(
            "Select drivers to analyze",
            options=drivers,
            default=drivers[:5] if len(drivers) >= 5 else drivers,
            key="tire_drivers"
        )
        
        if selected_drivers:
            # Filter for selected drivers
            driver_laps = laps_df[laps_df['Driver'].isin(selected_drivers)]
            
            # Show compound usage over race
            st.markdown("#### Tire Compound Usage")
            
            # Create stacked area chart of compound usage
            compound_data = []
            
            for driver in selected_drivers:
                driver_data = driver_laps[driver_laps['Driver'] == driver]
                
                for lap_num in range(1, int(driver_data['LapNumber'].max()) + 1):
                    lap_compound = driver_data[driver_data['LapNumber'] == lap_num]['Compound'].values
                    
                    if len(lap_compound) > 0 and pd.notna(lap_compound[0]):
                        compound_data.append({
                            'Driver': driver,
                            'Lap': lap_num,
                            'Compound': lap_compound[0]
                        })
            
            if compound_data:
                compound_df = pd.DataFrame(compound_data)
                
                # Create line chart with discrete colors
                fig = px.line(
                    compound_df,
                    x='Lap',
                    y='Driver',
                    color='Compound',
                    line_dash='Compound',
                    title="Tire Compound by Lap",
                    labels={
                        'Lap': 'Lap Number',
                        'Driver': 'Driver',
                        'Compound': 'Compound'
                    }
                )
                
                # Customize to make it more readable
                fig.update_layout(
                    yaxis=dict(
                        categoryorder='array',
                        categoryarray=selected_drivers
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyze pace by compound
            st.markdown("#### Pace by Compound")
            
            # Calculate lap time statistics by compound
            compound_stats = []
            
            for driver in selected_drivers:
                driver_data = driver_laps[driver_laps['Driver'] == driver]
                
                # Get team
                team = driver_data['Team'].iloc[0] if 'Team' in driver_data.columns else "Unknown"
                
                # Group by compound
                for compound, compound_laps in driver_data.groupby('Compound'):
                    if pd.isna(compound):
                        continue
                    
                    # Filter valid laps
                    valid_laps = compound_laps[(compound_laps['PitOutTime'].isnull()) & 
                                             (compound_laps['PitInTime'].isnull())]
                    
                    if len(valid_laps) < 3:
                        continue
                    
                    # Calculate statistics
                    lap_times = valid_laps['LapTime'].dt.total_seconds()
                    
                    stats = {
                        'Driver': driver,
                        'Team': team,
                        'Compound': compound,
                        'Laps': len(valid_laps),
                        'MedianLapTime': lap_times.median(),
                        'MeanLapTime': lap_times.mean(),
                        'MinLapTime': lap_times.min(),
                        'StdDev': lap_times.std()
                    }
                    
                    # Calculate degradation if enough laps
                    if len(valid_laps) >= 5:
                        # Sort by lap number
                        valid_laps = valid_laps.sort_values('LapNumber')
                        
                        # Create relative lap numbers for the compound stint
                        rel_lap_numbers = range(len(valid_laps))
                        
                        # Simple linear regression for degradation
                        try:
                            from scipy import stats as scipy_stats
                            slope, _, _, _, _ = scipy_stats.linregress(rel_lap_numbers, lap_times)
                            stats['DegradationPerLap'] = slope
                        except:
                            stats['DegradationPerLap'] = np.nan
                    
                    compound_stats.append(stats)
            
            # Create DataFrame
            if compound_stats:
                compound_stats_df = pd.DataFrame(compound_stats)
                
                # Show statistics table
                st.dataframe(compound_stats_df, use_container_width=True)
                
                # Create compound comparison chart
                fig = px.bar(
                    compound_stats_df,
                    x='Driver',
                    y='MedianLapTime',
                    color='Compound',
                    barmode='group',
                    error_y=compound_stats_df['StdDev'],
                    title="Lap Time by Compound",
                    labels={
                        'Driver': 'Driver',
                        'MedianLapTime': 'Median Lap Time (seconds)',
                        'Compound': 'Compound'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show degradation comparison if available
                if 'DegradationPerLap' in compound_stats_df.columns and not compound_stats_df['DegradationPerLap'].isnull().all():
                    fig = px.bar(
                        compound_stats_df,
                        x='Driver',
                        y='DegradationPerLap',
                        color='Compound',
                        barmode='group',
                        title="Tire Degradation by Compound",
                        labels={
                            'Driver': 'Driver',
                            'DegradationPerLap': 'Degradation (seconds per lap)',
                            'Compound': 'Compound'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("*Positive values indicate decreasing performance (higher lap times) with tire age*")
            else:
                st.warning("Insufficient data to analyze pace by compound.")
        else:
            st.warning("Please select at least one driver for tire strategy analysis.")
    
    # Tab 4: Position Battles
    with tab4:
        st.markdown("### Position Battles")
        
        # Check if position data is available
        if 'Position' not in laps_df.columns or laps_df['Position'].isnull().all():
            st.warning("Position data is not available for this race.")
            return
        
        # Allow selection of drivers for position battle analysis
        battle_drivers = st.multiselect(
            "Select drivers to analyze position battles",
            options=drivers,
            default=drivers[:5] if len(drivers) >= 5 else drivers,
            key="battle_drivers"
        )
        
        if battle_drivers:
            # Show position changes throughout the race
            st.markdown("#### Position Changes")
            
            position_fig = plot_race_position_changes(_laps_df=laps_df, drivers=battle_drivers)
            st.plotly_chart(position_fig, use_container_width=True)
            
            # Analyze key position battles
            st.markdown("#### Key Position Battles")
            
            # Allow selection of specific battle
            battle_1 = st.selectbox(
                "Select first driver",
                options=battle_drivers,
                index=0,
                key="battle_driver_1"
            )
            
            battle_2 = st.selectbox(
                "Select second driver",
                options=[d for d in battle_drivers if d != battle_1],
                index=0,
                key="battle_driver_2"
            )
            
            if battle_1 and battle_2:
                # Get position data for both drivers
                driver1_data = laps_df[laps_df['Driver'] == battle_1][['LapNumber', 'Position']].set_index('LapNumber')
                driver2_data = laps_df[laps_df['Driver'] == battle_2][['LapNumber', 'Position']].set_index('LapNumber')
                
                # Combine data
                battle_data = pd.DataFrame({
                    battle_1: driver1_data['Position'],
                    battle_2: driver2_data['Position']
                })
                
                # Calculate position difference
                battle_data['PositionDiff'] = battle_data[battle_1] - battle_data[battle_2]
                
                # Identify when positions changed
                battle_data['Overtake'] = battle_data['PositionDiff'].diff() != 0
                
                # Create figure
                fig = go.Figure()
                
                # Add position traces
                fig.add_trace(go.Scatter(
                    x=battle_data.index,
                    y=battle_data[battle_1],
                    mode='lines+markers',
                    name=battle_1
                ))
                
                fig.add_trace(go.Scatter(
                    x=battle_data.index,
                    y=battle_data[battle_2],
                    mode='lines+markers',
                    name=battle_2
                ))
                
                # Highlight overtakes
                overtakes = battle_data[battle_data['Overtake']]
                
                if not overtakes.empty:
                    for idx, row in overtakes.iterrows():
                        # Add annotation for overtake
                        if pd.notna(row[battle_1]) and pd.notna(row[battle_2]):
                            if row[battle_1] < row[battle_2]:
                                text = f"{battle_1} ahead"
                            else:
                                text = f"{battle_2} ahead"
                            
                            fig.add_annotation(
                                x=idx,
                                y=min(row[battle_1], row[battle_2]),
                                text=text,
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=1,
                                arrowcolor="#636363"
                            )
                
                # Update layout
                fig.update_layout(
                    title=f"Position Battle: {battle_1} vs {battle_2}",
                    xaxis_title="Lap Number",
                    yaxis_title="Position",
                    yaxis=dict(
                        autorange="reversed",  # Reverse to show P1 at the top
                        dtick=1
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show lap time comparison during battle
                st.markdown("#### Lap Time Comparison")
                
                # Get lap times for both drivers
                driver1_laps = laps_df[laps_df['Driver'] == battle_1]
                driver2_laps = laps_df[laps_df['Driver'] == battle_2]
                
                # Filter valid lap times
                driver1_valid = driver1_laps[(driver1_laps['PitOutTime'].isnull()) & 
                                          (driver1_laps['PitInTime'].isnull())]
                driver2_valid = driver2_laps[(driver2_laps['PitOutTime'].isnull()) & 
                                          (driver2_laps['PitInTime'].isnull())]
                
                # Create lap time comparison
                lap_time_data = pd.DataFrame({
                    'LapNumber': pd.concat([driver1_valid['LapNumber'], driver2_valid['LapNumber']]).unique()
                }).sort_values('LapNumber')
                
                # Add lap times for each driver
                for lap_num in lap_time_data['LapNumber']:
                    # Driver 1
                    d1_lap = driver1_valid[driver1_valid['LapNumber'] == lap_num]
                    if not d1_lap.empty and not d1_lap['LapTime'].isnull().all():
                        lap_time_data.loc[lap_time_data['LapNumber'] == lap_num, f'{battle_1} Time'] = d1_lap['LapTime'].iloc[0].total_seconds()
                    
                    # Driver 2
                    d2_lap = driver2_valid[driver2_valid['LapNumber'] == lap_num]
                    if not d2_lap.empty and not d2_lap['LapTime'].isnull().all():
                        lap_time_data.loc[lap_time_data['LapNumber'] == lap_num, f'{battle_2} Time'] = d2_lap['LapTime'].iloc[0].total_seconds()
                
                # Calculate time difference
                if f'{battle_1} Time' in lap_time_data.columns and f'{battle_2} Time' in lap_time_data.columns:
                    lap_time_data['Time Diff'] = lap_time_data[f'{battle_1} Time'] - lap_time_data[f'{battle_2} Time']
                    
                    # Create comparison chart
                    fig = make_subplots(rows=2, cols=1, 
                                      shared_xaxes=True,
                                      vertical_spacing=0.1,
                                      subplot_titles=("Lap Times", "Time Difference"))
                    
                    # Add lap time traces
                    fig.add_trace(
                        go.Scatter(
                            x=lap_time_data['LapNumber'],
                            y=lap_time_data[f'{battle_1} Time'],
                            mode='lines+markers',
                            name=battle_1
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=lap_time_data['LapNumber'],
                            y=lap_time_data[f'{battle_2} Time'],
                            mode='lines+markers',
                            name=battle_2
                        ),
                        row=1, col=1
                    )
                    
                    # Add time difference
                    fig.add_trace(
                        go.Bar(
                            x=lap_time_data['LapNumber'],
                            y=lap_time_data['Time Diff'],
                            name='Time Diff',
                            marker_color=np.where(lap_time_data['Time Diff'] > 0, 'indianred', 'royalblue')
                        ),
                        row=2, col=1
                    )
                    
                    # Add zero line to difference chart
                    fig.add_shape(
                        type="line",
                        x0=lap_time_data['LapNumber'].min(),
                        x1=lap_time_data['LapNumber'].max(),
                        y0=0,
                        y1=0,
                        line=dict(color="black", width=1, dash="dash"),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=700,
                        title_text=f"Lap Time Comparison: {battle_1} vs {battle_2}",
                        xaxis2_title="Lap Number",
                        yaxis_title="Lap Time (seconds)",
                        yaxis2_title="Time Difference (seconds)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown(f"*Negative values (blue) mean {battle_2} was faster, positive values (red) mean {battle_1} was faster*")
                else:
                    st.warning("Insufficient lap time data for comparison.")
            else:
                st.warning("Please select two different drivers for battle analysis.")
        else:
            st.warning("Please select at least one driver for position battle analysis.")

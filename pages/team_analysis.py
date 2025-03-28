import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import (
    get_available_seasons, get_available_events, load_session, 
    get_all_teams, load_laps_data, get_constructor_championship_standings
)
from utils.statistics import (
    calculate_team_performance_metrics, calculate_team_strategy_metrics
)
from utils.visualizations import (
    plot_team_performance_comparison, plot_stint_strategy_comparison
)

def show_team_analysis():
    """Display team analysis page"""
    st.title("Team Performance Analysis")
    
    # Session selection and data loading
    with st.sidebar:
        st.header("Data Selection")
        
        # Season selector
        seasons = get_available_seasons()
        season = st.selectbox("Season", options=seasons, index=len(seasons)-1, key="team_season")
        
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
                                        key="team_event")
            
            event_round = events_df[events_df['EventName'] == selected_event]['RoundNumber'].iloc[0]
        else:
            st.warning(f"No events available for season {season}")
            selected_event = None
            event_round = None
        
        # Session type selector
        if selected_event is not None:
            session_types = ['R', 'Q', 'S', 'FP3', 'FP2', 'FP1']
            session_labels = ['Race', 'Qualifying', 'Sprint', 'Practice 3', 'Practice 2', 'Practice 1']
            
            session_type = st.selectbox("Session", 
                                     options=session_types,
                                     format_func=lambda x: session_labels[session_types.index(x)],
                                     key="team_session_type")
        else:
            session_type = None
        
        # Load data button
        if selected_event is not None and session_type is not None:
            if st.button("Load Session Data", key="team_load_button"):
                with st.spinner(f"Loading {selected_event} {session_type} data..."):
                    session = load_session(season, event_round, session_type)
                    
                    if session is not None:
                        # Store session in session state
                        st.session_state.team_analysis_session = session
                        st.session_state.team_analysis_laps = session.laps
                        
                        # Store session info for reference
                        st.session_state.team_session_info = {
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
        
        # Load championship standings
        if season:
            if st.button("Load Championship Standings", key="team_standings_button"):
                with st.spinner("Loading constructor championship standings..."):
                    standings = get_constructor_championship_standings(season)
                    
                    if not standings.empty:
                        st.session_state.constructor_standings = standings
                        st.success("Successfully loaded constructor standings!")
                    else:
                        st.error("Failed to load constructor standings.")
    
    # Main content area
    if 'team_analysis_session' not in st.session_state or st.session_state.team_analysis_session is None:
        st.info("Please select and load a session from the sidebar to begin team analysis.")
        
        # Show championship standings if available
        if 'constructor_standings' in st.session_state and not st.session_state.constructor_standings.empty:
            show_championship_standings()
        
        return
    
    # Get loaded data
    session = st.session_state.team_analysis_session
    laps_df = st.session_state.team_analysis_laps
    session_info = st.session_state.team_session_info
    
    # Add team information if not present
    if 'Team' not in laps_df.columns and 'TeamName' in laps_df.columns:
        laps_df['Team'] = laps_df['TeamName']
    
    # Check if team information is available
    if 'Team' not in laps_df.columns:
        st.error("Team information not available in the loaded data. Please try another session.")
        return
    
    # Show session information
    st.markdown(f"### {session_info['EventName']} {session_info['Year']} - {session.name}")
    
    # Tabs for different analysis views
    tab1, tab2, tab3 = st.tabs([
        "Performance Comparison", 
        "Team Strategy", 
        "Championship Context"
    ])
    
    # Get list of teams
    teams = sorted(laps_df['Team'].unique())
    
    # Tab 1: Performance Comparison
    with tab1:
        st.markdown("### Team Performance Comparison")
        
        # Allow user to select teams for comparison
        selected_teams = st.multiselect(
            "Select teams to compare",
            options=teams,
            default=teams,
            key="perf_teams"
        )
        
        if selected_teams:
            # Show performance comparison
            st.markdown("#### Performance Metrics")
            performance_fig = plot_team_performance_comparison(_laps_df=laps_df, teams=selected_teams)
            st.plotly_chart(performance_fig, use_container_width=True)
            
            # Detailed metrics for each team
            st.markdown("#### Detailed Team Metrics")
            
            team_metrics_list = []
            for team in selected_teams:
                metrics = calculate_team_performance_metrics(laps_df, team)
                
                if metrics:
                    team_metrics = {
                        'Team': team
                    }
                    
                    # Add available metrics
                    if 'best_lap_time' in metrics:
                        team_metrics['Best Lap'] = metrics['best_lap_time']
                        team_metrics['Best Lap Driver'] = metrics['best_lap_driver']
                    
                    if 'avg_lap_time' in metrics:
                        team_metrics['Avg Lap Time'] = metrics['avg_lap_time']
                    
                    if 'lap_time_std' in metrics:
                        team_metrics['Consistency (σ)'] = metrics['lap_time_std']
                    
                    if 'total_pit_stops' in metrics:
                        team_metrics['Pit Stops'] = metrics['total_pit_stops']
                    
                    if 'avg_pit_time' in metrics:
                        team_metrics['Avg Pit Time'] = metrics['avg_pit_time']
                    
                    team_metrics_list.append(team_metrics)
            
            # Create DataFrame and show
            if team_metrics_list:
                metrics_df = pd.DataFrame(team_metrics_list)
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.warning("No metrics available for the selected teams.")
            
            # Driver comparison within teams
            st.markdown("#### Driver Comparison Within Teams")
            
            selected_team_drivers = st.selectbox(
                "Select team to analyze drivers",
                options=selected_teams,
                key="team_drivers_select"
            )
            
            if selected_team_drivers:
                # Get drivers for the selected team
                team_drivers = laps_df[laps_df['Team'] == selected_team_drivers]['Driver'].unique()
                
                if len(team_drivers) > 0:
                    # Calculate driver metrics
                    driver_metrics = []
                    
                    for driver in team_drivers:
                        driver_laps = laps_df[(laps_df['Team'] == selected_team_drivers) & 
                                             (laps_df['Driver'] == driver)]
                        
                        # Filter valid laps
                        valid_laps = driver_laps[(driver_laps['PitOutTime'].isnull()) & 
                                                (driver_laps['PitInTime'].isnull())]
                        
                        if not valid_laps.empty and not valid_laps['LapTime'].isnull().all():
                            lap_times = valid_laps['LapTime'].dt.total_seconds()
                            
                            metrics = {
                                'Driver': driver,
                                'Best Lap': valid_laps['LapTime'].min(),
                                'Median Lap': pd.Timedelta(seconds=lap_times.median()),
                                'Mean Lap': pd.Timedelta(seconds=lap_times.mean()),
                                'Std Dev (s)': lap_times.std(),
                                'Laps': len(valid_laps)
                            }
                            
                            driver_metrics.append(metrics)
                    
                    if driver_metrics:
                        # Create DataFrame and show
                        driver_df = pd.DataFrame(driver_metrics)
                        st.dataframe(driver_df, use_container_width=True)
                        
                        # Create comparison chart
                        if len(driver_metrics) > 1:
                            fig = px.bar(
                                driver_df,
                                x='Driver',
                                y=[time.total_seconds() for time in driver_df['Median Lap']],
                                error_y=driver_df['Std Dev (s)'],
                                labels={'x': 'Driver', 'y': 'Lap Time (seconds)'},
                                title=f"Driver Comparison - {selected_team_drivers}"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No driver metrics available for {selected_team_drivers}.")
                else:
                    st.warning(f"No drivers found for {selected_team_drivers}.")
        else:
            st.warning("Please select at least one team for comparison.")
    
    # Tab 2: Team Strategy
    with tab2:
        st.markdown("### Team Strategy Analysis")
        
        # Only applicable for race sessions
        if session.name == 'Race' or session.name == 'Sprint':
            # Allow user to select teams for comparison
            strategy_teams = st.multiselect(
                "Select teams to analyze",
                options=teams,
                default=teams[:2] if len(teams) >= 2 else teams,
                key="strategy_teams"
            )
            
            if strategy_teams:
                # Calculate strategy metrics
                strategy_metrics = calculate_team_strategy_metrics(laps_df)
                
                if not strategy_metrics.empty:
                    # Filter for selected teams
                    team_strategies = strategy_metrics[strategy_metrics['Team'].isin(strategy_teams)]
                    
                    # Display strategy metrics
                    st.markdown("#### Strategy Overview")
                    
                    # Prepare display dataframe
                    display_cols = ['Team', 'TotalPitStops']
                    
                    # Add compound columns if available
                    compound_cols = [col for col in team_strategies.columns if col.startswith('Laps')]
                    if compound_cols:
                        display_cols.extend(compound_cols)
                    
                    # Add most used compound if available
                    if 'MostUsedCompound' in team_strategies.columns:
                        display_cols.append('MostUsedCompound')
                    
                    # Add pit time metrics if available
                    pit_time_cols = ['AvgPitTime', 'MinPitTime', 'MaxPitTime']
                    for col in pit_time_cols:
                        if col in team_strategies.columns:
                            display_cols.append(col)
                    
                    # Create display dataframe
                    display_df = team_strategies[display_cols].copy()
                    
                    # Format time columns
                    time_cols = [col for col in display_df.columns if 'Time' in col]
                    for col in time_cols:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].round(3)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Pit stop comparison
                    if 'TotalPitStops' in team_strategies.columns:
                        st.markdown("#### Pit Stop Comparison")
                        
                        fig = px.bar(
                            team_strategies,
                            x='Team',
                            y='TotalPitStops',
                            title="Pit Stop Count by Team",
                            color='Team'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tire compound usage if available
                    compound_cols = [col for col in team_strategies.columns if col.startswith('Laps')]
                    if compound_cols:
                        st.markdown("#### Tire Compound Usage")
                        
                        # Reshape data for plotting
                        plot_data = []
                        for _, row in team_strategies.iterrows():
                            team = row['Team']
                            for col in compound_cols:
                                compound = col.replace('Laps', '')
                                plot_data.append({
                                    'Team': team,
                                    'Compound': compound,
                                    'Laps': row[col] if pd.notna(row[col]) else 0
                                })
                        
                        plot_df = pd.DataFrame(plot_data)
                        
                        # Create stacked bar chart
                        fig = px.bar(
                            plot_df,
                            x='Team',
                            y='Laps',
                            color='Compound',
                            title="Tire Compound Usage by Team",
                            barmode='stack'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show pit time comparison if available
                    if 'AvgPitTime' in team_strategies.columns:
                        st.markdown("#### Pit Stop Performance")
                        
                        fig = go.Figure()
                        
                        for _, row in team_strategies.iterrows():
                            team = row['Team']
                            
                            # Only add if pit times are available
                            if pd.notna(row.get('AvgPitTime')) and pd.notna(row.get('MinPitTime')):
                                fig.add_trace(go.Box(
                                    name=team,
                                    y=[row['MinPitTime'].total_seconds(), 
                                       row['AvgPitTime'].total_seconds(), 
                                       row.get('MaxPitTime', row['AvgPitTime']).total_seconds()],
                                    boxpoints=False,
                                    marker_color=None
                                ))
                        
                        fig.update_layout(
                            title="Pit Stop Time Comparison",
                            yaxis_title="Time (seconds)",
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Show stint strategy comparison
                st.markdown("#### Race Stint Strategy")
                
                # Get team drivers
                team_drivers = []
                for team in strategy_teams:
                    drivers = laps_df[laps_df['Team'] == team]['Driver'].unique()
                    team_drivers.extend(drivers)
                
                # Create stint dataframe
                stint_data = []
                
                for driver in team_drivers:
                    driver_laps = laps_df[laps_df['Driver'] == driver].copy()
                    
                    # Skip if too few laps
                    if len(driver_laps) < 5:
                        continue
                    
                    # Get team
                    team = driver_laps['Team'].iloc[0]
                    
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
                        if session.results is not None and not session.results.empty:
                            driver_result = session.results[session.results['Abbreviation'] == driver]
                            if not driver_result.empty:
                                final_position = driver_result['Position'].iloc[0]
                        
                        # Collect stint data
                        stint_info = {
                            'Driver': driver,
                            'Team': team,
                            'Stint': stint,
                            'Compound': compound,
                            'StartLap': start_lap,
                            'EndLap': end_lap,
                            'FinalPosition': final_position
                        }
                        
                        stint_data.append(stint_info)
                
                if stint_data:
                    stint_df = pd.DataFrame(stint_data)
                    
                    # Show stint strategy comparison
                    stint_fig = plot_stint_strategy_comparison(stint_df, team_drivers)
                    st.plotly_chart(stint_fig, use_container_width=True)
                else:
                    st.warning("No stint data available for the selected teams.")
            else:
                st.warning("Please select at least one team for strategy analysis.")
        else:
            st.info("Strategy analysis is only available for Race or Sprint sessions.")
    
    # Tab 3: Championship Context
    with tab3:
        st.markdown("### Championship Context")
        
        # Show championship standings if available
        if 'constructor_standings' in st.session_state and not st.session_state.constructor_standings.empty:
            show_championship_standings()
        else:
            st.info("Load championship standings from the sidebar to view championship context.")
            
            if st.button("Load Championship Standings", key="tab3_load_standings"):
                with st.spinner("Loading constructor championship standings..."):
                    standings = get_constructor_championship_standings(session_info['Season'])
                    
                    if not standings.empty:
                        st.session_state.constructor_standings = standings
                        st.success("Successfully loaded constructor standings!")
                        st.rerun()
                    else:
                        st.error("Failed to load constructor standings.")
        
        # Show team performance in championship context
        if 'constructor_standings' in st.session_state and not st.session_state.constructor_standings.empty:
            st.markdown("#### Session Performance vs Championship Standing")
            
            # Get session performance metrics
            if not teams:
                st.warning("No team data available in the current session.")
                return
            
            team_session_metrics = []
            for team in teams:
                team_laps = laps_df[laps_df['Team'] == team]
                
                # Skip if too few laps
                if len(team_laps) < 5:
                    continue
                
                # Calculate median lap time
                valid_laps = team_laps[(team_laps['PitOutTime'].isnull()) & 
                                     (team_laps['PitInTime'].isnull())]
                
                if valid_laps.empty or valid_laps['LapTime'].isnull().all():
                    continue
                
                median_lap = valid_laps['LapTime'].dt.total_seconds().median()
                
                # Get championship position
                standings = st.session_state.constructor_standings
                team_standing = standings[standings['TeamName'] == team]
                
                if team_standing.empty:
                    # Try alternative matching
                    for standing_team in standings['TeamName']:
                        if team in standing_team or standing_team in team:
                            team_standing = standings[standings['TeamName'] == standing_team]
                            break
                
                championship_pos = team_standing['Position'].iloc[0] if not team_standing.empty else np.nan
                championship_pts = team_standing['Points'].iloc[0] if not team_standing.empty else np.nan
                
                team_session_metrics.append({
                    'Team': team,
                    'MedianLapTime': median_lap,
                    'ChampionshipPosition': championship_pos,
                    'ChampionshipPoints': championship_pts
                })
            
            session_metrics_df = pd.DataFrame(team_session_metrics)
            
            if not session_metrics_df.empty and not session_metrics_df['ChampionshipPosition'].isnull().all():
                # Create scatter plot
                fig = px.scatter(
                    session_metrics_df,
                    x='ChampionshipPosition',
                    y='MedianLapTime',
                    text='Team',
                    size='ChampionshipPoints',
                    title=f"Session Performance vs Championship Standing - {session_info['EventName']}",
                    labels={
                        'ChampionshipPosition': 'Championship Position',
                        'MedianLapTime': 'Median Lap Time (seconds)',
                        'ChampionshipPoints': 'Championship Points'
                    }
                )
                
                fig.update_traces(textposition='top center')
                fig.update_yaxes(autorange="reversed")  # Lower lap time is better
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("*Better session performance is indicated by lower lap times (higher on the chart)*")
            else:
                st.warning("Insufficient data to compare session performance with championship standings.")

def show_championship_standings():
    """Display constructor championship standings"""
    if 'constructor_standings' not in st.session_state or st.session_state.constructor_standings.empty:
        return
    
    standings = st.session_state.constructor_standings
    
    st.markdown("#### Constructor Championship Standings")
    
    # Format standings for display
    display_standings = standings[['Position', 'TeamName', 'Points']].copy()
    display_standings.columns = ['Position', 'Team', 'Points']
    
    st.dataframe(display_standings, use_container_width=True)
    
    # Create bar chart
    fig = px.bar(
        standings,
        x='TeamName',
        y='Points',
        title="Constructor Championship Points",
        color='TeamName',
        labels={'TeamName': 'Team', 'Points': 'Points'}
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

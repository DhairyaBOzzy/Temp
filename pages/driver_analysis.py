import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import (
    get_available_seasons, get_available_events, load_session, 
    get_all_drivers, load_laps_data
)
from utils.statistics import (
    calculate_driver_consistency, analyze_race_pace_degradation,
    compare_stint_performance, calculate_multivariate_driver_rating
)
from utils.visualizations import (
    plot_driver_pace_comparison, plot_lap_time_comparison,
    plot_driver_performance_radar
)

def show_driver_analysis():
    """Display driver analysis page"""
    st.title("Driver Performance Analysis")
    
    # Session selection and data loading
    with st.sidebar:
        st.header("Data Selection")
        
        # Season selector
        seasons = get_available_seasons()
        season = st.selectbox("Season", options=seasons, index=len(seasons)-1, key="driver_season")
        
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
                                        key="driver_event")
            
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
                                     key="driver_session_type")
        else:
            session_type = None
        
        # Load data button
        if selected_event is not None and session_type is not None:
            if st.button("Load Session Data", key="driver_load_button"):
                with st.spinner(f"Loading {selected_event} {session_type} data..."):
                    session = load_session(season, event_round, session_type)
                    
                    if session is not None:
                        # Store session in session state
                        st.session_state.driver_analysis_session = session
                        st.session_state.driver_analysis_laps = session.laps
                        
                        # Store session info for reference
                        st.session_state.driver_session_info = {
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
    if 'driver_analysis_session' not in st.session_state or st.session_state.driver_analysis_session is None:
        st.info("Please select and load a session from the sidebar to begin driver analysis.")
        return
    
    # Get loaded data
    session = st.session_state.driver_analysis_session
    laps_df = st.session_state.driver_analysis_laps
    session_info = st.session_state.driver_session_info
    
    # Show session information
    st.markdown(f"### {session_info['EventName']} {session_info['Year']} - {session.name}")
    
    # Tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Pace Analysis", 
        "Consistency & Degradation", 
        "Stint Comparison",
        "Performance Rating"
    ])
    
    # Get list of drivers
    drivers = sorted(laps_df['Driver'].unique())
    
    # Tab 1: Pace Analysis
    with tab1:
        st.markdown("### Driver Pace Comparison")
        
        # Allow user to select drivers for comparison
        selected_drivers = st.multiselect(
            "Select drivers to compare",
            options=drivers,
            default=drivers[:5] if len(drivers) >= 5 else drivers,
            key="pace_drivers"
        )
        
        if selected_drivers:
            # Show lap time distribution
            st.markdown("#### Lap Time Distribution")
            lap_time_fig = plot_lap_time_comparison(_laps_df=laps_df, drivers=selected_drivers)
            st.plotly_chart(lap_time_fig, use_container_width=True)
            
            # Show lap time progression
            st.markdown("#### Lap Time Progression")
            pace_fig = plot_driver_pace_comparison(_laps_df=laps_df, drivers=selected_drivers, session_info=session_info)
            st.plotly_chart(pace_fig, use_container_width=True)
            
            # Detailed statistics table
            st.markdown("#### Detailed Pace Statistics")
            
            pace_stats = []
            for driver in selected_drivers:
                driver_laps = laps_df[laps_df['Driver'] == driver]
                valid_laps = driver_laps[(driver_laps['PitOutTime'].isnull()) & 
                                      (driver_laps['PitInTime'].isnull())]
                
                if not valid_laps.empty and not valid_laps['LapTime'].isnull().all():
                    lap_times = valid_laps['LapTime'].dt.total_seconds()
                    
                    stats = {
                        'Driver': driver,
                        'Best Lap': valid_laps['LapTime'].min(),
                        'Median Lap': pd.Timedelta(seconds=lap_times.median()),
                        'Mean Lap': pd.Timedelta(seconds=lap_times.mean()),
                        'Std Dev (s)': lap_times.std(),
                        'Laps': len(valid_laps)
                    }
                    
                    # Add team if available
                    if 'Team' in driver_laps.columns:
                        stats['Team'] = driver_laps['Team'].iloc[0]
                    
                    pace_stats.append(stats)
            
            # Create DataFrame and show
            stats_df = pd.DataFrame(pace_stats)
            
            # Arrange columns with Driver first, then Team if present
            cols = ['Driver']
            if 'Team' in stats_df.columns:
                cols.append('Team')
            cols.extend([col for col in stats_df.columns if col not in cols])
            
            st.dataframe(stats_df[cols], use_container_width=True)
        else:
            st.warning("Please select at least one driver for pace analysis.")
    
    # Tab 2: Consistency & Degradation
    with tab2:
        st.markdown("### Driver Consistency and Tire Degradation")
        
        # Driver selector
        consistency_driver = st.selectbox(
            "Select driver",
            options=drivers,
            key="consistency_driver"
        )
        
        if consistency_driver:
            # Calculate consistency metric
            consistency, avg_pace = calculate_driver_consistency(laps_df, consistency_driver)
            
            if consistency is not None and avg_pace is not None:
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Consistency (CV%)", f"{consistency:.2f}%", 
                             delta="-" if consistency < 5 else None,
                             delta_color="normal" if consistency < 5 else "off")
                    st.markdown("*Lower values indicate better consistency*")
                
                with col2:
                    st.metric("Average Pace", f"{avg_pace:.3f}s")
            else:
                st.warning("Insufficient data to calculate consistency metrics.")
            
            # Tire degradation analysis
            st.markdown("#### Tire Degradation Analysis")
            
            slope, degr_pct, r_value, x, y = analyze_race_pace_degradation(laps_df, consistency_driver)
            
            if slope is not None and degr_pct is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Degradation", f"{degr_pct:.2f}% per 10 laps",
                             delta="-" if degr_pct < 0 else f"+{degr_pct:.2f}%",
                             delta_color="inverse")
                
                with col2:
                    st.metric("Time Loss", f"{slope:.3f}s per lap")
                
                with col3:
                    st.metric("Fit Quality", f"{r_value:.3f}")
                
                # Plot degradation trend
                if x is not None and y is not None:
                    fig = px.scatter(
                        x=x, y=y,
                        labels={'x': 'Lap Number', 'y': 'Lap Time (seconds)'},
                        title=f"Pace Degradation - {consistency_driver}"
                    )
                    
                    # Add trend line
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=slope * x + (y.mean() - slope * x.mean()),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data to calculate degradation metrics.")
    
    # Tab 3: Stint Comparison
    with tab3:
        st.markdown("### Driver Stint Performance")
        
        # Driver selector
        stint_driver = st.selectbox(
            "Select driver",
            options=drivers,
            key="stint_driver"
        )
        
        if stint_driver:
            # Calculate stint metrics
            stint_metrics = compare_stint_performance(laps_df, stint_driver)
            
            if not stint_metrics.empty:
                # Display stint table
                st.markdown("#### Stint Comparison")
                
                # Format table for display
                display_df = stint_metrics.copy()
                
                # Format time columns
                time_cols = ['AvgLapTime', 'BestLapTime']
                for col in time_cols:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}s")
                
                # Show compound if available
                if 'Compound' in display_df.columns:
                    # Map compound to emoji
                    compound_emoji = {
                        'SOFT': '🔴 SOFT',
                        'MEDIUM': '🟡 MEDIUM',
                        'HARD': '⚪ HARD',
                        'INTERMEDIATE': '🟢 INTER',
                        'WET': '🔵 WET',
                        'Unknown': '❓ Unknown'
                    }
                    display_df['Compound'] = display_df['Compound'].map(lambda x: compound_emoji.get(x, x))
                
                # Reorder columns
                cols = ['Stint', 'Compound', 'NumLaps', 'AvgLapTime', 'BestLapTime', 'LapTimeStd']
                display_cols = [col for col in cols if col in display_df.columns]
                display_cols.extend([col for col in display_df.columns if col not in cols])
                
                st.dataframe(display_df[display_cols], use_container_width=True)
                
                # Plot stint comparison
                if 'AvgLapTime' in stint_metrics.columns:
                    fig = px.bar(
                        stint_metrics,
                        x='Stint',
                        y='AvgLapTime',
                        color='Compound' if 'Compound' in stint_metrics.columns else None,
                        labels={'Stint': 'Stint Number', 'AvgLapTime': 'Average Lap Time (seconds)'},
                        title=f"Stint Performance Comparison - {stint_driver}",
                        text='NumLaps'
                    )
                    
                    fig.update_traces(texttemplate='%{text} laps', textposition='outside')
                    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Plot degradation comparison if available
                if 'DegradationPerLap' in stint_metrics.columns and not stint_metrics['DegradationPerLap'].isnull().all():
                    fig = px.bar(
                        stint_metrics,
                        x='Stint',
                        y='DegradationPerLap',
                        color='Compound' if 'Compound' in stint_metrics.columns else None,
                        labels={'Stint': 'Stint Number', 'DegradationPerLap': 'Degradation (seconds per lap)'},
                        title=f"Tire Degradation by Stint - {stint_driver}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("*Positive values indicate decreasing performance (higher lap times) with tire age*")
            else:
                st.warning("Insufficient data to compare stint performance.")
    
    # Tab 4: Performance Rating
    with tab4:
        st.markdown("### Driver Performance Rating")
        
        # Get race results if available
        results = None
        if session.name == 'Race' or session.name == 'Sprint':
            results = session.results
        
        # Calculate performance metrics
        driver_metrics = calculate_multivariate_driver_rating(laps_df, results)
        
        if not driver_metrics.empty and 'DriverRating' in driver_metrics.columns:
            # Sort by rating
            sorted_metrics = driver_metrics.sort_values('DriverRating', ascending=False)
            
            # Display ratings
            st.markdown("#### Overall Driver Ratings")
            
            # Create columns for metrics
            cols = st.columns(3)
            
            # Show top 3 drivers with ratings
            for i in range(min(3, len(sorted_metrics))):
                driver = sorted_metrics.iloc[i]
                driver_name = driver['Driver']
                rating = driver['DriverRating']
                
                team = "Unknown"
                if 'Team' in driver and pd.notna(driver['Team']):
                    team = driver['Team']
                
                with cols[i]:
                    st.markdown(f"#### #{i+1}: {driver_name}")
                    st.markdown(f"**Team:** {team}")
                    st.markdown(f"**Rating:** {rating:.1f}/10")
                    
                    # Add position if available
                    if 'Position' in driver and pd.notna(driver['Position']):
                        st.markdown(f"**Race Position:** {driver['Position']}")
            
            # Display full ratings table
            st.markdown("#### Complete Driver Ratings")
            
            # Format table for display
            display_df = sorted_metrics[['Driver', 'DriverRating']].copy()
            
            # Add team if available
            if 'Team' in sorted_metrics.columns:
                display_df['Team'] = sorted_metrics['Team']
            
            # Add race position if available
            if 'Position' in sorted_metrics.columns:
                display_df['Position'] = sorted_metrics['Position']
            
            # Rename columns
            display_df.columns = [col if col != 'DriverRating' else 'Rating (0-10)' for col in display_df.columns]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Show radar chart for selected drivers
            st.markdown("#### Performance Comparison")
            
            # Allow selection of drivers to compare
            radar_drivers = st.multiselect(
                "Select drivers to compare",
                options=driver_metrics['Driver'].tolist(),
                default=driver_metrics.sort_values('DriverRating', ascending=False).head(3)['Driver'].tolist(),
                key="radar_drivers"
            )
            
            if radar_drivers:
                radar_df = driver_metrics[driver_metrics['Driver'].isin(radar_drivers)]
                radar_fig = plot_driver_performance_radar(radar_df, radar_drivers)
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Select drivers to compare their performance metrics.")
        else:
            st.warning("Insufficient data to calculate performance ratings.")

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import datetime

# Import format_timedelta and display_card functions from app.py
from app import format_timedelta, display_card, show_loading_animation

# Mock functions to replace those from utils.data_loader
def get_available_seasons():
    """Mock function to return available seasons"""
    return [2018, 2019, 2020, 2021, 2022, 2023, 2024]

def get_available_events(season):
    """Mock function to return available events"""
    events = {
        2024: [
            {"EventName": "Bahrain Grand Prix", "RoundNumber": 1, "EventDate": datetime.datetime(2024, 3, 2)},
            {"EventName": "Saudi Arabian Grand Prix", "RoundNumber": 2, "EventDate": datetime.datetime(2024, 3, 9)},
            {"EventName": "Australian Grand Prix", "RoundNumber": 3, "EventDate": datetime.datetime(2024, 3, 24)},
            {"EventName": "Japanese Grand Prix", "RoundNumber": 4, "EventDate": datetime.datetime(2024, 4, 7)}
        ]
    }
    
    # Return events as a DataFrame
    if season in events:
        return pd.DataFrame(events[season])
    else:
        # Return events for 2024 as default
        return pd.DataFrame(events[2024])

def show_home():
    """Home page showing application overview and features with an ultra-modern UI"""
    # Animation for loading effect
    st.markdown('<div class="loading-animation"></div>', unsafe_allow_html=True)
    
    # Welcome header
    st.markdown("""
    <div class="animate-in">
        <h2>Command Center <span class="blue-glow">Dashboard</span></h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Telemetry visualization panel
    st.markdown("""
    <div class="f1-card animate-in">
        <div class="f1-card-header">
            <div class="f1-card-icon red-glow">📈</div>
            <div class="f1-card-title">TELEMETRY</div>
        </div>
        <div class="telemetry-container">
            <svg viewBox="0 0 800 100" xmlns="http://www.w3.org/2000/svg">
                <!-- Speed line (red) -->
                <path d="M0,50 C50,20 100,60 150,30 C200,10 250,70 300,50 C350,30 400,80 450,40 C500,60 550,20 600,50 C650,30 700,60 750,40 C800,50 850,20 900,50" 
                      fill="none" stroke="#E20600" stroke-width="3" />
                
                <!-- RPM line (blue) -->
                <path d="M0,70 C50,40 100,80 150,50 C200,30 250,90 300,70 C350,50 400,60 450,20 C500,40 550,60 600,30 C650,50 700,80 750,60 C800,30 850,40 900,70" 
                      fill="none" stroke="#00C3FF" stroke-width="3" />
                
                <!-- Gear line (green) -->
                <path d="M0,60 C50,60 100,40 150,40 C200,40 250,60 300,60 C350,80 400,40 450,60 C500,20 550,40 600,40 C650,60 700,40 750,80 C800,60 850,60 900,60" 
                      fill="none" stroke="#21FC0D" stroke-width="3" />
            </svg>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; color: white;">
                <div style="color: #E20600;">SPEED</div>
                <div style="color: #00C3FF;">RPM</div>
                <div style="color: #21FC0D;">GEAR</div>
                <div>THROTTLE</div>
                <div style="color: #21FC0D;">OVERTAKE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main dashboard grid
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Race performance tracker with miniature circuit
        st.markdown("""
        <div class="f1-card animate-in">
            <div class="f1-card-header">
                <div class="f1-card-icon blue-glow">🏁</div>
                <div class="f1-card-title">RACE PERFORMANCE TRACKER</div>
            </div>
            <div class="circuit-map">
                <svg class="circuit-path" viewBox="0 0 100 60" xmlns="http://www.w3.org/2000/svg">
                    <!-- Circuit outline -->
                    <path d="M10,30 C15,10 30,5 50,5 C70,5 85,10 90,30 C85,50 70,55 50,55 C30,55 15,50 10,30 Z" 
                          fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="2" />
                    
                    <!-- Circuit sectors -->
                    <path d="M10,30 C15,10 30,5 50,5" fill="none" stroke="#E20600" stroke-width="2" />
                    <path d="M50,5 C70,5 85,10 90,30" fill="none" stroke="#21FC0D" stroke-width="2" />
                    <path d="M90,30 C85,50 70,55 50,55 C30,55 15,50 10,30" fill="none" stroke="#00C3FF" stroke-width="2" />
                    
                    <!-- Car indicators -->
                    <circle cx="45" cy="5" r="2" fill="#E20600" />
                    <circle cx="75" cy="10" r="2" fill="#00C3FF" />
                    <circle cx="90" cy="35" r="2" fill="#FFFFFF" />
                </svg>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">SECTOR 1</div>
                    <div style="font-size: 16px; color: #E20600; font-weight: bold;">31.254s</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">SECTOR 2</div>
                    <div style="font-size: 16px; color: #21FC0D; font-weight: bold;">28.765s</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">SECTOR 3</div>
                    <div style="font-size: 16px; color: #00C3FF; font-weight: bold;">33.482s</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">LAP TIME</div>
                    <div style="font-size: 16px; color: white; font-weight: bold;">1:33.501</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tire degradation visualization
        st.markdown("""
        <div class="f1-card animate-in">
            <div class="f1-card-header">
                <div class="f1-card-icon red-glow">🔄</div>
                <div class="f1-card-title">TIRE DEGRADATION</div>
            </div>
            <div style="display: flex; justify-content: space-around; margin: 15px 0;">
                <div class="tire-container">
                    <div class="tire soft"></div>
                    <div class="tire-label">SOFT</div>
                </div>
                <div class="tire-container">
                    <div class="tire medium"></div>
                    <div class="tire-label">MEDIUM</div>
                </div>
                <div class="tire-container">
                    <div class="tire hard"></div>
                    <div class="tire-label">HARD</div>
                </div>
                <div class="tire-container">
                    <div class="tire intermediate"></div>
                    <div class="tire-label">INTER</div>
                </div>
            </div>
            <div>
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>TIRE STRATEGY</span>
                        <span>LAPS 22-35</span>
                    </div>
                    <div style="height: 8px; border-radius: 4px; background: rgba(0,0,0,0.3); position: relative;">
                        <div style="position: absolute; top: 0; left: 0; width: 60%; height: 100%; background: linear-gradient(90deg, #ffcc00, #E20600); border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Driver comparison panel
        st.markdown("""
        <div class="f1-card animate-in">
            <div class="f1-card-header">
                <div class="f1-card-icon blue-glow">👤</div>
                <div class="f1-card-title">DRIVER COMPARISON</div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">DELTA TIME</div>
                    <div class="stat-value red">0.342s</div>
                </div>
                <div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">REACTION</div>
                    <div class="stat-value blue">0.249s</div>
                </div>
            </div>
            <div style="display: flex; margin-bottom: 10px; gap: 10px;">
                <div style="flex: 1; background: rgba(226, 6, 0, 0.2); border-radius: 5px; text-align: center; padding: 5px;">
                    22 LAPS
                </div>
                <div style="flex: 1; background: rgba(0, 195, 255, 0.2); border-radius: 5px; text-align: center; padding: 5px;">
                    20 LAPS
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;">
                <div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">PIT DELTA</div>
                    <div class="stat-value blue">2.3%</div>
                </div>
                <div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7);">TIRE WEAR</div>
                    <div style="display: flex; align-items: center; justify-content: center; width: 30px; height: 30px; border-radius: 50%; background: radial-gradient(circle, rgba(255, 0, 0, 0.8), rgba(100, 0, 0, 0.8)); margin: 0 auto;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Smart AI Insights panel
        st.markdown("""
        <div class="f1-card animate-in">
            <div class="f1-card-header">
                <div class="f1-card-icon green-glow">💡</div>
                <div class="f1-card-title">SMART AI INSIGHTS</div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background: rgba(33, 252, 13, 0.2); display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                    <div style="width: 15px; height: 15px; border-radius: 50%; background: #21FC0D;"></div>
                </div>
                <div>
                    <div style="font-size: 14px; color: white;">Optimal pit window</div>
                    <div style="font-size: 16px; color: #21FC0D; font-weight: bold;">LAPS 30-33</div>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="font-size: 12px; color: rgba(255,255,255,0.7); margin-bottom: 5px;">TIRE STRATEGY PREDICTION</div>
                <div style="height: 8px; border-radius: 4px; background: rgba(0,0,0,0.3); position: relative;">
                    <div style="position: absolute; top: 0; left: 0; width: 70%; height: 100%; background: linear-gradient(90deg, #21FC0D, #00C3FF); border-radius: 4px;"></div>
                    <div style="position: absolute; top: -5px; left: 70%; width: 3px; height: 18px; background: #E20600;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Live timing panel
    st.markdown("""
    <div class="f1-card animate-in">
        <div class="f1-card-header">
            <div class="f1-card-icon green-glow">⏱️</div>
            <div class="f1-card-title">LIVE TIMING</div>
        </div>
        <div style="text-align: center; font-size: 42px; font-weight: bold; margin: 15px 0;">
            1:23.274
        </div>
        <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 15px;">
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span>S1</span>
                    <span style="color: #E20600; font-weight: bold;">31.254s</span>
                </div>
                <div style="height: 4px; border-radius: 2px; background: rgba(0,0,0,0.3);">
                    <div style="height: 100%; width: 75%; background: #E20600; border-radius: 2px;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span>S2</span>
                    <span style="color: #21FC0D; font-weight: bold;">28.765s</span>
                </div>
                <div style="height: 4px; border-radius: 2px; background: rgba(0,0,0,0.3);">
                    <div style="height: 100%; width: 85%; background: #21FC0D; border-radius: 2px;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span>S3</span>
                    <span style="color: #00C3FF; font-weight: bold;">23.255s</span>
                </div>
                <div style="height: 4px; border-radius: 2px; background: rgba(0,0,0,0.3);">
                    <div style="height: 100%; width: 65%; background: #00C3FF; border-radius: 2px;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load default data for the latest season
    latest_season = get_available_seasons()[-1]
    st.session_state.selected_season = latest_season if 'selected_season' not in st.session_state else st.session_state.selected_season
    
    # Show quick session selector with improved styling
    st.markdown("""
    <div style="animation: fadeIn 1.2s ease-out; animation-delay: 0.5s; opacity: 0;">
        <h2>Quick Session Selection</h2>
        <p>Select a season, event, and session type to begin your analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session selector with gradient background
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(19, 30, 47, 0.9), rgba(226, 6, 0, 0.1)); 
                border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        season = st.selectbox("Season", 
                             options=get_available_seasons(), 
                             index=get_available_seasons().index(st.session_state.selected_season),
                             key="home_season_selector")
        
        # Update session state
        st.session_state.selected_season = season
    
    with col2:
        # Get events for the selected season
        events_df = get_available_events(season)
        
        if not events_df.empty:
            event_options = events_df['EventName'].tolist()
            event_index = 0  # Default to the first event
            
            # Find the last completed event
            for i, completed in enumerate(events_df['EventDate'] < pd.Timestamp.now()):
                if completed:
                    event_index = i
            
            selected_event = st.selectbox("Event", 
                                        options=event_options,
                                        index=min(event_index, len(event_options)-1),
                                        key="home_event_selector")
            
            event_round = events_df[events_df['EventName'] == selected_event]['RoundNumber'].iloc[0]
            st.session_state.selected_event_round = event_round
            st.session_state.selected_event_name = selected_event
        else:
            st.warning(f"No events available for season {season}")
            selected_event = None
            st.session_state.selected_event_round = None
            st.session_state.selected_event_name = None
    
    with col3:
        if selected_event is not None:
            session_types = ['R', 'Q', 'S', 'FP3', 'FP2', 'FP1']
            session_labels = ['Race', 'Qualifying', 'Sprint', 'Practice 3', 'Practice 2', 'Practice 1']
            
            session_type = st.selectbox("Session", 
                                     options=session_types,
                                     format_func=lambda x: session_labels[session_types.index(x)],
                                     key="home_session_selector")
            
            st.session_state.selected_session_type = session_type
        else:
            session_type = None
            st.session_state.selected_session_type = None
    
    # Load data button
    if selected_event is not None and session_type is not None:
        if st.button("Load Selected Session", key="home_load_button"):
            with st.spinner(f"Loading {selected_event} {session_type} data..."):
                session = load_session(season, event_round, session_type)
                
                if session is not None:
                    # Store session in session state
                    st.session_state.current_session = session
                    st.session_state.current_laps = session.laps
                    
                    # Display success message
                    st.success(f"Successfully loaded {selected_event} {session_type} data!")
                    
                    # Show session info
                    show_session_info()
                else:
                    st.error("Failed to load session data. Please try another session.")
    
    # Close the gradient div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display quick stats if a session is loaded
    if 'current_session' in st.session_state and st.session_state.current_session is not None:
        show_session_info()
    
    # Display latest F1 news if no session is loaded
    if 'current_session' not in st.session_state or st.session_state.current_session is None:
        st.markdown("### Using This Application")
        
        st.markdown("""
        To get started with your F1 data analysis:
        
        1. **Select a season, event, and session** using the selectors above
        2. **Click "Load Selected Session"** to fetch the data
        3. **Navigate to specific analysis sections** using the sidebar
        
        You can also directly navigate to any analysis page, where you'll be able to select and load data as needed.
        
        ### About the Data
        
        This application uses the FastF1 library, which provides access to official Formula 1 timing data. The data includes:
        
        - Detailed lap timing information
        - Car telemetry data (where available)
        - Driver and team information
        - Race results and session classifications
        
        FastF1 data is available from the 2018 season to the present.
        """)
    
    # Footer with credits
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
    <p>Data powered by the <a href="https://docs.fastf1.dev/" target="_blank">FastF1</a> library</p>
    </div>
    """, unsafe_allow_html=True)

def show_session_info():
    """Display information about the currently loaded session"""
    if 'current_session' not in st.session_state or st.session_state.current_session is None:
        return
    
    session = st.session_state.current_session
    laps = st.session_state.current_laps
    
    # Create an expander for session details
    with st.expander("Session Information", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Event:** {session.event['EventName']} {session.event.year}")
            st.markdown(f"**Circuit:** {session.event['OfficialEventName']}")
            st.markdown(f"**Session Type:** {session.name}")
            
            if session.date:
                st.markdown(f"**Date:** {session.date.strftime('%Y-%m-%d')}")
        
        with col2:
            st.markdown(f"**Laps Completed:** {len(laps)}")
            st.markdown(f"**Weather Condition:** {session.weather_data['AirTemp'].mean():.1f}°C, Wind: {session.weather_data['WindSpeed'].mean():.1f} km/h")
            
            # Number of drivers
            num_drivers = len(laps['Driver'].unique())
            st.markdown(f"**Drivers:** {num_drivers}")
    
    # Show quick visualization of results
    if session.name == 'Race' or session.name == 'Sprint':
        results = session.results
        
        if results is not None and not results.empty:
            # Show top 10 finishers
            st.markdown("### Race Results (Top 10)")
            
            top10 = results.head(10)[['Position', 'Abbreviation', 'TeamName', 'GridPosition', 'Time']]
            top10.columns = ['Position', 'Driver', 'Team', 'Grid', 'Race Time']
            
            # Add position change
            top10['Gain/Loss'] = top10['Grid'].astype(int) - top10['Position'].astype(int)
            
            # Format time
            if 'Race Time' in top10.columns:
                top10['Race Time'] = top10['Race Time'].astype(str).apply(lambda x: x.split(' ')[-1] if ' ' in x else x)
            
            st.dataframe(top10, use_container_width=True)
    
    elif session.name == 'Qualifying' or session.name == 'Sprint Qualifying':
        results = session.results
        
        if results is not None and not results.empty:
            # Show qualifying results
            st.markdown("### Qualifying Results")
            
            quali = results[['Position', 'Abbreviation', 'TeamName', 'Q1', 'Q2', 'Q3']]
            quali.columns = ['Position', 'Driver', 'Team', 'Q1', 'Q2', 'Q3']
            
            st.dataframe(quali, use_container_width=True)
    
    # Show fastest lap
    if not laps.empty and 'LapTime' in laps.columns and not laps['LapTime'].isnull().all():
        st.markdown("### Fastest Lap")
        
        fastest_lap = laps.loc[laps['LapTime'].idxmin()]
        driver = fastest_lap['Driver']
        team = fastest_lap['Team'] if 'Team' in fastest_lap.index else "Unknown"
        lap_time = fastest_lap['LapTime']
        lap_number = fastest_lap['LapNumber']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Driver", driver)
        
        with col2:
            st.metric("Team", team)
        
        with col3:
            # Convert Timedelta to string for st.metric using our formatter
            st.metric("Lap Time", format_timedelta(lap_time))
        
        # Add lap number and speed if available
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Lap Number", lap_number)
        
        with col2:
            if 'SpeedFL' in fastest_lap.index:
                st.metric("Top Speed", f"{fastest_lap['SpeedFL']:.1f} km/h")

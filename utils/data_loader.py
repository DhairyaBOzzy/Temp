import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import time

@st.cache_data(ttl=3600)
def get_available_seasons():
    """Get list of available seasons from 2018 to current year"""
    current_year = datetime.now().year
    return list(range(2018, current_year + 1))

@st.cache_data(ttl=3600)
def get_available_events(season):
    """Get all events for a specific season"""
    try:
        schedule = fastf1.get_event_schedule(season)
        return schedule
    except Exception as e:
        st.error(f"Error fetching events for season {season}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_session(season, event_round, session_type='R'):
    """
    Load a specific session with caching
    
    Parameters:
    -----------
    season : int
        The F1 season year
    event_round : int
        The round number of the event
    session_type : str
        The session type: 'FP1', 'FP2', 'FP3', 'Q', 'S', or 'R' (race)
        
    Returns:
    --------
    fastf1.core.Session
        The loaded session object
    """
    try:
        session = fastf1.get_session(season, event_round, session_type)
        session.load()
        return session
    except Exception as e:
        st.error(f"Error loading session: {e}")
        return None

@st.cache_data(ttl=3600)
def get_all_drivers(season):
    """Get all drivers for a specific season"""
    try:
        schedule = fastf1.get_event_schedule(season)
        if schedule.empty:
            return []
        
        # Get the last completed race
        completed_races = schedule[schedule['EventDate'] < datetime.now()]
        if completed_races.empty:
            # If no race has been completed yet, get the first event
            event = schedule.iloc[0]
        else:
            event = completed_races.iloc[-1]
        
        session = fastf1.get_session(season, event['RoundNumber'], 'R')
        session.load()
        driver_data = session.get_driver_info()
        
        driver_list = []
        for driver_number, data in driver_data.items():
            driver_dict = {
                'number': driver_number,
                'code': data['Abbreviation'],
                'firstname': data['FirstName'],
                'lastname': data['LastName'],
                'team': data['TeamName']
            }
            driver_list.append(driver_dict)
        
        return sorted(driver_list, key=lambda x: x['lastname'])
    except Exception as e:
        st.error(f"Error fetching drivers for season {season}: {e}")
        return []

@st.cache_data(ttl=3600)
def get_all_teams(season):
    """Get all teams for a specific season"""
    try:
        drivers = get_all_drivers(season)
        teams = sorted(list(set([driver['team'] for driver in drivers])))
        return teams
    except Exception as e:
        st.error(f"Error fetching teams for season {season}: {e}")
        return []

@st.cache_data(ttl=3600)
def load_laps_data(session):
    """Load and process lap data for a session"""
    if session is None:
        return None
    
    try:
        laps = session.laps
        return laps
    except Exception as e:
        st.error(f"Error loading lap data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_telemetry_data(lap):
    """Load telemetry data for a specific lap"""
    if lap is None:
        return None
    
    try:
        telemetry = lap.get_telemetry()
        return telemetry
    except Exception as e:
        st.error(f"Error loading telemetry data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_race_results(session):
    """Load race results data for a race session"""
    if session is None or session.session_type != 'Race':
        return None
    
    try:
        results = session.results
        return results
    except Exception as e:
        st.error(f"Error loading race results: {e}")
        return None

@st.cache_data(ttl=3600)
def load_multiple_seasons_data(seasons, session_type='R'):
    """Load and concatenate data from multiple seasons for comparison"""
    all_data = []
    progress_bar = st.progress(0)
    
    for i, season in enumerate(seasons):
        st.text(f"Loading data for {season}...")
        
        schedule = get_available_events(season)
        if schedule.empty:
            continue
            
        # Get completed events
        completed_events = schedule[schedule['EventDate'] < datetime.now()]
        if completed_events.empty:
            continue
            
        for _, event in completed_events.iterrows():
            try:
                session = load_session(season, event['RoundNumber'], session_type)
                if session:
                    laps = load_laps_data(session)
                    if laps is not None and not laps.empty:
                        laps['Season'] = season
                        laps['Event'] = event['EventName']
                        laps['Circuit'] = event['OfficialEventName']
                        all_data.append(laps)
            except Exception as e:
                pass  # Skip failed loads silently
                
        # Update progress
        progress_bar.progress((i + 1) / len(seasons))
        
    progress_bar.empty()
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        st.warning("No data could be loaded for the selected seasons.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_driver_championship_standings(season):
    """Get driver championship standings for a season"""
    try:
        standings = fastf1.get_driver_standings(season)
        return standings
    except Exception as e:
        st.error(f"Error fetching driver standings for season {season}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_constructor_championship_standings(season):
    """Get constructor championship standings for a season"""
    try:
        # FastF1 doesn't have a direct method for constructor standings
        # We'll use the events and then get standings for the last event
        schedule = fastf1.get_event_schedule(season)
        if schedule.empty:
            return pd.DataFrame()
            
        # Get the last completed event
        current_date = datetime.now()
        completed_events = schedule[schedule['EventDate'] < current_date]
        
        if completed_events.empty:
            # If no completed events, use the first event
            event = schedule.iloc[0]
        else:
            # Use the last completed event
            event = completed_events.iloc[-1]
            
        # Load the session to get access to constructor standings
        session = fastf1.get_session(season, event['EventName'], 'R')
        session.load()
        
        # Create a constructor standings DataFrame
        teams = {}
        for driver_data in session.results.itertuples():
            team_name = driver_data.TeamName
            points = driver_data.Points if hasattr(driver_data, 'Points') else 0
            
            if team_name in teams:
                teams[team_name] += points
            else:
                teams[team_name] = points
                
        # Convert to DataFrame and sort by points
        standings_data = [{"TeamName": team, "Points": points} for team, points in teams.items()]
        standings = pd.DataFrame(standings_data).sort_values('Points', ascending=False)
        standings['Position'] = range(1, len(standings) + 1)
        
        return standings[['Position', 'TeamName', 'Points']]
    except Exception as e:
        st.error(f"Error fetching constructor standings for season {season}: {e}")
        return pd.DataFrame()

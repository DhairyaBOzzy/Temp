import streamlit as st
import pandas as pd
import re
from utils.notifications import send_email_notification, create_race_alert_message
from utils.data_loader import get_all_drivers, get_all_teams
from app import display_card

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def show_notifications():
    """Display notifications settings page"""
    st.markdown("""
    <div style="animation: fadeIn 0.8s ease-out;">
        <h2>Race Notifications & Alerts</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if Email notification is configured
    if 'email_configured' not in st.session_state:
        import os
        has_email = all(key in os.environ for key in ["EMAIL_ADDRESS", "EMAIL_PASSWORD"])
        st.session_state.email_configured = has_email
    
    # Show email status
    if st.session_state.email_configured:
        st.success("✅ Email notifications are available! Configure your alerts below.")
    else:
        st.warning("⚠️ Email notifications are not available. Please configure email credentials.")
        
        with st.expander("How to setup Email notifications"):
            st.markdown("""
            ### Setting Up Email Notifications
            
            To enable email notifications for race alerts, you need to:
            
            1. Have an email account that allows app passwords or less secure apps
            2. For Gmail, create an App Password in your Google Account security settings
            3. Configure the following environment variables:
               - `EMAIL_ADDRESS` - Your email address
               - `EMAIL_PASSWORD` - Your email password or app password
               - `SMTP_SERVER` - SMTP server address (default: smtp.gmail.com)
               - `SMTP_PORT` - SMTP server port (default: 587)
               
            Once configured, you can receive alerts for:
            - Fastest laps
            - Position changes for selected drivers
            - Pit stop alerts
            - Race start/end notifications
            """)
    
    # Feature cards
    st.markdown("### Available Alert Types")
    
    col1, col2 = st.columns(2)
    
    with col1:
        display_card(
            "Race Start & End Alerts", 
            "Get notified when races start and end for selected events. Receive a summary of results after the race.",
            "🚦"
        )
        
        display_card(
            "Driver Position Alerts", 
            "Track position changes for your favorite drivers during the race.",
            "🏆"
        )
    
    with col2:
        display_card(
            "Fastest Lap Notifications", 
            "Be alerted when a new fastest lap is set during the race.",
            "⏱️"
        )
        
        display_card(
            "Pit Stop Strategy Alerts", 
            "Get notifications when drivers make pit stops, with details on tire changes and stop duration.",
            "🔧"
        )
    
    # Configuration section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="animation: fadeIn 1s ease-out; animation-delay: 0.3s; opacity: 0;">
        <h2>Configure Your Alerts</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Gradient background for configuration
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(19, 30, 47, 0.9), rgba(226, 6, 0, 0.1)); 
                border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
    """, unsafe_allow_html=True)
    
    # Email input
    email_address = st.text_input(
        "Your Email Address",
        value=st.session_state.get('email_address', ''),
        key="notification_email"
    )
    
    # Save to session state
    if email_address:
        st.session_state.email_address = email_address
    
    # Alert settings
    with st.expander("Alert Settings", expanded=True):
        # Driver selection
        if 'current_session' in st.session_state and st.session_state.current_session is not None:
            session = st.session_state.current_session
            season = session.event.year
            
            # Get drivers from current session
            drivers = st.session_state.current_laps['Driver'].unique().tolist()
            driver_names = [driver for driver in drivers]
            
            # Get teams from current session
            teams = st.session_state.current_laps['Team'].unique().tolist() if 'Team' in st.session_state.current_laps.columns else []
            
            st.write("#### Select Drivers to Monitor")
            selected_drivers = st.multiselect(
                "Drivers",
                options=driver_names,
                default=driver_names[:3] if driver_names else [],
                key="alert_drivers"
            )
            
            st.write("#### Select Teams to Monitor")
            selected_teams = st.multiselect(
                "Teams",
                options=teams,
                default=teams[:2] if teams else [],
                key="alert_teams"
            )
        else:
            st.info("Please load a session first to configure driver-specific alerts.")
            selected_drivers = []
            selected_teams = []
        
        # Alert types
        st.write("#### Alert Types")
        race_start_end = st.checkbox("Race Start & End", value=True, key="alert_race_start_end")
        fastest_laps = st.checkbox("Fastest Laps", value=True, key="alert_fastest_laps")
        position_changes = st.checkbox("Position Changes", value=True, key="alert_position_changes")
        pit_stops = st.checkbox("Pit Stops", value=True, key="alert_pit_stops")
    
    # Test alert button
    if st.button("Send Test Alert", disabled=not st.session_state.email_configured):
        if not email_address:
            st.error("Please enter an email address to receive alerts.")
        elif not validate_email(email_address):
            st.error("Please enter a valid email address (e.g., user@example.com)")
        else:
            # Create a test message
            subject = "F1 Analytics Platform - Test Alert"
            message = """🏎️ F1 ANALYTICS TEST ALERT

This is a test notification from your F1 Analytics Platform.
If you're receiving this, your alerts are configured correctly!

--
Sent by F1 Analytics Platform
"""
            
            # Send test message
            if send_email_notification(email_address, subject, message):
                st.success("Test alert sent successfully! Check your email inbox for the message.")
            else:
                st.error("Failed to send test alert. Please check your email configuration.")
    
    # Close gradient div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Instruction section
    if not st.session_state.email_configured:
        st.info("Note: Email notifications require proper credentials. The feature will be fully enabled once credentials are provided.")
    
    # Example alerts
    st.markdown("### Example Alert Messages")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        st.code("""
🏎️ F1 ALERT: Fastest Lap

Driver: HAM
Team: Mercedes
Event: British Grand Prix

Lewis Hamilton has set a new fastest lap of 1:27.369 on lap 44 of the race!
        """)
        
        st.code("""
🏎️ F1 ALERT: Race Start

Event: Monaco Grand Prix
Time: 14:00 CET

The Monaco Grand Prix is now starting! 20 drivers are on the grid with VER on pole.
        """)
    
    with example_col2:
        st.code("""
🏎️ F1 ALERT: Position Change

Driver: VER
Team: Red Bull Racing
Event: Belgian Grand Prix

Max Verstappen has moved up to P1 from P2 on lap 32!
        """)
        
        st.code("""
🏎️ F1 ALERT: Pit Stop

Driver: LEC
Team: Ferrari
Event: Italian Grand Prix

Charles Leclerc has completed a pit stop on lap 24:
- Tire change: Soft → Medium
- Stop time: 2.4 seconds
- Position: P2 (no change)
        """)
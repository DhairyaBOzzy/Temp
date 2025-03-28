import streamlit as st
import sys
import os
import pandas as pd
import time
from datetime import timedelta
import base64

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock FastF1 for development without the library
class MockFastF1:
    class Cache:
        @staticmethod
        def enable_cache(path):
            pass

# Create mock fastf1 module
sys.modules['fastf1'] = MockFastF1

# Must be the first Streamlit command
st.set_page_config(
    page_title="F1 Analytics Platform",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to format timedeltas for display
def format_timedelta(td):
    if pd.isna(td):
        return "N/A"
    
    if isinstance(td, timedelta):
        # Convert to total seconds
        total_seconds = td.total_seconds()
        
        # Format as minutes:seconds.milliseconds
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        
        return f"{minutes:01d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return str(td)

# Create loading animation
def show_loading_animation():
    with st.spinner("Loading data..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        progress_bar.empty()

# Function to display animated cards
def display_card(title, content, icon="🏎️"):
    st.markdown(f"""
    <div class="card">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# Load custom CSS
try:
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    # Create default styling if file doesn't exist
    st.markdown("""
    <style>
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(20px);}
        100% {opacity: 1; transform: translateY(0);}
    }

    .card {
        animation: fadeIn 0.5s ease-out;
        background: rgba(19, 30, 47, 0.8);
        border-left: 4px solid #E20600;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
    }

    .card-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #FFFFFF;
    }

    .card-content {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
    }

    .highlight {
        background: linear-gradient(90deg, #E20600, #ff8080);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-weight: bold;
    }

    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }

    .team-card {
        background: rgba(19, 30, 47, 0.5);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .team-card:hover {
        background: rgba(226, 6, 0, 0.2);
        transform: scale(1.05);
    }

    .metric-row {
        display: flex;
        gap: 15px;
        margin-bottom: 15px;
    }

    .metric-box {
        flex: 1;
        background: rgba(19, 30, 47, 0.7);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-bottom: 3px solid #E20600;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #E20600;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }

    /* Pulse animation for live data indicators */
    @keyframes pulse {
        0% {transform: scale(1);}
        50% {transform: scale(1.05);}
        100% {transform: scale(1);}
    }

    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #0CCE6B;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    .loading-animation {
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #E20600, transparent);
        background-size: 200% 100%;
        animation: loading 1.5s infinite linear;
    }

    @keyframes loading {
        0% {background-position: 200% 0;}
        100% {background-position: 0 0;}
    }
    </style>
    """, unsafe_allow_html=True)

# Application header with F1 logo
st.markdown("""
<div class="app-header">
    <svg class="f1-logo" viewBox="0 0 100 30" xmlns="http://www.w3.org/2000/svg">
        <path fill="#E20600" d="M0 0h30.5v30.5H0zM67.07 0H100v30.5H67.07z"/>
        <path fill="#E20600" d="M11.24 6.55V12H5.8V6.55h5.44zM59.67 5.77V12h-5.44V5.77h5.44zM36.34 5.77v12.78h-6.22v-2.33h-4.65v7h-6.23v-7H7.78v7H1.55V5.77h34.78z"/>
        <path fill="#E20600" d="M51.3 5.77v12.78h-8.75V12.4a12.92 12.92 0 0 1-7.76 0V5.77H51.3zM98.06 5.77v12.78h-7.76v-7.76h-7.77v7.76h-7.76V5.77h23.3z"/>
        <path fill="#E20600" d="M74.87 24.73a5.77 5.77 0 0 0 5.78-5.78h7.76a13.54 13.54 0 0 1-13.54 13.54A13.54 13.54 0 0 1 61.33 18.95V12.4a6.77 6.77 0 0 0 6.78 6.78h6.77z"/>
    </svg>
    <h1>Formula 1 Analytics Platform</h1>
</div>

<div class="animate-in" style="margin-bottom: 30px;">
    <p class="subtitle">
        This ultra-modern platform provides real-time <span class="red-glow">analytics</span>, 
        <span class="blue-glow">telemetry visualization</span>, and <span class="green-glow">machine learning insights</span> 
        for Formula 1 data. Explore driver performance, team strategies, and predictive models through 
        interactive visualizations and statistical analysis.
    </p>
    
    <div class="f1-quote">
        Formula 1 is all about data now. The more you understand your data, the more you understand what to do with the car.
        <span class="f1-quote-author">— Lewis Hamilton</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main navigation - Using interactive icons for sidebar
st.sidebar.markdown("""
<style>
.sidebar-nav {
    padding: 10px;
}
.sidebar-nav-item {
    display: flex;
    align-items: center;
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 8px;
    transition: all 0.3s ease;
    cursor: pointer;
    color: white;
}
.sidebar-nav-item:hover {
    background: rgba(226, 6, 0, 0.2);
    transform: translateX(5px);
}
.sidebar-nav-item.active {
    background: rgba(226, 6, 0, 0.3);
    border-left: 3px solid #E20600;
}
.sidebar-nav-icon {
    margin-right: 10px;
    font-size: 1.2rem;
    width: 25px;
    text-align: center;
}
</style>
<div class="sidebar-nav">
    <h2>Navigation</h2>
</div>
""", unsafe_allow_html=True)

# Define navigation options with icons
nav_options = {
    "Home": "🏠",
    "Driver Analysis": "👨‍💼",
    "Team Analysis": "🏎️",
    "Race Strategy": "⚙️",
    "Machine Learning Insights": "🧠",
    "Notifications": "🔔"
}

# Create styled sidebar items
selection = st.sidebar.radio(
    "Go to",
    list(nav_options.keys()),
    format_func=lambda x: f"{nav_options[x]} {x}",
    label_visibility="collapsed"
)

# Import all page modules
from pages.home import show_home
from pages.driver_analysis import show_driver_analysis
from pages.team_analysis import show_team_analysis
from pages.race_strategy import show_race_strategy
from pages.machine_learning import show_ml_insights
from pages.notifications import show_notifications

# Display the selected page
if selection == "Home":
    show_home()
elif selection == "Driver Analysis":
    show_driver_analysis()
elif selection == "Team Analysis":
    show_team_analysis()
elif selection == "Race Strategy":
    show_race_strategy()
elif selection == "Machine Learning Insights":
    show_ml_insights()
elif selection == "Notifications":
    show_notifications()

# Footer with information
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This F1 Analytics Platform uses the FastF1 library to provide insights into Formula 1 data.
    Data is sourced directly from the official F1 timing service.
    """
)
st.sidebar.markdown("### Data Source")
st.sidebar.markdown("[FastF1 Documentation](https://docs.fastf1.dev/)")

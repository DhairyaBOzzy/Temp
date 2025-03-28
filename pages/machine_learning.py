import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import (
    get_available_seasons, get_available_events, load_session, 
    get_all_drivers, load_multiple_seasons_data
)
from utils.ml_models import (
    create_driver_clusters, predict_lap_time, detect_anomalies,
    create_race_strategy_model
)
from utils.visualizations import (
    plot_driver_clusters, plot_feature_importance, plot_lap_time_prediction
)

def show_ml_insights():
    """Display machine learning insights page"""
    st.title("Machine Learning Insights")
    
    # Tabs for different ML analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Driver Clustering", 
        "Lap Time Prediction", 
        "Anomaly Detection",
        "Strategy Impact"
    ])
    
    # Tab 1: Driver Clustering
    with tab1:
        st.markdown("### Driver Performance Clustering")
        st.markdown("""
        This analysis uses unsupervised machine learning to identify groups of drivers with similar performance characteristics.
        The algorithm clusters drivers based on multiple performance metrics including lap times, consistency, and other factors.
        """)
        
        # Session selection for clustering
        st.sidebar.markdown("### Driver Clustering")
        
        # Season selector
        seasons = get_available_seasons()
        cluster_season = st.sidebar.selectbox("Season", options=seasons, index=len(seasons)-1, key="cluster_season")
        
        # Event selector
        events_df = get_available_events(cluster_season)
        
        if not events_df.empty:
            event_options = events_df['EventName'].tolist()
            
            # Find the last completed event
            event_index = 0
            for i, completed in enumerate(events_df['EventDate'] < pd.Timestamp.now()):
                if completed:
                    event_index = i
            
            cluster_event = st.sidebar.selectbox("Event", 
                                               options=event_options,
                                               index=min(event_index, len(event_options)-1),
                                               key="cluster_event")
            
            event_round = events_df[events_df['EventName'] == cluster_event]['RoundNumber'].iloc[0]
        else:
            st.sidebar.warning(f"No events available for season {cluster_season}")
            cluster_event = None
            event_round = None
        
        # Session type selector
        if cluster_event is not None:
            session_types = ['R', 'Q', 'S', 'FP3', 'FP2', 'FP1']
            session_labels = ['Race', 'Qualifying', 'Sprint', 'Practice 3', 'Practice 2', 'Practice 1']
            
            cluster_session_type = st.sidebar.selectbox("Session", 
                                                     options=session_types,
                                                     format_func=lambda x: session_labels[session_types.index(x)],
                                                     key="cluster_session_type")
        else:
            cluster_session_type = None
        
        # Load data button
        if cluster_event is not None and cluster_session_type is not None:
            if st.sidebar.button("Load Data for Clustering", key="cluster_load_button"):
                with st.spinner(f"Loading {cluster_event} {cluster_session_type} data..."):
                    session = load_session(cluster_season, event_round, cluster_session_type)
                    
                    if session is not None:
                        # Store session in session state
                        st.session_state.cluster_session = session
                        st.session_state.cluster_laps = session.laps
                        
                        # Store session info for reference
                        st.session_state.cluster_session_info = {
                            'Season': cluster_season,
                            'EventName': cluster_event,
                            'RoundNumber': event_round,
                            'SessionType': cluster_session_type,
                            'Year': session.event.year
                        }
                        
                        # Perform clustering
                        with st.spinner("Performing driver clustering..."):
                            cluster_df, scaler, kmeans, pca = create_driver_clusters(_laps_df=session.laps)
                            st.session_state.cluster_results = cluster_df
                            st.session_state.cluster_models = {
                                'scaler': scaler,
                                'kmeans': kmeans,
                                'pca': pca
                            }
                        
                        # Display success message
                        st.sidebar.success(f"Successfully loaded and clustered data!")
                    else:
                        st.sidebar.error("Failed to load session data. Please try another session.")
        
        # Display clustering results if available
        if 'cluster_results' in st.session_state and st.session_state.cluster_results is not None:
            cluster_df = st.session_state.cluster_results
            session_info = st.session_state.cluster_session_info
            
            st.markdown(f"#### Driver Clusters for {session_info['EventName']} {session_info['Year']}")
            
            # Show the cluster visualization
            fig = plot_driver_clusters(cluster_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show cluster details
            st.markdown("#### Cluster Details")
            
            clusters = sorted(cluster_df['Cluster'].unique())
            
            for cluster_num in clusters:
                with st.expander(f"Cluster {cluster_num}"):
                    # Get drivers in this cluster
                    cluster_drivers = cluster_df[cluster_df['Cluster'] == cluster_num]
                    
                    # Show driver list with team if available
                    if 'Team' in cluster_drivers.columns:
                        st.markdown("##### Drivers in this cluster:")
                        for _, driver in cluster_drivers.iterrows():
                            st.markdown(f"- **{driver['Driver']}** ({driver['Team']})")
                    else:
                        st.markdown("##### Drivers in this cluster:")
                        for driver in cluster_drivers['Driver']:
                            st.markdown(f"- **{driver}**")
                    
                    # Show cluster statistics
                    st.markdown("##### Cluster Statistics:")
                    
                    metric_cols = ['MeanLapTime', 'MinLapTime', 'StdLapTime']
                    cluster_stats = cluster_drivers[metric_cols].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg. Lap Time", f"{cluster_stats['MeanLapTime']:.3f}s")
                    
                    with col2:
                        st.metric("Best Lap Time", f"{cluster_stats['MinLapTime']:.3f}s")
                    
                    with col3:
                        st.metric("Consistency (σ)", f"{cluster_stats['StdLapTime']:.3f}s")
                    
                    # Additional metrics if available
                    additional_cols = [col for col in cluster_drivers.columns if col.startswith('Mean') and col != 'MeanLapTime']
                    
                    if additional_cols:
                        st.markdown("##### Additional Metrics:")
                        
                        cols = st.columns(min(3, len(additional_cols)))
                        
                        for i, col in enumerate(additional_cols):
                            metric_name = col.replace('Mean', '')
                            with cols[i % 3]:
                                st.metric(f"Avg. {metric_name}", f"{cluster_drivers[col].mean():.1f}")
            
            # Show cluster interpretation
            st.markdown("#### Cluster Interpretation")
            
            if len(clusters) >= 2:
                # Create a summary of clusters
                cluster_summary = []
                
                for cluster_num in clusters:
                    cluster_drivers = cluster_df[cluster_df['Cluster'] == cluster_num]
                    
                    summary = {
                        'Cluster': cluster_num,
                        'Count': len(cluster_drivers),
                        'MeanLapTime': cluster_drivers['MeanLapTime'].mean(),
                        'MinLapTime': cluster_drivers['MinLapTime'].mean(),
                        'Consistency': cluster_drivers['StdLapTime'].mean()
                    }
                    
                    # Add team distribution if available
                    if 'Team' in cluster_drivers.columns:
                        team_counts = cluster_drivers['Team'].value_counts()
                        top_team = team_counts.index[0] if not team_counts.empty else "Unknown"
                        top_team_pct = (team_counts.iloc[0] / team_counts.sum()) * 100 if not team_counts.empty else 0
                        
                        summary['TopTeam'] = top_team
                        summary['TopTeamPct'] = top_team_pct
                    
                    cluster_summary.append(summary)
                
                summary_df = pd.DataFrame(cluster_summary)
                
                # Describe clusters based on performance
                fastest_cluster = summary_df.loc[summary_df['MinLapTime'].idxmin()]['Cluster']
                slowest_cluster = summary_df.loc[summary_df['MeanLapTime'].idxmax()]['Cluster']
                most_consistent = summary_df.loc[summary_df['Consistency'].idxmin()]['Cluster']
                
                st.markdown(f"""
                Based on the clustering results:
                
                - **Cluster {fastest_cluster}** contains the fastest drivers with the best lap times
                - **Cluster {slowest_cluster}** contains drivers with slower average lap times
                - **Cluster {most_consistent}** contains the most consistent drivers (lowest standard deviation)
                """)
                
                # Add team insights if available
                if 'TopTeam' in summary_df.columns:
                    team_insights = []
                    
                    for _, cluster in summary_df.iterrows():
                        if cluster['TopTeamPct'] >= 50:  # At least half of drivers from same team
                            team_insights.append(f"Cluster {cluster['Cluster']} is dominated by {cluster['TopTeam']} ({cluster['TopTeamPct']:.1f}% of drivers)")
                    
                    if team_insights:
                        st.markdown("##### Team Distribution Insights:")
                        for insight in team_insights:
                            st.markdown(f"- {insight}")
            else:
                st.info("Not enough different performance profiles to create meaningful clusters.")
        else:
            st.info("Please load session data from the sidebar to perform driver clustering.")
    
    # Tab 2: Lap Time Prediction
    with tab2:
        st.markdown("### Lap Time Prediction")
        st.markdown("""
        This analysis uses machine learning to predict lap times based on various factors like
        tire compound, tire age, fuel load, and track position. The model identifies which factors
        have the biggest impact on lap times.
        """)
        
        # Session selection for lap time prediction
        st.sidebar.markdown("### Lap Time Prediction")
        
        # Season selector
        predict_season = st.sidebar.selectbox("Season", options=seasons, index=len(seasons)-1, key="predict_season")
        
        # Event selector
        events_df = get_available_events(predict_season)
        
        if not events_df.empty:
            event_options = events_df['EventName'].tolist()
            
            # Find the last completed event
            event_index = 0
            for i, completed in enumerate(events_df['EventDate'] < pd.Timestamp.now()):
                if completed:
                    event_index = i
            
            predict_event = st.sidebar.selectbox("Event", 
                                               options=event_options,
                                               index=min(event_index, len(event_options)-1),
                                               key="predict_event")
            
            event_round = events_df[events_df['EventName'] == predict_event]['RoundNumber'].iloc[0]
        else:
            st.sidebar.warning(f"No events available for season {predict_season}")
            predict_event = None
            event_round = None
        
        # Only race sessions are relevant for lap time prediction
        if predict_event is not None:
            session_types = ['R', 'S']  # Race or Sprint
            session_labels = ['Race', 'Sprint']
            
            predict_session_type = st.sidebar.selectbox("Session", 
                                                     options=session_types,
                                                     format_func=lambda x: session_labels[session_types.index(x)],
                                                     key="predict_session_type")
        else:
            predict_session_type = None
        
        # Load data button
        if predict_event is not None and predict_session_type is not None:
            if st.sidebar.button("Load Data for Prediction", key="predict_load_button"):
                with st.spinner(f"Loading {predict_event} {predict_session_type} data..."):
                    session = load_session(predict_season, event_round, predict_session_type)
                    
                    if session is not None:
                        # Store session in session state
                        st.session_state.predict_session = session
                        st.session_state.predict_laps = session.laps
                        
                        # Store session info for reference
                        st.session_state.predict_session_info = {
                            'Season': predict_season,
                            'EventName': predict_event,
                            'RoundNumber': event_round,
                            'SessionType': predict_session_type,
                            'Year': session.event.year
                        }
                        
                        # Display success message
                        st.sidebar.success(f"Successfully loaded data for lap time prediction!")
                    else:
                        st.sidebar.error("Failed to load session data. Please try another session.")
        
        # Prediction options
        if 'predict_laps' in st.session_state and st.session_state.predict_laps is not None:
            laps_df = st.session_state.predict_laps
            session_info = st.session_state.predict_session_info
            
            st.markdown(f"#### Lap Time Prediction for {session_info['EventName']} {session_info['Year']}")
            
            # Two options: predict for a specific driver or for a team
            prediction_option = st.radio(
                "Prediction scope",
                options=["Driver", "Team"],
                key="prediction_scope"
            )
            
            if prediction_option == "Driver":
                # Get list of drivers
                drivers = sorted(laps_df['Driver'].unique())
                
                # Allow user to select a driver
                selected_driver = st.selectbox(
                    "Select driver",
                    options=drivers,
                    key="prediction_driver"
                )
                
                if st.button("Build Prediction Model", key="driver_predict_button"):
                    with st.spinner("Building lap time prediction model..."):
                        model, feature_imp, features, r2, rmse = predict_lap_time(_laps_df=laps_df, driver_code=selected_driver)
                        
                        if model is not None:
                            st.session_state.lap_prediction_results = {
                                'model': model,
                                'feature_importance': feature_imp,
                                'features': features,
                                'r2': r2,
                                'rmse': rmse,
                                'scope': 'driver',
                                'driver': selected_driver
                            }
                            
                            st.success("Successfully built prediction model!")
                        else:
                            st.error("Insufficient data to build a prediction model. Try another driver with more laps.")
            else:  # Team
                # Get list of teams
                if 'Team' in laps_df.columns:
                    teams = sorted(laps_df['Team'].unique())
                    
                    # Allow user to select a team
                    selected_team = st.selectbox(
                        "Select team",
                        options=teams,
                        key="prediction_team"
                    )
                    
                    if st.button("Build Prediction Model", key="team_predict_button"):
                        with st.spinner("Building lap time prediction model..."):
                            model, feature_imp, features, r2, rmse = predict_lap_time(_laps_df=laps_df, team_name=selected_team)
                            
                            if model is not None:
                                st.session_state.lap_prediction_results = {
                                    'model': model,
                                    'feature_importance': feature_imp,
                                    'features': features,
                                    'r2': r2,
                                    'rmse': rmse,
                                    'scope': 'team',
                                    'team': selected_team
                                }
                                
                                st.success("Successfully built prediction model!")
                            else:
                                st.error("Insufficient data to build a prediction model. Try another team with more laps.")
                else:
                    st.warning("Team information not available in the data. Please use driver-based prediction.")
            
            # Show prediction results if available
            if 'lap_prediction_results' in st.session_state:
                results = st.session_state.lap_prediction_results
                
                # Get scope-specific information
                if results['scope'] == 'driver':
                    entity_name = results['driver']
                    title = f"Lap Time Prediction for {entity_name}"
                else:
                    entity_name = results['team']
                    title = f"Lap Time Prediction for {entity_name} Team"
                
                st.markdown(f"#### {title}")
                
                # Show model performance metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Model R² Score", f"{results['r2']:.3f}")
                    st.markdown(f"*Higher is better (max 1.0). Represents how well the model explains lap time variation.*")
                
                with col2:
                    st.metric("RMSE", f"{results['rmse']:.3f}s")
                    st.markdown(f"*Lower is better. Average prediction error in seconds.*")
                
                # Show feature importance
                st.markdown("#### Feature Importance")
                st.markdown("*Which factors have the biggest impact on lap times:*")
                
                if results['feature_importance'] is not None:
                    # Plot feature importance
                    fig = plot_feature_importance(results['feature_importance'], title="Factors Affecting Lap Time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explain key factors
                    top_features = results['feature_importance'].head(3)['Feature'].tolist()
                    
                    feature_explanations = {
                        'LapNumber': "Lap number (related to fuel load and track evolution)",
                        'FuelLoad': "Estimated fuel load (higher at race start, lower at end)",
                        'TyreLife': "Tire age in laps (affects grip)",
                        'StintLap': "Lap number within current stint",
                        'Position': "Track position (may affect clean/dirty air)",
                    }
                    
                    compound_features = [f for f in top_features if "Compound" in f]
                    if compound_features:
                        feature_explanations["Compound"] = "Tire compound type (affects grip and degradation)"
                    
                    st.markdown("#### Key Factors Explained")
                    
                    for feature in top_features:
                        base_feature = feature
                        for key in feature_explanations:
                            if key in feature:
                                base_feature = key
                                break
                        
                        if base_feature in feature_explanations:
                            st.markdown(f"- **{feature}**: {feature_explanations[base_feature]}")
                        else:
                            st.markdown(f"- **{feature}**")
                
                # Show example prediction
                st.markdown("#### Interactive Lap Time Prediction")
                
                if results['features']:
                    st.markdown("Adjust the factors below to predict lap time:")
                    
                    # Create input sliders for numerical features
                    input_values = {}
                    
                    for feature in results['features']:
                        # Skip one-hot encoded features for manual input
                        if "Compound" in feature:
                            continue
                        
                        # Get feature range from data
                        feature_data = laps_df[feature] if feature in laps_df.columns else None
                        
                        if feature_data is not None and not feature_data.isnull().all():
                            min_val = float(feature_data.min())
                            max_val = float(feature_data.max())
                            default_val = float(feature_data.mean())
                            
                            # Create slider
                            input_values[feature] = st.slider(
                                feature,
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                key=f"predict_{feature}"
                            )
                    
                    # Create selector for compound if needed
                    compound_features = [f for f in results['features'] if "Compound" in f]
                    if compound_features:
                        compounds = [f.replace('Compound_', '') for f in compound_features]
                        
                        selected_compound = st.selectbox(
                            "Tire Compound",
                            options=compounds,
                            key="predict_compound"
                        )
                        
                        # Set one-hot encoding values
                        for compound in compounds:
                            feature_name = f"Compound_{compound}"
                            input_values[feature_name] = 1.0 if compound == selected_compound else 0.0
                    
                    # Make prediction
                    if st.button("Predict Lap Time", key="make_prediction"):
                        # Create input array
                        X_input = np.array([[input_values.get(feature, 0.0) for feature in results['features']]])
                        
                        # Make prediction
                        prediction = results['model'].predict(X_input)[0]
                        
                        # Display prediction
                        st.markdown("#### Predicted Lap Time")
                        st.markdown(f"### {prediction:.3f} seconds")
                        
                        # Compare to best and average lap times
                        if results['scope'] == 'driver':
                            driver_laps = laps_df[laps_df['Driver'] == entity_name]
                            valid_laps = driver_laps[(driver_laps['PitOutTime'].isnull()) & 
                                                 (driver_laps['PitInTime'].isnull())]
                            
                            if not valid_laps.empty and not valid_laps['LapTime'].isnull().all():
                                lap_times = valid_laps['LapTime'].dt.total_seconds()
                                
                                st.markdown(f"""
                                *For comparison:*
                                - Best lap: {lap_times.min():.3f}s
                                - Average lap: {lap_times.mean():.3f}s
                                """)
                        else:
                            team_laps = laps_df[laps_df['Team'] == entity_name]
                            valid_laps = team_laps[(team_laps['PitOutTime'].isnull()) & 
                                               (team_laps['PitInTime'].isnull())]
                            
                            if not valid_laps.empty and not valid_laps['LapTime'].isnull().all():
                                lap_times = valid_laps['LapTime'].dt.total_seconds()
                                
                                st.markdown(f"""
                                *For comparison:*
                                - Team best lap: {lap_times.min():.3f}s
                                - Team average lap: {lap_times.mean():.3f}s
                                """)
        else:
            st.info("Please load race session data from the sidebar to build a lap time prediction model.")
    
    # Tab 3: Anomaly Detection
    with tab3:
        st.markdown("### Anomaly Detection")
        st.markdown("""
        This analysis uses machine learning to identify unusual or anomalous laps that deviate significantly 
        from a driver's normal pattern. These could represent mistakes, damage, exceptional performance, or 
        other unusual circumstances.
        """)
        
        # Session selection for anomaly detection
        st.sidebar.markdown("### Anomaly Detection")
        
        # Season selector
        anomaly_season = st.sidebar.selectbox("Season", options=seasons, index=len(seasons)-1, key="anomaly_season")
        
        # Event selector
        events_df = get_available_events(anomaly_season)
        
        if not events_df.empty:
            event_options = events_df['EventName'].tolist()
            
            # Find the last completed event
            event_index = 0
            for i, completed in enumerate(events_df['EventDate'] < pd.Timestamp.now()):
                if completed:
                    event_index = i
            
            anomaly_event = st.sidebar.selectbox("Event", 
                                               options=event_options,
                                               index=min(event_index, len(event_options)-1),
                                               key="anomaly_event")
            
            event_round = events_df[events_df['EventName'] == anomaly_event]['RoundNumber'].iloc[0]
        else:
            st.sidebar.warning(f"No events available for season {anomaly_season}")
            anomaly_event = None
            event_round = None
        
        # Session type selector
        if anomaly_event is not None:
            session_types = ['R', 'Q', 'S', 'FP3', 'FP2', 'FP1']
            session_labels = ['Race', 'Qualifying', 'Sprint', 'Practice 3', 'Practice 2', 'Practice 1']
            
            anomaly_session_type = st.sidebar.selectbox("Session", 
                                                     options=session_types,
                                                     format_func=lambda x: session_labels[session_types.index(x)],
                                                     key="anomaly_session_type")
        else:
            anomaly_session_type = None
        
        # Load data button
        if anomaly_event is not None and anomaly_session_type is not None:
            if st.sidebar.button("Load Data for Anomaly Detection", key="anomaly_load_button"):
                with st.spinner(f"Loading {anomaly_event} {anomaly_session_type} data..."):
                    session = load_session(anomaly_season, event_round, anomaly_session_type)
                    
                    if session is not None:
                        # Store session in session state
                        st.session_state.anomaly_session = session
                        st.session_state.anomaly_laps = session.laps
                        
                        # Store session info for reference
                        st.session_state.anomaly_session_info = {
                            'Season': anomaly_season,
                            'EventName': anomaly_event,
                            'RoundNumber': event_round,
                            'SessionType': anomaly_session_type,
                            'Year': session.event.year
                        }
                        
                        # Display success message
                        st.sidebar.success(f"Successfully loaded data for anomaly detection!")
                    else:
                        st.sidebar.error("Failed to load session data. Please try another session.")
        
        # Perform anomaly detection
        if 'anomaly_laps' in st.session_state and st.session_state.anomaly_laps is not None:
            laps_df = st.session_state.anomaly_laps
            session_info = st.session_state.anomaly_session_info
            
            st.markdown(f"#### Anomaly Detection for {session_info['EventName']} {session_info['Year']}")
            
            # Get list of drivers
            drivers = sorted(laps_df['Driver'].unique())
            
            # Allow user to select drivers or analyze all
            analysis_option = st.radio(
                "Analysis scope",
                options=["Specific Driver", "All Drivers"],
                key="anomaly_scope"
            )
            
            selected_driver = None
            if analysis_option == "Specific Driver":
                # Allow user to select a driver
                selected_driver = st.selectbox(
                    "Select driver",
                    options=drivers,
                    key="anomaly_driver"
                )
            
            # Button to perform detection
            if st.button("Detect Anomalies", key="detect_anomalies_button"):
                with st.spinner("Analyzing lap data for anomalies..."):
                    anomalies = detect_anomalies(_laps_df=laps_df, driver_code=selected_driver)
                    
                    if anomalies:
                        st.session_state.anomaly_results = anomalies
                        st.success("Anomaly detection completed!")
                    else:
                        st.warning("No significant anomalies detected or insufficient data for analysis.")
            
            # Show anomaly results if available
            if 'anomaly_results' in st.session_state and st.session_state.anomaly_results:
                anomalies = st.session_state.anomaly_results
                
                # Count total anomalies
                total_anomalies = sum(len(df) for df in anomalies.values())
                
                st.markdown(f"#### Detected {total_anomalies} Anomalous Laps")
                
                # Create tabs for each driver
                if len(anomalies) > 1:
                    driver_tabs = st.tabs(list(anomalies.keys()))
                    
                    for i, (driver, driver_anomalies) in enumerate(anomalies.items()):
                        with driver_tabs[i]:
                            show_driver_anomalies(driver, driver_anomalies, laps_df)
                else:
                    # Just one driver
                    driver = list(anomalies.keys())[0]
                    show_driver_anomalies(driver, anomalies[driver], laps_df)
            else:
                if 'anomaly_results' in st.session_state:
                    st.info("No significant anomalies detected in the session data.")
        else:
            st.info("Please load session data from the sidebar to perform anomaly detection.")
    
    # Tab 4: Strategy Impact
    with tab4:
        st.markdown("### Race Strategy Impact Analysis")
        st.markdown("""
        This analysis uses machine learning to quantify the impact of different strategy decisions 
        on race outcomes. It helps identify which strategy elements (pit stops, tire choices, stint lengths) 
        have the greatest effect on final results.
        """)
        
        # Session selection for strategy analysis
        st.sidebar.markdown("### Strategy Analysis")
        
        # Season selector
        strategy_season = st.sidebar.selectbox("Season", options=seasons, index=len(seasons)-1, key="strategy_ml_season")
        
        # Event selector
        events_df = get_available_events(strategy_season)
        
        if not events_df.empty:
            event_options = events_df['EventName'].tolist()
            
            # Find the last completed event
            event_index = 0
            for i, completed in enumerate(events_df['EventDate'] < pd.Timestamp.now()):
                if completed:
                    event_index = i
            
            strategy_event = st.sidebar.selectbox("Event", 
                                               options=event_options,
                                               index=min(event_index, len(event_options)-1),
                                               key="strategy_ml_event")
            
            event_round = events_df[events_df['EventName'] == strategy_event]['RoundNumber'].iloc[0]
        else:
            st.sidebar.warning(f"No events available for season {strategy_season}")
            strategy_event = None
            event_round = None
        
        # Only race sessions are relevant for strategy analysis
        if strategy_event is not None:
            session_types = ['R']  # Race only
            session_labels = ['Race']
            
            strategy_session_type = st.sidebar.selectbox("Session", 
                                                      options=session_types,
                                                      format_func=lambda x: session_labels[session_types.index(x)],
                                                      key="strategy_ml_session_type")
        else:
            strategy_session_type = None
        
        # Load data button
        if strategy_event is not None and strategy_session_type is not None:
            if st.sidebar.button("Load Data for Strategy Analysis", key="strategy_ml_load_button"):
                with st.spinner(f"Loading {strategy_event} {strategy_session_type} data..."):
                    session = load_session(strategy_season, event_round, strategy_session_type)
                    
                    if session is not None:
                        # Store session in session state
                        st.session_state.strategy_ml_session = session
                        st.session_state.strategy_ml_laps = session.laps
                        st.session_state.strategy_ml_results = session.results
                        
                        # Store session info for reference
                        st.session_state.strategy_ml_session_info = {
                            'Season': strategy_season,
                            'EventName': strategy_event,
                            'RoundNumber': event_round,
                            'SessionType': strategy_session_type,
                            'Year': session.event.year
                        }
                        
                        # Display success message
                        st.sidebar.success(f"Successfully loaded data for strategy analysis!")
                    else:
                        st.sidebar.error("Failed to load session data. Please try another session.")
        
        # Perform strategy analysis
        if 'strategy_ml_laps' in st.session_state and st.session_state.strategy_ml_laps is not None:
            laps_df = st.session_state.strategy_ml_laps
            results_df = st.session_state.strategy_ml_results if 'strategy_ml_results' in st.session_state else None
            session_info = st.session_state.strategy_ml_session_info
            
            st.markdown(f"#### Strategy Impact Analysis for {session_info['EventName']} {session_info['Year']}")
            
            # Check if we have race results
            if results_df is None or results_df.empty:
                st.warning("Race results are not available. Strategy impact analysis requires final race positions.")
                return
            
            # Button to perform analysis
            if st.button("Analyze Strategy Impact", key="analyze_strategy_button"):
                with st.spinner("Analyzing race strategy impact..."):
                    strategy_df, importance_df = create_race_strategy_model(_laps_df=laps_df, session_results=results_df)
                    
                    if strategy_df is not None:
                        st.session_state.strategy_model_results = {
                            'strategy_df': strategy_df,
                            'importance_df': importance_df
                        }
                        st.success("Strategy analysis completed!")
                    else:
                        st.warning("Insufficient data for strategy analysis. This may be due to missing compound information or incomplete race data.")
            
            # Show strategy analysis results if available
            if 'strategy_model_results' in st.session_state and st.session_state.strategy_model_results is not None:
                results = st.session_state.strategy_model_results
                strategy_df = results['strategy_df']
                importance_df = results['importance_df']
                
                st.markdown("#### Race Strategy Overview")
                
                # Show stint overview
                st.dataframe(strategy_df, use_container_width=True)
                
                # Show feature importance if available
                if importance_df is not None:
                    st.markdown("#### Strategy Factors Impact on Race Outcome")
                    
                    # Plot feature importance
                    fig = plot_feature_importance(importance_df, title="Factors Influencing Race Position")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explain key factors
                    st.markdown("#### Key Strategy Insights")
                    
                    # Get top factors
                    top_features = importance_df.head(3)['Feature'].tolist()
                    
                    # Build explanations
                    strategy_explanations = []
                    
                    for feature in top_features:
                        if 'NumStints' in feature:
                            strategy_explanations.append(f"- **Number of pit stops** was a significant factor in race outcome")
                        elif 'AvgStintLength' in feature:
                            strategy_explanations.append(f"- **Average stint length** had a strong influence on final position")
                        elif 'CompoundsUsed' in feature:
                            strategy_explanations.append(f"- **Tire compound variety** played an important role in race results")
                        elif 'MostUsed' in feature:
                            compound = feature.replace('MostUsed_', '')
                            strategy_explanations.append(f"- Teams primarily using **{compound} tires** showed distinct performance patterns")
                        elif 'Laps' in feature and any(compound in feature for compound in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']):
                            compound = ''.join([c for c in feature if c.isupper()])
                            strategy_explanations.append(f"- **Time spent on {compound} tires** was influential for final position")
                        elif 'AvgRacePace' in feature:
                            strategy_explanations.append(f"- **Overall race pace** was a key determinant of race outcome")
                        elif 'AvgDegradation' in feature:
                            strategy_explanations.append(f"- **Tire degradation management** significantly affected final results")
                        else:
                            strategy_explanations.append(f"- **{feature}** was an important factor in race outcome")
                    
                    # Display insights
                    for explanation in strategy_explanations:
                        st.markdown(explanation)
                    
                    # Additional insights based on data
                    st.markdown("#### Additional Strategy Insights")
                    
                    # Calculate average statistics by final position
                    if 'FinalPosition' in strategy_df.columns and not strategy_df['FinalPosition'].isnull().all():
                        # Group by position groups (podium, points, non-points)
                        strategy_df['PositionGroup'] = 'Other'
                        strategy_df.loc[strategy_df['FinalPosition'] <= 3, 'PositionGroup'] = 'Podium'
                        strategy_df.loc[(strategy_df['FinalPosition'] > 3) & (strategy_df['FinalPosition'] <= 10), 'PositionGroup'] = 'Points'
                        
                        # Group by position and calculate statistics
                        position_stats = strategy_df.groupby('PositionGroup').agg({
                            'Stint': 'nunique',  # Number of stints
                            'StintLength': 'mean',  # Average stint length
                        }).reset_index()
                        
                        position_stats.columns = ['Position Group', 'Avg Stints', 'Avg Stint Length']
                        
                        # Display statistics
                        st.dataframe(position_stats, use_container_width=True)
                        
                        # Check for compound preferences
                        if 'Compound' in strategy_df.columns:
                            # Calculate compound usage by position group
                            compound_usage = []
                            
                            for group in ['Podium', 'Points', 'Other']:
                                group_stints = strategy_df[strategy_df['PositionGroup'] == group]
                                
                                if not group_stints.empty:
                                    compound_counts = group_stints['Compound'].value_counts(normalize=True)
                                    
                                    for compound, pct in compound_counts.items():
                                        compound_usage.append({
                                            'Position Group': group,
                                            'Compound': compound,
                                            'Usage %': pct * 100
                                        })
                            
                            if compound_usage:
                                compound_df = pd.DataFrame(compound_usage)
                                
                                # Create compound usage chart
                                fig = px.bar(
                                    compound_df,
                                    x='Position Group',
                                    y='Usage %',
                                    color='Compound',
                                    title="Tire Compound Usage by Position Group",
                                    barmode='group'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Look for significant differences
                                podium_compounds = compound_df[compound_df['Position Group'] == 'Podium']['Compound'].tolist()
                                other_compounds = compound_df[compound_df['Position Group'] == 'Other']['Compound'].tolist()
                                
                                podium_preferences = [c for c in podium_compounds if c not in other_compounds]
                                if podium_preferences:
                                    st.markdown(f"- Podium finishers used **{', '.join(podium_preferences)}** tires which weren't used by lower finishers")
                        
                        # Analyze pit stop timing
                        if 'StartLap' in strategy_df.columns and 'EndLap' in strategy_df.columns:
                            # Calculate first pit stop lap by driver
                            driver_first_pit = strategy_df.groupby('Driver').agg({
                                'EndLap': 'min',  # First stint end = first pit stop
                                'FinalPosition': 'first'
                            }).reset_index()
                            
                            # Check correlation between first pit timing and result
                            if not driver_first_pit.empty and 'FinalPosition' in driver_first_pit.columns:
                                correlation = driver_first_pit['EndLap'].corr(driver_first_pit['FinalPosition'])
                                
                                if pd.notna(correlation) and abs(correlation) > 0.3:
                                    pit_direction = "later" if correlation < 0 else "earlier"
                                    st.markdown(f"- Teams that pitted **{pit_direction}** for their first stop tended to finish better")
                                    
                                    # Create scatter plot
                                    fig = px.scatter(
                                        driver_first_pit,
                                        x='EndLap',
                                        y='FinalPosition',
                                        text='Driver',
                                        title="First Pit Stop Timing vs Final Position",
                                        labels={
                                            'EndLap': 'First Pit Stop (Lap)',
                                            'FinalPosition': 'Final Position'
                                        }
                                    )
                                    
                                    fig.update_traces(textposition='top center')
                                    fig.update_yaxes(autorange="reversed")  # Lower position is better
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("The data analysis suggests that driver pace was the dominant factor, with strategy having minimal discernible impact on this race.")
        else:
            st.info("Please load race session data from the sidebar to analyze strategy impact.")

def show_driver_anomalies(driver, anomalies_df, all_laps_df):
    """Display anomalous laps for a specific driver"""
    st.markdown(f"#### Anomalous Laps for {driver}")
    
    # Sort by anomaly score
    sorted_anomalies = anomalies_df.sort_values('AnomalyScore', ascending=False)
    
    # Show table with anomalies
    display_cols = ['LapNumber', 'LapTime', 'AnomalyScore']
    
    # Add additional columns if available
    extra_cols = ['Compound', 'TyreLife']
    for col in extra_cols:
        if col in sorted_anomalies.columns:
            display_cols.append(col)
    
    # Add sector times if available
    sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in sector_cols:
        if col in sorted_anomalies.columns and not sorted_anomalies[col].isnull().all():
            display_cols.append(col)
    
    st.dataframe(sorted_anomalies[display_cols], use_container_width=True)
    
    # Plot anomalous laps in context
    driver_laps = all_laps_df[all_laps_df['Driver'] == driver].copy()
    
    # Convert lap times to seconds
    driver_laps['LapTimeSeconds'] = driver_laps['LapTime'].dt.total_seconds()
    
    # Create figure
    fig = go.Figure()
    
    # Add all laps
    fig.add_trace(go.Scatter(
        x=driver_laps['LapNumber'],
        y=driver_laps['LapTimeSeconds'],
        mode='markers',
        marker=dict(color='blue', size=8),
        name='Normal Laps'
    ))
    
    # Add anomalous laps
    anomalies_df['LapTimeSeconds'] = anomalies_df['LapTime'].dt.total_seconds()
    
    fig.add_trace(go.Scatter(
        x=anomalies_df['LapNumber'],
        y=anomalies_df['LapTimeSeconds'],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name='Anomalous Laps'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Lap Times for {driver} with Anomalies Highlighted",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (seconds)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyze anomalous laps
    st.markdown("#### Anomaly Analysis")
    
    # Check if lap times are unusually fast or slow
    if not anomalies_df.empty and 'LapTimeSeconds' in anomalies_df.columns and 'LapTimeSeconds' in driver_laps.columns:
        # Calculate mean and std for normal laps
        normal_laps_mask = ~driver_laps['LapNumber'].isin(anomalies_df['LapNumber'])
        normal_lap_times = driver_laps.loc[normal_laps_mask, 'LapTimeSeconds']
        
        if not normal_lap_times.empty:
            mean_time = normal_lap_times.mean()
            std_time = normal_lap_times.std()
            
            # Classify anomalies
            fast_anomalies = anomalies_df[anomalies_df['LapTimeSeconds'] < (mean_time - std_time)]
            slow_anomalies = anomalies_df[anomalies_df['LapTimeSeconds'] > (mean_time + std_time)]
            
            if not fast_anomalies.empty:
                st.markdown(f"##### Unusually Fast Laps")
                st.markdown(f"Found {len(fast_anomalies)} anomalously fast laps, potentially showing exceptional performance.")
                
                # Show fast laps
                fast_display = fast_anomalies[['LapNumber', 'LapTime', 'AnomalyScore']].sort_values('LapTimeSeconds')
                st.dataframe(fast_display, use_container_width=True)
            
            if not slow_anomalies.empty:
                st.markdown(f"##### Unusually Slow Laps")
                st.markdown(f"Found {len(slow_anomalies)} anomalously slow laps, potentially indicating issues.")
                
                # Show slow laps
                slow_display = slow_anomalies[['LapNumber', 'LapTime', 'AnomalyScore']].sort_values('LapTimeSeconds', ascending=False)
                st.dataframe(slow_display, use_container_width=True)
            
            # Check for unusual sector patterns
            sector_anomalies = []
            for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                if sector in anomalies_df.columns and not anomalies_df[sector].isnull().all():
                    # Convert to seconds
                    anomalies_df[f'{sector}Seconds'] = anomalies_df[sector].dt.total_seconds()
                    driver_laps[f'{sector}Seconds'] = driver_laps[sector].dt.total_seconds()
                    
                    # Calculate mean and std for normal laps
                    normal_sector_times = driver_laps.loc[normal_laps_mask, f'{sector}Seconds']
                    
                    if not normal_sector_times.empty:
                        sector_mean = normal_sector_times.mean()
                        sector_std = normal_sector_times.std()
                        
                        for _, lap in anomalies_df.iterrows():
                            if pd.notna(lap[f'{sector}Seconds']):
                                z_score = abs(lap[f'{sector}Seconds'] - sector_mean) / sector_std
                                
                                if z_score > 2:  # More than 2 standard deviations
                                    direction = "slow" if lap[f'{sector}Seconds'] > sector_mean else "fast"
                                    sector_anomalies.append({
                                        'LapNumber': lap['LapNumber'],
                                        'Sector': sector.replace('Time', ''),
                                        'Direction': direction,
                                        'Z-Score': z_score
                                    })
            
            if sector_anomalies:
                sector_df = pd.DataFrame(sector_anomalies)
                
                st.markdown(f"##### Unusual Sector Times")
                st.markdown(f"The following sectors showed significant deviation from normal performance:")
                
                st.dataframe(sector_df, use_container_width=True)
    
    # Provide overall analysis
    st.markdown("#### Possible Explanations for Anomalies")
    
    explanations = []
    
    # Check for compound related anomalies
    if 'Compound' in anomalies_df.columns and not anomalies_df['Compound'].isnull().all():
        compound_counts = anomalies_df['Compound'].value_counts()
        
        if len(compound_counts) == 1:
            compound = compound_counts.index[0]
            explanations.append(f"- All anomalies occurred on **{compound}** tires, suggesting potential issues with this compound for {driver}")
    
    # Check for lap position anomalies
    lap_positions = anomalies_df['LapNumber'].tolist()
    
    # Check for consecutive anomalies
    consecutive_groups = []
    group = []
    
    for i in range(len(lap_positions)):
        if i == 0 or lap_positions[i] == lap_positions[i-1] + 1:
            group.append(lap_positions[i])
        else:
            if len(group) > 1:
                consecutive_groups.append(group)
            group = [lap_positions[i]]
    
    if len(group) > 1:
        consecutive_groups.append(group)
    
    if consecutive_groups:
        for group in consecutive_groups:
            explanations.append(f"- Consecutive anomalies on laps {min(group)}-{max(group)} may indicate a temporary car issue or traffic situation")
    
    # Check for beginning/end of race
    early_anomalies = [lap for lap in lap_positions if lap <= 5]
    late_anomalies = [lap for lap in lap_positions if lap >= max(driver_laps['LapNumber']) - 5]
    
    if early_anomalies:
        explanations.append(f"- Anomalies at the start of the race (laps {', '.join(map(str, early_anomalies))}) may reflect full fuel load or opening lap incidents")
    
    if late_anomalies:
        explanations.append(f"- Anomalies at the end of the race (laps {', '.join(map(str, late_anomalies))}) could indicate low fuel, tire wear, or changed race priorities")
    
    # Display explanations
    if explanations:
        for explanation in explanations:
            st.markdown(explanation)
    else:
        st.markdown("- No clear patterns detected in the anomalies. They may be due to random factors or complex interactions.")

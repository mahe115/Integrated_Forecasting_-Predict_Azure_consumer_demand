# ===== TAB 7: ML FORECASTING (ENHANCED IMPLEMENTATION) =====
with tab7:
    st.subheader("🔮 Machine Learning Forecasting")

    # Load model information
    model_info = fetch_api("forecast/models")

    if model_info:
        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("### 🎯 Model Status")

            # Display model status for each region
            for region, info in model_info['models'].items():
                status_color = "#28a745" if info['loaded'] else "#dc3545"
                status_text = "✅ Loaded" if info['loaded'] else "❌ Not Loaded"

                st.markdown(f"""
                <div style="background: #e9ecef; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                    <strong>{region}</strong><br>
                    Model: {info['model_type']}<br>
                    Status: <span style="color: {status_color};">{status_text}</span>
                </div>
                """, unsafe_allow_html=True)

            # Forecasting controls
            st.markdown("### ⚙️ Forecast Settings")
            forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
            selected_region = st.selectbox("Focus Region", ["All Regions"] + list(model_info['models'].keys()))

            if st.button("🚀 Generate Forecasts", type="primary"):
                st.session_state.generate_forecast = True

        with col1:
            st.markdown("### 📈 Forecast Results")

            # Generate forecasts if button clicked
            if hasattr(st.session_state, 'generate_forecast') and st.session_state.generate_forecast:
                with st.spinner("Generating forecasts..."):
                    params = {'days': forecast_days}
                    if selected_region != "All Regions":
                        params['region'] = selected_region

                    forecast_data = fetch_api("forecast/predict", params=params)

                    if forecast_data:
                        # Create forecast visualization
                        fig = go.Figure()

                        for region, data in forecast_data.items():
                            if 'error' in data:
                                st.error(f"{region}: {data['error']}")
                                continue

                            # Plot historical data
                            if 'historical' in data:
                                hist = data['historical']
                                fig.add_trace(go.Scatter(
                                    x=hist['dates'],
                                    y=hist['actual_cpu'],
                                    mode='lines',
                                    name=f'{region} - Historical',
                                    line=dict(color='blue', width=2),
                                    hovertemplate=f'<b>{region} - Historical</b><br>Date: %{{x}}<br>CPU Usage: %{{y:.1f}}%<extra></extra>'
                                ))

                            # Plot forecast
                            fig.add_trace(go.Scatter(
                                x=data['dates'],
                                y=data['predicted_cpu'],
                                mode='lines+markers',
                                name=f'{region} - Forecast ({data["model_info"]["type"]})',
                                line=dict(dash='dash', width=2),
                                marker=dict(size=4),
                                hovertemplate=f'<b>{region} - Forecast</b><br>Date: %{{x}}<br>Predicted CPU: %{{y:.1f}}%<br>Model: {data["model_info"]["type"]}<extra></extra>'
                            ))

                        fig.update_layout(
                            title="Azure CPU Usage Forecast - ML Predictions",
                            xaxis_title="Date",
                            yaxis_title="CPU Usage (%)",
                            height=600,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Display forecast summary
                        st.markdown("### 📋 Forecast Summary")

                        summary_data = []
                        for region, data in forecast_data.items():
                            if 'predicted_cpu' in data:
                                avg_forecast = np.mean(data['predicted_cpu'])
                                max_forecast = np.max(data['predicted_cpu'])
                                min_forecast = np.min(data['predicted_cpu'])
                                model_type = data['model_info']['type']

                                summary_data.append({
                                    'Region': region,
                                    'Model': model_type,
                                    'Avg Predicted CPU': f"{avg_forecast:.1f}%",
                                    'Max Predicted CPU': f"{max_forecast:.1f}%",
                                    'Min Predicted CPU': f"{min_forecast:.1f}%",
                                    'Forecast Period': f"{forecast_days} days"
                                })

                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)

                        # Reset the session state
                        st.session_state.generate_forecast = False

                    else:
                        st.error("Failed to generate forecasts. Please check the API connection.")

        # Model Performance Comparison
        st.markdown("### 🏆 Model Performance Comparison")

        comparison_data = fetch_api("forecast/comparison")
        if comparison_data:
            performance = comparison_data['regional_performance']

            col1, col2 = st.columns(2)

            with col1:
                # RMSE comparison
                regions = list(performance.keys())
                rmse_values = [performance[region]['rmse'] for region in regions]
                models = [performance[region]['model'] for region in regions]

                fig_rmse = go.Figure(data=[
                    go.Bar(
                        x=regions,
                        y=rmse_values,
                        text=[f"{m}<br>RMSE: {r:.2f}" for m, r in zip(models, rmse_values)],
                        textposition='auto',
                        marker_color=['#ff6b6b' if m == 'ARIMA' else '#4ecdc4' for m in models]
                    )
                ])
                fig_rmse.update_layout(
                    title="Model Performance - RMSE by Region",
                    xaxis_title="Region",
                    yaxis_title="RMSE",
                    height=400
                )
                st.plotly_chart(fig_rmse, use_container_width=True)

            with col2:
                # MAE comparison
                mae_values = [performance[region]['mae'] for region in regions]

                fig_mae = go.Figure(data=[
                    go.Bar(
                        x=regions,
                        y=mae_values,
                        text=[f"{m}<br>MAE: {r:.2f}" for m, r in zip(models, mae_values)],
                        textposition='auto',
                        marker_color=['#ff6b6b' if m == 'ARIMA' else '#4ecdc4' for m in models]
                    )
                ])
                fig_mae.update_layout(
                    title="Model Performance - MAE by Region",
                    xaxis_title="Region", 
                    yaxis_title="MAE",
                    height=400
                )
                st.plotly_chart(fig_mae, use_container_width=True)

            # Performance summary
            overall_stats = comparison_data['overall_stats']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average RMSE", f"{overall_stats['avg_rmse']:.2f}")
            with col2:
                st.metric("Average MAE", f"{overall_stats['avg_mae']:.2f}")
            with col3:
                st.metric("Best Performing Region", overall_stats['best_rmse_region'])
            with col4:
                st.metric("LSTM Regions", len(overall_stats['lstm_regions']))

    else:
        st.error("⚠️ Unable to load model information. Please ensure the API server is running and models are properly loaded.")

        # Show placeholder information
        st.markdown("""
        <div style="background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 5px; border: 1px solid #ffeaa7; margin-bottom: 1rem;">
            <h4>🔧 Model Setup Required</h4>
            <p>To enable forecasting functionality, ensure:</p>
            <ul>
                <li>Trained models are saved in the <code>models/</code> directory</li>
                <li>Backend API server is running on localhost:5000</li>
                <li>Required Python packages are installed (tensorflow, statsmodels)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


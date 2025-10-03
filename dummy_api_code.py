#-------------------------------------------i have edited up -------------------------------------------------


# ===== TAB 7: ML FORECASTING (FULL IMPLEMENTATION) =====

# Prepare aggregated data for forecasting (same as notebook)
region_daily = df.groupby(['region', 'date']).agg({
    'usage_cpu': 'mean',
    'usage_storage': 'mean', 
    'users_active': 'sum',
    'economic_index': 'first',
    'cloud_market_demand': 'first',
    'holiday': 'max'
}).reset_index()

# Create region-specific DataFrames for forecasting
region_dfs = {}
for region in region_daily['region'].unique():
    region_data = region_daily[region_daily['region'] == region].copy()
    region_data = region_data.drop('region', axis=1).set_index('date').sort_index()
    region_dfs[region] = region_data

# Model configuration based on training results
FINAL_SELECTION = {
    'East US': 'LSTM',
    'North Europe': 'ARIMA', 
    'Southeast Asia': 'LSTM',
    'West US': 'LSTM'
}

MODEL_DIR = 'models/'

# Global variables for loaded models
loaded_models = {}
loaded_scalers = {}

def load_models_on_startup():
    """Load all trained models on application startup"""
    global loaded_models, loaded_scalers

    print("Loading trained models...")

    for region, model_type in FINAL_SELECTION.items():
        region_clean = region.replace(' ', '')

        try:
            if model_type == 'ARIMA':
                model_path = f"{MODEL_DIR}{region_clean}ARIMA.pkl"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        loaded_models[region] = pickle.load(f)
                    print(f"✓ Loaded ARIMA model for {region}")
                else:
                    print(f"✗ ARIMA model file not found for {region}: {model_path}")

            elif model_type == 'LSTM':
                model_path = f"{MODEL_DIR}{region_clean}LSTM.h5"
                scaler_path = f"{MODEL_DIR}{region_clean}LSTMscaler.pkl"

                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    loaded_models[region] = load_model(model_path)
                    with open(scaler_path, 'rb') as f:
                        loaded_scalers[region] = pickle.load(f)
                    print(f"✓ Loaded LSTM model and scaler for {region}")
                else:
                    print(f"✗ LSTM files not found for {region}")

        except Exception as e:
            print(f"✗ Error loading model for {region}: {str(e)}")

@app.route('/api/forecast/models')
def get_available_models():
    """Get information about available forecasting models"""
    try:
        model_info = {}

        for region, model_type in FINAL_SELECTION.items():
            model_info[region] = {
                'model_type': model_type,
                'loaded': region in loaded_models,
                'has_scaler': region in loaded_scalers if model_type == 'LSTM' else None
            }

        return jsonify({
            'models': model_info,
            'total_regions': len(FINAL_SELECTION),
            'model_types_used': list(set(FINAL_SELECTION.values()))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/predict')
def generate_forecasts():
    """Generate forecasts for all regions using trained models"""
    try:
        # Get parameters
        forecast_days = int(request.args.get('days', 30))  # Default 30 days
        region_filter = request.args.get('region', None)

        results = {}

        regions_to_process = [region_filter] if region_filter else FINAL_SELECTION.keys()

        for region in regions_to_process:
            if region not in loaded_models:
                results[region] = {'error': f'Model not loaded for {region}'}
                continue

            try:
                model_type = FINAL_SELECTION[region]
                region_data = region_dfs[region]

                if model_type == 'ARIMA':
                    forecast_data = generate_arima_forecast(
                        loaded_models[region], 
                        region_data, 
                        forecast_days
                    )
                elif model_type == 'LSTM':
                    forecast_data = generate_lstm_forecast(
                        loaded_models[region],
                        loaded_scalers[region],
                        region_data,
                        forecast_days
                    )
                else:
                    forecast_data = {'error': f'Unknown model type: {model_type}'}

                results[region] = forecast_data
                results[region]['model_type'] = model_type

            except Exception as e:
                results[region] = {'error': f'Forecasting error: {str(e)}'}

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_arima_forecast(model, region_data, forecast_days):
    """Generate ARIMA forecast"""
    try:
        # Get the last date and generate future dates
        last_date = region_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )

        # Generate forecast
        forecast = model.forecast(steps=forecast_days)

        # Prepare response
        forecast_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'predicted_cpu': [float(f) for f in forecast],
            'model_info': {
                'type': 'ARIMA',
                'forecast_horizon': forecast_days
            }
        }

        # Add recent historical data for context
        recent_data = region_data['usage_cpu'].tail(30)
        forecast_data['historical'] = {
            'dates': [d.strftime('%Y-%m-%d') for d in recent_data.index],
            'actual_cpu': [float(v) for v in recent_data.values]
        }

        return forecast_data

    except Exception as e:
        return {'error': f'ARIMA forecast error: {str(e)}'}

def generate_lstm_forecast(model, scaler, region_data, forecast_days):
    """Generate LSTM forecast"""
    try:
        # Prepare sequence data
        sequence_length = 7  # As used in training
        cpu_data = region_data['usage_cpu'].values

        # Scale the data
        scaled_data = scaler.transform(cpu_data.reshape(-1, 1))

        # Generate forecasts iteratively
        last_sequence = scaled_data[-sequence_length:]
        forecasts_scaled = []

        current_sequence = last_sequence.reshape(1, sequence_length, 1)

        for _ in range(forecast_days):
            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)
            forecasts_scaled.append(next_pred[0, 0])

            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]

        # Inverse transform predictions
        forecasts_scaled = np.array(forecasts_scaled).reshape(-1, 1)
        forecasts = scaler.inverse_transform(forecasts_scaled)

        # Generate future dates
        last_date = region_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )

        # Prepare response
        forecast_data = {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'predicted_cpu': [float(f[0]) for f in forecasts],
            'model_info': {
                'type': 'LSTM',
                'sequence_length': sequence_length,
                'forecast_horizon': forecast_days
            }
        }

        # Add recent historical data for context
        recent_data = region_data['usage_cpu'].tail(30)
        forecast_data['historical'] = {
            'dates': [d.strftime('%Y-%m-%d') for d in recent_data.index],
            'actual_cpu': [float(v) for v in recent_data.values]
        }

        return forecast_data

    except Exception as e:
        return {'error': f'LSTM forecast error: {str(e)}'}

@app.route('/api/forecast/comparison')
def model_comparison():
    """Get model performance comparison"""
    try:
        model_performance = {
            'East US': {'model': 'LSTM', 'rmse': 13.68, 'mae': 11.46},
            'North Europe': {'model': 'ARIMA', 'rmse': 15.90, 'mae': 14.25},
            'Southeast Asia': {'model': 'LSTM', 'rmse': 14.86, 'mae': 13.06},
            'West US': {'model': 'LSTM', 'rmse': 15.38, 'mae': 12.85}
        }

        # Calculate overall statistics
        all_rmse = [perf['rmse'] for perf in model_performance.values()]
        all_mae = [perf['mae'] for perf in model_performance.values()]

        summary = {
            'regional_performance': model_performance,
            'overall_stats': {
                'avg_rmse': np.mean(all_rmse),
                'avg_mae': np.mean(all_mae),
                'best_rmse_region': min(model_performance.items(), key=lambda x: x[1]['rmse'])[0],
                'lstm_regions': [k for k, v in model_performance.items() if v['model'] == 'LSTM'],
                'arima_regions': [k for k, v in model_performance.items() if v['model'] == 'ARIMA']
            }
        }

        return jsonify(summary)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load models when this section is added
load_models_on_startup()


#-------------------------------------------i have edited down-------------------------------------------------

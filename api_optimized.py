from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError


app = Flask(__name__)
CORS(app)

# Load datasets at startup
print("Loading datasets...")
df = pd.read_csv('data/processed/cleaned_merged.csv', parse_dates=['date'])
print(f"Loaded {len(df)} records from cleaned_merged.csv")

# Data preprocessing for API
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# ===== TAB 1: OVERVIEW & KPIs =====

@app.route('/api/kpis')
def get_kpis():
    """Get key performance indicators for dashboard overview"""
    try:
        kpis = {
            'peak_cpu': float(df['usage_cpu'].max()),
            'peak_cpu_details': {
                'date': df.loc[df['usage_cpu'].idxmax(), 'date'].isoformat(),
                'region': df.loc[df['usage_cpu'].idxmax(), 'region'],
                'resource_type': df.loc[df['usage_cpu'].idxmax(), 'resource_type']
            },
            'max_storage': float(df['usage_storage'].max()),
            'max_storage_details': {
                'date': df.loc[df['usage_storage'].idxmax(), 'date'].isoformat(),
                'region': df.loc[df['usage_storage'].idxmax(), 'region'],
                'resource_type': df.loc[df['usage_storage'].idxmax(), 'resource_type']
            },
            'peak_users': int(df['users_active'].max()),
            'peak_users_details': {
                'date': df.loc[df['users_active'].idxmax(), 'date'].isoformat(),
                'region': df.loc[df['users_active'].idxmax(), 'region'],
                'resource_type': df.loc[df['users_active'].idxmax(), 'resource_type']
            },
            'avg_cpu': float(df['usage_cpu'].mean()),
            'avg_storage': float(df['usage_storage'].mean()),
            'avg_users': float(df['users_active'].mean()),
            'total_regions': int(df['region'].nunique()),
            'total_resource_types': int(df['resource_type'].nunique()),
            'data_points': int(len(df)),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat(),
                'days': int((df['date'].max() - df['date'].min()).days)
            }
        }
        
        # Calculate holiday impact
        holiday_avg = df[df['holiday'] == 1]['usage_cpu'].mean()
        regular_avg = df[df['holiday'] == 0]['usage_cpu'].mean()
        holiday_impact = ((holiday_avg - regular_avg) / regular_avg) * 100
        
        kpis['holiday_impact'] = {
            'percentage': float(holiday_impact),
            'holiday_avg_cpu': float(holiday_avg),
            'regular_avg_cpu': float(regular_avg)
        }
        
        return jsonify(kpis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sparklines')
def get_sparklines():
    """Get mini trend data for sparkline charts"""
    try:
        # Get last 30 days of data
        latest_date = df['date'].max()
        last_30_days = df[df['date'] > (latest_date - timedelta(days=30))]
        
        daily_trends = last_30_days.groupby('date').agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()
        
        sparklines = {
            'cpu_trend': daily_trends[['date', 'usage_cpu']].to_dict('records'),
            'storage_trend': daily_trends[['date', 'usage_storage']].to_dict('records'),
            'users_trend': daily_trends[['date', 'users_active']].to_dict('records')
        }
        
        # Convert dates to ISO format
        for trend in sparklines.values():
            for point in trend:
                point['date'] = pd.to_datetime(point['date']).isoformat()
        
        return jsonify(sparklines)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== : USAGE TRENDS =====

@app.route('/api/time-series')
def get_time_series():
    """Get comprehensive time series data for trends analysis"""
    try:
        region_filter = request.args.get('region')
        resource_filter = request.args.get('resource_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        data = df.copy()
        
        # Apply filters
        if region_filter:
            data = data[data['region'] == region_filter]
        if resource_filter:
            data = data[data['resource_type'] == resource_filter]
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]
        
        # Group by date for time series
        time_series = data.groupby('date').agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean',
            'economic_index': 'mean',
            'cloud_market_demand': 'mean'
        }).reset_index()
        
        # Convert dates to ISO format
        time_series['date'] = time_series['date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(time_series.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends/regional')
def get_regional_trends():
    """Get time series data grouped by region"""
    try:
        regional_trends = df.groupby(['date', 'region']).agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()
        
        regional_trends['date'] = regional_trends['date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(regional_trends.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends/resource-types')
def get_resource_trends():
    """Get time series data grouped by resource type"""
    try:
        resource_trends = df.groupby(['date', 'resource_type']).agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()
        
        resource_trends['date'] = resource_trends['date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(resource_trends.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== TAB 3: REGIONAL COMPARISON =====

@app.route('/api/regional/comparison')
def get_regional_comparison():
    """Get regional performance comparison data"""
    try:
        regional_summary = df.groupby('region').agg({
            'usage_cpu': ['mean', 'max', 'min', 'std'],
            'usage_storage': ['mean', 'max', 'min', 'std'],
            'users_active': ['mean', 'max', 'min', 'std']
        }).round(2)
        
        # Flatten column names
        regional_summary.columns = ['_'.join(col).strip() for col in regional_summary.columns]
        regional_summary = regional_summary.reset_index()
        
        return jsonify(regional_summary.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/regional/heatmap')
def get_regional_heatmap():
    """Get data for regional performance heatmap"""
    try:
        heatmap_data = df.groupby(['region', 'resource_type']).agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()
        
        return jsonify(heatmap_data.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/regional/distribution')
def get_regional_distribution():
    """Get regional usage distribution data"""
    try:
        distribution = df.groupby('region').agg({
            'usage_cpu': 'sum',
            'usage_storage': 'sum',
            'users_active': 'sum'
        }).reset_index()
        
        return jsonify(distribution.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== TAB 4: RESOURCE TYPES =====

@app.route('/api/resources/utilization')
def get_resource_utilization():
    """Get resource utilization over time"""
    try:
        resource_util = df.groupby(['date', 'resource_type']).agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()
        
        resource_util['date'] = resource_util['date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(resource_util.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources/distribution')
def get_resource_distribution():
    """Get resource type distribution"""
    try:
        distribution = df.groupby('resource_type').agg({
            'usage_cpu': ['mean', 'sum'],
            'usage_storage': ['mean', 'sum'],
            'users_active': ['mean', 'sum']
        }).reset_index()
        
        # Flatten column names
        distribution.columns = ['_'.join(col).strip() if col[1] else col[0] for col in distribution.columns]
        
        return jsonify(distribution.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources/efficiency')
def get_resource_efficiency():
    """Get resource efficiency metrics"""
    try:
        efficiency = df.groupby('resource_type').agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()
        
        # Calculate efficiency ratios
        efficiency['cpu_per_user'] = efficiency['usage_cpu'] / efficiency['users_active']
        efficiency['storage_per_user'] = efficiency['usage_storage'] / efficiency['users_active']
        
        return jsonify(efficiency.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== TAB 5: CORRELATION ANALYSIS =====

@app.route('/api/correlations/matrix')
def get_correlation_matrix():
    """Get correlation matrix for numeric columns"""
    try:
        numeric_cols = ['usage_cpu', 'usage_storage', 'users_active', 'economic_index', 'cloud_market_demand']
        corr_matrix = df[numeric_cols].corr()
        
        # Convert to format suitable for heatmap
        correlation_data = []
        for i, row_name in enumerate(corr_matrix.index):
            for j, col_name in enumerate(corr_matrix.columns):
                correlation_data.append({
                    'row': row_name,
                    'column': col_name,
                    'correlation': float(corr_matrix.iloc[i, j])
                })
        
        return jsonify(correlation_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlations/scatter')
def get_scatter_data():
    """Get data for scatter plots"""
    try:
        x_axis = request.args.get('x_axis', 'economic_index')
        y_axis = request.args.get('y_axis', 'usage_cpu')
        
        scatter_data = df.groupby('region').apply(
            lambda x: pd.Series({
                'region': x['region'].iloc[0],
                f'{x_axis}_avg': x[x_axis].mean(),
                f'{y_axis}_avg': x[y_axis].mean(),
                'data_points': len(x)
            })
        ).reset_index(drop=True)
        
        return jsonify(scatter_data.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlations/bubble')
def get_bubble_data():
    """Get multi-dimensional bubble chart data"""
    try:
        bubble_data = df.groupby(['region', 'resource_type']).agg({
            'economic_index': 'mean',
            'cloud_market_demand': 'mean',
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()
        
        return jsonify(bubble_data.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== TAB 6: HOLIDAY EFFECTS =====

@app.route('/api/holiday/analysis')
def get_holiday_analysis():
    """Get holiday vs regular day analysis"""
    try:
        holiday_comparison = df.groupby('holiday').agg({
            'usage_cpu': ['mean', 'std', 'count'],
            'usage_storage': ['mean', 'std', 'count'],
            'users_active': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        holiday_comparison.columns = ['_'.join(col).strip() if col[1] else col[0] for col in holiday_comparison.columns]
        
        return jsonify(holiday_comparison.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/holiday/distribution')
def get_holiday_distribution():
    """Get detailed distribution data for holiday analysis"""
    try:
        # Get raw data for box plots and violin plots
        holiday_data = df[df['holiday'] == 1][['usage_cpu', 'usage_storage', 'users_active']].to_dict('records')
        regular_data = df[df['holiday'] == 0][['usage_cpu', 'usage_storage', 'users_active']].to_dict('records')
        
        return jsonify({
            'holiday_data': holiday_data,
            'regular_data': regular_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/holiday/calendar')
def get_calendar_data():
    """Get calendar heatmap data"""
    try:
        df_calendar = df.copy()
        df_calendar['day'] = df_calendar['date'].dt.day
        df_calendar['month'] = df_calendar['date'].dt.month
        df_calendar['month_name'] = df_calendar['date'].dt.strftime('%B')
        
        calendar_data = df_calendar.groupby(['month', 'month_name', 'day']).agg({
            'usage_cpu': 'mean',
            'holiday': 'max'  # 1 if any holiday, 0 otherwise
        }).reset_index()
        
        return jsonify(calendar_data.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== TAB 7: ML FORECASTING (PLACEHOLDER) =====
#-------------------------------------------i have edited down code-------------------------------------------------
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
        region_clean = region.replace(' ', '_')  # underscores instead of spaces

        
        try:
            if model_type == 'ARIMA':
                model_path = f"{MODEL_DIR}{region_clean}_ARIMA_model.pkl"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        loaded_models[region] = pickle.load(f)
                    print(f"✓ Loaded ARIMA model for {region}")
                else:
                    print(f"✗ ARIMA model file not found for {region}: {model_path}")
            
            elif model_type == 'LSTM':
                model_path = f"{MODEL_DIR}{region_clean}_LSTM_model.h5"
                scaler_path = f"{MODEL_DIR}{region_clean}_LSTM_scaler.pkl"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    loaded_models[region] = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
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

#-------------------------------------------i have edited up code-------------------------------------------------

# ===== TAB 8: USER ENGAGEMENT =====

@app.route('/api/engagement/efficiency')
def get_engagement_efficiency():
    """Get user engagement efficiency metrics"""
    try:
        engagement = df.groupby(['region', 'resource_type']).agg({
            'users_active': 'mean',
            'usage_cpu': 'mean',
            'usage_storage': 'mean'
        }).reset_index()
        
        # Calculate efficiency scores
        engagement['cpu_efficiency'] = engagement['users_active'] / engagement['usage_cpu']
        engagement['storage_efficiency'] = engagement['users_active'] / (engagement['usage_storage'] / 100)  # Normalize storage
        engagement['overall_efficiency'] = (engagement['cpu_efficiency'] + engagement['storage_efficiency']) / 2
        
        return jsonify(engagement.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/engagement/trends')
def get_engagement_trends():
    """Get user engagement trends over time"""
    try:
        engagement_trends = df.groupby('date').agg({
            'users_active': 'mean',
            'usage_cpu': 'mean',
            'usage_storage': 'mean'
        }).reset_index()
        
        # Calculate daily efficiency ratios
        engagement_trends['cpu_per_user'] = engagement_trends['usage_cpu'] / engagement_trends['users_active']
        engagement_trends['storage_per_user'] = engagement_trends['usage_storage'] / engagement_trends['users_active']
        
        engagement_trends['date'] = engagement_trends['date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(engagement_trends.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/engagement/bubble')
def get_engagement_bubble():
    """Get bubble chart data for user engagement analysis"""
    try:
        bubble_data = df.groupby(['region', 'resource_type']).agg({
            'users_active': 'mean',
            'usage_cpu': 'mean',
            'usage_storage': 'mean'
        }).reset_index()
        
        return jsonify(bubble_data.to_dict('records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== UTILITY ENDPOINTS =====

@app.route('/api/filters/options')
def get_filter_options():
    """Get available filter options for dropdowns"""
    try:
        options = {
            'regions': sorted(df['region'].unique().tolist()),
            'resource_types': sorted(df['resource_type'].unique().tolist()),
            'date_range': {
                'min_date': df['date'].min().isoformat(),
                'max_date': df['date'].max().isoformat()
            },
            'metrics': ['usage_cpu', 'usage_storage', 'users_active', 'economic_index', 'cloud_market_demand']
        }
        
        return jsonify(options)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/summary')
def get_data_summary():
    """Get dataset summary statistics"""
    try:
        numeric_cols = ['usage_cpu', 'usage_storage', 'users_active', 'economic_index', 'cloud_market_demand']
        summary = df[numeric_cols].describe().to_dict()
        
        # Add data info
        summary['dataset_info'] = {
            'total_records': len(df),
            'date_range_days': (df['date'].max() - df['date'].min()).days,
            'regions_count': df['region'].nunique(),
            'resource_types_count': df['resource_type'].nunique(),
            'holiday_records': int(df['holiday'].sum()),
            'regular_records': int(len(df) - df['holiday'].sum())
        }
        
        return jsonify(summary)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'available_endpoints': [
        '/api/kpis', '/api/sparklines', '/api/time-series', '/api/trends/regional',
        '/api/regional/comparison', '/api/resources/utilization', '/api/correlations/matrix',
        '/api/holiday/analysis', '/api/engagement/efficiency', '/api/filters/options',
        '/api/forecast/models', '/api/forecast/predict', '/api/forecast/comparison'  # Add these
    ]}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Azure Demand Forecasting API Server Starting...")
    print("📊 Available Endpoints:")
    print("   • (Overview): /api/kpis, /api/sparklines")
    print("   •  (Trends): /api/time-series, /api/trends/*")
    print("   •  (Regional): /api/regional/*")
    print("   •  (Resources): /api/resources/*")
    print("   •  (Correlations): /api/correlations/*")
    print("   •  (Holidays): /api/holiday/*")
    print("   •  (Forecasting): /api/forecast/placeholder")
    print("   •  (Engagement): /api/engagement/*")
    print("   • Utilities: /api/filters/options, /api/data/summary")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
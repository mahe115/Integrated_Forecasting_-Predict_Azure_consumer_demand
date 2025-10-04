# COMPLETE ULTRA-FAST API - WINDOWS COMPATIBLE WITH INTELLIGENT TRAINING
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
import os
import sys
from functools import lru_cache, wraps
from multiprocessing import Pool, Manager, Process, Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import time
import json
from threading import RLock
import pickle

# Add these imports at the top
import sqlite3

# Optimize imports and suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ML libraries - lazy imported to speed up startup
def lazy_import_ml():
    global load_model, MeanSquaredError, ARIMA, MinMaxScaler, tf
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.metrics import MeanSquaredError

        # Configure TensorFlow for production (Windows compatible)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)

        return True
    except ImportError as e:
        print(f"Warning: ML libraries not available: {e}")
        return False

app = Flask(__name__)
CORS(app)

# Global thread pool for concurrent operations
executor = ThreadPoolExecutor(max_workers=6)  # Reduced for Windows

# Simple in-memory cache (Windows compatible)
cache_dict = {}
cache_times = {}
cache_lock = RLock()

# Configuration
CACHE_TIMEOUT = {
    'fast': 120,      # 2 minutes for frequently changing data
    'medium': 600,    # 10 minutes for moderately stable data
    'slow': 1800,     # 30 minutes for stable data
    'forecast': 300   # 5 minutes for ML forecasts
}

# ===== ULTRA-FAST DATA LOADING =====

def load_data_optimized():
    """Ultra-fast data loading with memory optimization"""
    print("⚡ Fast-loading datasets...")
    start_time = time.time()

    try:
        # Read with optimized dtypes and minimal parsing
        df = pd.read_csv('data/processed/cleaned_merged.csv',
                        parse_dates=['date'],
                        dtype={
                            'usage_cpu': 'float32',
                            'usage_storage': 'float32', 
                            'users_active': 'int32',
                            'economic_index': 'float32',
                            'cloud_market_demand': 'float32',
                            'holiday': 'int8',
                            'region': 'category',
                            'resource_type': 'category'
                        },
                        engine='c',  # Use C engine for speed
                        low_memory=False)

        # Pre-compute all derived columns in vectorized operations
        df['month'] = df['date'].dt.month.astype('int8')
        df['quarter'] = df['date'].dt.quarter.astype('int8')
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype('int8')

        load_time = time.time() - start_time
        print(f"✅ Loaded {len(df):,} records in {load_time:.2f}s")
        return df

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        raise

# Load data at module level
df = load_data_optimized()

# Pre-compute expensive aggregations
def precompute_aggregations():
    """Pre-compute common aggregations in parallel"""
    print("🔄 Pre-computing aggregations...")

    def compute_regional_daily():
        return df.groupby(['region', 'date']).agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean', 
            'users_active': 'sum',
            'economic_index': 'first',
            'cloud_market_demand': 'first',
            'holiday': 'max'
        }).reset_index()

    def compute_region_dfs(region_daily):
        region_dfs = {}
        for region in region_daily['region'].unique():
            region_data = region_daily[region_daily['region'] == region].copy()
            region_data = region_data.drop('region', axis=1).set_index('date').sort_index()
            region_dfs[region] = region_data
        return region_dfs

    def compute_common_stats():
        return {
            'peak_cpu_idx': df['usage_cpu'].idxmax(),
            'max_storage_idx': df['usage_storage'].idxmax(),
            'peak_users_idx': df['users_active'].idxmax(),
            'holiday_mask': df['holiday'] == 1,
            'date_range': {
                'min': df['date'].min(),
                'max': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            }
        }

    # Execute computations in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_regional = executor.submit(compute_regional_daily)
        future_stats = executor.submit(compute_common_stats)

        regional_daily = future_regional.result()
        common_stats = future_stats.result()

        future_region_dfs = executor.submit(compute_region_dfs, regional_daily)
        region_dfs = future_region_dfs.result()

    return regional_daily, region_dfs, common_stats

# Pre-compute data
region_daily, region_dfs, common_stats = precompute_aggregations()
print("✅ Pre-computation complete")
print(f"📊 Regional data available for: {list(region_dfs.keys())}")

# ===== WINDOWS-COMPATIBLE CACHE SYSTEM =====

def windows_cache(cache_type='medium'):
    """Windows-compatible cache decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            current_time = time.time()
            timeout = CACHE_TIMEOUT[cache_type]

            # Check cache with minimal locking
            with cache_lock:
                if (cache_key in cache_dict and 
                    cache_key in cache_times and 
                    current_time - cache_times[cache_key] < timeout):
                    return cache_dict[cache_key]

            # Execute function
            result = func(*args, **kwargs)

            # Update cache
            with cache_lock:
                cache_dict[cache_key] = result
                cache_times[cache_key] = current_time

                # Cleanup old entries (every 100 entries)
                if len(cache_dict) > 100:
                    oldest_key = min(cache_times.keys(), key=cache_times.get)
                    del cache_dict[oldest_key]
                    del cache_times[oldest_key]

            return result
        return wrapper
    return decorator

# ===== MODEL CONFIGURATION =====

# Initialize intelligent pipeline after data loading
try:
    from model_training_pipeline import ModelTrainingPipeline
    intelligent_pipeline = ModelTrainingPipeline()
    print("🤖 Intelligent training pipeline connected")
except ImportError as e:
    print(f"⚠️ Warning: Intelligent pipeline not available: {e}")
    intelligent_pipeline = None

# Use intelligent pipeline models if available, fallback to static config
if intelligent_pipeline:
    FINAL_SELECTION = intelligent_pipeline.CPU_MODELS
    FINAL_SELECTION_USERS = intelligent_pipeline.USERS_MODELS
else:
    FINAL_SELECTION = {
        'East US': 'LSTM',
        'North Europe': 'ARIMA', 
        'Southeast Asia': 'LSTM',
        'West US': 'LSTM'
    }
    FINAL_SELECTION_USERS = {
        'East US':        'LSTM',
        'North Europe':   'XGBoost',
        'Southeast Asia': 'ARIMA',
        'West US':        'XGBoost',
    }

MODEL_DIR = 'models/'
loaded_models = {}
loaded_scalers = {}
ml_available = False

def load_single_model(region, model_type):
    """Load a single model (Windows compatible)"""
    try:
        region_clean = region.replace(' ', '_')

        if model_type == 'ARIMA':
            model_path = f"{MODEL_DIR}{region_clean}_ARIMA_model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return region, 'ARIMA', model, None

        elif model_type == 'LSTM':
            model_path = f"{MODEL_DIR}{region_clean}_LSTM_model.h5"
            scaler_path = f"{MODEL_DIR}{region_clean}_LSTM_scaler.pkl"

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                if lazy_import_ml():
                    model = load_model(model_path, 
                                     custom_objects={'mse': MeanSquaredError()},
                                     compile=False)
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    return region, 'LSTM', model, scaler

        return region, model_type, None, None

    except Exception as e:
        print(f"Model loading error for {region}: {e}")
        return region, model_type, None, None

def load_models_threaded():
    """Load all models using threading (Windows compatible)"""
    global loaded_models, loaded_scalers, ml_available

    print("🔄 Loading ML models in parallel...")
    start_time = time.time()

    # Try to import ML libraries first
    ml_available = lazy_import_ml()
    if not ml_available:
        print("⚠️ ML libraries not available, skipping model loading")
        return

    # Load models in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(load_single_model, region, model_type)
            for region, model_type in FINAL_SELECTION.items()
        ]

        for future in as_completed(futures):
            try:
                region, model_type, model, scaler = future.result(timeout=15)
                if model is not None:
                    loaded_models[region] = model
                    if scaler is not None:
                        loaded_scalers[region] = scaler
                    print(f"✅ Loaded {model_type} model for {region}")
                else:
                    print(f"❌ Failed to load {model_type} model for {region}")
            except Exception as e:
                print(f"❌ Model loading error: {e}")

    load_time = time.time() - start_time
    print(f"🏁 Model loading completed in {load_time:.2f}s")
    print(f"📈 Models loaded: {list(loaded_models.keys())}")
    print(f"🔧 Scalers loaded: {list(loaded_scalers.keys())}")

# Start model loading in background thread
model_loading_thread = threading.Thread(target=load_models_threaded, daemon=True)
model_loading_thread.start()

# ===== ALL OTHER ENDPOINTS (CACHED) =====

@app.route('/api/kpis')
@windows_cache('medium')
def get_kpis():
    try:
        peak_cpu_idx = common_stats['peak_cpu_idx']
        max_storage_idx = common_stats['max_storage_idx']
        peak_users_idx = common_stats['peak_users_idx']
        holiday_mask = common_stats['holiday_mask']

        holiday_avg = df.loc[holiday_mask, 'usage_cpu'].mean()
        regular_avg = df.loc[~holiday_mask, 'usage_cpu'].mean()
        holiday_impact = ((holiday_avg - regular_avg) / regular_avg) * 100 if regular_avg > 0 else 0

        kpis = {
            'peak_cpu': float(df.loc[peak_cpu_idx, 'usage_cpu']),
            'peak_cpu_details': {
                'date': df.loc[peak_cpu_idx, 'date'].isoformat(),
                'region': str(df.loc[peak_cpu_idx, 'region']),
                'resource_type': str(df.loc[peak_cpu_idx, 'resource_type'])
            },
            'max_storage': float(df.loc[max_storage_idx, 'usage_storage']),
            'max_storage_details': {
                'date': df.loc[max_storage_idx, 'date'].isoformat(),
                'region': str(df.loc[max_storage_idx, 'region']),
                'resource_type': str(df.loc[max_storage_idx, 'resource_type'])
            },
            'peak_users': int(df.loc[peak_users_idx, 'users_active']),
            'peak_users_details': {
                'date': df.loc[peak_users_idx, 'date'].isoformat(),
                'region': str(df.loc[peak_users_idx, 'region']),
                'resource_type': str(df.loc[peak_users_idx, 'resource_type'])
            },
            'avg_cpu': float(df['usage_cpu'].mean()),
            'avg_storage': float(df['usage_storage'].mean()),
            'avg_users': float(df['users_active'].mean()),
            'total_regions': int(df['region'].nunique()),
            'total_resource_types': int(df['resource_type'].nunique()),
            'data_points': int(len(df)),
            'date_range': {
                'start': common_stats['date_range']['min'].isoformat(),
                'end': common_stats['date_range']['max'].isoformat(),
                'days': common_stats['date_range']['days']
            },
            'holiday_impact': {
                'percentage': float(holiday_impact),
                'holiday_avg_cpu': float(holiday_avg),
                'regular_avg_cpu': float(regular_avg)
            }
        }

        return jsonify(kpis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sparklines')
@windows_cache('fast')
def get_sparklines():
    try:
        latest_date = df['date'].max()
        cutoff_date = latest_date - timedelta(days=30)
        mask = df['date'] > cutoff_date
        last_30_days = df[mask]

        daily_trends = last_30_days.groupby('date').agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()

        daily_trends['date'] = daily_trends['date'].dt.strftime('%Y-%m-%d')

        sparklines = {
            'cpu_trend': daily_trends[['date', 'usage_cpu']].to_dict('records'),
            'storage_trend': daily_trends[['date', 'usage_storage']].to_dict('records'),
            'users_trend': daily_trends[['date', 'users_active']].to_dict('records')
        }

        return jsonify(sparklines)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/data/raw')
@windows_cache('slow')
def get_data_raw():
    return jsonify(df.to_dict('records'))

@app.route('/api/time-series')
@windows_cache('fast')
def get_time_series():
    try:
        region_filter = request.args.get('region')
        resource_filter = request.args.get('resource_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        data = df

        if region_filter:
            data = data[data['region'] == region_filter]
        if resource_filter:
            data = data[data['resource_type'] == resource_filter]
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]

        time_series = data.groupby('date', sort=False).agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean',
            'economic_index': 'mean',
            'cloud_market_demand': 'mean'
        }).reset_index()

        time_series['date'] = time_series['date'].dt.strftime('%Y-%m-%d')
        return jsonify(time_series.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add all other endpoints with caching (keeping them brief for space)

@app.route('/api/trends/regional')
@windows_cache('fast')
def get_regional_trends():
    try:
        regional_trends = df.groupby(['date', 'region'], sort=False).agg({
            'usage_cpu': 'mean', 'usage_storage': 'mean', 'users_active': 'mean'
        }).reset_index()
        regional_trends['date'] = regional_trends['date'].dt.strftime('%Y-%m-%d')
        return jsonify(regional_trends.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends/resource-types')
@windows_cache('fast')
def get_resource_trends():
    try:
        resource_trends = df.groupby(['date', 'resource_type'], sort=False).agg({
            'usage_cpu': 'mean', 'usage_storage': 'mean', 'users_active': 'mean'
        }).reset_index()
        resource_trends['date'] = resource_trends['date'].dt.strftime('%Y-%m-%d')
        return jsonify(resource_trends.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/regional/comparison')
@windows_cache('medium')
def get_regional_comparison():
    try:
        regional_summary = df.groupby('region', sort=False).agg({
            'usage_cpu': ['mean', 'max', 'min', 'std'],
            'usage_storage': ['mean', 'max', 'min', 'std'],
            'users_active': ['mean', 'max', 'min', 'std']
        }).round(2)
        regional_summary.columns = ['_'.join(col).strip() for col in regional_summary.columns]
        return jsonify(regional_summary.reset_index().to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/regional/heatmap')
@windows_cache('medium')
def get_regional_heatmap():
    try:
        heatmap_data = df.groupby(['region', 'resource_type'], sort=False).agg({
            'usage_cpu': 'mean', 'usage_storage': 'mean', 'users_active': 'mean'
        }).reset_index()
        return jsonify(heatmap_data.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/regional/distribution')
@windows_cache('medium')
def get_regional_distribution():
    try:
        distribution = df.groupby('region', sort=False).agg({
            'usage_cpu': 'sum', 'usage_storage': 'sum', 'users_active': 'sum'
        }).reset_index()
        return jsonify(distribution.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources/utilization')
@windows_cache('fast')
def get_resource_utilization():
    try:
        resource_util = df.groupby(['date', 'resource_type'], sort=False).agg({
            'usage_cpu': 'mean', 'usage_storage': 'mean', 'users_active': 'mean'
        }).reset_index()
        resource_util['date'] = resource_util['date'].dt.strftime('%Y-%m-%d')
        return jsonify(resource_util.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources/distribution')
@windows_cache('medium')
def get_resource_distribution():
    try:
        distribution = df.groupby('resource_type', sort=False).agg({
            'usage_cpu': ['mean', 'sum'], 'usage_storage': ['mean', 'sum'], 'users_active': ['mean', 'sum']
        }).reset_index()
        distribution.columns = ['_'.join(col).strip() if col[1] else col[0] for col in distribution.columns]
        return jsonify(distribution.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources/efficiency')
@windows_cache('medium')
def get_resource_efficiency():
    try:
        efficiency = df.groupby('resource_type', sort=False).agg({
            'usage_cpu': 'mean', 'usage_storage': 'mean', 'users_active': 'mean'
        }).reset_index()
        efficiency['cpu_per_user'] = efficiency['usage_cpu'] / efficiency['users_active']
        efficiency['storage_per_user'] = efficiency['usage_storage'] / efficiency['users_active']
        return jsonify(efficiency.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlations/matrix')
@windows_cache('medium')
def get_correlation_matrix():
    try:
        numeric_cols = ['usage_cpu', 'usage_storage', 'users_active', 'economic_index', 'cloud_market_demand']
        corr_matrix = df[numeric_cols].corr(method='pearson')
        correlation_data = []
        for i, row_name in enumerate(corr_matrix.index):
            for j, col_name in enumerate(corr_matrix.columns):
                correlation_data.append({
                    'row': row_name, 'column': col_name, 'correlation': float(corr_matrix.iloc[i, j])
                })
        return jsonify(correlation_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlations/scatter')
@windows_cache('fast')
def get_scatter_data():
    try:
        x_axis = request.args.get('x_axis', 'economic_index')
        y_axis = request.args.get('y_axis', 'usage_cpu')
        scatter_data = df.groupby('region', sort=False).agg({
            x_axis: 'mean', y_axis: 'mean', 'region': 'first'
        }).reset_index(drop=True)
        scatter_data = scatter_data.rename(columns={x_axis: f'{x_axis}_avg', y_axis: f'{y_axis}_avg'})
        scatter_data['data_points'] = df.groupby('region').size().values
        return jsonify(scatter_data.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlations/bubble')
@windows_cache('medium')
def get_bubble_data():
    """Generate bubble chart data with meaningful variation"""
    try:
        bubble_data = df.groupby(['region', 'resource_type'], sort=False).agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean',
            'users_active': 'mean'
        }).reset_index()
        
        # Create meaningful metrics that actually vary
        # X-axis: CPU Efficiency (CPU per user) - Lower is better
        bubble_data['cpu_efficiency'] = bubble_data['usage_cpu'] / bubble_data['users_active']
        
        # Y-axis: Storage Efficiency (Storage per user) - Lower is better  
        bubble_data['storage_efficiency'] = bubble_data['usage_storage'] / bubble_data['users_active']
        
        # Bubble size: Total resource utilization
        bubble_data['total_utilization'] = bubble_data['usage_cpu'] + (bubble_data['usage_storage'] / 20)
        
        return jsonify(bubble_data.to_dict('records'))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/holiday/analysis')
@windows_cache('medium')
def get_holiday_analysis():
    try:
        holiday_comparison = df.groupby('holiday', sort=False).agg({
            'usage_cpu': ['mean', 'std', 'count'],
            'usage_storage': ['mean', 'std', 'count'],
            'users_active': ['mean', 'std', 'count']
        }).reset_index()
        holiday_comparison.columns = ['_'.join(col).strip() if col[1] else col[0] for col in holiday_comparison.columns]
        return jsonify(holiday_comparison.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/holiday/distribution')
@windows_cache('medium')
def get_holiday_distribution():
    try:
        holiday_mask = df['holiday'] == 1
        holiday_data = df[holiday_mask][['usage_cpu', 'usage_storage', 'users_active']].to_dict('records')
        regular_data = df[~holiday_mask][['usage_cpu', 'usage_storage', 'users_active']].to_dict('records')
        return jsonify({'holiday_data': holiday_data, 'regular_data': regular_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/holiday/calendar')
@windows_cache('medium')
def get_calendar_data():
    try:
        df_temp = df.copy()
        df_temp['day'] = df_temp['date'].dt.day
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['month_name'] = df_temp['date'].dt.strftime('%B')
        calendar_data = df_temp.groupby(['month', 'month_name', 'day'], sort=False).agg({
            'usage_cpu': 'mean', 'holiday': 'max'
        }).reset_index()
        return jsonify(calendar_data.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/engagement/efficiency')
@windows_cache('medium')
def get_engagement_efficiency():
    try:
        engagement = df.groupby(['region', 'resource_type'], sort=False).agg({
            'users_active': 'mean', 'usage_cpu': 'mean', 'usage_storage': 'mean'
        }).reset_index()
        engagement['cpu_efficiency'] = engagement['users_active'] / engagement['usage_cpu']
        engagement['storage_efficiency'] = engagement['users_active'] / (engagement['usage_storage'] / 100)
        engagement['overall_efficiency'] = (engagement['cpu_efficiency'] + engagement['storage_efficiency']) / 2
        return jsonify(engagement.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/engagement/trends')
@windows_cache('fast')
def get_engagement_trends():
    try:
        engagement_trends = df.groupby('date', sort=False).agg({
            'users_active': 'mean', 'usage_cpu': 'mean', 'usage_storage': 'mean'
        }).reset_index()
        engagement_trends['cpu_per_user'] = engagement_trends['usage_cpu'] / engagement_trends['users_active']
        engagement_trends['storage_per_user'] = engagement_trends['usage_storage'] / engagement_trends['users_active']
        engagement_trends['date'] = engagement_trends['date'].dt.strftime('%Y-%m-%d')
        return jsonify(engagement_trends.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/engagement/bubble')
@windows_cache('medium')
def get_engagement_bubble():
    try:
        bubble_data = df.groupby(['region', 'resource_type'], sort=False).agg({
            'users_active': 'mean', 'usage_cpu': 'mean', 'usage_storage': 'mean'
        }).reset_index()
        return jsonify(bubble_data.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filters/options')
@windows_cache('slow')
def get_filter_options():
    try:
        options = {
            'regions': sorted(df['region'].cat.categories.tolist()),
            'resource_types': sorted(df['resource_type'].cat.categories.tolist()),
            'date_range': {
                'min_date': common_stats['date_range']['min'].isoformat(),
                'max_date': common_stats['date_range']['max'].isoformat()
            },
            'metrics': ['usage_cpu', 'usage_storage', 'users_active', 'economic_index', 'cloud_market_demand']
        }
        return jsonify(options)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/summary')
@windows_cache('slow')
def get_data_summary():
    try:
        numeric_cols = ['usage_cpu', 'usage_storage', 'users_active', 'economic_index', 'cloud_market_demand']
        summary = df[numeric_cols].describe().to_dict()
        summary['dataset_info'] = {
            'total_records': len(df),
            'date_range_days': common_stats['date_range']['days'],
            'regions_count': df['region'].nunique(),
            'resource_types_count': df['resource_type'].nunique(),
            'holiday_records': int(df['holiday'].sum()),
            'regular_records': int(len(df) - df['holiday'].sum())
        }
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== FIXED FORECASTING ENDPOINTS (NO CACHING) =====

@app.route('/api/forecast/models')
@windows_cache('fast')
def get_available_models():
    """Get current best models (dynamically updated by intelligent pipeline)"""
    try:
        model_info = {}
        
        # Use current best CPU models
        for region, model_type in FINAL_SELECTION.items():
            model_info[region] = {
                'model_type': model_type,
                'loaded': region in loaded_models,
                'has_scaler': region in loaded_scalers if model_type == 'LSTM' else None,
                'selection_method': 'intelligent_auto_selection' if intelligent_pipeline else 'static',
                'last_updated': 'dynamic' if intelligent_pipeline else 'static'
            }
        
        return jsonify({
            'models': model_info,
            'total_regions': len(FINAL_SELECTION),
            'model_types_used': list(set(FINAL_SELECTION.values())),
            'ml_available': ml_available,
            'selection_method': 'intelligent_auto_selection' if intelligent_pipeline else 'static'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/predict')
def generate_forecasts():
    """Generate forecasts with proper region filtering (NO CACHING for dynamic results)"""
    try:
        forecast_days = min(int(request.args.get('days', 30)), 90)
        region_filter = request.args.get('region', None)

        if not ml_available:
            return jsonify({'error': 'ML libraries not available'}), 503

        # Determine regions to process
        if region_filter and region_filter != "All Regions":
            regions_to_process = [region_filter]
        else:
            regions_to_process = list(FINAL_SELECTION.keys())

        print(f"🔮 Generating forecasts for: {regions_to_process} ({forecast_days} days)")

        args_list = [(region, FINAL_SELECTION[region], forecast_days) for region in regions_to_process]
        results = {}

        # Use threading for multiple regions (Windows compatible)
        if len(args_list) > 1:
            print("📊 Processing multiple regions in parallel...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(threaded_forecast_worker, args): args[0] for args in args_list}

                for future in as_completed(futures):
                    try:
                        region, forecast_data = future.result(timeout=30)  # 30 second timeout per region
                        forecast_data['model_type'] = FINAL_SELECTION[region]
                        results[region] = forecast_data
                        print(f"✅ Completed forecast for {region}")
                    except Exception as e:
                        print(f"❌ Error forecasting {region}: {e}")
                        results[region] = {'error': f'Forecasting failed: {str(e)}'}
        else:
            # Single region processing
            print(f"📈 Processing single region: {regions_to_process[0]}")
            region, forecast_data = threaded_forecast_worker(args_list[0])
            forecast_data['model_type'] = FINAL_SELECTION[region]
            results[region] = forecast_data
            print(f"✅ Completed forecast for {region}")

        print(f"🎯 Forecast generation complete. Returning {len(results)} results.")
        return jsonify(results)

    except Exception as e:
        print(f"💥 Forecast generation error: {e}")
        return jsonify({'error': str(e)}), 500

def threaded_forecast_worker(args):
    """Worker function for threaded forecasting with better error handling"""
    region, model_type, forecast_days = args

    print(f"🔧 Worker processing: {region} ({model_type})")

    if not ml_available:
        return region, {'error': f'ML libraries not available'}

    if region not in loaded_models:
        return region, {'error': f'Model not loaded for {region}'}

    if region not in region_dfs:
        return region, {'error': f'Regional data not available for {region}'}

    try:
        model = loaded_models[region]
        region_data = region_dfs[region]

        print(f"📊 {region}: Using {model_type} model, data shape: {region_data.shape}")

        if model_type == 'ARIMA':
            result = generate_arima_forecast_fast(model, region_data, forecast_days)
        elif model_type == 'LSTM':
            if region not in loaded_scalers:
                return region, {'error': f'LSTM scaler not loaded for {region}'}
            scaler = loaded_scalers[region]
            result = generate_lstm_forecast_fast(model, scaler, region_data, forecast_days)
        else:
            return region, {'error': f'Unknown model type: {model_type}'}

        # Add region identifier to result for verification
        result['region'] = region
        result['generated_at'] = datetime.now().isoformat()

        print(f"✅ {region}: Generated {len(result.get('predicted_cpu', []))} predictions")
        return region, result

    except Exception as e:
        print(f"💥 {region}: Forecasting error - {str(e)}")
        return region, {'error': f'Forecasting error: {str(e)}'}

def generate_arima_forecast_fast(model, region_data, forecast_days):
    """Generate ARIMA forecast with region-specific data"""
    try:
        print(f"🔍 ARIMA: Processing {len(region_data)} data points for {forecast_days} day forecast")

        last_date = region_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')

        # Generate forecast
        forecast = model.forecast(steps=forecast_days)
        print(f"📈 ARIMA: Generated forecast range {min(forecast):.2f} - {max(forecast):.2f}")

        # Get recent historical data for context
        recent_data = region_data['usage_cpu'].tail(14)

        result = {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'predicted_cpu': [float(f) for f in forecast],
            'model_info': {
                'type': 'ARIMA',
                'forecast_horizon': forecast_days,
                'data_points_used': len(region_data)
            },
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in recent_data.index],
                'actual_cpu': recent_data.values.tolist()
            }
        }

        return result

    except Exception as e:
        print(f"💥 ARIMA forecast error: {str(e)}")
        return {'error': f'ARIMA forecast error: {str(e)}'}

def generate_lstm_forecast_fast(model, scaler, region_data, forecast_days):
    """Generate LSTM forecast with region-specific data"""
    try:
        print(f"🔍 LSTM: Processing {len(region_data)} data points for {forecast_days} day forecast")

        sequence_length = 7
        cpu_data = region_data['usage_cpu'].values

        # Scale the data
        scaled_data = scaler.transform(cpu_data.reshape(-1, 1))

        # Prepare the last sequence for prediction
        current_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        forecasts_scaled = []

        print(f"🧠 LSTM: Starting iterative predictions...")

        # Generate forecasts iteratively
        for i in range(forecast_days):
            next_pred = model.predict(current_sequence, verbose=0)
            forecasts_scaled.append(next_pred[0, 0])

            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]

            if (i + 1) % 10 == 0:  # Progress indicator
                print(f"🔄 LSTM: Generated {i+1}/{forecast_days} predictions")

        # Inverse transform predictions
        forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1))
        print(f"📈 LSTM: Generated forecast range {min(forecasts)[0]:.2f} - {max(forecasts)[0]:.2f}")

        # Generate future dates
        last_date = region_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')

        # Get recent historical data for context
        recent_data = region_data['usage_cpu'].tail(14)

        result = {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'predicted_cpu': [float(f[0]) for f in forecasts],
            'model_info': {
                'type': 'LSTM',
                'sequence_length': sequence_length,
                'forecast_horizon': forecast_days,
                'data_points_used': len(region_data)
            },
            'historical': {
                'dates': [d.strftime('%Y-%m-%d') for d in recent_data.index],
                'actual_cpu': recent_data.values.tolist()
            }
        }

        return result

    except Exception as e:
        print(f"💥 LSTM forecast error: {str(e)}")
        return {'error': f'LSTM forecast error: {str(e)}'}

# ===== USERS MODEL SECTION =====

# User model directory - CORRECTED PATH
USER_MODEL_DIR = r'D:\infosysspringboard projects\project1-1stmilestine\AZURE_BACKEND_TEAM-B\notebooks\users_active_forecasting_models\\'

loaded_user_models   = {}
loaded_user_scalers  = {}

def load_single_user_model(region, model_type):
    """Load ACTIVE-USERS model for one region based on actual directory structure."""
    try:
        # Convert region name to match file naming convention
        region_clean = region.replace(' ', '')  # Remove spaces: "East US" -> "EastUS"
        
        if model_type == 'ARIMA':
            m_path = f"{USER_MODEL_DIR}{region_clean}_ARIMA_users.pkl"
            print(f"Looking for ARIMA model at: {m_path}")
            if os.path.exists(m_path):
                with open(m_path, 'rb') as f:
                    model = pickle.load(f)
                return region, 'ARIMA', model, None
            else:
                print(f"ARIMA model file not found: {m_path}")
                
        elif model_type == 'XGBoost':
            m_path = f"{USER_MODEL_DIR}{region_clean}_XGBoost_users.pkl"
            print(f"Looking for XGBoost model at: {m_path}")
            if os.path.exists(m_path):
                with open(m_path, 'rb') as f:
                    model = pickle.load(f)
                return region, 'XGBoost', model, None
            else:
                print(f"XGBoost model file not found: {m_path}")
                
        elif model_type == 'LSTM':
            m_path = f"{USER_MODEL_DIR}{region_clean}_LSTM_users.h5"
            s_path = f"{USER_MODEL_DIR}{region_clean}_scaler_users.pkl"
            print(f"Looking for LSTM model at: {m_path}")
            print(f"Looking for LSTM scaler at: {s_path}")
            
            if os.path.exists(m_path) and os.path.exists(s_path):
                if lazy_import_ml():
                    model = load_model(m_path,
                                       custom_objects={'mse': MeanSquaredError()},
                                       compile=False)
                    with open(s_path, 'rb') as f:
                        scaler = pickle.load(f)
                    return region, 'LSTM', model, scaler
                else:
                    print("TensorFlow not available for LSTM loading")
            else:
                print(f"LSTM model or scaler file not found: {m_path}, {s_path}")
                
        return region, model_type, None, None
    except Exception as e:
        print(f"User-model load error for {region}: {e}")
        return region, model_type, None, None

def load_user_models_threaded():
    """Parallel loading for user-forecast models."""
    print("Starting user model loading...")
    print(f"User model directory: {USER_MODEL_DIR}")
    
    if not lazy_import_ml():
        print("ML libraries not available for user model loading")
        return
        
    # Check if directory exists
    if not os.path.exists(USER_MODEL_DIR):
        print(f"❌ User model directory does not exist: {USER_MODEL_DIR}")
        return
        
    # List files in directory for debugging
    try:
        files = os.listdir(USER_MODEL_DIR)
        print(f"Files found in user model directory: {files}")
    except Exception as e:
        print(f"Error listing directory contents: {e}")
        
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(load_single_user_model, r, t)
                   for r, t in FINAL_SELECTION_USERS.items()]
        for fut in as_completed(futures):
            region, mtype, model, scaler = fut.result()
            if model is not None:
                loaded_user_models[region]  = model
                if scaler is not None:
                    loaded_user_scalers[region] = scaler
                print(f"✅ User {mtype} model loaded for {region}")
            else:
                print(f"❌ User model missing for {region} ({mtype})")

# kick-off async load
threading.Thread(target=load_user_models_threaded, daemon=True).start()

# -------- FIXED GENERIC HELPERS (Matching Training Feature Engineering) --------
def _arima_generic(model, series, forecast_days):
    """ARIMA forecasting - unchanged as it works correctly."""
    fc   = model.forecast(steps=forecast_days)
    fmin = float(np.min(fc)); fmax = float(np.max(fc))
    return fc, fmin, fmax

def _lstm_generic(model, scaler, series, forecast_days):
    """LSTM forecasting - using optimized sequence length from notebook."""
    # Use sequence length based on optimization results
    seq_len = 21  # East US uses 21 steps based on notebook optimization
    
    # Ensure we have enough data
    if len(series) < seq_len:
        seq_len = min(7, len(series))
        
    data = scaler.transform(series.values.reshape(-1, 1))
    seq = data[-seq_len:].reshape(1, seq_len, 1)
    preds = []
    
    for _ in range(forecast_days):
        nxt = model.predict(seq, verbose=0)[0, 0]
        preds.append(nxt)
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = nxt
    
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds_inv, float(preds_inv.min()), float(preds_inv.max())

def _xgboost_generic(model, region_data, forecast_days):
    """
    PROPERLY FIXED XGBoost forecasting with correct recursive feature updates
    """
    try:
        # Create feature DataFrame matching training exactly
        df_xgb = region_data.copy()
        
        # Add lag features EXACTLY as in training
        df_xgb['lag1_users'] = df_xgb['users_active'].shift(1)
        df_xgb['lag7_users'] = df_xgb['users_active'].shift(7)
        df_xgb['lag1_cpu'] = df_xgb['usage_cpu'].shift(1)
        df_xgb['lag1_storage'] = df_xgb['usage_storage'].shift(1)
        df_xgb['roll7_mean_users'] = df_xgb['users_active'].rolling(7).mean()
        df_xgb['dow'] = df_xgb.index.dayofweek  # Day of week
        
        # Drop NaN rows created by lagging
        df_xgb = df_xgb.dropna()
        
        if len(df_xgb) == 0:
            print("Warning: No data left after feature engineering")
            return np.full(forecast_days, 1000.0), 1000.0, 1000.0
        
        # Feature columns EXACTLY as in training
        features = ['lag1_users', 'lag7_users', 'roll7_mean_users',
                   'lag1_cpu', 'lag1_storage', 'usage_cpu', 'usage_storage',
                   'economic_index', 'dow', 'holiday']
        
        # Verify all features exist
        available_features = [col for col in features if col in df_xgb.columns]
        print(f"Available features for XGBoost: {available_features}")
        
        # Get recent history for building proper features during forecasting
        recent_users = region_data['users_active'].tail(30).values  # Last 30 days
        recent_cpu = region_data['usage_cpu'].tail(10).values       # Last 10 days for lag features
        recent_storage = region_data['usage_storage'].tail(10).values
        
        # Get last known values for static features
        last_economic = region_data['economic_index'].iloc[-1]
        last_holiday = region_data['holiday'].iloc[-1]  # Will be updated for future dates
        last_date = region_data.index[-1]
        
        forecasts = []
        
        # For each forecast day - PROPERLY RECURSIVE
        for i in range(forecast_days):
            # Calculate current date
            current_date = last_date + pd.Timedelta(days=i+1)
            current_dow = current_date.dayofweek
            
            # Build features for this prediction step
            if i == 0:
                # First prediction - use actual historical data
                lag1_users = recent_users[-1]  # Yesterday's users
                lag7_users = recent_users[-7] if len(recent_users) >= 7 else recent_users[0]
                roll7_mean = np.mean(recent_users[-7:]) if len(recent_users) >= 7 else np.mean(recent_users)
                lag1_cpu = recent_cpu[-1]
                lag1_storage = recent_storage[-1]
            else:
                # Subsequent predictions - use combination of historical and predicted values
                if i >= 7:
                    lag7_users = forecasts[i-7]  # Use our own prediction from 7 days ago
                else:
                    lag7_users = recent_users[-(7-i)] if len(recent_users) >= (7-i) else recent_users[0]
                
                lag1_users = forecasts[i-1]  # Yesterday's prediction
                
                # Rolling mean using last 7 values (mix of historical and predicted)
                if i < 7:
                    recent_window = list(recent_users[-(7-i):]) + forecasts[:i]
                else:
                    recent_window = forecasts[i-7:i]
                roll7_mean = np.mean(recent_window)
                
                # For CPU/storage, assume they remain similar to recent values
                lag1_cpu = recent_cpu[-1]  # Could be enhanced with CPU forecasting
                lag1_storage = recent_storage[-1]
            
            # Current values (assume similar to recent for simplicity)
            current_cpu = recent_cpu[-1]
            current_storage = recent_storage[-1]
            
            # Build feature vector
            feature_vector = np.array([
                lag1_users,           # lag1_users
                lag7_users,           # lag7_users  
                roll7_mean,           # roll7_mean_users
                lag1_cpu,             # lag1_cpu
                lag1_storage,         # lag1_storage
                current_cpu,          # usage_cpu
                current_storage,      # usage_storage
                last_economic,        # economic_index
                current_dow,          # dow
                0  # holiday (assume no holidays, could be enhanced)
            ]).reshape(1, -1)
            
            # Make prediction
            pred = model.predict(feature_vector)[0]
            
            # Ensure reasonable bounds
            pred = max(100, pred)  # Minimum 100 users
            
            forecasts.append(pred)
        
        forecasts = np.array(forecasts)
        
        print(f"XGBoost recursive predictions range: {forecasts.min():.1f} to {forecasts.max():.1f}")
        return forecasts, float(forecasts.min()), float(forecasts.max())
        
    except Exception as e:
        print(f"XGBoost prediction error: {e}")
        # Return reasonable fallback values based on historical data
        if 'users_active' in region_data.columns:
            avg_users = region_data['users_active'].mean()
            return np.full(forecast_days, avg_users), avg_users, avg_users
        else:
            return np.full(forecast_days, 1000.0), 1000.0, 1000.0

# -------- ACTIVE-USERS FORECAST ENDPOINTS -----------------------------
@app.route('/api/forecast/users/models')
@windows_cache('fast')
def users_model_status():
    """Get current best user models (dynamically updated by intelligent pipeline)"""
    try:
        model_info = {}
        
        # Use current best USERS models
        for region, model_type in FINAL_SELECTION_USERS.items():
            model_info[region] = {
                'model_type': model_type,
                'loaded': region in loaded_user_models,
                'has_scaler': region in loaded_user_scalers if model_type == 'LSTM' else None,
                'selection_method': 'intelligent_auto_selection' if intelligent_pipeline else 'static',
                'last_updated': 'dynamic' if intelligent_pipeline else 'static'
            }
        
        return jsonify({
            'models': model_info,
            'total_regions': len(FINAL_SELECTION_USERS),
            'model_types_used': list(set(FINAL_SELECTION_USERS.values())),
            'ml_available': ml_available,
            'model_directory': USER_MODEL_DIR,
            'directory_exists': os.path.exists(USER_MODEL_DIR),
            'selection_method': 'intelligent_auto_selection' if intelligent_pipeline else 'static'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/users/predict')
def users_predict():
    """Forecast ACTIVE USERS; structure mirrors CPU endpoint."""
    try:
        days        = max(7, min(int(request.args.get('days', 30)), 90))
        region_req  = request.args.get('region')
        
        if not ml_available:
            return jsonify({'error': 'ML libs unavailable'}), 503

        regions = ([region_req] if region_req and region_req != "All Regions"
                   else list(FINAL_SELECTION_USERS.keys()))
        results = {}
        
        for region in regions:
            print(f"\n=== Processing region: {region} ===")
            
            if region not in loaded_user_models:
                results[region] = {'error': f'Model not loaded for {region}'}
                continue
                
            if region not in region_dfs:
                results[region] = {'error': f'Data not available for {region}'}
                continue

            model_type = FINAL_SELECTION_USERS[region]
            model = loaded_user_models[region]
            region_data = region_dfs[region]
            users_series = region_data['users_active']
            
            print(f"Model type: {model_type}")
            print(f"Data shape: {region_data.shape}")
            print(f"Users range: {users_series.min():.1f} to {users_series.max():.1f}")

            try:
                if model_type == 'ARIMA':
                    preds, pmin, pmax = _arima_generic(model, users_series, days)
                elif model_type == 'XGBoost':
                    # Pass the full region dataframe for proper feature engineering
                    preds, pmin, pmax = _xgboost_generic(model, region_data, days)
                else:  # LSTM
                    scaler = loaded_user_scalers.get(region)
                    if scaler is None:
                        results[region] = {'error': 'Scaler missing for LSTM'}
                        continue
                    preds, pmin, pmax = _lstm_generic(model, scaler, users_series, days)

                print(f"Predictions: {preds[:5]}... (showing first 5)")
                print(f"Range: {pmin:.1f} to {pmax:.1f}")
                
                # Generate future dates
                fut_dates = pd.date_range(users_series.index[-1] + timedelta(days=1),
                                          periods=days, freq='D').strftime('%Y-%m-%d')
                
                # Get historical data for context
                hist = users_series.tail(14)
                
                results[region] = {
                    'dates': list(fut_dates),
                    'predicted_users': [max(0, float(p)) for p in preds],  # Ensure non-negative
                    'model_info': {
                        'type': model_type,
                        'forecast_horizon': days,
                        'range': [pmin, pmax]
                    },
                    'historical': {
                        'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                        'actual_users': hist.values.tolist()
                    }
                }
                
            except Exception as model_error:
                print(f"Model prediction error for {region}: {model_error}")
                results[region] = {'error': f'Prediction failed: {str(model_error)}'}
                
        return jsonify(results)
        
    except Exception as e:
        print(f"General forecast error: {e}")
        return jsonify({'error': str(e)}), 500

# ===== INTELLIGENT TRAINING ENDPOINTS =====

@app.route('/api/training/intelligent/status')
@windows_cache('medium')
def get_intelligent_training_status():
    """Get intelligent training pipeline status - NEW ENDPOINT"""
    try:
        if not intelligent_pipeline:
            return jsonify({'error': 'Intelligent pipeline not available'}), 503
        
        # Get model performance status
        status_df = intelligent_pipeline.get_model_status()
        
        # Get recent data monitoring
        conn = sqlite3.connect(intelligent_pipeline.performance_db)
        monitoring_query = '''
            SELECT * FROM data_monitoring 
            ORDER BY check_date DESC LIMIT 10
        '''
        monitoring_df = pd.read_sql_query(monitoring_query, conn)
        conn.close()
        
        # Get current best model configurations
        current_models = {
            'cpu': intelligent_pipeline.CPU_MODELS,
            'users': intelligent_pipeline.USERS_MODELS
        }
        
        return jsonify({
            'current_models': status_df.to_dict('records'),
            'recent_monitoring': monitoring_df.to_dict('records'),
            'pipeline_active': True,
            'pipeline_type': 'intelligent_auto_selection',
            'model_configurations': current_models,
            'all_model_types_tested': intelligent_pipeline.ALL_MODEL_TYPES,
            'last_check': monitoring_df.iloc[0]['check_date'] if not monitoring_df.empty else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/intelligent/trigger', methods=['POST'])
def trigger_intelligent_training():
    """Manually trigger intelligent training pipeline - NEW ENDPOINT"""
    try:
        if not intelligent_pipeline:
            return jsonify({'error': 'Intelligent pipeline not available'}), 503
        
        # Run training in background thread to avoid blocking
        def run_training():
            intelligent_pipeline.run_training_pipeline(force_training=True)
        
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        
        return jsonify({
            'status': 'FORCED Intelligent Training Pipeline Started',
            'pipeline_type': 'intelligent_auto_selection',
            'models_to_test': ['ARIMA', 'XGBoost', 'LSTM'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/intelligent/history')
@windows_cache('medium')
def get_intelligent_training_history():
    """Get comprehensive intelligent training history - NEW ENDPOINT"""
    try:
        if not intelligent_pipeline:
            return jsonify({'error': 'Intelligent pipeline not available'}), 503
        
        conn = sqlite3.connect(intelligent_pipeline.performance_db)
        
        # Get performance history with model type breakdown
        perf_query = '''
            SELECT model_type, metric_type, region, rmse, mae, mape, 
                   training_date, data_size, is_active
            FROM model_performance 
            ORDER BY training_date DESC LIMIT 200
        '''
        perf_df = pd.read_sql_query(perf_query, conn)
        
        # Get data monitoring history
        monitor_query = '''
            SELECT check_date, data_size, new_records, training_triggered
            FROM data_monitoring 
            ORDER BY check_date DESC LIMIT 50
        '''
        monitor_df = pd.read_sql_query(monitor_query, conn)
        
        # Get model performance comparison
        comparison_query = '''
            SELECT region, model_type, metric_type, MIN(rmse) as best_rmse, 
                   AVG(rmse) as avg_rmse, COUNT(*) as training_count
            FROM model_performance 
            GROUP BY region, model_type, metric_type
            ORDER BY region, metric_type, best_rmse
        '''
        comparison_df = pd.read_sql_query(comparison_query, conn)
        
        conn.close()
        
        return jsonify({
            'performance_history': perf_df.to_dict('records'),
            'monitoring_history': monitor_df.to_dict('records'),
            'model_comparison': comparison_df.to_dict('records'),
            'pipeline_type': 'intelligent_auto_selection'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/intelligent/config')
def get_intelligent_training_config():
    """Get intelligent training configuration - NEW ENDPOINT"""
    try:
        if not intelligent_pipeline:
            return jsonify({'error': 'Intelligent pipeline not available'}), 503
        
        return jsonify({
            'pipeline_type': 'intelligent_auto_selection',
            'current_cpu_models': intelligent_pipeline.CPU_MODELS,
            'current_users_models': intelligent_pipeline.USERS_MODELS,
            'all_model_types_tested': intelligent_pipeline.ALL_MODEL_TYPES,
            'data_path': intelligent_pipeline.data_path,
            'models_directories': {
                'cpu': str(intelligent_pipeline.models_dir),
                'users': str(intelligent_pipeline.users_models_dir)
            },
            'performance_db': intelligent_pipeline.performance_db,
            'auto_model_selection': True,
            'evaluation_metrics': ['RMSE', 'MAE', 'MAPE']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/intelligent/model-comparison')
@windows_cache('medium')
def get_intelligent_model_comparison():
    """Get detailed intelligent model performance comparison - NEW ENDPOINT"""
    try:
        if not intelligent_pipeline:
            return jsonify({'error': 'Intelligent pipeline not available'}), 503
        
        conn = sqlite3.connect(intelligent_pipeline.performance_db)
        
        # Get latest performance for each model type per region
        comparison_query = '''
            WITH latest_training AS (
                SELECT region, metric_type, model_type, rmse, mae, mape,
                       ROW_NUMBER() OVER (PARTITION BY region, metric_type, model_type ORDER BY training_date DESC) as rn
                FROM model_performance
            )
            SELECT region, metric_type, model_type, rmse, mae, mape
            FROM latest_training
            WHERE rn = 1
            ORDER BY region, metric_type, rmse
        '''
        
        comparison_df = pd.read_sql_query(comparison_query, conn)
        
        # Get currently active models
        active_query = '''
            SELECT region, metric_type, model_type, rmse, mae, mape
            FROM model_performance
            WHERE is_active = 1
            ORDER BY region, metric_type
        '''
        
        active_df = pd.read_sql_query(active_query, conn)
        conn.close()
        
        return jsonify({
            'all_model_comparison': comparison_df.to_dict('records'),
            'currently_active': active_df.to_dict('records'),
            'model_types_tested': intelligent_pipeline.ALL_MODEL_TYPES
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Update model comparison endpoints to use actual performance data when available
@app.route('/api/forecast/comparison')
@windows_cache('slow')
def model_comparison():
    """Get current model performance comparison from database or fallback to static"""
    try:
        if intelligent_pipeline:
            conn = sqlite3.connect(intelligent_pipeline.performance_db)
            
            # Get current active CPU models performance
            cpu_query = '''
                SELECT region, model_type, rmse, mae, mape
                FROM model_performance
                WHERE metric_type = 'cpu' AND is_active = 1
            '''
            
            cpu_df = pd.read_sql_query(cpu_query, conn)
            conn.close()
            
            if not cpu_df.empty:
                # Format for API response
                model_performance = {}
                for _, row in cpu_df.iterrows():
                    model_performance[row['region']] = {
                        'model': row['model_type'],
                        'rmse': float(row['rmse']),
                        'mae': float(row['mae']),
                        'mape': float(row['mape']) if pd.notna(row['mape']) else None
                    }
                
                # Calculate summary statistics
                all_rmse = [perf['rmse'] for perf in model_performance.values()]
                all_mae = [perf['mae'] for perf in model_performance.values()]
                
                summary = {
                    'regional_performance': model_performance,
                    'overall_stats': {
                        'avg_rmse': float(np.mean(all_rmse)),
                        'avg_mae': float(np.mean(all_mae)),
                        'best_rmse_region': min(model_performance.items(), key=lambda x: x[1]['rmse'])[0],
                        'model_types_active': list(set([perf['model'] for perf in model_performance.values()]))
                    },
                    'selection_method': 'intelligent_auto_selection'
                }
                
                return jsonify(summary)
        
        # Fallback to static performance data
        model_performance = {
            'East US': {'model': 'LSTM', 'rmse': 13.68, 'mae': 11.46},
            'North Europe': {'model': 'ARIMA', 'rmse': 15.90, 'mae': 14.25},
            'Southeast Asia': {'model': 'LSTM', 'rmse': 14.86, 'mae': 13.06},
            'West US': {'model': 'LSTM', 'rmse': 15.38, 'mae': 12.85}
        }

        all_rmse = [perf['rmse'] for perf in model_performance.values()]
        all_mae = [perf['mae'] for perf in model_performance.values()]

        summary = {
            'regional_performance': model_performance,
            'overall_stats': {
                'avg_rmse': float(np.mean(all_rmse)),
                'avg_mae': float(np.mean(all_mae)),
                'best_rmse_region': min(model_performance.items(), key=lambda x: x[1]['rmse'])[0],
                'lstm_regions': [k for k, v in model_performance.items() if v['model'] == 'LSTM'],
                'arima_regions': [k for k, v in model_performance.items() if v['model'] == 'ARIMA']
            },
            'selection_method': 'static_fallback'
        }

        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/users/comparison')
@windows_cache('slow')  
def users_model_comparison():
    """Get current user model performance comparison from database or fallback"""
    try:
        if intelligent_pipeline:
            conn = sqlite3.connect(intelligent_pipeline.performance_db)
            
            # Get current active USERS models performance
            users_query = '''
                SELECT region, model_type, rmse, mae, mape
                FROM model_performance
                WHERE metric_type = 'users' AND is_active = 1
            '''
            
            users_df = pd.read_sql_query(users_query, conn)
            conn.close()
            
            if not users_df.empty:
                # Format for API response
                model_performance = {}
                for _, row in users_df.iterrows():
                    model_performance[row['region']] = {
                        'model': row['model_type'],
                        'rmse': float(row['rmse']),
                        'mae': float(row['mae']),
                        'mape': float(row['mape']) if pd.notna(row['mape']) else None
                    }
                
                # Calculate summary statistics
                all_rmse = [perf['rmse'] for perf in model_performance.values()]
                all_mae = [perf['mae'] for perf in model_performance.values()]
                
                summary = {
                    'regional_performance': model_performance,
                    'overall_stats': {
                        'avg_rmse': float(np.mean(all_rmse)),
                        'avg_mae': float(np.mean(all_mae)),
                        'best_rmse_region': min(model_performance.items(), key=lambda x: x[1]['rmse'])[0],
                        'model_types_active': list(set([perf['model'] for perf in model_performance.values()]))
                    },
                    'selection_method': 'intelligent_auto_selection'
                }
                
                return jsonify(summary)
                
        # Fallback to static performance data
        comp = {
            'East US':        {'model': 'LSTM',    'rmse': 162.67, 'mae': 127.56},
            'North Europe':   {'model': 'XGBoost', 'rmse': 195.64, 'mae': 161.28},
            'Southeast Asia': {'model': 'ARIMA',   'rmse': 108.65, 'mae':  85.84},
            'West US':        {'model': 'XGBoost', 'rmse': 198.01, 'mae': 169.96},
        }
        all_rmse = [v['rmse'] for v in comp.values()]
        all_mae  = [v['mae']  for v in comp.values()]
        return jsonify({
            'regional_performance': comp,
            'overall_stats': {
                'avg_rmse': float(np.mean(all_rmse)),
                'avg_mae':  float(np.mean(all_mae)),
                'best_rmse_region': min(comp, key=lambda k: comp[k]['rmse']),
                'lstm_regions':  [k for k, v in comp.items() if v['model'] == 'LSTM'],
                'arima_regions': [k for k, v in comp.items() if v['model'] == 'ARIMA'],
                'xgboost_regions': [k for k, v in comp.items() if v['model'] == 'XGBoost'],
            },
            'selection_method': 'static_fallback'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Debug endpoint for user models
@app.route('/api/forecast/users/debug')
def users_forecast_debug():
    """Enhanced debug endpoint to check user model loading and feature availability."""
    try:
        debug_info = {
            'user_model_directory': USER_MODEL_DIR,
            'directory_exists': os.path.exists(USER_MODEL_DIR),
            'available_regions': list(FINAL_SELECTION_USERS.keys()),
            'loaded_user_models': list(loaded_user_models.keys()),
            'loaded_user_scalers': list(loaded_user_scalers.keys()),
            'region_data_available': list(region_dfs.keys()),
            'ml_available': ml_available,
            'model_selection': FINAL_SELECTION_USERS,
            'intelligent_pipeline_available': intelligent_pipeline is not None
        }
        
        # List files in directory if it exists
        if os.path.exists(USER_MODEL_DIR):
            try:
                files = os.listdir(USER_MODEL_DIR)
                debug_info['directory_contents'] = files
            except Exception as e:
                debug_info['directory_list_error'] = str(e)
        
        # Check data columns for each region
        data_summary = {}
        for region, data in region_dfs.items():
            data_summary[region] = {
                'columns': list(data.columns),
                'shape': data.shape,
                'users_active_range': [float(data['users_active'].min()), 
                                     float(data['users_active'].max())] if 'users_active' in data.columns else None,
                'last_date': str(data.index[-1]) if len(data) > 0 else None
            }
        debug_info['region_data_summary'] = data_summary
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add debugging endpoint to check what regions are being processed
@app.route('/api/forecast/debug')
def forecast_debug():
    """Debug endpoint to check forecasting setup"""
    try:
        region_filter = request.args.get('region', None)

        debug_info = {
            'request_region': region_filter,
            'available_regions': list(FINAL_SELECTION.keys()),
            'loaded_models': list(loaded_models.keys()),
            'loaded_scalers': list(loaded_scalers.keys()),
            'region_data_available': list(region_dfs.keys()),
            'ml_available': ml_available,
            'model_selection': FINAL_SELECTION,
            'intelligent_pipeline_available': intelligent_pipeline is not None
        }

        # Determine what regions would be processed
        if region_filter and region_filter != "All Regions":
            debug_info['regions_to_process'] = [region_filter]
        else:
            debug_info['regions_to_process'] = list(FINAL_SELECTION.keys())

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== HEALTH AND MONITORING =====

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(loaded_models),
        'ml_available': ml_available,
        'cache_entries': len(cache_dict),
        'data_records': len(df),
        'platform': 'Windows Compatible',
        'boot_time': f"{time.time() - start_boot_time:.2f}s",
        'regional_data': list(region_dfs.keys()),
        'intelligent_pipeline': intelligent_pipeline is not None
    })

@app.route('/api/cache/stats')
def cache_stats():
    with cache_lock:
        return jsonify({
            'cache_entries': len(cache_dict),
            'cache_keys': list(cache_dict.keys())[:10],
            'memory_usage': sys.getsizeof(cache_dict)
        })

@app.route('/api/cache/clear')
def clear_cache():
    with cache_lock:
        cache_dict.clear()
        cache_times.clear()
    return jsonify({'status': 'cache cleared'})

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'available_endpoints': [
        '/api/kpis', '/api/sparklines', '/api/time-series', '/api/trends/regional',
        '/api/trends/resource-types', '/api/regional/comparison', '/api/regional/heatmap',
        '/api/regional/distribution', '/api/resources/utilization', '/api/resources/distribution',
        '/api/resources/efficiency', '/api/correlations/matrix', '/api/correlations/scatter',
        '/api/correlations/bubble', '/api/holiday/analysis', '/api/holiday/distribution',
        '/api/holiday/calendar', '/api/engagement/efficiency', '/api/engagement/trends',
        '/api/engagement/bubble', '/api/filters/options', '/api/data/summary',
        '/api/forecast/models', '/api/forecast/predict', '/api/forecast/comparison', '/api/forecast/debug',
        '/api/forecast/users/models', '/api/forecast/users/predict', '/api/forecast/users/comparison',
        '/api/training/intelligent/status', '/api/training/intelligent/trigger', '/api/training/intelligent/history',
        '/api/training/intelligent/config', '/api/training/intelligent/model-comparison'
    ]}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ===== STARTUP =====

start_boot_time = time.time()

if __name__ == '__main__':
    boot_time = time.time() - start_boot_time
    print(f"⚡ COMPLETE Windows Ultra-Fast Azure API Server Ready in {boot_time:.2f}s!")
    print(f"📊 {len(df):,} records loaded with optimized caching")
    print(f"🚀 ML Models: {len(loaded_models)}/{len(FINAL_SELECTION)} loaded")
    print(f"👥 User Models: {len(loaded_user_models)}/{len(FINAL_SELECTION_USERS)} loaded")
    print(f"🔍 Regional Data: {list(region_dfs.keys())}")
    print(f"🤖 Intelligent Pipeline: {'✅ Connected' if intelligent_pipeline else '❌ Not Available'}")
    print("🔥 Windows Optimizations: Threading Cache | Parallel Forecasting | Memory Optimized")
    print("🪟 Platform: Windows Compatible | All Issues FIXED")
    print("🔧 Debug endpoints: /api/forecast/debug | /api/forecast/users/debug")

    # Production server with optimized settings
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True,
        use_reloader=False
    )
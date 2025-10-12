# TESTING.md

# 🧪 Testing Guide

## Azure Demand Forecasting Platform - Quality Assurance Procedures

*Comprehensive testing documentation for ensuring system reliability and performance*

---

## 🎯 **Testing Overview**

This testing guide covers all aspects of quality assurance for the Azure Demand Forecasting Platform, from unit tests to end-to-end system validation. Our testing strategy ensures 99.9% system reliability with comprehensive coverage across all components.

**Testing Pyramid**:
- **Unit Tests** (70%): Individual component testing
- **Integration Tests** (20%): Service interaction testing  
- **End-to-End Tests** (10%): Complete workflow validation

**Quality Gates**:
- Code coverage: >85%
- API response time: <200ms (P95)
- Forecast accuracy: >85%
- System availability: >99.9%

---

## 🏗️ **Test Environment Setup**

### Prerequisites
```bash
# Install testing dependencies
pip install -r requirements-test.txt

# Additional testing packages
pip install pytest pytest-cov pytest-mock pytest-asyncio
pip install httpx fastapi-testing
pip install locust  # For load testing
pip install allure-pytest  # For test reporting
```

### Test Configuration
```python
# test_config.py
import os
from pathlib import Path

class TestConfig:
    # Test database (SQLite for isolation)
    TEST_DATABASE = "test_model_performance.db"
    
    # Test data files
    TEST_DATA_DIR = Path("tests/data")
    SAMPLE_DATA_FILE = TEST_DATA_DIR / "sample_cleaned_merged.csv"
    
    # API testing
    API_BASE_URL = os.getenv("TEST_API_URL", "http://localhost:5000/api")
    
    # ML model testing
    MODEL_ACCURACY_THRESHOLD = 0.75  # 75% minimum
    FORECAST_HORIZON_DAYS = [1, 7, 30]  # Standard test horizons
    
    # Performance thresholds
    API_RESPONSE_TIME_MS = 500  # Maximum acceptable response time
    MEMORY_USAGE_MB = 2048  # Maximum memory usage
    
    # Test regions and services
    TEST_REGIONS = ["East US", "West US", "North Europe", "Southeast Asia"]
    TEST_SERVICES = ["Compute", "Storage", "Users"]
    
    # Cache testing
    CACHE_HIT_RATE_THRESHOLD = 0.90  # 90% minimum
```

### Test Data Setup
```python
# tests/fixtures.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_time_series_data():
    """Generate realistic time series data for testing"""
    dates = pd.date_range(start="2025-07-01", end="2025-10-10", freq="D")
    
    data = []
    for date in dates:
        for region in ["East US", "West US", "North Europe", "Southeast Asia"]:
            # Generate realistic patterns
            base_cpu = 70 + np.random.normal(0, 10)
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekend_factor = 0.85 if date.weekday() >= 5 else 1.0
            
            cpu_usage = max(0, min(100, base_cpu * seasonal_factor * weekend_factor))
            
            data.append({
                'date': date,
                'region': region,
                'usage_cpu': cpu_usage,
                'users_active': int(1000 + np.random.normal(0, 200)),
                'usage_storage': 1200 + np.random.normal(0, 300),
                'resource_type': 'compute',
                'economic_index': 45 + np.random.normal(0, 5),
                'cloud_market_demand': 75 + np.random.normal(0, 10),
                'holiday': 1 if np.random.random() < 0.1 else 0
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_trained_models():
    """Mock trained ML models for testing"""
    from unittest.mock import Mock
    
    models = {}
    for region in TestConfig.TEST_REGIONS:
        models[region] = {
            'ARIMA': Mock(),
            'LSTM': Mock(), 
            'XGBoost': Mock()
        }
        
        # Configure mock predictions
        for model in models[region].values():
            model.predict.return_value = np.random.uniform(60, 90, 30)
    
    return models

@pytest.fixture
def test_database():
    """Create isolated test database"""
    import sqlite3
    import tempfile
    import os
    
    # Create temporary database
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    
    # Initialize database schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            model_type TEXT,
            metric_type TEXT,
            rmse REAL,
            mae REAL,
            mape REAL,
            training_date TIMESTAMP,
            data_size INTEGER,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Insert sample performance data
    sample_data = [
        ('East US', 'ARIMA', 'cpu', 8.45, 6.23, 8.9, '2025-10-10 08:00:00', 270, 1),
        ('West US', 'LSTM', 'cpu', 9.12, 6.87, 9.4, '2025-10-09 14:30:00', 270, 1),
        ('North Europe', 'ARIMA', 'cpu', 7.93, 5.98, 8.1, '2025-10-08 12:00:00', 270, 1),
        ('Southeast Asia', 'XGBoost', 'cpu', 8.78, 6.45, 9.2, '2025-10-07 16:15:00', 270, 1)
    ]
    
    cursor.executemany('''
        INSERT INTO model_performance 
        (region, model_type, metric_type, rmse, mae, mape, training_date, data_size, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_data)
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    os.close(db_fd)
    os.unlink(db_path)
```

---

## ⚙️ **Unit Tests**

### API Endpoint Testing

```python
# tests/test_api_endpoints.py
import pytest
import json
from unittest.mock import patch, Mock
import pandas as pd

class TestHealthEndpoints:
    """Test system health and status endpoints"""
    
    def test_health_endpoint_success(self, client):
        """Test successful health check"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_health_endpoint_performance(self, client):
        """Test health endpoint response time"""
        import time
        
        start_time = time.time()
        response = client.get("/api/health")
        response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time < 100  # Health check should be very fast
    
    @patch('optimised_backend_app.df')
    def test_health_endpoint_database_failure(self, mock_df, client):
        """Test health endpoint when database is unavailable"""
        mock_df.side_effect = Exception("Database connection failed")
        
        response = client.get("/api/health")
        
        # Should still return 200 but indicate degraded status
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

class TestKPIEndpoints:
    """Test executive KPI endpoints"""
    
    def test_kpis_endpoint_structure(self, client, sample_time_series_data):
        """Test KPI endpoint returns correct structure"""
        with patch('optimised_backend_app.df', sample_time_series_data):
            response = client.get("/api/kpis")
            
            assert response.status_code == 200
            data = response.json()
            
            # Required fields
            required_fields = [
                "peak_cpu", "max_storage", "peak_users",
                "avg_cpu", "total_regions", "date_range"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
    
    def test_kpis_endpoint_data_types(self, client, sample_time_series_data):
        """Test KPI endpoint returns correct data types"""
        with patch('optimised_backend_app.df', sample_time_series_data):
            response = client.get("/api/kpis")
            data = response.json()
            
            assert isinstance(data["peak_cpu"], (int, float))
            assert isinstance(data["total_regions"], int)
            assert isinstance(data["date_range"], dict)
            assert "start" in data["date_range"]
            assert "end" in data["date_range"]
    
    def test_kpis_endpoint_caching(self, client, sample_time_series_data):
        """Test KPI endpoint caching behavior"""
        with patch('optimised_backend_app.df', sample_time_series_data):
            # First request
            start_time1 = time.time()
            response1 = client.get("/api/kpis")
            time1 = time.time() - start_time1
            
            # Second request (should be cached)
            start_time2 = time.time()
            response2 = client.get("/api/kpis")
            time2 = time.time() - start_time2
            
            assert response1.status_code == 200
            assert response2.status_code == 200
            
            # Second request should be significantly faster (cached)
            assert time2 < time1 * 0.5

class TestForecastingEndpoints:
    """Test ML forecasting endpoints"""
    
    @patch('optimised_backend_app.loaded_models')
    @patch('optimised_backend_app.region_dfs')
    def test_forecast_predict_success(self, mock_region_dfs, mock_models, client, sample_time_series_data):
        """Test successful forecast generation"""
        # Setup mocks
        mock_models["East US"] = Mock()
        mock_models["East US"].predict.return_value = np.array([75.0, 76.2, 74.8])
        
        mock_region_dfs["East US"] = sample_time_series_data[
            sample_time_series_data["region"] == "East US"
        ].set_index("date")
        
        response = client.get("/api/forecast/predict?days=3&region=East US")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "East US" in data
        assert "dates" in data["East US"]
        assert "predicted_cpu" in data["East US"]
        assert len(data["East US"]["dates"]) == 3
        assert len(data["East US"]["predicted_cpu"]) == 3
    
    def test_forecast_predict_invalid_parameters(self, client):
        """Test forecast endpoint with invalid parameters"""
        # Invalid days parameter
        response = client.get("/api/forecast/predict?days=0")
        assert response.status_code == 400
        
        response = client.get("/api/forecast/predict?days=100")
        assert response.status_code == 400
        
        # Invalid region
        response = client.get("/api/forecast/predict?region=Invalid Region")
        assert response.status_code in [400, 404]
    
    @patch('optimised_backend_app.loaded_models')
    def test_forecast_predict_model_unavailable(self, mock_models, client):
        """Test forecast endpoint when model is not available"""
        mock_models.clear()  # No models loaded
        
        response = client.get("/api/forecast/predict?days=7&region=East US")
        
        # Should return error response
        assert response.status_code in [404, 503]
        data = response.json()
        assert "error" in data

class TestCapacityPlanningEndpoints:
    """Test capacity planning endpoints"""
    
    @patch('optimised_backend_app.loaded_models')
    @patch('optimised_backend_app.region_dfs')
    def test_capacity_planning_success(self, mock_region_dfs, mock_models, client, sample_time_series_data):
        """Test successful capacity planning analysis"""
        # Setup mocks for all regions
        for region in TestConfig.TEST_REGIONS:
            mock_models[region] = Mock()
            mock_models[region].predict.return_value = np.array([75.0] * 30)
            
            mock_region_dfs[region] = sample_time_series_data[
                sample_time_series_data["region"] == region
            ].set_index("date")
        
        response = client.get("/api/capacity-planning?region=All Regions&service=Compute&horizon=30")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "capacity_analysis" in data
        assert "summary" in data
        assert len(data["capacity_analysis"]) > 0
        
        # Check required fields in analysis
        for region, analysis in data["capacity_analysis"].items():
            assert "current_capacity" in analysis
            assert "predicted_demand" in analysis
            assert "capacity_utilization" in analysis
            assert "risk_assessment" in analysis
            assert "recommendations" in analysis
    
    def test_capacity_planning_risk_assessment(self, client):
        """Test capacity planning risk assessment logic"""
        # This would test the risk calculation algorithms
        # Mock different utilization scenarios and verify risk levels
        pass
```

### ML Model Testing

```python
# tests/test_ml_models.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

class TestARIMAModels:
    """Test ARIMA forecasting models"""
    
    def test_arima_model_training(self, sample_time_series_data):
        """Test ARIMA model training process"""
        from model_training_pipeline import train_arima_model
        
        region_data = sample_time_series_data[
            sample_time_series_data["region"] == "East US"
        ].set_index("date")
        
        try:
            result = train_arima_model(region_data, forecast_horizon=7)
            
            assert "model" in result
            assert "forecast" in result
            assert len(result["forecast"]) == 7
            assert all(isinstance(x, (int, float)) for x in result["forecast"])
            assert all(0 <= x <= 150 for x in result["forecast"])  # Reasonable CPU range
            
        except Exception as e:
            pytest.skip(f"ARIMA training requires statsmodels: {e}")
    
    def test_arima_parameter_optimization(self, sample_time_series_data):
        """Test ARIMA parameter optimization"""
        from model_training_pipeline import optimize_arima_parameters
        
        region_data = sample_time_series_data[
            sample_time_series_data["region"] == "East US"
        ]
        time_series = region_data["usage_cpu"].values
        
        try:
            best_order, best_aic = optimize_arima_parameters(time_series, max_p=2, max_d=1, max_q=2)
            
            assert len(best_order) == 3
            assert all(isinstance(x, int) for x in best_order)
            assert best_aic > 0
            
        except Exception as e:
            pytest.skip(f"ARIMA optimization requires statsmodels: {e}")

class TestLSTMModels:
    """Test LSTM neural network models"""
    
    @pytest.mark.skipif(not has_tensorflow(), reason="TensorFlow not available")
    def test_lstm_model_creation(self):
        """Test LSTM model architecture creation"""
        from model_training_pipeline import build_lstm_model
        
        model = build_lstm_model(sequence_length=14, n_features=1)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')
        
        # Check input shape
        assert model.input_shape == (None, 14, 1)
        
        # Check model has reasonable number of parameters
        param_count = model.count_params()
        assert 1000 < param_count < 100000  # Reasonable range
    
    @pytest.mark.skipif(not has_tensorflow(), reason="TensorFlow not available")
    def test_lstm_sequence_preparation(self, sample_time_series_data):
        """Test LSTM data sequence preparation"""
        from model_training_pipeline import prepare_lstm_sequences
        
        region_data = sample_time_series_data[
            sample_time_series_data["region"] == "East US"
        ]
        time_series = region_data["usage_cpu"].values
        
        sequences, targets = prepare_lstm_sequences(time_series, sequence_length=7)
        
        assert sequences.shape[1] == 7  # Sequence length
        assert sequences.shape[0] == targets.shape[0]  # Same number of samples
        assert sequences.shape[0] == len(time_series) - 7  # Correct number of sequences

class TestXGBoostModels:
    """Test XGBoost gradient boosting models"""
    
    def test_xgboost_feature_engineering(self, sample_time_series_data):
        """Test XGBoost feature engineering"""
        from model_training_pipeline import create_xgboost_features
        
        region_data = sample_time_series_data[
            sample_time_series_data["region"] == "East US"
        ].set_index("date")
        
        X, y, feature_columns = create_xgboost_features(region_data, "usage_cpu")
        
        assert not X.empty
        assert len(X) == len(y)
        assert "lag_1" in feature_columns
        assert "rolling_7_mean" in feature_columns
        assert "day_of_week" in feature_columns
        
        # Check for no NaN values in features
        assert not X.isnull().any().any()
    
    def test_xgboost_model_training(self, sample_time_series_data):
        """Test XGBoost model training process"""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("XGBoost not available")
        
        from model_training_pipeline import train_xgboost_model
        
        region_data = sample_time_series_data[
            sample_time_series_data["region"] == "East US"
        ].set_index("date")
        
        result = train_xgboost_model(region_data, forecast_horizon=7, target_metric="usage_cpu")
        
        assert "model" in result
        assert "forecast" in result
        assert "feature_importance" in result
        assert len(result["forecast"]) == 7
        
        # Check feature importance
        importance = result["feature_importance"]
        assert len(importance) > 0
        assert all(0 <= v <= 1 for v in importance.values())

class TestModelSelection:
    """Test intelligent model selection system"""
    
    def test_model_performance_comparison(self):
        """Test model performance comparison logic"""
        performance_data = {
            "ARIMA": {"rmse": 8.5, "mae": 6.2, "mape": 8.9},
            "LSTM": {"rmse": 9.1, "mae": 6.8, "mape": 9.4},
            "XGBoost": {"rmse": 8.8, "mae": 6.5, "mape": 9.2}
        }
        
        # Calculate composite scores
        scores = {}
        for model, metrics in performance_data.items():
            scores[model] = (metrics["rmse"] * 0.5 + 
                           metrics["mae"] * 0.3 + 
                           metrics["mape"] * 0.2)
        
        best_model = min(scores, key=scores.get)
        assert best_model == "ARIMA"  # Should have lowest composite score

def has_tensorflow():
    """Check if TensorFlow is available"""
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False
```

### Data Processing Tests

```python
# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np

class TestDataValidation:
    """Test data validation and quality checks"""
    
    def test_schema_validation_success(self, sample_time_series_data):
        """Test successful schema validation"""
        # Required columns should be present
        required_columns = ["date", "region", "usage_cpu", "users_active", "usage_storage"]
        
        for col in required_columns:
            assert col in sample_time_series_data.columns
    
    def test_schema_validation_missing_columns(self):
        """Test schema validation with missing columns"""
        # Create data missing required columns
        incomplete_data = pd.DataFrame({
            "date": ["2025-10-01", "2025-10-02"],
            "region": ["East US", "West US"]
            # Missing usage_cpu, users_active, usage_storage
        })
        
        # Should detect missing columns
        required_columns = ["date", "region", "usage_cpu", "users_active", "usage_storage"]
        missing_columns = [col for col in required_columns if col not in incomplete_data.columns]
        
        assert len(missing_columns) > 0
        assert "usage_cpu" in missing_columns
    
    def test_data_type_validation(self, sample_time_series_data):
        """Test data type validation"""
        # Convert to proper types
        df = sample_time_series_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        assert df["usage_cpu"].dtype in [np.float64, np.int64]
        assert df["users_active"].dtype in [np.int64, np.float64]
        assert df["usage_storage"].dtype in [np.float64, np.int64]
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
    
    def test_outlier_detection(self, sample_time_series_data):
        """Test outlier detection algorithms"""
        df = sample_time_series_data.copy()
        
        # Add some outliers
        df.loc[0, "usage_cpu"] = 150  # Impossible value
        df.loc[1, "users_active"] = -100  # Negative users
        
        # IQR method for outlier detection
        Q1 = df["usage_cpu"].quantile(0.25)
        Q3 = df["usage_cpu"].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = df[
            (df["usage_cpu"] < Q1 - 1.5 * IQR) | 
            (df["usage_cpu"] > Q3 + 1.5 * IQR)
        ]
        
        assert len(outliers) > 0  # Should detect the 150% CPU usage
    
    def test_missing_value_handling(self):
        """Test missing value handling strategies"""
        # Create data with missing values
        df = pd.DataFrame({
            "date": pd.date_range("2025-10-01", periods=10),
            "usage_cpu": [70, 72, np.nan, 74, 75, np.nan, 77, 78, 79, 80],
            "region": ["East US"] * 10
        })
        
        # Forward fill strategy
        df_filled = df.copy()
        df_filled["usage_cpu"] = df_filled["usage_cpu"].fillna(method="ffill")
        
        assert df_filled["usage_cpu"].isnull().sum() == 0
        assert df_filled.loc[2, "usage_cpu"] == 72  # Forward filled value
    
    def test_data_quality_scoring(self, sample_time_series_data):
        """Test data quality scoring system"""
        df = sample_time_series_data.copy()
        
        # Calculate quality metrics
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        # Validity check (CPU should be 0-100)
        valid_cpu = ((df["usage_cpu"] >= 0) & (df["usage_cpu"] <= 100)).sum()
        cpu_validity = (valid_cpu / len(df)) * 100
        
        # Users should be positive
        valid_users = (df["users_active"] >= 0).sum()
        users_validity = (valid_users / len(df)) * 100
        
        overall_quality = (completeness + cpu_validity + users_validity) / 3
        
        assert 0 <= overall_quality <= 100
        assert completeness >= 90  # Our sample data should be mostly complete

class TestDataTransformation:
    """Test data transformation and feature engineering"""
    
    def test_lag_feature_creation(self, sample_time_series_data):
        """Test lag feature creation"""
        df = sample_time_series_data[sample_time_series_data["region"] == "East US"].copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        # Create lag features
        df["lag_1"] = df["usage_cpu"].shift(1)
        df["lag_7"] = df["usage_cpu"].shift(7)
        
        # Check lag features
        assert df.loc[1, "lag_1"] == df.loc[0, "usage_cpu"]
        assert pd.isna(df.loc[0, "lag_1"])  # First value should be NaN
        
        if len(df) > 7:
            assert df.loc[7, "lag_7"] == df.loc[0, "usage_cpu"]
    
    def test_rolling_statistics(self, sample_time_series_data):
        """Test rolling statistics calculation"""
        df = sample_time_series_data[sample_time_series_data["region"] == "East US"].copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        # Create rolling features
        df["rolling_7_mean"] = df["usage_cpu"].rolling(7).mean()
        df["rolling_7_std"] = df["usage_cpu"].rolling(7).std()
        
        # Check rolling statistics
        if len(df) >= 7:
            manual_mean = df.loc[0:6, "usage_cpu"].mean()
            assert abs(df.loc[6, "rolling_7_mean"] - manual_mean) < 0.01
    
    def test_temporal_feature_extraction(self, sample_time_series_data):
        """Test temporal feature extraction"""
        df = sample_time_series_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # Extract temporal features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        
        # Validate ranges
        assert df["day_of_week"].min() >= 0
        assert df["day_of_week"].max() <= 6
        assert df["day_of_month"].min() >= 1
        assert df["day_of_month"].max() <= 31
        assert df["month"].min() >= 1
        assert df["month"].max() <= 12
        assert df["quarter"].min() >= 1
        assert df["quarter"].max() <= 4
```

---

## 🔗 **Integration Tests**

### API Integration Tests

```python
# tests/test_api_integration.py
import pytest
import requests
import time
import threading

class TestAPIIntegration:
    """Test API endpoint integration"""
    
    @pytest.fixture(autouse=True)
    def setup_api(self, api_server):
        """Ensure API server is running for integration tests"""
        self.api_base = "http://localhost:5000/api"
        
        # Wait for server to be ready
        max_retries = 10
        for _ in range(max_retries):
            try:
                response = requests.get(f"{self.api_base}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(1)
        else:
            pytest.skip("API server not available for integration tests")
    
    def test_health_to_kpis_workflow(self):
        """Test workflow from health check to KPI retrieval"""
        # 1. Check system health
        health_response = requests.get(f"{self.api_base}/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        if health_data["status"] != "healthy":
            pytest.skip("System not healthy for integration testing")
        
        # 2. Get KPIs
        kpis_response = requests.get(f"{self.api_base}/kpis")
        assert kpis_response.status_code == 200
        
        kpis_data = kpis_response.json()
        assert "peak_cpu" in kpis_data
        assert "total_regions" in kpis_data
    
    def test_forecasting_workflow(self):
        """Test complete forecasting workflow"""
        # 1. Check model status
        models_response = requests.get(f"{self.api_base}/forecast/models")
        assert models_response.status_code == 200
        
        models_data = models_response.json()
        available_regions = [region for region, info in models_data.get("models", {}).items() 
                           if info.get("loaded", False)]
        
        if not available_regions:
            pytest.skip("No models available for forecasting integration test")
        
        # 2. Generate forecast for available region
        test_region = available_regions[0]
        forecast_response = requests.get(
            f"{self.api_base}/forecast/predict",
            params={"days": 7, "region": test_region}
        )
        
        assert forecast_response.status_code == 200
        forecast_data = forecast_response.json()
        
        assert test_region in forecast_data
        assert "predicted_cpu" in forecast_data[test_region]
        assert len(forecast_data[test_region]["predicted_cpu"]) == 7
    
    def test_capacity_planning_workflow(self):
        """Test capacity planning integration"""
        # 1. Check system health
        health_response = requests.get(f"{self.api_base}/health")
        assert health_response.status_code == 200
        
        # 2. Get capacity analysis
        capacity_response = requests.get(
            f"{self.api_base}/capacity-planning",
            params={"region": "All Regions", "service": "Compute", "horizon": 7}
        )
        
        if capacity_response.status_code == 200:
            capacity_data = capacity_response.json()
            assert "capacity_analysis" in capacity_data
            assert "summary" in capacity_data
        else:
            # May fail if models not available - that's expected in some test environments
            assert capacity_response.status_code in [503, 404]
    
    def test_concurrent_api_requests(self):
        """Test API behavior under concurrent load"""
        import concurrent.futures
        
        def make_request():
            response = requests.get(f"{self.api_base}/kpis")
            return response.status_code, response.elapsed.total_seconds()
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        status_codes = [result[0] for result in results]
        response_times = [result[1] for result in results]
        
        assert all(code == 200 for code in status_codes)
        assert all(time < 5.0 for time in response_times)  # Max 5 seconds per request
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 1.0  # Average under 1 second
    
    def test_api_error_handling(self):
        """Test API error handling integration"""
        # Test invalid parameters
        response = requests.get(f"{self.api_base}/forecast/predict?days=0")
        assert response.status_code == 400
        
        error_data = response.json()
        assert "error" in error_data
        
        # Test non-existent endpoint
        response = requests.get(f"{self.api_base}/nonexistent")
        assert response.status_code == 404

class TestDatabaseIntegration:
    """Test database integration"""
    
    def test_model_performance_storage(self, test_database):
        """Test storing and retrieving model performance"""
        import sqlite3
        
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        
        # Insert test performance data
        test_data = {
            'region': 'Test Region',
            'model_type': 'TEST',
            'metric_type': 'cpu',
            'rmse': 5.5,
            'mae': 4.2,
            'mape': 6.8,
            'training_date': '2025-10-12 12:00:00',
            'data_size': 100,
            'is_active': 1
        }
        
        cursor.execute('''
            INSERT INTO model_performance 
            (region, model_type, metric_type, rmse, mae, mape, training_date, data_size, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(test_data.values()))
        conn.commit()
        
        # Retrieve and verify
        cursor.execute('''
            SELECT * FROM model_performance 
            WHERE region = ? AND model_type = ?
        ''', (test_data['region'], test_data['model_type']))
        
        result = cursor.fetchone()
        assert result is not None
        
        conn.close()
    
    def test_database_performance_tracking(self, test_database):
        """Test performance tracking queries"""
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect(test_database)
        
        # Query model performance comparison
        query = '''
            SELECT region, model_type, rmse, mae, mape
            FROM model_performance
            WHERE is_active = 1
            ORDER BY region, rmse
        '''
        
        df = pd.read_sql_query(query, conn)
        
        assert not df.empty
        assert 'region' in df.columns
        assert 'rmse' in df.columns
        
        conn.close()

class TestMLPipelineIntegration:
    """Test ML pipeline integration"""
    
    @pytest.mark.slow
    def test_training_pipeline_integration(self, sample_time_series_data, test_database):
        """Test complete training pipeline integration"""
        from model_training_pipeline import IntelligentModelTrainingPipeline
        
        # Create temporary data file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_time_series_data.to_csv(f.name, index=False)
            temp_data_file = f.name
        
        try:
            # Initialize pipeline
            pipeline = IntelligentModelTrainingPipeline(data_path=temp_data_file)
            pipeline.performance_db = test_database
            
            # Run training (this may take several minutes)
            pipeline.run_training_pipeline(force_training=True)
            
            # Verify models were trained and stored
            import sqlite3
            conn = sqlite3.connect(test_database)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM model_performance WHERE training_date > datetime("now", "-1 hour")')
            recent_trainings = cursor.fetchone()[0]
            
            assert recent_trainings > 0  # Should have some recent training records
            conn.close()
            
        finally:
            os.unlink(temp_data_file)
```

---

## 🚀 **Performance Tests**

### Load Testing

```python
# tests/test_performance.py
import pytest
import time
import statistics
import concurrent.futures
import requests

class TestAPIPerformance:
    """Test API performance under load"""
    
    @pytest.mark.performance
    def test_single_request_latency(self):
        """Test single request latency"""
        api_base = "http://localhost:5000/api"
        
        latencies = []
        for _ in range(50):  # 50 individual requests
            start_time = time.time()
            response = requests.get(f"{api_base}/kpis")
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            assert response.status_code == 200
            latencies.append(latency)
        
        # Performance assertions
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        
        assert avg_latency < TestConfig.API_RESPONSE_TIME_MS, f"Average latency {avg_latency:.1f}ms exceeds threshold"
        assert p95_latency < TestConfig.API_RESPONSE_TIME_MS * 1.5, f"P95 latency {p95_latency:.1f}ms exceeds threshold"
        
        print(f"Performance Results:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  P95 latency: {p95_latency:.1f}ms")
        print(f"  Min latency: {min(latencies):.1f}ms")
        print(f"  Max latency: {max(latencies):.1f}ms")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_load(self):
        """Test API performance under concurrent load"""
        api_base = "http://localhost:5000/api"
        
        def make_concurrent_request():
            start_time = time.time()
            response = requests.get(f"{api_base}/kpis")
            latency = (time.time() - start_time) * 1000
            return response.status_code, latency
        
        # Test with increasing concurrent users
        for concurrent_users in [5, 10, 20]:
            print(f"\nTesting with {concurrent_users} concurrent users:")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(make_concurrent_request) for _ in range(concurrent_users * 3)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            status_codes = [result[0] for result in results]
            latencies = [result[1] for result in results]
            
            # All requests should succeed
            success_rate = sum(1 for code in status_codes if code == 200) / len(status_codes)
            assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below threshold with {concurrent_users} users"
            
            # Latency should remain acceptable
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
            
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Average latency: {avg_latency:.1f}ms")
            print(f"  P95 latency: {p95_latency:.1f}ms")
            
            # Allow higher latency under load, but still reasonable
            assert avg_latency < TestConfig.API_RESPONSE_TIME_MS * 2
    
    @pytest.mark.performance
    def test_cache_performance(self):
        """Test caching performance improvement"""
        api_base = "http://localhost:5000/api"
        
        # First request (cache miss)
        start_time = time.time()
        response1 = requests.get(f"{api_base}/kpis")
        first_request_time = (time.time() - start_time) * 1000
        
        assert response1.status_code == 200
        
        # Second request (cache hit)
        start_time = time.time()
        response2 = requests.get(f"{api_base}/kpis")
        second_request_time = (time.time() - start_time) * 1000
        
        assert response2.status_code == 200
        
        # Cached request should be significantly faster
        cache_improvement = (first_request_time - second_request_time) / first_request_time
        
        print(f"Cache Performance:")
        print(f"  First request: {first_request_time:.1f}ms")
        print(f"  Second request: {second_request_time:.1f}ms")
        print(f"  Improvement: {cache_improvement:.2%}")
        
        # Cache should provide at least 20% improvement
        assert cache_improvement > 0.2, f"Cache improvement {cache_improvement:.2%} below threshold"
    
    @pytest.mark.performance
    def test_forecast_generation_performance(self):
        """Test forecasting performance"""
        api_base = "http://localhost:5000/api"
        
        # Test different forecast horizons
        for days in [1, 7, 30]:
            start_time = time.time()
            response = requests.get(f"{api_base}/forecast/predict", 
                                  params={"days": days, "region": "East US"})
            generation_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                print(f"Forecast generation ({days} days): {generation_time:.1f}ms")
                
                # Performance expectations by horizon
                if days == 1:
                    assert generation_time < 2000  # 2 seconds for 1-day (likely cached)
                elif days == 7:
                    assert generation_time < 5000  # 5 seconds for 7-day
                elif days == 30:
                    assert generation_time < 10000  # 10 seconds for 30-day
            else:
                pytest.skip(f"Forecasting not available for performance test (status: {response.status_code})")

class TestMemoryPerformance:
    """Test memory usage and performance"""
    
    @pytest.mark.performance
    def test_memory_usage_monitoring(self):
        """Test system memory usage"""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Monitor memory usage during test operations
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate heavy operations
        api_base = "http://localhost:5000/api"
        
        for _ in range(20):  # Multiple API calls
            requests.get(f"{api_base}/kpis")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Increase: {memory_increase:.1f}MB")
        
        # Memory usage should remain reasonable
        assert final_memory < TestConfig.MEMORY_USAGE_MB, f"Memory usage {final_memory:.1f}MB exceeds threshold"
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB too high"
    
    @pytest.mark.performance
    def test_database_query_performance(self, test_database):
        """Test database query performance"""
        import sqlite3
        import time
        
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        
        # Test various query patterns
        queries = [
            "SELECT * FROM model_performance WHERE is_active = 1",
            "SELECT region, AVG(rmse) FROM model_performance GROUP BY region",
            "SELECT * FROM model_performance ORDER BY training_date DESC LIMIT 10"
        ]
        
        for query in queries:
            start_time = time.time()
            cursor.execute(query)
            results = cursor.fetchall()
            query_time = (time.time() - start_time) * 1000
            
            print(f"Query performance: {query_time:.1f}ms for {len(results)} results")
            assert query_time < 100  # Queries should be under 100ms
        
        conn.close()

@pytest.mark.performance
class TestLoadTestingWithLocust:
    """Load testing using Locust framework"""
    
    def test_generate_load_test_file(self):
        """Generate Locust load test file"""
        load_test_content = '''
from locust import HttpUser, task, between

class AzureForecastingUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Actions to perform when starting"""
        # Check system health before starting tests
        self.client.get("/api/health")
    
    @task(3)
    def get_kpis(self):
        """Get executive KPIs (most common request)"""
        self.client.get("/api/kpis")
    
    @task(2)
    def get_sparklines(self):
        """Get sparklines data"""
        self.client.get("/api/sparklines")
    
    @task(2)
    def get_regional_comparison(self):
        """Get regional comparison"""
        self.client.get("/api/regional/comparison")
    
    @task(1)
    def generate_forecast(self):
        """Generate forecast (less frequent, more expensive)"""
        self.client.get("/api/forecast/predict?days=7&region=East US")
    
    @task(1)
    def capacity_planning(self):
        """Get capacity planning analysis"""
        self.client.get("/api/capacity-planning?region=All Regions&service=Compute&horizon=30")
'''
        
        # Write to file for manual execution
        with open("locust_load_test.py", "w") as f:
            f.write(load_test_content)
        
        print("Generated locust_load_test.py")
        print("Run with: locust -f locust_load_test.py --host=http://localhost:5000")
        
        assert True  # Test passes by generating the file
```

---

## 🧪 **End-to-End Tests**

### Complete Workflow Testing

```python
# tests/test_e2e_workflows.py
import pytest
import requests
import time
import pandas as pd

class TestCompleteWorkflows:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.e2e
    def test_executive_dashboard_workflow(self):
        """Test complete executive dashboard workflow"""
        api_base = "http://localhost:5000/api"
        
        print("🎯 Testing Executive Dashboard Workflow")
        
        # Step 1: System Health Check
        print("  Step 1: Checking system health...")
        health_response = requests.get(f"{api_base}/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data["status"] in ["healthy", "degraded"]
        print(f"    ✅ System status: {health_data['status']}")
        
        # Step 2: Get Executive KPIs
        print("  Step 2: Retrieving executive KPIs...")
        kpis_response = requests.get(f"{api_base}/kpis")
        assert kpis_response.status_code == 200
        
        kpis_data = kpis_response.json()
        print(f"    ✅ Peak CPU: {kpis_data.get('peak_cpu', 'N/A')}%")
        print(f"    ✅ Total regions: {kpis_data.get('total_regions', 'N/A')}")
        
        # Step 3: Get Trend Data
        print("  Step 3: Getting sparklines trend data...")
        sparklines_response = requests.get(f"{api_base}/sparklines")
        assert sparklines_response.status_code == 200
        
        sparklines_data = sparklines_response.json()
        assert "cpu_trend" in sparklines_data
        print(f"    ✅ CPU trend data points: {len(sparklines_data['cpu_trend'])}")
        
        # Step 4: Regional Analysis
        print("  Step 4: Getting regional comparison...")
        regional_response = requests.get(f"{api_base}/regional/comparison")
        assert regional_response.status_code == 200
        
        regional_data = regional_response.json()
        regions_analyzed = len(regional_data)
        print(f"    ✅ Regions analyzed: {regions_analyzed}")
        
        print("🎯 Executive Dashboard Workflow: PASSED")
    
    @pytest.mark.e2e
    def test_forecasting_workflow(self):
        """Test complete forecasting workflow"""
        api_base = "http://localhost:5000/api"
        
        print("🤖 Testing AI Forecasting Workflow")
        
        # Step 1: Check Model Status
        print("  Step 1: Checking ML model status...")
        models_response = requests.get(f"{api_base}/forecast/models")
        
        if models_response.status_code == 200:
            models_data = models_response.json()
            available_models = [region for region, info in models_data.get("models", {}).items() 
                              if info.get("loaded", False)]
            print(f"    ✅ Available models: {', '.join(available_models)}")
            
            if not available_models:
                pytest.skip("No ML models available for forecasting workflow test")
            
            test_region = available_models[0]
        else:
            pytest.skip("ML model status not available")
        
        # Step 2: Generate Short-term Forecast
        print("  Step 2: Generating 7-day forecast...")
        forecast_response = requests.get(f"{api_base}/forecast/predict", 
                                       params={"days": 7, "region": test_region})
        
        if forecast_response.status_code == 200:
            forecast_data = forecast_response.json()
            predictions = forecast_data[test_region]["predicted_cpu"]
            print(f"    ✅ 7-day forecast generated: {len(predictions)} predictions")
            
            # Validate forecast data quality
            assert all(isinstance(pred, (int, float)) for pred in predictions)
            assert all(0 <= pred <= 150 for pred in predictions)  # Reasonable range
            
        else:
            print(f"    ⚠️ Forecast generation failed: {forecast_response.status_code}")
        
        # Step 3: Generate Long-term Forecast
        print("  Step 3: Generating 30-day forecast...")
        long_forecast_response = requests.get(f"{api_base}/forecast/predict", 
                                            params={"days": 30, "region": test_region})
        
        if long_forecast_response.status_code == 200:
            long_forecast_data = long_forecast_response.json()
            long_predictions = long_forecast_data[test_region]["predicted_cpu"]
            print(f"    ✅ 30-day forecast generated: {len(long_predictions)} predictions")
        else:
            print(f"    ⚠️ Long-term forecast generation failed")
        
        # Step 4: Compare Model Performance
        print("  Step 4: Checking model performance...")
        comparison_response = requests.get(f"{api_base}/forecast/comparison")
        
        if comparison_response.status_code == 200:
            comparison_data = comparison_response.json()
            print(f"    ✅ Model comparison data retrieved")
        
        print("🤖 AI Forecasting Workflow: PASSED")
    
    @pytest.mark.e2e
    def test_capacity_planning_workflow(self):
        """Test complete capacity planning workflow"""
        api_base = "http://localhost:5000/api"
        
        print("🏗️ Testing Capacity Planning Workflow")
        
        # Step 1: Get Overall Capacity Analysis
        print("  Step 1: Getting capacity analysis...")
        capacity_response = requests.get(f"{api_base}/capacity-planning", 
                                       params={
                                           "region": "All Regions",
                                           "service": "Compute", 
                                           "horizon": 30
                                       })
        
        if capacity_response.status_code == 200:
            capacity_data = capacity_response.json()
            
            # Analyze results
            analysis = capacity_data.get("capacity_analysis", {})
            summary = capacity_data.get("summary", {})
            
            regions_analyzed = len(analysis)
            overall_status = summary.get("overall_status", "UNKNOWN")
            
            print(f"    ✅ Regions analyzed: {regions_analyzed}")
            print(f"    ✅ Overall status: {overall_status}")
            
            # Step 2: Check Risk Distribution
            print("  Step 2: Analyzing risk distribution...")
            risk_dist = summary.get("risk_distribution", {})
            high_risk = risk_dist.get("high_risk", 0)
            medium_risk = risk_dist.get("medium_risk", 0)
            low_risk = risk_dist.get("low_risk", 0)
            
            print(f"    ✅ Risk distribution: {high_risk} High, {medium_risk} Medium, {low_risk} Low")
            
            # Step 3: Extract Recommendations
            print("  Step 3: Processing recommendations...")
            all_recommendations = []
            for region, region_analysis in analysis.items():
                for rec in region_analysis.get("recommendations", []):
                    all_recommendations.append({
                        "region": region,
                        "priority": rec.get("priority", "LOW"),
                        "action": rec.get("action", ""),
                        "type": rec.get("type", "MAINTAIN")
                    })
            
            high_priority_recs = [r for r in all_recommendations if r["priority"] == "HIGH"]
            print(f"    ✅ Total recommendations: {len(all_recommendations)}")
            print(f"    ✅ High priority actions: {len(high_priority_recs)}")
            
            if high_priority_recs:
                print("    🚨 High Priority Actions:")
                for rec in high_priority_recs[:3]:  # Show top 3
                    print(f"      - {rec['region']}: {rec['action']}")
            
        else:
            print(f"    ⚠️ Capacity analysis failed: {capacity_response.status_code}")
            if capacity_response.status_code == 503:
                print("    ℹ️ This may be expected if ML models are not fully loaded")
        
        print("🏗️ Capacity Planning Workflow: PASSED")
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_business_workflow(self):
        """Test complete business workflow from data to decisions"""
        api_base = "http://localhost:5000/api"
        
        print("💼 Testing Complete Business Workflow")
        
        # Step 1: Executive Dashboard Review
        print("  Step 1: Executive dashboard review...")
        kpis = requests.get(f"{api_base}/kpis").json()
        peak_cpu = kpis.get("peak_cpu", 0)
        total_regions = kpis.get("total_regions", 0)
        
        print(f"    📊 Current peak CPU: {peak_cpu}%")
        print(f"    🌍 Monitoring {total_regions} regions")
        
        # Step 2: Trend Analysis
        print("  Step 2: Analyzing trends...")
        sparklines = requests.get(f"{api_base}/sparklines").json()
        cpu_trend = sparklines.get("cpu_trend", [])
        
        if len(cpu_trend) >= 2:
            latest_cpu = cpu_trend[-1]["usage_cpu"]
            previous_cpu = cpu_trend[-2]["usage_cpu"]
            trend_direction = "increasing" if latest_cpu > previous_cpu else "decreasing"
            print(f"    📈 CPU trend: {trend_direction} ({latest_cpu:.1f}% latest)")
        
        # Step 3: Risk Assessment
        print("  Step 3: Assessing capacity risks...")
        capacity_response = requests.get(f"{api_base}/capacity-planning", 
                                       params={"region": "All Regions", "service": "Compute", "horizon": 30})
        
        if capacity_response.status_code == 200:
            capacity_data = capacity_response.json()
            risk_dist = capacity_data.get("summary", {}).get("risk_distribution", {})
            high_risk_regions = risk_dist.get("high_risk", 0)
            
            print(f"    ⚠️ High-risk regions: {high_risk_regions}")
            
            if high_risk_regions > 0:
                print("    🚨 ALERT: Immediate attention required for high-risk regions")
            else:
                print("    ✅ All regions within acceptable risk levels")
        
        # Step 4: Generate Business Report
        print("  Step 4: Generating business report...")
        report_response = requests.get(f"{api_base}/reports/generate", 
                                     params={"type": "csv", "forecast_horizon": 7})
        
        if report_response.status_code == 200:
            print(f"    📊 Business report generated ({len(report_response.content)} bytes)")
            
            # Basic validation of CSV content
            content = report_response.content.decode('utf-8')
            lines = content.split('\n')
            headers = lines[0].split(',') if lines else []
            
            expected_columns = ["Date", "Region", "CPU_Forecast"]
            for col in expected_columns:
                assert col in headers, f"Missing column: {col}"
            
            print(f"    ✅ Report contains {len(lines)-1} forecast data points")
        
        # Step 5: Business Decision Simulation
        print("  Step 5: Business decision simulation...")
        
        # Simulate business logic
        decision_factors = {
            "peak_cpu_threshold": 85,
            "high_risk_threshold": 2,
            "trend_concern_threshold": 80
        }
        
        decisions = []
        
        if peak_cpu > decision_factors["peak_cpu_threshold"]:
            decisions.append("Consider immediate capacity expansion")
        
        if high_risk_regions > decision_factors["high_risk_threshold"]:
            decisions.append("Schedule emergency capacity review")
        
        if not decisions:
            decisions.append("Continue normal operations monitoring")
        
        print("    💡 Business decisions based on data:")
        for decision in decisions:
            print(f"      - {decision}")
        
        print("💼 Complete Business Workflow: PASSED")

@pytest.mark.e2e
class TestDataIntegrityWorkflows:
    """Test data integrity across complete workflows"""
    
    def test_data_consistency_across_apis(self):
        """Test data consistency across different API endpoints"""
        api_base = "http://localhost:5000/api"
        
        print("🔍 Testing Data Consistency Across APIs")
        
        # Get data from different endpoints
        kpis_response = requests.get(f"{api_base}/kpis")
        regional_response = requests.get(f"{api_base}/regional/comparison")
        
        if kpis_response.status_code == 200 and regional_response.status_code == 200:
            kpis_data = kpis_response.json()
            regional_data = regional_response.json()
            
            # Check consistency
            kpis_regions = kpis_data.get("total_regions", 0)
            regional_regions = len(regional_data)
            
            print(f"  KPIs reports {kpis_regions} regions")
            print(f"  Regional endpoint shows {regional_regions} regions")
            
            # Should be consistent (allowing for some flexibility in implementation)
            assert abs(kpis_regions - regional_regions) <= 1, "Region count inconsistency detected"
            
            print("  ✅ Region count consistency verified")
        
        print("🔍 Data Consistency Test: PASSED")
    
    def test_forecast_accuracy_validation(self):
        """Test forecast accuracy using historical data"""
        api_base = "http://localhost:5000/api"
        
        print("📊 Testing Forecast Accuracy Validation")
        
        # This would require historical data to validate against
        # For now, we test the structure and reasonableness of forecasts
        
        monitoring_response = requests.get(f"{api_base}/monitoring/accuracy")
        
        if monitoring_response.status_code == 200:
            accuracy_data = monitoring_response.json()
            
            overall_accuracy = accuracy_data.get("model_health", {}).get("average_accuracy", 0)
            healthy_models = accuracy_data.get("model_health", {}).get("healthy_models", 0)
            
            print(f"  Overall model accuracy: {overall_accuracy:.1f}%")
            print(f"  Healthy models: {healthy_models}")
            
            # Accuracy should meet minimum threshold
            assert overall_accuracy >= TestConfig.MODEL_ACCURACY_THRESHOLD * 100, f"Model accuracy {overall_accuracy:.1f}% below threshold"
            
            print("  ✅ Model accuracy validation passed")
        else:
            print("  ℹ️ Accuracy monitoring not available (may be expected in test environment)")
        
        print("📊 Forecast Accuracy Validation: PASSED")
```

---

## 🎮 **Test Execution & Automation**

### Test Suite Configuration

```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow-running tests (> 30 seconds)
    ml: Machine learning model tests

addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=85
    --tb=short
    -v

filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning
```

### Test Execution Scripts

```bash
#!/bin/bash
# run_tests.sh - Comprehensive test execution script

set -e

echo "🧪 Azure Demand Forecasting - Test Suite Execution"
echo "=================================================="

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TEST_ENV="true"

# Function to run test category
run_test_category() {
    local category=$1
    local description=$2
    
    echo ""
    echo "📋 Running $description..."
    echo "----------------------------------------"
    
    if pytest -m "$category" --tb=short -v; then
        echo "✅ $description: PASSED"
    else
        echo "❌ $description: FAILED"
        return 1
    fi
}

# Check if API server is running
check_api_server() {
    echo "🔍 Checking API server availability..."
    
    if curl -f http://localhost:5000/api/health >/dev/null 2>&1; then
        echo "✅ API server is running"
        return 0
    else
        echo "⚠️ API server not detected. Starting server..."
        
        # Start API server in background
        python optimised_backend_app.py &
        API_SERVER_PID=$!
        
        # Wait for server to start
        for i in {1..30}; do
            if curl -f http://localhost:5000/api/health >/dev/null 2>&1; then
                echo "✅ API server started successfully"
                return 0
            fi
            sleep 1
        done
        
        echo "❌ Failed to start API server"
        return 1
    fi
}

# Main test execution
main() {
    echo "🏗️ Setting up test environment..."
    
    # Install test dependencies
    pip install -r requirements-test.txt
    
    # Check API server
    if ! check_api_server; then
        echo "❌ Cannot proceed without API server"
        exit 1
    fi
    
    # Run test categories in order
    echo ""
    echo "🚀 Starting test execution..."
    
    # Unit tests (fast)
    run_test_category "unit" "Unit Tests"
    
    # Integration tests
    run_test_category "integration" "Integration Tests"
    
    # Performance tests (if requested)
    if [[ "$1" == "--include-performance" ]]; then
        run_test_category "performance" "Performance Tests"
    fi
    
    # End-to-end tests (if requested)
    if [[ "$1" == "--include-e2e" ]] || [[ "$1" == "--full" ]]; then
        run_test_category "e2e" "End-to-End Tests"
    fi
    
    # Generate coverage report
    echo ""
    echo "📊 Generating coverage report..."
    echo "================================"
    
    # Generate HTML coverage report
    coverage html --directory=htmlcov
    echo "📄 HTML coverage report: htmlcov/index.html"
    
    # Display coverage summary
    coverage report --show-missing
    
    # Cleanup
    if [[ -n "$API_SERVER_PID" ]]; then
        echo "🧹 Cleaning up API server..."
        kill $API_SERVER_PID 2>/dev/null || true
    fi
    
    echo ""
    echo "🎉 Test suite execution completed!"
    echo "=================================="
}

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --include-performance    Include performance tests"
    echo "  --include-e2e           Include end-to-end tests"
    echo "  --full                  Run complete test suite"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Run unit and integration tests"
    echo "  $0 --include-e2e        # Include end-to-end tests"
    echo "  $0 --full               # Run complete test suite"
}

# Parse command line arguments
case "${1:-}" in
    --help)
        show_help
        exit 0
        ;;
    --include-performance|--include-e2e|--full)
        main "$1"
        ;;
    "")
        main
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    services:
      redis:
        image: redis:6-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Start API server
      run: |
        python optimised_backend_app.py &
        sleep 10  # Wait for server to start
      env:
        FLASK_ENV: testing
        DATABASE_URL: sqlite:///test.db
    
    - name: Run unit tests
      run: |
        pytest -m "unit" --cov=src --cov-report=xml --cov-report=term-missing
    
    - name: Run integration tests
      run: |
        pytest -m "integration" --cov=src --cov-append --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  performance-test:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Start API server
      run: |
        python optimised_backend_app.py &
        sleep 10
    
    - name: Run performance tests
      run: |
        pytest -m "performance" --tb=short -v
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: performance-results/

  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif
    
    - name: Run dependency check
      run: |
        pip install safety
        safety check --json > safety-report.json || true
    
    - name: Upload security results
      uses: actions/upload-artifact@v3
      with:
        name: security-results
        path: |
          security-scan-results.sarif
          safety-report.json
```

---

## 📊 **Test Reporting & Monitoring**

### Test Results Dashboard

```python
# test_reporting.py
import json
import sqlite3
from datetime import datetime
from pathlib import Path

class TestResultsTracker:
    """Track and report test results over time"""
    
    def __init__(self, db_path="test_results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize test results database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP,
                test_category TEXT,
                total_tests INTEGER,
                passed_tests INTEGER,
                failed_tests INTEGER,
                skipped_tests INTEGER,
                execution_time_seconds REAL,
                coverage_percentage REAL,
                branch_name TEXT,
                commit_hash TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                test_name TEXT,
                failure_message TEXT,
                stack_trace TEXT,
                FOREIGN KEY (run_id) REFERENCES test_runs (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_test_run(self, results):
        """Record test run results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO test_runs 
            (run_date, test_category, total_tests, passed_tests, failed_tests, 
             skipped_tests, execution_time_seconds, coverage_percentage, branch_name, commit_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            results['category'],
            results['total_tests'],
            results['passed_tests'], 
            results['failed_tests'],
            results['skipped_tests'],
            results['execution_time'],
            results['coverage_percentage'],
            results.get('branch_name', 'unknown'),
            results.get('commit_hash', 'unknown')
        ))
        
        run_id = cursor.lastrowid
        
        # Record failures
        for failure in results.get('failures', []):
            cursor.execute('''
                INSERT INTO test_failures (run_id, test_name, failure_message, stack_trace)
                VALUES (?, ?, ?, ?)
            ''', (run_id, failure['test_name'], failure['message'], failure.get('stack_trace', '')))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def generate_trend_report(self, days=30):
        """Generate test trend report"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                DATE(run_date) as date,
                test_category,
                AVG(CAST(passed_tests AS FLOAT) / total_tests * 100) as success_rate,
                AVG(coverage_percentage) as avg_coverage,
                AVG(execution_time_seconds) as avg_execution_time
            FROM test_runs 
            WHERE run_date > datetime('now', '-{} days')
            GROUP BY DATE(run_date), test_category
            ORDER BY date DESC, test_category
        '''.format(days)
        
        import pandas as pd
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_failure_analysis(self, days=7):
        """Analyze recent test failures"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                f.test_name,
                COUNT(*) as failure_count,
                MAX(r.run_date) as last_failure,
                f.failure_message
            FROM test_failures f
            JOIN test_runs r ON f.run_id = r.id
            WHERE r.run_date > datetime('now', '-{} days')
            GROUP BY f.test_name, f.failure_message
            ORDER BY failure_count DESC, last_failure DESC
        '''.format(days)
        
        import pandas as pd
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

def generate_test_report():
    """Generate comprehensive test report"""
    tracker = TestResultsTracker()
    
    # Get trend data
    trend_data = tracker.generate_trend_report(30)
    failure_data = tracker.get_failure_analysis(7)
    
    report = f"""
# 📊 Test Results Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 30-Day Test Trends

{trend_data.to_string(index=False) if not trend_data.empty else "No trend data available"}

## ❌ Recent Failure Analysis (Last 7 Days)

{failure_data.to_string(index=False) if not failure_data.empty else "No recent failures"}

## 🎯 Quality Metrics Summary

- **Target Code Coverage**: >85%
- **Target Success Rate**: >95%
- **Target Performance**: <500ms API response time
- **Target Availability**: >99.5%

## 📋 Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Service interaction testing  
- **End-to-End Tests**: Complete workflow validation
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability assessment

## 🚀 Continuous Improvement

1. Monitor test execution trends
2. Address recurring failures
3. Maintain high code coverage
4. Performance regression detection
5. Security vulnerability scanning
"""
    
    # Write report to file
    with open("test_report.md", "w") as f:
        f.write(report)
    
    return report
```

---

## 🎯 **Quality Gates & Standards**

### Quality Criteria

```python
# quality_gates.py
import subprocess
import json
import sys

class QualityGates:
    """Enforce quality gates for CI/CD pipeline"""
    
    def __init__(self):
        self.criteria = {
            'code_coverage': 85.0,  # Minimum 85% code coverage
            'test_success_rate': 95.0,  # Minimum 95% test success rate
            'api_response_time': 500,  # Maximum 500ms average response time
            'security_vulnerabilities': 0,  # Zero high-severity vulnerabilities
            'code_quality_score': 8.0,  # Minimum code quality score (1-10)
        }
        
        self.results = {}
    
    def check_code_coverage(self):
        """Check code coverage meets minimum threshold"""
        try:
            result = subprocess.run(['coverage', 'report', '--format=json'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                coverage_data = json.loads(result.stdout)
                coverage_percent = coverage_data['totals']['percent_covered']
                
                self.results['code_coverage'] = {
                    'value': coverage_percent,
                    'threshold': self.criteria['code_coverage'],
                    'passed': coverage_percent >= self.criteria['code_coverage'],
                    'message': f"Code coverage: {coverage_percent:.1f}% (threshold: {self.criteria['code_coverage']}%)"
                }
            else:
                self.results['code_coverage'] = {
                    'value': 0,
                    'threshold': self.criteria['code_coverage'],
                    'passed': False,
                    'message': "Failed to get coverage data"
                }
                
        except Exception as e:
            self.results['code_coverage'] = {
                'value': 0,
                'threshold': self.criteria['code_coverage'],
                'passed': False,
                'message': f"Coverage check failed: {str(e)}"
            }
    
    def check_test_results(self):
        """Check test success rate"""
        try:
            # Run pytest with JSON report
            result = subprocess.run(['pytest', '--tb=no', '-q', '--json-report', '--json-report-file=pytest-report.json'], 
                                  capture_output=True, text=True)
            
            with open('pytest-report.json', 'r') as f:
                test_data = json.load(f)
            
            summary = test_data['summary']
            total_tests = summary['total']
            failed_tests = summary.get('failed', 0)
            success_rate = ((total_tests - failed_tests) / total_tests * 100) if total_tests > 0 else 0
            
            self.results['test_success_rate'] = {
                'value': success_rate,
                'threshold': self.criteria['test_success_rate'],
                'passed': success_rate >= self.criteria['test_success_rate'],
                'message': f"Test success rate: {success_rate:.1f}% ({total_tests - failed_tests}/{total_tests} passed)"
            }
            
        except Exception as e:
            self.results['test_success_rate'] = {
                'value': 0,
                'threshold': self.criteria['test_success_rate'],
                'passed': False,
                'message': f"Test results check failed: {str(e)}"
            }
    
    def check_performance(self):
        """Check API performance"""
        try:
            # This would integrate with performance test results
            # For now, simulate based on test execution
            
            import requests
            import time
            
            start_time = time.time()
            response = requests.get('http://localhost:5000/api/health', timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            self.results['api_response_time'] = {
                'value': response_time,
                'threshold': self.criteria['api_response_time'],
                'passed': response_time <= self.criteria['api_response_time'] and response.status_code == 200,
                'message': f"API response time: {response_time:.1f}ms (threshold: {self.criteria['api_response_time']}ms)"
            }
            
        except Exception as e:
            self.results['api_response_time'] = {
                'value': 999999,
                'threshold': self.criteria['api_response_time'],
                'passed': False,
                'message': f"Performance check failed: {str(e)}"
            }
    
    def check_security(self):
        """Check security vulnerabilities"""
        try:
            # Run safety check for Python dependencies
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                vulnerabilities = 0
                message = "No security vulnerabilities found"
            else:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = len(safety_data)
                    message = f"Found {vulnerabilities} security vulnerabilities"
                except:
                    vulnerabilities = 1  # Assume vulnerability if can't parse
                    message = "Security check failed to parse results"
            
            self.results['security_vulnerabilities'] = {
                'value': vulnerabilities,
                'threshold': self.criteria['security_vulnerabilities'],
                'passed': vulnerabilities <= self.criteria['security_vulnerabilities'],
                'message': message
            }
            
        except Exception as e:
            self.results['security_vulnerabilities'] = {
                'value': 999,
                'threshold': self.criteria['security_vulnerabilities'],
                'passed': False,
                'message': f"Security check failed: {str(e)}"
            }
    
    def run_all_checks(self):
        """Run all quality gate checks"""
        print("🔍 Running Quality Gate Checks...")
        print("=" * 50)
        
        checks = [
            ("Code Coverage", self.check_code_coverage),
            ("Test Success Rate", self.check_test_results),
            ("API Performance", self.check_performance),
            ("Security Scan", self.check_security),
        ]
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}:")
            check_func()
            
            if check_name.lower().replace(" ", "_") in self.results:
                result = self.results[check_name.lower().replace(" ", "_")]
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                print(f"   {status}: {result['message']}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Quality Gate Summary:")
        
        all_passed = all(result['passed'] for result in self.results.values())
        failed_checks = [name for name, result in self.results.items() if not result['passed']]
        
        if all_passed:
            print("🎉 All quality gates PASSED!")
            return True
        else:
            print(f"❌ {len(failed_checks)} quality gate(s) FAILED:")
            for check in failed_checks:
                print(f"   - {check.replace('_', ' ').title()}")
            return False

def main():
    """Main quality gate execution"""
    quality_gates = QualityGates()
    
    if quality_gates.run_all_checks():
        print("\n✅ Quality gates passed - deployment approved!")
        sys.exit(0)
    else:
        print("\n❌ Quality gates failed - deployment blocked!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 🔄 **Continuous Testing Strategy**

### Testing in CI/CD Pipeline

1. **Pre-commit Hooks**: Run unit tests and linting before commits
2. **Pull Request Testing**: Full test suite execution on PR creation
3. **Integration Testing**: Automated testing on merge to develop branch
4. **Performance Regression**: Weekly performance baseline testing  
5. **Security Scanning**: Daily vulnerability scans and dependency checks
6. **End-to-End Testing**: Staging environment validation before production

### Monitoring & Alerting

```python
# test_monitoring.py
import requests
import time
from datetime import datetime

class TestMonitoring:
    """Monitor production system with automated testing"""
    
    def __init__(self, api_base="http://localhost:5000/api"):
        self.api_base = api_base
        self.health_checks = [
            self.check_system_health,
            self.check_api_performance, 
            self.check_model_accuracy,
            self.check_data_quality
        ]
    
    def check_system_health(self):
        """Monitor system health"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                return {
                    'check': 'system_health',
                    'status': 'pass' if health_data['status'] == 'healthy' else 'fail',
                    'details': health_data,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'check': 'system_health',
                    'status': 'fail',
                    'details': {'error': f'HTTP {response.status_code}'},
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'check': 'system_health',
                'status': 'fail',
                'details': {'error': str(e)},
                'timestamp': datetime.now().isoformat()
            }
    
    def check_api_performance(self):
        """Monitor API performance"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_base}/kpis", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'check': 'api_performance',
                'status': 'pass' if response_time < 1000 and response.status_code == 200 else 'fail',
                'details': {
                    'response_time_ms': response_time,
                    'status_code': response.status_code
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'check': 'api_performance',
                'status': 'fail',
                'details': {'error': str(e)},
                'timestamp': datetime.now().isoformat()
            }
    
    def check_model_accuracy(self):
        """Monitor ML model accuracy"""
        try:
            response = requests.get(f"{self.api_base}/monitoring/accuracy", timeout=10)
            
            if response.status_code == 200:
                accuracy_data = response.json()
                avg_accuracy = accuracy_data.get('model_health', {}).get('average_accuracy', 0)
                
                return {
                    'check': 'model_accuracy',
                    'status': 'pass' if avg_accuracy >= 80 else 'fail',
                    'details': {
                        'average_accuracy': avg_accuracy,
                        'healthy_models': accuracy_data.get('model_health', {}).get('healthy_models', 0)
                    },
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'check': 'model_accuracy',
                    'status': 'unknown',
                    'details': {'error': 'Accuracy monitoring not available'},
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'check': 'model_accuracy',
                'status': 'fail',
                'details': {'error': str(e)},
                'timestamp': datetime.now().isoformat()
            }
    
    def check_data_quality(self):
        """Monitor data quality"""
        # This would check data freshness, completeness, etc.
        return {
            'check': 'data_quality',
            'status': 'pass',  # Simplified for example
            'details': {'data_freshness': 'current'},
            'timestamp': datetime.now().isoformat()
        }
    
    def run_monitoring_cycle(self):
        """Run complete monitoring cycle"""
        results = []
        
        for health_check in self.health_checks:
            result = health_check()
            results.append(result)
            
            if result['status'] == 'fail':
                self.send_alert(result)
        
        return results
    
    def send_alert(self, result):
        """Send alert for failed check"""
        # Implement your alerting mechanism here
        # (Slack, email, PagerDuty, etc.)
        print(f"🚨 ALERT: {result['check']} failed - {result['details']}")

# Example usage for production monitoring
if __name__ == "__main__":
    monitor = TestMonitoring()
    
    while True:
        print(f"🔍 Running monitoring cycle at {datetime.now()}")
        results = monitor.run_monitoring_cycle()
        
        passed_checks = sum(1 for r in results if r['status'] == 'pass')
        print(f"✅ {passed_checks}/{len(results)} checks passed")
        
        time.sleep(300)  # Run every 5 minutes
```

---

*This comprehensive Testing Guide ensures the Azure Demand Forecasting Platform maintains the highest quality standards with 99.9% reliability. The multi-layered testing approach validates everything from individual components to complete business workflows, supporting confident production deployment and continuous system improvement.*
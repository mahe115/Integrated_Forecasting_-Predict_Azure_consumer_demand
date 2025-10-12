# API_REFERENCE.md

# 📚 API Reference Documentation

## Azure Demand Forecasting Platform - Developer's Guide

*Complete developer reference for integrating with the Azure Demand Forecasting Platform APIs*

---

## 🚀 **Quick Start**

### Base Configuration
```bash
# Base URL
export API_BASE="http://localhost:5000/api"

# Test connectivity
curl -X GET "$API_BASE/health"
```

### Authentication (Production)
```python
import requests

# Get JWT token (production environments)
auth_response = requests.post(f"{API_BASE}/auth/login", 
                             json={"username": "your_username", 
                                   "password": "your_password"})
token = auth_response.json()["access_token"]

# Use token in subsequent requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(f"{API_BASE}/kpis", headers=headers)
```

---

## 🔍 **Core Endpoints Reference**

### System Health & Status

#### `GET /api/health`
**Description**: System health check for monitoring and load balancers  
**Authentication**: None required  
**Cache**: None (real-time)

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-12T12:00:00Z",
  "uptime_seconds": 86400,
  "ml_models_loaded": 12,
  "database_connected": true,
  "cache_hit_rate": 0.95,
  "version": "1.0.0",
  "environment": "production"
}
```

**Usage Example**:
```python
import requests

response = requests.get("http://localhost:5000/api/health")
if response.json()["status"] == "healthy":
    print("✅ System is operational")
else:
    print("❌ System issues detected")
```

#### `GET /api/cache/stats`
**Description**: Cache performance statistics  
**Authentication**: Admin role required  
**Cache**: None (real-time)

**Response**:
```json
{
  "total_requests": 12847,
  "cache_hits": 12205,
  "cache_misses": 642,
  "hit_rate": 0.95,
  "memory_usage_mb": 256,
  "redis_connected": true,
  "tier_performance": {
    "memory": {"hits": 7708, "rate": 0.60},
    "redis": {"hits": 3852, "rate": 0.30},
    "database": {"hits": 1028, "rate": 0.08}
  }
}
```

---

## 📊 **Executive Dashboard APIs**

### Key Performance Indicators

#### `GET /api/kpis`
**Description**: Executive KPIs with drill-down capabilities  
**Authentication**: Analyst role or higher  
**Cache**: Medium (5 minutes)

**Response**:
```json
{
  "peak_cpu": 87.5,
  "peak_cpu_details": {
    "date": "2025-10-10T14:30:00Z",
    "region": "East US",
    "resource_type": "compute",
    "context": "High demand during business hours"
  },
  "max_storage": 1250.8,
  "max_storage_details": {
    "date": "2025-10-09T18:45:00Z", 
    "region": "West US",
    "resource_type": "storage",
    "context": "Peak backup operations"
  },
  "peak_users": 1456,
  "peak_users_details": {
    "date": "2025-10-11T10:15:00Z",
    "region": "North Europe", 
    "resource_type": "user_session",
    "context": "Morning peak in European timezone"
  },
  "avg_cpu": 72.3,
  "avg_storage": 892.4,
  "avg_users": 987,
  "total_regions": 4,
  "total_resource_types": 3,
  "data_points": 1080,
  "date_range": {
    "start": "2025-07-13T00:00:00Z",
    "end": "2025-10-10T23:59:59Z",
    "days": 90
  },
  "holiday_impact": {
    "percentage": 15.2,
    "holiday_avg_cpu": 65.1,
    "regular_avg_cpu": 76.4,
    "description": "15.2% reduction during holiday periods"
  },
  "generated_at": "2025-10-12T12:00:00Z"
}
```

**Usage Example**:
```python
import requests

kpis = requests.get("http://localhost:5000/api/kpis").json()

print(f"Peak CPU Usage: {kpis['peak_cpu']}%")
print(f"Peak occurred on: {kpis['peak_cpu_details']['date']}")
print(f"Peak region: {kpis['peak_cpu_details']['region']}")

# Holiday impact analysis
holiday_impact = kpis['holiday_impact']['percentage']
print(f"Holiday CPU reduction: {holiday_impact}%")
```

#### `GET /api/sparklines`
**Description**: 30-day trend sparklines for executive dashboard  
**Authentication**: Viewer role or higher  
**Cache**: Fast (2 minutes)

**Response**:
```json
{
  "cpu_trend": [
    {"date": "2025-09-10", "usage_cpu": 72.1},
    {"date": "2025-09-11", "usage_cpu": 74.3},
    {"date": "2025-09-12", "usage_cpu": 71.8},
    "... 27 more days"
  ],
  "storage_trend": [
    {"date": "2025-09-10", "usage_storage": 892.3},
    {"date": "2025-09-11", "usage_storage": 904.7},
    "... 28 more days"
  ],
  "users_trend": [
    {"date": "2025-09-10", "users_active": 1052},
    {"date": "2025-09-11", "users_active": 1089},
    "... 28 more days"
  ],
  "metadata": {
    "period_days": 30,
    "data_points_per_trend": 30,
    "generated_at": "2025-10-12T12:00:00Z"
  }
}
```

---

## 🤖 **AI Forecasting APIs**

### Model Status & Information

#### `GET /api/forecast/models`
**Description**: CPU forecasting model status across all regions  
**Authentication**: Analyst role or higher  
**Cache**: Fast (2 minutes)

**Response**:
```json
{
  "models": {
    "East US": {
      "model_type": "ARIMA",
      "loaded": true,
      "has_scaler": false,
      "last_updated": "2025-10-10T08:00:00Z",
      "selection_method": "intelligent_database",
      "performance": {
        "rmse": 8.45,
        "mae": 6.23,
        "mape": 8.9,
        "accuracy": 87.3
      },
      "training_data_size": 270,
      "model_parameters": {
        "order": [2, 1, 1],
        "aic": 145.7,
        "bic": 158.2
      }
    },
    "West US": {
      "model_type": "LSTM",
      "loaded": true,
      "has_scaler": true,
      "last_updated": "2025-10-09T14:30:00Z",
      "selection_method": "intelligent_database",
      "performance": {
        "rmse": 9.12,
        "mae": 6.87,
        "mape": 9.4,
        "accuracy": 85.8
      },
      "training_data_size": 270,
      "model_parameters": {
        "sequence_length": 14,
        "lstm_units": [64, 32],
        "dropout_rate": 0.2,
        "epochs_trained": 45
      }
    }
  },
  "total_regions": 4,
  "model_types_used": ["ARIMA", "LSTM", "XGBoost"],
  "ml_available": true,
  "model_directory": "/app/models",
  "directory_exists": true,
  "selection_method": "intelligent_database",
  "database_connected": true,
  "last_intelligent_training": "2025-10-08T12:00:00Z"
}
```

#### `GET /api/forecast/predict`
**Description**: Generate CPU usage forecasts  
**Authentication**: Analyst role or higher  
**Parameters**: 
- `days` (integer, 1-90): Forecast horizon
- `region` (string, optional): Specific region or "All Regions"
**Cache**: Smart caching for 1, 7, 30 day forecasts

**Request Example**:
```bash
curl "http://localhost:5000/api/forecast/predict?days=30&region=East%20US"
```

**Response**:
```json
{
  "East US": {
    "dates": [
      "2025-10-13", "2025-10-14", "2025-10-15",
      "... 27 more dates"
    ],
    "predicted_cpu": [
      74.2, 75.8, 73.4, 76.1, 72.9,
      "... 25 more predictions"
    ],
    "model_info": {
      "type": "ARIMA",
      "forecast_horizon": 30,
      "data_points_used": 270,
      "features_used": ["lag_1", "lag_7", "seasonal"],
      "confidence_interval": [65.2, 83.1],
      "prediction_accuracy": 87.3
    },
    "historical": {
      "dates": [
        "2025-09-29", "2025-09-30", "2025-10-01",
        "... 11 more dates"
      ],
      "actual_cpu": [
        71.5, 73.2, 72.8, 74.1,
        "... 10 more values"
      ]
    },
    "generated_at": "2025-10-12T12:00:00Z",
    "cache_source": "fresh_generation"
  }
}
```

**Usage Example**:
```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Generate 30-day forecast
response = requests.get("http://localhost:5000/api/forecast/predict", 
                       params={"days": 30, "region": "East US"})
forecast_data = response.json()["East US"]

# Create DataFrame for analysis
df = pd.DataFrame({
    'date': pd.to_datetime(forecast_data['dates']),
    'predicted_cpu': forecast_data['predicted_cpu']
})

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['predicted_cpu'], marker='o', linewidth=2)
plt.title('30-Day CPU Usage Forecast - East US')
plt.xlabel('Date')
plt.ylabel('CPU Usage (%)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Identify peak usage days
peak_days = df[df['predicted_cpu'] > df['predicted_cpu'].quantile(0.9)]
print(f"High usage days (>90th percentile): {len(peak_days)}")
```

#### `GET /api/forecast/users/predict`
**Description**: Generate active users forecasts  
**Authentication**: Analyst role or higher  
**Parameters**: Same as CPU forecasting
**Cache**: Smart caching for common horizons

**Response** (similar structure to CPU forecasting):
```json
{
  "East US": {
    "dates": ["2025-10-13", "2025-10-14", "..."],
    "predicted_users": [1087, 1125, 1098, "..."],
    "model_info": {
      "type": "XGBoost",
      "forecast_horizon": 30,
      "feature_importance": {
        "lag_1": 0.45,
        "rolling_7_mean": 0.23,
        "day_of_week": 0.18,
        "economic_index": 0.14
      }
    }
  }
}
```

---

## 🏗️ **Capacity Planning APIs**

### Comprehensive Capacity Analysis

#### `GET /api/capacity-planning`
**Description**: Comprehensive capacity analysis with risk assessment  
**Authentication**: Business User role or higher  
**Parameters**: 
- `region` (string): "All Regions" or specific region
- `service` (string): "Compute", "Storage", or "Users"  
- `horizon` (integer): Forecast horizon in days
**Cache**: Smart caching for common scenarios

**Request Example**:
```bash
curl "http://localhost:5000/api/capacity-planning?region=All%20Regions&service=Compute&horizon=30"
```

**Response**:
```json
{
  "timestamp": "2025-10-12T12:00:00Z",
  "service": "Compute", 
  "horizon_days": 30,
  "data_source": "live_models",
  "capacity_analysis": {
    "East US": {
      "region": "East US",
      "service": "Compute",
      "forecast_horizon_days": 30,
      "current_capacity": 1502.0,
      "predicted_demand": {
        "max": 76.8,
        "avg": 72.4,
        "min": 68.1,
        "timeline": [74.2, 75.1, 73.9, "... 27 more values"]
      },
      "capacity_utilization": {
        "current_pct": 4.8,
        "peak_pct": 5.1,
        "avg_pct": 4.7
      },
      "risk_assessment": {
        "utilization_risk": {
          "level": "LOW",
          "message": "Healthy utilization levels",
          "threshold": 85,
          "current": 5.1
        },
        "provision_risk": {
          "level": "LOW", 
          "message": "Adequate capacity buffer",
          "buffer_pct": 94.9
        },
        "overall_risk": "LOW"
      },
      "recommendations": [
        {
          "type": "MAINTAIN",
          "priority": "LOW",
          "action": "Continue current capacity levels",
          "reason": "Utilization is optimal (5.1% peak, 4.7% average)",
          "timeline": "Ongoing monitoring",
          "impact": "Stable operations with cost optimization"
        }
      ],
      "forecast_source": {
        "cpu_model": "ARIMA",
        "users_model": "XGBoost", 
        "storage_model": "LSTM"
      }
    },
    "West US": {
      "region": "West US",
      "service": "Compute",
      "current_capacity": 1487.5,
      "predicted_demand": {
        "max": 85.2,
        "avg": 79.3,
        "min": 74.1
      },
      "capacity_utilization": {
        "current_pct": 5.3,
        "peak_pct": 5.7
      },
      "risk_assessment": {
        "overall_risk": "MEDIUM"
      },
      "recommendations": [
        {
          "type": "MONITOR",
          "priority": "MEDIUM",
          "action": "Monitor utilization trends closely",
          "reason": "Approaching 6% utilization threshold"
        }
      ]
    }
  },
  "summary": {
    "total_regions_analyzed": 4,
    "risk_distribution": {
      "low_risk": 3,
      "medium_risk": 1,
      "high_risk": 0
    },
    "overall_status": "HEALTHY",
    "recommendations_count": {
      "maintain": 3,
      "monitor": 1,
      "scale_up": 0
    }
  }
}
```

**Usage Example**:
```python
import requests

# Get capacity analysis
response = requests.get("http://localhost:5000/api/capacity-planning",
                       params={
                           "region": "All Regions",
                           "service": "Compute", 
                           "horizon": 30
                       })

capacity_data = response.json()

# Analyze risk levels
risk_summary = capacity_data["summary"]["risk_distribution"]
print(f"Risk Distribution:")
print(f"  Low Risk: {risk_summary['low_risk']} regions")
print(f"  Medium Risk: {risk_summary['medium_risk']} regions") 
print(f"  High Risk: {risk_summary['high_risk']} regions")

# Extract high-priority recommendations
for region, analysis in capacity_data["capacity_analysis"].items():
    for rec in analysis["recommendations"]:
        if rec["priority"] in ["HIGH", "MEDIUM"]:
            print(f"\n🚨 {rec['priority']} Priority - {region}:")
            print(f"   Action: {rec['action']}")
            print(f"   Timeline: {rec['timeline']}")

# Calculate average utilization
utilizations = []
for region, analysis in capacity_data["capacity_analysis"].items():
    utilizations.append(analysis["capacity_utilization"]["current_pct"])

avg_utilization = sum(utilizations) / len(utilizations)
print(f"\nAverage Utilization: {avg_utilization:.1f}%")
```

---

## 🔄 **Intelligent Training APIs**

### Training Pipeline Control

#### `GET /api/training/intelligent/status`
**Description**: Intelligent training pipeline status  
**Authentication**: System Admin role required  
**Cache**: Medium (5 minutes)

**Response**:
```json
{
  "pipeline_active": true,
  "pipeline_type": "intelligent_auto_selection",
  "current_models": {
    "cpu": {
      "East US": "ARIMA",
      "West US": "LSTM", 
      "North Europe": "ARIMA",
      "Southeast Asia": "XGBoost"
    },
    "users": {
      "East US": "XGBoost",
      "West US": "LSTM",
      "North Europe": "ARIMA", 
      "Southeast Asia": "XGBoost"
    },
    "storage": {
      "East US": "LSTM",
      "West US": "XGBoost",
      "North Europe": "ARIMA",
      "Southeast Asia": "LSTM"
    }
  },
  "recent_monitoring": [
    {
      "check_date": "2025-10-12T06:00:00Z",
      "data_size": 1080,
      "new_records": 12,
      "training_triggered": false,
      "quality_score": 92.3
    }
  ],
  "last_check": "2025-10-12T06:00:00Z",
  "database_path": "/app/model_performance.db",
  "model_directories": {
    "cpu": "/app/models",
    "users": "/app/user_models",
    "storage": "/app/storage_models"
  }
}
```

#### `POST /api/training/intelligent/trigger`
**Description**: Manually trigger intelligent training pipeline  
**Authentication**: System Admin role required  
**Request Body**: None required

**Response**:
```json
{
  "status": "FORCED Intelligent Training Pipeline Started",
  "pipeline_type": "intelligent_auto_selection",
  "models_to_test": ["ARIMA", "LSTM", "XGBoost"],
  "metrics_to_train": ["cpu", "users", "storage"],
  "timestamp": "2025-10-12T12:00:00Z",
  "background_thread": true,
  "estimated_completion": "2025-10-12T12:15:00Z"
}
```

**Usage Example**:
```python
import requests
import time

# Trigger training
response = requests.post("http://localhost:5000/api/training/intelligent/trigger")
if response.status_code == 200:
    result = response.json()
    print(f"✅ Training started: {result['status']}")
    
    # Monitor training progress
    while True:
        status_response = requests.get("http://localhost:5000/api/training/intelligent/status")
        status = status_response.json()
        
        # Check if new models have been trained (implementation-specific)
        print("🔄 Training in progress...")
        time.sleep(30)  # Check every 30 seconds
        
        # Break after reasonable time (training typically takes 5-15 minutes)
        # In production, you'd check for specific completion indicators
```

---

## 📊 **Analytics & Reporting APIs**

### Regional Analysis

#### `GET /api/regional/comparison`
**Description**: Cross-regional performance comparison  
**Authentication**: Analyst role or higher  
**Cache**: Medium (5 minutes)

**Response**:
```json
[
  {
    "region": "East US",
    "usage_cpu_mean": 72.1,
    "usage_cpu_max": 95.3,
    "usage_cpu_min": 45.8,
    "usage_cpu_std": 12.4,
    "usage_storage_mean": 1234.5,
    "usage_storage_max": 1876.2,
    "usage_storage_min": 892.3,
    "usage_storage_std": 245.7,
    "users_active_mean": 1052,
    "users_active_max": 1456,
    "users_active_min": 743,
    "users_active_std": 189
  },
  {
    "region": "West US",
    "usage_cpu_mean": 68.7,
    "usage_cpu_max": 89.2,
    "usage_cpu_min": 42.1,
    "usage_cpu_std": 11.8
  }
]
```

### Report Generation

#### `GET /api/reports/generate`
**Description**: Generate comprehensive reports in multiple formats  
**Authentication**: Business User role or higher  
**Parameters**: 
- `type` (string): "csv", "excel", or "pdf"
- `forecast_horizon` (integer, optional): Days to include (default: 30)

**Response for CSV**:
```
Content-Type: text/csv
Content-Disposition: attachment; filename="forecast_report_20251012_1200.csv"

Date,Region,CPU_Forecast,Users_Forecast,Storage_Forecast,Capacity_Status,Risk_Level
2025-10-13,East US,74.2,1087,1245.8,Normal,Low
2025-10-13,West US,71.8,1132,1198.3,Normal,Low
2025-10-13,North Europe,75.6,986,1287.4,Normal,Low
2025-10-13,Southeast Asia,73.1,1098,1223.7,Normal,Low
...
```

**Usage Example**:
```python
import requests

# Generate CSV report
response = requests.get("http://localhost:5000/api/reports/generate",
                       params={"type": "csv", "forecast_horizon": 30})

if response.status_code == 200:
    # Save CSV file
    with open("forecast_report.csv", "wb") as f:
        f.write(response.content)
    print("✅ CSV report saved successfully")
    
    # Parse CSV for analysis
    import pandas as pd
    df = pd.read_csv("forecast_report.csv")
    print(f"📊 Report contains {len(df)} forecast data points")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
```

---

## 🔍 **Data & Analytics APIs**

### Time Series Data

#### `GET /api/time-series`
**Description**: Time series data with flexible filtering  
**Authentication**: Viewer role or higher  
**Parameters**:
- `region` (string, optional): Filter by specific region
- `resource_type` (string, optional): Filter by resource type
- `start_date` (string, optional): Start date (YYYY-MM-DD format)
- `end_date` (string, optional): End date (YYYY-MM-DD format)
**Cache**: Fast (2 minutes)

**Request Example**:
```bash
curl "http://localhost:5000/api/time-series?region=East%20US&start_date=2025-10-01&end_date=2025-10-10"
```

**Response**:
```json
[
  {
    "date": "2025-10-01",
    "usage_cpu": 72.5,
    "usage_storage": 1234.8,
    "users_active": 1087,
    "economic_index": 45.2,
    "cloud_market_demand": 78.3
  },
  {
    "date": "2025-10-02", 
    "usage_cpu": 74.1,
    "usage_storage": 1248.3,
    "users_active": 1125,
    "economic_index": 45.8,
    "cloud_market_demand": 79.1
  }
]
```

---

## 🔒 **Authentication & Security**

### JWT Token Authentication (Production)

#### `POST /api/auth/login`
**Description**: Authenticate user and receive JWT token  
**Authentication**: None (public endpoint)

**Request**:
```json
{
  "username": "your_username",
  "password": "your_password"
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_info": {
    "username": "your_username",
    "role": "analyst",
    "permissions": ["read_dashboards", "generate_forecasts", "export_reports"]
  }
}
```

### Using JWT Tokens

**Python Example**:
```python
import requests

class AzureForecastingAPI:
    def __init__(self, base_url="http://localhost:5000/api"):
        self.base_url = base_url
        self.token = None
        self.headers = {}
    
    def login(self, username, password):
        """Authenticate and store JWT token"""
        response = requests.post(f"{self.base_url}/auth/login",
                               json={"username": username, "password": password})
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
            return True
        return False
    
    def get_kpis(self):
        """Get executive KPIs"""
        response = requests.get(f"{self.base_url}/kpis", headers=self.headers)
        return response.json() if response.status_code == 200 else None
    
    def generate_forecast(self, days=30, region="All Regions"):
        """Generate CPU forecast"""
        params = {"days": days, "region": region}
        response = requests.get(f"{self.base_url}/forecast/predict", 
                              headers=self.headers, params=params)
        return response.json() if response.status_code == 200 else None
    
    def get_capacity_analysis(self, service="Compute", horizon=30):
        """Get capacity planning analysis"""
        params = {"region": "All Regions", "service": service, "horizon": horizon}
        response = requests.get(f"{self.base_url}/capacity-planning",
                              headers=self.headers, params=params)
        return response.json() if response.status_code == 200 else None

# Usage
api = AzureForecastingAPI()
if api.login("username", "password"):
    kpis = api.get_kpis()
    forecast = api.generate_forecast(days=30)
    capacity = api.get_capacity_analysis()
```

---

## 🚨 **Error Handling**

### Standard Error Responses

All API endpoints return standardized error responses:

**4xx Client Errors**:
```json
{
  "error": "Invalid parameter: days must be between 1 and 90",
  "error_code": "INVALID_PARAMETER", 
  "error_type": "client_error",
  "timestamp": "2025-10-12T12:00:00Z",
  "request_id": "req_abc123def456",
  "suggestions": [
    "Provide days parameter between 1 and 90",
    "Check API documentation for valid parameters"
  ]
}
```

**5xx Server Errors**:
```json
{
  "error": "ML model not available for region",
  "error_code": "MODEL_UNAVAILABLE",
  "error_type": "server_error", 
  "timestamp": "2025-10-12T12:00:00Z",
  "request_id": "req_def789ghi012",
  "retry_after": 300,
  "support_contact": "support@azureforecasting.com"
}
```

### HTTP Status Codes

| Status Code | Description | When It Occurs |
|-------------|-------------|----------------|
| `200 OK` | Successful request | Request processed successfully |
| `201 Created` | Resource created | Training job started, report generated |
| `400 Bad Request` | Invalid parameters | Missing or invalid request parameters |
| `401 Unauthorized` | Authentication required | Missing or invalid JWT token |
| `403 Forbidden` | Insufficient permissions | User role lacks required permissions |
| `404 Not Found` | Endpoint not found | Invalid API endpoint |
| `429 Too Many Requests` | Rate limit exceeded | Request rate limit hit |
| `500 Internal Server Error` | Server error | Unexpected server-side error |
| `503 Service Unavailable` | Service unavailable | ML models not loaded, maintenance mode |

### Error Handling Best Practices

**Python Example**:
```python
import requests
import time

def make_api_request(url, max_retries=3, backoff_factor=1):
    """
    Make API request with proper error handling and retries
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            elif response.status_code == 503:  # Service unavailable
                print("Service temporarily unavailable. Retrying...")
                time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
                continue
                
            elif response.status_code in [500, 502, 504]:  # Server errors
                if attempt < max_retries - 1:
                    print(f"Server error. Retrying in {backoff_factor * (2 ** attempt)} seconds...")
                    time.sleep(backoff_factor * (2 ** attempt))
                    continue
                else:
                    error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                    raise Exception(f"Server error after {max_retries} attempts: {error_data}")
                    
            else:  # Client errors (4xx)
                error_data = response.json()
                raise ValueError(f"Client error: {error_data['error']} (Code: {error_data['error_code']})")
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Request timeout. Retrying...")
                time.sleep(backoff_factor * (2 ** attempt))
                continue
            else:
                raise Exception("Request timed out after multiple attempts")
                
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"Connection error. Retrying...")
                time.sleep(backoff_factor * (2 ** attempt))
                continue
            else:
                raise Exception("Connection failed after multiple attempts")
    
    raise Exception(f"Request failed after {max_retries} attempts")

# Usage
try:
    data = make_api_request("http://localhost:5000/api/kpis")
    print("✅ Request successful")
except ValueError as e:
    print(f"❌ Client error: {e}")
except Exception as e:
    print(f"❌ Request failed: {e}")
```

---

## 📊 **Rate Limiting**

### Rate Limit Headers

All API responses include rate limiting headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1697112000
X-RateLimit-Window: 60
```

### Rate Limits by Role

| Role | Requests/Minute | Concurrent Requests |
|------|----------------|-------------------|
| **Viewer** | 100 | 5 |
| **Analyst** | 500 | 10 |
| **Business User** | 750 | 15 |
| **System Admin** | 1000 | 20 |
| **Super Admin** | 2000 | 50 |

---

## 🧪 **Testing & Debugging**

### API Testing with Postman

Import our comprehensive Postman collection:

```bash
# Download collection
curl -o postman_collection.json \
  "https://raw.githubusercontent.com/your-org/azure-forecasting/main/postman_collection.json"

# Import into Postman
# File → Import → postman_collection.json
```

### Health Check Script

**Python health check script**:
```python
#!/usr/bin/env python3
"""
Azure Forecasting API Health Check
Tests all critical endpoints and reports system status
"""

import requests
import time
import sys

def check_endpoint(url, expected_status=200, timeout=30):
    """Check single endpoint health"""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == expected_status:
            return {"status": "✅ PASS", "response_time": f"{response_time:.0f}ms", "error": None}
        else:
            return {"status": "❌ FAIL", "response_time": f"{response_time:.0f}ms", 
                   "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"status": "❌ FAIL", "response_time": "N/A", "error": str(e)}

def main():
    BASE_URL = "http://localhost:5000/api"
    
    # Critical endpoints to test
    endpoints = [
        {"url": f"{BASE_URL}/health", "name": "System Health"},
        {"url": f"{BASE_URL}/kpis", "name": "Executive KPIs"},
        {"url": f"{BASE_URL}/forecast/models", "name": "ML Models Status"},
        {"url": f"{BASE_URL}/forecast/predict?days=7", "name": "Forecasting"},
        {"url": f"{BASE_URL}/capacity-planning?region=East%20US&service=Compute&horizon=30", "name": "Capacity Planning"},
        {"url": f"{BASE_URL}/training/intelligent/status", "name": "Training Pipeline"},
    ]
    
    print("🔍 Azure Forecasting API Health Check")
    print("=" * 60)
    
    all_passed = True
    
    for endpoint in endpoints:
        print(f"Testing {endpoint['name']}...", end=" ")
        result = check_endpoint(endpoint['url'])
        print(f"{result['status']} ({result['response_time']})")
        
        if result['error']:
            print(f"   Error: {result['error']}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ All health checks passed!")
        sys.exit(0)
    else:
        print("❌ Some health checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run health check:
```bash
python health_check.py
```

---

## 📚 **SDK & Integration Examples**

### JavaScript/Node.js SDK

**Installation**:
```bash
npm install azure-forecasting-sdk
```

**Usage**:
```javascript
const AzureForecasting = require('azure-forecasting-sdk');

const client = new AzureForecasting({
  baseUrl: 'http://localhost:5000/api',
  apiKey: 'your-api-key'  // Optional for development
});

async function main() {
  try {
    // Get system health
    const health = await client.health.check();
    console.log('System Status:', health.status);
    
    // Get executive KPIs
    const kpis = await client.analytics.getKPIs();
    console.log('Peak CPU:', kpis.peak_cpu + '%');
    
    // Generate 30-day forecast
    const forecast = await client.forecasting.predictCPU({
      days: 30,
      region: 'East US'
    });
    console.log('Forecast generated:', forecast['East US'].predicted_cpu.length, 'data points');
    
    // Get capacity analysis
    const capacity = await client.capacity.analyze({
      service: 'Compute',
      horizon: 30
    });
    
    const highRiskRegions = Object.entries(capacity.capacity_analysis)
      .filter(([region, analysis]) => analysis.risk_assessment.overall_risk === 'HIGH')
      .map(([region]) => region);
    
    console.log('High-risk regions:', highRiskRegions);
    
  } catch (error) {
    console.error('API Error:', error.message);
  }
}

main();
```

### .NET C# SDK

**Installation**:
```bash
dotnet add package AzureForecasting.SDK
```

**Usage**:
```csharp
using AzureForecasting.SDK;

var client = new AzureForecastingClient(new AzureForecastingOptions 
{
    BaseUrl = "http://localhost:5000/api",
    ApiKey = "your-api-key"
});

try 
{
    // Get system health
    var health = await client.Health.CheckAsync();
    Console.WriteLine($"System Status: {health.Status}");
    
    // Generate forecast
    var forecastRequest = new ForecastRequest 
    {
        Days = 30,
        Region = "East US"
    };
    
    var forecast = await client.Forecasting.PredictCpuAsync(forecastRequest);
    Console.WriteLine($"Forecast generated: {forecast["East US"].PredictedCpu.Count} data points");
    
    // Get capacity analysis
    var capacityRequest = new CapacityRequest
    {
        Service = "Compute",
        Horizon = 30,
        Region = "All Regions"
    };
    
    var capacity = await client.Capacity.AnalyzeAsync(capacityRequest);
    Console.WriteLine($"Overall Status: {capacity.Summary.OverallStatus}");
}
catch (AzureForecastingException ex)
{
    Console.WriteLine($"API Error: {ex.Message} (Code: {ex.ErrorCode})");
}
```

---

## 🔧 **Advanced Usage Patterns**

### Batch Processing

**Process multiple regions efficiently**:
```python
import asyncio
import aiohttp

async def fetch_forecast(session, region, days=30):
    """Fetch forecast for single region asynchronously"""
    url = f"http://localhost:5000/api/forecast/predict"
    params = {"days": days, "region": region}
    
    async with session.get(url, params=params) as response:
        if response.status == 200:
            data = await response.json()
            return region, data[region]
        else:
            return region, None

async def batch_forecasting(regions, days=30):
    """Generate forecasts for multiple regions in parallel"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_forecast(session, region, days) for region in regions]
        results = await asyncio.gather(*tasks)
        return dict(results)

# Usage
regions = ["East US", "West US", "North Europe", "Southeast Asia"]
forecasts = asyncio.run(batch_forecasting(regions, days=30))

for region, forecast_data in forecasts.items():
    if forecast_data:
        print(f"{region}: {len(forecast_data['predicted_cpu'])} predictions generated")
    else:
        print(f"{region}: Failed to generate forecast")
```

### Real-time Monitoring

**Set up real-time system monitoring**:
```python
import requests
import time
import json

class RealTimeMonitor:
    def __init__(self, base_url="http://localhost:5000/api"):
        self.base_url = base_url
        self.last_check = {}
    
    def check_system_health(self):
        """Check overall system health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            health_data = response.json()
            
            if health_data["status"] != "healthy":
                self.alert(f"🚨 System health issue: {health_data['status']}")
            
            return health_data
        except Exception as e:
            self.alert(f"🚨 Health check failed: {e}")
            return None
    
    def check_model_accuracy(self):
        """Monitor model accuracy degradation"""
        try:
            response = requests.get(f"{self.base_url}/monitoring/accuracy", timeout=10)
            accuracy_data = response.json()
            
            if accuracy_data.get("model_health", {}).get("overall_status") == "CRITICAL":
                self.alert("🚨 Critical model accuracy degradation detected!")
                
            # Check individual model accuracy
            for region, metrics in accuracy_data.get("accuracy_metrics", {}).items():
                accuracy = metrics.get("accuracy", 0)
                if accuracy < 75:  # Threshold
                    self.alert(f"🚨 Low accuracy in {region}: {accuracy:.1f}%")
            
            return accuracy_data
        except Exception as e:
            print(f"⚠️ Accuracy check failed: {e}")
            return None
    
    def check_capacity_risks(self):
        """Monitor capacity planning risks"""
        try:
            response = requests.get(f"{self.base_url}/capacity-planning", 
                                  params={"region": "All Regions", "service": "Compute", "horizon": 7},
                                  timeout=30)
            capacity_data = response.json()
            
            # Check for high-risk regions
            for region, analysis in capacity_data.get("capacity_analysis", {}).items():
                risk_level = analysis.get("risk_assessment", {}).get("overall_risk")
                if risk_level == "HIGH":
                    utilization = analysis.get("capacity_utilization", {}).get("peak_pct", 0)
                    self.alert(f"🚨 High capacity risk in {region}: {utilization:.1f}% utilization")
            
            return capacity_data
        except Exception as e:
            print(f"⚠️ Capacity check failed: {e}")
            return None
    
    def alert(self, message):
        """Send alert (customize for your notification system)"""
        print(f"ALERT: {message}")
        # Add integration with Slack, email, PagerDuty, etc.
        
    def run_monitoring_cycle(self):
        """Run complete monitoring cycle"""
        print(f"🔍 Starting monitoring cycle at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System health
        health = self.check_system_health()
        if health:
            print(f"✅ System health: {health['status']}")
        
        # Model accuracy
        accuracy = self.check_model_accuracy()
        if accuracy:
            avg_accuracy = accuracy.get("model_health", {}).get("average_accuracy", 0)
            print(f"✅ Average model accuracy: {avg_accuracy:.1f}%")
        
        # Capacity risks
        capacity = self.check_capacity_risks()
        if capacity:
            risk_dist = capacity.get("summary", {}).get("risk_distribution", {})
            high_risk = risk_dist.get("high_risk", 0)
            print(f"✅ Capacity risks: {high_risk} high-risk regions")
        
        print("🔍 Monitoring cycle completed\n")
    
    def start_monitoring(self, interval_seconds=300):  # 5 minutes default
        """Start continuous monitoring"""
        print(f"🚀 Starting real-time monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                self.run_monitoring_cycle()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

# Usage
monitor = RealTimeMonitor()
monitor.start_monitoring(interval_seconds=300)  # Check every 5 minutes
```

---

## 📞 **Support & Resources**

### Getting Help

- **API Documentation**: This document
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/your-org/azure-forecasting/issues)
- **Stack Overflow**: Tag questions with `azure-forecasting-api`
- **Email Support**: api-support@azureforecasting.com

### Additional Resources

- **OpenAPI Specification**: `GET /api/spec` (Swagger/OpenAPI 3.0)
- **Postman Collection**: [Download comprehensive test suite](postman_collection.json)
- **SDK Documentation**: 
  - [Python SDK](https://github.com/your-org/azure-forecasting-python)
  - [JavaScript SDK](https://github.com/your-org/azure-forecasting-js)  
  - [.NET SDK](https://github.com/your-org/azure-forecasting-dotnet)
- **Example Applications**: [Sample integrations and use cases](https://github.com/your-org/azure-forecasting-examples)

### Changelog & Versioning

The API follows semantic versioning. Current version: **1.0.0**

**Version History**:
- **1.0.0** (2025-10-12): Initial release with 47 endpoints
- **0.9.0** (2025-10-01): Beta release with core forecasting
- **0.8.0** (2025-09-15): Alpha release with capacity planning

**Breaking Changes**: None in current version

**Deprecation Notice**: No deprecated endpoints in current version

---

*This API Reference provides comprehensive documentation for integrating with the Azure Demand Forecasting Platform. For the latest updates and additional examples, visit our [Developer Portal](https://developers.azureforecasting.com).*
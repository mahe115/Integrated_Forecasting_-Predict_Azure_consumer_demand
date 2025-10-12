# API_DOCUMENTATION.md

# 📚 Complete API Documentation

## Azure Demand Forecasting - REST API Reference

*Complete reference guide for all 47 API endpoints in the Azure Demand Forecasting Platform*

---

## 🚀 **API Overview**

The Azure Demand Forecasting Platform provides a comprehensive REST API with 47 optimized endpoints organized into 10 functional categories. All endpoints support JSON responses with standardized error handling and enterprise-grade caching.

**Base URL**: `http://localhost:5000/api`  
**Authentication**: Currently open (token-based auth available for production)  
**Response Format**: JSON  
**Rate Limit**: 1000 requests/minute per IP  

---

## 📊 **1. Data Analytics Endpoints (6 endpoints)**

### `GET /api/kpis`
**Purpose**: Executive KPI metrics with drill-down capabilities  
**Cache**: Medium (5 minutes)  
**Response Time**: ~150ms  

```json
{
  "peak_cpu": 87.5,
  "peak_cpu_details": {
    "date": "2025-10-10",
    "region": "East US", 
    "resource_type": "compute"
  },
  "max_storage": 1250.8,
  "peak_users": 1456,
  "avg_cpu": 72.3,
  "total_regions": 4,
  "holiday_impact": {
    "percentage": 15.2,
    "holiday_avg_cpu": 65.1,
    "regular_avg_cpu": 76.4
  }
}
```

### `GET /api/sparklines`
**Purpose**: 30-day trend sparklines for executive dashboard  
**Cache**: Fast (2 minutes)  

```json
{
  "cpu_trend": [
    {"date": "2025-09-10", "usage_cpu": 72.1},
    {"date": "2025-09-11", "usage_cpu": 74.3}
  ],
  "storage_trend": [...],
  "users_trend": [...]
}
```

### `GET /api/data-raw`
**Purpose**: Raw dataset export (all historical data)  
**Cache**: Slow (15 minutes)  
**Use Case**: Data exports, external integrations  

### `GET /api/time-series`
**Purpose**: Time series data with flexible filtering  
**Parameters**: `region`, `resource_type`, `start_date`, `end_date`  
**Cache**: Fast (2 minutes)  

### `GET /api/data-summary`
**Purpose**: Dataset statistics and metadata  
**Cache**: Slow (15 minutes)  

```json
{
  "usage_cpu": {
    "mean": 72.45,
    "std": 12.33,
    "min": 45.2,
    "max": 98.7
  },
  "dataset_info": {
    "total_records": 1080,
    "date_range_days": 90,
    "regions_count": 4,
    "holiday_records": 120
  }
}
```

### `GET /api/filters/options`
**Purpose**: Available filter options for frontend dropdowns  
**Cache**: Slow (15 minutes)  

---

## 🌍 **2. Regional Analysis Endpoints (4 endpoints)**

### `GET /api/regional/comparison`
**Purpose**: Cross-regional performance comparison  
**Cache**: Medium (5 minutes)  

```json
[
  {
    "region": "East US",
    "usage_cpu_mean": 72.1,
    "usage_cpu_max": 95.3,
    "usage_storage_mean": 1234.5,
    "users_active_mean": 1052
  }
]
```

### `GET /api/regional/heatmap`
**Purpose**: Regional performance heatmap data  
**Use Case**: Geographic performance visualization  

### `GET /api/regional/distribution`
**Purpose**: Resource distribution across regions  
**Use Case**: Regional workload analysis  

### `GET /api/trends/regional`
**Purpose**: Regional trend analysis over time  
**Cache**: Fast (2 minutes)  

---

## ⚙️ **3. Resource Management Endpoints (4 endpoints)**

### `GET /api/resources/utilization`
**Purpose**: Resource utilization trends by type  
**Cache**: Fast (2 minutes)  

### `GET /api/resources/distribution`
**Purpose**: Resource allocation distribution  
**Cache**: Medium (5 minutes)  

### `GET /api/resources/efficiency`
**Purpose**: Resource efficiency scoring  
**Cache**: Medium (5 minutes)  

```json
[
  {
    "resource_type": "compute",
    "usage_cpu": 72.1,
    "users_active": 1052,
    "cpu_per_user": 0.0685,
    "storage_per_user": 1.174
  }
]
```

### `GET /api/trends/resource-types`
**Purpose**: Resource type trend analysis  
**Cache**: Fast (2 minutes)  

---

## 📈 **4. Correlation Analytics Endpoints (4 endpoints)**

### `GET /api/correlations/matrix`
**Purpose**: Correlation matrix for all metrics  
**Cache**: Medium (5 minutes)  

```json
[
  {
    "row": "usage_cpu",
    "column": "users_active", 
    "correlation": 0.847
  }
]
```

### `GET /api/correlations/scatter`
**Purpose**: Scatter plot data for correlation analysis  
**Parameters**: `x_axis`, `y_axis`  
**Cache**: Fast (2 minutes)  

### `GET /api/correlations/bubble`
**Purpose**: Multi-dimensional bubble chart data  
**Cache**: Medium (5 minutes)  

### `GET /api/engagement/bubble`
**Purpose**: User engagement bubble visualization  
**Cache**: Medium (5 minutes)  

---

## 🎄 **5. Holiday & Engagement Endpoints (6 endpoints)**

### `GET /api/holiday/analysis`
**Purpose**: Holiday impact analysis on system usage  
**Cache**: Medium (5 minutes)  

```json
[
  {
    "holiday": 1,
    "usage_cpu_mean": 65.2,
    "usage_cpu_std": 8.9,
    "users_active_mean": 892
  }
]
```

### `GET /api/holiday/distribution`
**Purpose**: Holiday vs regular day usage distribution  
**Cache**: Medium (5 minutes)  

### `GET /api/holiday/calendar`
**Purpose**: Calendar view of holiday impact  
**Cache**: Medium (5 minutes)  

### `GET /api/engagement/efficiency`
**Purpose**: User engagement efficiency metrics  
**Cache**: Medium (5 minutes)  

### `GET /api/engagement/trends`
**Purpose**: User engagement trends over time  
**Cache**: Fast (2 minutes)  

---

## 🤖 **6. ML Forecasting Endpoints (10 endpoints)**

### `GET /api/forecast/models`
**Purpose**: CPU forecasting model status  
**Cache**: Fast (2 minutes)  

```json
{
  "models": {
    "East US": {
      "model_type": "ARIMA",
      "loaded": true,
      "has_scaler": false,
      "last_updated": "2025-10-10",
      "performance": {
        "rmse": 8.45,
        "mae": 6.23
      }
    }
  },
  "total_regions": 4,
  "ml_available": true
}
```

### `GET /api/forecast/predict`
**Purpose**: Generate CPU usage forecasts  
**Parameters**: `days` (1-90), `region`  
**Cache**: Smart caching (1,7,30 day forecasts cached)  

```json
{
  "East US": {
    "dates": ["2025-10-11", "2025-10-12"],
    "predicted_cpu": [74.2, 75.8],
    "model_info": {
      "type": "ARIMA",
      "forecast_horizon": 30
    }
  }
}
```

### `GET /api/forecast/users/models`
**Purpose**: Users forecasting model status  

### `GET /api/forecast/users/predict`
**Purpose**: Generate active users forecasts  
**Parameters**: `days`, `region`  

### `GET /api/forecast/storage/models` *(New)*
**Purpose**: Storage forecasting model status  

### `GET /api/forecast/storage/predict` *(New)*
**Purpose**: Generate storage usage forecasts  

### `GET /api/forecast/comparison`
**Purpose**: Model performance comparison  
**Cache**: Slow (15 minutes)  

### `GET /api/forecast/debug`
**Purpose**: CPU model debugging information  
**Use Case**: Development and troubleshooting  

### `GET /api/forecast/users/debug`
**Purpose**: Users model debugging information  

### `GET /api/forecast/storage/debug` *(New)*
**Purpose**: Storage model debugging information  

---

## 🎯 **7. Intelligent Training Endpoints (5 endpoints)**

### `GET /api/training/intelligent/status`
**Purpose**: Intelligent training pipeline status  
**Cache**: Medium (5 minutes)  

```json
{
  "pipeline_active": true,
  "pipeline_type": "intelligent_auto_selection",
  "current_models": {
    "cpu": {"East US": "ARIMA", "West US": "LSTM"},
    "users": {"East US": "XGBoost"}
  },
  "last_check": "2025-10-10T14:30:00"
}
```

### `POST /api/training/intelligent/trigger`
**Purpose**: Manually trigger intelligent training  
**Response**: Training job started in background  

### `GET /api/training/intelligent/history`
**Purpose**: Training history and performance  
**Cache**: Medium (5 minutes)  

### `GET /api/training/intelligent/config`
**Purpose**: Training pipeline configuration  

### `GET /api/training/intelligent/model-comparison`
**Purpose**: Detailed model performance comparison  
**Cache**: Medium (5 minutes)  

---

## 📊 **8. System Monitoring Endpoints (4 endpoints)**

### `GET /api/health`
**Purpose**: System health check  
**Cache**: None (real-time)  

```json
{
  "status": "healthy",
  "timestamp": "2025-10-10T14:45:00",
  "uptime_seconds": 86400,
  "ml_models_loaded": 12,
  "database_connected": true,
  "cache_hit_rate": 0.95
}
```

### `GET /api/monitoring/accuracy`
**Purpose**: Model accuracy monitoring  
**Cache**: Fast (2 minutes)  

```json
{
  "model_health": {
    "overall_status": "HEALTHY",
    "average_accuracy": 84.7,
    "healthy_models": 10,
    "warning_models": 2
  },
  "accuracy_metrics": {
    "East US": {
      "accuracy": 87.3,
      "trend": "stable",
      "mae": 6.45,
      "rmse": 8.92
    }
  }
}
```

### `GET /api/cache/stats`
**Purpose**: Cache performance statistics  
**Cache**: None (real-time)  

### `GET /api/system/performance`
**Purpose**: System performance metrics  

---

## 📋 **9. Reporting & Export Endpoints (4 endpoints)**

### `GET /api/reports/generate`
**Purpose**: Generate comprehensive reports  
**Parameters**: `type` (csv|excel|pdf), `forecast_horizon`  

**CSV Export**:
```
Region,Date,CPU_Forecast,Users_Forecast,Capacity_Status,Risk_Level
East US,2025-10-11,75.2,1250,Normal,Low
West US,2025-10-11,73.8,1180,Normal,Low
```

**Excel Export**: Multi-sheet workbook with charts  
**PDF Export**: Executive summary with visualizations  

### `POST /api/reports/schedule`
**Purpose**: Schedule automated report generation  

### `GET /api/reports/history`
**Purpose**: Generated reports history  

### `DELETE /api/reports/{report_id}`
**Purpose**: Delete generated reports  

---

## 🏗️ **10. Capacity Planning Endpoints (3 endpoints)**

### `GET /api/capacity-planning`
**Purpose**: Comprehensive capacity analysis  
**Parameters**: `region`, `service`, `horizon`  
**Cache**: Smart caching for common scenarios  

```json
{
  "timestamp": "2025-10-10T14:45:00",
  "service": "Compute",
  "horizon_days": 30,
  "capacity_analysis": {
    "East US": {
      "current_capacity": 1502.0,
      "predicted_demand": {
        "max": 76.8,
        "avg": 72.4,
        "timeline": [74.2, 75.1, 73.9]
      },
      "capacity_utilization": {
        "current_pct": 4.8,
        "peak_pct": 5.1
      },
      "risk_assessment": {
        "overall_risk": "LOW",
        "utilization_risk": {
          "level": "LOW",
          "message": "Healthy utilization levels"
        }
      },
      "recommendations": [
        {
          "type": "MAINTAIN",
          "priority": "LOW",
          "action": "Continue current capacity levels",
          "timeline": "Ongoing"
        }
      ]
    }
  }
}
```

### `GET /api/capacity/recommendations`
**Purpose**: Specific capacity scaling recommendations  

### `GET /api/capacity/risk-matrix`
**Purpose**: Risk assessment matrix for all regions  

---

## 🛠️ **Error Handling**

All endpoints follow standardized error responses:

**4xx Client Errors**:
```json
{
  "error": "Invalid parameter: days must be between 1 and 90",
  "code": "INVALID_PARAMETER",
  "timestamp": "2025-10-10T14:45:00"
}
```

**5xx Server Errors**:
```json
{
  "error": "Model not available",
  "code": "MODEL_UNAVAILABLE",
  "timestamp": "2025-10-10T14:45:00"
}
```

**Common HTTP Status Codes**:
- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Endpoint not found  
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: ML models not loaded

---

## ⚡ **Performance Specifications**

| Metric | Target | Actual |
|--------|--------|---------|
| Average Response Time | <300ms | 200ms |
| P95 Response Time | <500ms | 450ms |
| Cache Hit Rate | >90% | 95% |
| Concurrent Users | 1000+ | Tested to 1200 |
| Forecast Generation | <10s | 5s (30-day) |
| Memory Usage | <2GB | 1.2GB typical |

---

## 🔧 **Integration Examples**

### Python Integration
```python
import requests

# Initialize API client
BASE_URL = "http://localhost:5000/api"

# Get system health
health = requests.get(f"{BASE_URL}/health")
print(f"System Status: {health.json()['status']}")

# Generate 30-day forecast
forecast = requests.get(f"{BASE_URL}/forecast/predict", 
                       params={"days": 30, "region": "East US"})
predictions = forecast.json()["East US"]["predicted_cpu"]
print(f"30-day forecast: {predictions[:5]}...")  # First 5 days

# Get capacity planning recommendations
capacity = requests.get(f"{BASE_URL}/capacity-planning",
                       params={"region": "All Regions", 
                              "service": "Compute", 
                              "horizon": 30})
risk_regions = [region for region, data in capacity.json()["capacity_analysis"].items() 
                if data.get("risk_assessment", {}).get("overall_risk") == "HIGH"]
print(f"High-risk regions: {risk_regions}")
```

### JavaScript/Node.js Integration
```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:5000/api';

// Get executive KPIs
async function getExecutiveDashboard() {
  const kpis = await axios.get(`${API_BASE}/kpis`);
  const sparklines = await axios.get(`${API_BASE}/sparklines`);
  
  return {
    metrics: kpis.data,
    trends: sparklines.data
  };
}

// Generate and export report
async function generateReport(type = 'csv') {
  const report = await axios.get(`${API_BASE}/reports/generate`, {
    params: { type, forecast_horizon: 30 }
  });
  
  return report.data;
}
```

### cURL Examples
```bash
# System health check
curl -X GET "http://localhost:5000/api/health"

# Generate 7-day CPU forecast
curl -X GET "http://localhost:5000/api/forecast/predict?days=7&region=East%20US"

# Trigger intelligent training
curl -X POST "http://localhost:5000/api/training/intelligent/trigger"

# Get capacity planning analysis
curl -X GET "http://localhost:5000/api/capacity-planning?region=All%20Regions&service=Compute&horizon=30"

# Export CSV report
curl -X GET "http://localhost:5000/api/reports/generate?type=csv" -o forecast_report.csv
```

---

## 📚 **Additional Resources**

- **[Postman Collection](postman_collection.json)**: Pre-configured API tests
- **[OpenAPI/Swagger Spec](swagger.yaml)**: Machine-readable API specification
- **[SDK Documentation](sdk/)**: Language-specific SDKs
- **[Rate Limiting Guide](rate_limits.md)**: API usage best practices
- **[Authentication Guide](auth.md)**: Token-based authentication setup

---

## 🚀 **Getting Started**

1. **Start the API server**:
   ```bash
   python optimised_backend_app.py
   ```

2. **Test basic connectivity**:
   ```bash
   curl http://localhost:5000/api/health
   ```

3. **Import Postman collection** for full API testing

4. **Reference this documentation** for endpoint details and integration patterns

---

*This API documentation covers all 47 endpoints in the Azure Demand Forecasting Platform. For updates and additional examples, see the [GitHub repository](https://github.com/your-repo).*
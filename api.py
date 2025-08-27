from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load datasets at startup
df = pd.read_csv('data/processed/cleaned_merged.csv', parse_dates=['date'])
df_features = pd.read_csv('data/processed/final_featured_dataset.csv', parse_dates=['date'])

@app.route('/api/raw-data')
def get_raw_data():
    limit = request.args.get('limit', 100, type=int)
    region_filter = request.args.get('region')
    resource_filter = request.args.get('resource_type')
    data = df.copy()
    if region_filter:
        data = data[data['region'] == region_filter]
    if resource_filter:
        data = data[data['resource_type'] == resource_filter]
    return jsonify(data.head(limit).to_dict(orient='records'))

@app.route('/api/features')
def get_features():
    limit = request.args.get('limit', 100, type=int)
    region_filter = request.args.get('region')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    features = request.args.get('features')
    data = df.copy()
    if region_filter:
        data = data[data['region'] == region_filter]
    if start_date:
        data = data[data['date'] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data['date'] <= pd.to_datetime(end_date)]
    if features:
        cols = ['date', 'region'] + features.split(',')
        data = data[cols]
    return jsonify(data.head(limit).to_dict(orient='records'))

@app.route('/api/insights')
def get_insights():
    top_regions = df.groupby('region')['usage_cpu'].sum().nlargest(5).to_dict()
    peak_usage = float(df['usage_cpu'].max())
    df_features['month'] = df['date'].dt.to_period('M')
    return jsonify({
        'top_regions': top_regions,
        'peak_usage_overall': peak_usage,
    })

@app.route('/api/usage-trends')
def usage_trends():
    data = df.groupby(['date', 'region'])['usage_cpu'].mean().reset_index()
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/top-regions')
def top_regions():
    metric = request.args.get('metric', 'usage_cpu')
    top_n = request.args.get('top_n', 10, type=int)
    data = df.groupby('region')[metric].sum().nlargest(top_n).to_dict()
    return jsonify(data)

@app.route('/api/region-timeseries/<region_name>')
def region_timeseries(region_name):
    data = df[df['region'].str.lower() == region_name.lower()]
    ts = data.groupby('date')['usage_cpu'].mean().reset_index()
    return jsonify(ts.to_dict(orient='records'))

@app.route('/api/custom-features')
def custom_features():
    cols = ['date', 'region', 'usage_cpu', 'usage_cpu_lag1', 'usage_cpu_roll7']
    data = df_features[cols]
    return jsonify(data.head(100).to_dict(orient='records'))

@app.route('/api/seasonality-insights')
def seasonality_insights():
    df_features['month'] = df_features['date'].dt.month
    month_dist = df_features.groupby('month')['usage_cpu'].mean().round(2).to_dict()
    return jsonify({'avg_usage_per_month': month_dist})

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)

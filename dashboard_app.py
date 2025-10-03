import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Azure Demand Forecasting Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
BASE_URL = "http://localhost:5000/api"

# Custom CSS for Azure theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0078d4 0%, #106ebe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #0078d4;
        margin-bottom: 1rem;
    }
    
    .tab-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .kpi-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .success-alert {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin-bottom: 1rem;
    }
    
    .warning-alert {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin-bottom: 1rem;Multi-dimensional Analysis - External Factors Impact


    }
    
    .plotly-chart {
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_api(endpoint, params=None):
    """Fetch data from API with error handling"""
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create custom metric card"""
    delta_html = ""
    if delta:
        color = "#28a745" if delta_color == "normal" else "#dc3545"
        delta_html = f'<small style="color: {color};">{delta}</small>'
    
    return f"""
    <div class="metric-card">
        <h4 style="margin: 0; color: #0078d4;">{title}</h4>
        <h2 style="margin: 0.5rem 0; color: #333;">{value}</h2>
        {delta_html}
    </div>
    """

# Main Header
st.markdown("""
<div class="main-header">
    <h1>☁️ Azure Demand Forecasting Dashboard</h1>
    <p>Real-time analytics and insights for Azure resource demand patterns</p>
</div>
""", unsafe_allow_html=True)

# Load filter options
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_filter_options():
    return fetch_api("filters/options")

filter_options = load_filter_options()

# Sidebar Filters
with st.sidebar:
    st.header("🎛️ Global Filters")
    
    if filter_options:
        # Region filter
        regions = filter_options.get('regions', [])
        selected_regions = st.multiselect(
            "🌍 Select Regions",
            options=regions,
            default=regions,
            help="Choose one or more regions to analyze"
        )
        
        # Resource type filter
        resources = filter_options.get('resource_types', [])
        selected_resources = st.multiselect(
            "⚙️ Select Resource Types",
            options=resources,
            default=resources,
            help="Choose resource types to include in analysis"
        )
        
        # Date range filter
        date_range = filter_options.get('date_range', {})
        if date_range:
            start_date = st.date_input(
                "📅 Start Date",
                value=pd.to_datetime(date_range['min_date']).date(),
                min_value=pd.to_datetime(date_range['min_date']).date(),
                max_value=pd.to_datetime(date_range['max_date']).date()
            )
            
            end_date = st.date_input(
                "📅 End Date", 
                value=pd.to_datetime(date_range['max_date']).date(),
                min_value=pd.to_datetime(date_range['min_date']).date(),
                max_value=pd.to_datetime(date_range['max_date']).date()
            )
    else:
        st.error("Unable to load filter options")
        selected_regions = []
        selected_resources = []
        start_date = None
        end_date = None

# Tab Navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "📈 Trends", 
    "🌍 Regional",
    "⚙️ Resources",
    "🔗 Correlations",
    "🎉 User Engagement",
    "🤖 Forecasting",
])

# ===== TAB 1: OVERVIEW & KPIs =====
with tab1:
    st.subheader("📊 Key Performance Indicators")
    
    # Load KPI data
    kpi_data = fetch_api("kpis")
    
    if kpi_data:
        # Top row - Main KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔥 Peak CPU Usage",
                value=f"{kpi_data['peak_cpu']:.1f}%",
                delta=f"+{kpi_data['peak_cpu'] - kpi_data['avg_cpu']:.1f}% above avg"
            )
            with st.expander("Details"):
                details = kpi_data['peak_cpu_details']
                st.write(f"**Date:** {details['date']}")
                st.write(f"**Region:** {details['region']}")
                st.write(f"**Resource:** {details['resource_type']}")
        
        with col2:
            st.metric(
                label="💾 Max Storage",
                value=f"{kpi_data['max_storage']:,.0f} GB",
                delta=f"+{kpi_data['max_storage'] - kpi_data['avg_storage']:.0f}GB above avg"
            )
            with st.expander("Details"):
                details = kpi_data['max_storage_details']
                st.write(f"**Date:** {details['date']}")
                st.write(f"**Region:** {details['region']}")
                st.write(f"**Resource:** {details['resource_type']}")
        
        with col3:
            st.metric(
                label="👥 Peak Users",
                value=f"{kpi_data['peak_users']:,}",
                delta=f"+{kpi_data['peak_users'] - kpi_data['avg_users']:.0f} above avg"
            )
            with st.expander("Details"):
                details = kpi_data['peak_users_details']
                st.write(f"**Date:** {details['date']}")
                st.write(f"**Region:** {details['region']}")
                st.write(f"**Resource:** {details['resource_type']}")
        
        with col4:
            holiday_impact = kpi_data['holiday_impact']['percentage']
            st.metric(
                label="🎉 Holiday Impact",
                value=f"{holiday_impact:+.1f}%",
                delta="CPU usage change on holidays",
                delta_color="inverse" if holiday_impact < 0 else "normal"
            )
        
        # Second row - System Overview
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("🌍 Total Regions", kpi_data['total_regions'])
        
        with col6:
            st.metric("⚙️ Resource Types", kpi_data['total_resource_types'])
        
        with col7:
            st.metric("📅 Data Points", f"{kpi_data['data_points']:,}")
        
        with col8:
            st.metric("⏱️ Time Span", f"{kpi_data['date_range']['days']} days")
        
        st.divider()
        
        # Sparklines section
        st.subheader("📈 Recent Trends (Last 30 Days)")
        sparklines = fetch_api("sparklines")
        
        if sparklines:
            spark_col1, spark_col2, spark_col3 = st.columns(3)
            
            with spark_col1:
                cpu_data = pd.DataFrame(sparklines['cpu_trend'])
                if not cpu_data.empty:
                    fig = px.line(cpu_data, x='date', y='usage_cpu',
                                title="CPU Usage Trend")
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with spark_col2:
                storage_data = pd.DataFrame(sparklines['storage_trend'])
                if not storage_data.empty:
                    fig = px.line(storage_data, x='date', y='usage_storage',
                                title="Storage Usage Trend", color_discrete_sequence=['#ff6b6b'])
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with spark_col3:
                users_data = pd.DataFrame(sparklines['users_trend'])
                if not users_data.empty:
                    fig = px.line(users_data, x='date', y='users_active',
                                title="User Activity Trend", color_discrete_sequence=['#4ecdc4'])
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)
        st.divider()
        # ===== Data Explorer =====
        st.subheader("🗃️ Data Explorer")

        # Fetch raw data
        raw_data = fetch_api("data/raw")
        if not raw_data:
           st.info("No data available for exploration.")
        else:
           df_explore = pd.DataFrame(raw_data)
           df_explore['date'] = pd.to_datetime(df_explore['date']).dt.date

           # --- Inline filter controls ---
           st.markdown("**Filters:**")
           fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1, 1])
           # Separate start and end date pickers
           default_start = df_explore['date'].min()
           default_end   = df_explore['date'].max()
           with fcol1:
                start = st.date_input("Start Date", default_start)
   
           with fcol2:
                end = st.date_input("End Date", default_end)

            # Region dropdown
           regions = ['All'] + sorted(df_explore['region'].unique().tolist())
           with fcol3:
                sel_region = st.selectbox("Region", regions)

            # Resource type dropdown
           resources = ['All'] + sorted(df_explore['resource_type'].unique().tolist())
           with fcol4:
                sel_resource = st.selectbox("Resource Type", resources)

           # --- Apply filters ---
           mask = df_explore['date'].between(start, end)
           if sel_region != 'All':
              mask &= df_explore['region'] == sel_region
           if sel_resource != 'All':
              mask &= df_explore['resource_type'] == sel_resource
           df_filtered = df_explore.loc[mask].copy()



           # --- COLUMN ORDERING & LABELING ---
           # Define meaningful column order and labels
           column_config = {
                             'date': 'Date',
        'region': 'Azure Region',
        'resource_type': 'Resource Type',
        'usage_cpu': 'CPU Usage (%)',
        'usage_storage': 'Storage (GB)',
        'users_active': 'Active Users',
        'economic_index': 'Economic Index',
        'cloud_market_demand': 'Market Demand',
        'holiday': 'Holiday'
    }
    
           # Reorder columns in meaningful sequence
           ordered_columns = ['date', 'region', 'resource_type', 'usage_cpu', 'usage_storage', 
                      'users_active', 'economic_index', 'cloud_market_demand', 'holiday']
    
           # Select and reorder columns
           df_display = df_filtered[ordered_columns].copy()
    
           # Format specific columns for better readability
           df_display['usage_cpu'] = df_display['usage_cpu'].round(1)
           df_display['usage_storage'] = df_display['usage_storage'].astype(int)
           df_display['economic_index'] = df_display['economic_index'].round(2)
           df_display['cloud_market_demand'] = df_display['cloud_market_demand'].round(3)
           df_display['holiday'] = df_display['holiday'].map({0: 'No', 1: 'Yes'})
    
           # Display count and table with custom column configuration
           st.markdown(f"**Showing {len(df_display):,} records**")
    
           st.dataframe(
                 df_display.sort_values('date', ascending=False),
                 use_container_width=True,
                 height=400,
                 column_config={
            'date': st.column_config.DateColumn('📅 Date'),
            'region': st.column_config.TextColumn('🌍 Azure Region'),
            'resource_type': st.column_config.TextColumn('⚙️ Resource Type'),
            'usage_cpu': st.column_config.NumberColumn('🔥 CPU Usage (%)', format="%.1f%%"),
            'usage_storage': st.column_config.NumberColumn('💾 Storage (GB)', format="%d GB"),
            'users_active': st.column_config.NumberColumn('👥 Active Users', format="%d"),
            'economic_index': st.column_config.NumberColumn('📈 Economic Index', format="%.2f"),
            'cloud_market_demand': st.column_config.NumberColumn('📊 Market Demand', format="%.3f"),
            'holiday': st.column_config.TextColumn('🎉 Holiday')
        }
    )
    
          # Additional insights section
           with st.expander("📊 Quick Insights from Filtered Data"):
              insights_col1, insights_col2, insights_col3 = st.columns(3)
        
           with insights_col1:
                st.metric("Avg CPU Usage", f"{df_display['usage_cpu'].mean():.1f}%")
                st.metric("Peak CPU Usage", f"{df_display['usage_cpu'].max():.1f}%")
        
           with insights_col2:
                st.metric("Avg Storage", f"{df_display['usage_storage'].mean():.0f} GB")
                st.metric("Total Users", f"{df_display['users_active'].sum():,}")
        
           with insights_col3:
               holiday_pct = (df_display['holiday'] == 'Yes').mean() * 100
               st.metric("Holiday Records", f"{holiday_pct:.1f}%")
               st.metric("Unique Regions", f"{df_display['region'].nunique()}")







# SPACE-OPTIMIZED TAB 2 - COMPACT LAYOUT

# ===== TAB 2: ENHANCED TRENDS ANALYSIS WITH COMPACT LAYOUT =====
with tab2:
    st.subheader("📈 Advanced Trends Analysis & Pattern Detection")

    # Load filter options and check filtering capability
    filter_options = fetch_api("filters/options")

    if not filter_options:
        st.error("❌ Unable to load filter options")
        st.stop()

    # === ENHANCED FILTER CONTROLS ===
    st.markdown("**🎛️ Advanced Trend Controls:**")

    # Row 1: Primary filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        metric_choice = st.selectbox(
            "📊 Primary Metric", 
            ["usage_cpu", "usage_storage", "users_active"], 
            format_func=lambda x: {
                "usage_cpu": "🔥 CPU Usage (%)",
                "usage_storage": "💾 Storage Usage (GB)", 
                "users_active": "👥 Active Users"
            }[x]
        )

    with filter_col2:
        # Get regions from filter options API
        available_regions = ['All Regions'] + filter_options.get('regions', [])
        region_filter = st.selectbox("🌍 Region Focus", available_regions)

    with filter_col3:
        # Get resource types from filter options API
        available_resources = ['All Resources'] + filter_options.get('resource_types', [])
        resource_filter = st.selectbox("⚙️ Resource Type", available_resources)

    with filter_col4:
        trend_period = st.selectbox("⏰ Time Period", ["7D", "30D", "90D", "All"])

    # Row 2: Analysis options
    analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)

    with analysis_col1:
        smoothing = st.checkbox("📈 Apply Smoothing", value=True)

    with analysis_col2:
        show_patterns = st.checkbox("🔍 Highlight Patterns", value=True)

    with analysis_col3:
        compare_mode = st.checkbox("⚖️ Comparison View", value=False)

    with analysis_col4:
        show_anomalies = st.checkbox("🚨 Detect Anomalies", value=False)

    # === DATA LOADING (COMPACT) ===
    # Try to use raw data for proper filtering
    raw_data = fetch_api("data/raw")

    if raw_data:
        # CLIENT-SIDE FILTERING (Most reliable approach)
        df_raw = pd.DataFrame(raw_data)
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # Apply region filter
        if region_filter != 'All Regions':
            df_filtered_raw = df_raw[df_raw['region'] == region_filter]
        else:
            df_filtered_raw = df_raw

        # Apply resource type filter
        if resource_filter != 'All Resources':
            df_filtered_raw = df_filtered_raw[df_filtered_raw['resource_type'] == resource_filter]

        # Apply time period filter
        if trend_period != "All":
            days = int(trend_period[:-1])
            latest_date = df_filtered_raw['date'].max()
            cutoff = latest_date - timedelta(days=days)
            df_filtered_raw = df_filtered_raw[df_filtered_raw['date'] >= cutoff]

        # Aggregate filtered data by date
        df_agg = df_filtered_raw.groupby('date').agg({
            'usage_cpu': 'mean',
            'usage_storage': 'mean', 
            'users_active': 'mean',
            'economic_index': 'mean',
            'cloud_market_demand': 'mean'
        }).reset_index().sort_values('date')

    else:
        # FALLBACK: Try time-series API with query parameters
        params = {}
        if region_filter != 'All Regions':
            params['region'] = region_filter
        if resource_filter != 'All Resources':
            params['resource_type'] = resource_filter

        if params:
            query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
            filtered_time_series = fetch_api(f"time-series?{query_params}")

            if filtered_time_series:
                df_agg = pd.DataFrame(filtered_time_series)
                df_agg['date'] = pd.to_datetime(df_agg['date'])
            else:
                time_series_data = fetch_api("time-series")
                df_agg = pd.DataFrame(time_series_data)
                df_agg['date'] = pd.to_datetime(df_agg['date'])
        else:
            # No filters, use regular time-series
            time_series_data = fetch_api("time-series")
            df_agg = pd.DataFrame(time_series_data)
            df_agg['date'] = pd.to_datetime(df_agg['date'])

        # Apply time period filter
        if trend_period != "All":
            days = int(trend_period[:-1])
            latest_date = df_agg['date'].max()
            cutoff = latest_date - timedelta(days=days)
            df_agg = df_agg[df_agg['date'] >= cutoff]

    if df_agg.empty:
        st.error("❌ No data available for selected filters")
        st.error(f"Filters: Region={region_filter}, Resource={resource_filter}, Period={trend_period}")
        st.stop()

    # === COMPACT DATA STATUS (HIDDEN IN EXPANDER) ===
    with st.expander("📊 Data Status & Debug Information", expanded=False):
        # Data loading status
        status_col1, status_col2, status_col3 = st.columns(3)

        if raw_data:
            status_col1.success("🔧 Using raw data for accurate filtering")
        else:
            status_col1.warning("⚠️ Using time-series API")

        # Filter status
        if region_filter != 'All Regions':
            status_col2.info(f"🌍 {region_filter}: {len(df_agg):,} records")
        else:
            status_col2.info("📊 All regions")

        if resource_filter != 'All Resources':
            status_col3.info(f"⚙️ {resource_filter}")
        else:
            status_col3.info("📊 All resources")

        st.divider()

        # Debug information (moved from main area)
        debug_col1, debug_col2 = st.columns(2)

        with debug_col1:
            st.markdown("**Filter Selection:**")
            st.write(f"📍 Region: {region_filter}")
            st.write(f"⚙️ Resource: {resource_filter}")
            st.write(f"⏰ Period: {trend_period}")
            st.write(f"📊 Metric: {metric_choice}")

        with debug_col2:
            st.markdown("**Data Summary:**")
            st.write(f"📈 Records: {len(df_agg):,}")
            st.write(f"📅 Date Range: {df_agg['date'].min().date()} to {df_agg['date'].max().date()}")
            st.write(f"🔢 {metric_choice} Range: {df_agg[metric_choice].min():.1f} - {df_agg[metric_choice].max():.1f}")

            # Show actual data preview
            st.markdown("**Sample Data:**")
            st.dataframe(df_agg.head(3), width="stretch")

        # Summary metrics (moved from main area) 
        st.markdown("**📊 Quick Summary Metrics:**")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.metric("📊 Data Points", f"{len(df_agg):,}")

        with summary_col2:
            date_range = (df_agg['date'].max() - df_agg['date'].min()).days
            st.metric("📅 Date Range", f"{date_range} days")

        with summary_col3:
            current_value = df_agg[metric_choice].iloc[-1] if len(df_agg) > 0 else 0
            st.metric("📈 Current Value", f"{current_value:.1f}")

        with summary_col4:
            if len(df_agg) > 1:
                trend_change = ((df_agg[metric_choice].iloc[-1] - df_agg[metric_choice].iloc[0]) / df_agg[metric_choice].iloc[0]) * 100
                st.metric("🔄 Period Change", f"{trend_change:+.1f}%")
            else:
                st.metric("🔄 Period Change", "N/A")

    # === 1. PRIMARY TREND ANALYSIS (NOW MUCH CLOSER TO CONTROLS) ===
    st.markdown("### 📈 Primary Trend Analysis")

    # Create primary trend chart
    fig_primary = go.Figure()

    # Base trend line
    base_color = '#0078d4'
    if smoothing and len(df_agg) >= 7:
        # Apply smoothing
        df_agg[f'{metric_choice}_smooth'] = df_agg[metric_choice].rolling(window=min(7, len(df_agg)), center=True).mean()

        # Original data (lighter)
        fig_primary.add_trace(go.Scatter(
            x=df_agg['date'],
            y=df_agg[metric_choice],
            mode='lines+markers',
            name=f'Daily {metric_choice.replace("_", " ").title()}',
            line=dict(color='lightgray', width=1),
            marker=dict(size=3, color='lightgray'),
            opacity=0.5
        ))

        # Smoothed trend (bold)
        fig_primary.add_trace(go.Scatter(
            x=df_agg['date'],
            y=df_agg[f'{metric_choice}_smooth'],
            mode='lines',
            name=f'Smoothed Trend',
            line=dict(color=base_color, width=3)
        ))

        trend_values = df_agg[f'{metric_choice}_smooth'].dropna()
    else:
        fig_primary.add_trace(go.Scatter(
            x=df_agg['date'],
            y=df_agg[metric_choice],
            mode='lines+markers',
            name=f'{metric_choice.replace("_", " ").title()}',
            line=dict(color=base_color, width=2),
            marker=dict(size=4)
        ))
        trend_values = df_agg[metric_choice]

    # Anomaly detection
    if show_anomalies and len(trend_values) > 10:
        # Simple anomaly detection using IQR
        Q1 = trend_values.quantile(0.25)
        Q3 = trend_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomalies = df_agg[(df_agg[metric_choice] < lower_bound) | (df_agg[metric_choice] > upper_bound)]

        if not anomalies.empty:
            fig_primary.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies[metric_choice],
                mode='markers',
                name='🚨 Anomalies',
                marker=dict(size=8, color='red', symbol='diamond'),
                hovertemplate='<b>Anomaly Detected</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))

    # Pattern highlighting
    if show_patterns and len(trend_values) > 5:
        try:
            from scipy.signal import find_peaks

            values = trend_values.values
            peaks, _ = find_peaks(values, height=np.percentile(values, 75))
            valleys, _ = find_peaks(-values, height=-np.percentile(values, 25))

            if len(peaks) > 0:
                peak_dates = df_agg.iloc[peaks]['date']
                peak_values = df_agg.iloc[peaks][metric_choice]

                fig_primary.add_trace(go.Scatter(
                    x=peak_dates,
                    y=peak_values,
                    mode='markers',
                    name='⛰️ Peaks',
                    marker=dict(size=10, color='green', symbol='triangle-up'),
                    hovertemplate='<b>Peak</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ))

            if len(valleys) > 0:
                valley_dates = df_agg.iloc[valleys]['date']
                valley_values = df_agg.iloc[valleys][metric_choice]

                fig_primary.add_trace(go.Scatter(
                    x=valley_dates,
                    y=valley_values,
                    mode='markers',
                    name='🏔️ Valleys',
                    marker=dict(size=10, color='orange', symbol='triangle-down'),
                    hovertemplate='<b>Valley</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ))
        except ImportError:
            pass  # Skip pattern detection if scipy not available

    # Add trend annotation
    if len(trend_values) > 1:
        trend_change = ((trend_values.iloc[-1] - trend_values.iloc[0]) / trend_values.iloc[0]) * 100
        trend_color = 'green' if trend_change > 0 else 'red'
        trend_arrow = '📈' if trend_change > 0 else '📉'

        fig_primary.add_annotation(
            x=df_agg['date'].iloc[-1],
            y=trend_values.iloc[-1],
            text=f"{trend_arrow} {trend_change:+.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=trend_color,
            bgcolor=trend_color,
            font=dict(color='white', size=12),
            bordercolor=trend_color,
            borderwidth=2
        )

    filter_text = ""
    if region_filter != 'All Regions':
        filter_text += f" | {region_filter}"
    if resource_filter != 'All Resources':
        filter_text += f" | {resource_filter}"

    fig_primary.update_layout(
        title=f"{metric_choice.replace('_', ' ').title()} Trends{filter_text}",
        xaxis_title="Date",
        yaxis_title=metric_choice.replace('_', ' ').title(),
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_primary, width="stretch")

    # === 2. COMPARISON VIEW ===
    if compare_mode:
        st.markdown("### ⚖️ Regional & Resource Comparison")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            # Regional comparison using raw data
            if raw_data:
                df_raw_regional = pd.DataFrame(raw_data)
                df_raw_regional['date'] = pd.to_datetime(df_raw_regional['date'])

                # Filter by time period
                if trend_period != "All":
                    days = int(trend_period[:-1])
                    latest_date = df_raw_regional['date'].max()
                    cutoff = latest_date - timedelta(days=days)
                    df_raw_regional = df_raw_regional[df_raw_regional['date'] >= cutoff]

                # Apply resource filter if selected
                if resource_filter != 'All Resources':
                    df_raw_regional = df_raw_regional[df_raw_regional['resource_type'] == resource_filter]

                # Aggregate by date and region
                regional_comparison = df_raw_regional.groupby(['date', 'region']).agg({
                    metric_choice: 'mean'
                }).reset_index()

                fig_regional = px.line(
                    regional_comparison,
                    x='date',
                    y=metric_choice,
                    color='region',
                    title=f"Regional Comparison - {metric_choice.replace('_', ' ').title()}",
                    color_discrete_map={
                        'East US': '#0078d4',
                        'West US': '#ff6b6b', 
                        'North Europe': '#4ecdc4',
                        'Southeast Asia': '#95e1d3'
                    }
                )
                fig_regional.update_layout(height=400)
                st.plotly_chart(fig_regional, width="stretch")
            else:
                st.info("📊 Raw data not available for regional comparison")

        with comp_col2:
            # Resource comparison using raw data
            if raw_data:
                df_raw_resource = pd.DataFrame(raw_data)
                df_raw_resource['date'] = pd.to_datetime(df_raw_resource['date'])

                # Filter by time period
                if trend_period != "All":
                    days = int(trend_period[:-1])
                    latest_date = df_raw_resource['date'].max()
                    cutoff = latest_date - timedelta(days=days)
                    df_raw_resource = df_raw_resource[df_raw_resource['date'] >= cutoff]

                # Apply region filter if selected
                if region_filter != 'All Regions':
                    df_raw_resource = df_raw_resource[df_raw_resource['region'] == region_filter]

                # Aggregate by date and resource type
                resource_comparison = df_raw_resource.groupby(['date', 'resource_type']).agg({
                    metric_choice: 'mean'
                }).reset_index()

                fig_resource = px.line(
                    resource_comparison,
                    x='date',
                    y=metric_choice,
                    color='resource_type',
                    title=f"Resource Type Comparison - {metric_choice.replace('_', ' ').title()}",
                    color_discrete_map={
                        'VM': '#8e44ad',
                        'Storage': '#e67e22',
                        'Container': '#27ae60'
                    }
                )
                fig_resource.update_layout(height=400)
                st.plotly_chart(fig_resource, width="stretch")
            else:
                st.info("📊 Raw data not available for resource comparison")

    # === 3. PATTERN ANALYSIS ===
    st.markdown("### 🔍 Pattern Analysis")

    pattern_col1, pattern_col2 = st.columns(2)

    with pattern_col1:
        # Weekly pattern
        df_agg['day_of_week'] = df_agg['date'].dt.day_name()
        weekly_pattern = df_agg.groupby('day_of_week')[metric_choice].mean().reset_index()

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern['day_of_week'] = pd.Categorical(weekly_pattern['day_of_week'], categories=day_order, ordered=True)
        weekly_pattern = weekly_pattern.sort_values('day_of_week')

        fig_weekly = px.bar(
            weekly_pattern,
            x='day_of_week',
            y=metric_choice,
            title=f"Weekly Pattern{filter_text}",
            color=metric_choice,
            color_continuous_scale='Blues'
        )
        fig_weekly.update_layout(height=350)
        st.plotly_chart(fig_weekly, width="stretch")

    with pattern_col2:
        # Monthly pattern (if enough data)
        if len(df_agg) > 30:
            df_agg['month'] = df_agg['date'].dt.month_name()
            monthly_pattern = df_agg.groupby('month')[metric_choice].mean().reset_index()

            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly_pattern['month'] = pd.Categorical(monthly_pattern['month'], categories=month_order, ordered=True)
            monthly_pattern = monthly_pattern.sort_values('month')

            fig_monthly = px.bar(
                monthly_pattern,
                x='month',
                y=metric_choice,
                title=f"Monthly Pattern{filter_text}",
                color=metric_choice,
                color_continuous_scale='Oranges'
            )
            fig_monthly.update_layout(height=350)
            fig_monthly.update_xaxes(tickangle=45)
            st.plotly_chart(fig_monthly, width="stretch")
        else:
            st.info("📊 Need more data points for monthly pattern analysis")

    # === 4. ADVANCED STATISTICS (MOVED TO EXPANDER) ===
    with st.expander("📊 Advanced Statistics & Insights", expanded=False):
        st.markdown("### 📊 Detailed Statistics")

        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)

        with stats_col1:
            volatility = df_agg[metric_choice].std()
            st.metric("📊 Volatility", f"{volatility:.2f}")

        with stats_col2:
            avg_value = df_agg[metric_choice].mean()
            st.metric("📈 Average", f"{avg_value:.1f}")

        with stats_col3:
            max_value = df_agg[metric_choice].max()
            min_value = df_agg[metric_choice].min()
            range_val = max_value - min_value
            st.metric("📏 Range", f"{range_val:.1f}")

        with stats_col4:
            median_value = df_agg[metric_choice].median()
            st.metric("🎯 Median", f"{median_value:.1f}")

        with stats_col5:
            if len(df_agg) > 1:
                correlation = np.corrcoef(range(len(df_agg)), df_agg[metric_choice])[0, 1]
                st.metric("📐 Trend Strength", f"{correlation:.3f}")
            else:
                st.metric("📐 Trend Strength", "N/A")

    # === 5. AI-POWERED INSIGHTS ===
    with st.expander("🤖 AI-Powered Insights & Recommendations", expanded=False):
        insights_col1, insights_col2 = st.columns(2)

        with insights_col1:
            st.markdown("**🔍 Pattern Detection:**")

            # Analyze trends
            if len(trend_values) > 7:
                recent_trend = trend_values.tail(7).mean()
                overall_trend = trend_values.mean()
                trend_momentum = ((recent_trend - overall_trend) / overall_trend) * 100

                if trend_momentum > 10:
                    st.success("📈 **Strong Upward Momentum** detected in recent data")
                elif trend_momentum < -10:
                    st.error("📉 **Strong Downward Momentum** detected in recent data")
                else:
                    st.info("➡️ **Stable Trend** - minimal momentum change")

                # Seasonality detection
                if show_patterns:
                    weekly_var = weekly_pattern[metric_choice].var()
                    if len(df_agg) > 0:
                        volatility = df_agg[metric_choice].std()
                        if weekly_var > volatility * 0.5:
                            st.warning("🗓️ **Strong Weekly Seasonality** detected")
                        else:
                            st.info("📅 **Minimal Weekly Seasonality** observed")
            else:
                st.info("Need more data points for advanced pattern detection")

        with insights_col2:
            st.markdown("**💡 Smart Recommendations:**")

            if len(df_agg) > 0:
                current_val = df_agg[metric_choice].iloc[-1]
                avg_val = df_agg[metric_choice].mean()

                if metric_choice == 'usage_cpu':
                    if current_val > avg_val * 1.2:
                        st.warning("⚠️ **High CPU Alert**: Consider scaling resources")
                    elif current_val < avg_val * 0.8:
                        st.success("✅ **CPU Optimized**: Resources efficiently utilized")
                    else:
                        st.info("📊 **CPU Normal**: Operating within expected range")

                elif metric_choice == 'usage_storage':
                    if current_val > avg_val * 1.15:
                        st.warning("💾 **Storage Growth**: Monitor capacity planning")
                    else:
                        st.success("💽 **Storage Stable**: Growth within normal range")

                elif metric_choice == 'users_active':
                    if current_val > avg_val * 1.1:
                        st.success("🚀 **User Growth**: Positive engagement trend")
                    elif current_val < avg_val * 0.9:
                        st.warning("👥 **User Decline**: Review engagement strategies")
                    else:
                        st.info("👤 **User Stable**: Consistent engagement levels")




# ===== TAB 3: ENHANCED REGIONAL PERFORMANCE ANALYSIS =====
with tab3:
    st.subheader("🌍 Regional Performance Analysis & Geographic Insights")

    # Load raw data for comprehensive regional analysis
    raw_data = fetch_api("data/raw")

    if not raw_data:
        st.error("❌ Unable to load regional data")
        st.stop()

    df_raw = pd.DataFrame(raw_data)
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    # === REGIONAL ANALYSIS CONTROLS ===
    st.markdown("**🎛️ Regional Analysis Settings:**")

    # Single row of controls - different from Tab 2's multi-row approach
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns(5)

    with ctrl_col1:
        analysis_metric = st.selectbox(
            "📊 Analysis Metric",
            ["usage_cpu", "usage_storage", "users_active"],
            format_func=lambda x: {
                "usage_cpu": "🔥 CPU Usage",
                "usage_storage": "💾 Storage Usage", 
                "users_active": "👥 Active Users"
            }[x]
        )

    with ctrl_col2:
        view_type = st.selectbox(
            "📈 View Type",
            ["Performance", "Distribution", "Comparison", "Rankings"]
        )

    with ctrl_col3:
        time_window = st.selectbox(
            "⏰ Time Window", 
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
        )

    with ctrl_col4:
        resource_focus = st.selectbox(
            "⚙️ Resource Focus",
            ["All Resources", "VM", "Storage", "Container"]
        )

    with ctrl_col5:
        show_insights = st.checkbox("🧠 Show Insights", value=True)

    # Apply filters
    df_filtered = df_raw.copy()

    # Time filter
    if time_window != "All Time":
        days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
        days = days_map[time_window]
        cutoff = df_filtered['date'].max() - timedelta(days=days)
        df_filtered = df_filtered[df_filtered['date'] >= cutoff]

    # Resource filter
    if resource_focus != "All Resources":
        df_filtered = df_filtered[df_filtered['resource_type'] == resource_focus]

    st.divider()

    # === REGIONAL PERFORMANCE OVERVIEW ===
    st.markdown("### 🎯 Regional Performance Overview")

    # Calculate regional statistics
    regional_stats = df_filtered.groupby('region').agg({
        analysis_metric: ['mean', 'max', 'min', 'std'],
        'date': 'count'
    }).round(2)

    # Flatten column names
    regional_stats.columns = ['avg', 'peak', 'min', 'volatility', 'data_points']
    regional_stats = regional_stats.reset_index()

    # Regional performance cards
    regions = regional_stats['region'].tolist()
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    colors = ['#0078d4', '#ff6b6b', '#4ecdc4', '#95e1d3']

    for idx, (col, region) in enumerate(zip([perf_col1, perf_col2, perf_col3, perf_col4], regions)):
        if idx < len(regions):
            stats = regional_stats[regional_stats['region'] == region].iloc[0]

            with col:
                # Create a colored container
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {colors[idx % len(colors)]}22, {colors[idx % len(colors)]}44); 
                           padding: 1rem; border-radius: 8px; border-left: 4px solid {colors[idx % len(colors)]};">
                    <h4 style="color: {colors[idx % len(colors)]}; margin: 0;">{region}</h4>
                    <h2 style="margin: 0.5rem 0;">{stats['avg']:.1f}</h2>
                    <p style="margin: 0; font-size: 0.9rem;">Average {analysis_metric.replace('_', ' ').title()}</p>
                    <hr style="margin: 0.5rem 0; border: 1px solid {colors[idx % len(colors)]}33;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                        <span>Peak: {stats['peak']:.1f}</span>
                        <span>Min: {stats['min']:.1f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # === MAIN VISUALIZATION BASED ON VIEW TYPE ===
    if view_type == "Performance":
        st.markdown("### 📊 Regional Performance Analysis")

        viz_col1, viz_col2 = st.columns([2, 1])

        with viz_col1:
            # Multi-metric regional comparison
            fig_performance = go.Figure()

            # Add bars for each metric component
            fig_performance.add_trace(go.Bar(
                name='Average Performance',
                x=regional_stats['region'],
                y=regional_stats['avg'],
                marker_color=colors,
                text=regional_stats['avg'].round(1),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Average: %{y:.1f}<extra></extra>'
            ))

            # Add line for volatility (on secondary y-axis)
            fig_performance.add_trace(go.Scatter(
                x=regional_stats['region'],
                y=regional_stats['volatility'],
                mode='lines+markers',
                name='Volatility',
                yaxis='y2',
                line=dict(color='orange', width=3),
                marker=dict(size=8, color='orange'),
                hovertemplate='<b>%{x}</b><br>Volatility: %{y:.1f}<extra></extra>'
            ))

            fig_performance.update_layout(
                title=f"Regional {analysis_metric.replace('_', ' ').title()} Performance",
                xaxis_title="Region",
                yaxis=dict(title=f"{analysis_metric.replace('_', ' ').title()}", side='left'),
                yaxis2=dict(title="Volatility", overlaying='y', side='right', showgrid=False),
                height=450,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_performance, width="stretch")

        with viz_col2:
            # Regional efficiency ranking
            st.markdown("**🏆 Regional Efficiency Ranking**")

            # Calculate efficiency (lower volatility + higher performance is better)
            regional_stats['efficiency_score'] = (
                (regional_stats['avg'] / regional_stats['avg'].max()) * 0.7 +
                (1 - regional_stats['volatility'] / regional_stats['volatility'].max()) * 0.3
            ) * 100

            ranked_regions = regional_stats.sort_values('efficiency_score', ascending=False)

            for idx, row in ranked_regions.iterrows():
                rank_emoji = ['🥇', '🥈', '🥉', '🏅'][min(idx, 3)]

                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.8rem; margin: 0.3rem 0; 
                           border-radius: 6px; border-left: 3px solid {colors[idx % len(colors)]};">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{rank_emoji}</span>
                            <strong>{row['region']}</strong>
                        </div>
                        <span style="font-weight: bold; color: {colors[idx % len(colors)]};">
                            {row['efficiency_score']:.0f}%
                        </span>
                    </div>
                    <div style="font-size: 0.8rem; color: #666; margin-top: 0.2rem;">
                        Avg: {row['avg']:.1f} | Volatility: {row['volatility']:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif view_type == "Distribution":
        st.markdown("### 🥧 Regional Resource Distribution")

        dist_col1, dist_col2 = st.columns(2)

        with dist_col1:
            # Resource distribution pie chart
            regional_totals = df_filtered.groupby('region')[analysis_metric].sum().reset_index()

            fig_pie = go.Figure(data=[go.Pie(
                labels=regional_totals['region'],
                values=regional_totals[analysis_metric],
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent}<br>%{value:.1f}'
            )])

            fig_pie.update_layout(
                title=f"Total {analysis_metric.replace('_', ' ').title()} Distribution",
                height=400,
                annotations=[dict(
                    text=f'Regional<br>Distribution', 
                    x=0.5, y=0.5, 
                    font_size=14, 
                    showarrow=False
                )]
            )

            st.plotly_chart(fig_pie, width="stretch")

        with dist_col2:
            # Resource type breakdown by region
            resource_breakdown = df_filtered.groupby(['region', 'resource_type'])[analysis_metric].mean().reset_index()

            fig_breakdown = px.bar(
                resource_breakdown,
                x='region',
                y=analysis_metric,
                color='resource_type',
                title=f"Average {analysis_metric.replace('_', ' ').title()} by Resource Type",
                color_discrete_map={'VM': '#8e44ad', 'Storage': '#e67e22', 'Container': '#27ae60'},
                height=400
            )

            fig_breakdown.update_layout(
                xaxis_title="Region",
                yaxis_title=analysis_metric.replace('_', ' ').title(),
                legend=dict(title="Resource Type")
            )

            st.plotly_chart(fig_breakdown, width="stretch")

    elif view_type == "Comparison":
        st.markdown("### ⚖️ Head-to-Head Regional Comparison")

        # Regional comparison matrix
        comparison_metrics = ['usage_cpu', 'usage_storage', 'users_active']

        # Create comparison data
        comparison_data = df_filtered.groupby('region')[comparison_metrics].mean().round(1)

        # Normalize for radar chart (0-100 scale)
        comparison_normalized = comparison_data.copy()
        for col in comparison_metrics:
            max_val = comparison_data[col].max()
            min_val = comparison_data[col].min()
            if max_val > min_val:
                comparison_normalized[col] = ((comparison_data[col] - min_val) / (max_val - min_val)) * 100
            else:
                comparison_normalized[col] = 50  # If all values are the same

        # Radar chart comparing regions
        fig_radar = go.Figure()

        for idx, region in enumerate(comparison_normalized.index):
            values = comparison_normalized.loc[region].tolist()
            values.append(values[0])  # Close the radar

            labels = ['CPU Usage', 'Storage Usage', 'User Activity']
            labels.append(labels[0])  # Close the radar

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=region,
                line_color=colors[idx % len(colors)],
                fillcolor=f'rgba({",".join([str(int(colors[idx % len(colors)][i:i+2], 16)) for i in (1, 3, 5)])}, 0.1)'
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickmode='linear',
                    tick0=0,
                    dtick=25
                )
            ),
            title="Regional Performance Radar Comparison (Normalized)",
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig_radar, width="stretch")

        # Detailed comparison table
        st.markdown("**📋 Detailed Regional Comparison**")

        # Add ranking for each metric
        for col in comparison_metrics:
            comparison_data[f'{col}_rank'] = comparison_data[col].rank(ascending=False).astype(int)

        # Display formatted table
        display_data = comparison_data.copy()

        # Format the display
        st.dataframe(
            display_data.style.format({
                'usage_cpu': '{:.1f}%',
                'usage_storage': '{:.1f} GB',
                'users_active': '{:.0f}',
                'usage_cpu_rank': '#{:.0f}',
                'usage_storage_rank': '#{:.0f}',
                'users_active_rank': '#{:.0f}'
            }).background_gradient(subset=['usage_cpu', 'usage_storage', 'users_active'], cmap='RdYlBu_r'),
            width="stretch"
        )

    elif view_type == "Rankings":
        st.markdown("### 🏆 Regional Performance Rankings")

        # Create comprehensive ranking system
        ranking_metrics = ['avg', 'peak', 'volatility']
        regional_stats_rank = regional_stats.copy()

        # Calculate ranks (lower volatility is better, so we reverse it)
        regional_stats_rank['avg_rank'] = regional_stats_rank['avg'].rank(ascending=False)
        regional_stats_rank['peak_rank'] = regional_stats_rank['peak'].rank(ascending=False)
        regional_stats_rank['volatility_rank'] = regional_stats_rank['volatility'].rank(ascending=True)  # Lower is better

        # Calculate overall score
        regional_stats_rank['overall_score'] = (
            regional_stats_rank['avg_rank'] * 0.4 +
            regional_stats_rank['peak_rank'] * 0.3 +
            regional_stats_rank['volatility_rank'] * 0.3
        )

        regional_stats_rank['overall_rank'] = regional_stats_rank['overall_score'].rank()

        # Sort by overall rank
        ranked_data = regional_stats_rank.sort_values('overall_rank')

        rank_col1, rank_col2 = st.columns([2, 1])

        with rank_col1:
            # Ranking bar chart
            fig_rank = go.Figure()

            fig_rank.add_trace(go.Bar(
                x=ranked_data['region'],
                y=ranked_data['avg'],
                name='Average Performance',
                marker_color=[colors[i % len(colors)] for i in range(len(ranked_data))],
                text=[f"#{int(rank)}" for rank in ranked_data['overall_rank']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg: %{y:.1f}<br>Rank: #%{text}<extra></extra>'
            ))

            fig_rank.update_layout(
                title=f"Regional Rankings - {analysis_metric.replace('_', ' ').title()}",
                xaxis_title="Region (Ordered by Overall Rank)",
                yaxis_title=analysis_metric.replace('_', ' ').title(),
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_rank, width="stretch")

        with rank_col2:
            st.markdown("**🏅 Overall Rankings**")

            for idx, (_, row) in enumerate(ranked_data.iterrows()):
                rank_icons = ['👑', '🥇', '🥈', '🥉']
                icon = rank_icons[min(idx, 3)] if idx < 4 else f"#{idx+1}"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {colors[idx % len(colors)]}22, {colors[idx % len(colors)]}44);
                           padding: 1rem; margin: 0.5rem 0; border-radius: 8px;
                           border-left: 4px solid {colors[idx % len(colors)]};">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                            <div>
                                <strong style="color: {colors[idx % len(colors)]};">{row['region']}</strong>
                                <br><small>Overall Score: {row['overall_score']:.1f}</small>
                            </div>
                        </div>
                        <div style="text-align: right; font-size: 0.8rem;">
                            <div>Avg: #{int(row['avg_rank'])}</div>
                            <div>Peak: #{int(row['peak_rank'])}</div>
                            <div>Stability: #{int(row['volatility_rank'])}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # === REGIONAL HEATMAP (ALWAYS SHOWN) ===
    st.divider()
    st.markdown("### 🔥 Regional Performance Heatmap")

    # Create heatmap data
    heatmap_data = df_filtered.groupby(['region', 'resource_type'])[analysis_metric].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='region', columns='resource_type', values=analysis_metric)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='RdYlBu_r',
        text=heatmap_pivot.values.round(1),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False,
        colorbar=dict(title=analysis_metric.replace('_', ' ').title())
    ))

    fig_heatmap.update_layout(
        title=f"Regional-Resource {analysis_metric.replace('_', ' ').title()} Heatmap",
        xaxis_title="Resource Type",
        yaxis_title="Region",
        height=300
    )

    st.plotly_chart(fig_heatmap, width="stretch")

    # === REGIONAL INSIGHTS (OPTIONAL) ===
    if show_insights:
        st.divider()
        st.markdown("### 🧠 Regional Performance Insights")

        insights_col1, insights_col2, insights_col3 = st.columns(3)

        with insights_col1:
            st.markdown("**🏆 Best Performers**")
            best_region = regional_stats.loc[regional_stats['avg'].idxmax()]
            most_stable = regional_stats.loc[regional_stats['volatility'].idxmin()]

            st.success(f"🥇 **Highest Average**: {best_region['region']} ({best_region['avg']:.1f})")
            st.info(f"📊 **Most Stable**: {most_stable['region']} (σ: {most_stable['volatility']:.1f})")

        with insights_col2:
            st.markdown("**📊 Key Statistics**")
            total_avg = regional_stats['avg'].mean()
            performance_gap = regional_stats['avg'].max() - regional_stats['avg'].min()

            st.metric("Global Average", f"{total_avg:.1f}")
            st.metric("Performance Gap", f"{performance_gap:.1f}")

        with insights_col3:
            st.markdown("**💡 Recommendations**")

            if performance_gap > total_avg * 0.2:  # 20% gap
                st.warning("⚠️ **High regional disparity** detected. Consider resource rebalancing.")
            else:
                st.success("✅ **Balanced regional performance** across all regions.")

            high_volatility = regional_stats[regional_stats['volatility'] > regional_stats['volatility'].mean()]
            if len(high_volatility) > 0:
                volatile_regions = ", ".join(high_volatility['region'].tolist())
                st.info(f"📈 **Monitor volatility** in: {volatile_regions}")

    else:
        st.error("❌ Unable to load regional data")



# ===== TAB 4: FIXED RESOURCE TYPE ANALYSIS =====
with tab4:
    st.subheader("⚙️ Resource Type Performance & Optimization Analysis")

    try:
        # Load raw data for comprehensive resource analysis
        raw_data = fetch_api("data/raw")

        if not raw_data:
            st.error("❌ Unable to load resource data")
            st.stop()

        df_raw = pd.DataFrame(raw_data)
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # Data validation
        required_columns = ['resource_type', 'usage_cpu', 'usage_storage', 'users_active']
        missing_columns = [col for col in required_columns if col not in df_raw.columns]

        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.stop()

        # Clean data - handle NaN and negative values
        df_raw = df_raw.dropna(subset=required_columns)
        df_raw['usage_cpu'] = df_raw['usage_cpu'].clip(0, 100)
        df_raw['usage_storage'] = df_raw['usage_storage'].clip(0, None)
        df_raw['users_active'] = df_raw['users_active'].clip(0, None)

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

    # === RESOURCE-FOCUSED CONTROL PANEL ===
    st.markdown("**🔧 Resource Analysis Dashboard:**")

    # Vertical control layout
    control_col1, control_col2 = st.columns([1, 3])

    with control_col1:
        st.markdown("**⚙️ Resource Settings**")

        available_resources = df_raw['resource_type'].unique().tolist()
        selected_resources = st.multiselect(
            "Resource Types",
            options=available_resources,
            default=available_resources,
            help="Select which resource types to analyze"
        )

        analysis_dimension = st.radio(
            "Analysis Focus",
            options=["Utilization", "Efficiency", "Capacity", "Cost-Benefit"],
            help="Choose the primary analysis dimension"
        )

        benchmark_mode = st.toggle(
            "⚡ Benchmark Mode",
            help="Compare resources against optimal performance thresholds"
        )

        show_optimization = st.toggle(
            "🎯 Show Optimization",
            value=True,
            help="Display optimization recommendations"
        )

    with control_col2:
        # Quick resource overview cards (horizontal layout)
        if selected_resources:
            st.markdown("**📊 Resource Overview**")

            try:
                overview_cols = st.columns(len(selected_resources))
                resource_colors = {'VM': '#8e44ad', 'Storage': '#e67e22', 'Container': '#27ae60'}

                for idx, resource in enumerate(selected_resources):
                    resource_data = df_raw[df_raw['resource_type'] == resource]

                    if not resource_data.empty:
                        # Safe calculations with error handling
                        avg_cpu = resource_data['usage_cpu'].mean() if len(resource_data) > 0 else 0
                        avg_storage = resource_data['usage_storage'].mean() if len(resource_data) > 0 else 0
                        avg_users = resource_data['users_active'].mean() if len(resource_data) > 0 else 0

                        color = resource_colors.get(resource, '#3498db')

                        with overview_cols[idx]:
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {color}22, {color}44);
                                       padding: 1rem; border-radius: 8px; text-align: center;
                                       border: 2px solid {color};">
                                <h3 style="color: {color}; margin: 0;">{resource}</h3>
                                <hr style="margin: 0.5rem 0; border-color: {color}66;">
                                <div style="font-size: 0.9rem;">
                                    <div>🔥 CPU: {avg_cpu:.1f}%</div>
                                    <div>💾 Storage: {avg_storage:.0f}GB</div>
                                    <div>👥 Users: {avg_users:.0f}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error creating overview cards: {str(e)}")

    if not selected_resources:
        st.warning("⚠️ Please select at least one resource type to analyze")
        st.stop()

    # Filter data by selected resources
    df_filtered = df_raw[df_raw['resource_type'].isin(selected_resources)]

    if df_filtered.empty:
        st.warning("⚠️ No data available for selected resource types")
        st.stop()

    st.divider()

    try:
        # === MAIN ANALYSIS SECTION ===
        if analysis_dimension == "Utilization":
            st.markdown("### 📈 Resource Utilization Analysis")

            util_col1, util_col2 = st.columns([2, 1])

            with util_col1:
                # FIXED: Import statement and error handling
                from plotly.subplots import make_subplots

                # Multi-resource utilization comparison
                fig_util = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('CPU Utilization Pattern', 'Storage Usage Pattern', 
                                   'User Activity Pattern', 'Resource Efficiency Score'),
                    specs=[[{}, {}], [{}, {}]],
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )

                for idx, resource in enumerate(selected_resources):
                    color = resource_colors.get(resource, '#3498db')
                    resource_data = df_filtered[df_filtered['resource_type'] == resource]

                    if not resource_data.empty:
                        # FIXED: Safe groupby with error handling
                        try:
                            daily_data = resource_data.groupby('date').agg({
                                'usage_cpu': 'mean',
                                'usage_storage': 'mean',
                                'users_active': 'mean'
                            }).reset_index()

                            if not daily_data.empty:
                                # CPU utilization (subplot 1)
                                fig_util.add_trace(
                                    go.Scatter(x=daily_data['date'], y=daily_data['usage_cpu'], 
                                              name=f'{resource} CPU', line=dict(color=color, width=2)),
                                    row=1, col=1
                                )

                                # Storage utilization (subplot 2)  
                                fig_util.add_trace(
                                    go.Scatter(x=daily_data['date'], y=daily_data['usage_storage'],
                                              name=f'{resource} Storage', line=dict(color=color, width=2, dash='dot')),
                                    row=1, col=2
                                )

                                # User activity (subplot 3)
                                fig_util.add_trace(
                                    go.Scatter(x=daily_data['date'], y=daily_data['users_active'],
                                              name=f'{resource} Users', line=dict(color=color, width=2, dash='dash')),
                                    row=2, col=1
                                )

                                # FIXED: Safe efficiency calculation
                                efficiency = []
                                for _, row in daily_data.iterrows():
                                    if row['users_active'] > 0:
                                        eff = (row['usage_cpu'] / row['users_active']) * 10
                                    else:
                                        eff = row['usage_cpu']  # Fallback if no users
                                    efficiency.append(eff)

                                # Efficiency score (subplot 4)
                                fig_util.add_trace(
                                    go.Scatter(x=daily_data['date'], y=efficiency,
                                              name=f'{resource} Efficiency', line=dict(color=color, width=3)),
                                    row=2, col=2
                                )
                        except Exception as subplot_error:
                            st.warning(f"Error processing {resource} data: {str(subplot_error)}")

                fig_util.update_layout(height=600, showlegend=True, 
                                      title_text="Comprehensive Resource Utilization Dashboard")
                st.plotly_chart(fig_util, width="stretch")

            with util_col2:
                # Resource performance matrix
                st.markdown("**🎯 Performance Matrix**")

                try:
                    for resource in selected_resources:
                        resource_data = df_filtered[df_filtered['resource_type'] == resource]
                        color = resource_colors.get(resource, '#3498db')

                        if not resource_data.empty:
                            # FIXED: Safe calculations with bounds checking
                            cpu_avg = resource_data['usage_cpu'].mean()
                            cpu_peak = resource_data['usage_cpu'].max() 
                            utilization_rate = min(100, max(0, cpu_avg))  # Clamp between 0-100

                            # Performance scoring
                            if utilization_rate > 80:
                                performance_status = "🔴 High Load"
                                performance_color = "#e74c3c"
                            elif utilization_rate > 60:
                                performance_status = "🟡 Moderate Load"
                                performance_color = "#f39c12"
                            else:
                                performance_status = "🟢 Optimal"
                                performance_color = "#27ae60"

                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; margin: 0.5rem 0; 
                                       border-radius: 8px; border-left: 4px solid {color};
                                       box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong style="color: {color};">{resource}</strong>
                                    <span style="color: {performance_color}; font-weight: bold;">{performance_status}</span>
                                </div>
                                <hr style="margin: 0.5rem 0; border-color: #eee;">
                                <div style="font-size: 0.9rem;">
                                    <div>📊 Avg CPU: {cpu_avg:.1f}%</div>
                                    <div>⚡ Peak CPU: {cpu_peak:.1f}%</div>
                                    <div>🎯 Utilization: {utilization_rate:.1f}%</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as matrix_error:
                    st.error(f"Error creating performance matrix: {str(matrix_error)}")

        elif analysis_dimension == "Efficiency":
            st.markdown("### ⚡ Resource Efficiency Analysis")

            eff_col1, eff_col2 = st.columns(2)

            with eff_col1:
                # Efficiency bubble chart
                st.markdown("**🎈 Resource Efficiency Bubble Chart**")

                try:
                    efficiency_data = []
                    for resource in selected_resources:
                        resource_data = df_filtered[df_filtered['resource_type'] == resource]
                        color = resource_colors.get(resource, '#3498db')

                        if not resource_data.empty:
                            # FIXED: Safe efficiency calculations
                            total_users = resource_data['users_active'].sum()
                            avg_cpu = resource_data['usage_cpu'].mean()
                            avg_storage = resource_data['usage_storage'].mean()
                            avg_users = resource_data['users_active'].mean()

                            # Avoid division by zero
                            if avg_users > 0:
                                cpu_efficiency = avg_cpu / avg_users
                                storage_efficiency = avg_storage / avg_users
                            else:
                                cpu_efficiency = avg_cpu
                                storage_efficiency = avg_storage

                            efficiency_data.append({
                                'resource_type': resource,
                                'cpu_efficiency': cpu_efficiency,
                                'storage_efficiency': storage_efficiency,
                                'total_users': total_users,
                                'color': color
                            })

                    if efficiency_data:
                        eff_df = pd.DataFrame(efficiency_data)

                        fig_bubble = go.Figure()

                        for _, row in eff_df.iterrows():
                            # FIXED: Safe bubble size calculation
                            bubble_size = max(20, min(80, row['total_users'] / 10 if row['total_users'] > 0 else 20))

                            fig_bubble.add_trace(go.Scatter(
                                x=[row['cpu_efficiency']],
                                y=[row['storage_efficiency']],
                                mode='markers',
                                marker=dict(
                                    size=bubble_size,
                                    color=row['color'],
                                    opacity=0.7,
                                    line=dict(width=2, color='white')
                                ),
                                name=row['resource_type'],
                                text=f"{row['resource_type']}<br>Users: {row['total_users']:.0f}",
                                hovertemplate='<b>%{text}</b><br>CPU/User: %{x:.2f}<br>Storage/User: %{y:.1f}<extra></extra>'
                            ))

                        fig_bubble.update_layout(
                            title="Resource Efficiency: CPU vs Storage per User",
                            xaxis_title="CPU Usage per User (%)",
                            yaxis_title="Storage Usage per User (GB)",
                            height=400,
                            showlegend=True
                        )

                        st.plotly_chart(fig_bubble, width="stretch")
                    else:
                        st.warning("No efficiency data available")

                except Exception as bubble_error:
                    st.error(f"Error creating bubble chart: {str(bubble_error)}")

            with eff_col2:
                # Efficiency scoring
                st.markdown("**🏆 Efficiency Scoring**")

                try:
                    efficiency_scores = []
                    for resource in selected_resources:
                        resource_data = df_filtered[df_filtered['resource_type'] == resource]
                        color = resource_colors.get(resource, '#3498db')

                        if not resource_data.empty:
                            # FIXED: Improved efficiency scoring logic
                            avg_cpu = resource_data['usage_cpu'].mean()
                            avg_users = resource_data['users_active'].mean()
                            avg_storage = resource_data['usage_storage'].mean()

                            # Efficiency metrics (higher is better)
                            cpu_efficiency = max(0, min(100, 100 - avg_cpu))  # Lower CPU usage is more efficient

                            if avg_cpu > 0:
                                user_efficiency = min(100, (avg_users / avg_cpu) * 50)  # Users per CPU unit
                            else:
                                user_efficiency = 50  # Neutral score

                            storage_efficiency = max(0, min(100, 100 - min(100, avg_storage / 20)))  # Normalized storage

                            # Overall score (weighted average)
                            overall_score = (cpu_efficiency * 0.4 + user_efficiency * 0.4 + storage_efficiency * 0.2)
                            overall_score = max(0, min(100, overall_score))

                            efficiency_scores.append({
                                'resource': resource,
                                'score': overall_score,
                                'cpu_eff': cpu_efficiency,
                                'user_eff': user_efficiency,
                                'storage_eff': storage_efficiency,
                                'color': color
                            })

                    # Sort by overall score
                    efficiency_scores.sort(key=lambda x: x['score'], reverse=True)

                    for idx, score_data in enumerate(efficiency_scores):
                        resource = score_data['resource']
                        score = score_data['score']
                        color = score_data['color']

                        # Determine grade
                        if score >= 80:
                            grade = "A+"
                            grade_color = "#27ae60"
                        elif score >= 70:
                            grade = "A"
                            grade_color = "#2ecc71"
                        elif score >= 60:
                            grade = "B+"
                            grade_color = "#f39c12"
                        elif score >= 50:
                            grade = "B"
                            grade_color = "#e67e22"
                        else:
                            grade = "C"
                            grade_color = "#e74c3c"

                        rank_emoji = ['🥇', '🥈', '🥉'][min(idx, 2)] if len(efficiency_scores) > idx else f"#{idx+1}"

                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color}11, {color}22);
                                   padding: 1rem; margin: 0.5rem 0; border-radius: 8px;
                                   border: 2px solid {color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="display: flex; align-items: center;">
                                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{rank_emoji}</span>
                                    <strong style="color: {color};">{resource}</strong>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.2rem; font-weight: bold; color: {grade_color};">{grade}</div>
                                    <div style="font-size: 0.9rem;">Score: {score:.1f}</div>
                                </div>
                            </div>
                            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>CPU: {score_data['cpu_eff']:.1f}</span>
                                    <span>User: {score_data['user_eff']:.1f}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as scoring_error:
                    st.error(f"Error creating efficiency scoring: {str(scoring_error)}")

        elif analysis_dimension == "Capacity":
            st.markdown("### 📊 Resource Capacity Analysis")

            cap_col1, cap_col2 = st.columns([3, 1])

            with cap_col1:
                # Capacity utilization over time
                st.markdown("**📈 Capacity Utilization Timeline**")

                try:
                    fig_capacity = go.Figure()

                    for resource in selected_resources:
                        resource_data = df_filtered[df_filtered['resource_type'] == resource]
                        color = resource_colors.get(resource, '#3498db')

                        if not resource_data.empty:
                            # FIXED: Safe daily aggregation
                            try:
                                daily_data = resource_data.groupby('date').agg({
                                    'usage_cpu': 'max',
                                    'usage_storage': 'max',
                                    'users_active': 'max'
                                }).reset_index()

                                if not daily_data.empty:
                                    # Calculate capacity utilization percentage (assuming 100% CPU is max capacity)
                                    cpu_utilization = daily_data['usage_cpu'].clip(0, 100)

                                    fig_capacity.add_trace(go.Scatter(
                                        x=daily_data['date'],
                                        y=cpu_utilization,
                                        mode='lines+markers',
                                        name=f'{resource} Capacity',
                                        line=dict(color=color, width=3),
                                        hovertemplate=f'<b>{resource}</b><br>Date: %{{x}}<br>Capacity: %{{y:.1f}}%<extra></extra>'
                                    ))
                            except Exception as daily_error:
                                st.warning(f"Error processing daily data for {resource}: {str(daily_error)}")

                    # Add capacity thresholds
                    fig_capacity.add_hline(y=80, line_dash="dash", line_color="orange", 
                                          annotation_text="Warning Threshold (80%)")
                    fig_capacity.add_hline(y=95, line_dash="dash", line_color="red", 
                                          annotation_text="Critical Threshold (95%)")

                    fig_capacity.update_layout(
                        title="Resource Capacity Utilization Over Time",
                        xaxis_title="Date",
                        yaxis_title="Capacity Utilization (%)",
                        height=400,
                        yaxis=dict(range=[0, 110])
                    )

                    st.plotly_chart(fig_capacity, width="stretch")

                except Exception as capacity_error:
                    st.error(f"Error creating capacity chart: {str(capacity_error)}")

            with cap_col2:
                # Capacity status indicators
                st.markdown("**🚦 Capacity Status**")

                try:
                    for resource in selected_resources:
                        resource_data = df_filtered[df_filtered['resource_type'] == resource]
                        color = resource_colors.get(resource, '#3498db')

                        if not resource_data.empty and len(resource_data) > 0:
                            # FIXED: Safe current and max calculation
                            current_cpu = resource_data['usage_cpu'].iloc[-1] if len(resource_data) > 0 else 0
                            max_cpu = resource_data['usage_cpu'].max()

                            # Capacity status
                            if max_cpu >= 95:
                                status = "🔴 Critical"
                                status_color = "#e74c3c"
                            elif max_cpu >= 80:
                                status = "🟡 Warning"
                                status_color = "#f39c12"
                            else:
                                status = "🟢 Normal"
                                status_color = "#27ae60"

                            # FIXED: Safe trend calculation
                            capacity_projection = "Insufficient data"
                            if len(resource_data) > 14:  # Need at least 14 days for trend
                                try:
                                    recent_avg = resource_data['usage_cpu'].tail(7).mean()
                                    previous_avg = resource_data['usage_cpu'].head(7).mean()

                                    if recent_avg > previous_avg and previous_avg > 0:
                                        trend_rate = (recent_avg - previous_avg) / 7  # Daily trend
                                        if trend_rate > 0.1 and current_cpu < 100:  # Significant upward trend
                                            days_to_capacity = (100 - current_cpu) / trend_rate
                                            if days_to_capacity < 365:
                                                capacity_projection = f"{days_to_capacity:.0f} days"
                                            else:
                                                capacity_projection = "1+ year"
                                        else:
                                            capacity_projection = "Stable"
                                    else:
                                        capacity_projection = "Stable/Decreasing"
                                except:
                                    capacity_projection = "Calculation error"

                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; margin: 0.5rem 0; 
                                       border-radius: 8px; border-left: 4px solid {color};
                                       box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <strong style="color: {color};">{resource}</strong>
                                    <span style="color: {status_color}; font-weight: bold; font-size: 0.9rem;">{status}</span>
                                </div>
                                <div style="font-size: 0.9rem;">
                                    <div>Current: {current_cpu:.1f}%</div>
                                    <div>Peak: {max_cpu:.1f}%</div>
                                    <div style="margin-top: 0.3rem; font-size: 0.8rem; color: #666;">
                                        Time to Capacity:<br><strong>{capacity_projection}</strong>
                                    </div>
                                </div>
                                <div style="background: linear-gradient(90deg, {status_color}22 0%, transparent 100%);
                                           padding: 0.2rem; border-radius: 3px; margin-top: 0.3rem;">
                                    <div style="width: {min(100, max_cpu)}%; background: {status_color}; height: 4px; border-radius: 2px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as status_error:
                    st.error(f"Error creating capacity status: {str(status_error)}")

        elif analysis_dimension == "Cost-Benefit":
            st.markdown("### 💰 Resource Cost-Benefit Analysis")

            cb_col1, cb_col2 = st.columns([2, 1])

            with cb_col1:
                # Cost-benefit matrix
                st.markdown("**💰 Resource Value Analysis**")

                try:
                    # FIXED: Safer cost-benefit calculations
                    cost_benefit_data = []
                    base_costs = {'VM': 1.0, 'Storage': 0.3, 'Container': 0.8}  # Relative cost per unit

                    for resource in selected_resources:
                        resource_data = df_filtered[df_filtered['resource_type'] == resource]
                        color = resource_colors.get(resource, '#3498db')

                        if not resource_data.empty:
                            # Calculate benefit metrics safely
                            avg_users_served = resource_data['users_active'].mean()
                            avg_cpu = resource_data['usage_cpu'].mean()

                            # Avoid division by zero in efficiency calculation
                            if avg_cpu > 0:
                                resource_efficiency = avg_users_served / avg_cpu
                            else:
                                resource_efficiency = avg_users_served  # Fallback

                            # Calculate cost metrics
                            estimated_cost = avg_cpu * base_costs.get(resource, 1.0)

                            # Safe cost per user calculation
                            if avg_users_served > 0:
                                cost_per_user = estimated_cost / avg_users_served
                            else:
                                cost_per_user = estimated_cost  # Fallback

                            # ROI calculation with safety checks
                            if estimated_cost > 0:
                                roi = (resource_efficiency * 100) / estimated_cost
                            else:
                                roi = resource_efficiency * 100  # Fallback

                            cost_benefit_data.append({
                                'resource': resource,
                                'cost': max(0.1, estimated_cost),  # Minimum cost to avoid zero
                                'benefit': max(0.1, resource_efficiency * 10),  # Scale and minimum
                                'roi': max(0, roi),
                                'cost_per_user': cost_per_user,
                                'users_served': avg_users_served,
                                'color': color
                            })

                    if cost_benefit_data:
                        cb_df = pd.DataFrame(cost_benefit_data)

                        # Cost-benefit scatter plot
                        fig_cb = go.Figure()

                        for _, row in cb_df.iterrows():
                            # FIXED: Safe bubble size calculation
                            bubble_size = max(20, min(60, row['users_served'] / 5 if row['users_served'] > 0 else 20))

                            fig_cb.add_trace(go.Scatter(
                                x=[row['cost']],
                                y=[row['benefit']],
                                mode='markers+text',
                                marker=dict(
                                    size=bubble_size,
                                    color=row['color'],
                                    opacity=0.7,
                                    line=dict(width=2, color='white')
                                ),
                                name=row['resource'],
                                text=row['resource'],
                                textposition='middle center',
                                textfont=dict(color='white', size=12, family='Arial Black'),
                                hovertemplate=f"<b>{row['resource']}</b><br>" +
                                             f"Cost: {row['cost']:.1f}<br>" +
                                             f"Benefit: {row['benefit']:.1f}<br>" +
                                             f"ROI: {row['roi']:.1f}%<br>" +
                                             f"Users: {row['users_served']:.0f}<extra></extra>"
                            ))

                        # Safe quadrant calculation
                        max_cost = cb_df['cost'].max() * 1.1 if cb_df['cost'].max() > 0 else 1
                        max_benefit = cb_df['benefit'].max() * 1.1 if cb_df['benefit'].max() > 0 else 1

                        fig_cb.add_shape(type="rect", x0=0, y0=max_benefit*0.7, x1=max_cost*0.5, y1=max_benefit,
                                        fillcolor="rgba(39, 174, 96, 0.1)", line=dict(color="rgba(39, 174, 96, 0.3)"))
                        fig_cb.add_annotation(x=max_cost*0.25, y=max_benefit*0.85, text="High Value<br>(Low Cost, High Benefit)",
                                            showarrow=False, font=dict(size=10, color="#27ae60"))

                        fig_cb.update_layout(
                            title="Cost-Benefit Analysis: Resource Value Matrix",
                            xaxis_title="Relative Cost",
                            yaxis_title="Relative Benefit",
                            height=400,
                            showlegend=False
                        )

                        st.plotly_chart(fig_cb, width="stretch")
                    else:
                        st.warning("No cost-benefit data available")

                except Exception as cb_error:
                    st.error(f"Error creating cost-benefit analysis: {str(cb_error)}")

            with cb_col2:
                # ROI ranking
                st.markdown("**📈 ROI Rankings**")

                try:
                    if 'cb_df' in locals() and not cb_df.empty:
                        cb_df_sorted = cb_df.sort_values('roi', ascending=False)

                        for idx, (_, row) in enumerate(cb_df_sorted.iterrows()):
                            resource = row['resource']
                            roi = row['roi']
                            color = row['color']

                            # ROI status
                            if roi >= 20:
                                roi_status = "🚀 Excellent"
                                roi_color = "#27ae60"
                            elif roi >= 15:
                                roi_status = "✅ Good"
                                roi_color = "#2ecc71"
                            elif roi >= 10:
                                roi_status = "⚠️ Average"
                                roi_color = "#f39c12"
                            else:
                                roi_status = "❌ Poor"
                                roi_color = "#e74c3c"

                            rank_emoji = ['🥇', '🥈', '🥉'][min(idx, 2)] if idx < 3 else f"#{idx+1}"

                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {color}15, {color}25);
                                       padding: 1rem; margin: 0.5rem 0; border-radius: 8px;
                                       border-left: 4px solid {color};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div style="display: flex; align-items: center;">
                                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{rank_emoji}</span>
                                        <strong style="color: {color};">{resource}</strong>
                                    </div>
                                    <span style="color: {roi_color}; font-weight: bold; font-size: 0.9rem;">{roi_status}</span>
                                </div>
                                <hr style="margin: 0.5rem 0; border-color: {color}44;">
                                <div style="font-size: 0.9rem;">
                                    <div><strong>ROI: {roi:.1f}%</strong></div>
                                    <div>Cost/User: {row['cost_per_user']:.2f}</div>
                                    <div>Users Served: {row['users_served']:.0f}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No ROI data available")

                except Exception as roi_error:
                    st.error(f"Error creating ROI rankings: {str(roi_error)}")

        # === OPTIMIZATION RECOMMENDATIONS (ALWAYS SHOWN IF ENABLED) ===
        if show_optimization:
            st.divider()
            st.markdown("### 🎯 Resource Optimization Recommendations")

            try:
                opt_col1, opt_col2, opt_col3 = st.columns(3)

                # Generate recommendations based on analysis
                recommendations = []

                for resource in selected_resources:
                    resource_data = df_filtered[df_filtered['resource_type'] == resource]

                    if not resource_data.empty:
                        # FIXED: Safe recommendation calculations
                        avg_cpu = resource_data['usage_cpu'].mean()
                        max_cpu = resource_data['usage_cpu'].max()

                        # Safe volatility calculation
                        if len(resource_data) > 1:
                            volatility = resource_data['usage_cpu'].std()
                        else:
                            volatility = 0

                        if max_cpu > 90:
                            recommendations.append({
                                'type': '🚨 Critical',
                                'resource': resource,
                                'message': f"{resource} approaching capacity limits",
                                'action': "Consider scaling up or load balancing",
                                'priority': 'high'
                            })
                        elif avg_cpu < 30 and volatility < 10:
                            recommendations.append({
                                'type': '💡 Efficiency',
                                'resource': resource,
                                'message': f"{resource} is underutilized",
                                'action': "Consider consolidation or downsizing",
                                'priority': 'medium'
                            })
                        elif volatility > 25:
                            recommendations.append({
                                'type': '📊 Stability',
                                'resource': resource,
                                'message': f"{resource} shows high variability",
                                'action': "Implement auto-scaling or load smoothing",
                                'priority': 'medium'
                            })

                if not recommendations:
                    recommendations.append({
                        'type': '✅ Optimal',
                        'resource': 'All Resources',
                        'message': 'Resources are operating within optimal ranges',
                        'action': 'Continue monitoring for trends',
                        'priority': 'low'
                    })

                # Display recommendations in columns
                rec_cols = [opt_col1, opt_col2, opt_col3]
                priority_colors = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#27ae60'}

                for idx, rec in enumerate(recommendations[:3]):  # Show top 3 recommendations
                    with rec_cols[idx % 3]:
                        priority_color = priority_colors.get(rec['priority'], '#3498db')

                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px;
                                   border-left: 4px solid {priority_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                   margin-bottom: 1rem;">
                            <div style="color: {priority_color}; font-weight: bold; margin-bottom: 0.5rem;">
                                {rec['type']}
                            </div>
                            <div style="font-weight: bold; margin-bottom: 0.3rem;">
                                {rec['resource']}
                            </div>
                            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">
                                {rec['message']}
                            </div>
                            <div style="background: {priority_color}22; padding: 0.5rem; border-radius: 4px;">
                                <strong>Action:</strong> {rec['action']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Additional recommendations if more than 3
                if len(recommendations) > 3:
                    with st.expander(f"📋 View All {len(recommendations)} Recommendations"):
                        for rec in recommendations[3:]:
                            priority_color = priority_colors.get(rec['priority'], '#3498db')
                            st.markdown(f"**{rec['type']}** - {rec['resource']}: {rec['message']} → *{rec['action']}*")

            except Exception as opt_error:
                st.error(f"Error generating recommendations: {str(opt_error)}")

    except Exception as main_error:
        st.error(f"❌ Main analysis error: {str(main_error)}")
        st.info("Please check your data and try again")





# ===== TAB 5: CORRECTED CORRELATION & STATISTICAL ANALYSIS =====
with tab5:
    st.subheader("🔗 Correlation Analysis & Statistical Insights")

    try:
        # Load raw data for comprehensive correlation analysis
        raw_data = fetch_api("data/raw")

        if not raw_data:
            st.error("❌ Unable to load correlation data")
            st.stop()

        df_raw = pd.DataFrame(raw_data)
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # Data validation for correlation analysis
        numeric_columns = ['usage_cpu', 'usage_storage', 'users_active', 'economic_index', 'cloud_market_demand']
        available_columns = [col for col in numeric_columns if col in df_raw.columns]

        if len(available_columns) < 2:
            st.error("❌ Insufficient numeric columns for correlation analysis")
            st.stop()

        # Clean data for correlation analysis
        df_clean = df_raw[available_columns + ['region', 'resource_type', 'date']].dropna()

        # === CORRELATION ANALYSIS CONTROL PANEL (UNIQUE GRID LAYOUT) ===
        st.markdown("**🔬 Statistical Analysis Configuration:**")

        # Grid layout (different from other tabs' layouts)
        config_container = st.container()

        with config_container:
            # Top row - main controls
            main_col1, main_col2, main_col3, main_col4 = st.columns(4)

            with main_col1:
                correlation_method = st.selectbox(
                    "📊 Correlation Method",
                    options=['pearson', 'spearman', 'kendall'],
                    format_func=lambda x: {
                        'pearson': '📈 Pearson (Linear)',
                        'spearman': '📉 Spearman (Rank)',
                        'kendall': '🔄 Kendall (Tau)'
                    }[x],
                    help="Choose correlation calculation method"
                )

            with main_col2:
                analysis_scope = st.selectbox(
                    "🎯 Analysis Scope",
                    options=['Overall', 'By Region', 'By Resource', 'Time Series'],
                    help="Define the scope of correlation analysis"
                )

            with main_col3:
                significance_level = st.selectbox(
                    "📏 Significance Level",
                    options=[0.05, 0.01, 0.001],
                    format_func=lambda x: f"α = {x}",
                    help="Statistical significance threshold"
                )

            with main_col4:
                min_correlation = st.slider(
                    "🎚️ Min Correlation",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Minimum correlation magnitude to display"
                )

            # Bottom row - visualization controls
            viz_col1, viz_col2, viz_col3, viz_col4 = st.columns(4)

            with viz_col1:
                show_p_values = st.checkbox("📊 Show P-Values", value=True)

            with viz_col2:
                advanced_stats = st.checkbox("🧮 Advanced Statistics", value=True)

            with viz_col3:
                interactive_mode = st.checkbox("🎮 Interactive Mode", value=True)

            with viz_col4:
                export_results = st.checkbox("💾 Export Results", value=False)

        st.divider()

        # Calculate correlations based on scope
        if analysis_scope == 'Overall':
            correlation_data = df_clean[available_columns].corr(method=correlation_method)
            analysis_groups = {'Overall': df_clean}
        elif analysis_scope == 'By Region':
            analysis_groups = {region: group for region, group in df_clean.groupby('region')}
        elif analysis_scope == 'By Resource':
            analysis_groups = {resource: group for resource, group in df_clean.groupby('resource_type')}
        elif analysis_scope == 'Time Series':
            # Monthly aggregation for time series analysis
            df_clean['month'] = df_clean['date'].dt.to_period('M')
            analysis_groups = {str(month): group for month, group in df_clean.groupby('month')}

        # === MAIN CORRELATION VISUALIZATION ===
        if analysis_scope == 'Overall':
            st.markdown("### 🌡️ Overall Correlation Heatmap")

            # Enhanced correlation heatmap
            correlation_matrix = df_clean[available_columns].corr(method=correlation_method)

            # Calculate p-values if requested
            if show_p_values:
                import scipy.stats as stats
                p_values = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)

                for i in correlation_matrix.index:
                    for j in correlation_matrix.columns:
                        if i != j:
                            try:
                                corr, p_val = stats.pearsonr(df_clean[i].dropna(), df_clean[j].dropna())
                                p_values.loc[i, j] = p_val
                            except:
                                p_values.loc[i, j] = 1.0  # No significance if calculation fails
                        else:
                            p_values.loc[i, j] = 0.0

                # Create significance mask
                significance_mask = p_values < significance_level

            # FIXED: Create enhanced heatmap with corrected colorbar properties
            fig_heatmap = go.Figure()

            # Add correlation heatmap with FIXED colorbar configuration
            heatmap_trace = go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=correlation_matrix.values.round(3),
                texttemplate='%{text}',
                textfont={"size": 12, "color": "white"},
                hoverongaps=False,
                colorbar=dict(
                    title="Correlation Coefficient",
                    # REMOVED: titleside (not supported in this Plotly version)
                    # REPLACED WITH: Standard colorbar properties
                    x=1.02,  # Position colorbar to the right
                    xanchor='left',
                    tickmode="linear",
                    tick0=-1,
                    dtick=0.5,
                    len=0.9  # Length of colorbar
                )
            )

            fig_heatmap.add_trace(heatmap_trace)

            # Add significance markers if p-values are shown
            if show_p_values and 'significance_mask' in locals():
                for i, row in enumerate(correlation_matrix.index):
                    for j, col in enumerate(correlation_matrix.columns):
                        try:
                            if significance_mask.iloc[i, j] and abs(correlation_matrix.iloc[i, j]) >= min_correlation:
                                fig_heatmap.add_shape(
                                    type="rect",
                                    x0=j-0.4, y0=i-0.4, x1=j+0.4, y1=i+0.4,
                                    line=dict(color="yellow", width=3),
                                    fillcolor="rgba(0,0,0,0)"
                                )
                        except:
                            continue  # Skip if significance calculation failed

            fig_heatmap.update_layout(
                title=f"Correlation Matrix ({correlation_method.title()} Method)",
                xaxis_title="Variables",
                yaxis_title="Variables",
                height=500,
                font=dict(size=12)
            )

            st.plotly_chart(fig_heatmap, width="stretch")

            # Correlation insights
            if advanced_stats:
                with st.expander("🔍 Correlation Insights"):
                    insights_col1, insights_col2 = st.columns(2)

                    with insights_col1:
                        st.markdown("**📈 Strongest Correlations:**")

                        # Find strongest correlations
                        correlations_list = []
                        for i in range(len(correlation_matrix.columns)):
                            for j in range(i+1, len(correlation_matrix.columns)):
                                var1 = correlation_matrix.columns[i]
                                var2 = correlation_matrix.columns[j]
                                corr_val = correlation_matrix.iloc[i, j]
                                if abs(corr_val) >= min_correlation and not np.isnan(corr_val):
                                    correlations_list.append({
                                        'pair': f"{var1} ↔ {var2}",
                                        'correlation': corr_val,
                                        'strength': 'Strong' if abs(corr_val) >= 0.7 else 'Moderate' if abs(corr_val) >= 0.3 else 'Weak'
                                    })

                        # Sort by absolute correlation
                        correlations_list.sort(key=lambda x: abs(x['correlation']), reverse=True)

                        if correlations_list:
                            for corr_data in correlations_list[:5]:  # Show top 5
                                strength_color = {
                                    'Strong': '#27ae60',
                                    'Moderate': '#f39c12', 
                                    'Weak': '#95a5a6'
                                }[corr_data['strength']]

                                st.markdown(f"""
                                <div style="background: {strength_color}22; padding: 0.5rem; margin: 0.3rem 0; border-radius: 6px;
                                           border-left: 3px solid {strength_color};">
                                    <strong>{corr_data['pair']}</strong><br>
                                    <span style="color: {strength_color};">
                                        r = {corr_data['correlation']:.3f} ({corr_data['strength']})
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No significant correlations found above the minimum threshold")

                    with insights_col2:
                        st.markdown("**📊 Statistical Summary:**")

                        # Calculate summary statistics
                        abs_correlations = np.abs(correlation_matrix.values)
                        # Remove diagonal (self-correlations) and get upper triangle
                        mask = np.triu(np.ones_like(abs_correlations, dtype=bool), k=1)
                        abs_correlations = abs_correlations[mask]

                        # Remove NaN values
                        abs_correlations = abs_correlations[~np.isnan(abs_correlations)]

                        if len(abs_correlations) > 0:
                            st.metric("🎯 Average Correlation", f"{np.mean(abs_correlations):.3f}")
                            st.metric("📏 Max Correlation", f"{np.max(abs_correlations):.3f}")
                            st.metric("📐 Std Deviation", f"{np.std(abs_correlations):.3f}")

                            # Correlation strength distribution
                            strong_count = sum(1 for c in abs_correlations if c >= 0.7)
                            moderate_count = sum(1 for c in abs_correlations if 0.3 <= c < 0.7)
                            weak_count = sum(1 for c in abs_correlations if c < 0.3)

                            st.markdown(f"""
                            **🔍 Correlation Distribution:**
                            - 🟢 Strong (≥0.7): {strong_count}
                            - 🟡 Moderate (0.3-0.7): {moderate_count}  
                            - ⚪ Weak (<0.3): {weak_count}
                            """)
                        else:
                            st.warning("No valid correlations calculated")

        else:
            # Multi-group correlation analysis
            st.markdown(f"### 📊 Correlation Analysis: {analysis_scope}")

            # Create comparison visualization
            group_correlations = {}

            # Calculate correlations for each group
            for group_name, group_data in list(analysis_groups.items())[:6]:  # Limit to 6 groups for performance
                if len(group_data) > 3:  # Need minimum data for correlation
                    try:
                        group_corr = group_data[available_columns].corr(method=correlation_method)
                        # Check if correlation matrix has valid values
                        if not group_corr.isnull().all().all():
                            group_correlations[group_name] = group_corr
                    except:
                        continue  # Skip groups with calculation issues

            if group_correlations:
                # Multi-group heatmap comparison
                n_groups = len(group_correlations)
                cols_per_row = 2
                n_rows = (n_groups + cols_per_row - 1) // cols_per_row

                from plotly.subplots import make_subplots

                fig_multi = make_subplots(
                    rows=n_rows,
                    cols=cols_per_row,
                    subplot_titles=list(group_correlations.keys()),
                    vertical_spacing=0.08,
                    horizontal_spacing=0.05
                )

                for idx, (group_name, corr_matrix) in enumerate(group_correlations.items()):
                    row = (idx // cols_per_row) + 1
                    col = (idx % cols_per_row) + 1

                    # FIXED: Simplified colorbar configuration for subplots
                    fig_multi.add_trace(
                        go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            colorscale='RdBu',
                            zmid=0,
                            zmin=-1,
                            zmax=1,
                            showscale=(idx == 0),  # Show colorbar only for first plot
                            text=corr_matrix.values.round(2),
                            texttemplate='%{text}',
                            textfont={"size": 8},
                            hoverongaps=False
                        ),
                        row=row, col=col
                    )

                fig_multi.update_layout(
                    title=f"Correlation Comparison: {analysis_scope}",
                    height=300 * n_rows,
                    showlegend=False
                )

                st.plotly_chart(fig_multi, width="stretch")
            else:
                st.warning("⚠️ No valid correlation data for selected groups")

        # === INTERACTIVE CORRELATION EXPLORER ===
        if interactive_mode:
            st.divider()
            st.markdown("### 🎮 Interactive Correlation Explorer")

            explorer_col1, explorer_col2 = st.columns([1, 2])

            with explorer_col1:
                st.markdown("**🎛️ Variable Selection:**")

                var_x = st.selectbox(
                    "X-Axis Variable",
                    options=available_columns,
                    index=0 if available_columns else None
                )

                var_y = st.selectbox(
                    "Y-Axis Variable", 
                    options=available_columns,
                    index=1 if len(available_columns) > 1 else 0
                )

                color_by = st.selectbox(
                    "Color By",
                    options=['None', 'region', 'resource_type'],
                    index=1
                )

                size_by = st.selectbox(
                    "Size By",
                    options=['None'] + available_columns,
                    index=0
                )

                # Correlation calculation for selected pair
                if var_x and var_y and var_x != var_y:
                    try:
                        correlation_value = df_clean[var_x].corr(df_clean[var_y], method=correlation_method)

                        if not np.isnan(correlation_value):
                            # Determine correlation strength and color
                            if abs(correlation_value) >= 0.7:
                                strength = "Strong"
                                strength_color = "#27ae60"
                            elif abs(correlation_value) >= 0.3:
                                strength = "Moderate"
                                strength_color = "#f39c12"
                            else:
                                strength = "Weak"
                                strength_color = "#95a5a6"

                            st.markdown(f"""
                            <div style="background: {strength_color}22; padding: 1rem; border-radius: 8px;
                                       border-left: 4px solid {strength_color}; text-align: center;">
                                <h3 style="color: {strength_color}; margin: 0;">{correlation_value:.3f}</h3>
                                <p style="margin: 0.5rem 0 0 0;">{strength} Correlation</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Cannot calculate correlation for selected variables")
                    except Exception as corr_error:
                        st.error(f"Error calculating correlation: {str(corr_error)}")

            with explorer_col2:
                if var_x and var_y and var_x != var_y:
                    try:
                        # Create interactive scatter plot
                        scatter_fig = go.Figure()

                        if color_by != 'None' and color_by in df_clean.columns:
                            # Group by color variable
                            color_map = {
                                'region': {'East US': '#0078d4', 'West US': '#ff6b6b', 'North Europe': '#4ecdc4', 'Southeast Asia': '#95e1d3'},
                                'resource_type': {'VM': '#8e44ad', 'Storage': '#e67e22', 'Container': '#27ae60'}
                            }

                            for group_value in df_clean[color_by].unique():
                                group_data = df_clean[df_clean[color_by] == group_value]

                                # Calculate size values if specified
                                if size_by != 'None' and size_by in available_columns:
                                    sizes = group_data[size_by]
                                    # Normalize sizes to reasonable range
                                    size_range = sizes.max() - sizes.min()
                                    if size_range > 0:
                                        normalized_sizes = ((sizes - sizes.min()) / size_range * 30) + 10
                                    else:
                                        normalized_sizes = [15] * len(sizes)
                                else:
                                    normalized_sizes = 8

                                color = color_map.get(color_by, {}).get(group_value, '#3498db')

                                scatter_fig.add_trace(go.Scatter(
                                    x=group_data[var_x],
                                    y=group_data[var_y],
                                    mode='markers',
                                    name=str(group_value),
                                    marker=dict(
                                        color=color,
                                        size=normalized_sizes,
                                        opacity=0.7,
                                        line=dict(width=1, color='white')
                                    ),
                                    hovertemplate=f'<b>{group_value}</b><br>' +
                                                 f'{var_x}: %{{x:.2f}}<br>' +
                                                 f'{var_y}: %{{y:.2f}}<br>' +
                                                 (f'{size_by}: %{{marker.size:.1f}}<br>' if size_by != 'None' else '') +
                                                 '<extra></extra>'
                                ))
                        else:
                            # Single series scatter plot
                            if size_by != 'None' and size_by in available_columns:
                                sizes = df_clean[size_by]
                                size_range = sizes.max() - sizes.min()
                                if size_range > 0:
                                    normalized_sizes = ((sizes - sizes.min()) / size_range * 30) + 10
                                else:
                                    normalized_sizes = [15] * len(sizes)
                            else:
                                normalized_sizes = 8

                            scatter_fig.add_trace(go.Scatter(
                                x=df_clean[var_x],
                                y=df_clean[var_y],
                                mode='markers',
                                name='Data Points',
                                marker=dict(
                                    color='#3498db',
                                    size=normalized_sizes,
                                    opacity=0.7
                                ),
                                hovertemplate=f'{var_x}: %{{x:.2f}}<br>' +
                                             f'{var_y}: %{{y:.2f}}<br>' +
                                             '<extra></extra>'
                            ))

                        # Add trend line
                        try:
                            x_clean = df_clean[var_x].dropna()
                            y_clean = df_clean[var_y].dropna()

                            # Align the data
                            common_idx = x_clean.index.intersection(y_clean.index)
                            x_aligned = x_clean.loc[common_idx]
                            y_aligned = y_clean.loc[common_idx]

                            if len(x_aligned) > 1:
                                z = np.polyfit(x_aligned, y_aligned, 1)
                                p = np.poly1d(z)
                                x_trend = np.linspace(x_aligned.min(), x_aligned.max(), 100)

                                scatter_fig.add_trace(go.Scatter(
                                    x=x_trend,
                                    y=p(x_trend),
                                    mode='lines',
                                    name='Trend Line',
                                    line=dict(color='red', width=2, dash='dash'),
                                    hovertemplate='Trend Line<extra></extra>'
                                ))
                        except:
                            pass  # Skip trend line if calculation fails

                        scatter_fig.update_layout(
                            title=f"Interactive Correlation: {var_x} vs {var_y}",
                            xaxis_title=var_x.replace('_', ' ').title(),
                            yaxis_title=var_y.replace('_', ' ').title(),
                            height=400,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )

                        st.plotly_chart(scatter_fig, width="stretch")

                    except Exception as scatter_error:
                        st.error(f"Error creating scatter plot: {str(scatter_error)}")

        # === ADVANCED STATISTICAL ANALYSIS ===
        if advanced_stats:
            st.divider()
            st.markdown("### 📈 Advanced Statistical Analysis")

            # Simplified version to avoid complexity
            st.markdown("**🧪 Correlation Summary**")

            # Create a simple summary table
            if 'correlation_matrix' in locals():
                summary_data = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        var1 = correlation_matrix.columns[i]
                        var2 = correlation_matrix.columns[j]
                        corr_val = correlation_matrix.iloc[i, j]

                        if not np.isnan(corr_val) and abs(corr_val) >= min_correlation:
                            summary_data.append({
                                'Variable 1': var1,
                                'Variable 2': var2,
                                'Correlation': f"{corr_val:.3f}",
                                'Strength': 'Strong' if abs(corr_val) >= 0.7 else 'Moderate' if abs(corr_val) >= 0.3 else 'Weak'
                            })

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, width="stretch")
                else:
                    st.info("No correlations above the minimum threshold found")

        # === EXPORT RESULTS ===
        if export_results:
            st.divider()
            st.markdown("### 💾 Export Analysis Results")

            if 'correlation_matrix' in locals():
                csv_data = correlation_matrix.to_csv()
                st.download_button(
                    label="📊 Download Correlation Matrix",
                    data=csv_data,
                    file_name=f"correlation_matrix_{correlation_method}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No correlation matrix available for export")

    except Exception as main_error:
        st.error(f"❌ Main correlation analysis error: {str(main_error)}")
        st.info("Please check your data and try again")


# ===== TAB 6: HOLIDAY EFFECTS =====

# ===== CORRECTED MERGED TAB 6: SEASONAL BEHAVIOR & USER ENGAGEMENT ANALYSIS =====
with tab6:
    st.subheader("🎄 Seasonal Behavior & User Engagement Intelligence")

    try:
        # Load raw data for comprehensive seasonal and engagement analysis
        raw_data = fetch_api("data/raw")

        if not raw_data:
            st.error("❌ Unable to load seasonal behavior data")
            st.stop()

        df_raw = pd.DataFrame(raw_data)
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # Enhanced date features for seasonal analysis
        df_raw['day_of_week'] = df_raw['date'].dt.day_name()
        df_raw['month'] = df_raw['date'].dt.month
        df_raw['month_name'] = df_raw['date'].dt.month_name()
        df_raw['day_of_month'] = df_raw['date'].dt.day
        df_raw['quarter'] = df_raw['date'].dt.quarter
        df_raw['week_of_year'] = df_raw['date'].dt.isocalendar().week

        # Create holiday indicator (simulated based on weekends and common dates)
        weekend_mask = df_raw['date'].dt.weekday >= 5  # Saturday=5, Sunday=6
        month_day_holidays = [
            (1, 1),   # New Year
            (7, 4),   # July 4th
            (12, 25), # Christmas
            (12, 31)  # New Year's Eve
        ]
        holiday_mask = df_raw[['month', 'day_of_month']].apply(
            lambda x: (x['month'], x['day_of_month']) in month_day_holidays, axis=1
        )
        df_raw['is_holiday'] = (weekend_mask | holiday_mask).astype(int)

        # Calculate user engagement metrics
        df_raw['cpu_per_user'] = df_raw['usage_cpu'] / (df_raw['users_active'] + 1)  # Avoid division by zero
        df_raw['storage_per_user'] = df_raw['usage_storage'] / (df_raw['users_active'] + 1)
        df_raw['user_efficiency'] = df_raw['users_active'] / (df_raw['usage_cpu'] + 1) * 100  # Users per CPU unit

        # Clean data
        df_clean = df_raw.dropna()

        # === BEHAVIORAL ANALYSIS CONTROL CENTER (UNIQUE DASHBOARD DESIGN) ===
        st.markdown("**🎯 Behavioral Analysis Control Center:**")

        # Three-tier control layout (different from all other tabs)
        tier1_container = st.container()
        tier2_container = st.container()
        tier3_container = st.container()

        with tier1_container:
            # Tier 1: Primary Analysis Selection
            primary_col1, primary_col2, primary_col3 = st.columns(3)

            with primary_col1:
                analysis_focus = st.selectbox(
                    "🎯 Analysis Focus",
                    options=['Seasonal Patterns', 'Holiday Effects', 'User Behavior', 'Engagement Efficiency'],
                    help="Choose the primary behavioral analysis focus"
                )

            with primary_col2:
                time_granularity = st.selectbox(
                    "⏰ Time Granularity",
                    options=['Daily', 'Weekly', 'Monthly', 'Quarterly'],
                    index=2,  # Default to Monthly
                    help="Select temporal analysis granularity"
                )

            with primary_col3:
                behavioral_metric = st.selectbox(
                    "📊 Key Metric",
                    options=['usage_cpu', 'usage_storage', 'users_active', 'cpu_per_user', 'user_efficiency'],
                    format_func=lambda x: {
                        'usage_cpu': '🔥 CPU Usage',
                        'usage_storage': '💾 Storage Usage',
                        'users_active': '👥 Active Users',
                        'cpu_per_user': '⚡ CPU Efficiency',
                        'user_efficiency': '🎯 User Efficiency'
                    }[x],
                    help="Primary metric for behavioral analysis"
                )

        with tier2_container:
            # Tier 2: Behavioral Filters
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

            with filter_col1:
                include_holidays = st.checkbox("🎉 Include Holidays", value=True)

            with filter_col2:
                include_weekends = st.checkbox("🛌 Include Weekends", value=True)

            with filter_col3:
                behavior_comparison = st.checkbox("⚖️ Behavior Comparison", value=True)

            with filter_col4:
                advanced_insights = st.checkbox("🧠 Advanced Insights", value=True)

        with tier3_container:
            # Tier 3: Visualization Options
            viz_col1, viz_col2, viz_col3 = st.columns(3)

            with viz_col1:
                show_annotations = st.toggle("📝 Show Annotations", value=True)

            with viz_col2:
                interactive_features = st.toggle("🎮 Interactive Features", value=True)

            with viz_col3:
                export_insights = st.toggle("💾 Export Insights", value=False)

        # Apply filters
        df_filtered = df_clean.copy()

        if not include_holidays:
            df_filtered = df_filtered[df_filtered['is_holiday'] == 0]

        if not include_weekends:
            df_filtered = df_filtered[df_filtered['date'].dt.weekday < 5]  # Monday=0 to Friday=4

        st.divider()

        # === MAIN BEHAVIORAL ANALYSIS SECTIONS ===

        if analysis_focus == 'Seasonal Patterns':
            st.markdown("### 🌅 Seasonal Pattern Analysis")

            pattern_col1, pattern_col2 = st.columns([2, 1])

            with pattern_col1:
                # Seasonal trend visualization
                st.markdown("**📈 Seasonal Trends Dashboard**")

                # Aggregate data based on time granularity
                if time_granularity == 'Daily':
                    time_group = df_filtered.groupby(df_filtered['date'].dt.date)[behavioral_metric].mean().reset_index()
                    time_group.columns = ['period', 'value']
                elif time_granularity == 'Weekly':
                    time_group = df_filtered.groupby('week_of_year')[behavioral_metric].mean().reset_index()
                    time_group.columns = ['period', 'value']
                elif time_granularity == 'Monthly':
                    time_group = df_filtered.groupby('month')[behavioral_metric].mean().reset_index()
                    time_group.columns = ['period', 'value']
                else:  # Quarterly
                    time_group = df_filtered.groupby('quarter')[behavioral_metric].mean().reset_index()
                    time_group.columns = ['period', 'value']

                # Create seasonal pattern chart
                fig_seasonal = go.Figure()

                # Main trend line
                fig_seasonal.add_trace(go.Scatter(
                    x=time_group['period'],
                    y=time_group['value'],
                    mode='lines+markers',
                    name=f'{behavioral_metric.replace("_", " ").title()}',
                    line=dict(color='#2E86AB', width=4),
                    marker=dict(size=8, color='#2E86AB'),
                    fill='tonexty' if time_granularity != 'Daily' else None,
                    fillcolor='rgba(46, 134, 171, 0.1)'
                ))

                # Add seasonal average line
                seasonal_avg = time_group['value'].mean()
                fig_seasonal.add_hline(
                    y=seasonal_avg, 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text=f"Average: {seasonal_avg:.1f}"
                )

                # Add peak and trough annotations if enabled
                if show_annotations and len(time_group) > 0:
                    peak_idx = time_group['value'].idxmax()
                    trough_idx = time_group['value'].idxmin()

                    fig_seasonal.add_annotation(
                        x=time_group.iloc[peak_idx]['period'],
                        y=time_group.iloc[peak_idx]['value'],
                        text=f"Peak: {time_group.iloc[peak_idx]['value']:.1f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="green",
                        bgcolor="lightgreen",
                        bordercolor="green"
                    )

                    fig_seasonal.add_annotation(
                        x=time_group.iloc[trough_idx]['period'],
                        y=time_group.iloc[trough_idx]['value'],
                        text=f"Low: {time_group.iloc[trough_idx]['value']:.1f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="red",
                        bgcolor="lightcoral",
                        bordercolor="red"
                    )

                fig_seasonal.update_layout(
                    title=f"{time_granularity} {behavioral_metric.replace('_', ' ').title()} Patterns",
                    xaxis_title=time_granularity,
                    yaxis_title=behavioral_metric.replace('_', ' ').title(),
                    height=400,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_seasonal, width="stretch")

            with pattern_col2:
                # Seasonal insights panel
                st.markdown("**🔍 Seasonal Insights**")

                if len(time_group) > 1:
                    # Calculate seasonal statistics
                    seasonal_variance = time_group['value'].var()
                    seasonal_range = time_group['value'].max() - time_group['value'].min()
                    seasonal_cv = (time_group['value'].std() / time_group['value'].mean()) * 100 if time_group['value'].mean() > 0 else 0

                    # Pattern classification
                    if seasonal_cv < 10:
                        pattern_type = "🟢 Stable"
                        pattern_color = "#27ae60"
                    elif seasonal_cv < 25:
                        pattern_type = "🟡 Moderate Variation"
                        pattern_color = "#f39c12"
                    else:
                        pattern_type = "🔴 High Volatility"
                        pattern_color = "#e74c3c"

                    st.markdown(f"""
                    <div style="background: {pattern_color}22; padding: 1rem; border-radius: 8px;
                               border-left: 4px solid {pattern_color};">
                        <h4 style="color: {pattern_color}; margin: 0;">Pattern Type</h4>
                        <h3 style="margin: 0.5rem 0;">{pattern_type}</h3>
                        <small>Coefficient of Variation: {seasonal_cv:.1f}%</small>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric("📊 Seasonal Range", f"{seasonal_range:.1f}")
                    st.metric("📈 Peak Value", f"{time_group['value'].max():.1f}")
                    st.metric("📉 Trough Value", f"{time_group['value'].min():.1f}")

                    # Seasonal recommendations
                    if advanced_insights:
                        st.markdown("**💡 Recommendations:**")

                        if seasonal_cv > 25:
                            st.markdown("- Consider implementing adaptive scaling during high volatility periods")
                            st.markdown("- Monitor resource allocation during peak seasons")

                        peak_period = time_group.iloc[time_group['value'].idxmax()]['period']
                        st.markdown(f"- Peak demand occurs in period {peak_period}")

                        if seasonal_range > seasonal_avg * 0.5:
                            st.markdown("- Significant seasonal variation detected - plan capacity accordingly")
                else:
                    st.info("Insufficient data for seasonal analysis")

            # Seasonal heatmap
            if len(df_filtered) > 0:
                st.markdown("**🗓️ Seasonal Calendar Heatmap**")

                # Create month-day heatmap
                calendar_pivot = df_filtered.groupby(['month', 'day_of_month'])[behavioral_metric].mean().reset_index()

                if not calendar_pivot.empty:
                    calendar_matrix = calendar_pivot.pivot(index='day_of_month', columns='month', values=behavioral_metric)

                    fig_calendar = go.Figure(data=go.Heatmap(
                        z=calendar_matrix.values,
                        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        y=calendar_matrix.index,
                        colorscale='RdYlBu_r',
                        text=calendar_matrix.values.round(1),
                        texttemplate='%{text}',
                        textfont={"size": 8},
                        hoverongaps=False,
                        colorbar=dict(
                            title=behavioral_metric.replace('_', ' ').title(),
                            x=1.02,
                            xanchor='left'
                        )
                    ))

                    fig_calendar.update_layout(
                        title=f"Seasonal Calendar - {behavioral_metric.replace('_', ' ').title()} Patterns",
                        xaxis_title="Month",
                        yaxis_title="Day of Month",
                        height=400
                    )

                    st.plotly_chart(fig_calendar, width="stretch")

        elif analysis_focus == 'Holiday Effects':
            st.markdown("### 🎉 Holiday Effects Analysis")

            holiday_col1, holiday_col2 = st.columns(2)

            with holiday_col1:
                # Holiday vs Non-Holiday comparison
                st.markdown("**📊 Holiday vs Regular Day Comparison**")

                holiday_data = df_filtered[df_filtered['is_holiday'] == 1]
                regular_data = df_filtered[df_filtered['is_holiday'] == 0]

                if not holiday_data.empty and not regular_data.empty:
                    # Comparative box plots
                    fig_comparison = go.Figure()

                    fig_comparison.add_trace(go.Box(
                        y=regular_data[behavioral_metric],
                        name='Regular Days',
                        marker_color='#4ecdc4',
                        boxmean=True
                    ))

                    fig_comparison.add_trace(go.Box(
                        y=holiday_data[behavioral_metric],
                        name='Holidays',
                        marker_color='#ff6b6b',
                        boxmean=True
                    ))

                    fig_comparison.update_layout(
                        title=f"Holiday Impact on {behavioral_metric.replace('_', ' ').title()}",
                        yaxis_title=behavioral_metric.replace('_', ' ').title(),
                        height=400,
                        showlegend=True
                    )

                    st.plotly_chart(fig_comparison, width="stretch")

                    # Holiday impact metrics
                    holiday_avg = holiday_data[behavioral_metric].mean()
                    regular_avg = regular_data[behavioral_metric].mean()
                    impact_pct = ((holiday_avg - regular_avg) / regular_avg) * 100 if regular_avg > 0 else 0

                    impact_color = "#27ae60" if impact_pct > 0 else "#e74c3c"
                    impact_direction = "📈 Increase" if impact_pct > 0 else "📉 Decrease"

                    st.markdown(f"""
                    <div style="background: {impact_color}22; padding: 1rem; border-radius: 8px;
                               border-left: 4px solid {impact_color}; text-align: center;">
                        <h4 style="color: {impact_color}; margin: 0;">Holiday Impact</h4>
                        <h2 style="margin: 0.5rem 0;">{impact_pct:+.1f}%</h2>
                        <p style="margin: 0;">{impact_direction} vs Regular Days</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Insufficient holiday or regular day data for comparison")

            with holiday_col2:
                # Day of week analysis
                st.markdown("**📅 Day of Week Behavioral Patterns**")

                dow_analysis = df_filtered.groupby('day_of_week')[behavioral_metric].agg(['mean', 'std']).reset_index()

                if not dow_analysis.empty:
                    # Reorder days of week
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    dow_analysis['day_of_week'] = pd.Categorical(dow_analysis['day_of_week'], categories=day_order, ordered=True)
                    dow_analysis = dow_analysis.sort_values('day_of_week')

                    fig_dow = go.Figure()

                    # Add bar chart with error bars
                    fig_dow.add_trace(go.Bar(
                        x=dow_analysis['day_of_week'],
                        y=dow_analysis['mean'],
                        error_y=dict(type='data', array=dow_analysis['std']),
                        marker_color=['#3498db' if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] 
                                     else '#e74c3c' for day in dow_analysis['day_of_week']],
                        hovertemplate='%{x}<br>Mean: %{y:.1f}<br>Std: %{error_y.array:.1f}<extra></extra>'
                    ))

                    # Add weekend highlighting
                    fig_dow.add_vrect(
                        x0=4.5, x1=6.5,
                        fillcolor="rgba(231, 76, 60, 0.1)",
                        layer="below",
                        line_width=0,
                        annotation_text="Weekend",
                        annotation_position="top left"
                    )

                    fig_dow.update_layout(
                        title=f"Weekly Pattern - {behavioral_metric.replace('_', ' ').title()}",
                        xaxis_title="Day of Week",
                        yaxis_title=behavioral_metric.replace('_', ' ').title(),
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig_dow, width="stretch")

        elif analysis_focus == 'User Behavior':
            st.markdown("### 👥 User Behavior Intelligence")

            behavior_col1, behavior_col2 = st.columns([3, 1])

            with behavior_col1:
                # User behavior scatter analysis
                st.markdown("**🎈 User Behavior Bubble Analysis**")

                # Create behavioral scatter plot
                fig_behavior = go.Figure()

                # Group by region for different colors
                region_colors = {'East US': '#0078d4', 'West US': '#ff6b6b', 'North Europe': '#4ecdc4', 'Southeast Asia': '#95e1d3'}

                for region in df_filtered['region'].unique():
                    region_data = df_filtered[df_filtered['region'] == region]

                    fig_behavior.add_trace(go.Scatter(
                        x=region_data['users_active'],
                        y=region_data['usage_cpu'],
                        mode='markers',
                        name=region,
                        marker=dict(
                            color=region_colors.get(region, '#3498db'),
                            size=region_data['usage_storage'] / 50,  # Scale bubble size
                            sizemin=5,
                            sizemax=30,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{region}</b><br>' +
                                     'Active Users: %{x}<br>' +
                                     'CPU Usage: %{y:.1f}%<br>' +
                                     'Storage: %{marker.size*50:.0f}GB<extra></extra>'
                    ))

                # Add efficiency trend line
                if len(df_filtered) > 10:
                    try:
                        z = np.polyfit(df_filtered['users_active'], df_filtered['usage_cpu'], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(df_filtered['users_active'].min(), df_filtered['users_active'].max(), 100)

                        fig_behavior.add_trace(go.Scatter(
                            x=x_trend,
                            y=p(x_trend),
                            mode='lines',
                            name='Efficiency Trend',
                            line=dict(color='red', width=2, dash='dash'),
                            hovertemplate='Trend Line<extra></extra>'
                        ))
                    except:
                        pass  # Skip trend line if calculation fails

                fig_behavior.update_layout(
                    title="User Behavior Analysis (Bubble size = Storage Usage)",
                    xaxis_title="Active Users",
                    yaxis_title="CPU Usage (%)",
                    height=450,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_behavior, width="stretch")

            with behavior_col2:
                # User behavior insights
                st.markdown("**🧠 Behavior Insights**")

                # Calculate user behavior metrics
                avg_users = df_filtered['users_active'].mean()
                avg_cpu_per_user = df_filtered['cpu_per_user'].mean()
                avg_efficiency = df_filtered['user_efficiency'].mean()

                # User classification
                if avg_cpu_per_user < 2:
                    user_class = "🟢 Efficient"
                    class_color = "#27ae60"
                elif avg_cpu_per_user < 5:
                    user_class = "🟡 Moderate"
                    class_color = "#f39c12"
                else:
                    user_class = "🔴 Resource Intensive"
                    class_color = "#e74c3c"

                st.markdown(f"""
                <div style="background: {class_color}22; padding: 1rem; border-radius: 8px;
                           border-left: 4px solid {class_color};">
                    <h4 style="color: {class_color}; margin: 0;">User Classification</h4>
                    <h3 style="margin: 0.5rem 0;">{user_class}</h3>
                    <small>Based on CPU usage per user</small>
                </div>
                """, unsafe_allow_html=True)

                st.metric("👥 Avg Active Users", f"{avg_users:.0f}")
                st.metric("⚡ CPU per User", f"{avg_cpu_per_user:.1f}%")
                st.metric("🎯 User Efficiency", f"{avg_efficiency:.1f}")

                # Behavior recommendations
                if advanced_insights:
                    st.markdown("**💡 Behavior Insights:**")

                    if not df_filtered.empty:
                        # Peak usage analysis
                        peak_usage = df_filtered.loc[df_filtered['usage_cpu'].idxmax()]
                        st.markdown(f"- Peak usage: {peak_usage['users_active']:.0f} users on {peak_usage['date'].strftime('%Y-%m-%d')}")

                        # Efficiency trends
                        correlation = df_filtered['users_active'].corr(df_filtered['usage_cpu'])
                        if correlation > 0.7:
                            st.markdown("- Strong positive correlation: More users = Higher CPU usage")
                        elif correlation < -0.3:
                            st.markdown("- Negative correlation: System becomes more efficient with more users")
                        else:
                            st.markdown("- Moderate correlation: User count has mixed impact on resources")

        elif analysis_focus == 'Engagement Efficiency':
            st.markdown("### 🎯 Engagement Efficiency Analysis")

            efficiency_col1, efficiency_col2 = st.columns(2)

            with efficiency_col1:
                # Efficiency heatmap by region and resource
                st.markdown("**🔥 Regional Efficiency Heatmap**")

                efficiency_pivot = df_filtered.groupby(['region', 'resource_type'])['user_efficiency'].mean().reset_index()

                if not efficiency_pivot.empty:
                    efficiency_matrix = efficiency_pivot.pivot(index='region', columns='resource_type', values='user_efficiency')

                    fig_efficiency_heatmap = go.Figure(data=go.Heatmap(
                        z=efficiency_matrix.values,
                        x=efficiency_matrix.columns,
                        y=efficiency_matrix.index,
                        colorscale='RdYlGn',
                        text=efficiency_matrix.values.round(1),
                        texttemplate='%{text}',
                        textfont={"size": 12},
                        hoverongaps=False,
                        colorbar=dict(
                            title="Efficiency Score",
                            x=1.02,
                            xanchor='left'
                        )
                    ))

                    fig_efficiency_heatmap.update_layout(
                        title="User Engagement Efficiency by Region & Resource",
                        xaxis_title="Resource Type",
                        yaxis_title="Region",
                        height=400
                    )

                    st.plotly_chart(fig_efficiency_heatmap, width="stretch")
                else:
                    st.info("Insufficient data for efficiency heatmap")

            with efficiency_col2:
                # Efficiency distribution analysis
                st.markdown("**📊 Efficiency Distribution**")

                fig_efficiency_dist = go.Figure()

                # Create violin plot for efficiency distribution
                for resource in df_filtered['resource_type'].unique():
                    resource_data = df_filtered[df_filtered['resource_type'] == resource]

                    fig_efficiency_dist.add_trace(go.Violin(
                        y=resource_data['user_efficiency'],
                        name=resource,
                        box_visible=True,
                        meanline_visible=True
                    ))

                fig_efficiency_dist.update_layout(
                    title="Efficiency Distribution by Resource Type",
                    yaxis_title="User Efficiency Score",
                    height=400,
                    showlegend=True
                )

                st.plotly_chart(fig_efficiency_dist, width="stretch")

        # === BEHAVIORAL INTELLIGENCE SUMMARY ===
        if advanced_insights:
            st.divider()
            st.markdown("### 🧠 Behavioral Intelligence Summary")

            intel_col1, intel_col2, intel_col3 = st.columns(3)

            with intel_col1:
                st.markdown("**🔍 Key Findings**")

                # Generate key insights
                insights = []

                # Seasonal insight
                if len(df_filtered) > 12:  # Need sufficient data for seasonal analysis
                    seasonal_var = df_filtered.groupby('month')[behavioral_metric].mean().var()
                    if seasonal_var > df_filtered[behavioral_metric].var() * 0.5:
                        insights.append("📅 Strong seasonal patterns detected")

                # Holiday insight
                holiday_data = df_filtered[df_filtered['is_holiday'] == 1]
                regular_data = df_filtered[df_filtered['is_holiday'] == 0]

                if not holiday_data.empty and not regular_data.empty:
                    holiday_avg = holiday_data[behavioral_metric].mean()
                    regular_avg = regular_data[behavioral_metric].mean()
                    holiday_impact = abs((holiday_avg - regular_avg) / regular_avg * 100) if regular_avg > 0 else 0
                    if holiday_impact > 15:
                        insights.append("🎉 Significant holiday effects observed")

                # User behavior insight
                user_cpu_corr = df_filtered['users_active'].corr(df_filtered['usage_cpu'])
                if user_cpu_corr > 0.7:
                    insights.append("👥 Strong user-resource correlation")

                for insight in insights[:5]:  # Show top 5 insights
                    st.markdown(f"- {insight}")

                if not insights:
                    st.info("No significant behavioral patterns detected")

            with intel_col2:
                st.markdown("**📊 Behavioral Metrics**")

                # Key behavioral metrics
                behavioral_consistency = 100 - (df_filtered[behavioral_metric].std() / df_filtered[behavioral_metric].mean() * 100) if df_filtered[behavioral_metric].mean() > 0 else 0
                user_predictability = 100 - (df_filtered['users_active'].std() / df_filtered['users_active'].mean() * 100) if df_filtered['users_active'].mean() > 0 else 0
                peak_to_trough = df_filtered[behavioral_metric].max() / df_filtered[behavioral_metric].min() if df_filtered[behavioral_metric].min() > 0 else 0

                st.metric("🎯 Behavioral Consistency", f"{behavioral_consistency:.1f}%")
                st.metric("👥 User Predictability", f"{user_predictability:.1f}%")
                st.metric("📈 Peak-to-Trough Ratio", f"{peak_to_trough:.1f}x")

            with intel_col3:
                st.markdown("**🎯 Action Items**")

                # Generate action items based on analysis
                actions = []

                if behavioral_consistency < 70:
                    actions.append("📊 Implement behavioral monitoring")

                if user_predictability < 60:
                    actions.append("👥 Analyze user engagement patterns")

                if peak_to_trough > 3:
                    actions.append("⚖️ Consider load balancing strategies")

                # Always include these general actions
                actions.extend([
                    "📈 Continue monitoring seasonal trends",
                    "🎉 Plan for holiday capacity changes"
                ])

                for action in actions[:5]:  # Show top 5 actions
                    st.markdown(f"- {action}")

        # === EXPORT FUNCTIONALITY ===
        if export_insights:
            st.divider()
            st.markdown("### 💾 Export Behavioral Analysis")

            export_col1, export_col2 = st.columns(2)

            with export_col1:
                # Export seasonal data
                if st.button("📅 Export Seasonal Analysis"):
                    seasonal_export = df_filtered.groupby(['month', 'day_of_week']).agg({
                        behavioral_metric: ['mean', 'std', 'min', 'max'],
                        'users_active': 'mean',
                        'is_holiday': 'sum'
                    }).round(2)

                    seasonal_export.columns = ['_'.join(col).strip() for col in seasonal_export.columns]
                    csv_data = seasonal_export.to_csv()

                    st.download_button(
                        label="⬇️ Download Seasonal Data",
                        data=csv_data,
                        file_name=f"seasonal_analysis_{analysis_focus.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )

            with export_col2:
                # Export behavioral insights
                if st.button("🧠 Export Behavioral Insights"):
                    insights_export = {
                        'analysis_focus': analysis_focus,
                        'time_granularity': time_granularity,
                        'behavioral_metric': behavioral_metric,
                        'total_records': len(df_filtered),
                        'date_range': f"{df_filtered['date'].min()} to {df_filtered['date'].max()}",
                        'avg_metric_value': df_filtered[behavioral_metric].mean(),
                        'metric_volatility': df_filtered[behavioral_metric].std(),
                        'user_efficiency': df_filtered['user_efficiency'].mean(),
                        'holiday_impact': 'N/A'  # Would be calculated based on analysis
                    }

                    insights_json = pd.DataFrame([insights_export]).to_json(orient='records', indent=2)

                    st.download_button(
                        label="⬇️ Download Insights Summary",
                        data=insights_json,
                        file_name=f"behavioral_insights_{analysis_focus.lower().replace(' ', '_')}.json",
                        mime="application/json"
                    )

    except Exception as main_error:
        st.error(f"❌ Behavioral analysis error: {str(main_error)}")
        st.info("Please check your data and try again")


#-------------------------------------------i have edited down code -------------------------------------------------

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
                     <div style="
                     background-color: #f0f4f8; /* Light blue-gray for contrast */
                     padding: 1rem; 
                     border-radius: 5px; 
                     margin: 0.5rem 0;
                     color: #333333; /* Dark text for readability */
                     box-shadow: 0 2px 5px rgba(0,0,0,0.1); /* subtle shadow for depth */
                   ">
                    <strong>{region}</strong><br>
                    Model: {info['model_type']}<br>
                    Status: <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
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


#-------------------------------------------i have edited up code-------------------------------------------------

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>☁️ <strong>Azure Demand Forecasting Dashboard</strong></p>
    <p>Real-time analytics and ML-powered predictions for Azure resource optimization</p>
    <p><em>Built with Streamlit • Powered by Azure Data</em></p>
</div>
""", unsafe_allow_html=True)
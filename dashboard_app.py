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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview",
    "📈 Trends", 
    "🌍 Regional",
    "⚙️ Resources",
    "🔗 Correlations",
    "🎉 Holidays",
    "🤖 Forecasting",
    "👥 Engagement"
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

# ===== TAB 4: RESOURCE TYPES =====
with tab4:
    st.subheader("⚙️ Resource Type Analysis")
    
    # Load resource data
    resource_util = fetch_api("resources/utilization")
    resource_dist = fetch_api("resources/distribution")
    resource_efficiency = fetch_api("resources/efficiency")
    
    if resource_util:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stacked area chart
            st.markdown("**Resource Utilization Over Time**")
            df_util = pd.DataFrame(resource_util)
            df_util['date'] = pd.to_datetime(df_util['date'])
            
            # Pivot for stacked area chart
            util_pivot = df_util.pivot_table(
                values='usage_cpu',
                index='date',
                columns='resource_type',
                aggfunc='mean'
            ).fillna(0)
            
            fig = go.Figure()
            
            colors = ['#0078d4', '#ff6b6b', '#4ecdc4']
            for i, resource in enumerate(util_pivot.columns):
                fig.add_trace(go.Scatter(
                    x=util_pivot.index,
                    y=util_pivot[resource],
                    mode='lines',
                    stackgroup='one',
                    name=resource,
                    fill='tonexty' if i > 0 else 'tozeroy',
                    line=dict(color=colors[i % len(colors)], width=0),
                    hovertemplate=f'<b>{resource}</b><br>Date: %{{x}}<br>Usage: %{{y:.1f}}%<extra></extra>'
                ))
            
            fig.update_layout(
                title="Resource Type Usage Over Time",
                xaxis_title="Date",
                yaxis_title="CPU Usage (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if resource_dist:
                st.markdown("**Resource Distribution**")
                df_dist = pd.DataFrame(resource_dist)
                
                fig = go.Figure(data=[go.Pie(
                    labels=df_dist['resource_type'],
                    values=df_dist['usage_cpu_mean'],
                    textinfo='label+percent',
                    marker_colors=['#0078d4', '#ff6b6b', '#4ecdc4']
                )])
                
                fig.update_layout(
                    title="Average Resource Type Distribution",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Resource efficiency
        if resource_efficiency:
            st.markdown("**Resource Efficiency Analysis**")
            df_eff = pd.DataFrame(resource_efficiency)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=df_eff['resource_type'],
                x=df_eff['cpu_per_user'],
                name='CPU per User (%/user)',
                orientation='h',
                marker_color='#0078d4'
            ))
            
            fig.update_layout(
                title="Resource Efficiency - CPU per Active User",
                xaxis_title="CPU Usage per User (%/user)",
                yaxis_title="Resource Type",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ===== TAB 5: CORRELATION ANALYSIS =====
with tab5:
    st.subheader("🔗 Correlation & External Factor Analysis")
    
    # Load correlation data
    corr_matrix = fetch_api("correlations/matrix")
    scatter_data = fetch_api("correlations/scatter")
    bubble_data = fetch_api("correlations/bubble")
    
    if corr_matrix:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Correlation Matrix**")
            df_corr = pd.DataFrame(corr_matrix)
            
            # Create pivot table for heatmap
            corr_pivot = df_corr.pivot(index='row', columns='column', values='correlation')
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_pivot.values,
                x=corr_pivot.columns,
                y=corr_pivot.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_pivot.values.round(3),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Feature Correlation Matrix",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if scatter_data:
                st.markdown("**Economic Index vs CPU Usage**")
                df_scatter = pd.DataFrame(scatter_data)
                
                fig = px.scatter(
                    df_scatter,
                    x='economic_index_avg',
                    y='usage_cpu_avg',
                    color='region',
                    size='data_points',
                    title="Economic Index vs CPU Usage by Region",
                    color_discrete_map={'East US': '#0078d4', 'West US': '#ff6b6b', 'North Europe': '#4ecdc4', 'Southeast Asia': '#95e1d3'}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
    # Multi-dimensional bubble chart with meaningful metrics
    if bubble_data:
       st.markdown("**Resource Efficiency Analysis - CPU vs Storage per User**")
       df_bubble = pd.DataFrame(bubble_data)
    
       st.write(f"📊 Analyzing {len(df_bubble)} region-resource combinations")
    
       fig = px.scatter(
        df_bubble,
        x='cpu_efficiency',  # CPU per user
        y='storage_efficiency',  # Storage per user
        size='total_utilization',  # Total resource usage
        color='region',
        symbol='resource_type',  # Different symbols for VM, Storage, Container
        hover_data=['usage_cpu', 'usage_storage', 'users_active'],
        title="Resource Efficiency: CPU/User vs Storage/User (Lower = More Efficient)",
        labels={
            'cpu_efficiency': 'CPU per User (Lower = Better)',
            'storage_efficiency': 'Storage per User (Lower = Better)'
        },
        color_discrete_map={
            'East US': '#0078d4', 
            'West US': '#ff6b6b', 
            'North Europe': '#4ecdc4', 
            'Southeast Asia': '#95e1d3'
        },
        size_max=60,
        opacity=0.8
    )
    
       fig.update_layout(height=600, showlegend=True)
       st.plotly_chart(fig, use_container_width=True)
    
       # Show raw data
       with st.expander("📋 Efficiency Data"):
           st.dataframe(df_bubble[['region', 'resource_type', 'cpu_efficiency', 'storage_efficiency', 'total_utilization']])

# ===== TAB 6: HOLIDAY EFFECTS =====
with tab6:
    st.subheader("🎉 Holiday Effects & Seasonal Analysis")
    
    # Load holiday data
    holiday_analysis = fetch_api("holiday/analysis")
    holiday_distribution = fetch_api("holiday/distribution")
    calendar_data = fetch_api("holiday/calendar")
    
    if holiday_analysis:
        st.markdown("**Holiday vs Regular Day Analysis**")
        df_holiday = pd.DataFrame(holiday_analysis)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show summary metrics
            st.markdown("**Summary Statistics**")
            holiday_stats = df_holiday[df_holiday['holiday'] == 1]
            regular_stats = df_holiday[df_holiday['holiday'] == 0]
            
            if not holiday_stats.empty and not regular_stats.empty:
                st.metric(
                    "Holiday Avg CPU",
                    f"{holiday_stats['usage_cpu_mean'].iloc[0]:.1f}%",
                    delta=f"{holiday_stats['usage_cpu_mean'].iloc[0] - regular_stats['usage_cpu_mean'].iloc[0]:+.1f}% vs regular days"
                )
                
                st.metric(
                    "Holiday Avg Storage",
                    f"{holiday_stats['usage_storage_mean'].iloc[0]:.0f} GB",
                    delta=f"{holiday_stats['usage_storage_mean'].iloc[0] - regular_stats['usage_storage_mean'].iloc[0]:+.0f}GB vs regular days"
                )
        
        with col2:
            if holiday_distribution:
                st.markdown("**Usage Distribution Comparison**")
                
                holiday_data = holiday_distribution.get('holiday_data', [])
                regular_data = holiday_distribution.get('regular_data', [])
                
                if holiday_data and regular_data:
                    fig = go.Figure()
                    
                    # Box plots for comparison
                    fig.add_trace(go.Box(
                        y=[d['usage_cpu'] for d in regular_data],
                        name='Regular Days',
                        marker_color='#4ecdc4'
                    ))
                    
                    fig.add_trace(go.Box(
                        y=[d['usage_cpu'] for d in holiday_data],
                        name='Holidays',
                        marker_color='#ff6b6b'
                    ))
                    
                    fig.update_layout(
                        title="CPU Usage Distribution - Holiday vs Regular",
                        yaxis_title="CPU Usage (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Calendar heatmap
        if calendar_data:
            st.markdown("**Calendar Heatmap - Daily Usage Patterns**")
            df_cal = pd.DataFrame(calendar_data)
            
            # Create calendar pivot
            cal_pivot = df_cal.pivot_table(
                values='usage_cpu',
                index='day',
                columns='month_name',
                aggfunc='mean'
            ).fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=cal_pivot.values,
                x=cal_pivot.columns,
                y=cal_pivot.index,
                colorscale='RdYlBu_r',
                text=cal_pivot.values.round(1),
                texttemplate='%{text}%',
                textfont={"size": 8},
                colorbar=dict(title="CPU Usage %")
            ))
            
            fig.update_layout(
                title="Daily Usage Calendar - Seasonal Pattern Analysis",
                xaxis_title="Month",
                yaxis_title="Day of Month",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

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


# ===== TAB 8: USER ENGAGEMENT =====
with tab8:
    st.subheader("👥 User Engagement & Resource Efficiency")
    
    # Load engagement data
    engagement_efficiency = fetch_api("engagement/efficiency")
    engagement_trends = fetch_api("engagement/trends")
    engagement_bubble = fetch_api("engagement/bubble")
    
    if engagement_efficiency:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Resource Efficiency Matrix**")
            df_eff = pd.DataFrame(engagement_efficiency)
            
            # Create efficiency heatmap
            eff_pivot = df_eff.pivot_table(
                values='overall_efficiency',
                index='region',
                columns='resource_type',
                aggfunc='mean'
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=eff_pivot.values,
                x=eff_pivot.columns,
                y=eff_pivot.index,
                colorscale='Greens',
                text=eff_pivot.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Efficiency Score")
            ))
            
            fig.update_layout(
                title="User Engagement Efficiency by Region & Resource",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if engagement_bubble:
                st.markdown("**User Engagement Bubble Chart**")
                df_bubble = pd.DataFrame(engagement_bubble)
                
                fig = px.scatter(
                    df_bubble,
                    x='users_active',
                    y='usage_cpu',
                    size='usage_storage',
                    color='region',
                    hover_data=['resource_type'],
                    title="Users vs CPU Usage (Bubble size = Storage)",
                    color_discrete_map={'East US': '#0078d4', 'West US': '#ff6b6b', 'North Europe': '#4ecdc4', 'Southeast Asia': '#95e1d3'}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency trends over time
        if engagement_trends:
            st.markdown("**Efficiency Trends Over Time**")
            df_trends = pd.DataFrame(engagement_trends)
            df_trends['date'] = pd.to_datetime(df_trends['date'])
            
            fig = go.Figure()
            
            # CPU per user trend
            fig.add_trace(go.Scatter(
                x=df_trends['date'],
                y=df_trends['cpu_per_user'],
                mode='lines+markers',
                name='CPU per User (%/user)',
                line=dict(color='#0078d4', width=3),
                yaxis='y'
            ))
            
            # Storage per user trend (secondary axis)
            fig.add_trace(go.Scatter(
                x=df_trends['date'],
                y=df_trends['storage_per_user'],
                mode='lines+markers',
                name='Storage per User (GB/user)',
                line=dict(color='#ff6b6b', width=3),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Resource Efficiency Trends - Usage per Active User",
                xaxis_title="Date",
                yaxis=dict(title="CPU per User (%/user)", side="left"),
                yaxis2=dict(title="Storage per User (GB/user)", side="right", overlaying="y"),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>☁️ <strong>Azure Demand Forecasting Dashboard</strong></p>
    <p>Real-time analytics and ML-powered predictions for Azure resource optimization</p>
    <p><em>Built with Streamlit • Powered by Azure Data</em></p>
</div>
""", unsafe_allow_html=True)
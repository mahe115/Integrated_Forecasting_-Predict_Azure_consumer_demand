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
        margin-bottom: 1rem;
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
                st.markdown("**CPU Usage Trend**")
                cpu_data = pd.DataFrame(sparklines['cpu_trend'])
                if not cpu_data.empty:
                    fig = px.line(cpu_data, x='date', y='usage_cpu',
                                title="CPU Usage Trend")
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with spark_col2:
                st.markdown("**Storage Usage Trend**")
                storage_data = pd.DataFrame(sparklines['storage_trend'])
                if not storage_data.empty:
                    fig = px.line(storage_data, x='date', y='usage_storage',
                                title="Storage Usage Trend", color_discrete_sequence=['#ff6b6b'])
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with spark_col3:
                st.markdown("**User Activity Trend**")
                users_data = pd.DataFrame(sparklines['users_trend'])
                if not users_data.empty:
                    fig = px.line(users_data, x='date', y='users_active',
                                title="User Activity Trend", color_discrete_sequence=['#4ecdc4'])
                    fig.update_layout(height=400, showlegend=False)
                    fig.update_xaxes(showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 2: USAGE TRENDS =====
with tab2:
    st.subheader("📈 Usage Trends Analysis")
    
    # Time series controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("**Chart Controls**")
        show_cpu = st.checkbox("Show CPU Usage", value=True)
        show_storage = st.checkbox("Show Storage Usage", value=True)
        show_users = st.checkbox("Show Active Users", value=True)
        
        chart_type = st.selectbox(
            "Chart Type",
            ["Combined View", "Regional Breakdown", "Resource Type Breakdown"]
        )
    
    with col1:
        # Prepare API parameters
        params = {}
        if start_date: params['start_date'] = start_date.isoformat()
        if end_date: params['end_date'] = end_date.isoformat()
        
        if chart_type == "Combined View":
            time_series_data = fetch_api("time-series", params)
        elif chart_type == "Regional Breakdown":
            time_series_data = fetch_api("trends/regional", params)
        else:
            time_series_data = fetch_api("trends/resource-types", params)
        
        if time_series_data:
            df_ts = pd.DataFrame(time_series_data)
            df_ts['date'] = pd.to_datetime(df_ts['date'])
            
            # Create interactive time series chart
            fig = go.Figure()
            
            if chart_type == "Combined View":
                if show_cpu:
                    fig.add_trace(go.Scatter(
                        x=df_ts['date'],
                        y=df_ts['usage_cpu'],
                        mode='lines+markers',
                        name='CPU Usage (%)',
                        line=dict(color='#0078d4', width=3),
                        hovertemplate='<b>CPU Usage</b><br>Date: %{x}<br>Value: %{y:.1f}%<extra></extra>'
                    ))
                
                if show_storage:
                    fig.add_trace(go.Scatter(
                        x=df_ts['date'],
                        y=df_ts['usage_storage'],
                        mode='lines+markers',
                        name='Storage Usage (GB)',
                        yaxis='y2',
                        line=dict(color='#ff6b6b', width=3),
                        hovertemplate='<b>Storage Usage</b><br>Date: %{x}<br>Value: %{y:.0f} GB<extra></extra>'
                    ))
                
                if show_users:
                    fig.add_trace(go.Scatter(
                        x=df_ts['date'],
                        y=df_ts['users_active'],
                        mode='lines+markers',
                        name='Active Users',
                        yaxis='y3',
                        line=dict(color='#4ecdc4', width=3),
                        hovertemplate='<b>Active Users</b><br>Date: %{x}<br>Value: %{y:.0f}<extra></extra>'
                    ))
                
                # Update layout for multi-axis
                fig.update_layout(
                    title="Azure Usage Trends - Multi-Dimensional Analysis",
                    xaxis_title="Date",
                    yaxis=dict(title="CPU Usage (%)", side="left"),
                    yaxis2=dict(title="Storage (GB)", side="right", overlaying="y"),
                    yaxis3=dict(title="Active Users", side="right", overlaying="y", position=0.95),
                    hovermode='x unified',
                    height=500
                )
            
            else:
                # Regional or Resource breakdown
                group_col = 'region' if chart_type == "Regional Breakdown" else 'resource_type'
                
                for group in df_ts[group_col].unique():
                    group_data = df_ts[df_ts[group_col] == group]
                    
                    if show_cpu:
                        fig.add_trace(go.Scatter(
                            x=group_data['date'],
                            y=group_data['usage_cpu'],
                            mode='lines+markers',
                            name=f'{group} - CPU',
                            line=dict(width=2),
                            hovertemplate=f'<b>{group} - CPU</b><br>Date: %{{x}}<br>Value: %{{y:.1f}}%<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=f"Usage Trends by {group_col.replace('_', ' ').title()}",
                    xaxis_title="Date",
                    yaxis_title="CPU Usage (%)",
                    hovermode='x unified',
                    height=500
                )
            
            # Add range selector
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="7D", step="day", stepmode="backward"),
                            dict(count=30, label="30D", step="day", stepmode="backward"),
                            dict(step="all", label="All")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Unable to load time series data")

# ===== TAB 3: REGIONAL COMPARISON =====
with tab3:
    st.subheader("🌍 Regional Performance Analysis")
    
    # Load regional data
    regional_comparison = fetch_api("regional/comparison")
    regional_heatmap = fetch_api("regional/heatmap")
    regional_distribution = fetch_api("regional/distribution")
    
    if regional_comparison:
        # Regional comparison bar chart
        df_regional = pd.DataFrame(regional_comparison)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Regional Performance Comparison**")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Average CPU Usage',
                x=df_regional['region'],
                y=df_regional['usage_cpu_mean'],
                marker_color='#0078d4'
            ))
            
            fig.add_trace(go.Bar(
                name='Peak CPU Usage',
                x=df_regional['region'],
                y=df_regional['usage_cpu_max'],
                marker_color='#ff6b6b'
            ))
            
            fig.update_layout(
                title="CPU Usage by Region",
                xaxis_title="Region",
                yaxis_title="CPU Usage (%)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if regional_distribution:
                st.markdown("**Regional Usage Distribution**")
                df_dist = pd.DataFrame(regional_distribution)
                
                fig = go.Figure(data=[go.Pie(
                    labels=df_dist['region'],
                    values=df_dist['usage_cpu'],
                    hole=0.4,
                    marker_colors=['#0078d4', '#ff6b6b', '#4ecdc4', '#95e1d3']
                )])
                
                fig.update_layout(
                    title="Total CPU Usage Distribution by Region",
                    height=400,
                    annotations=[dict(text='Regional<br>Share', x=0.5, y=0.5, font_size=16, showarrow=False)]
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Regional heatmap
        if regional_heatmap:
            st.markdown("**Regional Performance Heatmap**")
            df_heatmap = pd.DataFrame(regional_heatmap)
            
            heatmap_pivot = df_heatmap.pivot_table(
                values='usage_cpu',
                index='region',
                columns='resource_type',
                aggfunc='mean'
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='RdYlBu_r',
                text=heatmap_pivot.values.round(1),
                texttemplate='%{text}%',
                textfont={"size": 12},
                hoverongaps=False,
                colorbar=dict(title="CPU Usage %")
            ))
            
            fig.update_layout(
                title="Average CPU Usage by Region and Resource Type",
                xaxis_title="Resource Type",
                yaxis_title="Region",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

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
        
        # Multi-dimensional bubble chart
        if bubble_data:
            st.markdown("**Multi-dimensional Analysis - External Factors Impact**")
            df_bubble = pd.DataFrame(bubble_data)
            
            fig = px.scatter(
                df_bubble,
                x='economic_index',
                y='cloud_market_demand',
                size='usage_cpu',
                color='region',
                hover_data=['resource_type', 'usage_storage', 'users_active'],
                title="Economic Index vs Market Demand (Bubble size = CPU Usage)",
                color_discrete_map={'East US': '#0078d4', 'West US': '#ff6b6b', 'North Europe': '#4ecdc4', 'Southeast Asia': '#95e1d3'}
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

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
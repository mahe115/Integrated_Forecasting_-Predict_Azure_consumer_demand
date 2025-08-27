import streamlit as st
import pandas as pd
import requests

BASE_URL = "http://localhost:5000/api"

st.set_page_config(
    page_title="Azure Demand Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☁️ Azure Demand Forecasting Dashboard")

# --- Sidebar Controls --- #
st.sidebar.header("Filters & Settings")

section = st.sidebar.radio("Select Data Section", [
    "Raw Data", "Features", "Insights", "Usage Trends", "Top Regions"
])

# Common input controls
region_filter = st.sidebar.text_input("Region filter (e.g. East US)")
resource_filter = st.sidebar.text_input("Resource Type filter")
limit = st.sidebar.slider("Number of records to fetch", 10, 500, 100)

start_date = None
end_date = None
if section == "Features":
    start_date = st.sidebar.date_input("Start date (optional)")
    end_date = st.sidebar.date_input("End date (optional)")
    features = st.sidebar.text_input("Comma-separated feature names (optional)")
else:
    features = None

def fetch_json(endpoint, params=None):
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.sidebar.error(f"API request failed: {e}")
        return None

# --- Section: Raw Data --- #
if section == "Raw Data":
    st.subheader("Raw Azure Demand Data")
    params = {"limit": limit}
    if region_filter: params["region"] = region_filter
    if resource_filter: params["resource_type"] = resource_filter
    data = fetch_json("raw-data", params)
    if data:
        st.dataframe(pd.DataFrame(data))
    else:
        st.write("No data available.")

# --- Section: Features --- #
elif section == "Features":
    st.subheader("Feature-engineered Dataset Preview")
    params = {"limit": limit}
    if region_filter: params["region"] = region_filter
    if start_date: params["start_date"] = start_date.isoformat()
    if end_date: params["end_date"] = end_date.isoformat()
    if features: params["features"] = features
    data = fetch_json("features", params)
    if data:
        st.dataframe(pd.DataFrame(data))
    else:
        st.write("No data available.")

# --- Section: Insights --- #
elif section == "Insights":
    st.subheader("Comprehensive Data Insights")
    insights = fetch_json("insights")
    if insights:
        # Display KPIs as cards
        cols = st.columns(3)
        cols[0].metric("Overall Peak CPU Usage", f"{insights['peak_usage_overall']:.2f}")
        
        top_regions_series = pd.Series(insights['top_regions']).sort_values(ascending=False)
        cols[1].metric("Top Region", top_regions_series.idxmax(), f"{top_regions_series.max():.0f}")
        
        peak_months = insights['peak_usage_by_month']
        peak_months_series = pd.Series(peak_months).sort_index()
        top_month = peak_months_series.idxmax()
        cols[2].metric("Peak CPU Month", top_month, f"{peak_months_series.max():.2f}")

        st.markdown("### CPU Usage by Month (Peak)")
        df_month = pd.Series(peak_months)
        # Fix period string to standard datetime for plotting
        df_month.index = pd.to_datetime(df_month.index.to_series().astype(str).str.replace('M', '-01'))
        st.bar_chart(df_month)

        st.markdown("### Total CPU Usage by Region")
        st.bar_chart(top_regions_series)
    else:
        st.write("No insights available.")

# --- Section: Usage Trends --- #
elif section == "Usage Trends":
    st.subheader("Usage Trends Over Time")
    data = fetch_json("usage-trends")
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        usage_pivot = df.pivot(index='date', columns='region', values='usage_cpu')
        
        # Line chart for time series
        st.line_chart(usage_pivot)

        # Show raw table toggle
        if st.checkbox("Show raw usage data table"):
            st.dataframe(df)
    else:
        st.write("No usage trends data available.")

# --- Section: Top Regions --- #
elif section == "Top Regions":
    st.subheader("Top Regions by CPU Usage")
    top_n = st.slider("Number of top regions to display", 1, 20, 10)
    params = {"top_n": top_n}
    top_regions_data = fetch_json("top-regions", params)
    if top_regions_data:
        sr = pd.Series(top_regions_data).sort_values(ascending=False)
        st.bar_chart(sr)
        st.write(sr)
    else:
        st.write("No data available.")

# --- Footer --- #
st.markdown("---")
st.markdown(
    """<center>
    <small>Developed with ❤️ using Flask API and Streamlit</small>
    </center>""",
    unsafe_allow_html=True
)

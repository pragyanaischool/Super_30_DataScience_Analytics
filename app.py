import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go

# Load the data 
close_prices = pd.read_csv("close_price.csv", index_col="Date", parse_dates=True)
log_returns = pd.read_csv("log_returns.csv", index_col="Date", parse_dates=True)

# Helper function for downloading DataFrame as CSV
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

# Helper function for downloading figures as PNG
def download_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer

# Streamlit app configuration
st.set_page_config(page_title="PragyanAI - Nifty 50 Advanced Dashboard", layout="wide")
st.title("Nifty 50 Advanced Dashboard")

# Sidebar for stock selection and "Select All" button
if st.sidebar.button("Select All Stocks"):
    selected_stocks = close_prices.columns.tolist()
else:
    selected_stocks = st.sidebar.multiselect("Select Stocks for Analysis:", close_prices.columns, default=close_prices.columns)

# Option to clear selection
if st.sidebar.button("Clear Selection"):
    selected_stocks = []

all_stocks = close_prices[selected_stocks]
log_selected_stocks = log_returns[selected_stocks]

# Option to show live data
time_options = ["1 Day", "1 Month", "1 Year", "2 Years", "5 Years", "Max"]
time_period = st.sidebar.selectbox("Select Time Range for Live Nifty 50 Chart:", time_options)
show_live_chart = st.sidebar.checkbox("Show Live Nifty 50 Chart", value=False)

# Function to get live Nifty 50 data
@st.cache_data(ttl=300)
def get_live_nifty50(period="1y"):
    return yf.download("^NSEI", period=period, interval="1d")

# Live Nifty 50 Chart
if show_live_chart:
    st.subheader(f"Live Nifty 50 Chart for {time_period}")
    period_mapping = {
        "1 Day": "1d", "1 Month": "1mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"
    }
    nifty50 = get_live_nifty50(period=period_mapping[time_period])
    chart_type = st.radio("Select Chart Type:", ["Line Chart", "Candlestick Chart"], index=0, horizontal=True)

    if chart_type == "Line Chart":
        fig, ax = plt.subplots(figsize=(12, 6))
        nifty50['Close'].plot(ax=ax, color='green', linewidth=2)
        ax.set(title=f"Nifty 50 Live Chart - {time_period}", xlabel="Date", ylabel="Closing Price")
        st.pyplot(fig)
        st.download_button("Download Live Chart as PNG", data=download_figure(fig), file_name="live_chart.png", mime="image/png")
    elif chart_type == "Candlestick Chart":
        fig = go.Figure(data=[go.Candlestick(
            x=nifty50.index,
            open=nifty50['Open'],
            high=nifty50['High'],
            low=nifty50['Low'],
            close=nifty50['Close'],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        fig.update_layout(
            title=f"Nifty 50 Candlestick Chart - {time_period}",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig)

# Chart type selection
chart_type = st.radio("Select Chart Type:", ["Time Series", "Log Returns"], index=0, horizontal=True)

if chart_type == "Time Series":
    st.subheader("Time Series Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=all_stocks, ax=ax)
    ax.set(title="Time Series of Selected Stocks", xlabel="Date", ylabel="Closing Price")
    st.pyplot(fig)
    st.download_button("Download Time Series Chart as PNG", data=download_figure(fig), file_name="time_series.png", mime="image/png")
else:
    st.subheader("Log Returns Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=log_selected_stocks, ax=ax)
    ax.set(title="Log Returns of Selected Stocks", xlabel="Date", ylabel="Log Returns")
    st.pyplot(fig)
    st.download_button("Download Log Returns Chart as PNG", data=download_figure(fig), file_name="log_returns.png", mime="image/png")

# Distribution analysis (Fat tails)
st.subheader("Distribution Analysis for Fat Tails")
selected_stock_dist = st.selectbox("Select Stock for Distribution Analysis:", selected_stocks)

# Check for valid selection
if selected_stock_dist:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(log_selected_stocks[selected_stock_dist], kde=True, ax=ax)
    ax.set(title=f"Distribution of Log Returns for {selected_stock_dist}", xlabel="Log Returns", ylabel="Frequency")
    st.pyplot(fig)
    st.download_button("Download Distribution Chart as PNG", data=download_figure(fig), file_name=f"distribution_{selected_stock_dist}.png", mime="image/png")
else:
    st.warning("Please select a valid stock for distribution analysis.")

# Autocorrelation analysis
st.subheader("Autocorrelation Analysis")
selected_stock_auto = st.selectbox("Select Stock for Autocorrelation Analysis:", selected_stocks)

# Check for valid selection
if selected_stock_auto:
    fig, ax = plt.subplots(figsize=(10, 5))
    pd.plotting.autocorrelation_plot(log_selected_stocks[selected_stock_auto], ax=ax)
    ax.set(title=f"Autocorrelation of {selected_stock_auto} Log Returns")
    st.pyplot(fig)
    st.download_button("Download Autocorrelation Chart as PNG", data=download_figure(fig), file_name=f"autocorrelation_{selected_stock_auto}.png", mime="image/png")
else:
    st.warning("Please select a valid stock for autocorrelation analysis.")

# Heatmap type selection
heatmap_type = st.radio("Select Heatmap Type:", ["Correlation Matrix", "Distance Matrix"], index=0, horizontal=True)

if heatmap_type == "Correlation Matrix":
    st.subheader("Correlation Matrix Heatmap")
    corr_matrix = log_selected_stocks.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set(title="Correlation Matrix Heatmap")
    st.pyplot(fig)
    st.download_button("Download Correlation Matrix as PNG", data=download_figure(fig), file_name="correlation_matrix.png", mime="image/png")
else:
    st.subheader("Distance Matrix Heatmap")
    dist_matrix = pairwise_distances(log_selected_stocks.T, metric="euclidean")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(dist_matrix, cmap="viridis", xticklabels=selected_stocks, yticklabels=selected_stocks, ax=ax)
    ax.set(title="Distance Matrix Heatmap")
    st.pyplot(fig)
    st.download_button("Download Distance Matrix as PNG", data=download_figure(fig), file_name="distance_matrix.png", mime="image/png")

# Multidimensional Scaling (MDS)
dist_matrix = pairwise_distances(log_selected_stocks.T, metric="euclidean")
st.subheader("Multidimensional Scaling (MDS)")
if len(selected_stocks) > 1:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_coords = mds.fit_transform(dist_matrix)
    mds_df = pd.DataFrame(mds_coords, index=selected_stocks, columns=['MDS1', 'MDS2'])
    fig_mds = px.scatter(mds_df, x='MDS1', y='MDS2', text=mds_df.index, title="Multidimensional Scaling (MDS)")
    fig_mds.update_traces(textposition="top center")
    st.plotly_chart(fig_mds)
else:
    st.warning("At least 2 stocks must be selected for MDS visualization.")

# Elbow method to determine the optimal number of clusters
st.subheader("Optimal Number of Clusters (Elbow Method)")
inertia_values = []
k_range = range(1, min(11, len(selected_stocks) + 1))  # Limit range to the number of selected stocks

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    if len(selected_stocks) >= k:
        kmeans_temp.fit(dist_matrix)
        inertia_values.append(kmeans_temp.inertia_)

# Plotting the Elbow curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_range, inertia_values, marker='o', linestyle='--')
ax.set(title="Elbow Method for Optimal k", xlabel="Number of Clusters (k)", ylabel="Inertia")
st.pyplot(fig)

# Download option for Elbow method plot
st.download_button("Download Elbow Method Plot as PNG", data=download_figure(fig), file_name="elbow_method.png", mime="image/png")

# KMeans clustering
st.subheader("KMeans Clustering")
max_k = min(10, len(selected_stocks) - 1)

if len(selected_stocks) > 1:
    if max_k > 1:
        optimal_k = st.slider("Select number of clusters (k):", 2, max_k, min(4, max_k))
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(dist_matrix)
        mds_df['Cluster'] = clusters
        fig_kmeans = px.scatter(mds_df, x='MDS1', y='MDS2', color='Cluster', text=mds_df.index, title=f"KMeans Clustering with k={optimal_k}")
        fig_kmeans.update_traces(textposition="top center")
        st.plotly_chart(fig_kmeans)
    else:
        st.warning("Not enough stocks to form multiple clusters. Please select more stocks.")
else:
    st.warning("At least 2 stocks must be selected to perform clustering.")

# Download options for data
st.subheader("Download Data")
csv_data = convert_df_to_csv(all_stocks)
st.download_button(label="Download Stock Data as CSV", data=csv_data, file_name='stock_data.csv', mime='text/csv')

# Footer
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 15px 0;
            background-color: #282828;
            text-align: center;
            font-size: 16px;
            color: #fff;
            font-family: Arial, sans-serif;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
        }
        .footer strong {
            font-weight: bold;
            color: #e74c3c;;  /* Red color for "GSky" */
        }
    </style>
    <div class="footer">
        Developed by <strong>GSky</strong>
    </div>
    """, unsafe_allow_html=True)



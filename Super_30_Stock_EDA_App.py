import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from io import BytesIO

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
st.image("PragyanAI_logo.png")
st.set_page_config(page_title="PragyanAI - Nifty 50 Advanced Dashboard", layout="wide")
st.title("Nifty 50 Advanced Dashboard")

# --- Dashboard Selection and Stock Filters ---
st.sidebar.header("Dashboard Navigation")
dashboard_selection = st.sidebar.selectbox(
    "Choose a Dashboard:",
    ["Basic & Intermediate Analytics", "Advanced & Clustering", "Visualizations & Live Data"]
)

st.sidebar.header("Stock Selection")
if st.sidebar.button("Select All Stocks"):
    selected_stocks = close_prices.columns.tolist()
else:
    selected_stocks = st.sidebar.multiselect("Select Stocks for Analysis:", close_prices.columns, default=close_prices.columns)

# Option to clear selection
if st.sidebar.button("Clear Selection"):
    selected_stocks = []

all_stocks = close_prices[selected_stocks]
log_selected_stocks = log_returns[selected_stocks]

# --- Insights Box Function ---
def add_insights_box(dashboard_name, selected_stocks, log_selected_stocks):
    st.markdown("---")
    st.subheader(f"Insights for {dashboard_name}")
    
    if log_selected_stocks.empty:
        st.info("No stocks selected. Insights will appear here when you select stocks.")
        return

    insights = []
    
    if dashboard_name == "Basic & Intermediate Analytics":
        # Volatility Insight
        rolling_volatility = log_selected_stocks.rolling(window=30).std() * np.sqrt(252)
        if not rolling_volatility.empty:
            max_volatility_stock = rolling_volatility.iloc[-1].idxmax()
            max_volatility_value = rolling_volatility.iloc[-1].max() * 100
            insights.append(f"**Most Volatile Stock:** {max_volatility_stock} with a 30-day annualized volatility of **{max_volatility_value:.2f}%**.")
        
        # Fat Tails Insight
        kurtosis = log_selected_stocks.kurtosis()
        if not kurtosis.empty:
            high_kurtosis_stock = kurtosis.idxmax()
            high_kurtosis_value = kurtosis.max()
            insights.append(f"**Potential for 'Fat Tails':** The distribution analysis shows that **{high_kurtosis_stock}** has the highest kurtosis value of **{high_kurtosis_value:.2f}**, indicating a higher probability of extreme price movements (fat tails) compared to a normal distribution.")

    elif dashboard_name == "Advanced & Clustering":
        if len(selected_stocks) >= 2:
            # Correlation Insight
            corr_matrix = log_selected_stocks.corr()
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            
            # Find the index of the highest non-self correlation pair
            highest_corr_pair_index = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)].idxmax()
            stock1, stock2 = highest_corr_pair_index
            highest_corr_pair_value = corr_pairs.loc[highest_corr_pair_index]
            insights.append(f"**Highest Correlation:** The stocks **{stock1}** and **{stock2}** have the highest correlation of **{highest_corr_pair_value:.2f}**, suggesting they often move in the same direction.")

            # PCA Insight
            pca = PCA().fit(log_selected_stocks)
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            components_for_80 = np.where(cumulative_variance >= 0.80)[0][0] + 1
            insights.append(f"**Principal Components:** The first **{components_for_80}** principal components explain over **80%** of the total variance in the selected stocks' returns, indicating a few key factors drive market movements.")

    elif dashboard_name == "Visualizations & Live Data":
        # Cumulative Returns Insight
        cumulative_returns = np.exp(log_selected_stocks.cumsum())
        if not cumulative_returns.empty:
            highest_return_stock = cumulative_returns.iloc[-1].idxmax()
            highest_return_value = cumulative_returns.iloc[-1].max() * 100
            insights.append(f"**Best Performing Stock:** The stock with the highest cumulative return is **{highest_return_stock}**, with a total return of **{highest_return_value:.2f}%** over the period.")
    
    # Display insights
    for insight in insights:
        st.info(insight)

# --- Dashboard 1: Basic & Intermediate Analytics ---
def basic_dashboard():
    st.header("Basic & Intermediate Analytics")

    # Time Series and Log Returns Chart (Interactive)
    chart_type = st.radio("Select Chart Type:", ["Time Series", "Log Returns"], index=0, horizontal=True)
    if not all_stocks.empty:
        if chart_type == "Time Series":
            st.subheader("Time Series of Selected Stocks")
            fig = px.line(all_stocks, title="Time Series of Selected Stocks")
            fig.update_layout(xaxis_title="Date", yaxis_title="Closing Price")
            st.plotly_chart(fig)
        else:
            st.subheader("Log Returns of Selected Stocks")
            fig = px.line(log_selected_stocks, title="Log Returns of Selected Stocks")
            fig.update_layout(xaxis_title="Date", yaxis_title="Log Returns")
            st.plotly_chart(fig)
    else:
        st.warning("Please select at least one stock to view the charts.")
    
    st.markdown("---")
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics of Log Returns")
    if not log_selected_stocks.empty:
        stats_df = log_selected_stocks.describe().T
        st.dataframe(stats_df)
    else:
        st.warning("Please select at least one stock to view descriptive statistics.")

    st.markdown("---")

    # Rolling Volatility (Interactive)
    st.subheader("Rolling Volatility (30-day)")
    if not log_selected_stocks.empty:
        rolling_volatility = log_selected_stocks.rolling(window=30).std() * np.sqrt(252) # Annualized
        fig = px.line(rolling_volatility, title="30-day Annualized Rolling Volatility")
        fig.update_layout(xaxis_title="Date", yaxis_title="Volatility")
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least one stock to view rolling volatility.")

    st.markdown("---")

    # Autocorrelation analysis (Interactive)
    st.subheader("Autocorrelation Analysis")
    selected_stock_auto = st.selectbox("Select Stock for Autocorrelation Analysis:", selected_stocks, key='auto_select')
    if selected_stock_auto:
        # Calculate autocorrelation manually for plotting with Plotly
        autocorr_values = [log_selected_stocks[selected_stock_auto].autocorr(lag=i) for i in range(1, 51)]
        autocorr_df = pd.DataFrame({
            'Lag': range(1, 51),
            'Autocorrelation': autocorr_values
        })
        fig = px.bar(autocorr_df, x='Lag', y='Autocorrelation', title=f"Autocorrelation of {selected_stock_auto} Log Returns")
        st.plotly_chart(fig)
    else:
        st.warning("Please select a valid stock for autocorrelation analysis.")
    
    add_insights_box("Basic & Intermediate Analytics", selected_stocks, log_selected_stocks)

# --- Dashboard 2: Advanced & Clustering ---
def advanced_dashboard():
    st.header("Advanced & Clustering")
    if len(selected_stocks) < 2:
        st.warning("Please select at least two stocks for this dashboard.")
        return

    # Heatmap type selection (Interactive)
    st.subheader("Correlation & Distance Heatmaps")
    heatmap_type = st.radio("Select Heatmap Type:", ["Correlation Matrix", "Distance Matrix"], index=0, horizontal=True)
    if heatmap_type == "Correlation Matrix":
        corr_matrix = log_selected_stocks.corr()
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix Heatmap",
                        color_continuous_scale=px.colors.diverging.RdBu, zmin=-1, zmax=1)
        st.plotly_chart(fig)
    else:
        dist_matrix = pairwise_distances(log_selected_stocks.T, metric="euclidean")
        dist_df = pd.DataFrame(dist_matrix, index=selected_stocks, columns=selected_stocks)
        fig = px.imshow(dist_df, text_auto=True, title="Distance Matrix Heatmap",
                        color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

    st.markdown("---")

    # Multidimensional Scaling (MDS)
    dist_matrix = pairwise_distances(log_selected_stocks.T, metric="euclidean")
    st.subheader("Multidimensional Scaling (MDS)")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_coords = mds.fit_transform(dist_matrix)
    mds_df = pd.DataFrame(mds_coords, index=selected_stocks, columns=['MDS1', 'MDS2'])
    fig_mds = px.scatter(mds_df, x='MDS1', y='MDS2', text=mds_df.index, title="Multidimensional Scaling (MDS)")
    fig_mds.update_traces(textposition="top center")
    st.plotly_chart(fig_mds)

    st.markdown("---")

    # KMeans clustering
    st.subheader("KMeans Clustering")
    max_k = min(10, len(selected_stocks) - 1)
    if max_k >= 2:
        optimal_k = st.slider("Select number of clusters (k):", 2, max_k, min(4, max_k))
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(dist_matrix)
        mds_df['Cluster'] = clusters
        fig_kmeans = px.scatter(mds_df, x='MDS1', y='MDS2', color='Cluster', text=mds_df.index, title=f"KMeans Clustering with k={optimal_k}")
        fig_kmeans.update_traces(textposition="top center")
        st.plotly_chart(fig_kmeans)
    else:
        st.warning("Not enough stocks to form multiple clusters. Please select more stocks.")

    st.markdown("---")

    # Principal Component Analysis (PCA)
    st.subheader("Principal Component Analysis (PCA)")
    if len(selected_stocks) > 1:
        pca = PCA().fit(log_selected_stocks)
        
        # Explained Variance Plot (Interactive)
        st.write("Explained Variance Ratio:")
        fig_variance = px.bar(
            x=range(1, len(pca.explained_variance_ratio_) + 1),
            y=pca.explained_variance_ratio_,
            title='Explained Variance by Principal Component',
            labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
        )
        st.plotly_chart(fig_variance)

        # PCA Scatter Plot (Interactive)
        pca_coords = pca.transform(log_selected_stocks)
        pca_df = pd.DataFrame(pca_coords[:, :2], index=log_selected_stocks.index, columns=['PC1', 'PC2'])
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', title="PCA on Log Returns")
        st.plotly_chart(fig_pca)
    else:
        st.warning("Please select at least two stocks for PCA.")
    
    add_insights_box("Advanced & Clustering", selected_stocks, log_selected_stocks)

# --- Dashboard 3: Visualizations & Live Data ---
def visuals_dashboard():
    st.header("Visualizations & Live Data")
    
    # Live Nifty 50 Chart
    st.subheader("Live Nifty 50 Chart")
    time_options = ["1 Day", "1 Month", "1 Year", "2 Years", "5 Years", "Max"]
    time_period = st.selectbox("Select Time Range for Live Nifty 50 Chart:", time_options, key='live_chart_time')
    
    @st.cache_data(ttl=300)
    def get_live_nifty50(period="1y"):
        return yf.download("^NSEI", period=period, interval="1d")

    period_mapping = {
        "1 Day": "1d", "1 Month": "1mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"
    }
    
    try:
        nifty50 = get_live_nifty50(period=period_mapping[time_period])
        
        # Flatten the MultiIndex to a single level for easier plotting
        if isinstance(nifty50.columns, pd.MultiIndex):
            nifty50.columns = nifty50.columns.droplevel(level=0)

        # Use .dropna().empty to safely check for valid data points
        if nifty50.empty or 'Close' not in nifty50.columns or nifty50['Close'].dropna().empty:
            st.warning("No valid data available for the selected time range. Please select a different period.")
        else:
            chart_type = st.radio("Select Chart Type:", ["Area Chart", "Candlestick Chart"], index=0, horizontal=True, key='live_chart_type')
            if chart_type == "Area Chart":
                fig = px.area(nifty50, x=nifty50.index, y='Close', title=f"Nifty 50 Live Chart - {time_period}")
                st.plotly_chart(fig)
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
    except Exception as e:
        st.error(f"An error occurred while fetching live data. Please try again later.")
        st.exception(e)
        
    st.markdown("---")

    # Cumulative Returns (Interactive)
    st.subheader("Cumulative Returns")
    if not log_selected_stocks.empty:
        cumulative_returns = np.exp(log_selected_stocks.cumsum())
        fig = px.line(cumulative_returns, title="Cumulative Returns of Selected Stocks")
        fig.update_layout(yaxis_title="Cumulative Return")
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least one stock to view cumulative returns.")

    st.markdown("---")

    # Box Plots (Interactive)
    st.subheader("Distribution of Log Returns (Box Plot)")
    if not log_selected_stocks.empty:
        df_melted = log_selected_stocks.melt(var_name='Stock', value_name='Log Return')
        fig = px.box(df_melted, x='Stock', y='Log Return', title="Box Plot of Log Returns")
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least one stock for the box plot.")

    st.markdown("---")
    
    # Pair Plot
    st.subheader("Pair Plot of Log Returns")
    if len(selected_stocks) > 1 and len(selected_stocks) <= 5:
        fig_pair = sns.pairplot(log_selected_stocks)
        st.pyplot(fig_pair)
        st.download_button("Download Pair Plot as PNG", data=download_figure(fig_pair.fig), file_name="pair_plot.png", mime="image/png")
    else:
        st.warning("Pair plots are best for 2-5 stocks. Please adjust your selection.")
    
    add_insights_box("Visualizations & Live Data", selected_stocks, log_selected_stocks)

# --- Main App Logic to Display Dashboard ---
if dashboard_selection == "Basic & Intermediate Analytics":
    basic_dashboard()
elif dashboard_selection == "Advanced & Clustering":
    advanced_dashboard()
elif dashboard_selection == "Visualizations & Live Data":
    visuals_dashboard()

# --- Footer ---
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
            color: #e74c3c;
        }
    </style>
    <div class="footer">
        Developed by <strong>PragyanAI</strong>
    </div>
    """, unsafe_allow_html=True)

# Download options for data
st.sidebar.subheader("Download Data")
csv_data = convert_df_to_csv(all_stocks)
st.sidebar.download_button(label="Download Stock Data as CSV", data=csv_data, file_name='stock_data.csv', mime='text/csv')

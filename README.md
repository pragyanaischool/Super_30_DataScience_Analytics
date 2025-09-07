# Nifty50-AnalyticsHub

A comprehensive and interactive Streamlit dashboard designed for advanced analysis and visualization of Nifty 50 stock data. This app provides users with various tools to gain insights into stock market trends and correlations, making it a valuable resource for investors, financial analysts, and researchers.

## Features

- **Live Nifty 50 Chart**: View live stock price charts with options for line or candlestick visualizations for different time periods (1 Day, 1 Month, 1 Year, 2 Years, 5 Years, Max).
- **Stock Selection**: Choose individual stocks for detailed time series and log returns analysis.
- **Time Series & Log Returns Analysis**: Visualize selected stocks' historical prices and log returns.
- **Distribution Analysis**: Examine the distribution of log returns for selected stocks to identify fat tails.
- **Autocorrelation Analysis**: Analyze autocorrelation plots for selected stocks' log returns.
- **Heatmaps**: Generate correlation and distance matrix heatmaps to understand stock relationships.
- **Multidimensional Scaling (MDS)**: Visualize stock similarities through a scatter plot.
- **KMeans Clustering**: Perform clustering analysis on stocks to identify groupings based on selected criteria.
- **Data Download Options**: Download charts as PNG images and selected stock data as CSV files.

## Getting Started

### Prerequisites

Ensure you have Python 3.7 or later installed. You will also need the following Python packages:

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `yfinance`
- `scikit-learn`
- `plotly`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Nifty50-AnalyticsHub.git
   cd Nifty50-AnalyticsHub

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Run the Application
  Run the Streamlit app with the following command:
  ```bash
  streamlit run app.py
  ```

### Usage
- Navigate to the sidebar to select stocks for analysis, choose time periods, and configure chart options.
- Interact with visualizations such as time series, log returns, heatmaps, and MDS plots.
- Download data and charts for offline use or reporting.

### File Structure
- **app.py**: The main Streamlit application code.
- **close_price.csv**: CSV file containing historical closing prices for Nifty 50 stocks.
- **log_returns.csv**: CSV file with computed log returns for Nifty 50 stocks.
- **requirements.txt**: List of dependencies required to run the application.

### Deployment
You can access the deployed version of the app here: [Nifty50 Analytics Hub](---)

License
- This project is licensed under the MIT License. See the LICENSE file for details.

Contributing
- Contributions are welcome! Feel free to submit a pull request or report any issues.

Contact
- For questions or feedback, please reach out to GSky at [pragyan.ai.school@gmail.com].

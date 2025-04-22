Brent Oil and USD/KZT Exchange Rate Analysis
This project analyzes the relationship between Brent crude oil prices and the USD/KZT (US Dollar to Kazakhstani Tenge) exchange rate using time series analysis techniques.
Overview
Kazakhstan is a major oil producer, and its economy is significantly influenced by oil prices. This project examines the potential cointegration relationship between Brent crude oil prices and the USD/KZT exchange rate to understand how oil price movements might affect the Kazakhstani currency.
Features

Data Acquisition: Automatically downloads Brent crude oil and USD/KZT exchange rate data from Yahoo Finance
Stationarity Testing: Uses both ADF and KPSS tests to check if time series are stationary
Cointegration Analysis: Tests for long-term equilibrium relationships between the variables
Vector Error Correction Model (VECM): Builds a VECM when cointegration is detected
Vector Autoregression (VAR): Implements VAR modeling for differenced data
Volatility Analysis: Applies GARCH(1,1) models to analyze price volatility
Visualization: Generates multiple plots for time series, correlation, volatility, and more
Impulse Response Function (IRF): Analyzes how variables respond to shocks
Forecast Error Variance Decomposition (FEVD): Examines contribution of variables to forecast error

Requirements

Python 3.6+
Dependencies:

yfinance
pandas
numpy
matplotlib
seaborn
statsmodels
arch



Installation
bash# Clone the repository
git clone https://github.com/yourusername/brent-usdkzt-analysis.git
cd brent-usdkzt-analysis

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
Usage
Simply run the main script:
bashpython dataeng.py
The script will:

Download the most recent data for Brent oil prices and USD/KZT exchange rates
Perform stationarity and cointegration tests
Build appropriate time series models (VECM or VAR)
Generate visualizations
Analyze volatility using GARCH models
Output results to the console and save plots to the current directory

Output Files
The program generates the following output files:

brent_usdkzt_timeseries.png: Time series plots of both variables
Brent_vs_USD/KZT_correlation.png: Scatter plot showing correlation between variables
Brent_and_USD/KZT_dynamics.png: Normalized comparison of both variables
impulse_response_function.png: IRF analysis results
forecast_error_variance_decomposition.png: FEVD analysis results
Brent_volatility.png: Conditional volatility of Brent oil prices
USD/KZT_volatility.png: Conditional volatility of USD/KZT exchange rate

Methodology

Data Preprocessing: The raw data is cleaned and aligned to ensure matching dates.
Stationarity Testing:

Augmented Dickey-Fuller (ADF) test checks for unit roots
Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test verifies stationarity


Cointegration Analysis: Engle-Granger test checks for long-term equilibrium relationships
Model Selection:

If variables are cointegrated, a VECM model is built
If not, a VAR model is built on the differenced data


Impulse Response Analysis: Examines how shocks propagate through the system
Volatility Modeling: GARCH models capture and analyze conditional variance

Interpretation
The results can help understand:

The long-term relationship between oil prices and the Kazakhstani tenge
How shocks in oil prices affect the exchange rate and vice versa
Periods of high volatility in both markets
The correlation and direction of influence between the variables

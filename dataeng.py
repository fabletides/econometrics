import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def download_brent_data(start_date, end_date):
    try:
        
        brent = yf.download("BZ=F", start=start_date, end=end_date)
        
        if not brent.empty:
            print(f"Brent data successfully downloaded: {len(brent)} rows")
            return brent["Close"]
        else:
            print("Brent data not found")
            return None
    except Exception as e:
        print(f"Error downloading Brent data: {e}")
        return None




def download_usdkzt_data(start_date, end_date):
    try:
        
        usdkzt = yf.download("KZT=X", start=start_date, end=end_date, interval="1d")
        
        if not usdkzt.empty and len(usdkzt) > 10:  
            print(f"USD/KZT data successfully downloaded: {len(usdkzt)} rows")
            return usdkzt["Close"]
        else:
            print("USD/KZT data not found via Yahoo Finance")
            
            return None
    except Exception as e:
        print(f"Error downloading USD/KZT data: {e}")
        return None


def check_stationarity(series, name):
    print(f"\n--- Checking stationarity for {name} ---")
    
    
    adf_result = adfuller(series)
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value:.4f}")
    
    
    if adf_result[1] < 0.05:
        print("ADF Result: Time series is stationary (reject null hypothesis)")
    else:
        print("ADF Result: Time series is non-stationary (fail to reject null hypothesis)")
    
    
    kpss_result = kpss(series)
    print(f"\nKPSS Statistic: {kpss_result[0]:.4f}")
    print(f"p-value: {kpss_result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"\t{key}: {value:.4f}")
    
    
    if kpss_result[1] < 0.05:
        print("KPSS Result: Time series is non-stationary (reject null hypothesis)")
    else:
        print("KPSS Result: Time series is stationary (fail to reject null hypothesis)")
    
    return adf_result[1] < 0.05, kpss_result[1] >= 0.05


def check_cointegration(series1, series2, name1, name2):
    print(f"\n--- Cointegration test between {name1} and {name2} ---")
    
    
    score, p_value, _ = coint(series1, series2)
    print(f"Engle-Granger statistic: {score:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"✅ {name1} and {name2} are cointegrated (p < 0.05)")
        is_cointegrated = True
    else:
        print(f"❌ {name1} and {name2} are not cointegrated (p >= 0.05)")
        is_cointegrated = False
    
    return is_cointegrated, p_value


def build_var_model(data, max_lags=10):
    print("\n--- Building VAR model ---")
    
    
    model = VAR(data)
    lag_order_results = {}
    for i in range(1, max_lags + 1):
        result = model.fit(i)
        lag_order_results[i] = result.aic
    
    best_lag = min(lag_order_results, key=lag_order_results.get)
    print(f"Optimal number of lags according to AIC: {best_lag}")
    
    
    var_model = model.fit(best_lag)
    print(var_model.summary())
    
    return var_model, best_lag


def build_vecm_model(data, cointegration_rank=1, max_lags=10):
    print("\n--- Building VECM model ---")
    
    
    model = VAR(data)
    lag_order_results = {}
    for i in range(1, max_lags + 1):
        result = model.fit(i)
        lag_order_results[i] = result.aic
    
    best_lag = min(lag_order_results, key=lag_order_results.get)
    print(f"Optimal number of lags according to AIC: {best_lag}")
    
    
    vecm_model = VECM(data, k_ar_diff=best_lag, coint_rank=cointegration_rank, deterministic="ci")
    vecm_results = vecm_model.fit()
    print(vecm_results.summary())
    
    return vecm_results, best_lag


def analyze_volatility(series, name):
    print(f"\n--- Volatility analysis for {name} using GARCH model ---")
    
    
    returns = 100 * series.pct_change().dropna()
    
    
    garch_model = arch_model(returns, vol="Garch", p=1, q=1)
    garch_result = garch_model.fit(disp="off")
    print(garch_result.summary())
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(garch_result.conditional_volatility, color='red')
    plt.title(f'Conditional Volatility of {name}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}_volatility.png")
    
    return garch_result


def visualize_correlation(series1, series2, name1, name2):
    
    df = pd.DataFrame({name1: series1, name2: series2})
    
    
    correlation = df[name1].corr(df[name2])
    print(f"\nPearson correlation coefficient between {name1} and {name2}: {correlation:.4f}")
    
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df[name1], df[name2], alpha=0.5)
    plt.title(f'Scatter Plot: {name1} vs {name2}\nCorrelation: {correlation:.4f}')
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.grid(True)
    
    
    z = np.polyfit(df[name1], df[name2], 1)
    p = np.poly1d(z)
    plt.plot(df[name1], p(df[name1]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{name1}_vs_{name2}_correlation.png")
    
    
    plt.figure(figsize=(12, 8))
    
    
    s1_norm = (series1 - series1.mean()) / series1.std()
    s2_norm = (series2 - series2.mean()) / series2.std()
    
    plt.plot(s1_norm, label=f'{name1} (normalized)')
    plt.plot(s2_norm, label=f'{name2} (normalized)')
    plt.title(f'Comparing dynamics of {name1} and {name2}')
    plt.xlabel('Date')
    plt.ylabel('Normalized value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name1}_and_{name2}_dynamics.png")
    
    return correlation


def analyze_irf(model, periods=20):
    print("\n--- Impulse Response Function (IRF) Analysis ---")
    
    
    irf = model.irf(periods)
    irf.plot(orth=False)
    plt.tight_layout()
    plt.savefig("impulse_response_function.png")
    
    return irf


def analyze_fevd(model, periods=20):
    print("\n--- Forecast Error Variance Decomposition (FEVD) Analysis ---")
    
    
    fevd = model.fevd(periods)
    fevd.plot()
    plt.tight_layout()
    plt.savefig("forecast_error_variance_decomposition.png")
    
    return fevd


def main():
    
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print("=== Starting cointegration analysis between Brent oil and USD/KZT ===")
    print(f"Analysis period: from {start_date} to {end_date}")
    
    
    brent_price = download_brent_data(start_date, end_date)
    usdkzt_rate = download_usdkzt_data(start_date, end_date)
    
    if brent_price is None or usdkzt_rate is None:
        print("Failed to load the necessary data. Terminating analysis.")
        return
    
    
    data = pd.DataFrame({'Brent': brent_price, 'USD/KZT': usdkzt_rate})
    data = data.dropna()
    
    print(f"\n{len(data)} days of data available after processing")
    print(data.head())
    
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Brent'], color='blue')
    plt.title('Brent Oil Price')
    plt.ylabel('USD')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['USD/KZT'], color='red')
    plt.title('USD/KZT Exchange Rate')
    plt.ylabel('KZT per 1 USD')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("brent_usdkzt_timeseries.png")
    
    
    correlation = visualize_correlation(data['Brent'], data['USD/KZT'], 'Brent', 'USD/KZT')
    
    
    brent_stationary_adf, brent_stationary_kpss = check_stationarity(data['Brent'], 'Brent')
    usdkzt_stationary_adf, usdkzt_stationary_kpss = check_stationarity(data['USD/KZT'], 'USD/KZT')
    
    
    if not (brent_stationary_adf and brent_stationary_kpss) or not (usdkzt_stationary_adf and usdkzt_stationary_kpss):
        is_cointegrated, p_value = check_cointegration(data['Brent'], data['USD/KZT'], 'Brent', 'USD/KZT')
        
        if is_cointegrated:
            
            vecm_results, best_lag = build_vecm_model(data, cointegration_rank=1)
            
            
            
            data_diff = data.diff().dropna()
            var_model, var_best_lag = build_var_model(data_diff)
            
            
            analyze_irf(var_model)
            analyze_fevd(var_model)
        else:
            
            
            data_diff = data.diff().dropna()
            var_model, var_best_lag = build_var_model(data_diff)
            
            
            analyze_irf(var_model)
            analyze_fevd(var_model)
    
    
    brent_garch = analyze_volatility(data['Brent'], 'Brent')
    usdkzt_garch = analyze_volatility(data['USD/KZT'], 'USD/KZT')
    
    print("\n=== Analysis complete ===")

if __name__ == "__main__":
    main()
"""
=============================================================================
Huashu Cup 2026 Problem B - Question 3 Q313
Exponential Smoothing Prediction (Pure NumPy Implementation)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q313: Exponential Smoothing Prediction")
print("=" * 70)
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
# Data Loading
df = pd.read_csv('panel_data_38indicators.csv')
countries = df['Country'].unique()
years = sorted(df['Year'].unique())
indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
pred_years = list(range(2026, 2036))
n_pred = len(pred_years)

print(f"Countries: {len(countries)}, Indicators: {len(indicator_cols)}")
print(f"Prediction: {pred_years[0]}-{pred_years[-1]}")

# =============================================================================
# Exponential Smoothing Models
# =============================================================================

def simple_exp_smoothing(series, alpha=None, n_pred=10):
    """
    Simple Exponential Smoothing (SES)
    Formula: S_t = alpha * Y_t + (1-alpha) * S_{t-1}
    """
    series = np.array(series).flatten()
    n = len(series)
    
    # Optimize alpha if not provided
    if alpha is None:
        best_alpha, best_sse = 0.5, np.inf
        for a in np.arange(0.1, 1.0, 0.1):
            S = np.zeros(n)
            S[0] = series[0]
            for t in range(1, n):
                S[t] = a * series[t] + (1 - a) * S[t-1]
            sse = np.sum((series - S) ** 2)
            if sse < best_sse:
                best_alpha, best_sse = a, sse
        alpha = best_alpha
    
    # Fit
    S = np.zeros(n)
    S[0] = series[0]
    for t in range(1, n):
        S[t] = alpha * series[t] + (1 - alpha) * S[t-1]
    
    # Forecast (constant)
    forecast = np.full(n_pred, S[-1])
    
    return forecast, S, alpha

def holt_linear(series, alpha=None, beta=None, n_pred=10, damped=False, phi=0.9):
    """
    Holt's Linear Trend Method (Double Exponential Smoothing)
    Level: L_t = alpha * Y_t + (1-alpha) * (L_{t-1} + T_{t-1})
    Trend: T_t = beta * (L_t - L_{t-1}) + (1-beta) * T_{t-1}
    """
    series = np.array(series).flatten()
    n = len(series)
    
    # Initialize
    L = np.zeros(n)
    T = np.zeros(n)
    L[0] = series[0]
    T[0] = series[1] - series[0] if n > 1 else 0
    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    # Optimize parameters if not provided
    if alpha is None or beta is None:
        best_alpha, best_beta, best_sse = 0.5, 0.5, np.inf
        for a in np.arange(0.1, 1.0, 0.2):
            for b in np.arange(0.1, 1.0, 0.2):
                L_temp, T_temp = np.zeros(n), np.zeros(n)
                L_temp[0], T_temp[0] = series[0], series[1] - series[0] if n > 1 else 0
                
                for t in range(1, n):
                    L_temp[t] = a * series[t] + (1 - a) * (L_temp[t-1] + T_temp[t-1])
                    T_temp[t] = b * (L_temp[t] - L_temp[t-1]) + (1 - b) * T_temp[t-1]
                
                fitted = L_temp + T_temp
                sse = np.sum((series - fitted) ** 2)
                if sse < best_sse:
                    best_alpha, best_beta, best_sse = a, b, sse
        
        alpha, beta = best_alpha, best_beta
    
    # Fit
    for t in range(1, n):
        L[t] = alpha * series[t] + (1 - alpha) * (L[t-1] + T[t-1])
        T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
    
    # Forecast
    forecast = np.zeros(n_pred)
    if damped:
        phi_sum = 0
        for h in range(1, n_pred + 1):
            phi_sum += phi ** h
            forecast[h-1] = L[-1] + phi_sum * T[-1]
    else:
        for h in range(1, n_pred + 1):
            forecast[h-1] = L[-1] + h * T[-1]
    
    fitted = L + T
    
    return forecast, fitted, {'alpha': alpha, 'beta': beta, 'damped': damped}

def auto_exp_smoothing(series, n_pred=10):
    """
    Automatically select best exponential smoothing method
    """
    series = np.array(series).flatten()
    n = len(series)
    
    if n < 3:
        return np.full(n_pred, series[-1]), 'constant', 0
    
    if np.std(series) < 1e-10:
        return np.full(n_pred, series[-1]), 'constant', 0
    
    results = {}
    
    # 1. Simple Exponential Smoothing
    try:
        forecast, fitted, alpha = simple_exp_smoothing(series, n_pred=n_pred)
        mape = np.mean(np.abs((series - fitted) / (series + 1e-10))) * 100
        results['SES'] = {'forecast': forecast, 'mape': mape, 'params': {'alpha': alpha}}
    except:
        pass
    
    # 2. Holt's Linear
    try:
        forecast, fitted, params = holt_linear(series, n_pred=n_pred, damped=False)
        mape = np.mean(np.abs((series - fitted) / (series + 1e-10))) * 100
        results['Holt'] = {'forecast': forecast, 'mape': mape, 'params': params}
    except:
        pass
    
    # 3. Holt's Damped
    try:
        forecast, fitted, params = holt_linear(series, n_pred=n_pred, damped=True)
        mape = np.mean(np.abs((series - fitted) / (series + 1e-10))) * 100
        results['Holt_Damped'] = {'forecast': forecast, 'mape': mape, 'params': params}
    except:
        pass
    
    if not results:
        return np.full(n_pred, series[-1]), 'fallback', np.nan
    
    # Select best
    best_method = min(results.keys(), key=lambda x: results[x]['mape'])
    best = results[best_method]
    
    return best['forecast'], best_method, best['mape']

# =============================================================================
# Predictions
# =============================================================================
print("\n--- Predicting ---")

predictions = {}
model_info = {}

for country in countries:
    print(f"Processing: {country}")
    country_data = df[df['Country'] == country].sort_values('Year')
    predictions[country] = {}
    model_info[country] = {}
    
    for indicator in indicator_cols:
        x = country_data[indicator].values
        
        forecast, method, mape = auto_exp_smoothing(x, n_pred=n_pred)
        
        # Ensure non-negative
        forecast = np.maximum(forecast, 0)
        
        predictions[country][indicator] = forecast
        model_info[country][indicator] = {'method': method, 'mape': mape}

print("\nPrediction complete!")
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
# =============================================================================
# Create DataFrame
# =============================================================================
print("\n--- Creating DataFrames ---")

all_data = []
for _, row in df.iterrows():
    all_data.append(row.to_dict())

for country in countries:
    for i, year in enumerate(pred_years):
        row_dict = {'Country': country, 'Year': year}
        for indicator in indicator_cols:
            row_dict[indicator] = predictions[country][indicator][i]
        all_data.append(row_dict)

df_full = pd.DataFrame(all_data)
df_full = df_full.sort_values(['Country', 'Year']).reset_index(drop=True)
df_pred = df_full[df_full['Year'] >= 2026].copy()

print(f"Full data shape: {df_full.shape}")
print(f"Prediction data shape: {df_pred.shape}")

# =============================================================================
# Model Summary
# =============================================================================
print("\n--- Model Summary ---")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
model_summary = []
for country in countries:
    for indicator in indicator_cols:
        info = model_info[country][indicator]
        model_summary.append({
            'Country': country,
            'Indicator': indicator,
            'Method': info['method'],
            'MAPE': info['mape']
        })

df_model = pd.DataFrame(model_summary)

print("\nMethod Distribution:")
print(df_model['Method'].value_counts())

print(f"\nAverage MAPE: {df_model['MAPE'].mean():.2f}%")
print(f"Median MAPE: {df_model['MAPE'].median():.2f}%")

# =============================================================================
# Visualization
# =============================================================================
print("\n--- Generating Visualizations ---")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sample_indicators = indicator_cols[:6]
sample_countries = list(countries)[:2]

for idx, indicator in enumerate(sample_indicators):
    ax = axes[idx // 3, idx % 3]
    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    for country in sample_countries:
        hist_data = df[df['Country'] == country].sort_values('Year')
        ax.plot(hist_data['Year'], hist_data[indicator], 'o-', label=f'{country[:4]}(H)')
        
        pred_data = df_pred[df_pred['Country'] == country].sort_values('Year')
        ax.plot(pred_data['Year'], pred_data[indicator], 's--', label=f'{country[:4]}(P)')
    
    ax.axvline(x=2025.5, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Year')
    short_name = indicator[:18] + '...' if len(indicator) > 18 else indicator
    ax.set_title(short_name, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.suptitle('Q313: Exponential Smoothing Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q313_exp_smoothing_predictions.png', bbox_inches='tight')
plt.close()
print("Saved: Q313_exp_smoothing_predictions.png")

# =============================================================================
# Save Results
# =============================================================================
print("\n--- Saving Results ---")

df_pred.to_csv('Q313_exp_smoothing_predictions.csv', index=False)
print("Saved: Q313_exp_smoothing_predictions.csv")

df_full.to_csv('Q313_exp_smoothing_full_data.csv', index=False)
print("Saved: Q313_exp_smoothing_full_data.csv")

df_model.to_csv('Q313_exp_smoothing_model_info.csv', index=False)
print("Saved: Q313_exp_smoothing_model_info.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q313 EXPONENTIAL SMOOTHING SUMMARY")
print("=" * 70)
print(f"""
Models:
  1. SES: Simple Exponential Smoothing
  2. Holt: Holt's Linear Trend
  3. Holt_Damped: Holt's Damped Trend

Data:
  - Countries: {len(countries)}
  - Indicators: {len(indicator_cols)}
  - Prediction: 2026-2035

Method Distribution:
{df_model['Method'].value_counts().to_string()}

Performance:
  - Average MAPE: {df_model['MAPE'].mean():.2f}%
  - Median MAPE: {df_model['MAPE'].median():.2f}%
""")
print("=" * 70)

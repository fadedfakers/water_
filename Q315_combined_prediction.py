"""
=============================================================================
Huashu Cup 2026 Problem B - Question 3 Q315
Combined Prediction Model: GM(1,1) + ARIMA + Exp.Smoothing + Regression
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q315: Combined Prediction Model")
print("GM(1,1) + ARIMA + Exponential Smoothing + Regression")
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
# Individual Models
# =============================================================================

# 1. GM(1,1) Model
def gm11_predict(series, n_pred):
    """Grey Model GM(1,1)"""
    x = np.array(series).flatten()
    n = len(x)
    
    if n < 4 or np.std(x) < 1e-10:
        return np.full(n_pred, x[-1]), np.nan
    
    # Shift to positive
    shift = 0
    if np.any(x <= 0):
        shift = -x.min() + 1
        x = x + shift
    
    # AGO
    x1 = np.cumsum(x)
    z1 = 0.5 * (x1[:-1] + x1[1:])
    
    # Least squares
    B = np.column_stack([-z1, np.ones(n-1)])
    Y = x[1:].reshape(-1, 1)
    
    try:
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a, b = params[0, 0], params[1, 0]
    except:
        return np.full(n_pred, x[-1] - shift), np.nan
    
    # Predict
    n_total = n + n_pred
    x1_pred = np.zeros(n_total)
    x1_pred[0] = x[0]
    for k in range(1, n_total):
        x1_pred[k] = (x[0] - b/a) * np.exp(-a * k) + b/a
    
    x0_pred = np.zeros(n_total)
    x0_pred[0] = x1_pred[0]
    x0_pred[1:] = np.diff(x1_pred)
    
    fitted = x0_pred[:n] - shift
    forecast = x0_pred[-n_pred:] - shift
    
    # MAPE
    original = np.array(series)
    mape = np.mean(np.abs((original - fitted) / (original + 1e-10))) * 100
    
    return forecast, mape

# 2. ARIMA-like Model
def arima_predict(series, n_pred):
    """Simplified AR model with differencing"""
    x = np.array(series).flatten()
    n = len(x)
    
    if n < 4 or np.std(x) < 1e-10:
        return np.full(n_pred, x[-1]), np.nan
    
    # Difference
    diff = np.diff(x)
    
    # AR(1) on differenced series
    if len(diff) >= 2:
        X = diff[:-1].reshape(-1, 1)
        y = diff[1:]
        try:
            model = LinearRegression()
            model.fit(X, y)
            ar_coef = model.coef_[0]
            intercept = model.intercept_
        except:
            ar_coef, intercept = 0, np.mean(diff)
    else:
        ar_coef, intercept = 0, np.mean(diff)
    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    # Forecast differenced
    diff_pred = [diff[-1]]
    for _ in range(n_pred):
        next_diff = intercept + ar_coef * diff_pred[-1]
        diff_pred.append(next_diff)
    
    # Inverse difference
    forecast = x[-1] + np.cumsum(diff_pred[1:])
    
    # Fitted values for MAPE
    fitted_diff = [intercept + ar_coef * diff[i] if i > 0 else diff[0] for i in range(len(diff))]
    fitted = x[0] + np.cumsum([x[1] - x[0]] + fitted_diff[:-1])
    fitted = np.concatenate([[x[0]], fitted])[:n]
    
    mape = np.mean(np.abs((x - fitted) / (x + 1e-10))) * 100
    
    return forecast, mape

# 3. Exponential Smoothing (Holt)
def exp_smoothing_predict(series, n_pred):
    """Holt's Linear Exponential Smoothing"""
    x = np.array(series).flatten()
    n = len(x)
    
    if n < 3 or np.std(x) < 1e-10:
        return np.full(n_pred, x[-1]), np.nan
    
    # Optimize alpha and beta
    best_alpha, best_beta, best_sse = 0.5, 0.5, np.inf
    
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        for beta in [0.2, 0.4, 0.6]:
            L, T = np.zeros(n), np.zeros(n)
            L[0] = x[0]
            T[0] = x[1] - x[0] if n > 1 else 0
            
            for t in range(1, n):
                L[t] = alpha * x[t] + (1 - alpha) * (L[t-1] + T[t-1])
                T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
            
            sse = np.sum((x - (L + T)) ** 2)
            if sse < best_sse:
                best_alpha, best_beta, best_sse = alpha, beta, sse
    
    # Fit with best params
    L, T = np.zeros(n), np.zeros(n)
    L[0] = x[0]
    T[0] = x[1] - x[0] if n > 1 else 0
    
    for t in range(1, n):
        L[t] = best_alpha * x[t] + (1 - best_alpha) * (L[t-1] + T[t-1])
        T[t] = best_beta * (L[t] - L[t-1]) + (1 - best_beta) * T[t-1]
    
    # Forecast
    forecast = np.array([L[-1] + (h + 1) * T[-1] for h in range(n_pred)])
    
    # MAPE
    fitted = L + T
    mape = np.mean(np.abs((x - fitted) / (x + 1e-10))) * 100
    
    return forecast, mape

# 4. Regression Model
def regression_predict(series, n_pred):
    """Polynomial Regression (auto-select degree)"""
    x = np.array(series).flatten()
    n = len(x)
    t = np.arange(n)
    t_pred = np.arange(n, n + n_pred)
    
    if n < 3 or np.std(x) < 1e-10:
        return np.full(n_pred, x[-1]), np.nan
    
    best_mape, best_forecast = np.inf, None
    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    for degree in [1, 2]:
        try:
            poly = PolynomialFeatures(degree=degree)
            T = poly.fit_transform(t.reshape(-1, 1))
            T_pred = poly.transform(t_pred.reshape(-1, 1))
            
            model = Ridge(alpha=0.1)
            model.fit(T, x)
            
            fitted = model.predict(T)
            forecast = model.predict(T_pred)
            
            mape = np.mean(np.abs((x - fitted) / (x + 1e-10))) * 100
            
            if mape < best_mape:
                best_mape = mape
                best_forecast = forecast
        except:
            continue
    
    if best_forecast is None:
        trend = (x[-1] - x[0]) / (n - 1) if n > 1 else 0
        best_forecast = x[-1] + trend * np.arange(1, n_pred + 1)
        best_mape = np.nan
    
    return best_forecast, best_mape

# =============================================================================
# Combined Prediction
# =============================================================================

def combined_predict(series, n_pred):
    """
    Combine predictions from 4 models using inverse-variance weighting
    """
    series = np.array(series).flatten()
    
    if len(series) < 4 or np.std(series) < 1e-10:
        return np.full(n_pred, series[-1]), {'weights': {}, 'mapes': {}}
    
    # Get individual predictions
    models = {
        'GM11': gm11_predict,
        'ARIMA': arima_predict,
        'ExpSmooth': exp_smoothing_predict,
        'Regression': regression_predict
    }
    
    predictions = {}
    mapes = {}
    
    for name, func in models.items():
        try:
            pred, mape = func(series, n_pred)
            if not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                # Sanity check
                if np.abs(pred).max() < np.abs(series).max() * 100:
                    predictions[name] = pred
                    mapes[name] = mape if not np.isnan(mape) else 100
        except:
            pass
    
    if not predictions:
        return np.full(n_pred, series[-1]), {'weights': {}, 'mapes': {}}
    
    # Calculate weights (inverse variance / MAPE^2)
    model_names = list(predictions.keys())
    mape_values = np.array([max(mapes[n], 0.1) for n in model_names])  # Avoid div by zero
    
    inv_var = 1 / (mape_values ** 2)
    weights = inv_var / inv_var.sum()
    
    # Combine
    pred_matrix = np.array([predictions[n] for n in model_names])
    combined = np.dot(weights, pred_matrix)
    
    weight_dict = {n: w for n, w in zip(model_names, weights)}
    
    return combined, {'weights': weight_dict, 'mapes': mapes}

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
        
        forecast, info = combined_predict(x, n_pred)
        forecast = np.maximum(forecast, 0)
        
        predictions[country][indicator] = forecast
        model_info[country][indicator] = info

print("\nCombined prediction complete!")

# =============================================================================
# Create DataFrame# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
#     # =============================================================================
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
# Weight Analysis
# =============================================================================
print("\n--- Weight Analysis ---")

weight_records = []
for country in countries:
    for indicator in indicator_cols:
        info = model_info[country][indicator]
        weights = info.get('weights', {})
        mapes = info.get('mapes', {})
        
        weight_records.append({
            'Country': country,
            'Indicator': indicator,
            'W_GM11': weights.get('GM11', 0),
            'W_ARIMA': weights.get('ARIMA', 0),
            'W_ExpSmooth': weights.get('ExpSmooth', 0),
            'W_Regression': weights.get('Regression', 0),
            'MAPE_GM11': mapes.get('GM11', np.nan),
            'MAPE_ARIMA': mapes.get('ARIMA', np.nan),
            'MAPE_ExpSmooth': mapes.get('ExpSmooth', np.nan),
            'MAPE_Regression': mapes.get('Regression', np.nan)
        })

df_weights = pd.DataFrame(weight_records)

print("\nAverage Combination Weights:")
print(f"  GM(1,1):     {df_weights['W_GM11'].mean():.4f} ({df_weights['W_GM11'].mean()*100:.1f}%)")
print(f"  ARIMA:       {df_weights['W_ARIMA'].mean():.4f} ({df_weights['W_ARIMA'].mean()*100:.1f}%)")
print(f"  ExpSmooth:   {df_weights['W_ExpSmooth'].mean():.4f} ({df_weights['W_ExpSmooth'].mean()*100:.1f}%)")
print(f"  Regression:  {df_weights['W_Regression'].mean():.4f} ({df_weights['W_Regression'].mean()*100:.1f}%)")

print("\nAverage Individual MAPEs:")
print(f"  GM(1,1):     {df_weights['MAPE_GM11'].mean():.2f}%")
print(f"  ARIMA:       {df_weights['MAPE_ARIMA'].mean():.2f}%")
print(f"  ExpSmooth:   {df_weights['MAPE_ExpSmooth'].mean():.2f}%")
print(f"  Regression:  {df_weights['MAPE_Regression'].mean():.2f}%")

# =============================================================================
# Visualization
# =============================================================================
print("\n--- Generating Visualizations ---")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sample_indicators = indicator_cols[:6]
sample_countries = list(countries)[:2]

for idx, indicator in enumerate(sample_indicators):
    ax = axes[idx // 3, idx % 3]
    
    for country in sample_countries:
        hist_data = df[df['Country'] == country].sort_values('Year')
        ax.plot(hist_data['Year'], hist_data[indicator], 'o-', 
                label=f'{country[:4]}(H)', linewidth=2)
        
        pred_data = df_pred[df_pred['Country'] == country].sort_values('Year')
        ax.plot(pred_data['Year'], pred_data[indicator], 's-', 
                label=f'{country[:4]}(P)', linewidth=2)
    
    ax.axvline(x=2025.5, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Year')
    short_name = indicator[:18] + '...' if len(indicator) > 18 else indicator
    ax.set_title(short_name, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.suptitle('Q315: Combined Prediction Results\n(GM11 + ARIMA + ExpSmooth + Regression)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q315_combined_predictions.png', bbox_inches='tight')
plt.close()
print("Saved: Q315_combined_predictions.png")

# Weight distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
avg_weights = [df_weights['W_GM11'].mean(), df_weights['W_ARIMA'].mean(),
               df_weights['W_ExpSmooth'].mean(), df_weights['W_Regression'].mean()]
model_names = ['GM(1,1)', 'ARIMA', 'ExpSmooth', 'Regression']
colors = plt.cm.Set2(np.linspace(0, 1, 4))
bars = ax1.bar(model_names, avg_weights, color=colors)
ax1.set_ylabel('Average Weight')
ax1.set_title('Average Combination Weights', fontweight='bold')
for bar, w in zip(bars, avg_weights):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{w:.3f}', ha='center', fontsize=10)

ax2 = axes[1]
avg_mapes = [df_weights['MAPE_GM11'].mean(), df_weights['MAPE_ARIMA'].mean(),
             df_weights['MAPE_ExpSmooth'].mean(), df_weights['MAPE_Regression'].mean()]
bars = ax2.bar(model_names, avg_mapes, color=colors)
ax2.set_ylabel('Average MAPE (%)')
ax2.set_title('Average MAPE by Model', fontweight='bold')
for bar, m in zip(bars, avg_mapes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{m:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('Q315_weight_analysis.png', bbox_inches='tight')
plt.close()
print("Saved: Q315_weight_analysis.png")

# =============================================================================
# Save Results
# =============================================================================
print("\n--- Saving Results ---")

df_pred.to_csv('Q315_combined_predictions.csv', index=False)
print("Saved: Q315_combined_predictions.csv")

df_full.to_csv('Q315_combined_full_data.csv', index=False)
print("Saved: Q315_combined_full_data.csv")

df_weights.to_csv('Q315_combination_weights.csv', index=False)
print("Saved: Q315_combination_weights.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q315 COMBINED PREDICTION SUMMARY")
print("=" * 70)
print(f"""
Combined Model: GM(1,1) + ARIMA + Exponential Smoothing + Regression

Weight Method: Inverse Variance (1/MAPE²)

Data:
  - Countries: {len(countries)}
  - Indicators: {len(indicator_cols)}
  - Prediction: 2026-2035

Average Weights:
  - GM(1,1):     {df_weights['W_GM11'].mean()*100:.1f}%
  - ARIMA:       {df_weights['W_ARIMA'].mean()*100:.1f}%
  - ExpSmooth:   {df_weights['W_ExpSmooth'].mean()*100:.1f}%
  - Regression:  {df_weights['W_Regression'].mean()*100:.1f}%

Output Files:
  - Q315_combined_predictions.csv
  - Q315_combined_full_data.csv
  - Q315_combination_weights.csv
""")
print("=" * 70)

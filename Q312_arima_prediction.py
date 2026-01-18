"""
=============================================================================
Huashu Cup 2026 Problem B - Question 3 Q312
ARIMA-like Time Series Prediction (Pure NumPy Implementation)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
print("=" * 70)
print("Q312: ARIMA-like Time Series Prediction")
print("=" * 70)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# Data Loading
df = pd.read_csv('panel_data_38indicators.csv')
countries = df['Country'].unique()
years = sorted(df['Year'].unique())
indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
pred_years = list(range(2026, 2036))
n_pred = len(pred_years)

print(f"Countries: {len(countries)}, Indicators: {len(indicator_cols)}")
print(f"Prediction: {pred_years[0]}-{pred_years[-1]}")

# ARIMA-like Model
class SimpleARIMA:
    def __init__(self, p=1, d=1):
        self.p, self.d = p, d
        self.ar_params = None
        self.last_vals = None
        
    def fit(self, series):
        series = np.array(series).flatten()
        self.last_vals = series[-self.d:] if self.d > 0 else []
        
        # Differencing
        diff = series.copy()
        for _ in range(self.d):
            diff = np.diff(diff)
        
        if len(diff) < self.p + 1:
            self.ar_params = np.zeros(max(1, self.p))
            self.mean = np.mean(diff) if len(diff) > 0 else 0
            return self
        
        # Fit AR model
        X = np.column_stack([diff[i:len(diff)-self.p+i] for i in range(self.p)])
        y = diff[self.p:]
        
        try:
            self.ar_params = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            self.ar_params = np.zeros(self.p)
        
        self.diff_series = diff
        self.mean = np.mean(diff)
        return self
    
    def predict(self, n_pred):
        diff = list(self.diff_series)
        
        for _ in range(n_pred):
            if self.p > 0 and len(diff) >= self.p:
                pred = np.dot(self.ar_params, diff[-self.p:][::-1])
            else:
                pred = self.mean
            diff.append(pred)
        
        result = np.array(diff[-n_pred:])
        
        # Inverse differencing
        for val in reversed(self.last_vals):
            result = np.cumsum(np.concatenate([[val], result]))[1:]
        
        return result

def arima_predict(series, n_pred):
    series = np.array(series).flatten()
    if np.std(series) < 1e-10:
        return np.full(n_pred, series[-1]), (0, 0), 0
    
    best_mape, best_pred, best_order = np.inf, None, (1, 1)
    
    for p in range(3):
        for d in range(2):
            try:
                model = SimpleARIMA(p=p, d=d)
                model.fit(series)
                pred = model.predict(n_pred)
                
                # Simple MAPE on last 2 points
                fitted = model.predict(2)
                actual = series[-2:]
                mape = np.mean(np.abs((actual - fitted[:2]) / (actual + 1e-10))) * 100
                
                if mape < best_mape:
                    best_mape, best_pred, best_order = mape, pred, (p, d)
            except:
                continue
    
    if best_pred is None:
        trend = (series[-1] - series[0]) / (len(series) - 1) if len(series) > 1 else 0
        best_pred = series[-1] + trend * np.arange(1, n_pred + 1)
    
    return best_pred, best_order, best_mape

# Predictions
print("\n--- Predicting ---")
predictions, model_info = {}, {}
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
for country in countries:
    print(f"Processing: {country}")
    country_data = df[df['Country'] == country].sort_values('Year')
    predictions[country], model_info[country] = {}, {}
    
    for indicator in indicator_cols:
        x = country_data[indicator].values
        pred, order, mape = arima_predict(x, n_pred)
        predictions[country][indicator] = np.maximum(pred, 0)
        model_info[country][indicator] = {'order': order, 'mape': mape}

# Create DataFrame
all_data = [row.to_dict() for _, row in df.iterrows()]
for country in countries:
    for i, year in enumerate(pred_years):
        row = {'Country': country, 'Year': year}
        row.update({ind: predictions[country][ind][i] for ind in indicator_cols})
        all_data.append(row)

df_full = pd.DataFrame(all_data).sort_values(['Country', 'Year']).reset_index(drop=True)
df_pred = df_full[df_full['Year'] >= 2026].copy()

# Save
df_pred.to_csv('Q312_arima_predictions.csv', index=False)
df_full.to_csv('Q312_arima_full_data.csv', index=False)

model_summary = [{'Country': c, 'Indicator': i, 'Order': str(model_info[c][i]['order']), 
                  'MAPE': model_info[c][i]['mape']} 
                 for c in countries for i in indicator_cols]
pd.DataFrame(model_summary).to_csv('Q312_arima_model_info.csv', index=False)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, indicator in enumerate(indicator_cols[:6]):
    ax = axes[idx // 3, idx % 3]
    for country in list(countries)[:2]:
        hist = df[df['Country'] == country].sort_values('Year')
        pred = df_pred[df_pred['Country'] == country].sort_values('Year')
        ax.plot(hist['Year'], hist[indicator], 'o-', label=f'{country[:4]}(H)')
        ax.plot(pred['Year'], pred[indicator], 's--', label=f'{country[:4]}(P)')
    ax.axvline(x=2025.5, color='gray', linestyle=':')
    ax.set_title(indicator[:20], fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.suptitle('Q312: ARIMA-like Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q312_arima_predictions.png', bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("Q312 Complete!")
print(f"Saved: Q312_arima_predictions.csv, Q312_arima_full_data.csv")
print("=" * 70)

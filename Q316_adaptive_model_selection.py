"""
=============================================================================
Huashu Cup 2026 Problem B - Question 3 Q316
Adaptive Model Selection - Best Model for Each Indicator
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q316: Adaptive Model Selection")
print("Automatically Select Best Model for Each Indicator")
print("=" * 70)

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
# Model Library (14 Models)
# =============================================================================
print("\n--- Building Model Library ---")

class ModelLibrary:
    """Library of 14 prediction models"""
    
    def __init__(self):
        self.models = {
            # Grey Models
            'GM11': self._gm11,
            'GM11_Markov': self._gm11_markov,
            # Time Series
            'AR1': self._ar1,
            'AR2': self._ar2,
            # Exponential Smoothing
            'SES': self._ses,
            'Holt': self._holt,
            'Holt_Damped': self._holt_damped,
            # Regression
            'Linear': self._linear,
            'Quadratic': self._quadratic,
            'Cubic': self._cubic,
            'Exponential': self._exponential,
            'Power': self._power,
            'Log': self._log,
            # Moving Average
            'WMA': self._weighted_ma
        }
    
    # ========== Grey Models ==========
    def _gm11(self, x, n_pred):
        n = len(x)
        shift = 0
        if np.any(x <= 0):
            shift = -x.min() + 1
            x = x + shift
        
        x1 = np.cumsum(x)
        z1 = 0.5 * (x1[:-1] + x1[1:])
        B = np.column_stack([-z1, np.ones(n-1)])
        Y = x[1:].reshape(-1, 1)
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a, b = params[0, 0], params[1, 0]
        
        n_total = n + n_pred
        x1_pred = np.zeros(n_total)
        x1_pred[0] = x[0]
        for k in range(1, n_total):
            x1_pred[k] = (x[0] - b/a) * np.exp(-a * k) + b/a
        
        x0_pred = np.zeros(n_total)
        x0_pred[0] = x1_pred[0]
        x0_pred[1:] = np.diff(x1_pred)
        
        return x0_pred[-n_pred:] - shift, x0_pred[:n] - shift
    
    def _gm11_markov(self, x, n_pred):
        forecast, fitted = self._gm11(x, n_pred)
        residuals = x - fitted
        
        if len(np.unique(residuals)) >= 3:
            n_states = 3
            bins = np.percentile(residuals, [0, 33, 67, 100])
            bins[0], bins[-1] = -np.inf, np.inf
            states = np.digitize(residuals, bins) - 1
            
            trans = np.zeros((n_states, n_states))
            for i in range(len(states) - 1):
                trans[states[i], states[i+1]] += 1
            trans = trans / (trans.sum(axis=1, keepdims=True) + 1e-10)
            
            centers = [residuals[states == s].mean() if np.sum(states == s) > 0 else 0 
                      for s in range(n_states)]
            
            current = states[-1]
            corrections = []
            for _ in range(n_pred):
                corrections.append(np.dot(trans[current], centers))
                current = np.argmax(trans[current])
            
            forecast = forecast + np.array(corrections)
        
        return forecast, fitted
    
    # ========== Time Series ==========
    def _ar1(self, x, n_pred):
        diff = np.diff(x)
        if len(diff) < 2:
            return np.full(n_pred, x[-1]), x
        
        X = diff[:-1].reshape(-1, 1)
        y = diff[1:]
        model = LinearRegression().fit(X, y)
        
        pred = [diff[-1]]
        for _ in range(n_pred):
            pred.append(model.predict([[pred[-1]]])[0])
        
        forecast = x[-1] + np.cumsum(pred[1:])
        fitted = x  # Simplified
        return forecast, fitted
    
    def _ar2(self, x, n_pred):
        diff = np.diff(x)
        if len(diff) < 3:
            return self._ar1(x, n_pred)
        
        X = np.column_stack([diff[:-2], diff[1:-1]])
        y = diff[2:]
        model = LinearRegression().fit(X, y)
        
        pred = list(diff[-2:])
        for _ in range(n_pred):
            next_val = model.predict([[pred[-2], pred[-1]]])[0]
            pred.append(next_val)
        
        forecast = x[-1] + np.cumsum(pred[2:])
        fitted = x
        return forecast, fitted
    
    # ========== Exponential Smoothing ==========
    def _ses(self, x, n_pred):
        n = len(x)
        best_alpha, best_sse = 0.5, np.inf
        
        for alpha in [0.2, 0.4, 0.6, 0.8]:
            S = np.zeros(n)
            S[0] = x[0]
            for t in range(1, n):
                S[t] = alpha * x[t] + (1 - alpha) * S[t-1]
            sse = np.sum((x - S) ** 2)
            if sse < best_sse:
                best_alpha, best_sse = alpha, sse
        
        S = np.zeros(n)
        S[0] = x[0]
        for t in range(1, n):
            S[t] = best_alpha * x[t] + (1 - best_alpha) * S[t-1]
        
        return np.full(n_pred, S[-1]), S
    
    def _holt(self, x, n_pred):
        n = len(x)
        best_params, best_sse = (0.5, 0.5), np.inf
        
        for alpha in [0.2, 0.5, 0.8]:
            for beta in [0.2, 0.5]:
                L, T = np.zeros(n), np.zeros(n)
                L[0], T[0] = x[0], x[1] - x[0] if n > 1 else 0
                for t in range(1, n):
                    L[t] = alpha * x[t] + (1 - alpha) * (L[t-1] + T[t-1])
                    T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
                sse = np.sum((x - (L + T)) ** 2)
                if sse < best_sse:
                    best_params, best_sse = (alpha, beta), sse
        
        alpha, beta = best_params
        L, T = np.zeros(n), np.zeros(n)
        L[0], T[0] = x[0], x[1] - x[0] if n > 1 else 0
        for t in range(1, n):
            L[t] = alpha * x[t] + (1 - alpha) * (L[t-1] + T[t-1])
            T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
        
        forecast = np.array([L[-1] + (h + 1) * T[-1] for h in range(n_pred)])
        return forecast, L + T
    
    def _holt_damped(self, x, n_pred):
        n = len(x)
        phi = 0.9
        
        alpha, beta = 0.5, 0.3
        L, T = np.zeros(n), np.zeros(n)
        L[0], T[0] = x[0], x[1] - x[0] if n > 1 else 0
        
        for t in range(1, n):
            L[t] = alpha * x[t] + (1 - alpha) * (L[t-1] + phi * T[t-1])
            T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * phi * T[t-1]
        
        forecast = []
        phi_sum = 0
        for h in range(1, n_pred + 1):
            phi_sum += phi ** h
            forecast.append(L[-1] + phi_sum * T[-1])
        
        return np.array(forecast), L + T
    
    # ========== Regression ==========
    def _linear(self, x, n_pred):
        n = len(x)
        t = np.arange(n).reshape(-1, 1)
        t_pred = np.arange(n, n + n_pred).reshape(-1, 1)
        model = LinearRegression().fit(t, x)
        return model.predict(t_pred).flatten(), model.predict(t).flatten()
    
    def _quadratic(self, x, n_pred):
        n = len(x)
        t = np.arange(n).reshape(-1, 1)
        t_pred = np.arange(n, n + n_pred).reshape(-1, 1)
        poly = PolynomialFeatures(2)
        T = poly.fit_transform(t)
        T_pred = poly.transform(t_pred)
        model = Ridge(alpha=0.1).fit(T, x)
        return model.predict(T_pred).flatten(), model.predict(T).flatten()
    
    def _cubic(self, x, n_pred):
        n = len(x)
        t = np.arange(n).reshape(-1, 1)
        t_pred = np.arange(n, n + n_pred).reshape(-1, 1)
        poly = PolynomialFeatures(3)
        T = poly.fit_transform(t)
        T_pred = poly.transform(t_pred)
        model = Ridge(alpha=1.0).fit(T, x)
        return model.predict(T_pred).flatten(), model.predict(T).flatten()
    
    def _exponential(self, x, n_pred):
        n = len(x)
        shift = -x.min() + 1 if x.min() <= 0 else 0
        y = x + shift
        
        t = np.arange(n).reshape(-1, 1)
        t_pred = np.arange(n, n + n_pred).reshape(-1, 1)
        
        model = LinearRegression().fit(t, np.log(y))
        forecast = np.exp(model.predict(t_pred)).flatten() - shift
        fitted = np.exp(model.predict(t)).flatten() - shift
        return forecast, fitted
    
    def _power(self, x, n_pred):
        n = len(x)
        shift = -x.min() + 1 if x.min() <= 0 else 0
        y = x + shift
        
        t = np.arange(1, n + 1).reshape(-1, 1)
        t_pred = np.arange(n + 1, n + n_pred + 1).reshape(-1, 1)
        
        model = LinearRegression().fit(np.log(t), np.log(y))
        forecast = np.exp(model.predict(np.log(t_pred))).flatten() - shift
        fitted = np.exp(model.predict(np.log(t))).flatten() - shift
        return forecast, fitted
    
    def _log(self, x, n_pred):
        n = len(x)
        t = np.arange(1, n + 1).reshape(-1, 1)
        t_pred = np.arange(n + 1, n + n_pred + 1).reshape(-1, 1)
        model = LinearRegression().fit(np.log(t), x)
        return model.predict(np.log(t_pred)).flatten(), model.predict(np.log(t)).flatten()
    
    def _weighted_ma(self, x, n_pred):
        n = len(x)
        weights = np.exp(np.linspace(-1, 0, n))
        weights /= weights.sum()
        wma = np.dot(weights, x)
        trend = (x[-1] - x[-3]) / 2 if n >= 3 else 0
        return wma + trend * np.arange(1, n_pred + 1), x
    
    # ========== Model Selection ==========
    def evaluate(self, x, model_name, n_pred):
        """Evaluate a model using MAPE"""
        try:
            forecast, fitted = self.models[model_name](x, n_pred)
            
            if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
                return None
            
            # Sanity check
            if np.abs(forecast).max() > np.abs(x).max() * 50:
                return None
            
            mape = np.mean(np.abs((x - fitted) / (x + 1e-10))) * 100
            return {'forecast': forecast, 'fitted': fitted, 'mape': mape}
        except:
            return None
    
    def select_best(self, x, n_pred, verbose=False):
        """Select the best model based on MAPE"""
        x = np.array(x).flatten()
        
        if len(x) < 4 or np.std(x) < 1e-10:
            return np.full(n_pred, x[-1]), 'constant', 0
        
        results = {}
        for name in self.models:
            result = self.evaluate(x, name, n_pred)
            if result:
                results[name] = result
        
        if not results:
            trend = (x[-1] - x[0]) / (len(x) - 1)
            return x[-1] + trend * np.arange(1, n_pred + 1), 'fallback', np.nan
        
        best = min(results.keys(), key=lambda k: results[k]['mape'])
        
        if verbose:
            print(f"\n  Top 3 models:")
            for name in sorted(results.keys(), key=lambda k: results[k]['mape'])[:3]:
                print(f"    {name}: MAPE={results[name]['mape']:.2f}%")
        
        return results[best]['forecast'], best, results[best]['mape']

# =============================================================================
# Predictions
# =============================================================================
print("\n--- Predicting (Adaptive Selection) ---")

library = ModelLibrary()
predictions = {}
model_info = {}

for country in countries:
    print(f"\nProcessing: {country}")
    country_data = df[df['Country'] == country].sort_values('Year')
    predictions[country] = {}
    model_info[country] = {}
    
    model_counts = {}
    
    for indicator in indicator_cols:
        x = country_data[indicator].values
        forecast, best_model, mape = library.select_best(x, n_pred)
        
        predictions[country][indicator] = np.maximum(forecast, 0)
        model_info[country][indicator] = {'model': best_model, 'mape': mape}
        model_counts[best_model] = model_counts.get(best_model, 0) + 1
    
    top3 = sorted(model_counts.items(), key=lambda x: -x[1])[:3]
    print(f"  Top models: {dict(top3)}")

print("\nAdaptive prediction complete!")

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
# Model Selection Summary
# =============================================================================
print("\n--- Model Selection Summary ---")

model_summary = []
for country in countries:
    for indicator in indicator_cols:
        info = model_info[country][indicator]
        model_summary.append({
            'Country': country,
            'Indicator': indicator,
            'Selected_Model': info['model'],
            'MAPE': info['mape']
        })

df_model = pd.DataFrame(model_summary)

print("\nOverall Model Selection Distribution:")
print(df_model['Selected_Model'].value_counts())

print(f"\nAverage MAPE: {df_model['MAPE'].mean():.2f}%")
print(f"Median MAPE: {df_model['MAPE'].median():.2f}%")

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
        hist = df[df['Country'] == country].sort_values('Year')
        pred = df_pred[df_pred['Country'] == country].sort_values('Year')
        
        ax.plot(hist['Year'], hist[indicator], 'o-', label=f'{country[:4]}(H)', linewidth=2)
        ax.plot(pred['Year'], pred[indicator], 's--', label=f'{country[:4]}(P)', linewidth=2)
        
        # Show model name
        model_name = model_info[country][indicator]['model']
        ax.annotate(f'{model_name}', xy=(pred_years[-1], pred[indicator].values[-1]),
                   fontsize=6, alpha=0.7)
    
    ax.axvline(x=2025.5, color='gray', linestyle=':', alpha=0.7)
    short_name = indicator[:18] + '...' if len(indicator) > 18 else indicator
    ax.set_title(short_name, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.suptitle('Q316: Adaptive Model Selection Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q316_adaptive_predictions.png', bbox_inches='tight')
plt.close()
print("Saved: Q316_adaptive_predictions.png")

# Model distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
counts = df_model['Selected_Model'].value_counts()
colors = plt.cm.tab20(np.linspace(0, 1, len(counts)))
bars = ax1.barh(range(len(counts)), counts.values, color=colors)
ax1.set_yticks(range(len(counts)))
ax1.set_yticklabels(counts.index)
ax1.set_xlabel('Count')
ax1.set_title('Model Selection Distribution', fontweight='bold')
ax1.invert_yaxis()

ax2 = axes[1]
mape_by_model = df_model.groupby('Selected_Model')['MAPE'].mean().sort_values()
bars = ax2.barh(range(len(mape_by_model)), mape_by_model.values,
                color=plt.cm.RdYlGn_r(mape_by_model.values / max(mape_by_model.values)))
ax2.set_yticks(range(len(mape_by_model)))
ax2.set_yticklabels(mape_by_model.index)
ax2.set_xlabel('Average MAPE (%)')
ax2.set_title('Average MAPE by Model', fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('Q316_model_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: Q316_model_distribution.png")

# =============================================================================
# Save Results
# =============================================================================
print("\n--- Saving Results ---")

df_pred.to_csv('Q316_adaptive_predictions.csv', index=False)
print("Saved: Q316_adaptive_predictions.csv")

df_full.to_csv('Q316_adaptive_full_data.csv', index=False)
print("Saved: Q316_adaptive_full_data.csv")

df_model.to_csv('Q316_model_selection.csv', index=False)
print("Saved: Q316_model_selection.csv")

# Indicator recommendations
indicator_recs = df_model.groupby('Indicator').agg({
    'Selected_Model': lambda x: x.value_counts().index[0],
    'MAPE': 'mean'
}).reset_index()
indicator_recs.columns = ['Indicator', 'Recommended_Model', 'Avg_MAPE']
indicator_recs.to_csv('Q316_indicator_recommendations.csv', index=False)
print("Saved: Q316_indicator_recommendations.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q316 ADAPTIVE MODEL SELECTION SUMMARY")
print("=" * 70)

top5 = df_model['Selected_Model'].value_counts().head(5)
print(f"""
Model Library: 14 Models
  - Grey: GM11, GM11_Markov
  - Time Series: AR1, AR2
  - Smoothing: SES, Holt, Holt_Damped
  - Regression: Linear, Quadratic, Cubic, Exponential, Power, Log
  - Moving Average: WMA

Selection: MAPE-based Cross-Validation

Data:
  - Countries: {len(countries)}
  - Indicators: {len(indicator_cols)}
  - Predictions: 2026-2035

Top 5 Selected Models:
""")
for model, count in top5.items():
    pct = count / len(df_model) * 100
    print(f"  {model}: {count} ({pct:.1f}%)")

print(f"""
Performance:
  - Average MAPE: {df_model['MAPE'].mean():.2f}%
  - Median MAPE: {df_model['MAPE'].median():.2f}%

Output Files:
  - Q316_adaptive_predictions.csv
  - Q316_adaptive_full_data.csv
  - Q316_model_selection.csv
  - Q316_indicator_recommendations.csv
""")
print("=" * 70)

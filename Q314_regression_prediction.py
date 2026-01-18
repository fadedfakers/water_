"""
=============================================================================
Huashu Cup 2026 Problem B - Question 3 Q314
Regression Prediction Model for AI Competitiveness Forecasting
Predict 2026-2035 for 10 countries × 38 indicators
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("Q314: Regression Prediction Model")
print("=" * 70)

# =============================================================================
# 1. Data Loading
# =============================================================================
print("\n--- 1. Data Loading ---")

df = pd.read_csv('panel_data_38indicators.csv')
print(f"Data shape: {df.shape}")

countries = df['Country'].unique()
years = sorted(df['Year'].unique())
indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]

print(f"Countries: {len(countries)}")
print(f"Years: {years[0]} - {years[-1]}")
print(f"Indicators: {len(indicator_cols)}")

pred_years = list(range(2026, 2036))
n_pred = len(pred_years)
print(f"Prediction years: {pred_years[0]} - {pred_years[-1]}")

# =============================================================================
# 2. Regression Model Functions
# =============================================================================
print("\n--- 2. Regression Model Implementation ---")

def linear_regression_predict(x_time, y_values, pred_time):
    """Linear Regression: y = a + b*t"""
    X = x_time.reshape(-1, 1)
    X_pred = pred_time.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y_values)
    
    fitted = model.predict(X)
    forecast = model.predict(X_pred)
    
    r2 = r2_score(y_values, fitted)
    mape = mean_absolute_percentage_error(y_values, fitted) * 100
    
    return forecast, fitted, {'r2': r2, 'mape': mape, 'coef': model.coef_[0], 'intercept': model.intercept_}

def polynomial_regression_predict(x_time, y_values, pred_time, degree=2):
    """Polynomial Regression: y = a + b*t + c*t^2 + ..."""
    poly = PolynomialFeatures(degree=degree)
    X = poly.fit_transform(x_time.reshape(-1, 1))
    X_pred = poly.transform(pred_time.reshape(-1, 1))
    
    model = Ridge(alpha=0.1)  # Use Ridge to prevent overfitting
    model.fit(X, y_values)
    
    fitted = model.predict(X)
    forecast = model.predict(X_pred)
    
    r2 = r2_score(y_values, fitted)
    mape = mean_absolute_percentage_error(y_values, fitted) * 100
    
    return forecast, fitted, {'r2': r2, 'mape': mape, 'degree': degree}

def exponential_regression_predict(x_time, y_values, pred_time):
    """Exponential Regression: y = a * exp(b*t)"""
    # Shift y to positive if needed
    y_min = y_values.min()
    if y_min <= 0:
        y_shifted = y_values - y_min + 1
    else:
        y_shifted = y_values
    
    try:
        # Log-transform for linear fit
        log_y = np.log(y_shifted)
        
        X = x_time.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, log_y)
        
        # Predict
        log_fitted = model.predict(X)
        log_forecast = model.predict(pred_time.reshape(-1, 1))
        
        fitted = np.exp(log_fitted)
        forecast = np.exp(log_forecast)
        
        # Shift back
        if y_min <= 0:
            fitted = fitted + y_min - 1
            forecast = forecast + y_min - 1
        
        r2 = r2_score(y_values, fitted)
        mape = mean_absolute_percentage_error(y_values + 1e-10, fitted + 1e-10) * 100
        
        return forecast, fitted, {'r2': r2, 'mape': mape, 'type': 'exponential'}
    except:
        return linear_regression_predict(x_time, y_values, pred_time)

def logistic_regression_predict(x_time, y_values, pred_time):
    """Logistic Growth: y = K / (1 + exp(-r*(t-t0)))"""
    def logistic(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))
    
    try:
        # Initial guesses
        K0 = y_values.max() * 1.5
        r0 = 0.5
        t0_0 = x_time.mean()
        
        popt, _ = curve_fit(logistic, x_time, y_values, p0=[K0, r0, t0_0], maxfev=5000)
        
        fitted = logistic(x_time, *popt)
        forecast = logistic(pred_time, *popt)
        
        r2 = r2_score(y_values, fitted)
        mape = mean_absolute_percentage_error(y_values + 1e-10, fitted + 1e-10) * 100
        
        return forecast, fitted, {'r2': r2, 'mape': mape, 'K': popt[0], 'r': popt[1], 't0': popt[2]}
    except:
        return linear_regression_predict(x_time, y_values, pred_time)

def power_regression_predict(x_time, y_values, pred_time):
    """Power Regression: y = a * t^b"""
    # Ensure positive values
    x_pos = x_time - x_time.min() + 1
    pred_pos = pred_time - x_time.min() + 1
    
    y_pos = y_values.copy()
    if y_pos.min() <= 0:
        y_pos = y_pos - y_pos.min() + 1
    
    try:
        log_x = np.log(x_pos)
        log_y = np.log(y_pos)
        
        model = LinearRegression()
        model.fit(log_x.reshape(-1, 1), log_y)
        
        log_fitted = model.predict(log_x.reshape(-1, 1))
        log_forecast = model.predict(np.log(pred_pos).reshape(-1, 1))
        
        fitted = np.exp(log_fitted)
        forecast = np.exp(log_forecast)
        
        # Adjust back if shifted
        if y_values.min() <= 0:
            shift = y_values.min() - 1
            fitted = fitted + shift
            forecast = forecast + shift
        
        r2 = r2_score(y_values, fitted)
        mape = mean_absolute_percentage_error(y_values + 1e-10, fitted + 1e-10) * 100
        
        return forecast, fitted, {'r2': r2, 'mape': mape, 'type': 'power'}
    except:
        return linear_regression_predict(x_time, y_values, pred_time)

def auto_regression_predict(x_time, y_values, pred_time):
    """
    Automatically select the best regression model based on R² and MAPE
    """
    n = len(y_values)
    
    # Handle edge cases
    if n < 3:
        return np.full(len(pred_time), y_values[-1]), 'constant', {'r2': np.nan, 'mape': np.nan}
    
    if np.std(y_values) < 1e-10:  # Constant series
        return np.full(len(pred_time), y_values[-1]), 'constant', {'r2': 1.0, 'mape': 0}
    
    results = {}
    
    # 1. Linear Regression
    try:
        forecast, fitted, metrics = linear_regression_predict(x_time, y_values, pred_time)
        results['linear'] = {'forecast': forecast, 'fitted': fitted, 'metrics': metrics}
    except:
        pass
    
    # 2. Quadratic Regression
    try:
        forecast, fitted, metrics = polynomial_regression_predict(x_time, y_values, pred_time, degree=2)
        results['quadratic'] = {'forecast': forecast, 'fitted': fitted, 'metrics': metrics}
    except:
        pass
    
    # 3. Exponential Regression
    try:
        forecast, fitted, metrics = exponential_regression_predict(x_time, y_values, pred_time)
        # Check for reasonable predictions
        if not np.any(np.isnan(forecast)) and not np.any(np.isinf(forecast)):
            if np.abs(forecast).max() < np.abs(y_values).max() * 100:  # Sanity check
                results['exponential'] = {'forecast': forecast, 'fitted': fitted, 'metrics': metrics}
    except:
        pass
    
    # 4. Power Regression
    try:
        forecast, fitted, metrics = power_regression_predict(x_time, y_values, pred_time)
        if not np.any(np.isnan(forecast)) and not np.any(np.isinf(forecast)):
            if np.abs(forecast).max() < np.abs(y_values).max() * 100:
                results['power'] = {'forecast': forecast, 'fitted': fitted, 'metrics': metrics}
    except:
        pass
    
    # 5. Logistic Growth (for S-curve patterns)
    try:
        forecast, fitted, metrics = logistic_regression_predict(x_time, y_values, pred_time)
        if not np.any(np.isnan(forecast)) and not np.any(np.isinf(forecast)):
            if forecast.min() >= 0 and forecast.max() < y_values.max() * 10:
                results['logistic'] = {'forecast': forecast, 'fitted': fitted, 'metrics': metrics}
    except:
        pass
    
    if not results:
        # Fallback
        trend = (y_values[-1] - y_values[0]) / (n - 1) if n > 1 else 0
        forecast = y_values[-1] + trend * np.arange(1, len(pred_time) + 1)
        return forecast, 'fallback', {'r2': np.nan, 'mape': np.nan}
    
    # Select best model based on combined score (higher R², lower MAPE)
    def score_model(name):
        m = results[name]['metrics']
        r2 = m.get('r2', 0)
        mape = m.get('mape', 100)
        # Combined score: prioritize R², penalize high MAPE
        return r2 - mape / 100
    
    best_name = max(results.keys(), key=score_model)
    best = results[best_name]
    
    return best['forecast'], best_name, best['metrics']

# =============================================================================
# 3. Predict All Indicators for All Countries
# =============================================================================
print("\n--- 3. Predicting All Indicators ---")

# Time arrays
hist_time = np.array(years) - years[0]  # Normalize to start from 0
pred_time = np.array(pred_years) - years[0]

predictions = {}
model_info = {}

for country in countries:
    print(f"\nProcessing: {country}")
    country_data = df[df['Country'] == country].sort_values('Year')
    predictions[country] = {}
    model_info[country] = {}
    
    for indicator in indicator_cols:
        try:
            y = country_data[indicator].values
            
            # Auto select and predict
            pred, method, metrics = auto_regression_predict(hist_time, y, pred_time)
            
            # Ensure non-negative for most indicators
            pred = np.maximum(pred, 0)
            
            predictions[country][indicator] = pred
            model_info[country][indicator] = {
                'method': method,
                'r2': metrics.get('r2', np.nan),
                'mape': metrics.get('mape', np.nan)
            }
            
        except Exception as e:
            predictions[country][indicator] = np.full(n_pred, y[-1] if len(y) > 0 else 0)
            model_info[country][indicator] = {
                'method': 'error',
                'r2': np.nan,
                'mape': np.nan
            }
    
    print(f"  Completed {len(indicator_cols)} indicators")

print("\nPrediction complete!")

# =============================================================================
# 4. Create Prediction DataFrame
# =============================================================================
print("\n--- 4. Creating Prediction DataFrame ---")

all_data = []

# Historical data
for _, row in df.iterrows():
    all_data.append(row.to_dict())

# Predicted data
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
# 5. Model Summary
# =============================================================================
print("\n--- 5. Model Summary ---")

model_summary = []
for country in countries:
    for indicator in indicator_cols:
        info = model_info[country][indicator]
        model_summary.append({
            'Country': country,
            'Indicator': indicator,
            'Method': info['method'],
            'R2': info['r2'],
            'MAPE': info['mape']
        })

df_model = pd.DataFrame(model_summary)

print("\nRegression Method Distribution:")
print(df_model['Method'].value_counts())

print(f"\nAverage R²: {df_model['R2'].mean():.4f}")
print(f"Average MAPE: {df_model['MAPE'].mean():.2f}%")
print(f"Median MAPE: {df_model['MAPE'].median():.2f}%")

# =============================================================================
# 6. Visualization
# =============================================================================
print("\n--- 6. Generating Visualizations ---")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sample_indicators = indicator_cols[:6]  # First 6 indicators
# sample_indicators_old = ['AI_Patents', 'AI_Market', 'AI_Researchers', 
                     # 'Large_Models', 'VC_Investment', 'Gov_Investment']
sample_countries = list(countries)[:2]  # First 2 countries

for idx, indicator in enumerate(sample_indicators):
    ax = axes[idx // 3, idx % 3]
    
    for country in sample_countries:
        # Historical
        hist_data = df[df['Country'] == country].sort_values('Year')
        ax.plot(hist_data['Year'], hist_data[indicator], 'o-', label=f'{country} (Historical)')
        
        # Predicted
        pred_data = df_pred[df_pred['Country'] == country].sort_values('Year')
        ax.plot(pred_data['Year'], pred_data[indicator], 's--', label=f'{country} (Predicted)')
    
    ax.axvline(x=2025.5, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel(indicator)
    ax.set_title(f'{indicator} Prediction', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Q314: Regression Prediction Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q314_regression_predictions.png', bbox_inches='tight')
plt.close()
print("Saved: Q314_regression_predictions.png")

# Method distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Method counts
ax1 = axes[0]
method_counts = df_model['Method'].value_counts()
colors = plt.cm.Set2(np.linspace(0, 1, len(method_counts)))
method_counts.plot(kind='bar', ax=ax1, color=colors)
ax1.set_xlabel('Regression Method')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Selected Regression Methods', fontweight='bold')
plt.sca(ax1)
plt.xticks(rotation=45)

# R² distribution by method
ax2 = axes[1]
df_model.boxplot(column='R2', by='Method', ax=ax2)
ax2.set_xlabel('Regression Method')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Score Distribution by Method', fontweight='bold')
plt.sca(ax2)
plt.xticks(rotation=45)

plt.suptitle('')
plt.tight_layout()
plt.savefig('Q314_regression_methods.png', bbox_inches='tight')
plt.close()
print("Saved: Q314_regression_methods.png")

# =============================================================================
# 7. Save Results
# =============================================================================
print("\n--- 7. Saving Results ---")

df_pred.to_csv('Q314_regression_predictions.csv', index=False)
print("Saved: Q314_regression_predictions.csv")

df_full.to_csv('Q314_regression_full_data.csv', index=False)
print("Saved: Q314_regression_full_data.csv")

df_model.to_csv('Q314_regression_model_info.csv', index=False)
print("Saved: Q314_regression_model_info.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q314 REGRESSION PREDICTION SUMMARY")
print("=" * 70)
print(f"""
Models Used:
  1. Linear: y = a + b*t
  2. Quadratic: y = a + b*t + c*t²
  3. Exponential: y = a * exp(b*t)
  4. Power: y = a * t^b
  5. Logistic: y = K / (1 + exp(-r*(t-t0)))

Data:
  - Countries: {len(countries)}
  - Indicators: {len(indicator_cols)}
  - Historical period: 2016-2025
  - Prediction period: 2026-2035

Model Selection:
  - Selection criterion: R² and MAPE combined
  - Method distribution: {df_model['Method'].value_counts().to_dict()}

Model Performance:
  - Average R²: {df_model['R2'].mean():.4f}
  - Average MAPE: {df_model['MAPE'].mean():.2f}%
  - Median MAPE: {df_model['MAPE'].median():.2f}%

Output Files:
  - Q314_regression_predictions.csv: Predictions for 2026-2035
  - Q314_regression_full_data.csv: Historical + Predicted data
  - Q314_regression_model_info.csv: Model parameters and metrics
""")
print("=" * 70)

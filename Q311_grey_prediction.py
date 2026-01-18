"""
=============================================================================
Huashu Cup 2026 Problem B - Question 3 Q311
Grey Prediction Model GM(1,1) for AI Competitiveness Forecasting
Predict 2026-2035 for 10 countries × 38 indicators
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
print("=" * 70)
print("Q311: Grey Prediction Model GM(1,1)")
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

# Prediction horizon
pred_years = list(range(2026, 2036))
print(f"Prediction years: {pred_years[0]} - {pred_years[-1]}")

# =============================================================================
# 2. GM(1,1) Model Implementation
# =============================================================================
print("\n--- 2. GM(1,1) Model Implementation ---")

class GM11:
    """
    Grey Model GM(1,1) for time series prediction
    
    Model: dx^(1)/dt + a*x^(1) = b
    Solution: x^(1)(k+1) = (x^(0)(1) - b/a) * exp(-a*k) + b/a
    """
    
    def __init__(self):
        self.a = None  # Development coefficient
        self.b = None  # Grey action quantity
        self.x0 = None  # Original sequence
        self.x1 = None  # Accumulated sequence
        
    def fit(self, x):
        """Fit GM(1,1) model"""
        x = np.array(x).flatten()
        n = len(x)
        
        if n < 4:
            raise ValueError("GM(1,1) requires at least 4 data points")
        
        # Check for non-positive values
        if np.any(x <= 0):
            # Shift data to positive
            x = x - x.min() + 1
        
        self.x0 = x.copy()
        
        # Step 1: Accumulated Generating Operation (AGO)
        self.x1 = np.cumsum(x)
        
        # Step 2: Mean sequence with consecutive neighbors
        z1 = 0.5 * (self.x1[:-1] + self.x1[1:])
        
        # Step 3: Build data matrix
        B = np.column_stack([-z1, np.ones(n-1)])
        Y = x[1:].reshape(-1, 1)
        
        # Step 4: Least squares estimation
        try:
            params = np.linalg.lstsq(B, Y, rcond=None)[0]
            self.a = params[0, 0]
            self.b = params[1, 0]
        except:
            self.a = 0.01
            self.b = x[0]
        
        return self
    
    def predict(self, n_pred):
        """Predict n_pred future values"""
        if self.a is None:
            raise ValueError("Model not fitted yet")
        
        n_total = len(self.x0) + n_pred
        
        # Predicted accumulated sequence
        x1_pred = np.zeros(n_total)
        x1_pred[0] = self.x0[0]
        
        for k in range(1, n_total):
            x1_pred[k] = (self.x0[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
        
        # Inverse AGO to get original sequence
        x0_pred = np.zeros(n_total)
        x0_pred[0] = x1_pred[0]
        x0_pred[1:] = np.diff(x1_pred)
        
        return x0_pred[-n_pred:]
    
    def fit_predict(self, x, n_pred):
        """Fit and predict"""
        self.fit(x)
        return self.predict(n_pred)
    
    def get_fitted_values(self):
        """Get fitted values for training period"""
        n = len(self.x0)
        x1_fitted = np.zeros(n)
        x1_fitted[0] = self.x0[0]
        
        for k in range(1, n):
            x1_fitted[k] = (self.x0[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
        
        x0_fitted = np.zeros(n)
        x0_fitted[0] = x1_fitted[0]
        x0_fitted[1:] = np.diff(x1_fitted)
        
        return x0_fitted
    
    def evaluate(self):
        """Evaluate model accuracy"""
        fitted = self.get_fitted_values()
        
        # Residual analysis
        residuals = self.x0 - fitted
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs(residuals / (self.x0 + 1e-10))) * 100
        
        # Relative error for each point
        relative_errors = np.abs(residuals / (self.x0 + 1e-10)) * 100
        
        # Grade based on MAPE
        if mape < 1:
            grade = "Excellent"
        elif mape < 5:
            grade = "Good"
        elif mape < 10:
            grade = "Qualified"
        else:
            grade = "Poor"
        
        return {
            'mape': mape,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'grade': grade,
            'relative_errors': relative_errors
        }


def gm11_with_markov_correction(x, n_pred, n_states=3):
    """
    GM(1,1) with Markov chain correction for residual randomness
    """
    # Fit basic GM(1,1)
    gm = GM11()
    gm.fit(x)
    
    # Get fitted values and residuals
    fitted = gm.get_fitted_values()
    residuals = x - fitted
    
    # Classify residuals into states
    if len(np.unique(residuals)) >= n_states:
        bins = np.percentile(residuals, np.linspace(0, 100, n_states + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        states = np.digitize(residuals, bins) - 1
        
        # Transition probability matrix
        trans_matrix = np.zeros((n_states, n_states))
        for i in range(len(states) - 1):
            trans_matrix[states[i], states[i+1]] += 1
        
        # Normalize
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_matrix = trans_matrix / row_sums
        
        # State centers (mean residual for each state)
        state_centers = np.array([residuals[states == s].mean() if np.sum(states == s) > 0 else 0 
                                  for s in range(n_states)])
        
        # Predict with correction
        base_pred = gm.predict(n_pred)
        
        # Apply Markov correction
        current_state = states[-1]
        corrections = np.zeros(n_pred)
        
        for i in range(n_pred):
            # Expected correction based on transition probabilities
            corrections[i] = np.dot(trans_matrix[current_state], state_centers)
            # Update state (most probable)
            current_state = np.argmax(trans_matrix[current_state])
        
        corrected_pred = base_pred + corrections
    else:
        corrected_pred = gm.predict(n_pred)
    
    return corrected_pred, gm.evaluate()

# =============================================================================
# 3. Predict All Indicators for All Countries
# =============================================================================
print("\n--- 3. Predicting All Indicators ---")

n_pred = len(pred_years)
predictions = {}
model_evaluations = {}

for country in countries:
    print(f"\nProcessing: {country}")
    country_data = df[df['Country'] == country].sort_values('Year')
    predictions[country] = {}
    model_evaluations[country] = {}
    
    for indicator in indicator_cols:
        try:
            x = country_data[indicator].values
            
            # Handle edge cases
            if np.all(x == x[0]):  # Constant series
                pred = np.full(n_pred, x[0])
                eval_result = {'mape': 0, 'rmse': 0, 'grade': 'Constant'}
            elif np.any(np.isnan(x)) or len(x) < 4:
                # Use simple extrapolation
                pred = np.full(n_pred, x[-1])
                eval_result = {'mape': np.nan, 'rmse': np.nan, 'grade': 'Insufficient'}
            else:
                # Apply GM(1,1) with Markov correction
                pred, eval_result = gm11_with_markov_correction(x, n_pred)
                
                # Ensure non-negative for certain indicators
                pred = np.maximum(pred, 0)
            
            predictions[country][indicator] = pred
            model_evaluations[country][indicator] = eval_result
            
        except Exception as e:
            # Fallback: use last value
            predictions[country][indicator] = np.full(n_pred, x[-1] if len(x) > 0 else 0)
            model_evaluations[country][indicator] = {'mape': np.nan, 'rmse': np.nan, 'grade': 'Error'}

print("\nPrediction complete!")

# =============================================================================
# 4. Create Prediction DataFrame
# =============================================================================
print("\n--- 4. Creating Prediction DataFrame ---")

# Combine historical and predicted data
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

# Extract prediction only
df_pred = df_full[df_full['Year'] >= 2026].copy()

print(f"Full data shape: {df_full.shape}")
print(f"Prediction data shape: {df_pred.shape}")

# =============================================================================
# 5. Model Evaluation Summary
# =============================================================================
print("\n--- 5. Model Evaluation Summary ---")

eval_summary = []
for country in countries:
    for indicator in indicator_cols:
        eval_result = model_evaluations[country][indicator]
        eval_summary.append({
            'Country': country,
            'Indicator': indicator,
            'MAPE': eval_result.get('mape', np.nan),
            'RMSE': eval_result.get('rmse', np.nan),
            'Grade': eval_result.get('grade', 'N/A')
        })

df_eval = pd.DataFrame(eval_summary)

print("\nOverall Model Performance:")
print(f"  Average MAPE: {df_eval['MAPE'].mean():.2f}%")
print(f"  Median MAPE: {df_eval['MAPE'].median():.2f}%")

print("\nGrade Distribution:")
print(df_eval['Grade'].value_counts())

# =============================================================================
# 6. Visualization
# =============================================================================
print("\n--- 6. Generating Visualizations ---")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Sample indicators to visualize
# Use actual column names from data
sample_indicators = indicator_cols[:6]  # First 6 indicators
sample_countries = list(countries)[:2]  # First 2 countries
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
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

plt.suptitle('Q311: GM(1,1) Grey Prediction Model Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q311_gm11_predictions.png', bbox_inches='tight')
plt.close()
print("Saved: Q311_gm11_predictions.png")

# =============================================================================
# 7. Save Results
# =============================================================================
print("\n--- 7. Saving Results ---")

# Save predictions only (2026-2035)
df_pred.to_csv('Q311_gm11_predictions.csv', index=False)
print("Saved: Q311_gm11_predictions.csv")

# Save full data (historical + predicted)
df_full.to_csv('Q311_gm11_full_data.csv', index=False)
print("Saved: Q311_gm11_full_data.csv")

# Save evaluation results
df_eval.to_csv('Q311_gm11_evaluation.csv', index=False)
print("Saved: Q311_gm11_evaluation.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q311 GM(1,1) GREY PREDICTION SUMMARY")
print("=" * 70)
print(f"""
Model: GM(1,1) with Markov Chain Correction

Data:
  - Countries: {len(countries)}
  - Indicators: {len(indicator_cols)}
  - Historical period: 2016-2025
  - Prediction period: 2026-2035

Model Performance:
  - Average MAPE: {df_eval['MAPE'].mean():.2f}%
  - Excellent (MAPE < 1%): {(df_eval['Grade'] == 'Excellent').sum()}
  - Good (MAPE < 5%): {(df_eval['Grade'] == 'Good').sum()}
  - Qualified (MAPE < 10%): {(df_eval['Grade'] == 'Qualified').sum()}

Output Files:
  - Q311_gm11_predictions.csv: Predictions for 2026-2035
  - Q311_gm11_full_data.csv: Historical + Predicted data
  - Q311_gm11_evaluation.csv: Model evaluation metrics
""")
print("=" * 70)

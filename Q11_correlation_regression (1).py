"""
=============================================================================
Huashu Cup 2026 Problem B - Question 1 Q11
Pearson/Spearman Correlation Analysis + Heatmap + Multiple Regression
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Simple OLS implementation without statsmodels
def simple_ols(X, y):
    """Simple OLS regression returning coefficients, R-squared, and F-statistic"""
    n, k = X.shape
    # Add constant if not present
    if not np.allclose(X[:, 0], 1):
        X = np.column_stack([np.ones(n), X])
        k = k + 1
    
    # OLS estimates
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    
    # Predictions and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    
    # R-squared
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((y - np.mean(y))**2)
    R_squared = 1 - SS_res / SS_tot
    
    # Adjusted R-squared
    R_squared_adj = 1 - (1 - R_squared) * (n - 1) / (n - k)
    
    # F-statistic
    SS_reg = SS_tot - SS_res
    df_reg = k - 1
    df_res = n - k
    F_stat = (SS_reg / df_reg) / (SS_res / df_res) if df_res > 0 else np.nan
    
    # Standard errors
    MSE = SS_res / df_res if df_res > 0 else np.nan
    se = np.sqrt(np.diag(XtX_inv) * MSE)
    
    # t-statistics and p-values
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_res))
    
    return {
        'params': beta,
        'se': se,
        'pvalues': p_values,
        'rsquared': R_squared,
        'rsquared_adj': R_squared_adj,
        'fvalue': F_stat
    }

# Set plot style
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("=" * 70)
print("Q11: Correlation Analysis and Multiple Regression")
print("=" * 70)

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
print("\n--- 1. Data Loading ---")

df = pd.read_csv('panel_data_38indicators.csv')
print(f"Data dimension: {df.shape[0]} rows × {df.shape[1]} columns")

# Extract country and year
Country = df['Country']
Year = df['Year']

# Extract numeric indicators
indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
X = df[indicator_cols].values
n, p = X.shape
print(f"Number of indicators: {p}")

# English short names
short_names = ['Top500', 'GPU_Cluster', 'DC_Compute', 'AI_Chips', '5G_Coverage',
    'Internet_BW', 'Data_Centers', 'Internet_Pen', 'AI_Researchers', 'Talent_Flow',
    'Top_Scholars', 'STEM_Grads', 'AI_Papers', 'AI_Labs', 'Gov_Investment',
    'Enterprise_RD', 'VC_Investment', 'Paper_Citations', 'AI_Patents', 'AI_Market',
    'AI_Companies', 'AI_Unicorns', 'Large_Models', 'Industry_Market', 'Tax_Incentive',
    'Subsidy_Amount', 'Policy_Count', 'Subsidy_Intensity', 'Regulatory_FW', 'GDP',
    'GDP_Growth', 'FX_Reserves', 'Population', 'Working_Age', 'Higher_Edu',
    'GII_Rank', 'RD_Density', 'FDI_Inflow']

# =============================================================================
# 2. Data Normalization
# =============================================================================
print("\n--- 2. Data Normalization (Min-Max) ---")

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
df_norm = pd.DataFrame(X_norm, columns=short_names)

# =============================================================================
# 3. Pearson Correlation Matrix
# =============================================================================
print("\n--- 3. Computing Pearson Correlation ---")

R_pearson = df_norm.corr(method='pearson')

# Display highly correlated pairs
print("\nHighly correlated pairs (|r| > 0.85):")
high_corr_count = 0
for i in range(p):
    for j in range(i+1, p):
        if abs(R_pearson.iloc[i, j]) > 0.85:
            print(f"  {short_names[i]} <-> {short_names[j]}: r = {R_pearson.iloc[i, j]:.3f}")
            high_corr_count += 1
print(f"Total {high_corr_count} highly correlated pairs found")

# =============================================================================
# 4. Spearman Rank Correlation Matrix
# =============================================================================
print("\n--- 4. Computing Spearman Rank Correlation ---")

R_spearman = df_norm.corr(method='spearman')

# =============================================================================
# 5. Plot Correlation Heatmaps
# =============================================================================
print("\n--- 5. Plotting Correlation Heatmaps ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Dimension boundaries for visual separation
dim_bounds = [0, 8, 14, 20, 24, 29, 38]
dim_labels = ['Infra', 'Talent', 'R&D', 'Industry', 'Policy', 'National']

# Pearson Heatmap
ax1 = axes[0]
sns.heatmap(R_pearson, ax=ax1, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.1, cbar_kws={'shrink': 0.8})
ax1.set_title('Pearson Correlation Heatmap', fontsize=14, fontweight='bold')
ax1.set_xlabel('Indicator Index')
ax1.set_ylabel('Indicator Index')

# Add dimension separation lines
for bound in dim_bounds[1:-1]:
    ax1.axhline(y=bound, color='black', linewidth=1.5)
    ax1.axvline(x=bound, color='black', linewidth=1.5)

# Spearman Heatmap
ax2 = axes[1]
sns.heatmap(R_spearman, ax=ax2, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.1, cbar_kws={'shrink': 0.8})
ax2.set_title('Spearman Rank Correlation Heatmap', fontsize=14, fontweight='bold')
ax2.set_xlabel('Indicator Index')
ax2.set_ylabel('Indicator Index')

for bound in dim_bounds[1:-1]:
    ax2.axhline(y=bound, color='black', linewidth=1.5)
    ax2.axvline(x=bound, color='black', linewidth=1.5)

plt.suptitle('Correlation Analysis of 38 AI Development Indicators', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Q11_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: Q11_correlation_heatmap.png")

# =============================================================================
# 6. Construct AI Competitiveness Index (Dependent Variable)
# =============================================================================
print("\n--- 6. Constructing AI Competitiveness Index ---")

# Core output indicators indices
output_idx = [19, 20, 18, 21, 22]  # AI_Market, AI_Companies, AI_Patents, AI_Unicorns, Large_Models
output_names = [short_names[i] for i in output_idx]

# Compute composite competitiveness
Y = df_norm[output_names].mean(axis=1)
print(f"Composite Index = mean({', '.join(output_names)})")

# =============================================================================
# 7. Multiple Regression Analysis
# =============================================================================
print("\n--- 7. Multiple Regression Analysis ---")

# Select input indicators (exclude output indicators)
input_idx = [i for i in range(p) if i not in output_idx]
input_names = [short_names[i] for i in input_idx]
X_input = df_norm[input_names].values

# OLS Regression
X_reg = np.column_stack([np.ones(n), X_input])
model = simple_ols(X_reg, Y.values)

print(f"\nRegression Results:")
print(f"  R-squared = {model['rsquared']:.4f}")
print(f"  Adj. R-squared = {model['rsquared_adj']:.4f}")
print(f"  F-statistic = {model['fvalue']:.4f}")

# Significant coefficients (excluding constant)
print("\nSignificant coefficients (p < 0.05):")
for i, name in enumerate(input_names):
    coef = model['params'][i+1]  # +1 to skip constant
    pval = model['pvalues'][i+1]
    if pval < 0.05:
        print(f"  {name}: β = {coef:.4f}, p = {pval:.4f} ***")

# =============================================================================
# 8. Feature Importance Analysis
# =============================================================================
print("\n--- 8. Feature Importance Analysis ---")

# Get all coefficients and sort by absolute value
coef_df = pd.DataFrame({
    'Variable': input_names,
    'Coefficient': model['params'][1:],  # Skip constant
    'P_value': model['pvalues'][1:]
})
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('Abs_Coef', ascending=False)

print("\nTop 15 influential factors:")
print(coef_df.head(15).to_string(index=False))

# =============================================================================
# 9. Plot Regression Coefficients
# =============================================================================
print("\n--- 9. Plotting Regression Coefficients ---")

fig, ax = plt.subplots(figsize=(10, 8))

top_n = 20
top_coef = coef_df.head(top_n)

colors = ['#27ae60' if c > 0 else '#e74c3c' for c in top_coef['Coefficient']]

bars = ax.barh(range(top_n), top_coef['Coefficient'], color=colors, alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_coef['Variable'])
ax.set_xlabel('Regression Coefficient', fontsize=12)
ax.set_title('Impact of Factors on AI Competitiveness\n(Regression Coefficients)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_coef['Coefficient'])):
    x_pos = val + 0.01 if val > 0 else val - 0.01
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=8)

# Legend
ax.text(0.75, 0.95, 'Positive = Promoting', transform=ax.transAxes, 
        color='#27ae60', fontweight='bold', fontsize=10)
ax.text(0.75, 0.90, 'Negative = Constraining', transform=ax.transAxes, 
        color='#e74c3c', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('Q11_regression_coefficients.png', bbox_inches='tight')
plt.close()
print("Saved: Q11_regression_coefficients.png")

# =============================================================================
# 10. Dimension-Level Correlation Analysis
# =============================================================================
print("\n--- 10. Dimension-Level Correlation Analysis ---")

dim_names = ['Infrastructure', 'Talent', 'R&D Investment', 'Industry', 'Policy', 'National Strength']
dim_ranges = [(0, 8), (8, 14), (14, 20), (20, 24), (24, 29), (29, 38)]

# Compute dimension scores
dim_scores = pd.DataFrame()
for i, (dim_name, (start, end)) in enumerate(zip(dim_names, dim_ranges)):
    dim_cols = short_names[start:end]
    dim_scores[dim_name] = df_norm[dim_cols].mean(axis=1)

# Dimension correlation matrix
R_dim = dim_scores.corr()

# Plot dimension correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(R_dim, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 10, 'fontweight': 'bold'})
ax.set_title('Inter-Dimension Correlation Matrix', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('Q11_dimension_correlation.png', bbox_inches='tight')
plt.close()
print("Saved: Q11_dimension_correlation.png")

# =============================================================================
# 11. VIF Analysis for Multicollinearity
# =============================================================================
print("\n--- 11. VIF Analysis ---")

def calculate_vif(X):
    """Calculate VIF for each variable"""
    vif_data = []
    for i in range(X.shape[1]):
        # Get all columns except i
        X_others = np.delete(X, i, axis=1)
        y_i = X[:, i]
        
        # Regress column i on others
        X_reg = np.column_stack([np.ones(len(y_i)), X_others])
        result = simple_ols(X_reg, y_i)
        r_squared = result['rsquared']
        
        # VIF = 1 / (1 - R²)
        vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        vif_data.append(vif)
    return vif_data

# Calculate VIF for key variables
key_vars = ['GPU_Cluster', 'AI_Researchers', 'Gov_Investment', 'VC_Investment', 
            '5G_Coverage', 'AI_Labs', 'GDP']
X_vif = df_norm[key_vars].values

vif_values = calculate_vif(X_vif)

print("\nVariance Inflation Factor (VIF):")
for name, vif in zip(key_vars, vif_values):
    print(f"  {name}: VIF = {vif:.2f}")

# =============================================================================
# 12. Save Results
# =============================================================================
print("\n--- 12. Saving Results ---")

# Save correlation matrices
R_pearson.to_csv('Q11_pearson_correlation.csv')
R_spearman.to_csv('Q11_spearman_correlation.csv')

# Save regression results
coef_df.to_csv('Q11_regression_results.csv', index=False)

# Save dimension correlation
R_dim.to_csv('Q11_dimension_correlation.csv')

print("Saved: Q11_pearson_correlation.csv")
print("Saved: Q11_spearman_correlation.csv")
print("Saved: Q11_regression_results.csv")
print("Saved: Q11_dimension_correlation.csv")

print("\n" + "=" * 70)
print("Q11 Analysis Complete")
print("=" * 70)

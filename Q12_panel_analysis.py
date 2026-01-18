"""
=============================================================================
Huashu Cup 2026 Problem B - Question 1 Q12
Detrending (First Difference/HP Filter) + Panel Correlation + Fixed Effects
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Simple OLS implementation
def simple_ols(X, y):
    """Simple OLS regression"""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X + 0.001 * np.eye(k))  # Ridge for stability
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((y - np.mean(y))**2)
    R_squared = 1 - SS_res / SS_tot
    return {'params': beta, 'rsquared': R_squared}

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("Q12: Panel Data Analysis")
print("=" * 70)
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
# 1. Data Loading
# =============================================================================
print("\n--- 1. Data Loading ---")

df = pd.read_csv('panel_data_38indicators.csv')
Country = df['Country']
Year = df['Year']

indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
X = df[indicator_cols].values
n, p = X.shape

countries = Country.unique()
years = Year.unique()
n_country = len(countries)
n_year = len(years)
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
# English short names
short_names = ['Top500', 'GPU_Cluster', 'DC_Compute', 'AI_Chips', '5G_Coverage',
    'Internet_BW', 'Data_Centers', 'Internet_Pen', 'AI_Researchers', 'Talent_Flow',
    'Top_Scholars', 'STEM_Grads', 'AI_Papers', 'AI_Labs', 'Gov_Investment',
    'Enterprise_RD', 'VC_Investment', 'Paper_Citations', 'AI_Patents', 'AI_Market',
    'AI_Companies', 'AI_Unicorns', 'Large_Models', 'Industry_Market', 'Tax_Incentive',
    'Subsidy_Amount', 'Policy_Count', 'Subsidy_Intensity', 'Regulatory_FW', 'GDP',
    'GDP_Growth', 'FX_Reserves', 'Population', 'Working_Age', 'Higher_Edu',
    'GII_Rank', 'RD_Density', 'FDI_Inflow']

print(f"Countries: {n_country}, Years: {n_year}, Indicators: {p}")

# =============================================================================
# 2. Data Normalization
# =============================================================================
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
df_norm = pd.DataFrame(X_norm, columns=short_names)
df_norm['Country'] = Country.values
df_norm['Year'] = Year.values

# =============================================================================
# 3. First Difference Detrending
# =============================================================================
print("\n--- 3. First Difference Detrending ---")

diff_data = []
for country in countries:
    df_c = df_norm[df_norm['Country'] == country].sort_values('Year')
    for i in range(1, len(df_c)):
        diff_row = df_c[short_names].iloc[i].values - df_c[short_names].iloc[i-1].values
        diff_data.append(diff_row)

X_diff = np.array(diff_data)
df_diff = pd.DataFrame(X_diff, columns=short_names)
print(f"Sample size after differencing: {len(X_diff)}")

R_diff = df_diff.corr()

# =============================================================================
# 4. HP Filter Detrending
# =============================================================================
print("\n--- 4. HP Filter Detrending ---")

def hp_filter(y, lamb=100):
    """Hodrick-Prescott filter"""
    T = len(y)
    if T < 4:
        return y - np.mean(y)
    
    # Construct the penalty matrix
    K = np.zeros((T-2, T))
    for i in range(T-2):
        K[i, i] = 1
        K[i, i+1] = -2
        K[i, i+2] = 1
    
    # Solve for trend
    I = np.eye(T)
    trend = np.linalg.solve(I + lamb * K.T @ K, y)
    cycle = y - trend
    return cycle
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
X_cycle = np.zeros_like(X_norm)

for i, country in enumerate(countries):
    mask = df_norm['Country'] == country
    idx = df_norm[mask].sort_values('Year').index
    
    for j in range(p):
        y = X_norm[idx, j]
        X_cycle[idx, j] = hp_filter(y, lamb=100)

df_cycle = pd.DataFrame(X_cycle, columns=short_names)
R_hp = df_cycle.corr()
print("HP filter detrending complete")

# =============================================================================
# 5. Plot Detrending Comparison
# =============================================================================
print("\n--- 5. Plotting Detrending Comparison ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Original correlation
R_orig = df_norm[short_names].corr()

ax1 = axes[0]
sns.heatmap(R_orig, ax=ax1, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'shrink': 0.6})
ax1.set_title('Original Data Correlation', fontsize=12, fontweight='bold')
ax1.set_xlabel('Indicator Index')
ax1.set_ylabel('Indicator Index')

ax2 = axes[1]
sns.heatmap(R_diff, ax=ax2, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'shrink': 0.6})
ax2.set_title('After First Differencing', fontsize=12, fontweight='bold')
ax2.set_xlabel('Indicator Index')
ax2.set_ylabel('Indicator Index')

ax3 = axes[2]
sns.heatmap(R_hp, ax=ax3, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'shrink': 0.6})
ax3.set_title('After HP Filtering', fontsize=12, fontweight='bold')
ax3.set_xlabel('Indicator Index')
ax3.set_ylabel('Indicator Index')

plt.suptitle('Impact of Detrending on Correlation Structure', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q12_detrend_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: Q12_detrend_comparison.png")

# =============================================================================
# 6. Panel Correlation: Between vs Within
# =============================================================================
print("\n--- 6. Panel Correlation Analysis ---")

# Between correlation (country means)
X_between = []
for country in countries:
    mask = df_norm['Country'] == country
    X_between.append(df_norm[mask][short_names].mean().values)
X_between = np.array(X_between)
df_between = pd.DataFrame(X_between, columns=short_names)
R_between = df_between.corr()

# Within correlation (demeaned by country)
X_within = np.zeros_like(X_norm)
for country in countries:
    mask = (df_norm['Country'] == country).values
    country_mean = X_norm[mask].mean(axis=0)
    X_within[mask] = X_norm[mask] - country_mean
df_within = pd.DataFrame(X_within, columns=short_names)
R_within = df_within.corr()

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
sns.heatmap(R_between, ax=ax1, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'shrink': 0.6})
ax1.set_title('Between-Country Correlation\n(Cross-sectional variation)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Indicator Index')
ax1.set_ylabel('Indicator Index')

ax2 = axes[1]
sns.heatmap(R_within, ax=ax2, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'shrink': 0.6})
ax2.set_title('Within-Country Correlation\n(Time-series variation)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Indicator Index')
ax2.set_ylabel('Indicator Index')

plt.suptitle('Panel Correlation: Cross-Country vs Time-Series', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q12_panel_correlation.png', bbox_inches='tight')
plt.close()
print("Saved: Q12_panel_correlation.png")

# =============================================================================
# 7. Fixed Effects Panel Regression
# =============================================================================
print("\n--- 7. Fixed Effects Panel Regression ---")

# Construct dependent variable
output_idx = [19, 20, 18]  # AI_Market, AI_Companies, AI_Patents
output_names = [short_names[i] for i in output_idx]
Y = df_norm[output_names].mean(axis=1).values

# Key independent variables
key_idx = [1, 8, 14, 20, 26, 29]  # GPU_Cluster, AI_Researchers, Gov_Investment, AI_Companies, Policy_Count, GDP
key_names = [short_names[i] for i in key_idx]
X_key = df_norm[key_names].values

# Create dummy variables
country_dummies = pd.get_dummies(df_norm['Country'], prefix='C', drop_first=True).values
year_dummies = pd.get_dummies(df_norm['Year'], prefix='Y', drop_first=True).values

# Model 1: Pooled OLS
print("\nModel 1: Pooled OLS (No Fixed Effects)")
X_pooled = np.column_stack([np.ones(n), X_key])
model_pooled = simple_ols(X_pooled, Y)
print(f"  R-squared = {model_pooled['rsquared']:.4f}")
print("  Coefficients:")
for i, name in enumerate(key_names):
    print(f"    {name}: {model_pooled['params'][i+1]:.4f}")

# Model 2: Country Fixed Effects
print("\nModel 2: Country Fixed Effects")
X_fe_country = np.column_stack([np.ones(n), X_key, country_dummies])
model_fe_country = simple_ols(X_fe_country, Y)
print(f"  R-squared = {model_fe_country['rsquared']:.4f}")
print("  Core variable coefficients:")
for i, name in enumerate(key_names):
    print(f"    {name}: {model_fe_country['params'][i+1]:.4f}")

# Model 3: Two-Way Fixed Effects
print("\nModel 3: Two-Way Fixed Effects (Country + Year)")
X_fe_both = np.column_stack([np.ones(n), X_key, country_dummies, year_dummies])
model_fe_both = simple_ols(X_fe_both, Y)
print(f"  R-squared = {model_fe_both['rsquared']:.4f}")
print("  Core variable coefficients:")
for i, name in enumerate(key_names):
    print(f"    {name}: {model_fe_both['params'][i+1]:.4f}")
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
# Extract coefficients for plotting
coef_pooled = [model_pooled['params'][i+1] for i in range(len(key_names))]
coef_fe_country = [model_fe_country['params'][i+1] for i in range(len(key_names))]
coef_fe_both = [model_fe_both['params'][i+1] for i in range(len(key_names))]

# =============================================================================
# 8. Plot Model Comparison
# =============================================================================
print("\n--- 8. Plotting Model Comparison ---")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(key_names))
width = 0.25

bars1 = ax.bar(x - width, coef_pooled, width, label='Pooled OLS', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, coef_fe_country, width, label='Country FE', color='#e74c3c', alpha=0.8)
bars3 = ax.bar(x + width, coef_fe_both, width, label='Two-Way FE', color='#27ae60', alpha=0.8)

ax.set_xlabel('Variables', fontsize=12)
ax.set_ylabel('Regression Coefficient', fontsize=12)
ax.set_title('Coefficient Comparison Across Panel Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(key_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('Q12_panel_regression.png', bbox_inches='tight')
plt.close()
print("Saved: Q12_panel_regression.png")

# =============================================================================
# 9. Variance Decomposition
# =============================================================================
print("\n--- 9. Variance Decomposition ---")

print("\nVariance decomposition (Between/Total):")
for j in range(min(10, p)):
    total_var = X_norm[:, j].var()
    between_var = X_between[:, j].var()
    within_var = X_within[:, j].var()
    between_ratio = between_var / (between_var + within_var) * 100
    print(f"  {short_names[j]}: Between {between_ratio:.1f}%, Within {100-between_ratio:.1f}%")

# =============================================================================
# 10. Save Results
# =============================================================================
print("\n--- 10. Saving Results ---")

R_diff.to_csv('Q12_correlation_diff.csv')
R_hp.to_csv('Q12_correlation_hp.csv')

# Save regression results
results_df = pd.DataFrame({
    'Variable': key_names,
    'Pooled_OLS': coef_pooled,
    'Country_FE': coef_fe_country,
    'TwoWay_FE': coef_fe_both
})
results_df.to_csv('Q12_panel_regression_results.csv', index=False)

print("Saved: Q12_correlation_diff.csv")
print("Saved: Q12_correlation_hp.csv")
print("Saved: Q12_panel_regression_results.csv")

print("\n" + "=" * 70)
print("Q12 Analysis Complete")
print("=" * 70)

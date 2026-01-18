"""
=============================================================================
Huashu Cup 2026 Problem B - Question 1 Q14
Partial Correlation Analysis - Finding Net Correlations
=============================================================================
"""
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("Q14: Partial Correlation Analysis")
print("=" * 70)

# =============================================================================
# 1. Data Loading
# =============================================================================
print("\n--- 1. Data Loading ---")

df = pd.read_csv('panel_data_38indicators.csv')
indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
X = df[indicator_cols].values
n, p = X.shape

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# English short names
short_names = ['Top500', 'GPU_Cluster', 'DC_Compute', 'AI_Chips', '5G_Coverage',
    'Internet_BW', 'Data_Centers', 'Internet_Pen', 'AI_Researchers', 'Talent_Flow',
    'Top_Scholars', 'STEM_Grads', 'AI_Papers', 'AI_Labs', 'Gov_Investment',
    'Enterprise_RD', 'VC_Investment', 'Paper_Citations', 'AI_Patents', 'AI_Market',
    'AI_Companies', 'AI_Unicorns', 'Large_Models', 'Industry_Market', 'Tax_Incentive',
    'Subsidy_Amount', 'Policy_Count', 'Subsidy_Intensity', 'Regulatory_FW', 'GDP',
    'GDP_Growth', 'FX_Reserves', 'Population', 'Working_Age', 'Higher_Edu',
    'GII_Rank', 'RD_Density', 'FDI_Inflow']

df_norm = pd.DataFrame(X_norm, columns=short_names)

print(f"Sample size: {n}, Indicators: {p}")

# =============================================================================
# 2. Simple Correlation Matrix
# =============================================================================
print("\n--- 2. Computing Simple Correlation Matrix ---")

R_simple = df_norm.corr().values

# =============================================================================
# 3. Partial Correlation Matrix (Controlling All Other Variables)
# =============================================================================
print("\n--- 3. Computing Partial Correlation Matrix ---")
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
def compute_partial_correlation_matrix(X, ridge=0.1):
    """Compute partial correlation matrix from precision matrix"""
    Sigma = np.cov(X.T)
    Sigma_reg = Sigma + ridge * np.eye(Sigma.shape[0])
    Precision = np.linalg.inv(Sigma_reg)
    
    D = np.diag(Precision)
    p = len(D)
    R_partial = np.zeros((p, p))
    
    for i in range(p):
        for j in range(p):
            if i == j:
                R_partial[i, j] = 1
            else:
                R_partial[i, j] = -Precision[i, j] / np.sqrt(D[i] * D[j])
    
    return R_partial

R_partial_all = compute_partial_correlation_matrix(X_norm)
print("Partial correlation matrix computed")

# =============================================================================
# 4. First-Order Partial Correlation
# =============================================================================
print("\n--- 4. First-Order Partial Correlation ---")

def partial_corr(X, i, j, control_idx):
    """Compute partial correlation between i and j, controlling for control_idx"""
    if isinstance(control_idx, int):
        control_idx = [control_idx]
    
    Z = X[:, control_idx]
    Z = np.column_stack([np.ones(len(Z)), Z])
    
    # Regress i on Z, get residuals
    xi = X[:, i]
    beta_i = np.linalg.lstsq(Z, xi, rcond=None)[0]
    residual_i = xi - Z @ beta_i
    
    # Regress j on Z, get residuals
    xj = X[:, j]
    beta_j = np.linalg.lstsq(Z, xj, rcond=None)[0]
    residual_j = xj - Z @ beta_j
    
    # Correlation of residuals
    return np.corrcoef(residual_i, residual_j)[0, 1]

# Key variables
key_vars = ['GPU_Cluster', 'AI_Researchers', 'Gov_Investment', 'AI_Market', 'GDP']
key_idx = [short_names.index(v) for v in key_vars]

print("\nFirst-order partial correlation (controlling GDP):")
gdp_idx = short_names.index('GDP')

print(f"\n{'Variable Pair':<35} {'Simple r':>10} {'Partial r':>12} {'Difference':>12}")
print("-" * 70)

for i in range(len(key_idx)-1):
    for j in range(i+1, len(key_idx)-1):
        idx_i = key_idx[i]
        idx_j = key_idx[j]
        
        r_simple = R_simple[idx_i, idx_j]
        r_partial = partial_corr(X_norm, idx_i, idx_j, gdp_idx)
        diff = r_simple - r_partial
        
        print(f"{key_vars[i]} vs {key_vars[j]:<20} {r_simple:>10.3f} {r_partial:>12.3f} {diff:>12.3f}")

# =============================================================================
# 5. Simple vs Partial Correlation Comparison
# =============================================================================
print("\n--- 5. Simple vs Partial Correlation Comparison ---")

diff_matrix = np.abs(R_simple - R_partial_all)
np.fill_diagonal(diff_matrix, 0)

# Find pairs with largest difference
print("\nTop 15 pairs with largest difference (spurious correlations):")
print(f"{'Variable Pair':<50} {'Simple r':>10} {'Partial r':>10} {'Diff':>8}")
print("-" * 80)

# Flatten and sort
indices = np.triu_indices(p, k=1)
diff_values = diff_matrix[indices]
sorted_idx = np.argsort(diff_values)[::-1]

for k in range(15):
    idx = sorted_idx[k]
    i, j = indices[0][idx], indices[1][idx]
    r_s = R_simple[i, j]
    r_p = R_partial_all[i, j]
    d = diff_values[idx]
    
    name_pair = f"{short_names[i]} vs {short_names[j]}"
    print(f"{name_pair:<50} {r_s:>10.3f} {r_p:>10.3f} {d:>8.3f}")

# =============================================================================
# 6. Plot Comparison Heatmaps
# =============================================================================
print("\n--- 6. Plotting Comparison Heatmaps ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Simple Correlation
ax1 = axes[0]
sns.heatmap(R_simple, ax=ax1, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'shrink': 0.6})
ax1.set_title('Simple Correlation Matrix', fontsize=12, fontweight='bold')
ax1.set_xlabel('Indicator Index')
ax1.set_ylabel('Indicator Index')

# Partial Correlation
ax2 = axes[1]
sns.heatmap(R_partial_all, ax=ax2, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={'shrink': 0.6})
ax2.set_title('Partial Correlation Matrix', fontsize=12, fontweight='bold')
ax2.set_xlabel('Indicator Index')
ax2.set_ylabel('Indicator Index')

# Difference
ax3 = axes[2]
sns.heatmap(np.abs(R_simple - R_partial_all), ax=ax3, cmap='YlOrRd',
            square=True, cbar_kws={'shrink': 0.6})
ax3.set_title('|Simple - Partial| Difference', fontsize=12, fontweight='bold')
ax3.set_xlabel('Indicator Index')
ax3.set_ylabel('Indicator Index')

plt.suptitle('Simple Correlation vs Partial Correlation Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q14_partial_correlation_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: Q14_partial_correlation_comparison.png")
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
# =============================================================================
# 7. Direct Relationships from Partial Correlation
# =============================================================================
print("\n--- 7. Direct Relationships (Significant Partial Correlations) ---")

print("\nSignificant partial correlations (|r_partial| > 0.3):")
print(f"{'Variable Pair':<55} {'Partial r':>10}")
print("-" * 65)

significant_count = 0
for i in range(p):
    for j in range(i+1, p):
        if abs(R_partial_all[i, j]) > 0.3:
            print(f"{short_names[i]} <-> {short_names[j]:<30} {R_partial_all[i, j]:>10.3f}")
            significant_count += 1

print(f"\nTotal {significant_count} direct associations found")

# =============================================================================
# 8. Specific Factor Analysis
# =============================================================================
print("\n--- 8. Specific Factor Analysis ---")

target_name = 'AI_Market'
target_idx = short_names.index(target_name)

print(f"\n[{target_name}] relationships with all factors:")
print(f"{'Factor':<35} {'Simple r':>10} {'Partial r':>12} {'Effect Type':<15}")
print("-" * 75)

for i in range(p):
    if i != target_idx:
        r_s = R_simple[target_idx, i]
        r_p = R_partial_all[target_idx, i]
        
        if abs(r_p) > 0.2:
            if np.sign(r_s) == np.sign(r_p):
                effect_type = 'Direct Effect'
            else:
                effect_type = 'Suppression'
        elif abs(r_s) > 0.3 and abs(r_p) < 0.1:
            effect_type = 'Spurious Corr'
        else:
            effect_type = 'Weak Effect'
        
        print(f"{short_names[i]:<35} {r_s:>10.3f} {r_p:>12.3f} {effect_type:<15}")

# =============================================================================
# 9. Dimension-Level Partial Correlation
# =============================================================================
print("\n--- 9. Dimension-Level Partial Correlation ---")

dim_names = ['Infrastructure', 'Talent', 'R&D', 'Industry', 'Policy', 'National']
dim_ranges = [(0, 8), (8, 14), (14, 20), (20, 24), (24, 29), (29, 38)]

# Compute dimension scores
dim_scores = np.zeros((n, 6))
for d, (start, end) in enumerate(dim_ranges):
    dim_scores[:, d] = X_norm[:, start:end].mean(axis=1)

# Simple correlation
R_dim_simple = np.corrcoef(dim_scores.T)

# Partial correlation
R_dim_partial = compute_partial_correlation_matrix(dim_scores, ridge=0.01)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
sns.heatmap(R_dim_simple, ax=ax1, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', square=True,
            xticklabels=dim_names, yticklabels=dim_names,
            cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 9})
ax1.set_title('Inter-Dimension Simple Correlation', fontsize=12, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
ax2 = axes[1]
sns.heatmap(R_dim_partial, ax=ax2, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', square=True,
            xticklabels=dim_names, yticklabels=dim_names,
            cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 9})
ax2.set_title('Inter-Dimension Partial Correlation (Direct)', fontsize=12, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

plt.suptitle('Dimension Correlation: Simple vs Partial', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q14_dimension_partial_correlation.png', bbox_inches='tight')
plt.close()
print("Saved: Q14_dimension_partial_correlation.png")

# =============================================================================
# 10. Effect Classification Summary
# =============================================================================
print("\n--- 10. Effect Classification Summary ---")

direct_count = 0
spurious_count = 0
suppression_count = 0

for i in range(p):
    for j in range(i+1, p):
        r_s = R_simple[i, j]
        r_p = R_partial_all[i, j]
        
        if abs(r_p) > 0.2 and np.sign(r_s) == np.sign(r_p):
            direct_count += 1
        elif abs(r_s) > 0.3 and abs(r_p) < 0.1:
            spurious_count += 1
        elif abs(r_p) > 0.1 and np.sign(r_s) != np.sign(r_p):
            suppression_count += 1

print(f"\nEffect Classification Summary:")
print(f"  Direct Effects (r_s and r_p same sign, |r_p|>0.2): {direct_count}")
print(f"  Spurious Correlations (|r_s|>0.3 but |r_p|<0.1): {spurious_count}")
print(f"  Suppression Effects (r_s and r_p opposite signs): {suppression_count}")

# =============================================================================
# 11. Save Results
# =============================================================================
print("\n--- 11. Saving Results ---")

# Simple correlation
R_simple_df = pd.DataFrame(R_simple, columns=short_names, index=short_names)
R_simple_df.to_csv('Q14_simple_correlation.csv')

# Partial correlation
R_partial_df = pd.DataFrame(R_partial_all, columns=short_names, index=short_names)
R_partial_df.to_csv('Q14_partial_correlation.csv')

# Difference
R_diff_df = pd.DataFrame(R_simple - R_partial_all, columns=short_names, index=short_names)
R_diff_df.to_csv('Q14_correlation_difference.csv')

print("Saved: Q14_simple_correlation.csv")
print("Saved: Q14_partial_correlation.csv")
print("Saved: Q14_correlation_difference.csv")

print("\n" + "=" * 70)
print("Q14 Analysis Complete")
print("=" * 70)

"""
=============================================================================
Huashu Cup 2026 Problem B - Question 2 Q25
CRITIC + TOPSIS Evaluation Model
CRITIC: Criteria Importance Through Intercriteria Correlation
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
print("=" * 70)
print("Q25: CRITIC + TOPSIS Evaluation Model")
print("=" * 70)

# =============================================================================
# 1. Data Loading
# =============================================================================
print("\n--- 1. Data Loading ---")

df = pd.read_csv('panel_data_38indicators.csv')
latest_year = df['Year'].max()
df_2025 = df[df['Year'] == latest_year].copy().reset_index(drop=True)
countries = df_2025['Country'].values

indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
X = df_2025[indicator_cols].values.astype(float)
n, p = X.shape

short_names = ['Top500', 'GPU_Cluster', 'DC_Compute', 'AI_Chips', '5G_Coverage',
    'Internet_BW', 'Data_Centers', 'Internet_Pen', 'AI_Researchers', 'Talent_Flow',
    'Top_Scholars', 'STEM_Grads', 'AI_Papers', 'AI_Labs', 'Gov_Investment',
    'Enterprise_RD', 'VC_Investment', 'Paper_Citations', 'AI_Patents', 'AI_Market',
    'AI_Companies', 'AI_Unicorns', 'Large_Models', 'Industry_Market', 'Tax_Incentive',
    'Subsidy_Amount', 'Policy_Count', 'Subsidy_Intensity', 'Regulatory_FW', 'GDP',
    'GDP_Growth', 'FX_Reserves', 'Population', 'Working_Age', 'Higher_Edu',
    'GII_Rank', 'RD_Density', 'FDI_Inflow']

dim_names = ['Infrastructure', 'Talent', 'R&D', 'Industry', 'Policy', 'National']
dim_ranges = [(0, 8), (8, 14), (14, 20), (20, 24), (24, 29), (29, 38)]

negative_idx = [short_names.index('GII_Rank')]

print(f"Year: {latest_year}, Countries: {n}, Indicators: {p}")

# =============================================================================
# 2. CRITIC Weight Calculation
# =============================================================================
print("\n--- 2. CRITIC Weight Calculation ---")

def critic_weight(X):
    """
    CRITIC (Criteria Importance Through Intercriteria Correlation)
    Considers both:
    1. Contrast Intensity: Standard deviation (variability)
    2. Conflict: Correlation with other indicators (independence)
    
    C_j = σ_j × Σ(1 - r_jk)  for k ≠ j
    Weight_j = C_j / ΣC_j
    """
    n, p = X.shape
    
    # Normalize data (0-1)
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    
    # Standard deviation (contrast intensity)
    sigma = X_norm.std(axis=0)
    
    # Correlation matrix
    corr_matrix = np.corrcoef(X_norm.T)
    
    # Conflict measure: sum of (1 - |correlation|) with other indicators
    conflict = np.zeros(p)
    for j in range(p):
        conflict[j] = np.sum(1 - np.abs(corr_matrix[j, :]))
    # 本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    # Information content
    C = sigma * conflict
    
    # Normalize to weights
    weights = C / C.sum()
    
    return weights, sigma, conflict, C, corr_matrix

weights, sigma, conflict, info_content, corr_matrix = critic_weight(X)

print("\nCRITIC Weight Components:")
print(f"{'Indicator':<20} {'Sigma':<10} {'Conflict':<10} {'Info':<10} {'Weight':<10}")
print("-" * 60)

sorted_idx = np.argsort(weights)[::-1]
for i in sorted_idx[:15]:
    print(f"{short_names[i]:<20} {sigma[i]:<10.4f} {conflict[i]:<10.2f} "
          f"{info_content[i]:<10.4f} {weights[i]:<10.4f}")

print(f"\n... (showing top 15 of {p})")

# =============================================================================
# 3. Analyze Weight Components
# =============================================================================
print("\n--- 3. Weight Component Analysis ---")

# Indicators with high sigma but low conflict (redundant with others)
high_sigma_low_conflict = []
for i in range(p):
    if sigma[i] > np.median(sigma) and conflict[i] < np.median(conflict):
        high_sigma_low_conflict.append(short_names[i])

# Indicators with high conflict (independent information)
high_conflict_idx = np.argsort(conflict)[::-1][:5]
print("Most Independent Indicators (High Conflict):")
for i in high_conflict_idx:
    print(f"  {short_names[i]}: Conflict = {conflict[i]:.2f}")

print("\nMost Variable Indicators (High Sigma):")
high_sigma_idx = np.argsort(sigma)[::-1][:5]
for i in high_sigma_idx:
    print(f"  {short_names[i]}: Sigma = {sigma[i]:.4f}")

# =============================================================================
# 4. TOPSIS Evaluation
# =============================================================================
print("\n--- 4. TOPSIS Evaluation ---")

def topsis(X, weights, negative_idx=[]):
    """TOPSIS method"""
    n, p = X.shape
    
    # Vector normalization
    X_sq_sum = np.sqrt((X ** 2).sum(axis=0))
    X_sq_sum[X_sq_sum == 0] = 1e-10
    R = X / X_sq_sum
    
    # Weighted normalized matrix
    V = R * weights
    
    # Ideal solutions
    V_pos = np.zeros(p)
    V_neg = np.zeros(p)
    
    for j in range(p):
        if j in negative_idx:
            V_pos[j] = V[:, j].min()
            V_neg[j] = V[:, j].max()
        else:
            V_pos[j] = V[:, j].max()
            V_neg[j] = V[:, j].min()
    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    # Distances
    D_pos = np.sqrt(((V - V_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - V_neg) ** 2).sum(axis=1))
    
    # Closeness coefficient
    C = D_neg / (D_pos + D_neg + 1e-10)
    
    return C, D_pos, D_neg

closeness, D_pos, D_neg = topsis(X, weights, negative_idx)

# =============================================================================
# 5. Create Rankings
# =============================================================================
print("\n--- 5. Results ---")
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
scores_norm = (closeness - closeness.min()) / (closeness.max() - closeness.min()) * 100

results = pd.DataFrame({
    'Country': countries,
    'Closeness': closeness,
    'D_Positive': D_pos,
    'D_Negative': D_neg,
    'Normalized_Score': scores_norm
})
results['Rank'] = results['Normalized_Score'].rank(ascending=False).astype(int)
results = results.sort_values('Rank')

print("\n" + "=" * 60)
print("2025 AI Competitiveness Ranking (CRITIC-TOPSIS)")
print("=" * 60)
print(f"\n{'Rank':<6} {'Country':<15} {'Score':<10} {'Closeness':<12}")
print("-" * 45)
for _, row in results.iterrows():
    print(f"{row['Rank']:<6} {row['Country']:<15} {row['Normalized_Score']:.2f}     {row['Closeness']:.4f}")

# =============================================================================
# 6. Compare with Other Methods
# =============================================================================
print("\n--- 6. Weight Method Comparison ---")

# Entropy weights
def entropy_weight(X):
    n, p = X.shape
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    X_norm = np.clip(X_norm, 1e-10, 1)
    X_sum = X_norm.sum(axis=0)
    X_sum[X_sum == 0] = 1e-10
    P = X_norm / X_sum
    k = 1 / np.log(n)
    P_log = np.where(P > 0, P * np.log(P), 0)
    E = -k * P_log.sum(axis=0)
    D = 1 - E
    return D / D.sum()

# CV weights
def cv_weight(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    means[means == 0] = 1e-10
    cv = stds / np.abs(means)
    return cv / cv.sum()

entropy_weights = entropy_weight(X)
cv_weights = cv_weight(X)
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
# Correlations
corr_critic_entropy = np.corrcoef(weights, entropy_weights)[0, 1]
corr_critic_cv = np.corrcoef(weights, cv_weights)[0, 1]
corr_entropy_cv = np.corrcoef(entropy_weights, cv_weights)[0, 1]

print(f"Weight Correlations:")
print(f"  CRITIC vs Entropy: {corr_critic_entropy:.4f}")
print(f"  CRITIC vs CV:      {corr_critic_cv:.4f}")
print(f"  Entropy vs CV:     {corr_entropy_cv:.4f}")

# =============================================================================
# 7. Visualization
# =============================================================================
print("\n--- 7. Generating Visualizations ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 7.1 CRITIC Components
ax1 = axes[0, 0]
ax1.scatter(sigma, conflict, c=weights, cmap='YlOrRd', s=80, alpha=0.7)
for i in sorted_idx[:5]:
    ax1.annotate(short_names[i], (sigma[i], conflict[i]), fontsize=8)
ax1.set_xlabel('Contrast Intensity (σ)')
ax1.set_ylabel('Conflict (Independence)')
ax1.set_title('CRITIC: Sigma vs Conflict\n(Color = Weight)', fontweight='bold')
ax1.grid(alpha=0.3)
plt.colorbar(ax1.collections[0], ax=ax1, label='Weight')

# 7.2 Ranking Bar Chart
ax2 = axes[0, 1]
colors = plt.cm.RdYlGn(results['Normalized_Score'].values / 100)
ax2.barh(range(len(results)), results['Normalized_Score'].values, color=colors)
ax2.set_yticks(range(len(results)))
ax2.set_yticklabels(results['Country'].values)
ax2.set_xlabel('Normalized Score (0-100)')
ax2.set_title('2025 AI Competitiveness Ranking\n(CRITIC-TOPSIS)', fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)
for i, score in enumerate(results['Normalized_Score'].values):
    ax2.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9)

# 7.3 Weight Methods Comparison
ax3 = axes[0, 2]
methods = ['CRITIC', 'Entropy', 'CV']
weights_matrix = np.array([weights, entropy_weights, cv_weights])
im = ax3.imshow(weights_matrix, aspect='auto', cmap='YlOrRd')
ax3.set_yticks(range(3))
ax3.set_yticklabels(methods)
ax3.set_xlabel('Indicator Index')
ax3.set_title('Weight Methods Comparison', fontweight='bold')
plt.colorbar(im, ax=ax3, label='Weight')

# 7.4 Indicator Correlation Heatmap (subset)
ax4 = axes[1, 0]
subset_idx = sorted_idx[:12]
corr_subset = corr_matrix[np.ix_(subset_idx, subset_idx)]
sns.heatmap(corr_subset, ax=ax4, cmap='RdBu_r', center=0, 
            xticklabels=[short_names[i][:8] for i in subset_idx],
            yticklabels=[short_names[i][:8] for i in subset_idx],
            annot=True, fmt='.2f', annot_kws={'fontsize': 7})
ax4.set_title('Correlation Matrix (Top 12 Indicators)', fontweight='bold')

# 7.5 Weight Comparison Scatter
ax5 = axes[1, 1]
ax5.scatter(entropy_weights, weights, alpha=0.7, c='steelblue', s=60, label='CRITIC vs Entropy')
ax5.plot([0, max(weights)], [0, max(weights)], 'r--', label='y=x')
ax5.set_xlabel('Entropy Weight')
ax5.set_ylabel('CRITIC Weight')
ax5.set_title(f'CRITIC vs Entropy Weights\n(Corr={corr_critic_entropy:.3f})', fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# 7.6 Dimension Weights
ax6 = axes[1, 2]
dim_weights_critic = [weights[start:end].sum() for start, end in dim_ranges]
dim_weights_entropy = [entropy_weights[start:end].sum() for start, end in dim_ranges]

x = np.arange(6)
width = 0.35
ax6.bar(x - width/2, dim_weights_critic, width, label='CRITIC', color='steelblue', alpha=0.8)
ax6.bar(x + width/2, dim_weights_entropy, width, label='Entropy', color='coral', alpha=0.8)
ax6.set_xticks(x)
ax6.set_xticklabels(dim_names, rotation=45, ha='right')
ax6.set_ylabel('Dimension Weight')
ax6.set_title('Dimension Weights: CRITIC vs Entropy', fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.suptitle('CRITIC-TOPSIS AI Competitiveness Evaluation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Q25_critic_topsis.png', bbox_inches='tight')
plt.close()
print("Saved: Q25_critic_topsis.png")

# Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist()
angles += angles[:1]

# Dimension scores
dim_scores = np.zeros((n, 6))
for d, (start, end) in enumerate(dim_ranges):
    X_dim = X[:, start:end]
    W_dim = weights[start:end]
    W_dim = W_dim / W_dim.sum()
    neg_dim = [i-start for i in negative_idx if start <= i < end]
    dim_scores[:, d], _, _ = topsis(X_dim, W_dim, neg_dim)

for d in range(6):
    dim_scores[:, d] = (dim_scores[:, d] - dim_scores[:, d].min()) / \
                       (dim_scores[:, d].max() - dim_scores[:, d].min() + 1e-10) * 100

colors = plt.cm.Set1(np.linspace(0, 1, 5))
top5 = results.head(5)['Country'].values

for idx, country in enumerate(top5):
    country_idx = np.where(countries == country)[0][0]
    values = dim_scores[country_idx, :].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=country, color=colors[idx])
    ax.fill(angles, values, alpha=0.1, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dim_names)
ax.set_ylim(0, 100)
ax.set_title('Top 5 Countries Dimension Comparison\n(CRITIC-TOPSIS)', fontweight='bold', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('Q25_radar_chart.png', bbox_inches='tight')
plt.close()
print("Saved: Q25_radar_chart.png")

# =============================================================================
# 8. Save Results
# =============================================================================
print("\n--- 8. Saving Results ---")

results.to_csv('Q25_critic_topsis_ranking.csv', index=False)

weights_df = pd.DataFrame({
    'Indicator': short_names,
    'Sigma': sigma,
    'Conflict': conflict,
    'Info_Content': info_content,
    'CRITIC_Weight': weights,
    'Entropy_Weight': entropy_weights,
    'CV_Weight': cv_weights
})
weights_df.to_csv('Q25_critic_weights.csv', index=False)

print("Saved: Q25_critic_topsis_ranking.csv")
print("Saved: Q25_critic_weights.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q25 CRITIC-TOPSIS SUMMARY")
print("=" * 70)
print(f"""
Method: CRITIC (Criteria Importance Through Intercriteria Correlation) + TOPSIS

CRITIC considers both:
1. Contrast Intensity (σ): Variability of indicator
2. Conflict: Independence from other indicators (1 - |correlation|)

Information Content = σ × Conflict

Key Findings:
- Most informative indicator: {short_names[sorted_idx[0]]} (C={info_content[sorted_idx[0]]:.4f})
- Most independent indicator: {short_names[high_conflict_idx[0]]} (Conflict={conflict[high_conflict_idx[0]]:.2f})
- Most variable indicator: {short_names[high_sigma_idx[0]]} (σ={sigma[high_sigma_idx[0]]:.4f})

Weight Correlations:
- CRITIC vs Entropy: {corr_critic_entropy:.4f}
- CRITIC vs CV: {corr_critic_cv:.4f}

2025 Ranking:
1. {results.iloc[0]['Country']}: {results.iloc[0]['Normalized_Score']:.2f}
2. {results.iloc[1]['Country']}: {results.iloc[1]['Normalized_Score']:.2f}
3. {results.iloc[2]['Country']}: {results.iloc[2]['Normalized_Score']:.2f}

CRITIC Advantages:
- Captures both variability AND independence
- Reduces redundancy from highly correlated indicators
- More comprehensive than entropy or CV alone
""")
print("=" * 70)

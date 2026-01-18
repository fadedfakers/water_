"""
=============================================================================
Huashu Cup 2026 Problem B - Question 2 Q23
Entropy Weight + TOPSIS Evaluation Model
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
print("Q23: Entropy Weight + TOPSIS Evaluation Model")
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
X = df_2025[indicator_cols].values
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

# Negative indicators (lower is better)
negative_idx = [short_names.index('GII_Rank')]

print(f"Year: {latest_year}, Countries: {n}, Indicators: {p}")

# =============================================================================
# 2. Data Normalization (for Entropy)
# =============================================================================
print("\n--- 2. Data Normalization ---")

def normalize_minmax(X, negative_idx=[]):
    """Min-Max normalization with direction handling"""
    n, p = X.shape
    X_norm = np.zeros_like(X, dtype=float)
    
    for j in range(p):
        col = X[:, j].astype(float)
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            if j in negative_idx:
                X_norm[:, j] = (max_val - col) / (max_val - min_val)
            else:
                X_norm[:, j] = (col - min_val) / (max_val - min_val)
        else:
            X_norm[:, j] = 0.5
    
    return np.clip(X_norm, 1e-10, 1)

X_norm = normalize_minmax(X, negative_idx)

# ===#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# # =======================================================================================================================================================
# 3. Entropy Weight Calculation
# =============================================================================
print("\n--- 3. Entropy Weight Calculation ---")

def entropy_weight(X):
    """Calculate entropy weights"""
    n, p = X.shape
    
    # Proportion matrix
    X_sum = X.sum(axis=0)
    X_sum[X_sum == 0] = 1e-10
    P = X / X_sum
    
    # Entropy
    k = 1 / np.log(n)
    P_log = np.where(P > 0, P * np.log(P), 0)
    E = -k * P_log.sum(axis=0)
    
    # Utility and weights
    D = 1 - E
    W = D / D.sum()
    
    return W

weights = entropy_weight(X_norm)

print("Top 10 weights:")
sorted_idx = np.argsort(weights)[::-1]
for i in sorted_idx[:10]:
    print(f"  {short_names[i]}: {weights[i]:.4f}")

# =============================================================================
# 4. TOPSIS Evaluation
# =============================================================================
print("\n--- 4. TOPSIS Evaluation ---")

def topsis(X, weights, negative_idx=[]):
    """
    TOPSIS method
    Returns: closeness coefficient (higher is better)
    """
    n, p = X.shape
    
    # Step 1: Vector normalization
    X_sq_sum = np.sqrt((X ** 2).sum(axis=0))
    X_sq_sum[X_sq_sum == 0] = 1e-10
    R = X / X_sq_sum
    
    # Step 2: Weighted normalized matrix
    V = R * weights
    
    # Step 3: Ideal solutions
    V_pos = np.zeros(p)  # Positive ideal
    V_neg = np.zeros(p)  # Negative ideal
    
    for j in range(p):
        if j in negative_idx:
            V_pos[j] = V[:, j].min()
            V_neg[j] = V[:, j].max()
        else:
            V_pos[j] = V[:, j].max()
            V_neg[j] = V[:, j].min()
    
    # Step 4: Distance calculation
    D_pos = np.sqrt(((V - V_pos) ** 2).sum(axis=1))
    D_neg = np.sqrt(((V - V_neg) ** 2).sum(axis=1))
    
    # Step 5: Closeness coefficient
    C = D_neg / (D_pos + D_neg + 1e-10)
    
    return C, D_pos, D_neg, V

closeness, D_pos, D_neg, V = topsis(X, weights, negative_idx)

# =============================================================================
# 5. Create Rankings
# =============================================================================
print("\n--- 5. Results ---")

# Normalize to 0-100
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
print("2025 AI Competitiveness Ranking (Entropy-TOPSIS)")
print("=" * 60)
print(f"\n{'Rank':<6} {'Country':<15} {'Score':<10} {'Closeness':<12} {'D+':<10} {'D-':<10}")
print("-" * 65)
for _, row in results.iterrows():
    print(f"{row['Rank']:<6} {row['Country']:<15} {row['Normalized_Score']:.2f}     "
          f"{row['Closeness']:.4f}      {row['D_Positive']:.4f}    {row['D_Negative']:.4f}")

# =============================================================================
# 6. Dimension Analysis
# =============================================================================
print("\n--- 6. Dimension Analysis ---")

dim_scores = np.zeros((n, 6))
for d, (start, end) in enumerate(dim_ranges):
    # TOPSIS for each dimension
    X_dim = X[:, start:end]
    W_dim = weights[start:end]
    W_dim = W_dim / W_dim.sum()  # Re-normalize
    neg_dim = [i-start for i in negative_idx if start <= i < end]
    dim_scores[:, d], _, _, _ = topsis(X_dim, W_dim, neg_dim)

# Normalize dimension scores
for d in range(6):
    dim_scores[:, d] = (dim_scores[:, d] - dim_scores[:, d].min()) / \
                       (dim_scores[:, d].max() - dim_scores[:, d].min() + 1e-10) * 100

dim_df = pd.DataFrame(dim_scores, columns=dim_names)
dim_df['Country'] = countries
print("\nDimension Scores:")
print(dim_df.to_string(index=False))

# =============================================================================
# 7. Visualization
# =============================================================================
print("\n--- 7. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 7.1 Ranking Bar Chart
ax1 = axes[0, 0]
colors = plt.cm.RdYlGn(results['Normalized_Score'].values / 100)
bars = ax1.barh(range(len(results)), results['Normalized_Score'].values, color=colors)
ax1.set_yticks(range(len(results)))
ax1.set_yticklabels(results['Country'].values)
ax1.set_xlabel('Normalized Score (0-100)')
ax1.set_title('2025 AI Competitiveness Ranking\n(Entropy-TOPSIS)', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)
for i, score in enumerate(results['Normalized_Score'].values):
    ax1.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9)

# 7.2 Distance Comparison
ax2 = axes[0, 1]
x = np.arange(len(results))
width = 0.35
ax2.bar(x - width/2, results['D_Positive'].values, width, label='D+ (to Ideal)', color='green', alpha=0.7)
ax2.bar(x + width/2, results['D_Negative'].values, width, label='D- (to Anti-Ideal)', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(results['Country'].values, rotation=45, ha='right')
ax2.set_ylabel('Distance')
ax2.set_title('Distance to Ideal and Anti-Ideal Solutions', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 7.3 Heatmap of Dimension Scores
ax3 = axes[1, 0]
dim_df_sorted = dim_df.set_index('Country').loc[results['Country'].values]
sns.heatmap(dim_df_sorted, ax=ax3, cmap='RdYlGn', annot=True, fmt='.1f',
            cbar_kws={'label': 'Score'})
ax3.set_title('Dimension Scores Heatmap', fontweight='bold')
ax3.set_ylabel('')

# 7.4 Closeness Coefficient
ax4 = axes[1, 1]
ax4.bar(range(len(results)), results['Closeness'].values, color='steelblue', alpha=0.8)
ax4.set_xticks(range(len(results)))
ax4.set_xticklabels(results['Country'].values, rotation=45, ha='right')
ax4.set_ylabel('Closeness Coefficient')
ax4.set_title('TOPSIS Closeness Coefficient (C)', fontweight='bold')
ax4.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Entropy-TOPSIS AI Competitiveness Evaluation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Q23_entropy_topsis.png', bbox_inches='tight')
plt.close()
print("Saved: Q23_entropy_topsis.png")

# Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist()
angles += angles[:1]
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
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
ax.set_title('Top 5 Countries Dimension Comparison\n(Entropy-TOPSIS)', fontweight='bold', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('Q23_radar_chart.png', bbox_inches='tight')
plt.close()
print("Saved: Q23_radar_chart.png")

# =============================================================================
# 8. Save Results
# =============================================================================
print("\n--- 8. Saving Results ---")

results.to_csv('Q23_entropy_topsis_ranking.csv', index=False)
dim_df.to_csv('Q23_dimension_scores.csv', index=False)

weights_df = pd.DataFrame({'Indicator': short_names, 'Weight': weights})
weights_df.to_csv('Q23_weights.csv', index=False)

print("Saved: Q23_entropy_topsis_ranking.csv")
print("Saved: Q23_dimension_scores.csv")
print("Saved: Q23_weights.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q23 ENTROPY-TOPSIS SUMMARY")
print("=" * 70)
print(f"""
Method: Entropy Weight + TOPSIS
- Entropy for objective weighting
- TOPSIS for ranking based on ideal solution distance

2025 Ranking:
1. {results.iloc[0]['Country']}: {results.iloc[0]['Normalized_Score']:.2f} (C={results.iloc[0]['Closeness']:.4f})
2. {results.iloc[1]['Country']}: {results.iloc[1]['Normalized_Score']:.2f} (C={results.iloc[1]['Closeness']:.4f})
3. {results.iloc[2]['Country']}: {results.iloc[2]['Normalized_Score']:.2f} (C={results.iloc[2]['Closeness']:.4f})
""")
print("=" * 70)
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
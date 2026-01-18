"""
=============================================================================
Huashu Cup 2026 Problem B - Question 2 Q24
Coefficient of Variation (CV) + TOPSIS Evaluation Model
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

print("=" * 70)
print("Q24: Coefficient of Variation + TOPSIS Evaluation Model")
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
# 2. Coefficient of Variation Weight Calculation
# =============================================================================
print("\n--- 2. Coefficient of Variation Weight Calculation ---")

def cv_weight(X):
    """
    Calculate weights using Coefficient of Variation method
    CV = std / mean (measures relative variability)
    Higher CV -> more discriminative -> higher weight
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    
    # Avoid division by zero
    means[means == 0] = 1e-10
    
    # Coefficient of variation
    cv = stds / np.abs(means)
    
    # Normalize to weights
    weights = cv / cv.sum()
    
    return weights, cv, means, stds

weights, cv_values, means, stds = cv_weight(X)

print("\nCoefficient of Variation Results:")
print(f"{'Indicator':<20} {'Mean':<12} {'Std':<12} {'CV':<12} {'Weight':<12}")
print("-" * 68)

sorted_idx = np.argsort(weights)[::-1]
for i in sorted_idx[:15]:
    print(f"{short_names[i]:<20} {means[i]:<12.2f} {stds[i]:<12.2f} {cv_values[i]:<12.4f} {weights[i]:<12.4f}")

print(f"\nCV range: [{cv_values.min():.4f}, {cv_values.max():.4f}]")

# =============================================================================
# 3. Compare with Entropy Weights
# =============================================================================
print("\n--- 3. Compare CV vs Entropy Weights ---")

# Calculate entropy weights for comparison
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

entropy_weights = entropy_weight(X)

# Correlation between CV and entropy weights
corr = np.corrcoef(weights, entropy_weights)[0, 1]
print(f"Correlation between CV and Entropy weights: {corr:.4f}")

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
print("2025 AI Competitiveness Ranking (CV-TOPSIS)")
print("=" * 60)
print(f"\n{'Rank':<6} {'Country':<15} {'Score':<10} {'Closeness':<12}")
print("-" * 45)
for _, row in results.iterrows():
    print(f"{row['Rank']:<6} {row['Country']:<15} {row['Normalized_Score']:.2f}     {row['Closeness']:.4f}")

# =============================================================================
# 6. Dimension Analysis
# =============================================================================
print("\n--- 6. Dimension Analysis ---")

# Dimension weights (sum of indicator weights in each dimension)
dim_weights = []
for start, end in dim_ranges:
    dim_weights.append(weights[start:end].sum())

print("\nDimension Weights (CV Method):")
for d, name in enumerate(dim_names):
    print(f"  {name}: {dim_weights[d]*100:.2f}%")

# Dimension TOPSIS scores
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

# =============================================================================
# 7. Visualization
# =============================================================================
print("\n--- 7. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 7.1 CV Values vs Weights
ax1 = axes[0, 0]
ax1.scatter(cv_values, weights, alpha=0.7, c='steelblue', s=60)
for i in sorted_idx[:5]:
    ax1.annotate(short_names[i], (cv_values[i], weights[i]), fontsize=8)
ax1.set_xlabel('Coefficient of Variation')
ax1.set_ylabel('Weight')
ax1.set_title('CV Values vs Weights', fontweight='bold')
ax1.grid(alpha=0.3)

# Add regression line
z = np.polyfit(cv_values, weights, 1)
p_line = np.poly1d(z)
ax1.plot(cv_values, p_line(cv_values), "r--", alpha=0.8, label='Linear fit')
ax1.legend()

# 7.2 Ranking Bar Chart
ax2 = axes[0, 1]
colors = plt.cm.RdYlGn(results['Normalized_Score'].values / 100)
ax2.barh(range(len(results)), results['Normalized_Score'].values, color=colors)
ax2.set_yticks(range(len(results)))
ax2.set_yticklabels(results['Country'].values)
ax2.set_xlabel('Normalized Score (0-100)')
ax2.set_title('2025 AI Competitiveness Ranking\n(CV-TOPSIS)', fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)
for i, score in enumerate(results['Normalized_Score'].values):
    ax2.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9)

# 7.3 CV vs Entropy Weights Comparison
ax3 = axes[1, 0]
x = np.arange(p)
width = 0.35
ax3.bar(x - width/2, weights, width, label='CV Weight', alpha=0.7)
ax3.bar(x + width/2, entropy_weights, width, label='Entropy Weight', alpha=0.7)
ax3.set_xlabel('Indicator Index')
ax3.set_ylabel('Weight')
ax3.set_title(f'CV vs Entropy Weights (Corr={corr:.3f})', fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 7.4 Dimension Weights Comparison
ax4 = axes[1, 1]
x = np.arange(6)
width = 0.35

# Entropy dimension weights
entropy_dim_weights = []
for start, end in dim_ranges:
    entropy_dim_weights.append(entropy_weights[start:end].sum())

ax4.bar(x - width/2, dim_weights, width, label='CV Method', color='steelblue', alpha=0.8)
ax4.bar(x + width/2, entropy_dim_weights, width, label='Entropy Method', color='coral', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(dim_names, rotation=45, ha='right')
ax4.set_ylabel('Dimension Weight')
ax4.set_title('Dimension Weights: CV vs Entropy', fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('CV-TOPSIS AI Competitiveness Evaluation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Q24_cv_topsis.png', bbox_inches='tight')
plt.close()
print("Saved: Q24_cv_topsis.png")

# Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist()
angles += angles[:1]

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
ax.set_title('Top 5 Countries Dimension Comparison\n(CV-TOPSIS)', fontweight='bold', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('Q24_radar_chart.png', bbox_inches='tight')
plt.close()
print("Saved: Q24_radar_chart.png")

# =============================================================================
# 8. Save Results
# =============================================================================
print("\n--- 8. Saving Results ---")

results.to_csv('Q24_cv_topsis_ranking.csv', index=False)

weights_df = pd.DataFrame({
    'Indicator': short_names,
    'Mean': means,
    'Std': stds,
    'CV': cv_values,
    'CV_Weight': weights,
    'Entropy_Weight': entropy_weights
})
weights_df.to_csv('Q24_cv_weights.csv', index=False)

dim_df = pd.DataFrame(dim_scores, columns=dim_names)
dim_df['Country'] = countries
dim_df.to_csv('Q24_dimension_scores.csv', index=False)

print("Saved: Q24_cv_topsis_ranking.csv")
print("Saved: Q24_cv_weights.csv")
print("Saved: Q24_dimension_scores.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q24 CV-TOPSIS SUMMARY")
print("=" * 70)
print(f"""
Method: Coefficient of Variation + TOPSIS
- CV measures relative variability (std/mean)
- Higher CV -> more discriminative power -> higher weight

Key Findings:
- Highest CV indicator: {short_names[sorted_idx[0]]} (CV={cv_values[sorted_idx[0]]:.4f})
- Lowest CV indicator: {short_names[sorted_idx[-1]]} (CV={cv_values[sorted_idx[-1]]:.4f})
- CV-Entropy weight correlation: {corr:.4f}

2025 Ranking:
1. {results.iloc[0]['Country']}: {results.iloc[0]['Normalized_Score']:.2f}
2. {results.iloc[1]['Country']}: {results.iloc[1]['Normalized_Score']:.2f}
3. {results.iloc[2]['Country']}: {results.iloc[2]['Normalized_Score']:.2f}

CV Method Characteristics:
- Favors indicators with high relative variability
- Simple and intuitive interpretation
- May overweight noisy indicators
""")
print("=" * 70)

"""
=============================================================================
Huashu Cup 2026 Problem B - Question 2 Q27
PCA-Entropy-Grey Relational Analysis Combination Model
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("Q27: PCA-Entropy-Grey Relational Analysis Combination Model")
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
# 2. PCA Analysis
# =============================================================================
print("\n--- 2. PCA Analysis ---")

# Standardize data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Handle negative indicators by reversing sign
for idx in negative_idx:
    X_std[:, idx] = -X_std[:, idx]

# PCA
pca = PCA()
pca.fit(X_std)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# Select components (cumulative variance >= 80%)
n_components = np.argmax(cumulative_var >= 0.80) + 1
n_components = max(n_components, 3)

print(f"Variance Explained:")
for i in range(min(10, p)):
    print(f"  PC{i+1}: {explained_var[i]*100:.2f}% (Cumulative: {cumulative_var[i]*100:.2f}%)")

print(f"\nSelected components: {n_components} (Cumulative: {cumulative_var[n_components-1]*100:.2f}%)")

# Get PC scores
pca_selected = PCA(n_components=n_components)
PC_scores = pca_selected.fit_transform(X_std)

# PCA weighted score (variance contribution as weights)
pca_weights = explained_var[:n_components] / explained_var[:n_components].sum()
pca_composite = PC_scores @ pca_weights
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
print("\nPCA Weights:")
for i in range(n_components):
    print(f"  PC{i+1}: {pca_weights[i]*100:.2f}%")

# =============================================================================
# 3. Entropy Weight on PC Scores
# =============================================================================
print("\n--- 3. Entropy Weight on PC Scores ---")

def entropy_weight(X):
    """Calculate entropy weights"""
    n, m = X.shape
    
    # Normalize to positive
    X_shifted = X - X.min(axis=0) + 1e-10
    
    # Proportion
    X_sum = X_shifted.sum(axis=0)
    P = X_shifted / X_sum
    
    # Entropy
    k = 1 / np.log(n)
    P_log = np.where(P > 0, P * np.log(P), 0)
    E = -k * P_log.sum(axis=0)
    
    # Utility and weights
    D = 1 - E
    W = D / D.sum()
    
    return W

pc_entropy_weights = entropy_weight(PC_scores)

print("Entropy Weights for PCs:")
for i in range(n_components):
    print(f"  PC{i+1}: {pc_entropy_weights[i]*100:.2f}%")

# Combined PC-Entropy Score
pc_entropy_composite = PC_scores @ pc_entropy_weights

# =============================================================================
# 4. Grey Relational Analysis (GRA)
# =============================================================================
print("\n--- 4. Grey Relational Analysis ---")

def grey_relational_analysis(X, reference='max', rho=0.5, weights=None):
    """
    Grey Relational Analysis
    
    Parameters:
    - X: normalized data matrix
    - reference: 'max' for best values, or specific reference series
    - rho: distinguishing coefficient (0-1), typically 0.5
    - weights: indicator weights (optional)
    
    Returns:
    - Grey relational grades
    """
    n, p = X.shape
    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    # Reference series (ideal values)
    if reference == 'max':
        X_ref = X.max(axis=0)
    elif reference == 'min':
        X_ref = X.min(axis=0)
    else:
        X_ref = reference
    
    # Absolute differences
    delta = np.abs(X - X_ref)
    
    # Min and max differences
    delta_min = delta.min()
    delta_max = delta.max()
    
    # Grey relational coefficients
    xi = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-10)
    
    # Grey relational grades
    if weights is None:
        weights = np.ones(p) / p
    
    gamma = (xi * weights).sum(axis=1)
    
    return gamma, xi

# Normalize original data for GRA
X_norm = np.zeros_like(X, dtype=float)
for j in range(p):
    col = X[:, j]
    min_val, max_val = col.min(), col.max()
    if max_val > min_val:
        if j in negative_idx:
            X_norm[:, j] = (max_val - col) / (max_val - min_val)
        else:
            X_norm[:, j] = (col - min_val) / (max_val - min_val)
    else:
        X_norm[:, j] = 0.5

# Calculate entropy weights for GRA
def entropy_weight_norm(X):
    n, p = X.shape
    X_clip = np.clip(X, 1e-10, 1)
    X_sum = X_clip.sum(axis=0)
    X_sum[X_sum == 0] = 1e-10
    P = X_clip / X_sum
    k = 1 / np.log(n)
    P_log = np.where(P > 0, P * np.log(P), 0)
    E = -k * P_log.sum(axis=0)
    D = 1 - E
    return D / D.sum()

gra_weights = entropy_weight_norm(X_norm)
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
# GRA
gra_grades, gra_coefficients = grey_relational_analysis(X_norm, reference='max', rho=0.5, weights=gra_weights)

print("Grey Relational Analysis Results:")
print(f"  Reference: Ideal (max for positive, min for negative indicators)")
print(f"  Distinguishing coefficient (ρ): 0.5")
print(f"  Weighting: Entropy weights")

# =============================================================================
# 5. Combine Three Methods
# =============================================================================
print("\n--- 5. Combine Three Methods ---")

# Normalize all scores to 0-100
def normalize_100(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10) * 100

score_pca = normalize_100(pca_composite)
score_pc_entropy = normalize_100(pc_entropy_composite)
score_gra = normalize_100(gra_grades)

# Borda Count for rank aggregation
def borda_count(scores_list):
    """Borda count for rank aggregation"""
    n = len(scores_list[0])
    borda_scores = np.zeros(n)
    
    for scores in scores_list:
        ranks = pd.Series(scores).rank(ascending=False).values
        borda_scores += (n - ranks)  # Higher rank = higher Borda score
    
    return borda_scores

borda_scores = borda_count([score_pca, score_pc_entropy, score_gra])

# Simple average combination
combined_score = (score_pca + score_pc_entropy + score_gra) / 3

# Weighted combination (can adjust weights based on method reliability)
method_weights = [0.35, 0.35, 0.30]  # PCA, PC-Entropy, GRA
weighted_combined = (method_weights[0] * score_pca + 
                     method_weights[1] * score_pc_entropy + 
                     method_weights[2] * score_gra)

# =============================================================================
# 6. Create Rankings
# =============================================================================
print("\n--- 6. Results ---")

results = pd.DataFrame({
    'Country': countries,
    'PCA_Score': score_pca,
    'PC_Entropy_Score': score_pc_entropy,
    'GRA_Score': score_gra,
    'Average_Score': combined_score,
    'Weighted_Score': weighted_combined,
    'Borda_Score': borda_scores
})

# Rankings for each method
results['Rank_PCA'] = results['PCA_Score'].rank(ascending=False).astype(int)
results['Rank_PC_Entropy'] = results['PC_Entropy_Score'].rank(ascending=False).astype(int)
results['Rank_GRA'] = results['GRA_Score'].rank(ascending=False).astype(int)
results['Rank_Average'] = results['Average_Score'].rank(ascending=False).astype(int)
results['Rank_Borda'] = results['Borda_Score'].rank(ascending=False).astype(int)

results = results.sort_values('Rank_Average')

print("\n" + "=" * 85)
print("2025 AI Competitiveness Ranking (PCA-Entropy-GRA Combination)")
print("=" * 85)
print(f"\n{'Country':<15} {'PCA':<8} {'PC-Ent':<8} {'GRA':<8} {'Average':<10} {'R_PCA':<6} {'R_Ent':<6} {'R_GRA':<6} {'R_Avg':<6}")
print("-" * 85)
for _, row in results.iterrows():
    print(f"{row['Country']:<15} {row['PCA_Score']:.2f}   {row['PC_Entropy_Score']:.2f}   "
          f"{row['GRA_Score']:.2f}   {row['Average_Score']:.2f}      "
          f"{row['Rank_PCA']:<6} {row['Rank_PC_Entropy']:<6} {row['Rank_GRA']:<6} {row['Rank_Average']:<6}")

# =============================================================================
# 7. Rank Consistency Check
# =============================================================================
print("\n--- 7. Rank Consistency Check ---")

from scipy.stats import spearmanr, kendalltau

# Spearman correlations
rank_cols = ['Rank_PCA', 'Rank_PC_Entropy', 'Rank_GRA']
print("\nSpearman Rank Correlations:")
for i in range(len(rank_cols)):
    for j in range(i+1, len(rank_cols)):
        corr, pval = spearmanr(results[rank_cols[i]], results[rank_cols[j]])
        print(f"  {rank_cols[i]} vs {rank_cols[j]}: ρ = {corr:.4f} (p = {pval:.4f})")

# Kendall's W (coefficient of concordance)
def kendall_w(rankings_matrix):
    """Calculate Kendall's W coefficient of concordance"""
    k, n = rankings_matrix.shape  # k raters, n items
    
    # Sum of ranks for each item
    R = rankings_matrix.sum(axis=0)
    
    # Mean of rank sums
    R_mean = R.mean()
    
    # S statistic
    S = ((R - R_mean) ** 2).sum()
    
    # Kendall's W
    W = 12 * S / (k**2 * (n**3 - n))
    
    return W

rankings_matrix = np.array([
    results['Rank_PCA'].values,
    results['Rank_PC_Entropy'].values,
    results['Rank_GRA'].values
])

kendall_w_value = kendall_w(rankings_matrix)
print(f"\nKendall's W (Concordance): {kendall_w_value:.4f}")
print(f"  Interpretation: {'High agreement' if kendall_w_value > 0.7 else 'Moderate agreement' if kendall_w_value > 0.5 else 'Low agreement'}")

# =============================================================================
# 8. Visualization
# =============================================================================
print("\n--- 8. Generating Visualizations ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 8.1 PCA Scree Plot
ax1 = axes[0, 0]
n_show = min(15, len(explained_var))  # Use actual length of explained_var
x = range(1, n_show + 1)
ax1.bar(x, explained_var[:n_show]*100, alpha=0.7, color='steelblue', label='Individual')
ax1.plot(x, cumulative_var[:n_show]*100, 'ro-', linewidth=2, markersize=6, label='Cumulative')
ax1.axhline(y=80, color='green', linestyle='--', label='80% threshold')
ax1.axvline(x=n_components, color='red', linestyle='--', label=f'Selected: {n_components}')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained (%)')
ax1.set_title('PCA Scree Plot', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 8.2 Three Methods Comparison
ax2 = axes[0, 1]
x = np.arange(len(results))
width = 0.25
ax2.bar(x - width, results['PCA_Score'].values, width, label='PCA', alpha=0.8)
ax2.bar(x, results['PC_Entropy_Score'].values, width, label='PC-Entropy', alpha=0.8)
ax2.bar(x + width, results['GRA_Score'].values, width, label='GRA', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(results['Country'].values, rotation=45, ha='right')
ax2.set_ylabel('Score (0-100)')
ax2.set_title('Three Methods Score Comparison', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 8.3 Final Ranking
ax3 = axes[0, 2]
colors = plt.cm.RdYlGn(results['Average_Score'].values / 100)
ax3.barh(range(len(results)), results['Average_Score'].values, color=colors)
ax3.set_yticks(range(len(results)))
ax3.set_yticklabels(results['Country'].values)
ax3.set_xlabel('Average Score (0-100)')
ax3.set_title('2025 AI Competitiveness Ranking\n(PCA-Entropy-GRA Average)', fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)
for i, score in enumerate(results['Average_Score'].values):
    ax3.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9)

# 8.4 Rank Comparison Heatmap
ax4 = axes[1, 0]
rank_data = results[['Country', 'Rank_PCA', 'Rank_PC_Entropy', 'Rank_GRA', 'Rank_Average']].set_index('Country')
rank_data.columns = ['PCA', 'PC-Entropy', 'GRA', 'Average']
sns.heatmap(rank_data, ax=ax4, cmap='RdYlGn_r', annot=True, fmt='d',
            cbar_kws={'label': 'Rank'})
ax4.set_title('Rank Comparison Across Methods', fontweight='bold')

# 8.5 GRA Coefficients Heatmap (Top indicators)
ax5 = axes[1, 1]
top_indicators = np.argsort(gra_weights)[::-1][:15]
gra_subset = gra_coefficients[:, top_indicators]
sns.heatmap(gra_subset, ax=ax5, cmap='YlOrRd',
            xticklabels=[short_names[i][:10] for i in top_indicators],
            yticklabels=countries)
ax5.set_title('Grey Relational Coefficients\n(Top 15 Weighted Indicators)', fontweight='bold')
ax5.set_xlabel('Indicator')

# 8.6 PC1 vs PC2 Scatter
ax6 = axes[1, 2]
scatter = ax6.scatter(PC_scores[:, 0], PC_scores[:, 1], 
                      c=results['Average_Score'].values, cmap='RdYlGn',
                      s=200, edgecolors='black', linewidths=1)
for i, country in enumerate(countries):
    ax6.annotate(country, (PC_scores[i, 0], PC_scores[i, 1]),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)
ax6.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
ax6.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
ax6.set_title('Country Positions in PC Space', fontweight='bold')
ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax6.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax6.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='Average Score')

plt.suptitle('PCA-Entropy-GRA AI Competitiveness Evaluation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Q27_pca_entropy_gra.png', bbox_inches='tight')
plt.close()
print("Saved: Q27_pca_entropy_gra.png")

# Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist()
angles += angles[:1]

# Calculate dimension scores using GRA
dim_scores = np.zeros((n, 6))
for d, (start, end) in enumerate(dim_ranges):
    X_dim = X_norm[:, start:end]
    W_dim = gra_weights[start:end]
    W_dim = W_dim / W_dim.sum()
    dim_scores[:, d], _ = grey_relational_analysis(X_dim, reference='max', rho=0.5, weights=W_dim)

for d in range(6):
    dim_scores[:, d] = normalize_100(dim_scores[:, d])

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
ax.set_title('Top 5 Countries Dimension Comparison\n(GRA-based)', fontweight='bold', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('Q27_radar_chart.png', bbox_inches='tight')
plt.close()
print("Saved: Q27_radar_chart.png")

# =============================================================================
# 9. Save Results
# =============================================================================
print("\n--- 9. Saving Results ---")

results.to_csv('Q27_pca_entropy_gra_ranking.csv', index=False)

# PCA loadings
loadings = pca_selected.components_.T * np.sqrt(pca_selected.explained_variance_)
loadings_df = pd.DataFrame(loadings, index=short_names, 
                           columns=[f'PC{i+1}' for i in range(n_components)])
loadings_df.to_csv('Q27_pca_loadings.csv')

# GRA weights
gra_weights_df = pd.DataFrame({'Indicator': short_names, 'GRA_Weight': gra_weights})
gra_weights_df.to_csv('Q27_gra_weights.csv', index=False)

# Dimension scores
dim_df = pd.DataFrame(dim_scores, columns=dim_names)
dim_df['Country'] = countries
dim_df.to_csv('Q27_dimension_scores.csv', index=False)

print("Saved: Q27_pca_entropy_gra_ranking.csv")
print("Saved: Q27_pca_loadings.csv")
print("Saved: Q27_gra_weights.csv")
print("Saved: Q27_dimension_scores.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q27 PCA-ENTROPY-GRA SUMMARY")
print("=" * 70)
print(f"""
Method Combination:
1. PCA: Dimensionality reduction, variance-weighted scoring
2. PC-Entropy: Entropy weights on principal components
3. GRA: Grey relational analysis with entropy-weighted indicators

PCA Results:
- Components selected: {n_components}
- Variance explained: {cumulative_var[n_components-1]*100:.2f}%

Rank Consistency:
- Kendall's W: {kendall_w_value:.4f} ({'High' if kendall_w_value > 0.7 else 'Moderate' if kendall_w_value > 0.5 else 'Low'} agreement)

2025 Ranking (Average):
1. {results.iloc[0]['Country']}: {results.iloc[0]['Average_Score']:.2f}
2. {results.iloc[1]['Country']}: {results.iloc[1]['Average_Score']:.2f}
3. {results.iloc[2]['Country']}: {results.iloc[2]['Average_Score']:.2f}

Advantages:
- PCA reduces multicollinearity
- GRA handles uncertainty and small samples
- Multiple methods cross-validate results
""")
print("=" * 70)

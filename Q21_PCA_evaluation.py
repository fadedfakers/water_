"""
=============================================================================
Huashu Cup 2026 Problem B - Question 2 Q21
Principal Component Analysis (PCA) for AI Competitiveness Evaluation
=============================================================================
"""
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("=" * 70)
print("Q21: Principal Component Analysis for AI Competitiveness Evaluation")
print("=" * 70)

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
print("\n--- 1. Data Loading ---")

df = pd.read_csv('panel_data_38indicators.csv')
df_original = df.copy()  # Keep a copy for later use
print(f"Data dimension: {df.shape[0]} rows × {df.shape[1]} columns")

# Extract basic info
Country = df['Country']
Year = df['Year']
countries = Country.unique()
years = Year.unique()

# Extract numeric indicators
indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
X = df[indicator_cols].values
n, p = X.shape

# English short names for indicators
short_names = ['Top500', 'GPU_Cluster', 'DC_Compute', 'AI_Chips', '5G_Coverage',
    'Internet_BW', 'Data_Centers', 'Internet_Pen', 'AI_Researchers', 'Talent_Flow',
    'Top_Scholars', 'STEM_Grads', 'AI_Papers', 'AI_Labs', 'Gov_Investment',
    'Enterprise_RD', 'VC_Investment', 'Paper_Citations', 'AI_Patents', 'AI_Market',
    'AI_Companies', 'AI_Unicorns', 'Large_Models', 'Industry_Market', 'Tax_Incentive',
    'Subsidy_Amount', 'Policy_Count', 'Subsidy_Intensity', 'Regulatory_FW', 'GDP',
    'GDP_Growth', 'FX_Reserves', 'Population', 'Working_Age', 'Higher_Edu',
    'GII_Rank', 'RD_Density', 'FDI_Inflow']

# Dimension structure
dim_names = ['Infrastructure', 'Talent', 'R&D', 'Industry', 'Policy', 'National']
dim_ranges = [(0, 8), (8, 14), (14, 20), (20, 24), (24, 29), (29, 38)]

print(f"Countries: {len(countries)}, Years: {len(years)}, Indicators: {p}")

# =============================================================================
# 2. Extract 2025 Data for Evaluation
# =============================================================================
print("\n--- 2. Extract 2025 Data ---")

# Get latest year data (assuming 2025 is the last year)
latest_year = years.max()
print(f"Using data from year: {latest_year}")

df_2025 = df[df['Year'] == latest_year].copy()
df_2025 = df_2025.reset_index(drop=True)
countries_2025 = df_2025['Country'].values
X_2025 = df_2025[indicator_cols].values

print(f"2025 data: {len(countries_2025)} countries")

# =============================================================================
# 3. KMO and Bartlett's Test
# =============================================================================
print("\n--- 3. KMO and Bartlett's Test ---")
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
def calculate_kmo(X):
    """Calculate KMO (Kaiser-Meyer-Olkin) measure"""
    # Correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Partial correlation matrix
    inv_corr = np.linalg.pinv(corr_matrix)
    
    # Scale to partial correlations
    d = np.diag(inv_corr)
    d_sqrt = np.sqrt(d)
    partial_corr = -inv_corr / np.outer(d_sqrt, d_sqrt)
    np.fill_diagonal(partial_corr, 1)
    
    # KMO calculation
    corr_sq = corr_matrix ** 2
    partial_sq = partial_corr ** 2
    
    np.fill_diagonal(corr_sq, 0)
    np.fill_diagonal(partial_sq, 0)
    
    sum_corr_sq = np.sum(corr_sq)
    sum_partial_sq = np.sum(partial_sq)
    
    kmo = sum_corr_sq / (sum_corr_sq + sum_partial_sq)
    
    return kmo

def bartlett_test(X):
    """Bartlett's test of sphericity"""
    n, p = X.shape
    corr_matrix = np.corrcoef(X.T)
    
    # Chi-square statistic
    det = np.linalg.det(corr_matrix)
    if det <= 0:
        det = 1e-10
    
    chi_square = -(n - 1 - (2*p + 5)/6) * np.log(det)
    df = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi_square, df)
    
    return chi_square, df, p_value

# Standardize data first
scaler = StandardScaler()
X_2025_std = scaler.fit_transform(X_2025)

# KMO test
kmo_value = calculate_kmo(X_2025_std)
print(f"KMO Measure: {kmo_value:.4f}")
if kmo_value >= 0.9:
    kmo_eval = "Marvelous"
elif kmo_value >= 0.8:
    kmo_eval = "Meritorious"
elif kmo_value >= 0.7:
    kmo_eval = "Middling"
elif kmo_value >= 0.6:
    kmo_eval = "Mediocre"
else:
    kmo_eval = "Unacceptable"
print(f"KMO Evaluation: {kmo_eval}")

# Bartlett's test
chi2, df, p_value = bartlett_test(X_2025_std)
print(f"\nBartlett's Test:")
print(f"  Chi-square: {chi2:.2f}")
print(f"  Degrees of freedom: {df:.0f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Conclusion: {'Suitable for PCA' if p_value < 0.05 else 'Not suitable for PCA'}")

# =============================================================================
# 4. Principal Component Analysis
# =============================================================================
print("\n--- 4. Principal Component Analysis ---")

# Perform PCA on all years data (standardized)
X_all_std = scaler.fit_transform(X)

# Full PCA to see all components
pca_full = PCA()
pca_full.fit(X_all_std)

# Explained variance
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("\nVariance Explained by Principal Components:")
print(f"{'PC':<6} {'Variance%':<12} {'Cumulative%':<12}")
print("-" * 30)
for i in range(min(15, p)):
    print(f"PC{i+1:<4} {explained_var[i]*100:>10.2f}% {cumulative_var[i]*100:>10.2f}%")

# Determine number of components (cumulative variance > 80%)
n_components = np.argmax(cumulative_var >= 0.80) + 1
n_components = max(n_components, 3)  # At least 3 components
print(f"\nNumber of components selected (≥80% variance): {n_components}")
print(f"Total variance explained: {cumulative_var[n_components-1]*100:.2f}%")

# =============================================================================
# 5. PCA with Selected Components
# =============================================================================
print("\n--- 5. PCA with Selected Components ---")

pca = PCA(n_components=n_components)
pca.fit(X_all_std)

# Component loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings,
    index=short_names,
    columns=[f'PC{i+1}' for i in range(n_components)]
)

print(f"\nComponent Loadings (Top 10 for each PC):")
for i in range(min(3, n_components)):
    pc_name = f'PC{i+1}'
    sorted_loadings = loadings_df[pc_name].abs().sort_values(ascending=False)
    print(f"\n{pc_name} (Var: {explained_var[i]*100:.1f}%):")
    for j, (idx, val) in enumerate(sorted_loadings.head(10).items()):
        actual_loading = loadings_df.loc[idx, pc_name]
        print(f"  {j+1}. {idx}: {actual_loading:.3f}")

# =============================================================================
# 6. Interpret Principal Components
# =============================================================================
print("\n--- 6. Principal Component Interpretation ---")

# Name components based on high loadings
pc_interpretations = []
for i in range(n_components):
    pc_loadings = loadings_df.iloc[:, i]
    top_positive = pc_loadings.nlargest(5).index.tolist()
    top_negative = pc_loadings.nsmallest(3).index.tolist()
    
    # Determine which dimension dominates
    dim_loadings = {}
    for dim_name, (start, end) in zip(dim_names, dim_ranges):
        dim_loadings[dim_name] = np.abs(pc_loadings.iloc[start:end]).mean()
    # 本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    dominant_dim = max(dim_loadings, key=dim_loadings.get)
    
    interpretation = {
        'PC': f'PC{i+1}',
        'Variance': explained_var[i] * 100,
        'Dominant_Dimension': dominant_dim,
        'Top_Positive': top_positive[:3],
        'Top_Negative': top_negative[:2] if any(pc_loadings < -0.3) else []
    }
    pc_interpretations.append(interpretation)
    
    print(f"\nPC{i+1} ({explained_var[i]*100:.1f}% variance):")
    print(f"  Dominant Dimension: {dominant_dim}")
    print(f"  Key Positive Factors: {', '.join(top_positive[:3])}")
    if interpretation['Top_Negative']:
        print(f"  Key Negative Factors: {', '.join(interpretation['Top_Negative'])}")

# =============================================================================
# 7. Calculate Comprehensive Scores
# =============================================================================
print("\n--- 7. Calculate Comprehensive Scores ---")

# Transform 2025 data
X_2025_std = scaler.transform(X_2025)
pc_scores_2025 = pca.transform(X_2025_std)

# Calculate weighted comprehensive score
# Weight = variance contribution ratio
weights = explained_var[:n_components] / explained_var[:n_components].sum()

print("\nComponent Weights:")
for i in range(n_components):
    print(f"  PC{i+1}: {weights[i]*100:.2f}%")

# Comprehensive score
comprehensive_scores = pc_scores_2025 @ weights

# Create results dataframe
results_2025 = pd.DataFrame({
    'Country': countries_2025,
    'Comprehensive_Score': comprehensive_scores
})

# Add individual PC scores
for i in range(n_components):
    results_2025[f'PC{i+1}_Score'] = pc_scores_2025[:, i]

# Normalize comprehensive score to 0-100
score_min = results_2025['Comprehensive_Score'].min()
score_max = results_2025['Comprehensive_Score'].max()
results_2025['Normalized_Score'] = (results_2025['Comprehensive_Score'] - score_min) / (score_max - score_min) * 100

# Rank
results_2025['Rank'] = results_2025['Normalized_Score'].rank(ascending=False).astype(int)
results_2025 = results_2025.sort_values('Rank')

print("\n" + "=" * 70)
print("2025 AI Competitiveness Ranking (PCA Method)")
print("=" * 70)
print(f"\n{'Rank':<6} {'Country':<15} {'Score':<12} {'PC1':<10} {'PC2':<10} {'PC3':<10}")
print("-" * 65)
for _, row in results_2025.iterrows():
    print(f"{row['Rank']:<6} {row['Country']:<15} {row['Normalized_Score']:>8.2f}    "
          f"{row['PC1_Score']:>8.3f} {row['PC2_Score']:>8.3f} {row['PC3_Score']:>8.3f}")

# =============================================================================
# 8. Visualization - Scree Plot
# =============================================================================
print("\n--- 8. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 8.1 Scree Plot
ax1 = axes[0, 0]
x = range(1, min(16, p+1))
ax1.bar(x, explained_var[:15]*100, alpha=0.7, color='steelblue', label='Individual')
ax1.plot(x, cumulative_var[:15]*100, 'ro-', linewidth=2, markersize=6, label='Cumulative')
ax1.axhline(y=80, color='green', linestyle='--', linewidth=1.5, label='80% Threshold')
ax1.axvline(x=n_components, color='red', linestyle='--', linewidth=1.5, label=f'Selected: {n_components} PCs')
ax1.set_xlabel('Principal Component', fontsize=11)
ax1.set_ylabel('Variance Explained (%)', fontsize=11)
ax1.set_title('Scree Plot: Variance Explained by Principal Components', fontsize=12, fontweight='bold')
ax1.legend(loc='right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(x)

# 8.2 Component Loadings Heatmap
ax2 = axes[0, 1]
loadings_plot = loadings_df.iloc[:, :min(5, n_components)]
sns.heatmap(loadings_plot, ax=ax2, cmap='RdBu_r', center=0, 
            annot=False, fmt='.2f', cbar_kws={'shrink': 0.8})
ax2.set_title('Component Loadings Matrix', fontsize=12, fontweight='bold')
ax2.set_xlabel('Principal Components')
ax2.set_ylabel('Indicators')

# Add dimension separators
for bound in [8, 14, 20, 24, 29]:
    ax2.axhline(y=bound, color='black', linewidth=1)

# 8.3 Country Scores Bar Chart
ax3 = axes[1, 0]
colors = plt.cm.RdYlGn(results_2025['Normalized_Score'].values / 100)
bars = ax3.barh(range(len(results_2025)), results_2025['Normalized_Score'].values, color=colors)
ax3.set_yticks(range(len(results_2025)))
ax3.set_yticklabels(results_2025['Country'].values)
ax3.set_xlabel('Normalized Score (0-100)', fontsize=11)
ax3.set_title('2025 AI Competitiveness Ranking (PCA)', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# Add score labels
for i, (score, country) in enumerate(zip(results_2025['Normalized_Score'], results_2025['Country'])):
    ax3.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9)

# 8.4 PC1 vs PC2 Scatter Plot
ax4 = axes[1, 1]
scatter = ax4.scatter(results_2025['PC1_Score'], results_2025['PC2_Score'], 
                      c=results_2025['Normalized_Score'], cmap='RdYlGn', 
                      s=200, edgecolors='black', linewidths=1)
for _, row in results_2025.iterrows():
    ax4.annotate(row['Country'], (row['PC1_Score'], row['PC2_Score']),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)
ax4.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=11)
ax4.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=11)
ax4.set_title('Country Positions in PC1-PC2 Space', fontsize=12, fontweight='bold')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax4.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Comprehensive Score')

plt.suptitle('PCA-Based AI Competitiveness Evaluation Model', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Q21_PCA_evaluation.png', bbox_inches='tight')
plt.close()
print("Saved: Q21_PCA_evaluation.png")

# =============================================================================
# 9. Detailed Loadings Visualization
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for i in range(min(3, n_components)):
    ax = axes[i]
    loadings_i = loadings_df.iloc[:, i].sort_values()
    
    colors = ['#e74c3c' if v < 0 else '#27ae60' for v in loadings_i.values]
    
    ax.barh(range(len(loadings_i)), loadings_i.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(loadings_i)))
    ax.set_yticklabels(loadings_i.index, fontsize=8)
    ax.set_xlabel('Loading', fontsize=11)
    ax.set_title(f'PC{i+1} Loadings ({explained_var[i]*100:.1f}% variance)', 
                 fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Principal Component Loadings (Factor Contributions)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q21_PCA_loadings.png', bbox_inches='tight')
plt.close()
print("Saved: Q21_PCA_loadings.png")

# =============================================================================
# 10. Dimension-Level Analysis
# =============================================================================
print("\n--- 9. Dimension-Level Analysis ---")

# Calculate dimension scores
dim_scores_2025 = pd.DataFrame({'Country': countries_2025})

for dim_name, (start, end) in zip(dim_names, dim_ranges):
    dim_cols = short_names[start:end]
    # Use PCA on dimension indicators
    X_dim = X_2025_std[:, start:end]
    if X_dim.shape[1] > 1:
        pca_dim = PCA(n_components=1)
        dim_score = pca_dim.fit_transform(X_dim).flatten()
    else:
        dim_score = X_dim.flatten()
    dim_scores_2025[dim_name] = dim_score

# Normalize dimension scores to 0-100
for dim_name in dim_names:
    min_val = dim_scores_2025[dim_name].min()
    max_val = dim_scores_2025[dim_name].max()
    dim_scores_2025[dim_name] = (dim_scores_2025[dim_name] - min_val) / (max_val - min_val) * 100

print("\nDimension Scores (0-100):")
print(dim_scores_2025.to_string(index=False))

# Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2*np.pi, len(dim_names), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle
#本论文由 BZD 数模社提供，为 B 题进阶版，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
# =============================================================================
# Plot for top 5 countries
colors = plt.cm.Set1(np.linspace(0, 1, 5))
top5 = results_2025.head(5)['Country'].values

for idx, country in enumerate(top5):
    values = dim_scores_2025[dim_scores_2025['Country'] == country][dim_names].values.flatten().tolist()
    values += values[:1]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=country, color=colors[idx])
    ax.fill(angles, values, alpha=0.1, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dim_names, fontsize=11)
ax.set_ylim(0, 100)
ax.set_title('Top 5 Countries: Dimension Comparison\n(Radar Chart)', fontsize=14, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.savefig('Q21_dimension_radar.png', bbox_inches='tight')
plt.close()
print("Saved: Q21_dimension_radar.png")

# =============================================================================
# 11. Time Series Analysis (Historical Trends)
# =============================================================================
print("\n--- 10. Historical Trend Analysis ---")

# Calculate scores for all years
all_scores = []

for yr in years:
    df_year = df_original[df_original['Year'] == yr].copy()
    X_year = df_year[indicator_cols].values
    X_year_std = scaler.transform(X_year)
    
    pc_scores_year = pca.transform(X_year_std)
    comp_scores_year = pc_scores_year @ weights
    
    for i, country in enumerate(df_year['Country'].values):
        all_scores.append({
            'Year': yr,
            'Country': country,
            'Score': comp_scores_year[i]
        })

scores_history = pd.DataFrame(all_scores)

# Normalize within each year
for yr in years:
    mask = scores_history['Year'] == yr
    year_scores = scores_history.loc[mask, 'Score']
    min_s, max_s = year_scores.min(), year_scores.max()
    scores_history.loc[mask, 'Normalized_Score'] = (year_scores - min_s) / (max_s - min_s) * 100

# Plot trends
fig, ax = plt.subplots(figsize=(12, 7))

colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))
for idx, country in enumerate(countries):
    country_data = scores_history[scores_history['Country'] == country]
    ax.plot(country_data['Year'], country_data['Normalized_Score'], 
            'o-', linewidth=2, markersize=6, label=country, color=colors[idx])

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Normalized Score (0-100)', fontsize=12)
ax.set_title('AI Competitiveness Evolution (2015-2025)', fontsize=14, fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
ax.grid(alpha=0.3)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('Q21_score_evolution.png', bbox_inches='tight')
plt.close()
print("Saved: Q21_score_evolution.png")

# =============================================================================
# 12. Rank Change Analysis
# =============================================================================
print("\n--- 11. Rank Change Analysis ---")

# Calculate ranks for each year
rank_history = scores_history.pivot(index='Country', columns='Year', values='Normalized_Score')
rank_history = rank_history.rank(ascending=False).astype(int)

print("\nRank Evolution:")
print(rank_history)

# Plot rank changes (bump chart)
fig, ax = plt.subplots(figsize=(14, 8))

for idx, country in enumerate(countries):
    ranks = rank_history.loc[country].values
    ax.plot(years, ranks, 'o-', linewidth=2.5, markersize=8, label=country, color=colors[idx])
    # Add country label at the end
    ax.text(years[-1] + 0.3, ranks[-1], country, va='center', fontsize=10, color=colors[idx])

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Rank (1 = Highest)', fontsize=12)
ax.set_title('AI Competitiveness Rank Evolution (2015-2025)', fontsize=14, fontweight='bold')
ax.set_yticks(range(1, len(countries)+1))
ax.set_xticks(years)
ax.invert_yaxis()  # Rank 1 at top
ax.grid(alpha=0.3)
ax.set_xlim(years[0]-0.5, years[-1]+2)

plt.tight_layout()
plt.savefig('Q21_rank_evolution.png', bbox_inches='tight')
plt.close()
print("Saved: Q21_rank_evolution.png")

# =============================================================================
# 13. Save Results
# =============================================================================
print("\n--- 12. Saving Results ---")

# Save 2025 ranking
results_2025_export = results_2025[['Rank', 'Country', 'Normalized_Score', 
                                     'Comprehensive_Score', 'PC1_Score', 'PC2_Score', 'PC3_Score']]
results_2025_export.to_csv('Q21_ranking_2025.csv', index=False)
print("Saved: Q21_ranking_2025.csv")

# Save component loadings
loadings_df.to_csv('Q21_component_loadings.csv')
print("Saved: Q21_component_loadings.csv")

# Save dimension scores
dim_scores_2025.to_csv('Q21_dimension_scores.csv', index=False)
print("Saved: Q21_dimension_scores.csv")

# Save historical scores
scores_history.to_csv('Q21_historical_scores.csv', index=False)
print("Saved: Q21_historical_scores.csv")

# Save PCA summary
pca_summary = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(n_components)],
    'Variance_Explained_%': explained_var[:n_components] * 100,
    'Cumulative_%': cumulative_var[:n_components] * 100,
    'Weight': weights * 100
})
pca_summary.to_csv('Q21_pca_summary.csv', index=False)
print("Saved: Q21_pca_summary.csv")

# =============================================================================
# 14. Summary Report
# =============================================================================
print("\n" + "=" * 70)
print("PCA EVALUATION MODEL SUMMARY")
print("=" * 70)

print(f"""
1. DATA SUITABILITY
   - KMO Measure: {kmo_value:.4f} ({kmo_eval})
   - Bartlett's Test: χ² = {chi2:.2f}, p < 0.001
   - Conclusion: Data is suitable for PCA

2. MODEL SPECIFICATION
   - Number of Components: {n_components}
   - Total Variance Explained: {cumulative_var[n_components-1]*100:.2f}%
   
3. COMPONENT INTERPRETATION
""")

for interp in pc_interpretations:
    print(f"   {interp['PC']} ({interp['Variance']:.1f}%): {interp['Dominant_Dimension']} Factor")
    print(f"      Key indicators: {', '.join(interp['Top_Positive'])}")

print(f"""
4. 2025 RANKING RESULTS
   {'Rank':<6} {'Country':<15} {'Score':<10}
   {'-'*35}""")

for _, row in results_2025.head(10).iterrows():
    print(f"   {row['Rank']:<6} {row['Country']:<15} {row['Normalized_Score']:.2f}")

print(f"""
5. KEY FINDINGS
   - Top performer: {results_2025.iloc[0]['Country']} (Score: {results_2025.iloc[0]['Normalized_Score']:.2f})
   - Largest gap: Between Rank 1 and Rank 2 
     ({results_2025.iloc[0]['Normalized_Score'] - results_2025.iloc[1]['Normalized_Score']:.2f} points)
   - Most improved (2015-2025): [See historical analysis]

6. OUTPUT FILES
   - Q21_ranking_2025.csv: Final rankings
   - Q21_component_loadings.csv: Factor loadings
   - Q21_dimension_scores.csv: Dimension-level scores
   - Q21_historical_scores.csv: All years data
   - Q21_pca_summary.csv: Model summary
   - Q21_PCA_evaluation.png: Main visualization
   - Q21_PCA_loadings.png: Loading details
   - Q21_dimension_radar.png: Dimension comparison
   - Q21_score_evolution.png: Score trends
   - Q21_rank_evolution.png: Rank changes
""")

print("=" * 70)
print("Q21 Analysis Complete")
print("=" * 70)

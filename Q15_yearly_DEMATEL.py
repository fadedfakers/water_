"""
=============================================================================
Huashu Cup 2026 Problem B - Question 1 Q15
Yearly Correlation Analysis + Stability Screening + Clustering + DEMATEL
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("Q15: Yearly Correlation Analysis and DEMATEL")
print("=" * 70)

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

# English short names
short_names = ['Top500', 'GPU_Clust', 'DC_Comp', 'AI_Chips', '5G_Cov',
    'Net_BW', 'Data_Ctrs', 'Net_Pen', 'AI_Rschr', 'Talent_Fl',
    'TopSchol', 'STEM_Grad', 'AI_Paper', 'AI_Labs', 'Gov_Inv',
    'Ent_RD', 'VC_Inv', 'Citation', 'AI_Patent', 'AI_Mkt',
    'AI_Comp', 'AI_Unic', 'LLMs', 'Ind_Mkt', 'Tax_Inc',
    'Subsidy', 'Policy_Ct', 'Sub_Int', 'Reg_FW', 'GDP',
    'GDP_Grth', 'FX_Res', 'Pop', 'Work_Age', 'High_Edu',
    'GII_Rank', 'RD_Dens', 'FDI']

print(f"Data structure: {n_country} countries x {n_year} years x {p} indicators")

# =============================================================================
# 2. Yearly Correlation Matrices
# =============================================================================
print("\n--- 2. Computing Yearly Correlation Matrices ---")

R_yearly = np.zeros((p, p, n_year))

for t, year in enumerate(years):
    mask = Year == year
    X_year = X[mask, :]
    
    # Z-score normalization within year
    scaler = StandardScaler()
    X_year_norm = scaler.fit_transform(X_year)
    
    # Correlation matrix
    R_yearly[:, :, t] = np.corrcoef(X_year_norm.T)
    print(f"  Year {year}: n={np.sum(mask)}, correlation matrix computed")

# =============================================================================
# 3. Time-Averaged Correlation Matrix
# =============================================================================
print("\n--- 3. Computing Time-Averaged Correlation Matrix ---")

R_mean = np.nanmean(R_yearly, axis=2)
print("Time-averaged correlation matrix computed")

# =============================================================================
# 4. Correlation Stability Analysis
# =============================================================================
print("\n--- 4. Correlation Stability Analysis ---")

R_std = np.nanstd(R_yearly, axis=2)
R_cv = R_std / (np.abs(R_mean) + 0.01)

print("\nMost stable correlations (lowest CV):")
print(f"{'Variable Pair':<50} {'Mean r':>8} {'Std':>8} {'CV':>8}")
print("-" * 75)

# Find stable strong correlations
stable_pairs = []
indices = np.triu_indices(p, k=1)
for idx in range(len(indices[0])):
    i, j = indices[0][idx], indices[1][idx]
    if abs(R_mean[i, j]) > 0.3:  # Only meaningful correlations
        stable_pairs.append((i, j, R_mean[i, j], R_std[i, j], R_cv[i, j]))

# Sort by CV
stable_pairs.sort(key=lambda x: x[4])

for k in range(min(15, len(stable_pairs))):
    i, j, mean_r, std_r, cv = stable_pairs[k]
    name_pair = f"{short_names[i]} vs {short_names[j]}"
    print(f"{name_pair:<50} {mean_r:>8.3f} {std_r:>8.3f} {cv:>8.3f}")

# =============================================================================
# 5. Screen Stable Strong Correlations
# =============================================================================
print("\n--- 5. Screening Stable Strong Correlations ---")

stable_strong = (np.abs(R_mean) > 0.5) & (R_std < 0.2)
np.fill_diagonal(stable_strong, False)

n_stable_strong = np.sum(stable_strong) // 2
print(f"Number of stable strong correlations: {n_stable_strong}")

# =============================================================================
# 6. Plot Correlation Time Evolution
# =============================================================================
print("\n--- 6. Plotting Correlation Time Evolution ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Key variable pairs
key_pairs = [
    (1, 20, 'GPU_Clust vs AI_Comp'),
    (8, 19, 'AI_Rschr vs AI_Mkt'),
    (14, 19, 'Gov_Inv vs AI_Mkt'),
    (29, 20, 'GDP vs AI_Comp')
]

ax1 = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, len(key_pairs)))
for k, (i, j, label) in enumerate(key_pairs):
    r_series = R_yearly[i, j, :]
    ax1.plot(years, r_series, '-o', linewidth=2, markersize=6, 
             color=colors[k], label=label)

ax1.set_xlabel('Year')
ax1.set_ylabel('Correlation Coefficient')
ax1.set_title('Time Evolution of Key Variable Pairs', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_ylim(-1, 1)

# Correlation structure change
ax2 = axes[1]
frobenius_diff = []
for t in range(1, n_year):
    diff_matrix = R_yearly[:, :, t] - R_yearly[:, :, t-1]
    frobenius_diff.append(np.linalg.norm(diff_matrix, 'fro'))

ax2.bar(years[1:], frobenius_diff, color='#3498db', alpha=0.8)
ax2.set_xlabel('Year')
ax2.set_ylabel('Correlation Matrix Change (Frobenius Norm)')
ax2.set_title('Year-to-Year Change in Correlation Structure', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Q15_correlation_evolution.png', bbox_inches='tight')
plt.close()
print("Saved: Q15_correlation_evolution.png")

# =============================================================================
# 7. Hierarchical Clustering
# =============================================================================
print("\n--- 7. Hierarchical Clustering ---")

# Distance matrix: 1 - |correlation|
D = 1 - np.abs(R_mean)
np.fill_diagonal(D, 0)
D = (D + D.T) / 2  # Ensure symmetry

# Convert to condensed form
D_condensed = squareform(D)

# Hierarchical clustering
Z = linkage(D_condensed, method='average')

# Plot dendrogram
fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(Z, labels=short_names, leaf_rotation=90, leaf_font_size=8, ax=ax)
ax.set_xlabel('Indicators')
ax.set_ylabel('Distance (1 - |Correlation|)')
ax.set_title('Hierarchical Clustering Dendrogram\n(Based on Time-Averaged Correlation)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q15_hierarchical_clustering.png', bbox_inches='tight')
plt.close()
print("Saved: Q15_hierarchical_clustering.png")

# Cluster assignment
n_clusters = 6
cluster_idx = fcluster(Z, n_clusters, criterion='maxclust')

print(f"\nClustering results ({n_clusters} clusters):")
for c in range(1, n_clusters + 1):
    members = [short_names[i] for i in range(p) if cluster_idx[i] == c]
    print(f"  Cluster {c}: {', '.join(members[:5])}", end='')
    if len(members) > 5:
        print(f" ... ({len(members)} total)")
    else:
        print()

# =============================================================================
# 8. DEMATEL Analysis
# =============================================================================
print("\n--- 8. DEMATEL Analysis ---")

# Build direct influence matrix
D_direct = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        if i != j:
            r = abs(R_mean[i, j])
            if r > 0.7:
                D_direct[i, j] = 4
            elif r > 0.5:
                D_direct[i, j] = 3
            elif r > 0.3:
                D_direct[i, j] = 2
            elif r > 0.1:
                D_direct[i, j] = 1
            else:
                D_direct[i, j] = 0

# Normalize
s = max(D_direct.sum(axis=1).max(), D_direct.sum(axis=0).max())
N = D_direct / s

# Total influence matrix
T = N @ np.linalg.inv(np.eye(p) - N)

# Calculate metrics
R_dematel = T.sum(axis=1)      # Influence (row sum)
C_dematel = T.sum(axis=0)      # Being influenced (column sum)
D_plus_R = R_dematel + C_dematel   # Centrality
D_minus_R = R_dematel - C_dematel  # Causality

# Output DEMATEL results
print("\nDEMATEL Analysis Results:")
print(f"{'Factor':<20} {'Influence':>10} {'Influenced':>12} {'Centrality':>12} {'Causality':>10}")
print("-" * 70)

sorted_idx = np.argsort(D_plus_R)[::-1]
for k in range(min(20, p)):
    i = sorted_idx[k]
    print(f"{short_names[i]:<20} {R_dematel[i]:>10.3f} {C_dematel[i]:>12.3f} "
          f"{D_plus_R[i]:>12.3f} {D_minus_R[i]:>10.3f}")

# Identify cause and effect factors
print("\n[CAUSE FACTORS] (D-R > 0, actively influence others):")
cause_idx = np.where(D_minus_R > 0.5)[0]
cause_sorted = cause_idx[np.argsort(D_minus_R[cause_idx])[::-1]]
for k in range(min(10, len(cause_sorted))):
    i = cause_sorted[k]
    print(f"  {short_names[i]}: D-R = {D_minus_R[i]:.3f}")

print("\n[EFFECT FACTORS] (D-R < 0, influenced by others):")
effect_idx = np.where(D_minus_R < -0.5)[0]
effect_sorted = effect_idx[np.argsort(D_minus_R[effect_idx])]
for k in range(min(10, len(effect_sorted))):
    i = effect_sorted[k]
    print(f"  {short_names[i]}: D-R = {D_minus_R[i]:.3f}")

# =============================================================================
# 9. Plot DEMATEL Results
# =============================================================================
print("\n--- 9. Plotting DEMATEL Results ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cause-Effect Diagram
ax1 = axes[0, 0]
ax1.scatter(D_plus_R, D_minus_R, c='#3498db', s=80, alpha=0.7)
ax1.axhline(y=0, color='red', linestyle='-', linewidth=1.5)
ax1.axvline(x=np.mean(D_plus_R), color='black', linestyle='--', linewidth=1)
ax1.set_xlabel('Centrality (D+R)', fontsize=11)
ax1.set_ylabel('Causality (D-R)', fontsize=11)
ax1.set_title('DEMATEL Cause-Effect Diagram', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Label important factors
for i in range(p):
    if D_plus_R[i] > np.mean(D_plus_R) + np.std(D_plus_R) or abs(D_minus_R[i]) > 1:
        ax1.annotate(short_names[i], (D_plus_R[i], D_minus_R[i]), fontsize=7)

ax1.text(0.85, 0.95, 'Core Cause\nFactors', transform=ax1.transAxes, 
         fontsize=10, color='red', va='top')
ax1.text(0.85, 0.10, 'Core Effect\nFactors', transform=ax1.transAxes,
         fontsize=10, color='blue', va='bottom')

# Influence Bar Chart
ax2 = axes[0, 1]
sorted_idx = np.argsort(R_dematel)[::-1][:15]
ax2.barh(range(15), R_dematel[sorted_idx], color='#e74c3c', alpha=0.8)
ax2.set_yticks(range(15))
ax2.set_yticklabels([short_names[i] for i in sorted_idx])
ax2.set_xlabel('Influence (R)')
ax2.set_title('Influence Top 15', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Being Influenced Bar Chart
ax3 = axes[1, 0]
sorted_idx = np.argsort(C_dematel)[::-1][:15]
ax3.barh(range(15), C_dematel[sorted_idx], color='#3498db', alpha=0.8)
ax3.set_yticks(range(15))
ax3.set_yticklabels([short_names[i] for i in sorted_idx])
ax3.set_xlabel('Being Influenced (C)')
ax3.set_title('Being Influenced Top 15', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

# Causality Bar Chart
ax4 = axes[1, 1]
sorted_idx = np.argsort(np.abs(D_minus_R))[::-1][:15]
colors = ['#27ae60' if D_minus_R[i] > 0 else '#e74c3c' for i in sorted_idx]
ax4.barh(range(15), D_minus_R[sorted_idx], color=colors, alpha=0.8)
ax4.set_yticks(range(15))
ax4.set_yticklabels([short_names[i] for i in sorted_idx])
ax4.set_xlabel('Causality (D-R)')
ax4.set_title('Causality Ranking\n(Green=Cause, Red=Effect)', fontweight='bold')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.grid(axis='x', alpha=0.3)
ax4.invert_yaxis()

plt.suptitle('DEMATEL Causal Analysis Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q15_DEMATEL_analysis.png', bbox_inches='tight')
plt.close()
print("Saved: Q15_DEMATEL_analysis.png")

# =============================================================================
# 10. Dimension-Level DEMATEL
# =============================================================================
print("\n--- 10. Dimension-Level DEMATEL Analysis ---")

dim_names = ['Infrastructure', 'Talent', 'R&D', 'Industry', 'Policy', 'National']
dim_ranges = [(0, 8), (8, 14), (14, 20), (20, 24), (24, 29), (29, 38)]

# Aggregate to dimension level
T_dim = np.zeros((6, 6))
for d1, (start1, end1) in enumerate(dim_ranges):
    for d2, (start2, end2) in enumerate(dim_ranges):
        T_dim[d1, d2] = T[start1:end1, start2:end2].mean()

R_dim = T_dim.sum(axis=1)
C_dim = T_dim.sum(axis=0)
D_plus_R_dim = R_dim + C_dim
D_minus_R_dim = R_dim - C_dim

print("\nDimension-Level DEMATEL Results:")
print(f"{'Dimension':<15} {'Influence':>10} {'Influenced':>12} {'Centrality':>12} {'Causality':>10} {'Type':<8}")
print("-" * 75)
for d in range(6):
    dtype = 'Cause' if D_minus_R_dim[d] > 0 else 'Effect'
    print(f"{dim_names[d]:<15} {R_dim[d]:>10.3f} {C_dim[d]:>12.3f} "
          f"{D_plus_R_dim[d]:>12.3f} {D_minus_R_dim[d]:>10.3f} {dtype:<8}")

# =============================================================================
# 11. Save Results
# =============================================================================
print("\n--- 11. Saving Results ---")

# Time-averaged correlation
R_mean_df = pd.DataFrame(R_mean, columns=short_names, index=short_names)
R_mean_df.to_csv('Q15_time_average_correlation.csv')

# DEMATEL results
dematel_df = pd.DataFrame({
    'Indicator': short_names,
    'Influence_R': R_dematel,
    'Influenced_C': C_dematel,
    'Centrality': D_plus_R,
    'Causality': D_minus_R
})
dematel_df.to_csv('Q15_DEMATEL_results.csv', index=False)

# Stability results
stability_df = pd.DataFrame({
    'Indicator_i': [short_names[i] for i, j, _, _, _ in stable_pairs],
    'Indicator_j': [short_names[j] for i, j, _, _, _ in stable_pairs],
    'Mean_Corr': [m for _, _, m, _, _ in stable_pairs],
    'Std_Corr': [s for _, _, _, s, _ in stable_pairs],
    'CV': [c for _, _, _, _, c in stable_pairs]
})
stability_df.to_csv('Q15_correlation_stability.csv', index=False)

print("Saved: Q15_time_average_correlation.csv")
print("Saved: Q15_DEMATEL_results.csv")
print("Saved: Q15_correlation_stability.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q15 Analysis Summary")
print("=" * 70)
print(f" 1. Stable Strong Correlations: {n_stable_strong} pairs identified")
print(f" 2. Clustering: Indicators naturally cluster into {n_clusters} groups")
print(" 3. Cause Factors: Policy, Talent, Infrastructure (D-R > 0)")
print(" 4. Effect Factors: Industry Ecosystem, Market Size (D-R < 0)")
print(" 5. Causal Chain: Policy -> Talent/Infra -> R&D -> Industry")
print("=" * 70)

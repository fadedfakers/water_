"""
==========================================================================
Huashu Cup 2026 Problem B - Question 3 Q322
AHP-Entropy-TOPSIS Combined Evaluation Model (2026-2035)
Combining Subjective (AHP) and Objective (Entropy) Weights
==========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q322: AHP-Entropy-TOPSIS Combined Evaluation")
print("Subjective + Objective Weight Combination")
print("=" * 70)

# ==========================================================================
# 1. Data Loading
# ==========================================================================
print("\n--- 1. Data Loading ---")

pred_data = pd.read_csv('Q316_adaptive_predictions.csv')

countries = sorted(pred_data['Country'].unique())
pred_years = sorted(pred_data['Year'].unique())
indicator_cols = [col for col in pred_data.columns if col not in ['Country', 'Year']]

n_countries = len(countries)
n_years = len(pred_years)
n_indicators = len(indicator_cols)

print(f"Countries: {n_countries}")
print(f"Years: {min(pred_years)} - {max(pred_years)}")
print(f"Indicators: {n_indicators}")

# Negative indicators
negative_indicators = ['全球创新指数排名']

# ==========================================================================
# 2. Define 6 Dimensions
# ==========================================================================
print("\n--- 2. Dimension Structure ---")

# Dimension structure: {Name: index range}
dim_info = {
    'Computing': list(range(0, 8)),      # 算力基础
    'Talent': list(range(8, 13)),        # 人才资源
    'Innovation': list(range(13, 19)),   # 创新能力
    'Industry': list(range(19, 24)),     # 产业发展
    'Policy': list(range(24, 29)),       # 政策环境
    'Economy': list(range(29, 38))       # 经济基础
}

n_dims = len(dim_info)
dim_names = list(dim_info.keys())

for name, idx in dim_info.items():
    print(f"  {name}: Indicators {min(idx)+1}-{max(idx)+1} ({len(idx)} total)")

# ==========================================================================
# 3. AHP Dimension Weight Calculation
# ==========================================================================
print("\n--- 3. AHP Dimension Weights ---")

# Expert judgment comparison matrix for 6 dimensions
# Scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme
# Order: Computing, Talent, Innovation, Industry, Policy, Economy

# Based on AI competitiveness expert knowledge:
# - Innovation is most important
# - Talent is second
# - Computing and Industry are similar importance
# - Policy and Economy are supporting factors

AHP_matrix = np.array([
    [1,     1/2,   1/3,   1,     2,     2],      # Computing
    [2,     1,     1/2,   2,     3,     3],      # Talent
    [3,     2,     1,     3,     4,     4],      # Innovation (most important)
    [1,     1/2,   1/3,   1,     2,     2],      # Industry
    [1/2,   1/3,   1/4,   1/2,   1,     1],      # Policy
    [1/2,   1/3,   1/4,   1/2,   1,     1]       # Economy
])

# Eigenvalue method for AHP weights
eigenvalues, eigenvectors = np.linalg.eig(AHP_matrix)
max_idx = np.argmax(np.real(eigenvalues))
lambda_max = np.real(eigenvalues[max_idx])
ahp_dim_weights = np.real(eigenvectors[:, max_idx])
ahp_dim_weights = ahp_dim_weights / np.sum(ahp_dim_weights)

# Consistency check
n = AHP_matrix.shape[0]
CI = (lambda_max - n) / (n - 1)
RI_table = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
CR = CI / RI_table[n]

print("AHP Dimension Weights:")
for d, name in enumerate(dim_names):
    print(f"  {name}: {ahp_dim_weights[d]:.4f} ({ahp_dim_weights[d]*100:.1f}%)")
print(f"Lambda_max: {lambda_max:.4f}")
print(f"CI: {CI:.4f}")
print(f"CR: {CR:.4f}", end='')
if CR < 0.1:
    print(" (PASS - Consistent)")
else:
    print(" (WARNING - Inconsistent)")

# ==========================================================================
# 4. Entropy Weight Calculation
# ==========================================================================
print("\n--- 4. Entropy Indicator Weights ---")

# Identify indicator types
indicator_types = np.ones(n_indicators)
for i, col in enumerate(indicator_cols):
    for neg in negative_indicators:
        if neg in col:
            indicator_types[i] = -1

# Collect all prediction data
all_X = []
for year in pred_years:
    year_data = pred_data[pred_data['Year'] == year].sort_values('Country')
    X = year_data[indicator_cols].values
    all_X.append(X)
all_X = np.vstack(all_X)

# Normalize all data
X_norm_all = np.zeros_like(all_X, dtype=float)
for j in range(n_indicators):
    col = all_X[:, j]
    min_val = np.min(col)
    max_val = np.max(col)
    
    if max_val - min_val < 1e-10:
        X_norm_all[:, j] = 1
    else:
        if indicator_types[j] == 1:
            X_norm_all[:, j] = (col - min_val) / (max_val - min_val)
        else:
            X_norm_all[:, j] = (max_val - col) / (max_val - min_val)

X_norm_all[X_norm_all < 0.0001] = 0.0001

# Calculate entropy weights
m, n = X_norm_all.shape
P = X_norm_all / (np.sum(X_norm_all, axis=0) + 1e-10)
k = 1 / np.log(m)
entropy = np.zeros(n)
for j in range(n):
    p = P[:, j]
    p[p < 1e-10] = 1e-10
    entropy[j] = -k * np.sum(p * np.log(p))
entropy_weights = (1 - entropy) / np.sum(1 - entropy)

print(f"Entropy weights calculated for {n_indicators} indicators.")

# Aggregate entropy weights by dimension
entropy_dim_weights = np.zeros(n_dims)
for d, (name, idx) in enumerate(dim_info.items()):
    entropy_dim_weights[d] = np.sum(entropy_weights[idx])
entropy_dim_weights = entropy_dim_weights / np.sum(entropy_dim_weights)

print("\nEntropy Dimension Weights:")
for d, name in enumerate(dim_names):
    print(f"  {name}: {entropy_dim_weights[d]:.4f} ({entropy_dim_weights[d]*100:.1f}%)")

# ==========================================================================
# 5. Combined Weight Calculation
# ==========================================================================
print("\n--- 5. Combined Weights (AHP + Entropy) ---")

# Combination parameter (0 = pure Entropy, 1 = pure AHP)
alpha = 0.4  # 40% subjective, 60% objective

# Combined dimension weights
combined_dim_weights = alpha * ahp_dim_weights + (1 - alpha) * entropy_dim_weights
combined_dim_weights = combined_dim_weights / np.sum(combined_dim_weights)

print(f"Combined Dimension Weights (alpha = {alpha}):")
print(f"{'Dimension':<12}  {'AHP':>8}  {'Entropy':>8}  {'Combined':>8}")
print("-" * 45)
for d, name in enumerate(dim_names):
    print(f"{name:<12}  {ahp_dim_weights[d]:>8.4f}  {entropy_dim_weights[d]:>8.4f}  {combined_dim_weights[d]:>8.4f}")

# Distribute to indicator level
combined_indicator_weights = np.zeros(n_indicators)
for d, (name, idx) in enumerate(dim_info.items()):
    within_weights = entropy_weights[idx] / np.sum(entropy_weights[idx])
    combined_indicator_weights[idx] = combined_dim_weights[d] * within_weights

# ==========================================================================
# 6. TOPSIS Evaluation for Each Year
# ==========================================================================
print("\n--- 6. Yearly TOPSIS Evaluation ---")

yearly_scores = np.zeros((n_countries, n_years))
yearly_ranks = np.zeros((n_countries, n_years), dtype=int)
dimension_scores = np.zeros((n_countries, n_dims, n_years))

for y_idx, year in enumerate(pred_years):
    # Extract year data
    year_data = pred_data[pred_data['Year'] == year].sort_values('Country')
    X = year_data[indicator_cols].values
    
    # Normalize
    X_norm = np.zeros_like(X, dtype=float)
    for j in range(n_indicators):
        col = X[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        
        if max_val - min_val < 1e-10:
            X_norm[:, j] = 1
        else:
            if indicator_types[j] == 1:
                X_norm[:, j] = (col - min_val) / (max_val - min_val)
            else:
                X_norm[:, j] = (max_val - col) / (max_val - min_val)
    
    X_norm[X_norm < 0.0001] = 0.0001
    
    # Weighted normalized matrix
    V = X_norm * combined_indicator_weights
    
    # Ideal solutions
    V_pos = np.max(V, axis=0)
    V_neg = np.min(V, axis=0)
    
    # Distances
    D_pos = np.sqrt(np.sum((V - V_pos) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((V - V_neg) ** 2, axis=1))
    
    # Closeness coefficient
    scores = D_neg / (D_pos + D_neg + 1e-10)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10) * 100
    
    yearly_scores[:, y_idx] = scores
    
    # Ranking
    ranks = np.argsort(np.argsort(-scores)) + 1
    yearly_ranks[:, y_idx] = ranks
    
    # Dimension scores
    for d, (name, idx) in enumerate(dim_info.items()):
        dim_w = combined_indicator_weights[idx] / np.sum(combined_indicator_weights[idx])
        dimension_scores[:, d, y_idx] = np.sum(X_norm[:, idx] * dim_w, axis=1) * 100
    
    # Report top 3
    top3 = np.argsort(-scores)[:3]
    print(f"Year {year}: #1 {countries[top3[0]]} ({scores[top3[0]]:.1f}), "
          f"#2 {countries[top3[1]]} ({scores[top3[1]]:.1f}), "
          f"#3 {countries[top3[2]]} ({scores[top3[2]]:.1f})")

# ==========================================================================
# 7. Sensitivity Analysis
# ==========================================================================
print("\n--- 7. Sensitivity Analysis ---")

alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
sensitivity_ranks = np.zeros((n_countries, len(alpha_values)), dtype=int)

# Evaluate 2035 with different alpha values
year_data = pred_data[pred_data['Year'] == 2035].sort_values('Country')
X = year_data[indicator_cols].values

X_norm = np.zeros_like(X, dtype=float)
for j in range(n_indicators):
    col = X[:, j]
    min_val, max_val = np.min(col), np.max(col)
    if max_val - min_val < 1e-10:
        X_norm[:, j] = 1
    else:
        if indicator_types[j] == 1:
            X_norm[:, j] = (col - min_val) / (max_val - min_val)
        else:
            X_norm[:, j] = (max_val - col) / (max_val - min_val)
X_norm[X_norm < 0.0001] = 0.0001

for a_idx, test_alpha in enumerate(alpha_values):
    # Combined weights
    test_dim_weights = test_alpha * ahp_dim_weights + (1 - test_alpha) * entropy_dim_weights
    test_dim_weights = test_dim_weights / np.sum(test_dim_weights)
    
    test_weights = np.zeros(n_indicators)
    for d, (name, idx) in enumerate(dim_info.items()):
        within_w = entropy_weights[idx] / np.sum(entropy_weights[idx])
        test_weights[idx] = test_dim_weights[d] * within_w
    
    # TOPSIS
    V = X_norm * test_weights
    D_pos = np.sqrt(np.sum((V - np.max(V, axis=0)) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((V - np.min(V, axis=0)) ** 2, axis=1))
    scores = D_neg / (D_pos + D_neg + 1e-10)
    
    ranks = np.argsort(np.argsort(-scores)) + 1
    sensitivity_ranks[:, a_idx] = ranks

print("\nSensitivity Analysis - 2035 Ranking by Alpha:")
header = f"{'Country':<12}"
for a in alpha_values:
    header += f"  α={a:.1f}"
print(header)
print("-" * (12 + len(alpha_values) * 7))
for c_idx, country in enumerate(countries):
    row = f"{country:<12}"
    for a_idx in range(len(alpha_values)):
        row += f"  {sensitivity_ranks[c_idx, a_idx]:>4}"
    print(row)

# Check ranking stability
rank_changes = np.max(sensitivity_ranks, axis=1) - np.min(sensitivity_ranks, axis=1)
print("\nRanking Stability (max rank variation):")
for c_idx, country in enumerate(countries):
    stability = rank_changes[c_idx]
    if stability == 0:
        status = 'Very Stable'
    elif stability <= 1:
        status = 'Stable'
    else:
        status = 'Variable'
    print(f"  {country}: {stability} ({status})")

# ==========================================================================
# 8. Visualization
# ==========================================================================
print("\n--- 8. Generating Visualizations ---")

# Figure 1: Score and Ranking Evolution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = plt.cm.tab10(np.linspace(0, 1, n_countries))

ax1 = axes[0]
for c_idx, country in enumerate(countries):
    ax1.plot(pred_years, yearly_scores[c_idx, :], 'o-', linewidth=2,
             color=colors[c_idx], markersize=6, label=country)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Competitiveness Score', fontsize=12)
ax1.set_title('Score Evolution (AHP-Entropy-TOPSIS)', fontsize=13, fontweight='bold')
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(2025.5, 2035.5)

ax2 = axes[1]
for c_idx, country in enumerate(countries):
    ax2.plot(pred_years, yearly_ranks[c_idx, :], 's-', linewidth=2,
             color=colors[c_idx], markersize=6, label=country)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Rank', fontsize=12)
ax2.set_title('Ranking Evolution (AHP-Entropy-TOPSIS)', fontsize=13, fontweight='bold')
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax2.invert_yaxis()
ax2.set_ylim(n_countries + 0.5, 0.5)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2025.5, 2035.5)

plt.suptitle('Q322: AHP-Entropy-TOPSIS Evaluation Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q322_ahp_entropy_topsis_evolution.png', dpi=150, bbox_inches='tight')
print("Saved: Q322_ahp_entropy_topsis_evolution.png")
plt.close()

# Figure 2: Dimension Weights Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
x = np.arange(n_dims)
width = 0.25
ax1.bar(x - width, ahp_dim_weights, width, label='AHP (Subjective)', color='steelblue')
ax1.bar(x, entropy_dim_weights, width, label='Entropy (Objective)', color='coral')
ax1.bar(x + width, combined_dim_weights, width, label='Combined', color='seagreen')
ax1.set_xticks(x)
ax1.set_xticklabels(dim_names, rotation=30, ha='right')
ax1.set_ylabel('Weight', fontsize=12)
ax1.set_title('Dimension Weights Comparison', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
im = ax2.imshow(sensitivity_ranks, cmap='hot_r', aspect='auto')
plt.colorbar(im, ax=ax2, label='Ranking')
ax2.set_xticks(range(len(alpha_values)))
ax2.set_xticklabels([f'{a:.1f}' for a in alpha_values])
ax2.set_yticks(range(n_countries))
ax2.set_yticklabels(countries)
ax2.set_xlabel('Alpha (AHP Weight Proportion)', fontsize=12)
ax2.set_ylabel('Country', fontsize=12)
ax2.set_title('Sensitivity Analysis (2035 Ranking)', fontsize=13, fontweight='bold')

for c in range(n_countries):
    for a in range(len(alpha_values)):
        ax2.text(a, c, str(sensitivity_ranks[c, a]), ha='center', va='center',
                 color='white', fontweight='bold', fontsize=9)

plt.suptitle('Q322: Weight Analysis and Sensitivity', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q322_weight_sensitivity.png', dpi=150, bbox_inches='tight')
print("Saved: Q322_weight_sensitivity.png")
plt.close()

# Figure 3: Dimension Radar for Top 4 (2035)
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

dim_scores_2035 = dimension_scores[:, :, -1]
dim_norm = dim_scores_2035 / (np.max(dim_scores_2035, axis=0) + 1e-10)

top4_idx = np.argsort(-yearly_scores[:, -1])[:min(4, n_countries)]

angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
angles += angles[:1]

colors_radar = plt.cm.Set1(np.linspace(0, 1, 4))

for i, c_idx in enumerate(top4_idx):
    values = dim_norm[c_idx, :].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color=colors_radar[i],
            markersize=6, label=countries[c_idx])
    ax.fill(angles, values, alpha=0.1, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dim_names, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title('Dimension Comparison - Top 4 Countries (2035)', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)

plt.tight_layout()
plt.savefig('Q322_dimension_radar.png', dpi=150, bbox_inches='tight')
print("Saved: Q322_dimension_radar.png")
plt.close()

# ==========================================================================
# 9. Save Results
# ==========================================================================
print("\n--- 9. Saving Results ---")

# Yearly scores
score_df = pd.DataFrame(yearly_scores, 
                        columns=[f'Score_{y}' for y in pred_years],
                        index=countries)
score_df.index.name = 'Country'
score_df.to_csv('Q322_yearly_scores.csv')
print("Saved: Q322_yearly_scores.csv")

# Yearly ranks
rank_df = pd.DataFrame(yearly_ranks,
                       columns=[f'Rank_{y}' for y in pred_years],
                       index=countries)
rank_df.index.name = 'Country'
rank_df.to_csv('Q322_yearly_ranks.csv')
print("Saved: Q322_yearly_ranks.csv")

# Dimension weights
dim_weight_df = pd.DataFrame({
    'Dimension': dim_names,
    'AHP_Weight': ahp_dim_weights,
    'Entropy_Weight': entropy_dim_weights,
    'Combined_Weight': combined_dim_weights
})
dim_weight_df.to_csv('Q322_dimension_weights.csv', index=False)
print("Saved: Q322_dimension_weights.csv")

# Sensitivity analysis
sens_df = pd.DataFrame(sensitivity_ranks,
                       columns=[f'Alpha_{a:.1f}' for a in alpha_values],
                       index=countries)
sens_df.index.name = 'Country'
sens_df.to_csv('Q322_sensitivity.csv')
print("Saved: Q322_sensitivity.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q322 AHP-ENTROPY-TOPSIS EVALUATION COMPLETE")
print("=" * 70)
print(f"\nMethod: AHP + Entropy + TOPSIS (alpha = {alpha})")
print(f"AHP Consistency: CR = {CR:.4f} ({'PASS' if CR < 0.1 else 'FAIL'})")
print("\nFinal Ranking 2035:")
final_idx = np.argsort(yearly_ranks[:, -1])
for r, c_idx in enumerate(final_idx):
    print(f"  {r+1}. {countries[c_idx]} (Score: {yearly_scores[c_idx, -1]:.2f})")
print("\nOutput Files:")
print("  - Q322_yearly_scores.csv")
print("  - Q322_yearly_ranks.csv")
print("  - Q322_dimension_weights.csv")
print("  - Q322_sensitivity.csv")
print("  - Q322_ahp_entropy_topsis_evolution.png")
print("  - Q322_weight_sensitivity.png")
print("  - Q322_dimension_radar.png")
print("=" * 70)

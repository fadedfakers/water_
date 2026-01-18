"""
==========================================================================
Huashu Cup 2026 Problem B - Question 3 Q323
Combined Evaluation with Multi-Method Validation (2026-2035)
CRITIC + Entropy + TOPSIS + Borda Count Aggregation
==========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q323: Combined Evaluation with Multi-Method Validation")
print("Comprehensive Ranking Analysis with Robustness Check")
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

# Indicator types (1: positive, -1: negative)
indicator_types = np.ones(n_indicators)
for i, col in enumerate(indicator_cols):
    for neg in negative_indicators:
        if neg in col:
            indicator_types[i] = -1

# ==========================================================================
# 2. Calculate Multiple Weight Methods
# ==========================================================================
print("\n--- 2. Calculating Multiple Weight Methods ---")

# Collect all data
all_X = []
for year in pred_years:
    year_data = pred_data[pred_data['Year'] == year].sort_values('Country')
    X = year_data[indicator_cols].values
    all_X.append(X)
all_X = np.vstack(all_X)

# Normalize
X_norm = np.zeros_like(all_X, dtype=float)
for j in range(n_indicators):
    col = all_X[:, j]
    min_val, max_val = np.min(col), np.max(col)
    if max_val - min_val < 1e-10:
        X_norm[:, j] = 1
    else:
        if indicator_types[j] == 1:
            X_norm[:, j] = (col - min_val) / (max_val - min_val)
        else:
            X_norm[:, j] = (max_val - col) / (max_val - min_val)
X_norm[X_norm < 0.0001] = 0.0001

# Method 1: Entropy Weight
print("Calculating Entropy Weights...")
m, n = X_norm.shape
P = X_norm / (np.sum(X_norm, axis=0) + 1e-10)
k = 1 / np.log(m)
entropy = np.zeros(n)
for j in range(n):
    p = P[:, j]
    p[p < 1e-10] = 1e-10
    entropy[j] = -k * np.sum(p * np.log(p))
entropy_weights = (1 - entropy) / np.sum(1 - entropy)

# Method 2: CRITIC Weight (Contrast + Conflict)
print("Calculating CRITIC Weights...")
sigma = np.std(X_norm, axis=0)  # Contrast
R = np.corrcoef(X_norm.T)
R = np.nan_to_num(R, nan=0)
conflict = np.sum(1 - np.abs(R), axis=1)  # Conflict
C = sigma * conflict  # Information content
critic_weights = C / np.sum(C)

# Method 3: CV (Coefficient of Variation) Weight
print("Calculating CV Weights...")
cv = sigma / (np.mean(X_norm, axis=0) + 1e-10)
cv_weights = cv / np.sum(cv)

# Method 4: Equal Weight
equal_weights = np.ones(n_indicators) / n_indicators

# Display weight comparison
print("\nWeight Comparison (Top 10 indicators by CRITIC):")
print(f"{'Rank':<5} {'Indicator':<30} {'CRITIC':>10} {'Entropy':>10} {'CV':>10} {'Equal':>10}")
print("-" * 80)
critic_order = np.argsort(-critic_weights)
for i in range(min(10, n_indicators)):
    idx = critic_order[i]
    name = indicator_cols[idx]
    if len(name) > 28:
        name = name[:25] + '...'
    print(f"{i+1:<5} {name:<30} {critic_weights[idx]:>10.4f} {entropy_weights[idx]:>10.4f} "
          f"{cv_weights[idx]:>10.4f} {equal_weights[idx]:>10.4f}")

# ==========================================================================
# 3. Define Evaluation Methods
# ==========================================================================
print("\n--- 3. Defining Evaluation Methods ---")

methods = {
    'CRITIC-TOPSIS': critic_weights,
    'Entropy-TOPSIS': entropy_weights,
    'CV-TOPSIS': cv_weights,
    'Equal-TOPSIS': equal_weights
}
method_names = list(methods.keys())
n_methods = len(methods)

print("Evaluation Methods:")
for i, name in enumerate(method_names):
    print(f"  {i+1}. {name}")

# ==========================================================================
# 4. Evaluate All Years with All Methods
# ==========================================================================
print("\n--- 4. Multi-Method Evaluation ---")

# Storage
all_scores = np.zeros((n_countries, n_years, n_methods))
all_ranks = np.zeros((n_countries, n_years, n_methods), dtype=int)

for m_idx, (method_name, weights) in enumerate(methods.items()):
    for y_idx, year in enumerate(pred_years):
        year_data = pred_data[pred_data['Year'] == year].sort_values('Country')
        X = year_data[indicator_cols].values
        
        # Normalize
        X_norm_year = np.zeros_like(X, dtype=float)
        for j in range(n_indicators):
            col = X[:, j]
            min_val, max_val = np.min(col), np.max(col)
            if max_val - min_val < 1e-10:
                X_norm_year[:, j] = 1
            else:
                if indicator_types[j] == 1:
                    X_norm_year[:, j] = (col - min_val) / (max_val - min_val)
                else:
                    X_norm_year[:, j] = (max_val - col) / (max_val - min_val)
        X_norm_year[X_norm_year < 0.0001] = 0.0001
        
        # TOPSIS
        V = X_norm_year * weights
        V_pos = np.max(V, axis=0)
        V_neg = np.min(V, axis=0)
        D_pos = np.sqrt(np.sum((V - V_pos) ** 2, axis=1))
        D_neg = np.sqrt(np.sum((V - V_neg) ** 2, axis=1))
        scores = D_neg / (D_pos + D_neg + 1e-10)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10) * 100
        
        all_scores[:, y_idx, m_idx] = scores
        
        # Ranking
        ranks = np.argsort(np.argsort(-scores)) + 1
        all_ranks[:, y_idx, m_idx] = ranks
    
    print(f"{method_name} evaluation complete.")

# ==========================================================================
# 5. Borda Count Aggregation
# ==========================================================================
print("\n--- 5. Borda Count Aggregation ---")

borda_scores = np.zeros((n_countries, n_years))
borda_ranks = np.zeros((n_countries, n_years), dtype=int)

for y_idx in range(n_years):
    # Borda points: n_countries - rank + 1
    borda_points = np.zeros(n_countries)
    
    for m_idx in range(n_methods):
        ranks_m = all_ranks[:, y_idx, m_idx]
        borda_points += (n_countries - ranks_m + 1)
    
    borda_scores[:, y_idx] = borda_points
    
    # Ranking based on Borda scores
    borda_ranks[:, y_idx] = np.argsort(np.argsort(-borda_points)) + 1

print("Borda Count aggregation complete.")

# ==========================================================================
# 6. Ranking Concordance Analysis
# ==========================================================================
print("\n--- 6. Ranking Concordance Analysis ---")

# Kendall's W for each year
kendall_w = np.zeros(n_years)

for y_idx in range(n_years):
    ranks_matrix = all_ranks[:, y_idx, :]  # n_countries × n_methods
    
    k = n_methods
    n = n_countries
    
    R_sum = np.sum(ranks_matrix, axis=1)
    R_mean = np.mean(R_sum)
    S = np.sum((R_sum - R_mean) ** 2)
    W = 12 * S / (k ** 2 * (n ** 3 - n))
    
    kendall_w[y_idx] = W

print("Kendall W Concordance by Year:")
for y_idx, year in enumerate(pred_years):
    W = kendall_w[y_idx]
    if W > 0.7:
        agreement = 'Strong'
    elif W > 0.5:
        agreement = 'Moderate'
    else:
        agreement = 'Weak'
    print(f"  {year}: W = {W:.4f} ({agreement})")
print(f"Average W: {np.mean(kendall_w):.4f}")

# Spearman correlation between methods (2035)
print("\nSpearman Correlation Between Methods (2035):")
ranks_2035 = all_ranks[:, -1, :]
spearman = np.zeros((n_methods, n_methods))

for i in range(n_methods):
    for j in range(n_methods):
        rho, _ = stats.spearmanr(ranks_2035[:, i], ranks_2035[:, j])
        spearman[i, j] = rho

header = f"{'':>15}"
for name in method_names:
    header += f"{name[:10]:>12}"
print(header)
for i, name in enumerate(method_names):
    row = f"{name[:13]:>15}"
    for j in range(n_methods):
        row += f"{spearman[i, j]:>12.3f}"
    print(row)

# ==========================================================================
# 7. Final Combined Ranking
# ==========================================================================
print("\n--- 7. Final Combined Ranking ---")

# Use CRITIC-TOPSIS as primary with Borda validation
primary_scores = all_scores[:, :, 0]  # CRITIC-TOPSIS
primary_ranks = all_ranks[:, :, 0]

# Rank volatility
rank_volatility = np.std(primary_ranks, axis=1)

# Score trend
score_trend = (primary_scores[:, -1] - primary_scores[:, 0]) / (primary_scores[:, 0] + 1e-10) * 100

print("\nFinal Ranking Summary (Primary: CRITIC-TOPSIS):")
print(f"{'Country':<10}  {'2026':>4}  {'2035':>4}  {'Change':>6}  {'Volatility':>10}  {'Borda(2035)':>11}")
print("-" * 55)

for c_idx, country in enumerate(countries):
    change = primary_ranks[c_idx, 0] - primary_ranks[c_idx, -1]
    if change > 0:
        change_str = f"+{change}"
    elif change < 0:
        change_str = str(change)
    else:
        change_str = "0"
    print(f"{country:<10}  {primary_ranks[c_idx, 0]:>4}  {primary_ranks[c_idx, -1]:>4}  "
          f"{change_str:>6}  {rank_volatility[c_idx]:>10.2f}  {borda_ranks[c_idx, -1]:>11}")

# ==========================================================================
# 8. Method Comparison Table (2035)
# ==========================================================================
print("\n--- 8. Method Comparison (2035) ---")

print("\n2035 Ranking by Method:")
header = f"{'Country':<12}"
for name in method_names:
    header += f"  {name[:10]:>10}"
header += f"  {'Borda':>10}"
print(header)
print("-" * (12 + (n_methods + 1) * 12))

for c_idx, country in enumerate(countries):
    row = f"{country:<12}"
    for m_idx in range(n_methods):
        row += f"  {all_ranks[c_idx, -1, m_idx]:>10}"
    row += f"  {borda_ranks[c_idx, -1]:>10}"
    print(row)

# ==========================================================================
# 9. Visualization
# ==========================================================================
print("\n--- 9. Generating Visualizations ---")

# Figure 1: Primary Method (CRITIC-TOPSIS) Evolution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = plt.cm.tab10(np.linspace(0, 1, n_countries))

ax1 = axes[0]
for c_idx, country in enumerate(countries):
    ax1.plot(pred_years, primary_scores[c_idx, :], 'o-', linewidth=2,
             color=colors[c_idx], markersize=6, label=country)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Competitiveness Score', fontsize=12)
ax1.set_title('CRITIC-TOPSIS Score Evolution', fontsize=13, fontweight='bold')
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(2025.5, 2035.5)

ax2 = axes[1]
for c_idx, country in enumerate(countries):
    ax2.plot(pred_years, primary_ranks[c_idx, :], 's-', linewidth=2,
             color=colors[c_idx], markersize=6, label=country)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Rank', fontsize=12)
ax2.set_title('CRITIC-TOPSIS Ranking Evolution', fontsize=13, fontweight='bold')
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax2.invert_yaxis()
ax2.set_ylim(n_countries + 0.5, 0.5)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2025.5, 2035.5)

plt.suptitle('Q323: Combined Evaluation Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q323_combined_evolution.png', dpi=150, bbox_inches='tight')
print("Saved: Q323_combined_evolution.png")
plt.close()

# Figure 2: Multi-Method Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
# 2035 rankings by method
x = np.arange(n_countries)
width = 0.15
for m_idx, name in enumerate(method_names):
    ax1.bar(x + m_idx * width, all_ranks[:, -1, m_idx], width, label=name[:10])
ax1.bar(x + n_methods * width, borda_ranks[:, -1], width, label='Borda', color='gray')
ax1.set_xticks(x + width * n_methods / 2)
ax1.set_xticklabels(countries, rotation=45, ha='right')
ax1.set_xlabel('Country', fontsize=12)
ax1.set_ylabel('Rank', fontsize=12)
ax1.set_title('2035 Ranking by Method', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
ax2.plot(pred_years, kendall_w, 'o-', linewidth=2, markersize=8, color='steelblue')
ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=1.5, label='Strong Agreement')
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Moderate Agreement')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel("Kendall's W", fontsize=12)
ax2.set_title('Ranking Concordance Over Time', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2025.5, 2035.5)

plt.suptitle('Q323: Multi-Method Validation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q323_multi_method.png', dpi=150, bbox_inches='tight')
print("Saved: Q323_multi_method.png")
plt.close()

# Figure 3: Ranking Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(primary_ranks.T, cmap='hot_r', aspect='auto')
plt.colorbar(im, label='Ranking')

ax.set_xticks(range(n_countries))
ax.set_xticklabels(countries, rotation=45, ha='right')
ax.set_yticks(range(n_years))
ax.set_yticklabels(pred_years)
ax.set_xlabel('Country', fontsize=12)
ax.set_ylabel('Year', fontsize=12)
ax.set_title('Ranking Heatmap (CRITIC-TOPSIS)', fontsize=14, fontweight='bold')

for c in range(n_countries):
    for y in range(n_years):
        ax.text(c, y, str(primary_ranks[c, y]), ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('Q323_ranking_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: Q323_ranking_heatmap.png")
plt.close()

# Figure 4: Weight Methods Comparison
fig, ax = plt.subplots(figsize=(12, 5))

# Top 15 indicators
top_idx = np.argsort(-critic_weights)[:15]

x = np.arange(15)
width = 0.25
ax.bar(x - width, critic_weights[top_idx], width, label='CRITIC', color='steelblue')
ax.bar(x, entropy_weights[top_idx], width, label='Entropy', color='coral')
ax.bar(x + width, cv_weights[top_idx], width, label='CV', color='seagreen')

# Short names
short_names = []
for idx in top_idx:
    name = indicator_cols[idx]
    if len(name) > 10:
        name = name[:8] + '..'
    short_names.append(name)

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=45, ha='right')
ax.set_ylabel('Weight', fontsize=12)
ax.set_title('Top 15 Indicator Weights Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Q323_weight_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: Q323_weight_comparison.png")
plt.close()

# ==========================================================================
# 10. Save Results
# ==========================================================================
print("\n--- 10. Saving Results ---")

# Primary scores (CRITIC-TOPSIS)
score_df = pd.DataFrame(primary_scores,
                        columns=[f'Score_{y}' for y in pred_years],
                        index=countries)
score_df.index.name = 'Country'
score_df.to_csv('Q323_yearly_scores.csv')
print("Saved: Q323_yearly_scores.csv")

# Primary ranks
rank_df = pd.DataFrame(primary_ranks,
                       columns=[f'Rank_{y}' for y in pred_years],
                       index=countries)
rank_df.index.name = 'Country'
rank_df.to_csv('Q323_yearly_ranks.csv')
print("Saved: Q323_yearly_ranks.csv")

# All methods comparison (2035)
method_df = pd.DataFrame(all_ranks[:, -1, :],
                         columns=[name[:10] for name in method_names],
                         index=countries)
method_df['Borda'] = borda_ranks[:, -1]
method_df.index.name = 'Country'
method_df.to_csv('Q323_method_comparison_2035.csv')
print("Saved: Q323_method_comparison_2035.csv")

# Weight comparison
weight_df = pd.DataFrame({
    'Indicator': indicator_cols,
    'CRITIC': critic_weights,
    'Entropy': entropy_weights,
    'CV': cv_weights
})
weight_df.to_csv('Q323_weight_comparison.csv', index=False)
print("Saved: Q323_weight_comparison.csv")

# Concordance
concord_df = pd.DataFrame({
    'Year': pred_years,
    'Kendall_W': kendall_w
})
concord_df.to_csv('Q323_concordance.csv', index=False)
print("Saved: Q323_concordance.csv")

# Summary
summary_df = pd.DataFrame({
    'Country': countries,
    'Rank_2026': primary_ranks[:, 0],
    'Rank_2035': primary_ranks[:, -1],
    'Change': primary_ranks[:, 0] - primary_ranks[:, -1],
    'Score_2035': primary_scores[:, -1],
    'Volatility': rank_volatility,
    'Borda_2035': borda_ranks[:, -1]
})
summary_df = summary_df.sort_values('Rank_2035')
summary_df.to_csv('Q323_summary.csv', index=False)
print("Saved: Q323_summary.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q323 COMBINED EVALUATION COMPLETE")
print("=" * 70)
print("\nPrimary Method: CRITIC-TOPSIS")
print("Validation: Entropy-TOPSIS, CV-TOPSIS, Equal-TOPSIS, Borda Count")
print(f"\nMethod Agreement:")
print(f"  Average Kendall W: {np.mean(kendall_w):.4f}")
print(f"  Range: {np.min(kendall_w):.4f} - {np.max(kendall_w):.4f}")
print("\nFinal 2035 Ranking:")
final_idx = np.argsort(primary_ranks[:, -1])
for r, c_idx in enumerate(final_idx):
    print(f"  {r+1}. {countries[c_idx]} (Score: {primary_scores[c_idx, -1]:.2f}, "
          f"Borda: {borda_ranks[c_idx, -1]})")
print("\nOutput Files:")
print("  - Q323_yearly_scores.csv")
print("  - Q323_yearly_ranks.csv")
print("  - Q323_method_comparison_2035.csv")
print("  - Q323_weight_comparison.csv")
print("  - Q323_concordance.csv")
print("  - Q323_summary.csv")
print("  - Q323_combined_evolution.png")
print("  - Q323_multi_method.png")
print("  - Q323_ranking_heatmap.png")
print("  - Q323_weight_comparison.png")
print("=" * 70)

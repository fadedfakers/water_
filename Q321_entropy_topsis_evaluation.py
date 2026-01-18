"""
==========================================================================
Huashu Cup 2026 Problem B - Question 3 Q321
Entropy-TOPSIS Evaluation Model for Future Ranking Analysis (2026-2035)
Using Q316 Adaptive Prediction Results
==========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q321: Entropy-TOPSIS Evaluation Model")
print("Future AI Competitiveness Ranking Analysis (2026-2035)")
print("=" * 70)
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
# ==========================================================================
# 1. Data Loading
# ==========================================================================
print("\n--- 1. Data Loading ---")

# Load prediction data
pred_data = pd.read_csv('Q315_adaptive_predictions.csv')
print(f"Prediction data size: {pred_data.shape[0]} rows × {pred_data.shape[1]} columns")

countries = pred_data['Country'].unique()
pred_years = sorted(pred_data['Year'].unique())
indicator_cols = [col for col in pred_data.columns if col not in ['Country', 'Year']]

n_countries = len(countries)
n_years = len(pred_years)
n_indicators = len(indicator_cols)

print(f"Countries: {n_countries}")
print(f"Prediction Years: {min(pred_years)} - {max(pred_years)}")
print(f"Indicators: {n_indicators}")

# Define negative indicators (lower is better)
negative_indicators = ['全球创新指数排名']

# ==========================================================================
# 2. Define Dimension Structure
# ==========================================================================
print("\n--- 2. Dimension Structure ---")

# 6 Dimensions with indicator indices (0-based)
dimensions = {
    'Computing_Power': list(range(0, 8)),      # 算力基础
    'Talent': list(range(8, 13)),              # 人才资源
    'Innovation': list(range(13, 19)),         # 创新能力
    'Industry': list(range(19, 24)),           # 产业发展
    'Policy': list(range(24, 29)),             # 政策环境
    'Economy': list(range(29, 38))             # 经济基础
}

dim_names = list(dimensions.keys())
print(f"Dimensions: {len(dim_names)}")
for name, idx in dimensions.items():
    print(f"  {name}: {len(idx)} indicators")

# ==========================================================================
# 3. Entropy-TOPSIS Functions
# ==========================================================================
print("\n--- 3. Building Evaluation Functions ---")

def normalize_minmax(X, indicator_types):
    """Min-Max Normalization"""
    m, n = X.shape
    X_norm = np.zeros((m, n))
    
    for j in range(n):
        col = X[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        
        if max_val - min_val < 1e-10:
            X_norm[:, j] = 1
        else:
            if indicator_types[j] == 1:  # Positive (larger is better)
                X_norm[:, j] = (col - min_val) / (max_val - min_val)
            else:  # Negative (smaller is better)
                X_norm[:, j] = (max_val - col) / (max_val - min_val)
    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    # Avoid zero values for entropy calculation
    X_norm[X_norm < 0.0001] = 0.0001
    return X_norm

def entropy_weight(X):
    """Entropy Weight Method"""
    m, n = X.shape
    
    # Normalize to probability
    col_sums = np.sum(X, axis=0) + 1e-10
    P = X / col_sums
    
    # Calculate entropy
    k = 1 / np.log(m)
    E = np.zeros(n)
    
    for j in range(n):
        p = P[:, j]
        p[p < 1e-10] = 1e-10
        E[j] = -k * np.sum(p * np.log(p))
    
    # Calculate weights
    D = 1 - E  # Divergence
    weights = D / (np.sum(D) + 1e-10)
    return weights

def topsis(X_norm, weights):
    """TOPSIS Method"""
    m, n = X_norm.shape
    
    # Weighted normalized matrix
    V = X_norm * weights
    
    # Ideal and anti-ideal solutions
    V_pos = np.max(V, axis=0)
    V_neg = np.min(V, axis=0)
    
    # Distance calculation
    D_pos = np.sqrt(np.sum((V - V_pos) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((V - V_neg) ** 2, axis=1))
    
    # Closeness coefficient
    scores = D_neg / (D_pos + D_neg + 1e-10)
    return scores, D_pos, D_neg

# ==========================================================================
# 4. Evaluate Each Year
# ==========================================================================
print("\n--- 4. Evaluating Each Year (2026-2035) ---")

# Determine indicator types (1: positive, -1: negative)
indicator_types = np.ones(n_indicators)
for i, col in enumerate(indicator_cols):
    for neg in negative_indicators:
        if neg in col:
            indicator_types[i] = -1

# Storage for results
yearly_scores = np.zeros((n_countries, n_years))
yearly_ranks = np.zeros((n_countries, n_years), dtype=int)
dimension_scores = []

for y_idx, year in enumerate(pred_years):
    print(f"Processing Year {year}...")
    
    # Extract data for this year
    year_data = pred_data[pred_data['Year'] == year].copy()
    year_data = year_data.sort_values('Country').reset_index(drop=True)
    
    # Build data matrix
    X = year_data[indicator_cols].values
    
    # Normalize
    X_norm = normalize_minmax(X, indicator_types)
    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    # Calculate entropy weights
    weights = entropy_weight(X_norm)
    
    # Apply TOPSIS
    scores, _, _ = topsis(X_norm, weights)
    
    # Normalize scores to 0-100
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10) * 100
    
    yearly_scores[:, y_idx] = scores
    
    # Calculate ranks (1 = best)
    ranks = np.argsort(np.argsort(-scores)) + 1
    yearly_ranks[:, y_idx] = ranks
    
    # Calculate dimension scores
    dim_scores_year = np.zeros((n_countries, len(dim_names)))
    for d_idx, (dim_name, dim_idx) in enumerate(dimensions.items()):
        dim_X = X_norm[:, dim_idx]
        dim_weights = weights[dim_idx]
        dim_weights = dim_weights / np.sum(dim_weights)
        dim_scores_year[:, d_idx] = np.sum(dim_X * dim_weights, axis=1)
    dimension_scores.append(dim_scores_year)

print("\nEvaluation complete!")

# Get country order (sorted)
country_order = sorted(countries)

# ==========================================================================
# 5. Results Analysis
# ==========================================================================
print("\n--- 5. Results Analysis ---")

# Display final year ranking
print("\n2035 AI Competitiveness Ranking:")
print(f"{'Rank':<6} {'Country':<12} {'Score':>10} {'Change':>8}")
print("-" * 40)

final_rank_idx = np.argsort(-yearly_scores[:, -1])
for r, c_idx in enumerate(final_rank_idx):
    country = country_order[c_idx]
    initial_rank = yearly_ranks[c_idx, 0]
    final_rank = yearly_ranks[c_idx, -1]
    change = initial_rank - final_rank
    
    if change > 0:
        change_str = f"+{change}"
    elif change < 0:
        change_str = str(change)
    else:
        change_str = "0"
    
    print(f"{r+1:<6} {country:<12} {yearly_scores[c_idx, -1]:>10.2f} {change_str:>8}")

# Ranking evolution
print("\nRanking Evolution (2026-2035):")
header = f"{'Country':<12}"
for year in pred_years:
    header += f" {year}"
print(header)
print("-" * (12 + len(pred_years) * 5))

for c_idx, country in enumerate(country_order):
    row = f"{country:<12}"
    for y_idx in range(n_years):
        row += f" {yearly_ranks[c_idx, y_idx]:>4}"
    print(row)

# ==========================================================================
# 6. Key Insights
# ==========================================================================
print("\n--- 6. Key Insights ---")

# Find overtaking events
print("\nKey Overtaking Events:")
for y_idx in range(1, n_years):
    for c1 in range(n_countries):
        for c2 in range(c1 + 1, n_countries):
            # Check if c1 overtook c2
            if yearly_ranks[c1, y_idx-1] > yearly_ranks[c2, y_idx-1] and \
               yearly_ranks[c1, y_idx] < yearly_ranks[c2, y_idx]:
                print(f"  {pred_years[y_idx]}: {country_order[c1]} overtook {country_order[c2]}")
            # Check if c2 overtook c1
            if yearly_ranks[c2, y_idx-1] > yearly_ranks[c1, y_idx-1] and \
               yearly_ranks[c2, y_idx] < yearly_ranks[c1, y_idx]:
                print(f"  {pred_years[y_idx]}: {country_order[c2]} overtook {country_order[c1]}")

# Score gap analysis (Top 2)
top2_idx = np.argsort(-yearly_scores[:, 0])[:2]
top1, top2 = top2_idx[0], top2_idx[1]

print(f"\nScore Gap Analysis (Top 2: {country_order[top1]} vs {country_order[top2]}):")
for y_idx, year in enumerate(pred_years):
    gap = yearly_scores[top1, y_idx] - yearly_scores[top2, y_idx]
    print(f"  {year}: Gap = {gap:.2f} points")

# ==========================================================================
# 7. Visualization
# ==========================================================================
print("\n--- 7. Generating Visualizations ---")

# Figure 1: Score and Ranking Evolution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.tab10(np.linspace(0, 1, n_countries))

# Score Evolution
ax1 = axes[0]
for c_idx, country in enumerate(country_order):
    ax1.plot(pred_years, yearly_scores[c_idx, :], 'o-', linewidth=2,
             color=colors[c_idx], markersize=6, label=country)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Competitiveness Score', fontsize=12)
ax1.set_title('AI Competitiveness Score Evolution (2026-2035)', fontsize=13, fontweight='bold')
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(2025.5, 2035.5)

# Ranking Evolution
ax2 = axes[1]
for c_idx, country in enumerate(country_order):
    ax2.plot(pred_years, yearly_ranks[c_idx, :], 's-', linewidth=2,
             color=colors[c_idx], markersize=6, label=country)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Rank', fontsize=12)
ax2.set_title('AI Competitiveness Rank Evolution (2026-2035)', fontsize=13, fontweight='bold')
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax2.invert_yaxis()
ax2.set_ylim(n_countries + 0.5, 0.5)
ax2.set_yticks(range(1, n_countries + 1))
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2025.5, 2035.5)
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
plt.suptitle('Q321: Entropy-TOPSIS Evaluation Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q321_entropy_topsis_evolution.png', dpi=150, bbox_inches='tight')
print("Saved: Q321_entropy_topsis_evolution.png")
plt.close()

# Figure 2: Ranking Heatmap
fig, ax = plt.subplots(figsize=(12, 8))

im = ax.imshow(yearly_ranks.T, cmap='hot_r', aspect='auto')
plt.colorbar(im, label='Ranking')

ax.set_xticks(range(n_countries))
ax.set_xticklabels(country_order, rotation=45, ha='right')
ax.set_yticks(range(n_years))
ax.set_yticklabels(pred_years)
ax.set_xlabel('Country', fontsize=12)
ax.set_ylabel('Year', fontsize=12)
ax.set_title('Ranking Heatmap (2026-2035) - Entropy-TOPSIS', fontsize=14, fontweight='bold')

# Add rank numbers
for c in range(n_countries):
    for y in range(n_years):
        ax.text(c, y, str(yearly_ranks[c, y]), ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('Q321_ranking_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: Q321_ranking_heatmap.png")
plt.close()

# Figure 3: Dimension Radar for Top 4 (2035)
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

final_dim_scores = dimension_scores[-1]
final_dim_norm = final_dim_scores / (np.max(final_dim_scores, axis=0) + 1e-10)

top4_idx = np.argsort(-yearly_scores[:, -1])[:min(4, n_countries)]

angles = np.linspace(0, 2 * np.pi, len(dim_names), endpoint=False).tolist()
angles += angles[:1]

colors_radar = plt.cm.Set1(np.linspace(0, 1, 4))

for i, c_idx in enumerate(top4_idx):
    values = final_dim_norm[c_idx, :].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color=colors_radar[i],
            markersize=6, label=country_order[c_idx])
    ax.fill(angles, values, alpha=0.1, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dim_names, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title('Dimension Comparison - Top 4 Countries (2035)', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)

plt.tight_layout()
plt.savefig('Q321_dimension_radar.png', dpi=150, bbox_inches='tight')
print("Saved: Q321_dimension_radar.png")
plt.close()

# ==========================================================================
# 8. Save Results
# ==========================================================================
print("\n--- 8. Saving Results ---")

# Yearly scores table
score_df = pd.DataFrame(yearly_scores, 
                        columns=[f'Score_{y}' for y in pred_years],
                        index=country_order)
score_df.index.name = 'Country'
score_df.to_csv('Q321_yearly_scores.csv')
print("Saved: Q321_yearly_scores.csv")

# Yearly ranks table
rank_df = pd.DataFrame(yearly_ranks,
                       columns=[f'Rank_{y}' for y in pred_years],
                       index=country_order)
rank_df.index.name = 'Country'
rank_df.to_csv('Q321_yearly_ranks.csv')
print("Saved: Q321_yearly_ranks.csv")

# Final ranking summary
summary_data = {
    'Country': country_order,
    'Score_2026': yearly_scores[:, 0],
    'Rank_2026': yearly_ranks[:, 0],
    'Score_2035': yearly_scores[:, -1],
    'Rank_2035': yearly_ranks[:, -1],
    'Rank_Change': yearly_ranks[:, 0] - yearly_ranks[:, -1]
}
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Rank_2035')
summary_df.to_csv('Q321_ranking_summary.csv', index=False)
print("Saved: Q321_ranking_summary.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q321 ENTROPY-TOPSIS EVALUATION COMPLETE")
print("=" * 70)
print(f"\nMethod: Entropy Weight + TOPSIS")
print(f"Period: 2026-2035 (10 years)")
print("\n2035 Final Ranking:")
for r, c_idx in enumerate(final_rank_idx):
    print(f"  {r+1}. {country_order[c_idx]} (Score: {yearly_scores[c_idx, -1]:.2f})")
print("\nOutput Files:")
print("  - Q321_yearly_scores.csv")
print("  - Q321_yearly_ranks.csv")
print("  - Q321_ranking_summary.csv")
print("  - Q321_entropy_topsis_evolution.png")
print("  - Q321_ranking_heatmap.png")
print("  - Q321_dimension_radar.png")
print("=" * 70)

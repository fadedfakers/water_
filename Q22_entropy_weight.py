"""
=============================================================================
Huashu Cup 2026 Problem B - Question 2 Q22
Entropy Weight Method - Pure Data-Driven Evaluation
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("Q22: Entropy Weight Method - Pure Data-Driven Evaluation")
print("=" * 70)

# =============================================================================
# 1. Data Loading
# =============================================================================
print("\n--- 1. Data Loading ---")

df = pd.read_csv('panel_data_38indicators.csv')
print(f"Data dimension: {df.shape}")

# Extract 2025 data
latest_year = df['Year'].max()
df_2025 = df[df['Year'] == latest_year].copy().reset_index(drop=True)
countries = df_2025['Country'].values

indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
X = df_2025[indicator_cols].values
n, p = X.shape

# Short names
short_names = ['Top500', 'GPU_Cluster', 'DC_Compute', 'AI_Chips', '5G_Coverage',
    'Internet_BW', 'Data_Centers', 'Internet_Pen', 'AI_Researchers', 'Talent_Flow',
    'Top_Scholars', 'STEM_Grads', 'AI_Papers', 'AI_Labs', 'Gov_Investment',
    'Enterprise_RD', 'VC_Investment', 'Paper_Citations', 'AI_Patents', 'AI_Market',
    'AI_Companies', 'AI_Unicorns', 'Large_Models', 'Industry_Market', 'Tax_Incentive',
    'Subsidy_Amount', 'Policy_Count', 'Subsidy_Intensity', 'Regulatory_FW', 'GDP',
    'GDP_Growth', 'FX_Reserves', 'Population', 'Working_Age', 'Higher_Edu',
    'GII_Rank', 'RD_Density', 'FDI_Inflow']

# Dimension info
dim_names = ['Infrastructure', 'Talent', 'R&D', 'Industry', 'Policy', 'National']
dim_ranges = [(0, 8), (8, 14), (14, 20), (20, 24), (24, 29), (29, 38)]

print(f"Year: {latest_year}, Countries: {n}, Indicators: {p}")

# =============================================================================
# 2. Data Preprocessing - Normalization
# =============================================================================
print("\n--- 2. Data Normalization ---")

# Identify negative indicators (lower is better)
negative_indicators = ['GII_Rank']  # Global Innovation Index Rank (lower is better)
negative_idx = [short_names.index(name) for name in negative_indicators if name in short_names]

# Min-Max Normalization
X_norm = np.zeros_like(X, dtype=float)
for j in range(p):
    col = X[:, j]
    min_val, max_val = col.min(), col.max()
    if max_val > min_val:
        if j in negative_idx:
            # Negative indicator: reverse
            X_norm[:, j] = (max_val - col) / (max_val - min_val)
        else:
            # Positive indicator
            X_norm[:, j] = (col - min_val) / (max_val - min_val)
    else:
        X_norm[:, j] = 0.5

# Avoid zero for log calculation
X_norm = np.clip(X_norm, 1e-10, 1)

print("Normalization complete")

# =============================================================================
# 3. Entropy Weight Calculation
# =============================================================================
print("\n--- 3. Entropy Weight Calculation ---")

def entropy_weight(X):
    """
    Calculate entropy weights
    Input: X - normalized matrix (n x p)
    Output: weights (p,)
    """
    n, p = X.shape
    
    # Step 1: Calculate proportion
    X_sum = X.sum(axis=0)
    X_sum[X_sum == 0] = 1e-10
    P = X / X_sum
    
    # Step 2: Calculate entropy
    k = 1 / np.log(n)
    P_log = np.where(P > 0, P * np.log(P), 0)
    E = -k * P_log.sum(axis=0)
    
    # Step 3: Calculate utility value
    D = 1 - E
    
    # Step 4: Calculate weights
    W = D / D.sum()
    
    return W, E, D

weights, entropy, utility = entropy_weight(X_norm)

print("\nEntropy Weight Results:")
print(f"{'Indicator':<20} {'Entropy':<12} {'Utility':<12} {'Weight':<12}")
print("-" * 56)

# Sort by weight
sorted_idx = np.argsort(weights)[::-1]
for i in sorted_idx[:15]:
    print(f"{short_names[i]:<20} {entropy[i]:<12.4f} {utility[i]:<12.4f} {weights[i]:<12.4f}")

print(f"\n... (showing top 15 of {p})")
print(f"\nWeight range: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"Sum of weights: {weights.sum():.4f}")

# =============================================================================
# 4. Calculate Comprehensive Scores
# =============================================================================
print("\n--- 4. Calculate Comprehensive Scores ---")

# Weighted sum
scores = (X_norm * weights).sum(axis=1)

# Normalize to 0-100
scores_norm = (scores - scores.min()) / (scores.max() - scores.min()) * 100

# Create results
results = pd.DataFrame({
    'Country': countries,
    'Raw_Score': scores,
    'Normalized_Score': scores_norm
})
results['Rank'] = results['Normalized_Score'].rank(ascending=False).astype(int)
results = results.sort_values('Rank')

print("\n" + "=" * 60)
print("2025 AI Competitiveness Ranking (Entropy Weight Method)")
print("=" * 60)
print(f"\n{'Rank':<6} {'Country':<15} {'Score':<12}")
print("-" * 35)
for _, row in results.iterrows():
    print(f"{row['Rank']:<6} {row['Country']:<15} {row['Normalized_Score']:.2f}")

# =============================================================================
# 5. Dimension Analysis
# =============================================================================
print("\n--- 5. Dimension Analysis ---")

# Calculate dimension weights and scores
dim_weights = []
dim_scores = np.zeros((n, 6))

for d, (start, end) in enumerate(dim_ranges):
    dim_w = weights[start:end].sum()
    dim_weights.append(dim_w)
    dim_scores[:, d] = (X_norm[:, start:end] * weights[start:end]).sum(axis=1)

# Normalize dimension scores
for d in range(6):
    dim_scores[:, d] = (dim_scores[:, d] - dim_scores[:, d].min()) / \
                       (dim_scores[:, d].max() - dim_scores[:, d].min() + 1e-10) * 100

print("\nDimension Weights:")
for d, name in enumerate(dim_names):
    print(f"  {name}: {dim_weights[d]*100:.2f}%")

# =============================================================================
# 6. Visualization
# =============================================================================
print("\n--- 6. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 6.1 Weight Distribution
ax1 = axes[0, 0]
colors = []
for i in range(p):
    for d, (start, end) in enumerate(dim_ranges):
        if start <= i < end:
            colors.append(plt.cm.Set2(d/6))
            break
ax1.bar(range(p), weights, color=colors, alpha=0.8)
ax1.set_xlabel('Indicator Index')
ax1.set_ylabel('Weight')
ax1.set_title('Entropy Weights Distribution', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add dimension labels
for d, (start, end) in enumerate(dim_ranges):
    mid = (start + end) / 2
    ax1.annotate(dim_names[d], xy=(mid, weights[start:end].max()), 
                 fontsize=8, ha='center', rotation=45)

# 6.2 Ranking Bar Chart
ax2 = axes[0, 1]
colors = plt.cm.RdYlGn(results['Normalized_Score'].values / 100)
bars = ax2.barh(range(len(results)), results['Normalized_Score'].values, color=colors)
ax2.set_yticks(range(len(results)))
ax2.set_yticklabels(results['Country'].values)
ax2.set_xlabel('Normalized Score (0-100)')
ax2.set_title('2025 AI Competitiveness Ranking (Entropy Weight)', fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

for i, score in enumerate(results['Normalized_Score'].values):
    ax2.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9)

# 6.3 Dimension Weights Pie Chart
ax3 = axes[1, 0]
ax3.pie(dim_weights, labels=dim_names, autopct='%1.1f%%', 
        colors=plt.cm.Set2(np.linspace(0, 1, 6)), startangle=90)
ax3.set_title('Dimension Weight Distribution', fontweight='bold')

# 6.4 Top Indicators
ax4 = axes[1, 1]
top_n = 15
top_idx = sorted_idx[:top_n]
ax4.barh(range(top_n), weights[top_idx], color='steelblue', alpha=0.8)
ax4.set_yticks(range(top_n))
ax4.set_yticklabels([short_names[i] for i in top_idx])
ax4.set_xlabel('Weight')
ax4.set_title('Top 15 Important Indicators (by Entropy Weight)', fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.suptitle('Entropy Weight Method - AI Competitiveness Evaluation', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Q22_entropy_weight.png', bbox_inches='tight')
plt.close()
print("Saved: Q22_entropy_weight.png")

# =============================================================================
# 7. Radar Chart for Top 5 Countries
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist()
angles += angles[:1]

colors = plt.cm.Set1(np.linspace(0, 1, 5))
top5_countries = results.head(5)['Country'].values

for idx, country in enumerate(top5_countries):
    country_idx = np.where(countries == country)[0][0]
    values = dim_scores[country_idx, :].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=country, color=colors[idx])
    ax.fill(angles, values, alpha=0.1, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dim_names)
ax.set_ylim(0, 100)
ax.set_title('Top 5 Countries Dimension Comparison\n(Entropy Weight Method)', 
             fontsize=12, fontweight='bold', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('Q22_radar_chart.png', bbox_inches='tight')
plt.close()
print("Saved: Q22_radar_chart.png")

# =============================================================================
# 8. Save Results
# =============================================================================
print("\n--- 7. Saving Results ---")

# Save ranking
results.to_csv('Q22_entropy_ranking.csv', index=False)
print("Saved: Q22_entropy_ranking.csv")

# Save weights
weights_df = pd.DataFrame({
    'Indicator': short_names,
    'Entropy': entropy,
    'Utility': utility,
    'Weight': weights
})
weights_df.to_csv('Q22_entropy_weights.csv', index=False)
print("Saved: Q22_entropy_weights.csv")

# Save dimension scores
dim_scores_df = pd.DataFrame(dim_scores, columns=dim_names)
dim_scores_df['Country'] = countries
dim_scores_df = dim_scores_df[['Country'] + dim_names]
dim_scores_df.to_csv('Q22_dimension_scores.csv', index=False)
print("Saved: Q22_dimension_scores.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q22 ENTROPY WEIGHT METHOD SUMMARY")
print("=" * 70)
print(f"""
1. Method: Pure data-driven objective weighting
   - No subjective judgment required
   - Weights based on information entropy

2. Key Findings:
   - Top 3 important indicators: {', '.join([short_names[i] for i in sorted_idx[:3]])}
   - Most important dimension: {dim_names[np.argmax(dim_weights)]} ({max(dim_weights)*100:.1f}%)
   - Least important dimension: {dim_names[np.argmin(dim_weights)]} ({min(dim_weights)*100:.1f}%)

3. Ranking Results:
   - 1st: {results.iloc[0]['Country']} ({results.iloc[0]['Normalized_Score']:.2f})
   - 2nd: {results.iloc[1]['Country']} ({results.iloc[1]['Normalized_Score']:.2f})
   - Gap between 1st and 2nd: {results.iloc[0]['Normalized_Score'] - results.iloc[1]['Normalized_Score']:.2f}

4. Limitations:
   - Ignores domain knowledge
   - Sensitive to extreme values
   - May overweight low-variation indicators
""")
print("=" * 70)
print("Q22 Analysis Complete")
print("=" * 70)

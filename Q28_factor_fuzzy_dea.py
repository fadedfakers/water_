"""
=============================================================================
Huashu Cup 2026 Problem B - Question 2 Q28
Factor Analysis - Fuzzy Comprehensive - DEA Multi-Dimensional Model
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("Q28: Factor Analysis - Fuzzy Comprehensive - DEA Multi-Dimensional Model")
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
# 2. Factor Analysis
# =============================================================================
print("\n--- 2. Factor Analysis ---")

# Standardize data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Handle negative indicators
for idx in negative_idx:
    X_std[:, idx] = -X_std[:, idx]

# Determine number of factors (eigenvalue > 1 rule)
cov_matrix = np.cov(X_std.T)
eigenvalues, _ = np.linalg.eig(cov_matrix)
eigenvalues = np.real(eigenvalues)
eigenvalues = np.sort(eigenvalues)[::-1]

n_factors = np.sum(eigenvalues > 1)
n_factors = max(min(n_factors, 8), 3)  # Between 3 and 8 factors

print(f"Eigenvalues > 1: {np.sum(eigenvalues > 1)}")
print(f"Selected factors: {n_factors}")

# Factor Analysis
fa = FactorAnalysis(n_components=n_factors, random_state=42)
factor_scores = fa.fit_transform(X_std)
loadings = fa.components_.T

# Variance explained (approximate)
total_var = X_std.var(axis=0).sum()
explained_var = []
for i in range(n_factors):
    var_i = (loadings[:, i] ** 2).sum()
    explained_var.append(var_i / total_var)
explained_var = np.array(explained_var)
cumulative_var = np.cumsum(explained_var)

print(f"\nFactor Analysis Results:")
for i in range(n_factors):
    print(f"  Factor {i+1}: Variance = {explained_var[i]*100:.2f}% (Cumulative: {cumulative_var[i]*100:.2f}%)")

# Factor interpretation
print("\nFactor Interpretation (Top 5 loadings each):")
for i in range(min(4, n_factors)):
    sorted_idx = np.argsort(np.abs(loadings[:, i]))[::-1][:5]
    print(f"  Factor {i+1}:")
    for j in sorted_idx:
        print(f"    {short_names[j]}: {loadings[j, i]:.3f}")

# Calculate factor composite score
factor_weights = explained_var / explained_var.sum()
factor_composite = factor_scores @ factor_weights

# Normalize to 0-100
def normalize_100(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10) * 100

score_factor = normalize_100(factor_composite)

# =============================================================================
# 3. Fuzzy Comprehensive Evaluation
# =============================================================================
print("\n--- 3. Fuzzy Comprehensive Evaluation ---")

def fuzzy_comprehensive_evaluation(X, levels=5):
    """
    Fuzzy Comprehensive Evaluation
    
    Parameters:
    - X: normalized data (0-1)
    - levels: number of evaluation levels (default 5: excellent, good, medium, fair, poor)
    
    Returns:
    - Fuzzy evaluation results
    """
    n, p = X.shape
    
    # Define membership functions for each level
    # Trapezoidal membership functions
    level_bounds = np.linspace(0, 1, levels + 1)
    
    # Calculate membership matrix for each country
    membership_matrix = np.zeros((n, levels))
    
    for i in range(n):
        level_scores = np.zeros(levels)
        for j in range(p):
            x = X[i, j]
            for k in range(levels):
                low = level_bounds[k]
                high = level_bounds[k + 1]
                mid = (low + high) / 2
                
                # Triangular membership function
                if low <= x <= mid:
                    membership = (x - low) / (mid - low + 1e-10)
                elif mid < x <= high:
                    membership = (high - x) / (high - mid + 1e-10)
                else:
                    membership = 0
                
                # Peak membership for center value
                if k == levels - 1:  # Highest level
                    if x >= level_bounds[k]:
                        membership = max(membership, (x - level_bounds[k]) / (1 - level_bounds[k] + 1e-10))
                if k == 0:  # Lowest level
                    if x <= level_bounds[1]:
                        membership = max(membership, (level_bounds[1] - x) / (level_bounds[1] + 1e-10))
                
                level_scores[k] += membership
        
        # Normalize
        total = level_scores.sum()
        if total > 0:
            membership_matrix[i] = level_scores / total
    
    # Calculate fuzzy comprehensive score
    # Higher level = higher weight
    level_values = np.array([20, 40, 60, 80, 100])  # Score for each level
    fuzzy_scores = membership_matrix @ level_values
    
    return fuzzy_scores, membership_matrix

# Normalize data for fuzzy evaluation
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

score_fuzzy, membership_matrix = fuzzy_comprehensive_evaluation(X_norm)

print("Fuzzy Evaluation Levels: Excellent(100), Good(80), Medium(60), Fair(40), Poor(20)")
print("\nMembership Matrix:")
level_names = ['Poor', 'Fair', 'Medium', 'Good', 'Excellent']
mem_df = pd.DataFrame(membership_matrix, columns=level_names, index=countries)
print(mem_df.round(3))

# =============================================================================
# 4. DEA (Data Envelopment Analysis)
# =============================================================================
print("\n--- 4. DEA Analysis ---")

def dea_ccr(inputs, outputs):
    """
    DEA CCR Model (Constant Returns to Scale)
    
    For each DMU, solve:
    max θ = Σ u_r × y_rk / Σ v_i × x_ik
    s.t. Σ u_r × y_rj / Σ v_i × x_ij <= 1 for all j
         u, v >= ε
    
    Simplified using linear programming transformation
    """
    n_dmu = inputs.shape[0]
    n_inputs = inputs.shape[1]
    n_outputs = outputs.shape[1]
    
    efficiencies = np.zeros(n_dmu)
    
    for k in range(n_dmu):
        # Decision variables: [u (outputs), v (inputs)]
        # Objective: max Σ u_r × y_rk (with Σ v_i × x_ik = 1 constraint)
        
        # Coefficients for objective (maximize output virtual value)
        c = np.concatenate([-outputs[k], np.zeros(n_inputs)])
        
        # Inequality constraints: u'Y - v'X <= 0 for all DMUs
        A_ub = np.zeros((n_dmu, n_outputs + n_inputs))
        for j in range(n_dmu):
            A_ub[j, :n_outputs] = outputs[j]
            A_ub[j, n_outputs:] = -inputs[j]
        b_ub = np.zeros(n_dmu)
        
        # Equality constraint: v'x_k = 1
        A_eq = np.zeros((1, n_outputs + n_inputs))
        A_eq[0, n_outputs:] = inputs[k]
        b_eq = np.array([1.0])
        
        # Bounds (non-negative weights)
        bounds = [(1e-6, None) for _ in range(n_outputs + n_inputs)]
        
        # Solve
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            if result.success:
                efficiencies[k] = -result.fun
            else:
                efficiencies[k] = 0.5
        except:
            efficiencies[k] = 0.5
    
    return efficiencies

# Define inputs and outputs for AI competitiveness
# Inputs: Resources invested (infrastructure, talent pool, R&D investment)
# Outputs: Results achieved (enterprises, patents, market size)

# Input indicators
input_idx = [
    0,   # Top500 supercomputers
    1,   # GPU clusters
    8,   # AI researchers
    14,  # Government investment
    16,  # VC investment
    29,  # GDP (resource base)
]

# Output indicators
output_idx = [
    18,  # AI patents
    19,  # AI market size
    20,  # AI companies
    21,  # AI unicorns
    22,  # Large models
]

# Prepare DEA data (ensure positive values)
X_input = X[:, input_idx]
X_output = X[:, output_idx]

# Add small constant to avoid zeros
X_input = X_input - X_input.min(axis=0) + 1
X_output = X_output - X_output.min(axis=0) + 1

# Run DEA
dea_efficiency = dea_ccr(X_input, X_output)

print("DEA Analysis Setup:")
print(f"  Inputs ({len(input_idx)}): {[short_names[i] for i in input_idx]}")
print(f"  Outputs ({len(output_idx)}): {[short_names[i] for i in output_idx]}")

print("\nDEA Efficiency Scores:")
for i, country in enumerate(countries):
    status = "Efficient" if dea_efficiency[i] >= 0.99 else "Inefficient"
    print(f"  {country}: {dea_efficiency[i]:.4f} ({status})")

score_dea = normalize_100(dea_efficiency)

# =============================================================================
# 5. Combine Three Methods
# =============================================================================
print("\n--- 5. Combine Three Methods ---")

# All three scores are now on 0-100 scale
print("Individual Scores:")
print(f"{'Country':<15} {'Factor':<10} {'Fuzzy':<10} {'DEA':<10}")
print("-" * 45)
for i, country in enumerate(countries):
    print(f"{country:<15} {score_factor[i]:.2f}     {score_fuzzy[i]:.2f}     {score_dea[i]:.2f}")

# Weighted combination
method_weights = {
    'Factor': 0.40,   # Statistical strength
    'Fuzzy': 0.30,    # Handles uncertainty
    'DEA': 0.30       # Efficiency perspective
}

combined_score = (method_weights['Factor'] * score_factor + 
                  method_weights['Fuzzy'] * score_fuzzy + 
                  method_weights['DEA'] * score_dea)

# =============================================================================
# 6. Create Rankings
# =============================================================================
print("\n--- 6. Results ---")

results = pd.DataFrame({
    'Country': countries,
    'Factor_Score': score_factor,
    'Fuzzy_Score': score_fuzzy,
    'DEA_Score': score_dea,
    'DEA_Efficiency': dea_efficiency,
    'Combined_Score': combined_score
})

results['Rank_Factor'] = results['Factor_Score'].rank(ascending=False).astype(int)
results['Rank_Fuzzy'] = results['Fuzzy_Score'].rank(ascending=False).astype(int)
results['Rank_DEA'] = results['DEA_Score'].rank(ascending=False).astype(int)
results['Rank_Combined'] = results['Combined_Score'].rank(ascending=False).astype(int)

results = results.sort_values('Rank_Combined')

print("\n" + "=" * 85)
print("2025 AI Competitiveness Ranking (Factor-Fuzzy-DEA Combination)")
print("=" * 85)
print(f"\n{'Country':<15} {'Factor':<10} {'Fuzzy':<10} {'DEA':<10} {'Combined':<12} {'Rank':<6}")
print("-" * 70)
for _, row in results.iterrows():
    print(f"{row['Country']:<15} {row['Factor_Score']:.2f}     {row['Fuzzy_Score']:.2f}     "
          f"{row['DEA_Score']:.2f}     {row['Combined_Score']:.2f}       {row['Rank_Combined']}")

# =============================================================================
# 7. Multi-Dimensional Insight Analysis
# =============================================================================
print("\n--- 7. Multi-Dimensional Insights ---")

# Classification based on three perspectives
print("\nCountry Classification by Three Perspectives:")
print("-" * 60)

for i, country in enumerate(countries):
    factor_r = results[results['Country'] == country]['Rank_Factor'].values[0]
    fuzzy_r = results[results['Country'] == country]['Rank_Fuzzy'].values[0]
    dea_r = results[results['Country'] == country]['Rank_DEA'].values[0]
    dea_eff = results[results['Country'] == country]['DEA_Efficiency'].values[0]
    
    # Classify
    strength_high = factor_r <= 3
    fuzzy_good = fuzzy_r <= 3
    efficient = dea_eff >= 0.8
    
    insights = []
    if strength_high and efficient:
        insights.append("Strong & Efficient (Leader)")
    elif strength_high and not efficient:
        insights.append("Strong but Inefficient (Room for improvement)")
    elif not strength_high and efficient:
        insights.append("Efficient but Developing (Potential)")
    else:
        insights.append("Needs Development")
    
    print(f"{country:<15}: Factor R{factor_r}, Fuzzy R{fuzzy_r}, DEA R{dea_r} (Eff={dea_eff:.2f})")
    print(f"                → {insights[0]}")

# =============================================================================
# 8. Visualization
# =============================================================================
print("\n--- 8. Generating Visualizations ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 8.1 Factor Loadings Heatmap
ax1 = axes[0, 0]
loadings_subset = loadings[:, :min(5, n_factors)]
im = ax1.imshow(loadings_subset, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_yticks(range(0, p, 2))
ax1.set_yticklabels([short_names[i][:12] for i in range(0, p, 2)], fontsize=7)
ax1.set_xticks(range(loadings_subset.shape[1]))
ax1.set_xticklabels([f'F{i+1}' for i in range(loadings_subset.shape[1])])
ax1.set_title('Factor Loadings Matrix', fontweight='bold')
plt.colorbar(im, ax=ax1)

# 8.2 Fuzzy Membership Stacked Bar
ax2 = axes[0, 1]
bottom = np.zeros(n)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, 5))
for k, level in enumerate(level_names):
    ax2.bar(range(n), membership_matrix[:, k], bottom=bottom, 
            label=level, color=colors[k], alpha=0.8)
    bottom += membership_matrix[:, k]
ax2.set_xticks(range(n))
ax2.set_xticklabels(countries, rotation=45, ha='right')
ax2.set_ylabel('Membership Degree')
ax2.set_title('Fuzzy Membership Distribution', fontweight='bold')
ax2.legend(loc='upper right')

# 8.3 DEA Efficiency
ax3 = axes[0, 2]
colors = ['green' if e >= 0.99 else 'orange' if e >= 0.8 else 'red' for e in dea_efficiency]
ax3.barh(range(n), dea_efficiency, color=colors, alpha=0.8)
ax3.axvline(x=1.0, color='green', linestyle='--', label='Efficient Frontier')
ax3.axvline(x=0.8, color='orange', linestyle='--', label='80% Threshold')
ax3.set_yticks(range(n))
ax3.set_yticklabels(countries)
ax3.set_xlabel('DEA Efficiency')
ax3.set_title('DEA Efficiency Scores', fontweight='bold')
ax3.legend()
ax3.invert_yaxis()

# 8.4 Three Methods Comparison
ax4 = axes[1, 0]
x = np.arange(len(results))
width = 0.25
ax4.bar(x - width, results['Factor_Score'].values, width, label='Factor', alpha=0.8)
ax4.bar(x, results['Fuzzy_Score'].values, width, label='Fuzzy', alpha=0.8)
ax4.bar(x + width, results['DEA_Score'].values, width, label='DEA', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(results['Country'].values, rotation=45, ha='right')
ax4.set_ylabel('Score (0-100)')
ax4.set_title('Three Methods Comparison', fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 8.5 Final Ranking
ax5 = axes[1, 1]
colors = plt.cm.RdYlGn(results['Combined_Score'].values / 100)
ax5.barh(range(len(results)), results['Combined_Score'].values, color=colors)
ax5.set_yticks(range(len(results)))
ax5.set_yticklabels(results['Country'].values)
ax5.set_xlabel('Combined Score (0-100)')
ax5.set_title('2025 AI Competitiveness Ranking\n(Factor-Fuzzy-DEA)', fontweight='bold')
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)
for i, score in enumerate(results['Combined_Score'].values):
    ax5.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9)

# 8.6 Strength vs Efficiency Scatter
ax6 = axes[1, 2]
scatter = ax6.scatter(results['Factor_Score'].values, results['DEA_Efficiency'].values * 100,
                      c=results['Combined_Score'].values, cmap='RdYlGn',
                      s=200, edgecolors='black', linewidths=1)
for _, row in results.iterrows():
    ax6.annotate(row['Country'], (row['Factor_Score'], row['DEA_Efficiency']*100),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)
ax6.axhline(y=80, color='orange', linestyle='--', alpha=0.7)
ax6.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
ax6.set_xlabel('Factor Score (Strength)')
ax6.set_ylabel('DEA Efficiency (%)')
ax6.set_title('Strength vs Efficiency Matrix', fontweight='bold')
ax6.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='Combined Score')

# Add quadrant labels
ax6.text(75, 90, 'Leaders', fontsize=10, fontweight='bold', color='green')
ax6.text(25, 90, 'Efficient\nDevelopers', fontsize=9, color='blue')
ax6.text(75, 70, 'Inefficient\nGiants', fontsize=9, color='orange')
ax6.text(25, 70, 'Challengers', fontsize=9, color='gray')

plt.suptitle('Factor-Fuzzy-DEA AI Competitiveness Evaluation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Q28_factor_fuzzy_dea.png', bbox_inches='tight')
plt.close()
print("Saved: Q28_factor_fuzzy_dea.png")

# Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2*np.pi, 6, endpoint=False).tolist()
angles += angles[:1]

# Calculate dimension scores (using factor scores approach)
dim_scores = np.zeros((n, 6))
for d, (start, end) in enumerate(dim_ranges):
    dim_scores[:, d] = X_norm[:, start:end].mean(axis=1)

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
ax.set_title('Top 5 Countries Dimension Comparison', fontweight='bold', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('Q28_radar_chart.png', bbox_inches='tight')
plt.close()
print("Saved: Q28_radar_chart.png")

# =============================================================================
# 9. Save Results
# =============================================================================
print("\n--- 9. Saving Results ---")

results.to_csv('Q28_factor_fuzzy_dea_ranking.csv', index=False)

# Factor loadings
loadings_df = pd.DataFrame(loadings, index=short_names,
                           columns=[f'Factor{i+1}' for i in range(n_factors)])
loadings_df.to_csv('Q28_factor_loadings.csv')

# Fuzzy membership
mem_df.to_csv('Q28_fuzzy_membership.csv')

# DEA details
dea_df = pd.DataFrame({
    'Country': countries,
    'DEA_Efficiency': dea_efficiency,
    'Input_Total': X_input.sum(axis=1),
    'Output_Total': X_output.sum(axis=1)
})
dea_df.to_csv('Q28_dea_results.csv', index=False)

print("Saved: Q28_factor_fuzzy_dea_ranking.csv")
print("Saved: Q28_factor_loadings.csv")
print("Saved: Q28_fuzzy_membership.csv")
print("Saved: Q28_dea_results.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Q28 FACTOR-FUZZY-DEA SUMMARY")
print("=" * 70)
print(f"""
Three-Dimensional Evaluation:

1. Factor Analysis (40%): "How strong is the overall capability?"
   - Factors extracted: {n_factors}
   - Variance explained: {cumulative_var[-1]*100:.1f}%
   - Perspective: Statistical strength assessment

2. Fuzzy Comprehensive (30%): "What quality level does it belong to?"
   - Evaluation levels: 5 (Poor → Excellent)
   - Perspective: Handles uncertainty and imprecision

3. DEA Analysis (30%): "How efficient is the resource utilization?"
   - Inputs: {len(input_idx)} indicators
   - Outputs: {len(output_idx)} indicators
   - Efficient DMUs: {np.sum(dea_efficiency >= 0.99)}
   - Perspective: Input-output efficiency

Combined Ranking (2025):
1. {results.iloc[0]['Country']}: {results.iloc[0]['Combined_Score']:.2f}
2. {results.iloc[1]['Country']}: {results.iloc[1]['Combined_Score']:.2f}
3. {results.iloc[2]['Country']}: {results.iloc[2]['Combined_Score']:.2f}

Key Insights:
- Strong & Efficient: Countries with high factor scores AND high DEA efficiency
- Inefficient Giants: High strength but room for efficiency improvement
- Efficient Developers: Lower resources but good utilization
""")
print("=" * 70)

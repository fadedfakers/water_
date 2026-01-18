"""
==========================================================================
Huashu Cup 2026 Problem B - Question 4 Q412
Differential Evolution (DE) Global Optimization
Optimal Investment Allocation for China's AI Competitiveness
==========================================================================

Method: Differential Evolution - Population-based stochastic optimization
Characteristics:
- Global optimization (explores entire search space)
- No gradient required
- Handles multimodal objective functions
- More robust but slower than gradient methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
print("=" * 70)
print("Q412: Differential Evolution Optimization")
print("Global Stochastic Search Algorithm")
print("=" * 70)

# ==========================================================================
# 1. Data Loading and Preprocessing
# ==========================================================================
print("\n--- 1. Loading Data ---")

hist_data = pd.read_csv('panel_data_38indicators.csv')
countries = sorted(hist_data['Country'].unique())
hist_years = sorted(hist_data['Year'].unique())
indicator_cols = [col for col in hist_data.columns if col not in ['Country', 'Year']]

n_countries = len(countries)
n_indicators = len(indicator_cols)

print(f"Countries: {n_countries}")
print(f"Indicators: {n_indicators}")

# Negative indicators
negative_indicators = ['全球创新指数排名']
indicator_types = np.ones(n_indicators)
for i, col in enumerate(indicator_cols):
    for neg in negative_indicators:
        if neg in col:
            indicator_types[i] = -1

# ==========================================================================
# 2. Define Investment Dimensions
# ==========================================================================
print("\n--- 2. Investment Dimensions ---")

dimensions = {
    'Computing_Power': {
        'indices': list(range(0, 8)),
        'name_cn': '算力基础',
        'efficiency': 0.8,
        'lag_years': 2,
    },
    'Talent': {
        'indices': list(range(8, 13)),
        'name_cn': '人才资源',
        'efficiency': 0.6,
        'lag_years': 4,
    },
    'Innovation': {
        'indices': list(range(13, 19)),
        'name_cn': '创新能力',
        'efficiency': 0.5,
        'lag_years': 5,
    },
    'Industry': {
        'indices': list(range(19, 24)),
        'name_cn': '产业发展',
        'efficiency': 0.9,
        'lag_years': 2,
    },
    'Policy': {
        'indices': list(range(24, 29)),
        'name_cn': '政策环境',
        'efficiency': 0.7,
        'lag_years': 1,
    },
    'Economy': {
        'indices': list(range(29, 38)),
        'name_cn': '经济基础',
        'efficiency': 0.4,
        'lag_years': 3,
    }
}

n_dims = len(dimensions)
dim_names = list(dimensions.keys())

for name, info in dimensions.items():
    print(f"  {name} ({info['name_cn']}): {len(info['indices'])} indicators")

# ==========================================================================
# 3. Calculate Baseline Scores
# ==========================================================================
print("\n--- 3. Calculating Baseline Scores ---")

china_idx = countries.index('中国')
usa_idx = countries.index('美国')

def normalize_minmax(X, indicator_types):
    m, n = X.shape
    X_norm = np.zeros((m, n))
    for j in range(n):
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
    return X_norm

def entropy_weight(X):
    m, n = X.shape
    P = X / (np.sum(X, axis=0) + 1e-10)
    k = 1 / np.log(m)
    entropy = np.zeros(n)
    for j in range(n):
        p = P[:, j]
        p[p < 1e-10] = 1e-10
        entropy[j] = -k * np.sum(p * np.log(p))
    weights = (1 - entropy) / (np.sum(1 - entropy) + 1e-10)
    return weights

# Global weights calculation
all_hist_X = []
for year in hist_years:
    year_data = hist_data[hist_data['Year'] == year].sort_values('Country')
    X = year_data[indicator_cols].values
    all_hist_X.append(X)
all_hist_X = np.vstack(all_hist_X)
all_X_norm = normalize_minmax(all_hist_X, indicator_types)
global_weights = entropy_weight(all_X_norm)

# Dimension weights
dim_weights = {}
for name, info in dimensions.items():
    dim_weights[name] = np.sum(global_weights[info['indices']])

# 2025 baseline
data_2025 = hist_data[hist_data['Year'] == 2025].sort_values('Country')
X_2025 = data_2025[indicator_cols].values
X_2025_norm = normalize_minmax(X_2025, indicator_types)

china_norm = X_2025_norm[china_idx]
usa_norm = X_2025_norm[usa_idx]

# Dimension scores
dim_china_scores = {}
dim_usa_scores = {}
dim_gaps = {}

for name, info in dimensions.items():
    idx = info['indices']
    w = global_weights[idx] / np.sum(global_weights[idx])
    dim_china_scores[name] = np.sum(china_norm[idx] * w) * 100
    dim_usa_scores[name] = np.sum(usa_norm[idx] * w) * 100
    dim_gaps[name] = dim_usa_scores[name] - dim_china_scores[name]

print("\nChina-USA Gap by Dimension:")
for name in dim_names:
    print(f"  {name}: China={dim_china_scores[name]:.1f}, USA={dim_usa_scores[name]:.1f}, Gap={dim_gaps[name]:+.1f}")

# ==========================================================================
# 4. Investment Response Function
# ==========================================================================
print("\n--- 4. Investment Response Model ---")

def estimate_improvement_rate(dim_name):
    info = dimensions[dim_name]
    idx = info['indices']
    china_hist = hist_data[hist_data['Country'] == '中国'].sort_values('Year')
    
    improvements = []
    for i in range(1, len(hist_years)):
        data_prev = china_hist[china_hist['Year'] == hist_years[i-1]][indicator_cols].values.flatten()
        data_curr = china_hist[china_hist['Year'] == hist_years[i]][indicator_cols].values.flatten()
        
        for j in idx:
            if indicator_types[j] == 1 and data_prev[j] > 0:
                improvements.append((data_curr[j] - data_prev[j]) / data_prev[j])
            elif indicator_types[j] == -1 and data_prev[j] > 0:
                improvements.append((data_prev[j] - data_curr[j]) / data_prev[j])
    
    return np.mean(improvements) if improvements else 0.05

base_improvement = {name: max(estimate_improvement_rate(name), 0.01) for name in dim_names}

def investment_response(investment, dim_name, current_score, years=10):
    info = dimensions[dim_name]
    efficiency = info['efficiency']
    lag = info['lag_years']
    
    effective_years = max(years - lag, 1)
    ceiling_factor = ((100 - current_score) / 100) ** 0.5
    
    reference_investment = 2000
    investment_effect = 1 - np.exp(-investment / reference_investment)
    
    base_capacity = min(base_improvement[dim_name] * effective_years * 100, 50)
    investment_capacity = 50 * efficiency * investment_effect
    
    improvement = (base_capacity + investment_capacity) * ceiling_factor
    max_improvement = (100 - current_score) * 0.85
    
    return max(min(improvement, max_improvement), 0)

# ==========================================================================
# 5. TOPSIS Scoring Function
# ==========================================================================
print("\n--- 5. TOPSIS Scoring Function ---")

def calculate_topsis_score(china_scores_new):
    china_data_new = X_2025_norm[china_idx].copy()
    
    for name, score in china_scores_new.items():
        idx = dimensions[name]['indices']
        current_dim_score = dim_china_scores[name]
        if current_dim_score > 0:
            ratio = score / current_dim_score
        else:
            ratio = 1
        china_data_new[idx] = X_2025_norm[china_idx, idx] * min(ratio, 2.0)
    
    data_matrix = X_2025_norm.copy()
    data_matrix[china_idx] = china_data_new
    
    data_norm = np.zeros_like(data_matrix)
    for j in range(n_indicators):
        col = data_matrix[:, j]
        min_val, max_val = np.min(col), np.max(col)
        if max_val - min_val < 1e-10:
            data_norm[:, j] = 1
        else:
            if indicator_types[j] == 1:
                data_norm[:, j] = (col - min_val) / (max_val - min_val)
            else:
                data_norm[:, j] = (max_val - col) / (max_val - min_val)
    data_norm[data_norm < 0.0001] = 0.0001
    
    V = data_norm * global_weights
    V_pos = np.max(V, axis=0)
    V_neg = np.min(V, axis=0)
    
    D_pos = np.sqrt(np.sum((V - V_pos) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((V - V_neg) ** 2, axis=1))
    
    scores = D_neg / (D_pos + D_neg + 1e-10)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10) * 100
    
    return scores[china_idx], scores

# Baseline
baseline_scores = {name: dim_china_scores[name] for name in dim_names}
china_baseline, all_scores_baseline = calculate_topsis_score(baseline_scores)
usa_baseline = all_scores_baseline[usa_idx]

print(f"China Baseline Score: {china_baseline:.2f}")
print(f"USA Baseline Score: {usa_baseline:.2f}")

# ==========================================================================
# 6. Objective Function with Constraints
# ==========================================================================
print("\n--- 6. Building Optimization Model ---")

TOTAL_BUDGET = 10000  # 1万亿 = 10000亿
min_bound = 0.05 * TOTAL_BUDGET  # 5% minimum
max_bound = 0.40 * TOTAL_BUDGET  # 40% maximum

def marginal_benefit(x):
    new_scores = {}
    for i, name in enumerate(dim_names):
        improvement = investment_response(x[i], name, dim_china_scores[name])
        new_scores[name] = dim_china_scores[name] + improvement
    china_score, _ = calculate_topsis_score(new_scores)
    return china_score

def objective_de(x):
    """
    Differential Evolution Objective with penalty for constraint violations
    """
    score = marginal_benefit(x)
    
    # Penalty for budget violation
    budget_diff = abs(np.sum(x) - TOTAL_BUDGET)
    penalty = 10 * budget_diff
    
    # Penalty for bound violations
    min_violation = max(0, min_bound - np.min(x))
    max_violation = max(0, np.max(x) - max_bound)
    penalty += 50 * (min_violation + max_violation)
    
    # Diversification bonus
    shares = x / (np.sum(x) + 1e-10)
    hhi = np.sum(shares ** 2)
    diversification_bonus = (1 - hhi) * 5
    
    # Weight alignment bonus
    weight_alignment = sum(shares[i] * dim_weights[name] for i, name in enumerate(dim_names))
    alignment_bonus = weight_alignment * 3
    
    return -(score + diversification_bonus + alignment_bonus - penalty)

# ==========================================================================
# 7. Differential Evolution Optimization
# ==========================================================================
print("\n--- 7. Differential Evolution Optimization ---")

bounds = [(min_bound, max_bound) for _ in range(n_dims)]

print(f"Budget: {TOTAL_BUDGET}亿元")
print(f"Bounds: {min_bound:.0f}亿 to {max_bound:.0f}亿 per dimension")
print(f"Method: Differential Evolution (DE/rand/1/bin)")
print("\nDE Parameters:")
print(f"  Population size: 15 × {n_dims} = {15 * n_dims}")
print(f"  Mutation: (0.5, 1.0)")
print(f"  Recombination: 0.7")
print(f"  Max iterations: 500")

# Callback for progress tracking
best_scores = []
def callback(xk, convergence):
    score = marginal_benefit(xk)
    best_scores.append(score)
    if len(best_scores) % 50 == 0:
        print(f"  Iteration {len(best_scores)}: Score = {score:.2f}")
    return False

print("\nRunning Differential Evolution...")
result = differential_evolution(
    objective_de,
    bounds=bounds,
    maxiter=500,
    seed=42,
    workers=1,
    polish=True,          # Local refinement at the end
    mutation=(0.5, 1.0),  # Mutation factor range
    recombination=0.7,    # Crossover probability
    strategy='best1bin',  # DE strategy
    tol=1e-7,
    callback=callback,
    disp=False
)

# Normalize to exactly use budget
optimal_x = result.x / np.sum(result.x) * TOTAL_BUDGET
optimal_x = np.clip(optimal_x, min_bound, max_bound)
optimal_x = optimal_x / np.sum(optimal_x) * TOTAL_BUDGET

optimal_score = marginal_benefit(optimal_x)

print(f"\nOptimization Result:")
print(f"  Success: {result.success}")
print(f"  Iterations: {result.nit}")
print(f"  Function evaluations: {result.nfev}")
print(f"  Final Score: {optimal_score:.2f}")

# ==========================================================================
# 8. Results Analysis
# ==========================================================================
print("\n--- 8. Results Analysis ---")

# Calculate final scores
final_scores = {}
for i, name in enumerate(dim_names):
    improvement = investment_response(optimal_x[i], name, dim_china_scores[name])
    final_scores[name] = dim_china_scores[name] + improvement

china_final, all_final = calculate_topsis_score(final_scores)

print("\n" + "=" * 70)
print("Q412 DIFFERENTIAL EVOLUTION OPTIMAL INVESTMENT ALLOCATION")
print("=" * 70)

print(f"\n{'Dimension':<18} {'Investment':>12} {'Pct':>8} {'2025':>8} {'2035':>8} {'Δ':>8}")
print("-" * 66)

for i, name in enumerate(dim_names):
    inv = optimal_x[i]
    pct = inv / TOTAL_BUDGET * 100
    score_2025 = dim_china_scores[name]
    score_2035 = final_scores[name]
    delta = score_2035 - score_2025
    print(f"{name:<18} {inv:>10.0f}亿 {pct:>7.1f}% {score_2025:>8.1f} {score_2035:>8.1f} {delta:>+7.1f}")

print("-" * 66)
print(f"{'Total':<18} {np.sum(optimal_x):>10.0f}亿 {100:>7.1f}%")

# Performance metrics
hhi = np.sum((optimal_x / TOTAL_BUDGET) ** 2)
print(f"\nPerformance Metrics:")
print(f"  China Score: {china_baseline:.2f} → {china_final:.2f} (+{china_final - china_baseline:.2f})")
print(f"  USA Gap: {china_baseline - usa_baseline:.2f} → {china_final - all_final[usa_idx]:.2f}")
print(f"  HHI (diversification): {hhi:.4f} (lower is more diversified)")
print(f"  Function evaluations: {result.nfev}")

# ==========================================================================
# 9. Visualization
# ==========================================================================
print("\n--- 9. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Investment allocation pie chart
ax1 = axes[0, 0]
colors = plt.cm.Set2(np.linspace(0, 1, n_dims))
wedges, texts, autotexts = ax1.pie(
    optimal_x,
    labels=[f"{dimensions[name]['name_cn']}\n{optimal_x[i]:.0f}亿" for i, name in enumerate(dim_names)],
    autopct='%1.1f%%',
    colors=colors,
    explode=[0.02] * n_dims
)
ax1.set_title('DE Optimal Investment Allocation', fontsize=12, fontweight='bold')

# 2. Convergence history
ax2 = axes[0, 1]
if best_scores:
    ax2.plot(range(1, len(best_scores) + 1), best_scores, 'b-', linewidth=1.5)
    ax2.axhline(y=optimal_score, color='r', linestyle='--', label=f'Final: {optimal_score:.2f}')
    ax2.axhline(y=china_baseline, color='gray', linestyle=':', label=f'Baseline: {china_baseline:.2f}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Score')
ax2.set_title('DE Convergence History', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Before vs After comparison
ax3 = axes[1, 0]
x = np.arange(n_dims)
width = 0.35

bars1 = ax3.bar(x - width/2, [dim_china_scores[name] for name in dim_names],
                width, label='2025', color='steelblue', alpha=0.7)
bars2 = ax3.bar(x + width/2, [final_scores[name] for name in dim_names],
                width, label='2035', color='coral', alpha=0.7)

ax3.set_xticks(x)
ax3.set_xticklabels([dimensions[name]['name_cn'] for name in dim_names], rotation=30, ha='right')
ax3.set_ylabel('Score')
ax3.set_title('Dimension Score Improvement', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Gap reduction
ax4 = axes[1, 1]
gaps_2025 = [dim_gaps[name] for name in dim_names]
gaps_2035 = [dim_usa_scores[name] - final_scores[name] for name in dim_names]

ax4.bar(x - width/2, gaps_2025, width, label='2025 Gap', color='coral', alpha=0.7)
ax4.bar(x + width/2, gaps_2035, width, label='2035 Gap', color='steelblue', alpha=0.7)
ax4.axhline(y=0, color='black', linewidth=1)

ax4.set_xticks(x)
ax4.set_xticklabels([dimensions[name]['name_cn'] for name in dim_names], rotation=30, ha='right')
ax4.set_ylabel('Gap (USA - China)')
ax4.set_title('China-USA Gap Reduction', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Q412: Differential Evolution Optimization Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q412_de_optimization.png', dpi=150, bbox_inches='tight')
print("Saved: Q412_de_optimization.png")
plt.close()

# ==========================================================================
# 10. Save Results
# ==========================================================================
print("\n--- 10. Saving Results ---")

results_df = pd.DataFrame({
    'Dimension': dim_names,
    'Dimension_CN': [dimensions[name]['name_cn'] for name in dim_names],
    'Investment_Billion': optimal_x,
    'Percentage': optimal_x / TOTAL_BUDGET * 100,
    'Score_2025': [dim_china_scores[name] for name in dim_names],
    'Score_2035': [final_scores[name] for name in dim_names],
    'Improvement': [final_scores[name] - dim_china_scores[name] for name in dim_names],
    'Gap_2025': [dim_gaps[name] for name in dim_names],
    'Gap_2035': [dim_usa_scores[name] - final_scores[name] for name in dim_names]
})
results_df.to_csv('Q412_de_results.csv', index=False)
print("Saved: Q412_de_results.csv")

# Summary
summary_df = pd.DataFrame({
    'Metric': ['Method', 'Total_Budget', 'China_Score_2025', 'China_Score_2035', 
               'Score_Improvement', 'HHI_Index', 'Optimization_Success', 
               'Iterations', 'Function_Evaluations'],
    'Value': ['Differential_Evolution', TOTAL_BUDGET, china_baseline, china_final,
              china_final - china_baseline, hhi, result.success, result.nit, result.nfev]
})
summary_df.to_csv('Q412_de_summary.csv', index=False)
print("Saved: Q412_de_summary.csv")

# Convergence history
if best_scores:
    conv_df = pd.DataFrame({
        'Iteration': range(1, len(best_scores) + 1),
        'Score': best_scores
    })
    conv_df.to_csv('Q412_de_convergence.csv', index=False)
    print("Saved: Q412_de_convergence.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q412 DIFFERENTIAL EVOLUTION OPTIMIZATION COMPLETE")
print("=" * 70)
print(f"""
Method: Differential Evolution (DE/best1bin)
Characteristics:
  - Global stochastic optimization
  - Population-based search
  - {result.nfev} function evaluations
  - Polished with local optimizer

Optimal Allocation:
""")
for i, name in enumerate(dim_names):
    print(f"  {dimensions[name]['name_cn']}: {optimal_x[i]:.0f}亿 ({optimal_x[i]/TOTAL_BUDGET*100:.1f}%)")

print(f"""
Results:
  - Score: {china_baseline:.2f} → {china_final:.2f} (+{china_final-china_baseline:.2f})
  - Diversification (HHI): {hhi:.4f}
  
Output Files:
  - Q412_de_results.csv
  - Q412_de_summary.csv
  - Q412_de_convergence.csv
  - Q412_de_optimization.png
""")
print("=" * 70)

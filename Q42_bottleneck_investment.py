"""
==========================================================================
Huashu Cup 2026 Problem B - Question 4 Q42
Bottleneck-First Investment Strategy (短板优先法)
Based on the Barrel Theory (木桶原理)
==========================================================================

Core Principle:
- China's AI competitiveness ceiling is determined by its weakest dimension
- Identify 1-2 dimensions with the largest gap to USA
- Prioritize investment to fill these bottlenecks first

Strategy: "A barrel's capacity is determined by its shortest stave"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q42: Bottleneck-First Investment Strategy")
print("短板优先法 - 基于木桶原理")
print("=" * 70)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
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
# 3. Calculate Baseline Scores and Identify Bottlenecks
# ==========================================================================
print("\n--- 3. Identifying Bottlenecks (短板分析) ---")

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

# Calculate dimension scores and gaps
dim_china_scores = {}
dim_usa_scores = {}
dim_gaps = {}

for name, info in dimensions.items():
    idx = info['indices']
    w = global_weights[idx] / np.sum(global_weights[idx])
    dim_china_scores[name] = np.sum(china_norm[idx] * w) * 100
    dim_usa_scores[name] = np.sum(usa_norm[idx] * w) * 100
    dim_gaps[name] = dim_usa_scores[name] - dim_china_scores[name]

# Sort by gap (largest gap = biggest bottleneck)
sorted_by_gap = sorted(dim_gaps.items(), key=lambda x: -x[1])

print("\n" + "=" * 60)
print("BOTTLENECK ANALYSIS (短板分析)")
print("=" * 60)
print(f"\n{'Rank':<6} {'Dimension':<18} {'China':>8} {'USA':>8} {'Gap':>10} {'Status':<12}")
print("-" * 70)

bottlenecks = []
for rank, (name, gap) in enumerate(sorted_by_gap):
    china_score = dim_china_scores[name]
    usa_score = dim_usa_scores[name]
    
    if gap > 20:
        status = "🔴 严重短板"
        bottlenecks.append(name)
    elif gap > 0:
        status = "🟡 轻微落后"
        if len(bottlenecks) < 2:
            bottlenecks.append(name)
    else:
        status = "🟢 领先"
    
    print(f"{rank+1:<6} {name:<18} {china_score:>8.1f} {usa_score:>8.1f} {gap:>+10.1f} {status:<12}")

print("-" * 70)
print(f"\n识别的关键短板 (Top Bottlenecks): {bottlenecks[:2]}")

# ==========================================================================
# 4. Barrel Theory Model (木桶模型)
# ==========================================================================
print("\n--- 4. Barrel Theory Model (木桶原理模型) ---")

"""
Barrel Theory Score Model:
Overall score is heavily influenced by the weakest dimension.

Score = α × min(dim_scores) + (1-α) × weighted_average(dim_scores)

Where α controls the bottleneck effect strength:
- α = 0: Pure weighted average (no bottleneck effect)
- α = 1: Pure minimum (extreme bottleneck effect)
- α = 0.3~0.5: Balanced model (recommended)
"""

def barrel_score(dim_scores_dict, alpha=0.4):
    """
    Calculate overall score using barrel theory
    
    Parameters:
    - dim_scores_dict: {dim_name: score}
    - alpha: bottleneck effect strength (0-1)
    
    Returns:
    - Overall score considering bottleneck effect
    """
    scores = np.array([dim_scores_dict[name] for name in dim_names])
    weights = np.array([dim_weights[name] for name in dim_names])
    weights = weights / np.sum(weights)
    
    # Minimum score (bottleneck)
    min_score = np.min(scores)
    
    # Weighted average
    weighted_avg = np.sum(scores * weights)
    
    # Barrel score: combination of min and average
    barrel = alpha * min_score + (1 - alpha) * weighted_avg
    
    return barrel, min_score, weighted_avg

# Current barrel score
current_barrel, current_min, current_avg = barrel_score(dim_china_scores)
print(f"\nCurrent China Scores:")
print(f"  Weighted Average: {current_avg:.2f}")
print(f"  Minimum (Bottleneck): {current_min:.2f} ({sorted_by_gap[0][0]})")
print(f"  Barrel Score (α=0.4): {current_barrel:.2f}")

# USA barrel score for reference
usa_barrel, usa_min, usa_avg = barrel_score(dim_usa_scores)
print(f"\nUSA Scores (Reference):")
print(f"  Weighted Average: {usa_avg:.2f}")
print(f"  Minimum: {usa_min:.2f}")
print(f"  Barrel Score (α=0.4): {usa_barrel:.2f}")

print(f"\nChina-USA Barrel Gap: {current_barrel - usa_barrel:.2f}")

# ==========================================================================
# 5. Investment Response Function
# ==========================================================================
print("\n--- 5. Investment Response Model ---")

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
    """Calculate score improvement given investment"""
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
# 6. TOPSIS Scoring Function
# ==========================================================================
def calculate_topsis_score(china_scores_new):
    """Calculate TOPSIS score for given dimension scores"""
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

# Baseline TOPSIS
baseline_scores_dict = {name: dim_china_scores[name] for name in dim_names}
china_baseline_topsis, all_scores_baseline = calculate_topsis_score(baseline_scores_dict)
usa_baseline_topsis = all_scores_baseline[usa_idx]

print(f"\nBaseline TOPSIS Scores:")
print(f"  China: {china_baseline_topsis:.2f}")
print(f"  USA: {usa_baseline_topsis:.2f}")

# ==========================================================================
# 7. Bottleneck-First Allocation Strategy
# ==========================================================================
print("\n--- 7. Bottleneck-First Allocation Strategy ---")

TOTAL_BUDGET = 10000  # 1万亿 = 10000亿
min_bound = 0.03 * TOTAL_BUDGET  # 3% minimum (allow more concentration)
max_bound = 0.50 * TOTAL_BUDGET  # 50% maximum (allow more focus on bottlenecks)

print(f"\nBudget: {TOTAL_BUDGET}亿元")
print(f"Bounds: {min_bound:.0f}亿 to {max_bound:.0f}亿 per dimension")
print(f"Strategy: Prioritize bottleneck dimensions")

# ==========================================================================
# Strategy 1: Pure Bottleneck Focus (纯短板聚焦)
# ==========================================================================
print("\n--- Strategy 1: Pure Bottleneck Focus ---")
print("Allocate 70% to top 2 bottlenecks, 30% to others")

bottleneck_ratio = 0.70  # 70% to bottlenecks
other_ratio = 0.30       # 30% to others

allocation_s1 = {}
top_bottlenecks = bottlenecks[:2]  # Top 2 bottlenecks
other_dims = [name for name in dim_names if name not in top_bottlenecks]

# Allocate to bottlenecks based on gap size
bottleneck_gaps = {name: max(dim_gaps[name], 1) for name in top_bottlenecks}
total_bottleneck_gap = sum(bottleneck_gaps.values())

for name in top_bottlenecks:
    allocation_s1[name] = (bottleneck_gaps[name] / total_bottleneck_gap) * (bottleneck_ratio * TOTAL_BUDGET)

# Allocate to others based on weights
other_weights = {name: dim_weights[name] for name in other_dims}
total_other_weight = sum(other_weights.values())

for name in other_dims:
    allocation_s1[name] = (other_weights[name] / total_other_weight) * (other_ratio * TOTAL_BUDGET)

# Apply bounds
for name in dim_names:
    allocation_s1[name] = np.clip(allocation_s1[name], min_bound, max_bound)

total_alloc = sum(allocation_s1.values())
for name in dim_names:
    allocation_s1[name] = allocation_s1[name] / total_alloc * TOTAL_BUDGET

optimal_x_s1 = np.array([allocation_s1[name] for name in dim_names])

print(f"Top 2 Bottlenecks: {top_bottlenecks}")
for name in dim_names:
    marker = "★" if name in top_bottlenecks else " "
    print(f"  {marker} {name}: {allocation_s1[name]:.0f}亿 ({allocation_s1[name]/TOTAL_BUDGET*100:.1f}%)")

# ==========================================================================
# Strategy 2: Gap-Proportional Allocation (差距比例分配)
# ==========================================================================
print("\n--- Strategy 2: Gap-Proportional Allocation ---")
print("Allocate proportionally to gap size (larger gap → more investment)")

allocation_s2 = {}

# Use gap for behind dimensions, weight for leading dimensions
priority_scores = {}
for name in dim_names:
    gap = dim_gaps[name]
    weight = dim_weights[name]
    
    if gap > 0:  # Behind USA - prioritize by gap
        priority_scores[name] = gap * weight * 2  # Amplify gap effect
    else:  # Ahead of USA - use weight only
        priority_scores[name] = weight * 0.5

total_priority = sum(priority_scores.values())
for name in dim_names:
    allocation_s2[name] = (priority_scores[name] / total_priority) * TOTAL_BUDGET

# Apply bounds
for name in dim_names:
    allocation_s2[name] = np.clip(allocation_s2[name], min_bound, max_bound)

total_alloc = sum(allocation_s2.values())
for name in dim_names:
    allocation_s2[name] = allocation_s2[name] / total_alloc * TOTAL_BUDGET

optimal_x_s2 = np.array([allocation_s2[name] for name in dim_names])

for name in dim_names:
    gap = dim_gaps[name]
    marker = "★" if gap > 20 else ("●" if gap > 0 else " ")
    print(f"  {marker} {name}: {allocation_s2[name]:.0f}亿 ({allocation_s2[name]/TOTAL_BUDGET*100:.1f}%) [Gap={gap:+.1f}]")

# ==========================================================================
# Strategy 3: Barrel Optimization (木桶优化)
# ==========================================================================
print("\n--- Strategy 3: Barrel Theory Optimization ---")
print("Optimize to maximize barrel score (weighted by bottleneck effect)")

def barrel_objective(x, alpha=0.4):
    """
    Objective: Maximize barrel score
    Higher alpha = stronger bottleneck penalty
    """
    # Calculate new dimension scores
    new_scores = {}
    for i, name in enumerate(dim_names):
        improvement = investment_response(x[i], name, dim_china_scores[name])
        new_scores[name] = dim_china_scores[name] + improvement
    
    # Calculate barrel score
    barrel, min_score, avg_score = barrel_score(new_scores, alpha)
    
    # Penalty for budget violation
    budget_diff = abs(np.sum(x) - TOTAL_BUDGET)
    penalty = 10 * budget_diff
    
    # Penalty for bound violations
    min_violation = max(0, min_bound - np.min(x))
    max_violation = max(0, np.max(x) - max_bound)
    penalty += 50 * (min_violation + max_violation)
    
    return -(barrel - penalty)

# Initial guess: focus on bottlenecks
x0 = np.array([allocation_s1[name] for name in dim_names])

bounds = [(min_bound, max_bound) for _ in range(n_dims)]

# Optimize with different alpha values
print("\nOptimizing with different bottleneck effect strengths (α):")
alpha_results = []

for alpha in [0.2, 0.4, 0.6]:
    result = minimize(
        lambda x: barrel_objective(x, alpha),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - TOTAL_BUDGET},
        options={'maxiter': 500}
    )
    
    # Calculate scores
    new_scores = {}
    for i, name in enumerate(dim_names):
        improvement = investment_response(result.x[i], name, dim_china_scores[name])
        new_scores[name] = dim_china_scores[name] + improvement
    
    barrel, min_s, avg_s = barrel_score(new_scores, alpha)
    topsis, _ = calculate_topsis_score(new_scores)
    
    alpha_results.append({
        'alpha': alpha,
        'x': result.x,
        'barrel': barrel,
        'min_score': min_s,
        'topsis': topsis,
        'success': result.success
    })
    
    print(f"  α={alpha}: Barrel={barrel:.2f}, Min={min_s:.2f}, TOPSIS={topsis:.2f}")

# Select best alpha based on TOPSIS (practical outcome)
best_alpha_result = max(alpha_results, key=lambda r: r['topsis'])
optimal_x_s3 = best_alpha_result['x']
best_alpha = best_alpha_result['alpha']

print(f"\nBest α = {best_alpha} selected")

# ==========================================================================
# 8. Compare Strategies
# ==========================================================================
print("\n--- 8. Strategy Comparison ---")

def evaluate_allocation(x, strategy_name):
    """Evaluate an allocation"""
    new_scores = {}
    for i, name in enumerate(dim_names):
        improvement = investment_response(x[i], name, dim_china_scores[name])
        new_scores[name] = dim_china_scores[name] + improvement
    
    barrel, min_s, avg_s = barrel_score(new_scores)
    topsis, _ = calculate_topsis_score(new_scores)
    hhi = np.sum((x / TOTAL_BUDGET) ** 2)
    
    # Bottleneck improvement
    bottleneck_improvement = 0
    for bn in top_bottlenecks:
        bottleneck_improvement += new_scores[bn] - dim_china_scores[bn]
    
    return {
        'name': strategy_name,
        'x': x,
        'new_scores': new_scores,
        'barrel': barrel,
        'min_score': min_s,
        'avg_score': avg_s,
        'topsis': topsis,
        'hhi': hhi,
        'bottleneck_improvement': bottleneck_improvement
    }

strategies = [
    evaluate_allocation(optimal_x_s1, "S1: Pure Bottleneck Focus"),
    evaluate_allocation(optimal_x_s2, "S2: Gap-Proportional"),
    evaluate_allocation(optimal_x_s3, f"S3: Barrel Optimization (α={best_alpha})")
]

print(f"\n{'Strategy':<35} {'Barrel':>8} {'Min':>8} {'TOPSIS':>8} {'HHI':>8} {'BN Δ':>8}")
print("-" * 80)
for s in strategies:
    print(f"{s['name']:<35} {s['barrel']:>8.2f} {s['min_score']:>8.2f} {s['topsis']:>8.2f} {s['hhi']:>8.4f} {s['bottleneck_improvement']:>+8.1f}")

# Select best strategy (based on barrel score as this is the focus)
best_strategy = max(strategies, key=lambda s: s['barrel'])
print(f"\n*** Best Strategy: {best_strategy['name']} ***")

optimal_x = best_strategy['x']
final_scores = best_strategy['new_scores']

# ==========================================================================
# 9. Results Analysis
# ==========================================================================
print("\n--- 9. Results Analysis ---")

china_final_topsis, all_final = calculate_topsis_score(final_scores)
final_barrel, final_min, final_avg = barrel_score(final_scores)

print("\n" + "=" * 70)
print("Q42 BOTTLENECK-FIRST OPTIMAL INVESTMENT ALLOCATION")
print("=" * 70)

print(f"\n{'Dimension':<18} {'Investment':>12} {'Pct':>8} {'2025':>8} {'2035':>8} {'Δ':>8} {'Gap→':>10}")
print("-" * 78)

for i, name in enumerate(dim_names):
    inv = optimal_x[i]
    pct = inv / TOTAL_BUDGET * 100
    score_2025 = dim_china_scores[name]
    score_2035 = final_scores[name]
    delta = score_2035 - score_2025
    gap_2025 = dim_gaps[name]
    gap_2035 = dim_usa_scores[name] - score_2035
    
    # Mark bottlenecks
    marker = "★★" if name in top_bottlenecks[:1] else ("★" if name in top_bottlenecks else "  ")
    
    print(f"{marker}{name:<16} {inv:>10.0f}亿 {pct:>7.1f}% {score_2025:>8.1f} {score_2035:>8.1f} {delta:>+7.1f} {gap_2025:>+5.1f}→{gap_2035:>+.1f}")

print("-" * 78)
print(f"{'Total':<18} {np.sum(optimal_x):>10.0f}亿 {100:>7.1f}%")

print(f"\n{'Metric':<30} {'2025':>15} {'2035':>15} {'Change':>15}")
print("-" * 75)
print(f"{'TOPSIS Score':<30} {china_baseline_topsis:>15.2f} {china_final_topsis:>15.2f} {china_final_topsis-china_baseline_topsis:>+15.2f}")
print(f"{'Barrel Score (α=0.4)':<30} {current_barrel:>15.2f} {final_barrel:>15.2f} {final_barrel-current_barrel:>+15.2f}")
print(f"{'Minimum Dimension Score':<30} {current_min:>15.2f} {final_min:>15.2f} {final_min-current_min:>+15.2f}")
print(f"{'Weighted Average Score':<30} {current_avg:>15.2f} {final_avg:>15.2f} {final_avg-current_avg:>+15.2f}")

# Bottleneck analysis
print(f"\nBottleneck (短板) Analysis:")
print(f"  Original bottleneck: {sorted_by_gap[0][0]} (Score: {dim_china_scores[sorted_by_gap[0][0]]:.1f})")
new_scores_sorted = sorted(final_scores.items(), key=lambda x: x[1])
print(f"  New bottleneck: {new_scores_sorted[0][0]} (Score: {new_scores_sorted[0][1]:.1f})")
print(f"  Bottleneck improvement: +{final_min - current_min:.1f} points")

# ==========================================================================
# 10. Visualization
# ==========================================================================
print("\n--- 10. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Investment allocation with bottleneck highlighting
ax1 = axes[0, 0]
colors = ['#FF6B6B' if name in top_bottlenecks else '#4ECDC4' for name in dim_names]
bars = ax1.bar(range(n_dims), optimal_x, color=colors, edgecolor='black', linewidth=1.5)

ax1.set_xticks(range(n_dims))
ax1.set_xticklabels([dimensions[name]['name_cn'] for name in dim_names], rotation=30, ha='right')
ax1.set_ylabel('Investment (亿元)')
ax1.set_title('Bottleneck-First Investment Allocation\n(Red = Bottleneck Dimensions)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, (bar, val) in enumerate(zip(bars, optimal_x)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val/TOTAL_BUDGET*100:.1f}%', ha='center', va='bottom', fontsize=9)

# 2. Before vs After with gap visualization
ax2 = axes[0, 1]
x = np.arange(n_dims)
width = 0.25

bars1 = ax2.bar(x - width, [dim_china_scores[name] for name in dim_names],
                width, label='China 2025', color='steelblue', alpha=0.7)
bars2 = ax2.bar(x, [final_scores[name] for name in dim_names],
                width, label='China 2035', color='coral', alpha=0.7)
bars3 = ax2.bar(x + width, [dim_usa_scores[name] for name in dim_names],
                width, label='USA 2025', color='seagreen', alpha=0.7)

ax2.set_xticks(x)
ax2.set_xticklabels([dimensions[name]['name_cn'] for name in dim_names], rotation=30, ha='right')
ax2.set_ylabel('Score')
ax2.set_title('Dimension Score Comparison', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Barrel theory visualization
ax3 = axes[1, 0]

# Create barrel-like visualization
angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
angles += angles[:1]

china_2025 = [dim_china_scores[name] for name in dim_names]
china_2035 = [final_scores[name] for name in dim_names]
usa_scores = [dim_usa_scores[name] for name in dim_names]

china_2025 += china_2025[:1]
china_2035 += china_2035[:1]
usa_scores += usa_scores[:1]

ax3 = plt.subplot(2, 2, 3, projection='polar')
ax3.plot(angles, china_2025, 'o-', linewidth=2, color='steelblue', label='China 2025', alpha=0.7)
ax3.plot(angles, china_2035, 's-', linewidth=2, color='coral', label='China 2035')
ax3.plot(angles, usa_scores, '^--', linewidth=2, color='seagreen', label='USA 2025', alpha=0.7)
ax3.fill(angles, china_2035, alpha=0.1, color='coral')

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels([dimensions[name]['name_cn'] for name in dim_names], fontsize=9)
ax3.set_ylim(0, 100)
ax3.set_title('Barrel Diagram (木桶图)\nBottleneck Improvement', fontsize=12, fontweight='bold', pad=20)
ax3.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), ncol=3, fontsize=8)

# 4. Strategy comparison
ax4 = plt.subplot(2, 2, 4)
strategy_names = [s['name'].replace('S', 'Strategy ').split(':')[0] for s in strategies]
barrel_scores = [s['barrel'] for s in strategies]
topsis_scores = [s['topsis'] for s in strategies]

x = np.arange(len(strategies))
width = 0.35

bars1 = ax4.bar(x - width/2, barrel_scores, width, label='Barrel Score', color='steelblue')
bars2 = ax4.bar(x + width/2, topsis_scores, width, label='TOPSIS Score', color='coral')

ax4.set_xticks(x)
ax4.set_xticklabels(['S1: Bottleneck\nFocus', 'S2: Gap\nProportional', f'S3: Barrel\nOptimized'], fontsize=9)
ax4.set_ylabel('Score')
ax4.set_title('Strategy Comparison', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Mark best strategy
best_idx = strategies.index(best_strategy)
ax4.annotate('★ Best', xy=(best_idx, barrel_scores[best_idx]), xytext=(best_idx, barrel_scores[best_idx] + 3),
             ha='center', fontsize=10, fontweight='bold', color='red')

plt.suptitle('Q42: Bottleneck-First Investment Strategy (短板优先法)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q42_bottleneck_optimization.png', dpi=150, bbox_inches='tight')
print("Saved: Q42_bottleneck_optimization.png")
plt.close()

# ==========================================================================
# 11. Save Results
# ==========================================================================
print("\n--- 11. Saving Results ---")

# Main results
results_df = pd.DataFrame({
    'Dimension': dim_names,
    'Dimension_CN': [dimensions[name]['name_cn'] for name in dim_names],
    'Is_Bottleneck': [name in top_bottlenecks for name in dim_names],
    'Investment_Billion': optimal_x,
    'Percentage': optimal_x / TOTAL_BUDGET * 100,
    'Score_2025': [dim_china_scores[name] for name in dim_names],
    'Score_2035': [final_scores[name] for name in dim_names],
    'Improvement': [final_scores[name] - dim_china_scores[name] for name in dim_names],
    'Gap_2025': [dim_gaps[name] for name in dim_names],
    'Gap_2035': [dim_usa_scores[name] - final_scores[name] for name in dim_names],
    'USA_Score': [dim_usa_scores[name] for name in dim_names]
})
results_df.to_csv('Q42_bottleneck_results.csv', index=False)
print("Saved: Q42_bottleneck_results.csv")

# Strategy comparison
strategy_df = pd.DataFrame([{
    'Strategy': s['name'],
    'Barrel_Score': s['barrel'],
    'Min_Score': s['min_score'],
    'Avg_Score': s['avg_score'],
    'TOPSIS_Score': s['topsis'],
    'HHI': s['hhi'],
    'Bottleneck_Improvement': s['bottleneck_improvement']
} for s in strategies])
strategy_df.to_csv('Q42_strategy_comparison.csv', index=False)
print("Saved: Q42_strategy_comparison.csv")

# Bottleneck analysis
bottleneck_df = pd.DataFrame({
    'Dimension': dim_names,
    'China_Score': [dim_china_scores[name] for name in dim_names],
    'USA_Score': [dim_usa_scores[name] for name in dim_names],
    'Gap': [dim_gaps[name] for name in dim_names],
    'Is_Bottleneck': [name in top_bottlenecks for name in dim_names],
    'Bottleneck_Rank': [sorted_by_gap.index((name, dim_gaps[name])) + 1 if (name, dim_gaps[name]) in sorted_by_gap else 0 for name in dim_names]
})
bottleneck_df = bottleneck_df.sort_values('Gap', ascending=False)
bottleneck_df.to_csv('Q42_bottleneck_analysis.csv', index=False)
print("Saved: Q42_bottleneck_analysis.csv")

# Summary
summary_df = pd.DataFrame({
    'Metric': ['Method', 'Total_Budget', 'Top_Bottlenecks', 'TOPSIS_2025', 'TOPSIS_2035',
               'Barrel_2025', 'Barrel_2035', 'Min_Score_2025', 'Min_Score_2035',
               'Bottleneck_Improvement', 'Best_Strategy'],
    'Value': ['Bottleneck_First', TOTAL_BUDGET, str(top_bottlenecks), 
              china_baseline_topsis, china_final_topsis,
              current_barrel, final_barrel, current_min, final_min,
              final_min - current_min, best_strategy['name']]
})
summary_df.to_csv('Q42_summary.csv', index=False)
print("Saved: Q42_summary.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q42 BOTTLENECK-FIRST STRATEGY COMPLETE")
print("=" * 70)
print(f"""
Core Principle: 木桶原理 (Barrel Theory)
"A barrel's capacity is determined by its shortest stave"

Identified Bottlenecks (短板):
  1. {top_bottlenecks[0]} (Gap: {dim_gaps[top_bottlenecks[0]]:+.1f})
  2. {top_bottlenecks[1]} (Gap: {dim_gaps[top_bottlenecks[1]]:+.1f})

Best Strategy: {best_strategy['name']}

Optimal Allocation:
""")
for i, name in enumerate(dim_names):
    marker = "★" if name in top_bottlenecks else " "
    print(f"  {marker} {dimensions[name]['name_cn']}: {optimal_x[i]:.0f}亿 ({optimal_x[i]/TOTAL_BUDGET*100:.1f}%)")

print(f"""
Key Results:
  TOPSIS Score: {china_baseline_topsis:.2f} → {china_final_topsis:.2f} (+{china_final_topsis-china_baseline_topsis:.2f})
  Barrel Score: {current_barrel:.2f} → {final_barrel:.2f} (+{final_barrel-current_barrel:.2f})
  Min Dimension: {current_min:.2f} → {final_min:.2f} (+{final_min-current_min:.2f})

Strategic Insight:
  By prioritizing bottleneck dimensions, China can raise its 
  competitiveness "floor" and achieve more balanced development.

Output Files:
  - Q42_bottleneck_results.csv
  - Q42_strategy_comparison.csv
  - Q42_bottleneck_analysis.csv
  - Q42_summary.csv
  - Q42_bottleneck_optimization.png
""")
print("=" * 70)

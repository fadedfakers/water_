"""
==========================================================================
Huashu Cup 2026 Problem B - Question 4 Q413
Priority-Based Heuristic Investment Allocation
Optimal Investment Allocation for China's AI Competitiveness
==========================================================================

Method: Multi-criteria priority scoring heuristic
Characteristics:
- Rule-based allocation using domain knowledge
- Considers: gap, weight, efficiency, improvement rate
- Fast computation, interpretable results
- No optimization iterations required
"""
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
print("=" * 70)
print("Q413: Priority-Based Heuristic Allocation")
print("Multi-Criteria Decision Making Approach")
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
    },# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
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
# 4. Historical Improvement Rates
# ==========================================================================
print("\n--- 4. Historical Improvement Analysis ---")

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

print("Historical Improvement Rates:")
for name in dim_names:
    print(f"  {name}: {base_improvement[name]*100:.1f}% per year")

# ==========================================================================
# 5. Investment Response Function
# ==========================================================================
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
# 6. TOPSIS Scoring Function# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
#     # =============================================================================
# ==========================================================================
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
baseline_scores_dict = {name: dim_china_scores[name] for name in dim_names}
china_baseline, all_scores_baseline = calculate_topsis_score(baseline_scores_dict)
usa_baseline = all_scores_baseline[usa_idx]

print(f"\nChina Baseline Score: {china_baseline:.2f}")
print(f"USA Baseline Score: {usa_baseline:.2f}")

# ==========================================================================
# 7. Priority-Based Heuristic Allocation
# ==========================================================================
print("\n--- 7. Priority-Based Heuristic Allocation ---")

TOTAL_BUDGET = 10000  # 1万亿 = 10000亿
min_bound = 0.05 * TOTAL_BUDGET  # 5% minimum
max_bound = 0.40 * TOTAL_BUDGET  # 40% maximum

print(f"\nBudget: {TOTAL_BUDGET}亿元")
print(f"Bounds: {min_bound:.0f}亿 to {max_bound:.0f}亿 per dimension")

# ==========================================================================
# Method 1: Gap-Weighted Priority
# ==========================================================================
print("\n--- Method 1: Gap-Weighted Priority ---")
print("Priority = Weight × (1 + Gap/50) × Efficiency × (1 + ImprovementRate)")

priority_scores_m1 = {}
for name in dim_names:
    w = dim_weights[name]
    gap = dim_gaps[name]
    eff = dimensions[name]['efficiency']
    imp_rate = base_improvement[name]
    
    if gap > 0:  # Behind USA
        priority = w * (1 + gap/50) * eff * (1 + imp_rate)
    else:  # Ahead of USA
        priority = w * 0.5 * eff * (1 + imp_rate)
    
    priority_scores_m1[name] = priority
    print(f"  {name}: Weight={w:.3f}, Gap={gap:+.1f}, Eff={eff}, Priority={priority:.4f}")

# Normalize and allocate
total_priority = sum(priority_scores_m1.values())
allocation_m1 = {}
for name in dim_names:
    raw = (priority_scores_m1[name] / total_priority) * TOTAL_BUDGET
    allocation_m1[name] = np.clip(raw, min_bound, max_bound)

# Adjust to meet budget
total_alloc = sum(allocation_m1.values())
for name in dim_names:
    allocation_m1[name] = allocation_m1[name] / total_alloc * TOTAL_BUDGET

optimal_x_m1 = np.array([allocation_m1[name] for name in dim_names])

# ==========================================================================
# Method 2: Balanced Multi-Factor Priority
# ==========================================================================
print("\n--- Method 2: Balanced Multi-Factor Priority ---")
print("Priority = α×Weight + β×Gap_Score + γ×Efficiency + δ×ImprovementRate")

# Normalize factors
weights_norm = np.array([dim_weights[name] for name in dim_names])
weights_norm = weights_norm / np.max(weights_norm)

gaps_raw = np.array([max(dim_gaps[name], 0) for name in dim_names])
gaps_norm = gaps_raw / (np.max(gaps_raw) + 1e-10)

eff_raw = np.array([dimensions[name]['efficiency'] for name in dim_names])
eff_norm = eff_raw / np.max(eff_raw)

imp_raw = np.array([base_improvement[name] for name in dim_names])
imp_norm = imp_raw / (np.max(imp_raw) + 1e-10)

# Weights for factors
alpha, beta, gamma, delta = 0.3, 0.3, 0.2, 0.2

priority_scores_m2 = alpha * weights_norm + beta * gaps_norm + gamma * eff_norm + delta * imp_norm

print(f"Factor weights: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
for i, name in enumerate(dim_names):
    print(f"  {name}: W={weights_norm[i]:.2f}, G={gaps_norm[i]:.2f}, "
          f"E={eff_norm[i]:.2f}, I={imp_norm[i]:.2f} → Priority={priority_scores_m2[i]:.3f}")

# Allocate
priority_scores_m2 = priority_scores_m2 / np.sum(priority_scores_m2)
allocation_m2 = priority_scores_m2 * TOTAL_BUDGET
allocation_m2 = np.clip(allocation_m2, min_bound, max_bound)
allocation_m2 = allocation_m2 / np.sum(allocation_m2) * TOTAL_BUDGET

optimal_x_m2 = allocation_m2

# ==========================================================================
# Method 3: Strategic Focus Allocation
# ==========================================================================
print("\n--- Method 3: Strategic Focus Allocation ---")
print("Categorize dimensions and allocate by strategic importance")

# Strategic categories
catching_up = []    # Behind USA, high priority
leading = []        # Ahead of USA
foundation = []     # Support dimensions

for name in dim_names:
    if dim_gaps[name] > 10:
        catching_up.append(name)
    elif dim_gaps[name] < -5:
        leading.append(name)
    else:
        foundation.append(name)

print(f"  Catching Up (Gap > 10): {catching_up}")
print(f"  Leading (Gap < -5): {leading}")
print(f"  Foundation (-5 to 10): {foundation}")

# Allocation ratios by category
catching_up_ratio = 0.55  # 55% to catching up areas
leading_ratio = 0.20      # 20% to maintain leadership
foundation_ratio = 0.25   # 25% to foundation

allocation_m3 = {}

# Allocate within each category based on weights
if catching_up:
    catching_weights = {name: dim_weights[name] for name in catching_up}
    total_w = sum(catching_weights.values())
    for name in catching_up:
        allocation_m3[name] = (catching_weights[name] / total_w) * (catching_up_ratio * TOTAL_BUDGET)

if leading:
    leading_weights = {name: dim_weights[name] for name in leading}
    total_w = sum(leading_weights.values())
    for name in leading:
        allocation_m3[name] = (leading_weights[name] / total_w) * (leading_ratio * TOTAL_BUDGET)

if foundation:
    foundation_weights = {name: dim_weights[name] for name in foundation}
    total_w = sum(foundation_weights.values())
    for name in foundation:
        allocation_m3[name] = (foundation_weights[name] / total_w) * (foundation_ratio * TOTAL_BUDGET)

# Apply bounds
for name in dim_names:
    if name not in allocation_m3:
        allocation_m3[name] = min_bound
    allocation_m3[name] = np.clip(allocation_m3[name], min_bound, max_bound)

total_alloc = sum(allocation_m3.values())
for name in dim_names:
    allocation_m3[name] = allocation_m3[name] / total_alloc * TOTAL_BUDGET

optimal_x_m3 = np.array([allocation_m3[name] for name in dim_names])

# ==========================================================================
# 8. Compare Heuristic Methods
# ==========================================================================
print("\n--- 8. Comparing Heuristic Methods ---")

def marginal_benefit(x):
    new_scores = {}
    for i, name in enumerate(dim_names):
        improvement = investment_response(x[i], name, dim_china_scores[name])
        new_scores[name] = dim_china_scores[name] + improvement
    china_score, _ = calculate_topsis_score(new_scores)
    return china_score

score_m1 = marginal_benefit(optimal_x_m1)
score_m2 = marginal_benefit(optimal_x_m2)
score_m3 = marginal_benefit(optimal_x_m3)

hhi_m1 = np.sum((optimal_x_m1 / TOTAL_BUDGET) ** 2)
hhi_m2 = np.sum((optimal_x_m2 / TOTAL_BUDGET) ** 2)
hhi_m3 = np.sum((optimal_x_m3 / TOTAL_BUDGET) ** 2)

print(f"\n{'Method':<30} {'Score':>10} {'HHI':>10}")
print("-" * 52)
print(f"{'M1: Gap-Weighted Priority':<30} {score_m1:>10.2f} {hhi_m1:>10.4f}")
print(f"{'M2: Balanced Multi-Factor':<30} {score_m2:>10.2f} {hhi_m2:>10.4f}")
print(f"{'M3: Strategic Focus':<30} {score_m3:>10.2f} {hhi_m3:>10.4f}")

# Select best method
methods = [
    ('Gap-Weighted Priority', optimal_x_m1, score_m1, hhi_m1),
    ('Balanced Multi-Factor', optimal_x_m2, score_m2, hhi_m2),
    ('Strategic Focus', optimal_x_m3, score_m3, hhi_m3)
]

best_method = max(methods, key=lambda x: x[2])
print(f"\n*** Best Heuristic: {best_method[0]} (Score: {best_method[2]:.2f}) ***")

optimal_x = best_method[1]
optimal_score = best_method[2]
best_method_name = best_method[0]

# ==========================================================================
# 9. Results Analysis
# ==========================================================================
print("\n--- 9. Results Analysis ---")

# Calculate final scores
final_scores = {}
for i, name in enumerate(dim_names):
    improvement = investment_response(optimal_x[i], name, dim_china_scores[name])
    final_scores[name] = dim_china_scores[name] + improvement

china_final, all_final = calculate_topsis_score(final_scores)

print("\n" + "=" * 70)
print(f"Q413 HEURISTIC OPTIMAL INVESTMENT ALLOCATION ({best_method_name})")
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

hhi = np.sum((optimal_x / TOTAL_BUDGET) ** 2)
print(f"\nPerformance Metrics:")
print(f"  China Score: {china_baseline:.2f} → {china_final:.2f} (+{china_final - china_baseline:.2f})")
print(f"  USA Gap: {china_baseline - usa_baseline:.2f} → {china_final - all_final[usa_idx]:.2f}")
print(f"  HHI (diversification): {hhi:.4f}")

# ==========================================================================
# 10. Visualization
# ==========================================================================
print("\n--- 10. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Compare three heuristic methods
ax1 = axes[0, 0]
x = np.arange(n_dims)
width = 0.25

ax1.bar(x - width, optimal_x_m1 / TOTAL_BUDGET * 100, width, label='M1: Gap-Weighted', color='steelblue')
ax1.bar(x, optimal_x_m2 / TOTAL_BUDGET * 100, width, label='M2: Multi-Factor', color='coral')
ax1.bar(x + width, optimal_x_m3 / TOTAL_BUDGET * 100, width, label='M3: Strategic', color='seagreen')

ax1.set_xticks(x)
ax1.set_xticklabels([dimensions[name]['name_cn'] for name in dim_names], rotation=30, ha='right')
ax1.set_ylabel('Allocation (%)')
ax1.set_title('Heuristic Methods Comparison', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Best method allocation pie
ax2 = axes[0, 1]
colors = plt.cm.Pastel1(np.linspace(0, 1, n_dims))
wedges, texts, autotexts = ax2.pie(
    optimal_x,
    labels=[f"{dimensions[name]['name_cn']}\n{optimal_x[i]:.0f}亿" for i, name in enumerate(dim_names)],
    autopct='%1.1f%%',
    colors=colors,
    explode=[0.02] * n_dims
)
ax2.set_title(f'Best Heuristic: {best_method_name}', fontsize=12, fontweight='bold')

# 3. Priority factors visualization
ax3 = axes[1, 0]
factor_data = np.array([weights_norm, gaps_norm, eff_norm, imp_norm])
factor_names = ['Weight', 'Gap', 'Efficiency', 'Improvement']

im = ax3.imshow(factor_data, cmap='YlOrRd', aspect='auto')
ax3.set_xticks(range(n_dims))
ax3.set_xticklabels([dimensions[name]['name_cn'] for name in dim_names], rotation=30, ha='right')
ax3.set_yticks(range(4))
ax3.set_yticklabels(factor_names)
ax3.set_title('Priority Factor Scores (Normalized)', fontsize=12, fontweight='bold')

for i in range(4):
    for j in range(n_dims):
        ax3.text(j, i, f'{factor_data[i, j]:.2f}', ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=ax3)

# 4. Score improvement comparison
ax4 = axes[1, 1]
improvements = [final_scores[name] - dim_china_scores[name] for name in dim_names]
investments = optimal_x / TOTAL_BUDGET * 100

colors_scatter = ['red' if dim_gaps[name] > 0 else 'green' for name in dim_names]
scatter = ax4.scatter(investments, improvements, c=colors_scatter, s=200, edgecolors='black', linewidths=2)

for i, name in enumerate(dim_names):
    ax4.annotate(dimensions[name]['name_cn'], (investments[i], improvements[i]),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)

ax4.set_xlabel('Investment Share (%)')
ax4.set_ylabel('Score Improvement')
ax4.set_title('Investment vs Improvement (Red=Behind USA, Green=Ahead)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle('Q413: Priority-Based Heuristic Optimization Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q413_heuristic_optimization.png', dpi=150, bbox_inches='tight')
print("Saved: Q413_heuristic_optimization.png")
plt.close()

# ==========================================================================
# 11. Save Results
# ==========================================================================
print("\n--- 11. Saving Results ---")

# Best method results
results_df = pd.DataFrame({
    'Dimension': dim_names,
    'Dimension_CN': [dimensions[name]['name_cn'] for name in dim_names],
    'Investment_Billion': optimal_x,
    'Percentage': optimal_x / TOTAL_BUDGET * 100,
    'Score_2025': [dim_china_scores[name] for name in dim_names],
    'Score_2035': [final_scores[name] for name in dim_names],
    'Improvement': [final_scores[name] - dim_china_scores[name] for name in dim_names],
    'Gap_2025': [dim_gaps[name] for name in dim_names],
    'Gap_2035': [dim_usa_scores[name] - final_scores[name] for name in dim_names],
    'Weight': [dim_weights[name] for name in dim_names],
    'Efficiency': [dimensions[name]['efficiency'] for name in dim_names]
})
results_df.to_csv('Q413_heuristic_results.csv', index=False)
print("Saved: Q413_heuristic_results.csv")

# Method comparison
comparison_df = pd.DataFrame({
    'Method': ['Gap-Weighted Priority', 'Balanced Multi-Factor', 'Strategic Focus'],
    'Score': [score_m1, score_m2, score_m3],
    'HHI': [hhi_m1, hhi_m2, hhi_m3],
    'Score_Improvement': [score_m1 - china_baseline, score_m2 - china_baseline, score_m3 - china_baseline]
})
comparison_df.to_csv('Q413_method_comparison.csv', index=False)
print("Saved: Q413_method_comparison.csv")

# All allocations
all_alloc_df = pd.DataFrame({
    'Dimension': dim_names,
    'M1_GapWeighted': optimal_x_m1,
    'M2_MultiFactor': optimal_x_m2,
    'M3_Strategic': optimal_x_m3
})
all_alloc_df.to_csv('Q413_all_allocations.csv', index=False)
print("Saved: Q413_all_allocations.csv")

# Summary
summary_df = pd.DataFrame({
    'Metric': ['Method', 'Total_Budget', 'China_Score_2025', 'China_Score_2035', 
               'Score_Improvement', 'HHI_Index', 'Best_Heuristic'],
    'Value': ['Priority_Heuristic', TOTAL_BUDGET, china_baseline, china_final,
              china_final - china_baseline, hhi, best_method_name]
})
summary_df.to_csv('Q413_heuristic_summary.csv', index=False)
print("Saved: Q413_heuristic_summary.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q413 PRIORITY-BASED HEURISTIC ALLOCATION COMPLETE")
print("=" * 70)
print(f"""
Method: Priority-Based Heuristic (Multi-Criteria)
Best Heuristic: {best_method_name}

Three Heuristic Approaches:
  1. Gap-Weighted: Prioritizes catching up to USA
  2. Multi-Factor: Balances weight, gap, efficiency, improvement
  3. Strategic Focus: Categorizes and allocates by strategy

Optimal Allocation ({best_method_name}):
""")
for i, name in enumerate(dim_names):
    print(f"  {dimensions[name]['name_cn']}: {optimal_x[i]:.0f}亿 ({optimal_x[i]/TOTAL_BUDGET*100:.1f}%)")

print(f"""
Results:
  - Score: {china_baseline:.2f} → {china_final:.2f} (+{china_final-china_baseline:.2f})
  - Diversification (HHI): {hhi:.4f}
  
Characteristics:
  - No iterations required
  - Highly interpretable
  - Based on domain knowledge
  
Output Files:
  - Q413_heuristic_results.csv
  - Q413_method_comparison.csv
  - Q413_all_allocations.csv
  - Q413_heuristic_summary.csv
  - Q413_heuristic_optimization.png
""")
print("=" * 70)

"""
==========================================================================
Huashu Cup 2026 Problem B - Question 4 Q43
Dynamic Programming: Multi-Stage Investment Optimization
动态规划：分阶段投资优化
==========================================================================

Core Idea:
- 1 trillion RMB invested over 10 years (2026-2035), not all at once
- Different stages have different investment priorities
- Consider time lag effects (基础研究5-10年, 人才3-5年, 算力1-3年)
- Use dynamic programming to optimize year-by-year allocation

Three Stages:
- Foundation Period (2026-2028): Computing infrastructure, talent cultivation
- Breakthrough Period (2029-2032): Innovation, industry development
- Harvest Period (2033-2035): Industry consolidation, policy optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q43: Dynamic Programming Multi-Stage Investment Optimization")
print("动态规划分阶段投资优化")
print("=" * 70)

# ==========================================================================
# 1. Data Loading and Preprocessing
# ==========================================================================
print("\n--- 1. Loading Data ---")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
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
# 2. Define Investment Dimensions with Time Lag Characteristics
# ==========================================================================
print("\n--- 2. Investment Dimensions with Time Lag ---")

dimensions = {
    'Computing_Power': {
        'indices': list(range(0, 8)),
        'name_cn': '算力基础',
        'efficiency': 0.8,
        'lag_years': 2,          # Time to see effect
        'peak_effect_year': 3,   # Year of maximum effect
        'decay_rate': 0.1,       # Annual decay after peak
        'stage_priority': {      # Priority weight by stage
            'foundation': 1.5,   # High priority in foundation stage
            'breakthrough': 0.8,
            'harvest': 0.5
        }
    },
    'Talent': {
        'indices': list(range(8, 13)),
        'name_cn': '人才资源',
        'efficiency': 0.6,
        'lag_years': 4,
        'peak_effect_year': 6,
        'decay_rate': 0.05,
        'stage_priority': {
            'foundation': 1.8,   # Very high - long cultivation time
            'breakthrough': 1.0,
            'harvest': 0.4
        }
    },
    'Innovation': {
        'indices': list(range(13, 19)),
        'name_cn': '创新能力',
        'efficiency': 0.5,
        'lag_years': 5,
        'peak_effect_year': 7,
        'decay_rate': 0.08,
        'stage_priority': {
            'foundation': 1.2,
            'breakthrough': 1.5,  # High priority in breakthrough
            'harvest': 0.8
        }
    },
    'Industry': {
        'indices': list(range(19, 24)),
        'name_cn': '产业发展',
        'efficiency': 0.9,
        'lag_years': 2,
        'peak_effect_year': 3,
        'decay_rate': 0.15,
        'stage_priority': {
            'foundation': 0.8,
            'breakthrough': 1.3,
            'harvest': 1.5       # High priority in harvest
        }
    },
    'Policy': {
        'indices': list(range(24, 29)),
        'name_cn': '政策环境',
        'efficiency': 0.7,
        'lag_years': 1,
        'peak_effect_year': 2,
        'decay_rate': 0.2,
        'stage_priority': {
            'foundation': 1.0,
            'breakthrough': 1.0,
            'harvest': 1.2
        }
    },
    'Economy': {
        'indices': list(range(29, 38)),
        'name_cn': '经济基础',
        'efficiency': 0.4,
        'lag_years': 3,
        'peak_effect_year': 5,
        'decay_rate': 0.1,
        'stage_priority': {
            'foundation': 1.0,
            'breakthrough': 1.0,
            'harvest': 0.8
        }
    }
}

n_dims = len(dimensions)
dim_names = list(dimensions.keys())

print(f"\n{'Dimension':<18} {'Lag':>6} {'Peak':>6} {'Foundation':>12} {'Breakthrough':>12} {'Harvest':>10}")
print("-" * 75)
for name, info in dimensions.items():
    print(f"{name:<18} {info['lag_years']:>5}yr {info['peak_effect_year']:>5}yr "
          f"{info['stage_priority']['foundation']:>12.1f} {info['stage_priority']['breakthrough']:>12.1f} "
          f"{info['stage_priority']['harvest']:>10.1f}")

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

# Global weights
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

print("\nBaseline Dimension Scores (2025):")
for name in dim_names:
    print(f"  {name}: {dim_china_scores[name]:.1f} (Gap to USA: {dim_gaps[name]:+.1f})")

# Historical improvement rates
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

# ==========================================================================
# 4. Define Time Lag Investment Effect Model
# ==========================================================================
print("\n--- 4. Time Lag Investment Effect Model ---")

"""
Investment Effect Model with Time Lag:

The effect of investment in year t on score improvement follows a bell curve:
- No effect during lag period
- Gradually increasing effect
- Peak effect at peak_effect_year
- Gradual decay after peak

Effect(t, t_invest) = efficiency × investment × bell_curve(t - t_invest)

Where bell_curve models the temporal distribution of investment effects
"""

def investment_effect_curve(years_since_investment, dim_name):
    """
    Calculate the effect multiplier for investment made years_since_investment ago
    
    Returns a value between 0 and 1 representing the proportion of 
    investment effect realized in the current year
    """
    info = dimensions[dim_name]
    lag = info['lag_years']
    peak = info['peak_effect_year']
    decay = info['decay_rate']
    
    if years_since_investment < lag:
        # Still in lag period - no effect yet
        return 0.0
    elif years_since_investment <= peak:
        # Rising phase - linear increase from lag to peak
        progress = (years_since_investment - lag) / (peak - lag)
        return progress
    else:
        # Decay phase - exponential decay after peak
        years_past_peak = years_since_investment - peak
        return np.exp(-decay * years_past_peak)

# Visualize effect curves
print("\nInvestment Effect Curves (proportion of effect by years since investment):")
print(f"{'Years':<8}", end="")
for name in dim_names:
    print(f"{name[:8]:>10}", end="")
print()
print("-" * 70)

for years in range(11):
    print(f"{years:<8}", end="")
    for name in dim_names:
        effect = investment_effect_curve(years, name)
        print(f"{effect:>10.2f}", end="")
    print()

# ==========================================================================
# 5. State Transition Function
# ==========================================================================
print("\n--- 5. State Transition Model ---")

def calculate_yearly_improvement(dim_name, current_score, yearly_investment, 
                                  investment_history, current_year_idx):
    """
    Calculate the score improvement for a dimension in the current year
    
    Parameters:
    - dim_name: dimension name
    - current_score: current score (0-100)
    - yearly_investment: investment in current year (亿元)
    - investment_history: list of past investments [(year_idx, amount), ...]
    - current_year_idx: index of current year (0-9 for 2026-2035)
    
    Returns:
    - Score improvement for current year
    """
    info = dimensions[dim_name]
    efficiency = info['efficiency']
    
    # Ceiling factor - harder to improve when already high
    ceiling_factor = ((100 - current_score) / 100) ** 0.5
    
    # Base natural improvement
    base_imp = base_improvement[dim_name] * 100 * 0.3 * ceiling_factor
    
    # Accumulated effect from all past investments
    investment_effect = 0
    reference_investment = 2000  # Reference for normalization
    
    # Add current year investment to history for calculation
    all_investments = investment_history + [(current_year_idx, yearly_investment)]
    
    for invest_year_idx, invest_amount in all_investments:
        years_since = current_year_idx - invest_year_idx
        if years_since >= 0:
            # Get effect multiplier for this investment
            effect_mult = investment_effect_curve(years_since, dim_name)
            # Investment effect with diminishing returns
            invest_effect = efficiency * (invest_amount / reference_investment) ** 0.7
            investment_effect += effect_mult * invest_effect
    
    # Total improvement
    total_improvement = base_imp + investment_effect * 5 * ceiling_factor
    
    # Cap improvement
    max_yearly_improvement = (100 - current_score) * 0.15  # Max 15% of gap per year
    total_improvement = min(total_improvement, max_yearly_improvement)
    
    return max(total_improvement, 0)

# ==========================================================================
# 6. Define Investment Stages
# ==========================================================================
print("\n--- 6. Investment Stages Definition ---")

TOTAL_BUDGET = 10000  # 1万亿 = 10000亿
YEARS = 10  # 2026-2035
YEAR_LABELS = [f"{2026+i}" for i in range(YEARS)]

# Stage definitions
stages = {
    'foundation': {
        'name_cn': '奠基期',
        'years': [0, 1, 2],  # 2026-2028
        'year_labels': ['2026', '2027', '2028'],
        'budget_share': 0.35,  # 35% of total budget
        'focus': ['Computing_Power', 'Talent'],
        'description': '基础设施建设，人才培养启动'
    },
    'breakthrough': {
        'name_cn': '突破期',
        'years': [3, 4, 5, 6],  # 2029-2032
        'year_labels': ['2029', '2030', '2031', '2032'],
        'budget_share': 0.45,  # 45% of total budget
        'focus': ['Innovation', 'Industry'],
        'description': '技术突破，产业化推进'
    },
    'harvest': {
        'name_cn': '收获期',
        'years': [7, 8, 9],  # 2033-2035
        'year_labels': ['2033', '2034', '2035'],
        'budget_share': 0.20,  # 20% of total budget
        'focus': ['Industry', 'Policy'],
        'description': '成果转化，生态完善'
    }
}

print(f"\n{'Stage':<15} {'Period':<15} {'Budget':>10} {'Focus Areas':<30}")
print("-" * 75)
for stage_name, stage_info in stages.items():
    budget = stage_info['budget_share'] * TOTAL_BUDGET
    years = f"{stage_info['year_labels'][0]}-{stage_info['year_labels'][-1]}"
    focus = ', '.join(stage_info['focus'])
    print(f"{stage_info['name_cn']:<15} {years:<15} {budget:>8.0f}亿 {focus:<30}")

# ==========================================================================
# 7. Dynamic Programming Optimization
# ==========================================================================
print("\n--- 7. Dynamic Programming Optimization ---")

def get_stage(year_idx):
    """Get stage name for a given year index"""
    for stage_name, stage_info in stages.items():
        if year_idx in stage_info['years']:
            return stage_name
    return 'harvest'

def stage_priority_weights(year_idx):
    """Get dimension priority weights for the current stage"""
    stage = get_stage(year_idx)
    weights = {}
    for name in dim_names:
        weights[name] = dimensions[name]['stage_priority'][stage]
    # Normalize
    total = sum(weights.values())
    for name in weights:
        weights[name] /= total
    return weights

# Forward simulation with given allocation strategy
def simulate_trajectory(yearly_allocations):
    """
    Simulate the 10-year trajectory given yearly investment allocations
    
    Parameters:
    - yearly_allocations: array of shape (YEARS, n_dims) - investment per year per dimension
    
    Returns:
    - scores_trajectory: array of shape (YEARS+1, n_dims) - scores from 2025 to 2035
    - total_score_trajectory: array of shape (YEARS+1,) - TOPSIS-like total score
    """
    # Initialize with 2025 scores
    scores = np.array([dim_china_scores[name] for name in dim_names])
    scores_trajectory = [scores.copy()]
    
    # Investment history for each dimension
    investment_histories = {name: [] for name in dim_names}
    
    for year_idx in range(YEARS):
        new_scores = []
        for i, name in enumerate(dim_names):
            current_score = scores[i]
            yearly_invest = yearly_allocations[year_idx, i]
            
            # Calculate improvement
            improvement = calculate_yearly_improvement(
                name, current_score, yearly_invest,
                investment_histories[name], year_idx
            )
            
            new_score = min(current_score + improvement, 100)
            new_scores.append(new_score)
            
            # Record investment
            investment_histories[name].append((year_idx, yearly_invest))
        
        scores = np.array(new_scores)
        scores_trajectory.append(scores.copy())
    
    scores_trajectory = np.array(scores_trajectory)
    
    # Calculate weighted total score for each year
    weights = np.array([dim_weights[name] for name in dim_names])
    weights = weights / np.sum(weights)
    total_scores = np.sum(scores_trajectory * weights, axis=1)
    
    return scores_trajectory, total_scores

# Objective function for optimization
def objective_dp(x):
    """
    Objective: Maximize final year total score
    x: flattened array of yearly allocations (YEARS × n_dims)
    """
    yearly_allocations = x.reshape(YEARS, n_dims)
    
    # Penalty for budget violations
    yearly_totals = np.sum(yearly_allocations, axis=1)
    stage_budgets = {
        'foundation': stages['foundation']['budget_share'] * TOTAL_BUDGET,
        'breakthrough': stages['breakthrough']['budget_share'] * TOTAL_BUDGET,
        'harvest': stages['harvest']['budget_share'] * TOTAL_BUDGET
    }
    
    penalty = 0
    for stage_name, stage_info in stages.items():
        stage_years = stage_info['years']
        stage_total = np.sum(yearly_totals[stage_years])
        target = stage_budgets[stage_name]
        penalty += 0.1 * abs(stage_total - target)
    
    # Total budget penalty
    total_used = np.sum(yearly_allocations)
    penalty += 10 * abs(total_used - TOTAL_BUDGET)
    
    # Simulate trajectory
    scores_trajectory, total_scores = simulate_trajectory(yearly_allocations)
    
    # Final score (2035)
    final_score = total_scores[-1]
    
    # Bonus for smooth trajectory (avoid wild swings)
    smoothness_bonus = -np.std(np.diff(total_scores)) * 0.5
    
    return -(final_score + smoothness_bonus - penalty)

# Generate initial allocation based on stage priorities
def generate_initial_allocation():
    """Generate initial allocation based on stage priority weights"""
    allocation = np.zeros((YEARS, n_dims))
    
    for stage_name, stage_info in stages.items():
        stage_years = stage_info['years']
        stage_budget = stage_info['budget_share'] * TOTAL_BUDGET
        yearly_budget = stage_budget / len(stage_years)
        
        # Get priority weights for this stage
        priority = np.array([dimensions[name]['stage_priority'][stage_name] for name in dim_names])
        priority = priority / np.sum(priority)
        
        for year_idx in stage_years:
            allocation[year_idx] = priority * yearly_budget
    
    return allocation

print("Generating initial allocation based on stage priorities...")
initial_allocation = generate_initial_allocation()

print("\nInitial Stage-Based Allocation:")
print(f"{'Year':<8}", end="")
for name in dim_names:
    print(f"{name[:10]:>12}", end="")
print(f"{'Total':>12}")
print("-" * 86)

for year_idx in range(YEARS):
    print(f"{YEAR_LABELS[year_idx]:<8}", end="")
    for i in range(n_dims):
        print(f"{initial_allocation[year_idx, i]:>12.0f}", end="")
    print(f"{np.sum(initial_allocation[year_idx]):>12.0f}")

# Optimize
print("\nOptimizing multi-stage allocation...")

# Bounds
min_yearly = 50   # Minimum 50亿 per dimension per year
max_yearly = 1500  # Maximum 1500亿 per dimension per year

bounds = [(min_yearly, max_yearly) for _ in range(YEARS * n_dims)]

# Constraints for stage budgets
def stage_budget_constraint(x, stage_name):
    yearly_allocations = x.reshape(YEARS, n_dims)
    stage_years = stages[stage_name]['years']
    stage_budget = stages[stage_name]['budget_share'] * TOTAL_BUDGET
    stage_actual = np.sum(yearly_allocations[stage_years])
    return stage_budget - abs(stage_actual - stage_budget)

constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - TOTAL_BUDGET},  # Total budget
]

# Run optimization
x0 = initial_allocation.flatten()

result = minimize(
    objective_dp,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-8}
)

optimal_allocation = result.x.reshape(YEARS, n_dims)

# Adjust to exactly meet budget
optimal_allocation = optimal_allocation / np.sum(optimal_allocation) * TOTAL_BUDGET

print(f"\nOptimization Result:")
print(f"  Success: {result.success}")
print(f"  Iterations: {result.nit}")

# ==========================================================================
# 8. Results Analysis
# ==========================================================================
print("\n--- 8. Results Analysis ---")

# Simulate optimal trajectory
scores_trajectory, total_scores = simulate_trajectory(optimal_allocation)

print("\n" + "=" * 90)
print("Q43 DYNAMIC PROGRAMMING OPTIMAL MULTI-STAGE ALLOCATION")
print("=" * 90)

# Yearly allocation table
print(f"\n{'Year':<8} {'Stage':<10}", end="")
for name in dim_names:
    print(f"{dimensions[name]['name_cn'][:4]:>8}", end="")
print(f"{'年度合计':>10}")
print("-" * 90)

stage_totals = {stage: np.zeros(n_dims) for stage in stages.keys()}

for year_idx in range(YEARS):
    stage = get_stage(year_idx)
    stage_cn = stages[stage]['name_cn']
    print(f"{YEAR_LABELS[year_idx]:<8} {stage_cn:<10}", end="")
    for i in range(n_dims):
        print(f"{optimal_allocation[year_idx, i]:>8.0f}", end="")
        stage_totals[stage][i] += optimal_allocation[year_idx, i]
    print(f"{np.sum(optimal_allocation[year_idx]):>10.0f}")

print("-" * 90)

# Stage subtotals
print("\n阶段汇总:")
print(f"{'Stage':<15} {'Period':<12}", end="")
for name in dim_names:
    print(f"{dimensions[name]['name_cn'][:4]:>8}", end="")
print(f"{'合计':>10} {'占比':>8}")
print("-" * 90)

for stage_name, stage_info in stages.items():
    period = f"{stage_info['year_labels'][0]}-{stage_info['year_labels'][-1]}"
    total = np.sum(stage_totals[stage_name])
    print(f"{stage_info['name_cn']:<15} {period:<12}", end="")
    for i in range(n_dims):
        print(f"{stage_totals[stage_name][i]:>8.0f}", end="")
    print(f"{total:>10.0f} {total/TOTAL_BUDGET*100:>7.1f}%")

# Dimension totals
print("\n维度汇总:")
dim_totals = np.sum(optimal_allocation, axis=0)
print(f"{'Dimension':<18} {'Total':>12} {'Percentage':>12}")
print("-" * 45)
for i, name in enumerate(dim_names):
    print(f"{dimensions[name]['name_cn']:<18} {dim_totals[i]:>10.0f}亿 {dim_totals[i]/TOTAL_BUDGET*100:>11.1f}%")

# Score trajectory
print("\n得分演变轨迹:")
print(f"{'Year':<8}", end="")
for name in dim_names:
    print(f"{dimensions[name]['name_cn'][:4]:>8}", end="")
print(f"{'加权总分':>10}")
print("-" * 75)

for year_idx in range(YEARS + 1):
    year_label = "2025" if year_idx == 0 else YEAR_LABELS[year_idx - 1]
    print(f"{year_label:<8}", end="")
    for i in range(n_dims):
        print(f"{scores_trajectory[year_idx, i]:>8.1f}", end="")
    print(f"{total_scores[year_idx]:>10.2f}")

# Final metrics
print("\n" + "=" * 70)
print("KEY METRICS")
print("=" * 70)

initial_total = total_scores[0]
final_total = total_scores[-1]

print(f"\n{'Metric':<35} {'2025':>12} {'2035':>12} {'Change':>12}")
print("-" * 72)
print(f"{'Weighted Total Score':<35} {initial_total:>12.2f} {final_total:>12.2f} {final_total-initial_total:>+12.2f}")

for i, name in enumerate(dim_names):
    initial = scores_trajectory[0, i]
    final = scores_trajectory[-1, i]
    print(f"{dimensions[name]['name_cn']:<35} {initial:>12.1f} {final:>12.1f} {final-initial:>+12.1f}")

# ==========================================================================
# 9. Comparison with Static Allocation
# ==========================================================================
print("\n--- 9. Comparison: Dynamic vs Static Allocation ---")

# Static allocation (uniform across years)
static_allocation = np.zeros((YEARS, n_dims))
yearly_budget = TOTAL_BUDGET / YEARS
dim_weights_arr = np.array([dim_weights[name] for name in dim_names])
dim_weights_arr = dim_weights_arr / np.sum(dim_weights_arr)

for year_idx in range(YEARS):
    static_allocation[year_idx] = dim_weights_arr * yearly_budget

static_scores_trajectory, static_total_scores = simulate_trajectory(static_allocation)

print(f"\n{'Method':<25} {'2030 Score':>12} {'2035 Score':>12} {'Improvement':>12}")
print("-" * 65)
print(f"{'Dynamic (Stage-Based)':<25} {total_scores[5]:>12.2f} {total_scores[-1]:>12.2f} {total_scores[-1]-total_scores[0]:>+12.2f}")
print(f"{'Static (Uniform)':<25} {static_total_scores[5]:>12.2f} {static_total_scores[-1]:>12.2f} {static_total_scores[-1]-static_total_scores[0]:>+12.2f}")
print(f"{'Dynamic Advantage':<25} {total_scores[5]-static_total_scores[5]:>+12.2f} {total_scores[-1]-static_total_scores[-1]:>+12.2f}")

# ==========================================================================
# 10. Visualization
# ==========================================================================
print("\n--- 10. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Stacked area chart of yearly allocation
ax1 = axes[0, 0]
years_plot = np.arange(2026, 2036)
colors = plt.cm.Set2(np.linspace(0, 1, n_dims))

bottom = np.zeros(YEARS)
for i, name in enumerate(dim_names):
    ax1.bar(years_plot, optimal_allocation[:, i], bottom=bottom, 
            label=dimensions[name]['name_cn'], color=colors[i], edgecolor='white', linewidth=0.5)
    bottom += optimal_allocation[:, i]

# Add stage dividers
ax1.axvline(x=2028.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(x=2032.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Stage labels
ax1.text(2027, max(bottom)*0.95, '奠基期', ha='center', fontsize=11, fontweight='bold', color='red')
ax1.text(2030.5, max(bottom)*0.95, '突破期', ha='center', fontsize=11, fontweight='bold', color='red')
ax1.text(2034, max(bottom)*0.95, '收获期', ha='center', fontsize=11, fontweight='bold', color='red')

ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Investment (亿元)', fontsize=11)
ax1.set_title('Multi-Stage Investment Allocation by Year', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(years_plot)

# 2. Score trajectory comparison
ax2 = axes[0, 1]
years_full = np.arange(2025, 2036)

ax2.plot(years_full, total_scores, 'o-', linewidth=2, markersize=6, 
         color='coral', label='Dynamic (Stage-Based)')
ax2.plot(years_full, static_total_scores, 's--', linewidth=2, markersize=6, 
         color='steelblue', label='Static (Uniform)')

# Stage dividers
ax2.axvline(x=2028.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.axvline(x=2032.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

ax2.fill_between(years_full, total_scores, static_total_scores, 
                  where=(total_scores > static_total_scores), alpha=0.3, color='coral',
                  label='Dynamic Advantage')

ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Weighted Total Score', fontsize=11)
ax2.set_title('Score Trajectory: Dynamic vs Static', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(years_full)

# 3. Dimension score evolution
ax3 = axes[1, 0]
for i, name in enumerate(dim_names):
    ax3.plot(years_full, scores_trajectory[:, i], 'o-', linewidth=2, 
             label=dimensions[name]['name_cn'], color=colors[i])

ax3.axvline(x=2028.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax3.axvline(x=2032.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Dimension Score', fontsize=11)
ax3.set_title('Dimension Score Evolution (Dynamic Strategy)', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(years_full)

# 4. Stage allocation pie charts
ax4 = axes[1, 1]

# Create subplot for each stage
stage_names_list = list(stages.keys())
stage_colors = ['#FF9999', '#66B2FF', '#99FF99']

bar_width = 0.25
x = np.arange(n_dims)

for idx, (stage_name, stage_info) in enumerate(stages.items()):
    stage_total = stage_totals[stage_name]
    bars = ax4.bar(x + idx * bar_width, stage_total, bar_width, 
                   label=f"{stage_info['name_cn']} ({stage_info['year_labels'][0]}-{stage_info['year_labels'][-1]})",
                   color=stage_colors[idx], alpha=0.8)

ax4.set_xticks(x + bar_width)
ax4.set_xticklabels([dimensions[name]['name_cn'] for name in dim_names], rotation=30, ha='right')
ax4.set_ylabel('Investment (亿元)', fontsize=11)
ax4.set_title('Investment by Stage and Dimension', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Q43: Dynamic Programming Multi-Stage Investment Optimization\n动态规划分阶段投资优化', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q43_dynamic_programming.png', dpi=150, bbox_inches='tight')
print("Saved: Q43_dynamic_programming.png")
plt.close()

# Additional visualization: Investment effect timing
fig2, ax = plt.subplots(figsize=(12, 6))

years_effect = np.arange(0, 11)
for i, name in enumerate(dim_names):
    effects = [investment_effect_curve(y, name) for y in years_effect]
    ax.plot(years_effect, effects, 'o-', linewidth=2, label=dimensions[name]['name_cn'], color=colors[i])

ax.set_xlabel('Years Since Investment', fontsize=11)
ax.set_ylabel('Effect Multiplier (0-1)', fontsize=11)
ax.set_title('Investment Effect Timing by Dimension\n(Why early investment in long-lag dimensions matters)', 
             fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xticks(years_effect)

plt.tight_layout()
plt.savefig('Q43_effect_timing.png', dpi=150, bbox_inches='tight')
print("Saved: Q43_effect_timing.png")
plt.close()

# ==========================================================================
# 11. Save Results
# ==========================================================================
print("\n--- 11. Saving Results ---")

# Yearly allocation
yearly_df = pd.DataFrame(optimal_allocation, columns=dim_names)
yearly_df.insert(0, 'Year', YEAR_LABELS)
yearly_df['Stage'] = [stages[get_stage(i)]['name_cn'] for i in range(YEARS)]
yearly_df['Yearly_Total'] = np.sum(optimal_allocation, axis=1)
yearly_df.to_csv('Q43_yearly_allocation.csv', index=False)
print("Saved: Q43_yearly_allocation.csv")

# Stage summary
stage_summary = []
for stage_name, stage_info in stages.items():
    row = {'Stage': stage_info['name_cn'], 'Period': f"{stage_info['year_labels'][0]}-{stage_info['year_labels'][-1]}"}
    for i, name in enumerate(dim_names):
        row[dimensions[name]['name_cn']] = stage_totals[stage_name][i]
    row['Total'] = np.sum(stage_totals[stage_name])
    row['Percentage'] = np.sum(stage_totals[stage_name]) / TOTAL_BUDGET * 100
    stage_summary.append(row)
stage_df = pd.DataFrame(stage_summary)
stage_df.to_csv('Q43_stage_summary.csv', index=False)
print("Saved: Q43_stage_summary.csv")

# Score trajectory
trajectory_df = pd.DataFrame(scores_trajectory, columns=[dimensions[name]['name_cn'] for name in dim_names])
trajectory_df.insert(0, 'Year', ['2025'] + YEAR_LABELS)
trajectory_df['Weighted_Total'] = total_scores
trajectory_df.to_csv('Q43_score_trajectory.csv', index=False)
print("Saved: Q43_score_trajectory.csv")

# Comparison with static
comparison_df = pd.DataFrame({
    'Year': ['2025'] + YEAR_LABELS,
    'Dynamic_Score': total_scores,
    'Static_Score': static_total_scores,
    'Advantage': total_scores - static_total_scores
})
comparison_df.to_csv('Q43_dynamic_vs_static.csv', index=False)
print("Saved: Q43_dynamic_vs_static.csv")

# Summary
summary_df = pd.DataFrame({
    'Metric': ['Method', 'Total_Budget', 'Years', 'Stages',
               'Initial_Score_2025', 'Final_Score_2035', 'Total_Improvement',
               'Static_Final_Score', 'Dynamic_Advantage'],
    'Value': ['Dynamic_Programming', TOTAL_BUDGET, YEARS, 3,
              total_scores[0], total_scores[-1], total_scores[-1] - total_scores[0],
              static_total_scores[-1], total_scores[-1] - static_total_scores[-1]]
})
summary_df.to_csv('Q43_summary.csv', index=False)
print("Saved: Q43_summary.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q43 DYNAMIC PROGRAMMING OPTIMIZATION COMPLETE")
print("=" * 70)
print(f"""
Core Principle: 分阶段动态优化
"Different stages require different investment priorities"

Three Stages:
  1. 奠基期 (2026-2028): Focus on Computing, Talent
     - Budget: {stages['foundation']['budget_share']*100:.0f}% = {stages['foundation']['budget_share']*TOTAL_BUDGET:.0f}亿
     - Key insight: Long-lag investments (talent, innovation) must start early
     
  2. 突破期 (2029-2032): Focus on Innovation, Industry
     - Budget: {stages['breakthrough']['budget_share']*100:.0f}% = {stages['breakthrough']['budget_share']*TOTAL_BUDGET:.0f}亿
     - Key insight: Foundation built, now push for breakthroughs
     
  3. 收获期 (2033-2035): Focus on Industry, Policy
     - Budget: {stages['harvest']['budget_share']*100:.0f}% = {stages['harvest']['budget_share']*TOTAL_BUDGET:.0f}亿
     - Key insight: Consolidate gains, optimize ecosystem

Dimension Totals (10-year):
""")
for i, name in enumerate(dim_names):
    pct = dim_totals[i] / TOTAL_BUDGET * 100
    print(f"  {dimensions[name]['name_cn']}: {dim_totals[i]:.0f}亿 ({pct:.1f}%)")

print(f"""
Key Results:
  Weighted Score: {total_scores[0]:.2f} → {total_scores[-1]:.2f} (+{total_scores[-1]-total_scores[0]:.2f})
  vs Static: +{total_scores[-1] - static_total_scores[-1]:.2f} advantage

Strategic Insights:
  1. Early investment in long-lag dimensions (Talent, Innovation) pays off
  2. Stage-based allocation outperforms uniform allocation
  3. Flexibility to adjust based on stage progress

Output Files:
  - Q43_yearly_allocation.csv
  - Q43_stage_summary.csv
  - Q43_score_trajectory.csv
  - Q43_dynamic_vs_static.csv
  - Q43_summary.csv
  - Q43_dynamic_programming.png
  - Q43_effect_timing.png
""")
print("=" * 70)

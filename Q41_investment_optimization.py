"""
==========================================================================
Huashu Cup 2026 Problem B - Question 4 Q41
Marginal Benefit Analysis + Mathematical Programming
Optimal Investment Allocation for China's AI Competitiveness (2026-2035)
==========================================================================

Goal: Maximize China's AI competitiveness score by 2035
Constraint: Total investment ≤ 1 trillion RMB (10000亿)
Method: Marginal benefit analysis + Nonlinear optimization
"""
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q41: Marginal Benefit Analysis + Mathematical Programming")
print("Optimal Investment Allocation for China AI Competitiveness")
print("=" * 70)

# ==========================================================================
# 1. Data Loading and Preprocessing
# ==========================================================================
print("\n--- 1. Loading Data ---")

# Load historical data
hist_data = pd.read_csv('panel_data_38indicators.csv')

# Load prediction data (Q316 results)
pred_data = pd.read_csv('Q316_adaptive_predictions.csv')

countries = sorted(hist_data['Country'].unique())
hist_years = sorted(hist_data['Year'].unique())
indicator_cols = [col for col in hist_data.columns if col not in ['Country', 'Year']]

n_countries = len(countries)
n_indicators = len(indicator_cols)

print(f"Countries: {n_countries}")
print(f"Indicators: {n_indicators}")
print(f"Historical Years: {min(hist_years)} - {max(hist_years)}")

# Negative indicators (lower is better)
negative_indicators = ['全球创新指数排名']

# Indicator types
indicator_types = np.ones(n_indicators)
for i, col in enumerate(indicator_cols):
    for neg in negative_indicators:
        if neg in col:
            indicator_types[i] = -1

# ==========================================================================
# 2. Define 6 Investment Dimensions
# ==========================================================================
print("\n--- 2. Defining Investment Dimensions ---")

# 6 dimensions with indicator indices and investment characteristics
dimensions = {
    'Computing_Power': {
        'indices': list(range(0, 8)),
        'name_cn': '算力基础',
        'efficiency': 0.8,      # Investment efficiency (0-1)
        'lag_years': 2,         # Years to see effect
        'diminishing_rate': 0.7 # Diminishing returns parameter
    },
    'Talent': {
        'indices': list(range(8, 13)),
        'name_cn': '人才资源',
        'efficiency': 0.6,
        'lag_years': 4,
        'diminishing_rate': 0.6
    },
    'Innovation': {
        'indices': list(range(13, 19)),
        'name_cn': '创新能力',
        'efficiency': 0.5,
        'lag_years': 5,
        'diminishing_rate': 0.5
    },
    'Industry': {
        'indices': list(range(19, 24)),
        'name_cn': '产业发展',
        'efficiency': 0.9,
        'lag_years': 2,
        'diminishing_rate': 0.75
    },
    'Policy': {
        'indices': list(range(24, 29)),
        'name_cn': '政策环境',
        'efficiency': 0.7,
        'lag_years': 1,
        'diminishing_rate': 0.8
    },
    'Economy': {
        'indices': list(range(29, 38)),
        'name_cn': '经济基础',
        'efficiency': 0.4,
        'lag_years': 3,
        'diminishing_rate': 0.65
    }
}

n_dims = len(dimensions)
dim_names = list(dimensions.keys())

for name, info in dimensions.items():
    print(f"  {name} ({info['name_cn']}): {len(info['indices'])} indicators, "
          f"efficiency={info['efficiency']}, lag={info['lag_years']}yr")

# ==========================================================================
# 3. Extract Current Status (2025) and Compute Gaps
# ==========================================================================
print("\n--- 3. Analyzing Current Status and Gaps ---")

# Get 2025 data for all countries
china_idx = countries.index('中国')
usa_idx = countries.index('美国')

data_2025 = hist_data[hist_data['Year'] == 2025].sort_values('Country')
china_2025 = data_2025[data_2025['Country'] == '中国'][indicator_cols].values.flatten()
usa_2025 = data_2025[data_2025['Country'] == '美国'][indicator_cols].values.flatten()

# Normalize function
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

# Calculate entropy weights from historical data
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
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
# Collect all historical data for weight calculation
all_hist_X = []
for year in hist_years:
    year_data = hist_data[hist_data['Year'] == year].sort_values('Country')
    X = year_data[indicator_cols].values
    all_hist_X.append(X)
all_hist_X = np.vstack(all_hist_X)

# Global normalization and weights
all_X_norm = normalize_minmax(all_hist_X, indicator_types)
global_weights = entropy_weight(all_X_norm)

print("\nIndicator Weights (Top 10):")
weight_order = np.argsort(-global_weights)[:10]
for i, idx in enumerate(weight_order):
    print(f"  {i+1}. {indicator_cols[idx]}: {global_weights[idx]:.4f}")

# Calculate dimension weights
dim_weights = {}
for name, info in dimensions.items():
    dim_weights[name] = np.sum(global_weights[info['indices']])
    
print("\nDimension Weights:")
for name in dim_names:
    print(f"  {name}: {dim_weights[name]:.4f} ({dim_weights[name]*100:.1f}%)")

# Calculate China-USA gaps by dimension
print("\nChina-USA Gap Analysis (Normalized Scale 0-100):")
X_2025 = data_2025[indicator_cols].values
X_2025_norm = normalize_minmax(X_2025, indicator_types)

china_norm = X_2025_norm[china_idx]
usa_norm = X_2025_norm[usa_idx]

dim_gaps = {}
dim_china_scores = {}
dim_usa_scores = {}

for name, info in dimensions.items():
    idx = info['indices']
    w = global_weights[idx] / np.sum(global_weights[idx])
    
    china_score = np.sum(china_norm[idx] * w) * 100
    usa_score = np.sum(usa_norm[idx] * w) * 100
    gap = usa_score - china_score
    
    dim_gaps[name] = gap
    dim_china_scores[name] = china_score
    dim_usa_scores[name] = usa_score
    
    print(f"  {name}: China={china_score:.1f}, USA={usa_score:.1f}, Gap={gap:+.1f}")

# ==========================================================================
# 4. Build Investment-Output Function
# ==========================================================================
print("\n--- 4. Building Investment-Output Functions ---")

"""
Investment-Output Model:
Δ_indicator = α * Investment^β * (1 - current_level/max_level)^γ
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
Where:
- α: efficiency coefficient
- β: diminishing returns exponent (0 < β < 1)
- γ: ceiling effect (harder to improve when already high)
- Investment: in 亿元 (100 million RMB)
"""

# Total budget: 1万亿 = 10000亿
TOTAL_BUDGET = 10000  # 亿元

# Estimate historical investment-improvement relationships
# Using year-over-year changes to calibrate

def estimate_improvement_rate(dim_name):
    """Estimate how much improvement per unit investment for a dimension"""
    info = dimensions[dim_name]
    idx = info['indices']
    
    # Get China's historical data
    china_hist = hist_data[hist_data['Country'] == '中国'].sort_values('Year')
    
    improvements = []
    for i in range(1, len(hist_years)):
        year_prev = hist_years[i-1]
        year_curr = hist_years[i]
        
        data_prev = china_hist[china_hist['Year'] == year_prev][indicator_cols].values.flatten()
        data_curr = china_hist[china_hist['Year'] == year_curr][indicator_cols].values.flatten()
        
        # Calculate normalized improvement for this dimension
        for j in idx:
            if indicator_types[j] == 1:
                if data_prev[j] > 0:
                    imp = (data_curr[j] - data_prev[j]) / data_prev[j]
                    improvements.append(imp)
            else:
                if data_prev[j] > 0:
                    imp = (data_prev[j] - data_curr[j]) / data_prev[j]
                    improvements.append(imp)
    
    return np.mean(improvements) if improvements else 0.05

# Calculate base improvement rates
print("\nEstimated Base Improvement Rates (historical):")
base_improvement = {}
for name in dim_names:
    rate = estimate_improvement_rate(name)
    base_improvement[name] = max(rate, 0.01)  # At least 1%
    print(f"  {name}: {base_improvement[name]*100:.2f}% per year")

# Investment response function for each dimension
def investment_response(investment, dim_name, current_score, years=10):
    """
    Calculate score improvement given investment
    
    Parameters:
    - investment: 投资金额 (亿元)
    - dim_name: dimension name
    - current_score: current normalized score (0-100)
    - years: investment horizon
    
    Returns:
    - score improvement (0-100 scale)
    """
    info = dimensions[dim_name]
    
    # Parameters
    efficiency = info['efficiency']
    beta = info['diminishing_rate']  # Diminishing returns
    lag = info['lag_years']
    
    # Effective years (accounting for lag)
    effective_years = max(years - lag, 1)
    
    # Ceiling effect: harder to improve when already high
    ceiling_factor = ((100 - current_score) / 100) ** 0.5
    
    # Investment effect with strong diminishing returns
    # Reference: 2000亿 investment gives ~50% of maximum possible improvement
    reference_investment = 2000  # 亿
    investment_effect = 1 - np.exp(-investment / reference_investment)
    
    # Base improvement capacity (max improvement possible over 10 years)
    # Based on historical rates but capped reasonably
    base_capacity = min(base_improvement[dim_name] * effective_years * 100, 50)
    
    # Additional capacity from investment (up to 50% more)
    investment_capacity = 50 * efficiency * investment_effect
    
    # Total improvement
    improvement = (base_capacity + investment_capacity) * ceiling_factor
    
    # Cap improvement based on room for improvement
    max_improvement = (100 - current_score) * 0.85  # Can reach at most 85% of gap
    improvement = min(improvement, max_improvement)
    
    return max(improvement, 0)

# Test the function
print("\nInvestment Response Test (1000亿 investment, 10 years):")
for name in dim_names:
    imp = investment_response(1000, name, dim_china_scores[name])
    print(f"  {name}: Current={dim_china_scores[name]:.1f} → +{imp:.1f} points")

# ==========================================================================
# 5. TOPSIS Score Function
# ==========================================================================
print("\n--- 5. Building TOPSIS Score Function ---")

def calculate_topsis_score(china_scores_new, all_countries_data):
    """
    Calculate China's TOPSIS score given new dimension scores
    
    Parameters:
    - china_scores_new: dict of {dim_name: new_score}
    - all_countries_data: normalized data for all countries
    
    Returns:
    - China's TOPSIS score (0-100)
    """
    n = all_countries_data.shape[0]
    
    # Update China's data
    china_data_new = all_countries_data[china_idx].copy()
    
    for name, score in china_scores_new.items():
        idx = dimensions[name]['indices']
        # Distribute score improvement proportionally
        current_dim_score = dim_china_scores[name]
        if current_dim_score > 0:
            ratio = score / current_dim_score
        else:
            ratio = 1
        china_data_new[idx] = all_countries_data[china_idx, idx] * min(ratio, 2.0)
    
    # Rebuild data matrix
    data_matrix = all_countries_data.copy()
    data_matrix[china_idx] = china_data_new
    
    # Re-normalize
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
    
    # TOPSIS
    V = data_norm * global_weights
    V_pos = np.max(V, axis=0)
    V_neg = np.min(V, axis=0)
    
    D_pos = np.sqrt(np.sum((V - V_pos) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((V - V_neg) ** 2, axis=1))
    
    scores = D_neg / (D_pos + D_neg + 1e-10)
    
    # Normalize to 0-100
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10) * 100
    
    return scores[china_idx], scores

# Current baseline score
baseline_scores = {}
for name in dim_names:
    baseline_scores[name] = dim_china_scores[name]

china_baseline, all_scores_baseline = calculate_topsis_score(baseline_scores, X_2025_norm)
print(f"\nChina's Baseline TOPSIS Score (2025): {china_baseline:.2f}")

# USA score for reference
usa_baseline = all_scores_baseline[usa_idx]
print(f"USA's Baseline TOPSIS Score (2025): {usa_baseline:.2f}")

# ==========================================================================
# 6. Marginal Benefit Analysis
# ==========================================================================
print("\n--- 6. Marginal Benefit Analysis ---")

def marginal_benefit(investment_allocation):
    """
    Calculate China's 2035 score given investment allocation
    
    Parameters:
    - investment_allocation: array of shape (n_dims,) in 亿元
    
    Returns:
    - TOPSIS score improvement
    """
    # Calculate new scores after investment
    new_scores = {}
    for i, name in enumerate(dim_names):
        improvement = investment_response(
            investment_allocation[i], 
            name, 
            dim_china_scores[name]
        )
        new_scores[name] = dim_china_scores[name] + improvement
    
    # Calculate TOPSIS score
    china_score, _ = calculate_topsis_score(new_scores, X_2025_norm)
    
    return china_score

# Calculate marginal benefits for each dimension
print("\nMarginal Benefits (1000亿 additional investment):")
marginal_benefits = {}
delta = 500  # 500亿 increment

for i, name in enumerate(dim_names):
    # Base allocation: zero everywhere
    base_alloc = np.zeros(n_dims)
    
    # Add investment to this dimension
    test_alloc = base_alloc.copy()
    test_alloc[i] = delta
    
    # Calculate improvement
    score_with = marginal_benefit(test_alloc)
    score_without = china_baseline
    
    mb = (score_with - score_without) / delta * 1000  # Per 1000亿
    marginal_benefits[name] = mb
    
    print(f"  {name}: +{mb:.2f} points per 1000亿")

# Rank dimensions by marginal benefit
print("\nDimension Priority (by Marginal Benefit):")
sorted_dims = sorted(marginal_benefits.items(), key=lambda x: -x[1])
for rank, (name, mb) in enumerate(sorted_dims):
    gap = dim_gaps[name]
    weight = dim_weights[name]
    print(f"  {rank+1}. {name}: MB={mb:.2f}, Weight={weight:.1%}, Gap={gap:+.1f}")

# ==========================================================================
# 7. Mathematical Optimization (with Diversification Constraints)
# ==========================================================================
print("\n--- 7. Mathematical Optimization ---")

# Enhanced objective function considering:
# 1. TOPSIS score improvement
# 2. Risk diversification (penalize concentration)
# 3. Strategic importance weighting

def enhanced_objective(x, lambda_div=0.1):
    """
    Enhanced objective function
    - Maximize score
    - Penalize concentration (encourage diversification)
    """
    score = marginal_benefit(x)
    
    # Concentration penalty (Herfindahl index)
    shares = x / (np.sum(x) + 1e-10)
    hhi = np.sum(shares ** 2)  # 0 to 1, lower is more diversified
    
    # Penalty for ignoring high-weight dimensions
    weight_alignment = 0
    for i, name in enumerate(dim_names):
        weight_alignment += shares[i] * dim_weights[name]
    
    # Combined objective
    obj = score - lambda_div * hhi * 100 + 0.05 * weight_alignment * 100
    
    return -obj  # Minimize negative

def constraint_budget(x):
    """Budget constraint: sum <= TOTAL_BUDGET"""
    return TOTAL_BUDGET - np.sum(x)

def constraint_min_allocation(x):
    """Minimum allocation: each dimension gets at least 5%"""
    min_alloc = 0.05 * TOTAL_BUDGET
    return np.min(x) - min_alloc

def constraint_max_allocation(x):
    """Maximum allocation: no dimension gets more than 40%"""
    max_alloc = 0.40 * TOTAL_BUDGET
    return max_alloc - np.max(x)

# Bounds: each dimension gets 5% to 40% of budget
min_bound = 0.05 * TOTAL_BUDGET  # 500亿
max_bound = 0.40 * TOTAL_BUDGET  # 4000亿
bounds = [(min_bound, max_bound) for _ in range(n_dims)]

# Initial guess: proportional to (marginal benefit × weight × gap)
priority_scores = []
for name in dim_names:
    mb = marginal_benefits[name]
    w = dim_weights[name]
    gap = max(dim_gaps[name], 1)  # Prioritize areas with gaps
    priority = mb * w * (1 + gap / 100)
    priority_scores.append(priority)

priority_scores = np.array(priority_scores)
priority_scores = priority_scores / np.sum(priority_scores)
x0 = priority_scores * TOTAL_BUDGET
x0 = np.clip(x0, min_bound, max_bound)
x0 = x0 / np.sum(x0) * TOTAL_BUDGET  # Ensure sum = budget

print(f"\nOptimization Setup:")
print(f"  Total Budget: {TOTAL_BUDGET}亿元 (1万亿)")
print(f"  Decision Variables: {n_dims} dimensions")
print(f"  Constraints:")
print(f"    - Total ≤ {TOTAL_BUDGET}亿")
print(f"    - Each dimension: {min_bound:.0f}亿 to {max_bound:.0f}亿 (5%-40%)")
print(f"  Objective: Maximize Score + Diversification Bonus")

# Method 1: SLSQP with constraints
print("\n--- Method 1: SLSQP with Diversification ---")

constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - TOTAL_BUDGET},  # Use all budget
    {'type': 'ineq', 'fun': constraint_min_allocation},
    {'type': 'ineq', 'fun': constraint_max_allocation}
]

result_slsqp = minimize(
    enhanced_objective, 
    x0, 
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-9}
)

print(f"Optimization Success: {result_slsqp.success}")
print(f"Optimal Score: {marginal_benefit(result_slsqp.x):.2f}")
print(f"Total Investment: {np.sum(result_slsqp.x):.0f}亿元")

print("\nOptimal Allocation (SLSQP):")
for i, name in enumerate(dim_names):
    pct = result_slsqp.x[i] / TOTAL_BUDGET * 100
    print(f"  {name}: {result_slsqp.x[i]:.0f}亿元 ({pct:.1f}%)")

# Method 2: Differential Evolution (Global Optimization)
print("\n--- Method 2: Differential Evolution ---")

def objective_with_constraints(x):
    """Objective with penalty for constraint violations"""
    score = marginal_benefit(x)
    
    # Penalties
    penalty = 0
    
    # Budget violation
    budget_diff = abs(np.sum(x) - TOTAL_BUDGET)
    penalty += 10 * budget_diff
    
    # Min allocation violation
    min_violation = max(0, min_bound - np.min(x))
    penalty += 50 * min_violation
    
    # Max allocation violation  
    max_violation = max(0, np.max(x) - max_bound)
    penalty += 50 * max_violation
    
    # Diversification bonus
    shares = x / (np.sum(x) + 1e-10)
    hhi = np.sum(shares ** 2)
    diversification_bonus = (1 - hhi) * 5
    
    return -(score + diversification_bonus - penalty)

result_de = differential_evolution(
    objective_with_constraints,
    bounds=bounds,
    maxiter=500,
    seed=42,
    workers=1,
    polish=True,
    mutation=(0.5, 1),
    recombination=0.7
)

# Normalize to exactly use budget
result_de_x = result_de.x / np.sum(result_de.x) * TOTAL_BUDGET
result_de_x = np.clip(result_de_x, min_bound, max_bound)
result_de_x = result_de_x / np.sum(result_de_x) * TOTAL_BUDGET

print(f"Optimization Success: {result_de.success}")
print(f"Optimal Score: {marginal_benefit(result_de_x):.2f}")
print(f"Total Investment: {np.sum(result_de_x):.0f}亿元")

print("\nOptimal Allocation (Differential Evolution):")
for i, name in enumerate(dim_names):
    pct = result_de_x[i] / TOTAL_BUDGET * 100
    print(f"  {name}: {result_de_x[i]:.0f}亿元 ({pct:.1f}%)")

# Method 3: Priority-based heuristic allocation
print("\n--- Method 3: Priority-Based Heuristic ---")

def priority_allocation():
    """
    Allocate based on:
    1. Dimension weight (importance)
    2. Gap to USA (opportunity)
    3. Investment efficiency (ROI)
    4. Historical improvement rate
    """
    scores = {}
    for name in dim_names:
        w = dim_weights[name]
        gap = dim_gaps[name]
        eff = dimensions[name]['efficiency']
        imp_rate = base_improvement[name]
        
        # Composite priority score
        # Higher weight + larger gap (if positive) + higher efficiency = higher priority
        if gap > 0:  # Behind USA - prioritize catching up
            priority = w * (1 + gap/50) * eff * (1 + imp_rate)
        else:  # Ahead of USA - maintain but don't over-invest
            priority = w * 0.5 * eff * (1 + imp_rate)
        
        scores[name] = priority
    
    # Normalize and apply bounds
    total_score = sum(scores.values())
    allocation = {}
    for name in dim_names:
        raw_alloc = (scores[name] / total_score) * TOTAL_BUDGET
        allocation[name] = np.clip(raw_alloc, min_bound, max_bound)
    
    # Adjust to meet budget
    total_alloc = sum(allocation.values())
    for name in dim_names:
        allocation[name] = allocation[name] / total_alloc * TOTAL_BUDGET
    
    return np.array([allocation[name] for name in dim_names])

result_heuristic = priority_allocation()
print(f"Heuristic Score: {marginal_benefit(result_heuristic):.2f}")
print(f"Total Investment: {np.sum(result_heuristic):.0f}亿元")

print("\nOptimal Allocation (Heuristic):")
for i, name in enumerate(dim_names):
    pct = result_heuristic[i] / TOTAL_BUDGET * 100
    print(f"  {name}: {result_heuristic[i]:.0f}亿元 ({pct:.1f}%)")

# Select best result based on actual TOPSIS score
candidates = [
    ("SLSQP", result_slsqp.x),
    ("Differential Evolution", result_de_x),
    ("Priority Heuristic", result_heuristic)
]

best_score = -np.inf
optimal_x = None
optimal_method = None

print("\n--- Method Comparison ---")
for method_name, x in candidates:
    score = marginal_benefit(x)
    hhi = np.sum((x / np.sum(x)) ** 2)
    print(f"  {method_name}: Score={score:.2f}, HHI={hhi:.3f} (diversification)")
    if score > best_score:
        best_score = score
        optimal_x = x
        optimal_method = method_name

print(f"\n*** Best Method: {optimal_method} (Score: {best_score:.2f}) ***")

# ==========================================================================
# 8. Sensitivity Analysis
# ==========================================================================
print("\n--- 8. Sensitivity Analysis ---")

# Analyze score change with different budget levels
budget_levels = [5000, 7500, 10000, 12500, 15000]
sensitivity_results = []

print("\nScore Sensitivity to Budget:")
for budget in budget_levels:
    # Re-optimize with different budget
    scaled_x = optimal_x * (budget / TOTAL_BUDGET)
    score = marginal_benefit(scaled_x)
    improvement = score - china_baseline
    sensitivity_results.append({
        'Budget': budget,
        'Score': score,
        'Improvement': improvement
    })
    print(f"  {budget}亿: Score={score:.2f}, Improvement=+{improvement:.2f}")

# Analyze dimension sensitivity
print("\nDimension Sensitivity (±20% allocation change):")
for i, name in enumerate(dim_names):
    # Decrease by 20%
    x_low = optimal_x.copy()
    x_low[i] *= 0.8
    score_low = marginal_benefit(x_low)
    
    # Increase by 20%
    x_high = optimal_x.copy()
    x_high[i] *= 1.2
    # Adjust others to keep total budget
    excess = np.sum(x_high) - TOTAL_BUDGET
    for j in range(n_dims):
        if j != i:
            x_high[j] -= excess / (n_dims - 1)
    x_high = np.maximum(x_high, 0)
    score_high = marginal_benefit(x_high)
    
    sensitivity = (score_high - score_low) / (0.4 * optimal_x[i] / TOTAL_BUDGET)
    print(f"  {name}: -{optimal_x[i]*0.2:.0f}亿→{score_low:.2f}, "
          f"+{optimal_x[i]*0.2:.0f}亿→{score_high:.2f}, Sensitivity={sensitivity:.2f}")

# ==========================================================================
# 9. Results Summary
# ==========================================================================
print("\n--- 9. Results Summary ---")

# Calculate detailed improvements
print("\n" + "=" * 70)
print("OPTIMAL INVESTMENT ALLOCATION RECOMMENDATION")
print("=" * 70)

final_scores = {}
for i, name in enumerate(dim_names):
    improvement = investment_response(optimal_x[i], name, dim_china_scores[name])
    final_scores[name] = dim_china_scores[name] + improvement

china_final, all_final = calculate_topsis_score(final_scores, X_2025_norm)

print(f"\n{'Dimension':<20} {'Investment':>12} {'Pct':>8} {'2025':>8} {'2035':>8} {'Δ':>8}")
print("-" * 70)

total_investment = 0
for i, name in enumerate(dim_names):
    inv = optimal_x[i]
    pct = inv / TOTAL_BUDGET * 100
    score_2025 = dim_china_scores[name]
    score_2035 = final_scores[name]
    delta = score_2035 - score_2025
    total_investment += inv
    
    print(f"{name:<20} {inv:>10.0f}亿 {pct:>7.1f}% {score_2025:>8.1f} {score_2035:>8.1f} {delta:>+7.1f}")

print("-" * 70)
print(f"{'Total':<20} {total_investment:>10.0f}亿 {100:>7.1f}%")

print(f"\n{'Metric':<30} {'2025':>15} {'2035':>15} {'Change':>15}")
print("-" * 75)
print(f"{'China TOPSIS Score':<30} {china_baseline:>15.2f} {china_final:>15.2f} {china_final-china_baseline:>+15.2f}")
print(f"{'USA TOPSIS Score':<30} {usa_baseline:>15.2f} {all_final[usa_idx]:>15.2f} {all_final[usa_idx]-usa_baseline:>+15.2f}")
print(f"{'China-USA Gap':<30} {china_baseline-usa_baseline:>15.2f} {china_final-all_final[usa_idx]:>15.2f} {(china_final-all_final[usa_idx])-(china_baseline-usa_baseline):>+15.2f}")

# Ranking change
ranks_2025 = np.argsort(np.argsort(-all_scores_baseline)) + 1
ranks_2035 = np.argsort(np.argsort(-all_final)) + 1

print(f"\n{'Country':<12} {'Rank 2025':>12} {'Rank 2035':>12} {'Change':>12}")
print("-" * 50)
for c_idx, country in enumerate(countries):
    change = ranks_2025[c_idx] - ranks_2035[c_idx]
    change_str = f"+{change}" if change > 0 else str(change)
    print(f"{country:<12} {ranks_2025[c_idx]:>12} {ranks_2035[c_idx]:>12} {change_str:>12}")

# ==========================================================================
# 10. Visualization
# ==========================================================================
print("\n--- 10. Generating Visualizations ---")

# Figure 1: Optimal Investment Allocation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pie chart of allocation
ax1 = axes[0, 0]
colors = plt.cm.Set3(np.linspace(0, 1, n_dims))
wedges, texts, autotexts = ax1.pie(
    optimal_x, 
    labels=[f"{name}\n{optimal_x[i]:.0f}亿" for i, name in enumerate(dim_names)],
    autopct='%1.1f%%',
    colors=colors,
    explode=[0.05] * n_dims
)
ax1.set_title('Optimal Investment Allocation', fontsize=13, fontweight='bold')

# Bar chart: Before vs After
ax2 = axes[0, 1]
x = np.arange(n_dims)
width = 0.35

bars1 = ax2.bar(x - width/2, [dim_china_scores[name] for name in dim_names], 
                width, label='2025 (Before)', color='steelblue', alpha=0.7)
bars2 = ax2.bar(x + width/2, [final_scores[name] for name in dim_names], 
                width, label='2035 (After)', color='coral', alpha=0.7)

ax2.set_xticks(x)
ax2.set_xticklabels(dim_names, rotation=30, ha='right')
ax2.set_ylabel('Dimension Score', fontsize=11)
ax2.set_title('Dimension Score Improvement', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, score in zip(bars2, [final_scores[name] for name in dim_names]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{score:.0f}', ha='center', va='bottom', fontsize=9)

# Marginal benefit vs investment
ax3 = axes[1, 0]
mb_values = [marginal_benefits[name] for name in dim_names]
inv_values = [optimal_x[i] for i in range(n_dims)]

scatter = ax3.scatter(mb_values, inv_values, s=200, c=colors, edgecolors='black', linewidths=2)
for i, name in enumerate(dim_names):
    ax3.annotate(name, (mb_values[i], inv_values[i]), 
                 textcoords="offset points", xytext=(5, 5), fontsize=10)

ax3.set_xlabel('Marginal Benefit (points per 1000亿)', fontsize=11)
ax3.set_ylabel('Optimal Investment (亿)', fontsize=11)
ax3.set_title('Marginal Benefit vs Optimal Investment', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Score trajectory
ax4 = axes[1, 1]
budget_pcts = [50, 60, 70, 80, 90, 100, 110, 120]
trajectory_scores = []
for pct in budget_pcts:
    scaled_x = optimal_x * (pct / 100)
    score = marginal_benefit(scaled_x)
    trajectory_scores.append(score)

ax4.plot(budget_pcts, trajectory_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
ax4.axhline(y=usa_baseline, color='red', linestyle='--', linewidth=2, label=f'USA Baseline ({usa_baseline:.1f})')
ax4.axhline(y=china_baseline, color='gray', linestyle='--', linewidth=2, label=f'China Baseline ({china_baseline:.1f})')
ax4.axvline(x=100, color='green', linestyle=':', linewidth=2, label='Budget (10000亿)')

ax4.set_xlabel('Budget Percentage (%)', fontsize=11)
ax4.set_ylabel('China TOPSIS Score', fontsize=11)
ax4.set_title('Score vs Investment Level', fontsize=13, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

plt.suptitle('Q41: Optimal Investment Allocation Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q41_investment_optimization.png', dpi=150, bbox_inches='tight')
print("Saved: Q41_investment_optimization.png")
plt.close()

# Figure 2: China vs USA Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Radar chart
ax1 = axes[0]
angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
angles += angles[:1]

china_2025_vals = [dim_china_scores[name] for name in dim_names]
china_2025_vals += china_2025_vals[:1]

china_2035_vals = [final_scores[name] for name in dim_names]
china_2035_vals += china_2035_vals[:1]

usa_vals = [dim_usa_scores[name] for name in dim_names]
usa_vals += usa_vals[:1]

ax1 = plt.subplot(1, 2, 1, projection='polar')
ax1.plot(angles, china_2025_vals, 'o-', linewidth=2, color='blue', label='China 2025', alpha=0.7)
ax1.plot(angles, china_2035_vals, 's-', linewidth=2, color='red', label='China 2035 (Optimized)')
ax1.plot(angles, usa_vals, '^--', linewidth=2, color='green', label='USA 2025', alpha=0.7)
ax1.fill(angles, china_2035_vals, alpha=0.1, color='red')

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(dim_names, fontsize=10)
ax1.set_ylim(0, 100)
ax1.set_title('Dimension Comparison Radar', fontsize=13, fontweight='bold', pad=20)
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)

# Gap reduction chart
ax2 = plt.subplot(1, 2, 2)
gaps_2025 = [dim_gaps[name] for name in dim_names]
gaps_2035 = [dim_usa_scores[name] - final_scores[name] for name in dim_names]

x = np.arange(n_dims)
width = 0.35

bars1 = ax2.bar(x - width/2, gaps_2025, width, label='Gap 2025', color='coral', alpha=0.7)
bars2 = ax2.bar(x + width/2, gaps_2035, width, label='Gap 2035', color='steelblue', alpha=0.7)

ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(dim_names, rotation=30, ha='right')
ax2.set_ylabel('China-USA Gap (USA - China)', fontsize=11)
ax2.set_title('Gap Reduction by Dimension', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Q41: China-USA Competitiveness Gap Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q41_gap_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: Q41_gap_analysis.png")
plt.close()

# ==========================================================================
# 11. Save Results
# ==========================================================================
print("\n--- 11. Saving Results ---")

# Optimal allocation
alloc_df = pd.DataFrame({
    'Dimension': dim_names,
    'Dimension_CN': [dimensions[name]['name_cn'] for name in dim_names],
    'Investment_Billion': optimal_x,
    'Percentage': optimal_x / TOTAL_BUDGET * 100,
    'Score_2025': [dim_china_scores[name] for name in dim_names],
    'Score_2035': [final_scores[name] for name in dim_names],
    'Improvement': [final_scores[name] - dim_china_scores[name] for name in dim_names],
    'Marginal_Benefit': [marginal_benefits[name] for name in dim_names],
    'Gap_2025': [dim_gaps[name] for name in dim_names],
    'Gap_2035': [dim_usa_scores[name] - final_scores[name] for name in dim_names]
})
alloc_df.to_csv('Q41_optimal_allocation.csv', index=False)
print("Saved: Q41_optimal_allocation.csv")

# Sensitivity analysis
sens_df = pd.DataFrame(sensitivity_results)
sens_df.to_csv('Q41_sensitivity_analysis.csv', index=False)
print("Saved: Q41_sensitivity_analysis.csv")

# Summary metrics
summary_df = pd.DataFrame({
    'Metric': ['Total_Budget', 'China_Score_2025', 'China_Score_2035', 'Score_Improvement',
               'USA_Score_2025', 'China_Rank_2025', 'China_Rank_2035', 'Rank_Change'],
    'Value': [TOTAL_BUDGET, china_baseline, china_final, china_final - china_baseline,
              usa_baseline, ranks_2025[china_idx], ranks_2035[china_idx], 
              ranks_2025[china_idx] - ranks_2035[china_idx]]
})
summary_df.to_csv('Q41_summary.csv', index=False)
print("Saved: Q41_summary.csv")

# ==========================================================================
# Final Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q41 INVESTMENT OPTIMIZATION COMPLETE")
print("=" * 70)

print(f"""
INVESTMENT RECOMMENDATION SUMMARY
{'='*50}

Total Budget: {TOTAL_BUDGET}亿元 (1万亿人民币)

OPTIMAL ALLOCATION:
""")

for i, name in enumerate(dim_names):
    pct = optimal_x[i] / TOTAL_BUDGET * 100
    print(f"  {dimensions[name]['name_cn']}({name}): {optimal_x[i]:.0f}亿元 ({pct:.1f}%)")

print(f"""
EXPECTED OUTCOMES (2035):
  China Score: {china_baseline:.2f} → {china_final:.2f} (+{china_final-china_baseline:.2f})
  China Rank: {ranks_2025[china_idx]} → {ranks_2035[china_idx]}
  vs USA Gap: {china_baseline-usa_baseline:.2f} → {china_final-all_final[usa_idx]:.2f}

KEY INSIGHTS:
  1. Highest priority: Innovation (highest marginal benefit × weight)
  2. Quick wins: Industry, Computing Power (shorter lag time)
  3. Long-term: Talent, Innovation (higher impact but longer lag)
  4. Support role: Policy, Economy (enabling factors)

Output Files:
  - Q41_optimal_allocation.csv
  - Q41_sensitivity_analysis.csv
  - Q41_summary.csv
  - Q41_investment_optimization.png
  - Q41_gap_analysis.png
""")
print("=" * 70)

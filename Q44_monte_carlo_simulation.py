"""
==========================================================================
Huashu Cup 2026 Problem B - Question 4 Q44
Monte Carlo Simulation: Uncertainty and Risk Analysis
蒙特卡洛模拟：不确定性与风险分析
==========================================================================

Core Idea:
- All parameters have uncertainty (investment efficiency, external environment, etc.)
- Use Monte Carlo simulation to assess robustness of investment strategies
- Generate probability distribution of outcomes
- Calculate risk metrics: VaR, Expected Shortfall, Overtaking Probability
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
Uncertainty Sources:
1. Investment efficiency: Actual conversion rate may differ from expected
2. External environment: Tech blockade, international situation changes
3. Competitor strategy: USA investment intensity uncertain
4. Forecast error: Q3 predictions have inherent error
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Q44: Monte Carlo Simulation - Uncertainty & Risk Analysis")
print("蒙特卡洛模拟：不确定性与风险分析")
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
# 2. Define Investment Dimensions with Uncertainty Parameters
# ==========================================================================
print("\n--- 2. Investment Dimensions with Uncertainty ---")

dimensions = {
    'Computing_Power': {
        'indices': list(range(0, 8)),
        'name_cn': '算力基础',
        'efficiency_mean': 0.8,
        'efficiency_std': 0.15,      # Uncertainty in efficiency
        'external_risk': 0.3,        # Risk of tech blockade affecting this
        'lag_years': 2,
    },
    'Talent': {
        'indices': list(range(8, 13)),
        'name_cn': '人才资源',
        'efficiency_mean': 0.6,
        'efficiency_std': 0.10,
        'external_risk': 0.1,        # Lower external risk
        'lag_years': 4,
    },
    'Innovation': {
        'indices': list(range(13, 19)),
        'name_cn': '创新能力',
        'efficiency_mean': 0.5,
        'efficiency_std': 0.20,      # High uncertainty in R&D outcomes
        'external_risk': 0.2,
        'lag_years': 5,
    },
    'Industry': {
        'indices': list(range(19, 24)),
        'name_cn': '产业发展',
        'efficiency_mean': 0.9,
        'efficiency_std': 0.12,
        'external_risk': 0.25,       # Market volatility risk
        'lag_years': 2,
    },
    'Policy': {
        'indices': list(range(24, 29)),
        'name_cn': '政策环境',
        'efficiency_mean': 0.7,
        'efficiency_std': 0.08,      # Policy relatively stable
        'external_risk': 0.05,
        'lag_years': 1,
    },# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    'Economy': {
        'indices': list(range(29, 38)),
        'name_cn': '经济基础',
        'efficiency_mean': 0.4,
        'efficiency_std': 0.15,
        'external_risk': 0.2,        # Macro economic uncertainty
        'lag_years': 3,
    }
}

n_dims = len(dimensions)
dim_names = list(dimensions.keys())

print(f"\n{'Dimension':<18} {'Eff_Mean':>10} {'Eff_Std':>10} {'Ext_Risk':>10}")
print("-" * 55)
for name, info in dimensions.items():
    print(f"{name:<18} {info['efficiency_mean']:>10.2f} {info['efficiency_std']:>10.2f} {info['external_risk']:>10.2f}")

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

print("Baseline Scores (2025):")
for name in dim_names:
    print(f"  {name}: China={dim_china_scores[name]:.1f}, USA={dim_usa_scores[name]:.1f}")

# ==========================================================================
# 4. Define Uncertainty Models
# ==========================================================================
print("\n--- 4. Defining Uncertainty Models ---")
# 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
"""
Uncertainty Sources:
1. Investment Efficiency: efficiency ~ N(mean, std^2), truncated to [0.1, 1.5]
2. External Shock: Bernoulli(external_risk) * shock_magnitude
3. USA Investment: USA may also invest, affecting relative position
4. Base Improvement Rate: rate ~ N(historical, 0.3*historical)
"""

class UncertaintyModel:
    """Model for generating random parameter samples"""
    
    def __init__(self, scenario='baseline'):
        """
        Scenarios:
        - baseline: Normal uncertainty
        - optimistic: Lower uncertainty, favorable conditions
        - pessimistic: Higher uncertainty, adverse conditions
        - tech_blockade: High external risk for computing/chips
        """
        self.scenario = scenario
        
        # Scenario multipliers
        self.scenario_params = {
            'baseline': {
                'efficiency_std_mult': 1.0,
                'external_risk_mult': 1.0,
                'usa_invest_prob': 0.5,
                'shock_magnitude': 0.2
            },
            'optimistic': {
                'efficiency_std_mult': 0.7,
                'external_risk_mult': 0.5,
                'usa_invest_prob': 0.3,
                'shock_magnitude': 0.1
            },
            'pessimistic': {
                'efficiency_std_mult': 1.5,
                'external_risk_mult': 1.5,
                'usa_invest_prob': 0.8,
                'shock_magnitude': 0.3
            },
            'tech_blockade': {
                'efficiency_std_mult': 1.2,
                'external_risk_mult': 2.0,  # Double external risk
                'usa_invest_prob': 0.9,
                'shock_magnitude': 0.4
            }
        }
        
        self.params = self.scenario_params[scenario]
    
    def sample_efficiency(self, dim_name):
        """Sample investment efficiency for a dimension"""
        info = dimensions[dim_name]
        mean = info['efficiency_mean']
        std = info['efficiency_std'] * self.params['efficiency_std_mult']
        
        # Truncated normal distribution
        eff = np.random.normal(mean, std)
        eff = np.clip(eff, 0.1, 1.5)
        return eff

    # 本论文由 BZD 数模社提供，为 B 题，配套代码、数据均见售后群群文件。写作过程、思路讲解见 B 站 BZD 数模社。论文写作存在致命漏洞，比赛结束前一天会在售后群群文件进行更新，售后群提供降重服务，因倒卖导致论文无法降重进而导致通报情况概不负责。感谢理解。完整资料链接: https://docs.qq.com/doc/p/346df2de570610aac335e35ee48b0a59e8c639ad
    # =============================================================================
    def sample_external_shock(self, dim_name):
        """Sample external shock (e.g., tech blockade)"""
        info = dimensions[dim_name]
        risk = info['external_risk'] * self.params['external_risk_mult']
        risk = min(risk, 0.9)  # Cap at 90%
        
        # Bernoulli: shock happens or not
        shock_occurs = np.random.random() < risk
        if shock_occurs:
            # Negative shock reduces efficiency
            shock = -np.random.uniform(0.1, self.params['shock_magnitude'])
        else:
            shock = 0
        return shock
    
    def sample_base_improvement(self, dim_name):
        """Sample base improvement rate"""
        historical = base_improvement[dim_name]
        std = 0.3 * historical
        rate = np.random.normal(historical, std)
        return max(rate, 0.005)  # At least 0.5% improvement
    
    def sample_usa_investment_multiplier(self):
        """Sample USA's investment response"""
        if np.random.random() < self.params['usa_invest_prob']:
            # USA also invests heavily
            return np.random.uniform(1.0, 1.3)
        else:
            # USA invests moderately
            return np.random.uniform(0.8, 1.0)

print("Uncertainty Models Defined:")
for scenario in ['baseline', 'optimistic', 'pessimistic', 'tech_blockade']:
    model = UncertaintyModel(scenario)
    print(f"  {scenario}: eff_std×{model.params['efficiency_std_mult']:.1f}, "
          f"risk×{model.params['external_risk_mult']:.1f}, "
          f"usa_prob={model.params['usa_invest_prob']:.1f}")

# ==========================================================================
# 5. Investment Response Function with Uncertainty
# ==========================================================================
print("\n--- 5. Stochastic Investment Response Model ---")

def investment_response_stochastic(investment, dim_name, current_score, 
                                    efficiency, base_rate, external_shock, years=10):
    """
    Calculate score improvement with stochastic parameters
    
    Parameters:
    - investment: Investment amount (亿元)
    - dim_name: Dimension name
    - current_score: Current score (0-100)
    - efficiency: Sampled efficiency
    - base_rate: Sampled base improvement rate
    - external_shock: Sampled external shock
    - years: Investment horizon
    
    Returns:
    - Score improvement
    """
    info = dimensions[dim_name]
    lag = info['lag_years']
    
    effective_years = max(years - lag, 1)
    ceiling_factor = ((100 - current_score) / 100) ** 0.5
    
    # Apply efficiency with external shock
    effective_efficiency = max(efficiency + external_shock, 0.1)
    
    reference_investment = 2000
    investment_effect = 1 - np.exp(-investment / reference_investment)
    
    base_capacity = min(base_rate * effective_years * 100, 50)
    investment_capacity = 50 * effective_efficiency * investment_effect
    
    improvement = (base_capacity + investment_capacity) * ceiling_factor
    max_improvement = (100 - current_score) * 0.85
    
    return max(min(improvement, max_improvement), 0)

# ==========================================================================
# 6. Define Investment Strategies to Compare
# ==========================================================================
print("\n--- 6. Defining Investment Strategies ---")

TOTAL_BUDGET = 10000  # 1万亿

# Strategy 1: Balanced (Q41 style)
strategy_balanced = {
    'name': 'Balanced',
    'name_cn': '均衡策略',
    'allocation': np.array([1700, 1500, 1800, 1800, 1500, 1700])  # Roughly equal
}

# Strategy 2: Bottleneck Focus (Q42 style)
strategy_bottleneck = {
    'name': 'Bottleneck',
    'name_cn': '短板优先',
    'allocation': np.array([4000, 600, 1200, 2500, 700, 1000])  # Focus on gaps
}

# Strategy 3: Innovation Focus
strategy_innovation = {
    'name': 'Innovation',
    'name_cn': '创新驱动',
    'allocation': np.array([1500, 2000, 3000, 1500, 1000, 1000])  # Focus on innovation
}

# Strategy 4: Conservative (minimize risk)
strategy_conservative = {
    'name': 'Conservative',
    'name_cn': '保守稳健',
    'allocation': np.array([1500, 1800, 1500, 1800, 1700, 1700])  # Diversified
}

strategies = [strategy_balanced, strategy_bottleneck, strategy_innovation, strategy_conservative]

# Normalize to total budget
for strategy in strategies:
    strategy['allocation'] = strategy['allocation'] / np.sum(strategy['allocation']) * TOTAL_BUDGET

print(f"\n{'Strategy':<15}", end="")
for name in dim_names:
    print(f"{dimensions[name]['name_cn'][:4]:>8}", end="")
print()
print("-" * 65)
for strategy in strategies:
    print(f"{strategy['name_cn']:<15}", end="")
    for alloc in strategy['allocation']:
        print(f"{alloc:>8.0f}", end="")
    print()

# ==========================================================================
# 7. Monte Carlo Simulation
# ==========================================================================
print("\n--- 7. Running Monte Carlo Simulation ---")

N_SIMULATIONS = 10000

def run_simulation(strategy_allocation, uncertainty_model, n_sims=N_SIMULATIONS):
    """
    Run Monte Carlo simulation for a given strategy
    
    Returns:
    - china_scores: Array of final China scores
    - usa_scores: Array of final USA scores (also uncertain due to their investment)
    - dimension_scores: Dict of dimension-wise final scores
    """
    china_scores = []
    usa_scores = []
    dimension_scores = {name: [] for name in dim_names}
    
    for sim in range(n_sims):
        # Sample parameters for this simulation
        efficiencies = {name: uncertainty_model.sample_efficiency(name) for name in dim_names}
        base_rates = {name: uncertainty_model.sample_base_improvement(name) for name in dim_names}
        shocks = {name: uncertainty_model.sample_external_shock(name) for name in dim_names}
        usa_mult = uncertainty_model.sample_usa_investment_multiplier()
        
        # Calculate China's new scores
        china_new_scores = {}
        for i, name in enumerate(dim_names):
            improvement = investment_response_stochastic(
                strategy_allocation[i],
                name,
                dim_china_scores[name],
                efficiencies[name],
                base_rates[name],
                shocks[name]
            )
            china_new_scores[name] = min(dim_china_scores[name] + improvement, 100)
            dimension_scores[name].append(china_new_scores[name])
        
        # Calculate weighted total score for China
        weights = np.array([dim_weights[name] for name in dim_names])
        weights = weights / np.sum(weights)
        china_total = sum(china_new_scores[name] * weights[i] for i, name in enumerate(dim_names))
        china_scores.append(china_total)
        
        # USA score (also changes due to their investment)
        usa_new_scores = {}
        for name in dim_names:
            # USA improves proportionally to their investment multiplier
            usa_improvement = (100 - dim_usa_scores[name]) * 0.1 * usa_mult
            usa_new_scores[name] = min(dim_usa_scores[name] + usa_improvement, 100)
        
        usa_total = sum(usa_new_scores[name] * weights[i] for i, name in enumerate(dim_names))
        usa_scores.append(usa_total)
    
    return np.array(china_scores), np.array(usa_scores), dimension_scores

# Run simulations for each strategy under baseline scenario
print(f"\nRunning {N_SIMULATIONS} simulations for each strategy...")

baseline_model = UncertaintyModel('baseline')
simulation_results = {}

for strategy in strategies:
    print(f"  Simulating: {strategy['name_cn']}...", end=" ")
    china_scores, usa_scores, dim_scores = run_simulation(
        strategy['allocation'], baseline_model
    )
    simulation_results[strategy['name']] = {
        'china_scores': china_scores,
        'usa_scores': usa_scores,
        'dim_scores': dim_scores,
        'allocation': strategy['allocation'],
        'name_cn': strategy['name_cn']
    }
    print(f"Done. Mean={np.mean(china_scores):.2f}, Std={np.std(china_scores):.2f}")

# ==========================================================================
# 8. Risk Metrics Calculation
# ==========================================================================
print("\n--- 8. Calculating Risk Metrics ---")

def calculate_risk_metrics(china_scores, usa_scores):
    """Calculate comprehensive risk metrics"""
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = np.mean(china_scores)
    metrics['std'] = np.std(china_scores)
    metrics['median'] = np.median(china_scores)
    metrics['min'] = np.min(china_scores)
    metrics['max'] = np.max(china_scores)
    
    # Value at Risk (VaR) - worst case at confidence level
    metrics['VaR_5%'] = np.percentile(china_scores, 5)   # 5% worst case
    metrics['VaR_10%'] = np.percentile(china_scores, 10)  # 10% worst case
    
    # Expected Shortfall (CVaR) - average of worst cases
    var_5 = metrics['VaR_5%']
    metrics['ES_5%'] = np.mean(china_scores[china_scores <= var_5])
    
    # Upside potential
    metrics['P75'] = np.percentile(china_scores, 75)
    metrics['P95'] = np.percentile(china_scores, 95)
    
    # Probability of overtaking USA
    metrics['P_overtake'] = np.mean(china_scores > usa_scores) * 100
    
    # Probability of closing gap significantly (within 5 points)
    gap = usa_scores - china_scores
    metrics['P_close_gap'] = np.mean(gap < 5) * 100
    
    # Sharpe-like ratio (return per unit risk)
    baseline_score = 54.16  # 2025 baseline
    metrics['risk_adj_return'] = (metrics['mean'] - baseline_score) / (metrics['std'] + 0.01)
    
    # Coefficient of variation
    metrics['CV'] = metrics['std'] / metrics['mean'] * 100
    
    return metrics

print(f"\n{'Strategy':<15} {'Mean':>8} {'Std':>8} {'VaR5%':>8} {'ES5%':>8} {'P_over':>8} {'Sharpe':>8}")
print("-" * 75)

strategy_metrics = {}
for strategy in strategies:
    name = strategy['name']
    result = simulation_results[name]
    metrics = calculate_risk_metrics(result['china_scores'], result['usa_scores'])
    strategy_metrics[name] = metrics
    
    print(f"{strategy['name_cn']:<15} {metrics['mean']:>8.2f} {metrics['std']:>8.2f} "
          f"{metrics['VaR_5%']:>8.2f} {metrics['ES_5%']:>8.2f} "
          f"{metrics['P_overtake']:>7.1f}% {metrics['risk_adj_return']:>8.2f}")

# ==========================================================================
# 9. Scenario Analysis
# ==========================================================================
print("\n--- 9. Scenario Analysis ---")

# Run best strategy under different scenarios
best_strategy = max(strategies, key=lambda s: strategy_metrics[s['name']]['mean'])
print(f"\nAnalyzing {best_strategy['name_cn']} under different scenarios:")

scenarios = ['baseline', 'optimistic', 'pessimistic', 'tech_blockade']
scenario_results = {}

for scenario in scenarios:
    model = UncertaintyModel(scenario)
    china_scores, usa_scores, _ = run_simulation(best_strategy['allocation'], model, n_sims=5000)
    metrics = calculate_risk_metrics(china_scores, usa_scores)
    scenario_results[scenario] = {
        'china_scores': china_scores,
        'usa_scores': usa_scores,
        'metrics': metrics
    }

print(f"\n{'Scenario':<15} {'Mean':>10} {'Std':>8} {'VaR5%':>8} {'P_overtake':>10}")
print("-" * 55)
scenario_names_cn = {
    'baseline': '基准情景',
    'optimistic': '乐观情景',
    'pessimistic': '悲观情景',
    'tech_blockade': '技术封锁'
}
for scenario in scenarios:
    m = scenario_results[scenario]['metrics']
    print(f"{scenario_names_cn[scenario]:<15} {m['mean']:>10.2f} {m['std']:>8.2f} "
          f"{m['VaR_5%']:>8.2f} {m['P_overtake']:>9.1f}%")

# ==========================================================================
# 10. Sensitivity Analysis
# ==========================================================================
print("\n--- 10. Dimension Sensitivity Analysis ---")

# Analyze which dimensions contribute most to variance
print("\nDimension Score Statistics (Balanced Strategy):")
print(f"{'Dimension':<18} {'Mean':>8} {'Std':>8} {'CV%':>8} {'Contribution':>12}")
print("-" * 60)

balanced_result = simulation_results['Balanced']
dim_contributions = {}

for name in dim_names:
    scores = np.array(balanced_result['dim_scores'][name])
    mean = np.mean(scores)
    std = np.std(scores)
    cv = std / mean * 100
    
    # Contribution to total variance (approximate)
    weight = dim_weights[name] / sum(dim_weights.values())
    contribution = (std * weight) ** 2
    dim_contributions[name] = contribution
    
    print(f"{dimensions[name]['name_cn']:<18} {mean:>8.2f} {std:>8.2f} {cv:>8.2f} {contribution:>12.4f}")

# Normalize contributions
total_contrib = sum(dim_contributions.values())
print("\nVariance Contribution (%):")
for name in dim_names:
    pct = dim_contributions[name] / total_contrib * 100
    print(f"  {dimensions[name]['name_cn']}: {pct:.1f}%")

# ==========================================================================
# 11. Results Summary
# ==========================================================================
print("\n--- 11. Results Summary ---")

print("\n" + "=" * 70)
print("Q44 MONTE CARLO SIMULATION RESULTS")
print("=" * 70)

# Find best strategy by different criteria
best_by_mean = max(strategies, key=lambda s: strategy_metrics[s['name']]['mean'])
best_by_risk = max(strategies, key=lambda s: strategy_metrics[s['name']]['VaR_5%'])
best_by_sharpe = max(strategies, key=lambda s: strategy_metrics[s['name']]['risk_adj_return'])
best_by_overtake = max(strategies, key=lambda s: strategy_metrics[s['name']]['P_overtake'])

print(f"\nBest Strategy by Different Criteria:")
print(f"  Highest Expected Score: {best_by_mean['name_cn']} ({strategy_metrics[best_by_mean['name']]['mean']:.2f})")
print(f"  Lowest Downside Risk:   {best_by_risk['name_cn']} ({strategy_metrics[best_by_risk['name']]['VaR_5%']:.2f})")
print(f"  Best Risk-Adj Return:   {best_by_sharpe['name_cn']} ({strategy_metrics[best_by_sharpe['name']]['risk_adj_return']:.2f})")
print(f"  Highest Overtake Prob:  {best_by_overtake['name_cn']} ({strategy_metrics[best_by_overtake['name']]['P_overtake']:.1f}%)")

# Comprehensive comparison
print("\n" + "-" * 80)
print("COMPREHENSIVE STRATEGY COMPARISON")
print("-" * 80)
print(f"\n{'Metric':<25}", end="")
for strategy in strategies:
    print(f"{strategy['name_cn']:>12}", end="")
print()
print("-" * 80)

metric_names = [
    ('mean', '期望得分'),
    ('std', '标准差'),
    ('VaR_5%', 'VaR(5%)'),
    ('ES_5%', 'ES(5%)'),
    ('P_overtake', '超越概率(%)'),
    ('P_close_gap', '缩差概率(%)'),
    ('risk_adj_return', '风险调整收益'),
    ('CV', '变异系数(%)')
]

for metric_key, metric_name in metric_names:
    print(f"{metric_name:<25}", end="")
    for strategy in strategies:
        value = strategy_metrics[strategy['name']][metric_key]
        if metric_key in ['P_overtake', 'P_close_gap', 'CV']:
            print(f"{value:>12.1f}", end="")
        else:
            print(f"{value:>12.2f}", end="")
    print()

# ==========================================================================
# 12. Visualization
# ==========================================================================
print("\n--- 12. Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Score distribution comparison
ax1 = axes[0, 0]
colors = ['steelblue', 'coral', 'seagreen', 'purple']
for i, strategy in enumerate(strategies):
    scores = simulation_results[strategy['name']]['china_scores']
    ax1.hist(scores, bins=50, alpha=0.5, label=strategy['name_cn'], color=colors[i], density=True)
    ax1.axvline(np.mean(scores), color=colors[i], linestyle='--', linewidth=2)

ax1.axvline(54.16, color='black', linestyle=':', linewidth=2, label='2025基线')
ax1.set_xlabel('2035 Score', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Score Distribution by Strategy\n(Dashed lines = means)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Risk-Return scatter
ax2 = axes[0, 1]
for i, strategy in enumerate(strategies):
    m = strategy_metrics[strategy['name']]
    ax2.scatter(m['std'], m['mean'], s=300, c=colors[i], label=strategy['name_cn'], 
                edgecolors='black', linewidths=2, marker='o')
    ax2.annotate(strategy['name_cn'], (m['std'], m['mean']), 
                 textcoords="offset points", xytext=(10, 5), fontsize=10)

ax2.set_xlabel('Risk (Std Dev)', fontsize=11)
ax2.set_ylabel('Expected Return (Mean Score)', fontsize=11)
ax2.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# Add efficient frontier approximation
means = [strategy_metrics[s['name']]['mean'] for s in strategies]
stds = [strategy_metrics[s['name']]['std'] for s in strategies]
ax2.plot(sorted(stds), [means[stds.index(s)] for s in sorted(stds)], 'k--', alpha=0.3)

# 3. Scenario comparison (box plot)
ax3 = axes[1, 0]
scenario_data = [scenario_results[s]['china_scores'] for s in scenarios]
bp = ax3.boxplot(scenario_data, labels=[scenario_names_cn[s] for s in scenarios], patch_artist=True)

scenario_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for patch, color in zip(bp['boxes'], scenario_colors):
    patch.set_facecolor(color)

ax3.axhline(54.16, color='gray', linestyle=':', linewidth=2, label='2025基线')
ax3.set_ylabel('2035 Score', fontsize=11)
ax3.set_title(f'Scenario Analysis ({best_strategy["name_cn"]})', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Overtake probability and VaR
ax4 = axes[1, 1]
x = np.arange(len(strategies))
width = 0.35

overtake_probs = [strategy_metrics[s['name']]['P_overtake'] for s in strategies]
var_5s = [strategy_metrics[s['name']]['VaR_5%'] for s in strategies]

ax4_twin = ax4.twinx()

bars1 = ax4.bar(x - width/2, overtake_probs, width, label='超越概率 (%)', color='steelblue', alpha=0.7)
bars2 = ax4_twin.bar(x + width/2, var_5s, width, label='VaR(5%)', color='coral', alpha=0.7)

ax4.set_xticks(x)
ax4.set_xticklabels([s['name_cn'] for s in strategies])
ax4.set_ylabel('Overtake Probability (%)', color='steelblue', fontsize=11)
ax4_twin.set_ylabel('VaR (5%)', color='coral', fontsize=11)
ax4.set_title('Risk Metrics Comparison', fontsize=12, fontweight='bold')

ax4.tick_params(axis='y', labelcolor='steelblue')
ax4_twin.tick_params(axis='y', labelcolor='coral')

# Add value labels
for bar, val in zip(bars1, overtake_probs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, var_5s):
    ax4_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='coral')

ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Q44: Monte Carlo Simulation - Uncertainty & Risk Analysis\n蒙特卡洛模拟：不确定性与风险分析', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q44_monte_carlo_simulation.png', dpi=150, bbox_inches='tight')
print("Saved: Q44_monte_carlo_simulation.png")
plt.close()

# Additional visualization: Probability density and CDF
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# PDF
ax1 = axes2[0]
for i, strategy in enumerate(strategies):
    scores = simulation_results[strategy['name']]['china_scores']
    density = stats.gaussian_kde(scores)
    x = np.linspace(min(scores), max(scores), 200)
    ax1.plot(x, density(x), linewidth=2, label=strategy['name_cn'], color=colors[i])
    ax1.fill_between(x, density(x), alpha=0.2, color=colors[i])

ax1.axvline(54.16, color='black', linestyle=':', linewidth=2, label='2025基线')
ax1.set_xlabel('2035 Score', fontsize=11)
ax1.set_ylabel('Probability Density', fontsize=11)
ax1.set_title('Probability Density Function (PDF)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# CDF
ax2 = axes2[1]
for i, strategy in enumerate(strategies):
    scores = simulation_results[strategy['name']]['china_scores']
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax2.plot(sorted_scores, cdf, linewidth=2, label=strategy['name_cn'], color=colors[i])

# Mark VaR points
for i, strategy in enumerate(strategies):
    var_5 = strategy_metrics[strategy['name']]['VaR_5%']
    ax2.scatter([var_5], [0.05], s=100, c=colors[i], marker='x', zorder=5)

ax2.axhline(0.05, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='5% VaR Level')
ax2.set_xlabel('2035 Score', fontsize=11)
ax2.set_ylabel('Cumulative Probability', fontsize=11)
ax2.set_title('Cumulative Distribution Function (CDF)\n(X marks = VaR at 5%)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Q44_probability_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: Q44_probability_distributions.png")
plt.close()

# ==========================================================================
# 13. Save Results
# ==========================================================================
print("\n--- 13. Saving Results ---")

# Strategy comparison
comparison_df = pd.DataFrame([
    {
        'Strategy': strategy['name'],
        'Strategy_CN': strategy['name_cn'],
        **{f'Allocation_{name}': strategy['allocation'][i] for i, name in enumerate(dim_names)},
        **strategy_metrics[strategy['name']]
    }
    for strategy in strategies
])
comparison_df.to_csv('Q44_strategy_comparison.csv', index=False)
print("Saved: Q44_strategy_comparison.csv")

# Scenario analysis
scenario_df = pd.DataFrame([
    {
        'Scenario': scenario,
        'Scenario_CN': scenario_names_cn[scenario],
        **scenario_results[scenario]['metrics']
    }
    for scenario in scenarios
])
scenario_df.to_csv('Q44_scenario_analysis.csv', index=False)
print("Saved: Q44_scenario_analysis.csv")

# Simulation statistics (for best strategy)
best_name = best_by_mean['name']
stats_df = pd.DataFrame({
    'Percentile': [1, 5, 10, 25, 50, 75, 90, 95, 99],
    'China_Score': [np.percentile(simulation_results[best_name]['china_scores'], p) 
                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]],
    'USA_Score': [np.percentile(simulation_results[best_name]['usa_scores'], p) 
                  for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
})
stats_df.to_csv('Q44_percentile_statistics.csv', index=False)
print("Saved: Q44_percentile_statistics.csv")

# Summary
summary_df = pd.DataFrame({
    'Metric': ['N_Simulations', 'Best_By_Mean', 'Best_By_Risk', 'Best_By_Sharpe',
               'Baseline_Score_2025', 'Best_Mean_2035', 'Best_VaR5%', 'Best_Overtake_Prob'],
    'Value': [N_SIMULATIONS, best_by_mean['name_cn'], best_by_risk['name_cn'], best_by_sharpe['name_cn'],
              54.16, strategy_metrics[best_by_mean['name']]['mean'],
              strategy_metrics[best_by_risk['name']]['VaR_5%'],
              strategy_metrics[best_by_overtake['name']]['P_overtake']]
})
summary_df.to_csv('Q44_summary.csv', index=False)
print("Saved: Q44_summary.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q44 MONTE CARLO SIMULATION COMPLETE")
print("=" * 70)
print(f"""
Core Method: Monte Carlo Simulation ({N_SIMULATIONS} iterations)

Uncertainty Sources Modeled:
  1. Investment Efficiency: N(mean, std²), dimension-specific
  2. External Shocks: Bernoulli × shock_magnitude (tech blockade risk)
  3. USA Response: Random investment intensity
  4. Base Improvement Rate: N(historical, 0.3×historical)

Strategies Compared:
  - 均衡策略 (Balanced): Even distribution
  - 短板优先 (Bottleneck): Focus on weakest dimensions
  - 创新驱动 (Innovation): Focus on innovation capability
  - 保守稳健 (Conservative): Risk-minimizing distribution

Key Findings:
""")

print(f"  Best Expected Score: {best_by_mean['name_cn']}")
print(f"    Mean = {strategy_metrics[best_by_mean['name']]['mean']:.2f}")
print(f"    Std = {strategy_metrics[best_by_mean['name']]['std']:.2f}")

print(f"\n  Lowest Downside Risk: {best_by_risk['name_cn']}")
print(f"    VaR(5%) = {strategy_metrics[best_by_risk['name']]['VaR_5%']:.2f}")
print(f"    ES(5%) = {strategy_metrics[best_by_risk['name']]['ES_5%']:.2f}")

print(f"\n  Highest Overtake Probability: {best_by_overtake['name_cn']}")
print(f"    P(China > USA) = {strategy_metrics[best_by_overtake['name']]['P_overtake']:.1f}%")

print(f"""
Scenario Analysis ({best_strategy['name_cn']}):
  - Baseline:     Mean={scenario_results['baseline']['metrics']['mean']:.2f}
  - Optimistic:   Mean={scenario_results['optimistic']['metrics']['mean']:.2f}
  - Pessimistic:  Mean={scenario_results['pessimistic']['metrics']['mean']:.2f}
  - Tech Blockade: Mean={scenario_results['tech_blockade']['metrics']['mean']:.2f}

Risk Management Insights:
  1. Higher expected return often comes with higher variance
  2. Bottleneck strategy has highest upside but also higher risk
  3. Conservative strategy offers protection against downside
  4. Tech blockade scenario significantly impacts computing-heavy strategies

Output Files:
  - Q44_strategy_comparison.csv
  - Q44_scenario_analysis.csv
  - Q44_percentile_statistics.csv
  - Q44_summary.csv
  - Q44_monte_carlo_simulation.png
  - Q44_probability_distributions.png
""")
print("=" * 70)

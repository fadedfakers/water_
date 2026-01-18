"""
==========================================================================
Huashu Cup 2026 Problem B - Question 3 Q33
Markov Ranking Transition Model
Using 2016-2025 ranking sequences to estimate transition matrix
and predict future ranking distributions
==========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Q33: Markov Ranking Transition Model")
print("Predicting Future Ranking Distributions")
print("=" * 70)

# ==========================================================================
# 1. Data Loading and Historical Ranking Calculation
# ==========================================================================
print("\n--- 1. Loading Data and Computing Historical Rankings ---")

# Load historical data (2016-2025)
hist_data = pd.read_csv('panel_data_38indicators.csv')

countries = sorted(hist_data['Country'].unique())
hist_years = sorted(hist_data['Year'].unique())
indicator_cols = [col for col in hist_data.columns if col not in ['Country', 'Year']]

n_countries = len(countries)
n_hist_years = len(hist_years)
n_indicators = len(indicator_cols)

print(f"Countries: {n_countries}")
print(f"Historical Years: {min(hist_years)} - {max(hist_years)} ({n_hist_years} years)")
print(f"Indicators: {n_indicators}")

# Negative indicators
negative_indicators = ['全球创新指数排名']

# Indicator types
indicator_types = np.ones(n_indicators)
for i, col in enumerate(indicator_cols):
    for neg in negative_indicators:
        if neg in col:
            indicator_types[i] = -1

# ==========================================================================
# 2. Compute Historical Rankings Using Entropy-TOPSIS
# ==========================================================================
print("\n--- 2. Computing Historical Rankings (Entropy-TOPSIS) ---")

def normalize_minmax(X, indicator_types):
    """Min-Max Normalization"""
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
    """Entropy Weight Method"""
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

def topsis_score(X_norm, weights):
    """TOPSIS scoring"""
    V = X_norm * weights
    V_pos = np.max(V, axis=0)
    V_neg = np.min(V, axis=0)
    D_pos = np.sqrt(np.sum((V - V_pos) ** 2, axis=1))
    D_neg = np.sqrt(np.sum((V - V_neg) ** 2, axis=1))
    scores = D_neg / (D_pos + D_neg + 1e-10)
    return scores

# Compute rankings for each historical year
historical_ranks = np.zeros((n_countries, n_hist_years), dtype=int)
historical_scores = np.zeros((n_countries, n_hist_years))

# Collect all historical data for global weights
all_hist_X = []
for year in hist_years:
    year_data = hist_data[hist_data['Year'] == year].sort_values('Country')
    X = year_data[indicator_cols].values
    all_hist_X.append(X)
all_hist_X = np.vstack(all_hist_X)

# Global normalization and weights
all_X_norm = normalize_minmax(all_hist_X, indicator_types)
global_weights = entropy_weight(all_X_norm)

# Compute yearly rankings
for y_idx, year in enumerate(hist_years):
    year_data = hist_data[hist_data['Year'] == year].sort_values('Country')
    X = year_data[indicator_cols].values
    
    X_norm = normalize_minmax(X, indicator_types)
    scores = topsis_score(X_norm, global_weights)
    
    historical_scores[:, y_idx] = scores
    ranks = np.argsort(np.argsort(-scores)) + 1
    historical_ranks[:, y_idx] = ranks

print("Historical Rankings Computed:")
print(f"{'Country':<12}", end='')
for year in hist_years:
    print(f" {year}", end='')
print()
print("-" * (12 + len(hist_years) * 5))
for c_idx, country in enumerate(countries):
    print(f"{country:<12}", end='')
    for y_idx in range(n_hist_years):
        print(f" {historical_ranks[c_idx, y_idx]:>4}", end='')
    print()

# ==========================================================================
# 3. Build Markov Transition Matrix
# ==========================================================================
print("\n--- 3. Building Markov Transition Matrix ---")

# States are rankings: 1, 2, ..., n_countries
n_states = n_countries

# Count transitions
transition_counts = np.zeros((n_states, n_states))

for c_idx in range(n_countries):
    for y_idx in range(n_hist_years - 1):
        current_rank = historical_ranks[c_idx, y_idx]
        next_rank = historical_ranks[c_idx, y_idx + 1]
        # Convert to 0-based index
        transition_counts[current_rank - 1, next_rank - 1] += 1

# Normalize to get transition probabilities
transition_matrix = np.zeros((n_states, n_states))
for i in range(n_states):
    row_sum = np.sum(transition_counts[i, :])
    if row_sum > 0:
        transition_matrix[i, :] = transition_counts[i, :] / row_sum
    else:
        # If no transitions observed from this state, assume uniform
        transition_matrix[i, :] = 1.0 / n_states

print("\nTransition Matrix P(i→j):")
print("From\\To", end='')
for j in range(1, n_states + 1):
    print(f"  Rank{j}", end='')
print()
print("-" * (8 + n_states * 8))
for i in range(n_states):
    print(f"Rank{i+1:<3}", end='')
    for j in range(n_states):
        print(f"  {transition_matrix[i, j]:>5.2f}", end='')
    print()

# Analyze transition characteristics
print("\nTransition Matrix Characteristics:")
print(f"  Diagonal sum (staying in same rank): {np.trace(transition_matrix):.2f}")
print(f"  Average self-transition probability: {np.trace(transition_matrix) / n_states:.3f}")

# Check if matrix is ergodic (irreducible and aperiodic)
eigenvalues = np.linalg.eigvals(transition_matrix)
print(f"  Largest eigenvalue: {np.max(np.abs(eigenvalues)):.4f}")

# ==========================================================================
# 4. Compute Stationary Distribution
# ==========================================================================
print("\n--- 4. Computing Stationary Distribution ---")

# Method 1: Eigenvalue decomposition
# Find left eigenvector for eigenvalue 1: π * P = π
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
idx = np.argmin(np.abs(eigenvalues - 1))
stationary = np.real(eigenvectors[:, idx])
stationary = stationary / np.sum(stationary)

# Ensure non-negative
stationary = np.abs(stationary)
stationary = stationary / np.sum(stationary)

print("Stationary Distribution (long-term probability of each rank):")
for i in range(n_states):
    print(f"  Rank {i+1}: {stationary[i]:.4f} ({stationary[i]*100:.1f}%)")

# Method 2: Power iteration verification
print("\nVerification via power iteration (1000 steps):")
pi = np.ones(n_states) / n_states
for _ in range(1000):
    pi = pi @ transition_matrix
print(f"  Converged distribution matches: {np.allclose(pi, stationary, atol=0.01)}")

# ==========================================================================
# 5. Predict Future Ranking Distributions (2026-2035)
# ==========================================================================
print("\n--- 5. Predicting Future Ranking Distributions ---")

pred_years = list(range(2026, 2036))
n_pred_years = len(pred_years)

# Initial state: 2025 ranking distribution (deterministic)
# Each country starts from their 2025 rank
initial_ranks_2025 = historical_ranks[:, -1]

# Store predicted distributions for each country
# pred_distributions[country][year] = probability distribution over ranks
pred_distributions = np.zeros((n_countries, n_pred_years, n_states))

# For each country, evolve their rank distribution
for c_idx in range(n_countries):
    # Initial distribution: 100% at their 2025 rank
    current_dist = np.zeros(n_states)
    current_dist[initial_ranks_2025[c_idx] - 1] = 1.0
    
    for y_idx in range(n_pred_years):
        # Evolve one step
        current_dist = current_dist @ transition_matrix
        pred_distributions[c_idx, y_idx, :] = current_dist

# Expected ranks for each country each year
expected_ranks = np.zeros((n_countries, n_pred_years))
most_likely_ranks = np.zeros((n_countries, n_pred_years), dtype=int)

for c_idx in range(n_countries):
    for y_idx in range(n_pred_years):
        dist = pred_distributions[c_idx, y_idx, :]
        # Expected rank (weighted average)
        expected_ranks[c_idx, y_idx] = np.sum(dist * np.arange(1, n_states + 1))
        # Most likely rank
        most_likely_ranks[c_idx, y_idx] = np.argmax(dist) + 1

print("\nExpected Rankings (2026-2035):")
print(f"{'Country':<12}", end='')
for year in pred_years:
    print(f"  {year}", end='')
print()
print("-" * (12 + len(pred_years) * 7))
for c_idx, country in enumerate(countries):
    print(f"{country:<12}", end='')
    for y_idx in range(n_pred_years):
        print(f"  {expected_ranks[c_idx, y_idx]:>5.1f}", end='')
    print()

print("\nMost Likely Rankings (2026-2035):")
print(f"{'Country':<12}", end='')
for year in pred_years:
    print(f" {year}", end='')
print()
print("-" * (12 + len(pred_years) * 5))
for c_idx, country in enumerate(countries):
    print(f"{country:<12}", end='')
    for y_idx in range(n_pred_years):
        print(f" {most_likely_ranks[c_idx, y_idx]:>4}", end='')
    print()

# ==========================================================================
# 6. Rank Change Probability Analysis
# ==========================================================================
print("\n--- 6. Rank Change Probability Analysis ---")

# Probability of improving, staying, or declining for each country
print("\n2035 Rank Probability Distribution:")
print(f"{'Country':<12} {'2025':>6} {'E[2035]':>8} {'Mode':>6} {'P(improve)':>10} {'P(stay)':>8} {'P(decline)':>10}")
print("-" * 72)

for c_idx, country in enumerate(countries):
    rank_2025 = initial_ranks_2025[c_idx]
    dist_2035 = pred_distributions[c_idx, -1, :]
    
    p_improve = np.sum(dist_2035[:rank_2025-1]) if rank_2025 > 1 else 0
    p_stay = dist_2035[rank_2025-1]
    p_decline = np.sum(dist_2035[rank_2025:]) if rank_2025 < n_states else 0
    
    print(f"{country:<12} {rank_2025:>6} {expected_ranks[c_idx, -1]:>8.2f} "
          f"{most_likely_ranks[c_idx, -1]:>6} {p_improve:>10.1%} {p_stay:>8.1%} {p_decline:>10.1%}")

# ==========================================================================
# 7. Overtaking Probability Analysis
# ==========================================================================
print("\n--- 7. Overtaking Probability Analysis ---")

# Calculate probability that country A overtakes country B by 2035
print("\nOvertaking Probability Matrix (P(row overtakes column) by 2035):")
overtaking_prob = np.zeros((n_countries, n_countries))

for c1 in range(n_countries):
    for c2 in range(n_countries):
        if c1 == c2:
            continue
        # P(c1 rank < c2 rank) in 2035
        prob = 0
        dist1 = pred_distributions[c1, -1, :]
        dist2 = pred_distributions[c2, -1, :]
        for r1 in range(n_states):
            for r2 in range(r1 + 1, n_states):
                prob += dist1[r1] * dist2[r2]
        overtaking_prob[c1, c2] = prob

print(f"{'':>12}", end='')
for country in countries:
    print(f" {country[:6]:>7}", end='')
print()
print("-" * (12 + n_countries * 8))
for c1, country1 in enumerate(countries):
    print(f"{country1:<12}", end='')
    for c2 in range(n_countries):
        if c1 == c2:
            print(f"     -  ", end='')
        else:
            print(f" {overtaking_prob[c1, c2]:>6.1%}", end='')
    print()

# Key overtaking insights
print("\nKey Overtaking Insights (P > 30%):")
for c1 in range(n_countries):
    for c2 in range(n_countries):
        if c1 != c2 and overtaking_prob[c1, c2] > 0.30:
            # Check if c1 is currently behind c2
            if initial_ranks_2025[c1] > initial_ranks_2025[c2]:
                print(f"  {countries[c1]} may overtake {countries[c2]}: "
                      f"P = {overtaking_prob[c1, c2]:.1%} "
                      f"(current: {initial_ranks_2025[c1]} vs {initial_ranks_2025[c2]})")

# ==========================================================================
# 8. Convergence Analysis
# ==========================================================================
print("\n--- 8. Convergence Analysis ---")

# How quickly does the distribution converge to stationary?
convergence_rates = np.zeros(n_countries)
convergence_years = np.zeros(n_countries, dtype=int)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
for c_idx in range(n_countries):
    for y_idx in range(n_pred_years):
        dist = pred_distributions[c_idx, y_idx, :]
        # KL divergence from stationary (approximation)
        kl_div = np.sum(np.where(dist > 1e-10, dist * np.log(dist / (stationary + 1e-10)), 0))
        if kl_div < 0.1:
            convergence_years[c_idx] = pred_years[y_idx]
            convergence_rates[c_idx] = y_idx + 1
            break
    else:
        convergence_years[c_idx] = 0  # Not converged within horizon
        convergence_rates[c_idx] = n_pred_years

print("Convergence to Stationary Distribution:")
for c_idx, country in enumerate(countries):
    if convergence_years[c_idx] > 0:
        print(f"  {country}: ~{int(convergence_rates[c_idx])} years (by {convergence_years[c_idx]})")
    else:
        print(f"  {country}: >10 years (not converged)")

# ==========================================================================
# 9. Visualization
# ==========================================================================
print("\n--- 9. Generating Visualizations ---")

# Figure 1: Transition Matrix Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
im = ax.imshow(transition_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, label='Transition Probability')

ax.set_xticks(range(n_states))
ax.set_xticklabels([f'Rank {i+1}' for i in range(n_states)])
ax.set_yticks(range(n_states))
ax.set_yticklabels([f'Rank {i+1}' for i in range(n_states)])
ax.set_xlabel('To Rank', fontsize=12)
ax.set_ylabel('From Rank', fontsize=12)
ax.set_title('Q33: Markov Transition Matrix (2016-2025)', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(n_states):
    for j in range(n_states):
        text = f'{transition_matrix[i, j]:.2f}'
        color = 'white' if transition_matrix[i, j] > 0.5 else 'black'
        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

plt.tight_layout()
plt.savefig('Q33_transition_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: Q33_transition_matrix.png")
plt.close()

# Figure 2: Expected Rank Evolution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.tab10(np.linspace(0, 1, n_countries))

# Historical + Predicted rankings
ax1 = axes[0]
all_years = list(hist_years) + pred_years
for c_idx, country in enumerate(countries):
    hist_ranks = historical_ranks[c_idx, :]
    pred_ranks = expected_ranks[c_idx, :]
    
    ax1.plot(hist_years, hist_ranks, 'o-', color=colors[c_idx], 
             linewidth=2, markersize=5, label=country)
    ax1.plot(pred_years, pred_ranks, 's--', color=colors[c_idx], 
             linewidth=2, markersize=5, alpha=0.7)

ax1.axvline(x=2025.5, color='red', linestyle='--', linewidth=2, label='Forecast Start')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Rank (Expected)', fontsize=12)
ax1.set_title('Historical + Predicted Rank Evolution', fontsize=13, fontweight='bold')
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
ax1.invert_yaxis()
ax1.set_ylim(n_states + 0.5, 0.5)
ax1.grid(True, alpha=0.3)

# Rank distribution uncertainty (2035)
ax2 = axes[1]
x = np.arange(n_countries)
width = 0.08
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
for rank in range(n_states):
    probs = [pred_distributions[c, -1, rank] for c in range(n_countries)]
    ax2.bar(x + rank * width, probs, width, label=f'Rank {rank+1}', 
            color=plt.cm.RdYlGn_r(rank / (n_states - 1)))

ax2.set_xticks(x + width * (n_states - 1) / 2)
ax2.set_xticklabels(countries, rotation=45, ha='right')
ax2.set_xlabel('Country', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('2035 Rank Probability Distribution', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=7, ncol=2)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Q33: Markov Ranking Transition Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q33_rank_evolution.png', dpi=150, bbox_inches='tight')
print("Saved: Q33_rank_evolution.png")
plt.close()

# Figure 3: Rank Distribution Evolution for Top 3 Countries
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
top3_idx = np.argsort(initial_ranks_2025)[:3]

for plot_idx, c_idx in enumerate(top3_idx):
    ax = axes[plot_idx]
    country = countries[c_idx]
    
    # Create distribution heatmap over time
    dist_matrix = pred_distributions[c_idx, :, :].T  # n_states × n_years
    
    im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(n_pred_years))
    ax.set_xticklabels(pred_years, rotation=45)
    ax.set_yticks(range(n_states))
    ax.set_yticklabels([f'{i+1}' for i in range(n_states)])
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Rank', fontsize=11)
    ax.set_title(f'{country} (2025 Rank: {initial_ranks_2025[c_idx]})', 
                 fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Probability')

plt.suptitle('Q33: Rank Distribution Evolution (Top 3 Countries)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q33_distribution_evolution.png', dpi=150, bbox_inches='tight')
print("Saved: Q33_distribution_evolution.png")
plt.close()

# Figure 4: Stationary Distribution vs 2035 Prediction
fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
x = np.arange(n_states)
width = 0.35

# Stationary distribution
ax.bar(x - width/2, stationary, width, label='Stationary Distribution', color='steelblue', alpha=0.8)

# Average 2035 distribution across countries
avg_2035_dist = np.mean([pred_distributions[c, -1, :] for c in range(n_countries)], axis=0)
ax.bar(x + width/2, avg_2035_dist, width, label='Average 2035 Distribution', color='coral', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([f'Rank {i+1}' for i in range(n_states)])
ax.set_xlabel('Rank', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Q33: Stationary vs 2035 Predicted Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Q33_stationary_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: Q33_stationary_comparison.png")
plt.close()

# Figure 5: Overtaking Probability Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# Mask diagonal
mask = np.eye(n_countries, dtype=bool)
overtaking_display = np.ma.array(overtaking_prob, mask=mask)

im = ax.imshow(overtaking_display, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, label='Overtaking Probability')

ax.set_xticks(range(n_countries))
ax.set_xticklabels(countries, rotation=45, ha='right')
ax.set_yticks(range(n_countries))
ax.set_yticklabels(countries)
ax.set_xlabel('Country Being Overtaken', fontsize=12)
ax.set_ylabel('Country Overtaking', fontsize=12)
ax.set_title('Q33: Overtaking Probability Matrix (by 2035)', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(n_countries):
    for j in range(n_countries):
        if i != j:
            text = f'{overtaking_prob[i, j]:.0%}'
            color = 'white' if overtaking_prob[i, j] > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

plt.tight_layout()
plt.savefig('Q33_overtaking_probability.png', dpi=150, bbox_inches='tight')
print("Saved: Q33_overtaking_probability.png")
plt.close()

# ==========================================================================
# 10. Save Results
# ==========================================================================
print("\n--- 10. Saving Results ---")

# Transition matrix
trans_df = pd.DataFrame(transition_matrix, 
                        index=[f'From_Rank{i+1}' for i in range(n_states)],
                        columns=[f'To_Rank{i+1}' for i in range(n_states)])
trans_df.to_csv('Q33_transition_matrix.csv')
print("Saved: Q33_transition_matrix.csv")

# Historical rankings
hist_rank_df = pd.DataFrame(historical_ranks,
                            columns=[f'Rank_{y}' for y in hist_years],
                            index=countries)
hist_rank_df.index.name = 'Country'
hist_rank_df.to_csv('Q33_historical_ranks.csv')
print("Saved: Q33_historical_ranks.csv")

# Expected future rankings
expected_df = pd.DataFrame(expected_ranks,
                           columns=[f'E_Rank_{y}' for y in pred_years],
                           index=countries)
expected_df.index.name = 'Country'
expected_df.to_csv('Q33_expected_ranks.csv')
print("Saved: Q33_expected_ranks.csv")

# Most likely rankings
likely_df = pd.DataFrame(most_likely_ranks,
                         columns=[f'Mode_Rank_{y}' for y in pred_years],
                         index=countries)
likely_df.index.name = 'Country'
likely_df.to_csv('Q33_most_likely_ranks.csv')
print("Saved: Q33_most_likely_ranks.csv")

# Rank probability distributions (2035)
prob_2035_df = pd.DataFrame(pred_distributions[:, -1, :],
                            columns=[f'P_Rank{i+1}' for i in range(n_states)],
                            index=countries)
prob_2035_df.index.name = 'Country'
prob_2035_df.to_csv('Q33_rank_distribution_2035.csv')
print("Saved: Q33_rank_distribution_2035.csv")

# Overtaking probability matrix
overtake_df = pd.DataFrame(overtaking_prob,
                           index=countries,
                           columns=countries)
overtake_df.to_csv('Q33_overtaking_probability.csv')
print("Saved: Q33_overtaking_probability.csv")

# Stationary distribution
stat_df = pd.DataFrame({
    'Rank': [f'Rank {i+1}' for i in range(n_states)],
    'Stationary_Probability': stationary
})
stat_df.to_csv('Q33_stationary_distribution.csv', index=False)
print("Saved: Q33_stationary_distribution.csv")

# Summary statistics
summary_data = []
for c_idx, country in enumerate(countries):
    rank_2025 = initial_ranks_2025[c_idx]
    dist_2035 = pred_distributions[c_idx, -1, :]
    p_improve = np.sum(dist_2035[:rank_2025-1]) if rank_2025 > 1 else 0
    p_stay = dist_2035[rank_2025-1]
    p_decline = np.sum(dist_2035[rank_2025:]) if rank_2025 < n_states else 0
    
    summary_data.append({
        'Country': country,
        'Rank_2025': rank_2025,
        'Expected_Rank_2035': expected_ranks[c_idx, -1],
        'Most_Likely_Rank_2035': most_likely_ranks[c_idx, -1],
        'P_Improve': p_improve,
        'P_Stay': p_stay,
        'P_Decline': p_decline,
        'Rank_Uncertainty_Std': np.sqrt(np.sum(dist_2035 * (np.arange(1, n_states+1) - expected_ranks[c_idx, -1])**2))
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('Q33_summary.csv', index=False)
print("Saved: Q33_summary.csv")

# ==========================================================================
# Summary
# ==========================================================================
print("\n" + "=" * 70)
print("Q33 MARKOV RANKING TRANSITION MODEL COMPLETE")
print("=" * 70)
print("\nMethod: Markov Chain Transition Model")
print(f"Training Period: {min(hist_years)}-{max(hist_years)} ({n_hist_years} years)")
print(f"Prediction Period: {min(pred_years)}-{max(pred_years)} ({n_pred_years} years)")
print(f"\nTransition Matrix Properties:")
print(f"  Average self-transition probability: {np.trace(transition_matrix) / n_states:.1%}")
print(f"  Stationary distribution entropy: {-np.sum(stationary * np.log(stationary + 1e-10)):.3f}")
print("\n2035 Ranking Predictions:")
print(f"{'Country':<12} {'2025':>6} {'E[2035]':>8} {'Mode':>6}")
print("-" * 36)
for c_idx, country in enumerate(countries):
    print(f"{country:<12} {initial_ranks_2025[c_idx]:>6} "
          f"{expected_ranks[c_idx, -1]:>8.2f} {most_likely_ranks[c_idx, -1]:>6}")
print("\nOutput Files:")
print("  - Q33_transition_matrix.csv")
print("  - Q33_historical_ranks.csv")
print("  - Q33_expected_ranks.csv")
print("  - Q33_most_likely_ranks.csv")
print("  - Q33_rank_distribution_2035.csv")
print("  - Q33_overtaking_probability.csv")
print("  - Q33_stationary_distribution.csv")
print("  - Q33_summary.csv")
print("  - Q33_transition_matrix.png")
print("  - Q33_rank_evolution.png")
print("  - Q33_distribution_evolution.png")
print("  - Q33_stationary_comparison.png")
print("  - Q33_overtaking_probability.png")
print("=" * 70)

"""
=============================================================================
Huashu Cup 2026 Problem B - Question 1 Q13
Correlation Network + Partial Correlation Network + Centrality Analysis
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

print("=" * 70)
print("Q13: Network Analysis")
print("=" * 70)

# =============================================================================
# 1. Data Loading
# =============================================================================
print("\n--- 1. Data Loading ---")

df = pd.read_csv('panel_data_38indicators.csv')
indicator_cols = [col for col in df.columns if col not in ['Country', 'Year']]
X = df[indicator_cols].values
n, p = X.shape

# Standardize
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# English short names
short_names = ['Top500', 'GPU_Clust', 'DC_Comp', 'AI_Chips', '5G_Cov',
    'Net_BW', 'Data_Ctrs', 'Net_Pen', 'AI_Rschr', 'Talent_Fl',
    'TopSchol', 'STEM_Grad', 'AI_Paper', 'AI_Labs', 'Gov_Inv',
    'Ent_RD', 'VC_Inv', 'Citation', 'AI_Patent', 'AI_Mkt',
    'AI_Comp', 'AI_Unic', 'LLMs', 'Ind_Mkt', 'Tax_Inc',
    'Subsidy', 'Policy_Ct', 'Sub_Int', 'Reg_FW', 'GDP',
    'GDP_Grth', 'FX_Res', 'Pop', 'Work_Age', 'High_Edu',
    'GII_Rank', 'RD_Dens', 'FDI']

# Dimension colors
dim_colors = {
    'Infrastructure': '#3498db',
    'Talent': '#e67e22',
    'R&D': '#27ae60',
    'Industry': '#e74c3c',
    'Policy': '#9b59b6',
    'National': '#7f8c8d'
}
dim_ranges = [(0, 8), (8, 14), (14, 20), (20, 24), (24, 29), (29, 38)]
dim_names = ['Infrastructure', 'Talent', 'R&D', 'Industry', 'Policy', 'National']

# Assign colors to nodes
node_colors = []
node_dims = []
for i in range(p):
    for dim_name, (start, end) in zip(dim_names, dim_ranges):
        if start <= i < end:
            node_colors.append(dim_colors[dim_name])
            node_dims.append(dim_name)
            break

print(f"Indicators: {p}")

# =============================================================================
# 2. Build Correlation Network
# =============================================================================
print("\n--- 2. Building Correlation Network ---")

df_norm = pd.DataFrame(X_norm, columns=short_names)
R = df_norm.corr().values

threshold = 0.5
A_corr = (np.abs(R) > threshold).astype(int)
np.fill_diagonal(A_corr, 0)

W_corr = R * A_corr

n_edges = np.sum(A_corr) // 2
print(f"Correlation Network: Nodes = {p}, Edges = {n_edges} (threshold = {threshold})")

# =============================================================================
# 3. Compute Partial Correlation Matrix
# =============================================================================
print("\n--- 3. Computing Partial Correlation Network ---")

# Regularized precision matrix
Sigma = np.cov(X_norm.T)
ridge = 0.1
Sigma_reg = Sigma + ridge * np.eye(p)
Precision = np.linalg.inv(Sigma_reg)

# Convert to partial correlation
D = np.diag(Precision)
Partial_R = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        if i != j:
            Partial_R[i, j] = -Precision[i, j] / np.sqrt(D[i] * D[j])

threshold_partial = 0.15
A_partial = (np.abs(Partial_R) > threshold_partial).astype(int)
np.fill_diagonal(A_partial, 0)

W_partial = Partial_R * A_partial

n_edges_partial = np.sum(A_partial) // 2
print(f"Partial Correlation Network: Nodes = {p}, Edges = {n_edges_partial} (threshold = {threshold_partial})")

# =============================================================================
# 4. Network Metrics Calculation
# =============================================================================
print("\n--- 4. Network Metrics Calculation ---")

def compute_betweenness(A):
    """Compute betweenness centrality using BFS"""
    n = len(A)
    betweenness = np.zeros(n)
    
    for s in range(n):
        # BFS from source s
        dist = np.full(n, -1)
        dist[s] = 0
        paths = np.zeros(n)
        paths[s] = 1
        pred = [[] for _ in range(n)]
        
        queue = [s]
        order = []
        
        while queue:
            v = queue.pop(0)
            order.append(v)
            
            neighbors = np.where(A[v] == 1)[0]
            for u in neighbors:
                if dist[u] < 0:
                    dist[u] = dist[v] + 1
                    queue.append(u)
                if dist[u] == dist[v] + 1:
                    paths[u] += paths[v]
                    pred[u].append(v)
        
        # Backward accumulation
        delta = np.zeros(n)
        while order:
            w = order.pop()
            for v in pred[w]:
                if paths[w] > 0:
                    delta[v] += (paths[v] / paths[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    
    return betweenness / 2

def compute_clustering(A):
    """Compute clustering coefficient"""
    n = len(A)
    clustering = np.zeros(n)
    
    for i in range(n):
        neighbors = np.where(A[i] == 1)[0]
        k = len(neighbors)
        
        if k < 2:
            clustering[i] = 0
        else:
            # Count edges among neighbors
            edges = 0
            for ni in neighbors:
                for nj in neighbors:
                    if ni < nj and A[ni, nj] == 1:
                        edges += 1
            max_edges = k * (k - 1) / 2
            clustering[i] = edges / max_edges if max_edges > 0 else 0
    
    return clustering

# Degree Centrality
degree_corr = np.sum(A_corr, axis=1)
degree_partial = np.sum(A_partial, axis=1)

# Betweenness Centrality
betweenness_corr = compute_betweenness(A_corr)
betweenness_partial = compute_betweenness(A_partial)

# Clustering Coefficient
clustering_corr = compute_clustering(A_corr)
clustering_partial = compute_clustering(A_partial)

# Weighted Degree (Strength)
strength_corr = np.sum(np.abs(W_corr), axis=1)
strength_partial = np.sum(np.abs(W_partial), axis=1)

# =============================================================================
# 5. Output Network Metrics Ranking
# =============================================================================
print("\n[Correlation Network] Core Factors (by Degree):")
sorted_idx = np.argsort(degree_corr)[::-1]
for i in range(10):
    idx = sorted_idx[i]
    print(f"  {i+1}. {short_names[idx]}: Degree={degree_corr[idx]}, "
          f"Betweenness={betweenness_corr[idx]:.3f}, Clustering={clustering_corr[idx]:.3f}")

print("\n[Partial Correlation Network] Core Factors (by Degree):")
sorted_idx = np.argsort(degree_partial)[::-1]
for i in range(10):
    idx = sorted_idx[i]
    print(f"  {i+1}. {short_names[idx]}: Degree={degree_partial[idx]}, "
          f"Betweenness={betweenness_partial[idx]:.3f}, Clustering={clustering_partial[idx]:.3f}")

print("\n[Bridge Factors] (High Betweenness Centrality):")
sorted_idx = np.argsort(betweenness_corr)[::-1]
for i in range(5):
    idx = sorted_idx[i]
    print(f"  {i+1}. {short_names[idx]}: Betweenness={betweenness_corr[idx]:.3f}")

# =============================================================================
# 6. Plot Network Graphs
# =============================================================================
print("\n--- 6. Plotting Network Graphs ---")

def spring_layout(A, iterations=100, seed=42):
    """Simple spring layout algorithm"""
    np.random.seed(seed)
    n = len(A)
    pos = np.random.rand(n, 2) - 0.5
    k = 1 / np.sqrt(n)
    
    for _ in range(iterations):
        # Repulsive forces
        disp = np.zeros((n, 2))
        for i in range(n):
            for j in range(n):
                if i != j:
                    delta = pos[i] - pos[j]
                    dist = np.linalg.norm(delta) + 0.01
                    disp[i] += (delta / dist) * (k**2 / dist)
        
        # Attractive forces
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    delta = pos[i] - pos[j]
                    dist = np.linalg.norm(delta) + 0.01
                    force = (delta / dist) * (dist**2 / k)
                    disp[i] -= force * 0.1
                    disp[j] += force * 0.1
        
        # Update positions
        for i in range(n):
            disp_norm = np.linalg.norm(disp[i]) + 0.01
            pos[i] += (disp[i] / disp_norm) * min(disp_norm, 0.1)
        
        # Keep in bounds
        pos = np.clip(pos, -1, 1)
    
    return pos

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Helper function to draw network
def draw_network(ax, A, R_matrix, title, node_sizes, node_colors, short_names, threshold_val):
    pos = spring_layout(A, iterations=200)
    
    # Draw edges
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[i, j] == 1:
                weight = R_matrix[i, j]
                if weight > 0:
                    edge_color = '#e74c3c'  # Red for positive
                else:
                    edge_color = '#3498db'  # Blue for negative
                alpha = min(abs(weight), 0.6)
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                       color=edge_color, alpha=alpha, linewidth=abs(weight)*2)
    
    # Draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], s=node_sizes, c=node_colors, 
              alpha=0.9, edgecolors='black', linewidths=0.5)
    
    # Label high-degree nodes
    median_size = np.median(node_sizes)
    for i in range(len(A)):
        if node_sizes[i] > median_size:
            ax.annotate(short_names[i], (pos[i, 0], pos[i, 1]), 
                       fontsize=7, ha='center', va='bottom')
    
    ax.set_title(f'{title}\n(threshold={threshold_val}, edges={np.sum(A)//2})', 
                 fontsize=12, fontweight='bold')
    ax.axis('off')

# Correlation Network
node_sizes_corr = 100 + degree_corr * 30
draw_network(axes[0], A_corr, R, 'Correlation Network', 
             node_sizes_corr, node_colors, short_names, threshold)

# Partial Correlation Network
node_sizes_partial = 100 + degree_partial * 50
draw_network(axes[1], A_partial, Partial_R, 'Partial Correlation Network',
             node_sizes_partial, node_colors, short_names, threshold_partial)

plt.suptitle('Factor Association Networks\n(Node Size = Degree, Red = Positive, Blue = Negative)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q13_network_graph.png', bbox_inches='tight')
plt.close()
print("Saved: Q13_network_graph.png")

# =============================================================================
# 7. Plot Centrality Bar Charts
# =============================================================================
print("\n--- 7. Plotting Centrality Charts ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

top_n = 15

# Degree - Correlation
ax1 = axes[0, 0]
sorted_idx = np.argsort(degree_corr)[::-1][:top_n]
ax1.barh(range(top_n), degree_corr[sorted_idx], color='#3498db', alpha=0.8)
ax1.set_yticks(range(top_n))
ax1.set_yticklabels([short_names[i] for i in sorted_idx])
ax1.set_xlabel('Degree Centrality')
ax1.set_title('Correlation Network - Degree Top15', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Betweenness - Correlation
ax2 = axes[0, 1]
sorted_idx = np.argsort(betweenness_corr)[::-1][:top_n]
ax2.barh(range(top_n), betweenness_corr[sorted_idx], color='#e74c3c', alpha=0.8)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([short_names[i] for i in sorted_idx])
ax2.set_xlabel('Betweenness Centrality')
ax2.set_title('Correlation Network - Betweenness Top15\n(Bridge Factors)', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Degree - Partial
ax3 = axes[1, 0]
sorted_idx = np.argsort(degree_partial)[::-1][:top_n]
ax3.barh(range(top_n), degree_partial[sorted_idx], color='#27ae60', alpha=0.8)
ax3.set_yticks(range(top_n))
ax3.set_yticklabels([short_names[i] for i in sorted_idx])
ax3.set_xlabel('Degree Centrality')
ax3.set_title('Partial Correlation Network - Degree Top15', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

# Clustering
ax4 = axes[1, 1]
sorted_idx = np.argsort(clustering_corr)[::-1][:top_n]
ax4.barh(range(top_n), clustering_corr[sorted_idx], color='#9b59b6', alpha=0.8)
ax4.set_yticks(range(top_n))
ax4.set_yticklabels([short_names[i] for i in sorted_idx])
ax4.set_xlabel('Clustering Coefficient')
ax4.set_title('Correlation Network - Clustering Top15', fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
ax4.invert_yaxis()

plt.suptitle('Network Centrality Metrics Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q13_network_centrality.png', bbox_inches='tight')
plt.close()
print("Saved: Q13_network_centrality.png")

# =============================================================================
# 8. Inter-Dimension Connection Analysis
# =============================================================================
print("\n--- 8. Inter-Dimension Connection Analysis ---")

dim_connections = np.zeros((6, 6))
for d1, (start1, end1) in enumerate(dim_ranges):
    for d2, (start2, end2) in enumerate(dim_ranges):
        dim_connections[d1, d2] = np.sum(A_corr[start1:end1, start2:end2])

# Zero out diagonal for between-dimension only
dim_connections_between = dim_connections.copy()
np.fill_diagonal(dim_connections_between, 0)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(dim_connections_between, ax=ax, cmap='YlOrRd', 
            annot=True, fmt='.0f', square=True,
            xticklabels=dim_names, yticklabels=dim_names,
            cbar_kws={'shrink': 0.8})
ax.set_title('Inter-Dimension Edge Count', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Q13_dimension_connections.png', bbox_inches='tight')
plt.close()
print("Saved: Q13_dimension_connections.png")

# =============================================================================
# 9. Save Results
# =============================================================================
print("\n--- 9. Saving Results ---")

# Network metrics table
metrics_df = pd.DataFrame({
    'Indicator': short_names,
    'Dimension': node_dims,
    'Degree_Corr': degree_corr,
    'Degree_Partial': degree_partial,
    'Betweenness_Corr': betweenness_corr,
    'Betweenness_Partial': betweenness_partial,
    'Clustering_Corr': clustering_corr,
    'Clustering_Partial': clustering_partial,
    'Strength_Corr': strength_corr,
    'Strength_Partial': strength_partial
})
metrics_df.to_csv('Q13_network_metrics.csv', index=False)

# Adjacency matrix
adj_df = pd.DataFrame(A_corr, columns=short_names, index=short_names)
adj_df.to_csv('Q13_adjacency_matrix.csv')

print("Saved: Q13_network_metrics.csv")
print("Saved: Q13_adjacency_matrix.csv")

print("\n" + "=" * 70)
print("Q13 Analysis Complete")
print("=" * 70)

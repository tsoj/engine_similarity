"""
Script to perform similarity analysis and visualization using pre-computed embeddings.
Usage:
    python visualize_similarity.py embeddings_file.npz --output-graph output.png
The embeddings_file.npz should contain:
    - embeddings: numpy array of embeddings (one row per file)
    - labels: list of file labels
"""

import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster import hierarchy
from matplotlib.collections import LineCollection

def calculate_similarity_rankings(similarity_matrix, labels):
    """Calculate average similarity of each file to all other files and return a ranked list."""
    avg_similarities = []
    for i in range(similarity_matrix.shape[0]):
        other_similarities = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
        avg_similarities.append(np.mean(other_similarities))
    similarity_rankings = [{"label": label, "avg_similarity": avg} for label, avg in zip(labels, avg_similarities)]
    similarity_rankings.sort(key=lambda x: x["avg_similarity"], reverse=True)
    return similarity_rankings

def normalize_array(values):
    """Normalize a numpy array to [0, 1]."""
    mn, mx = values.min(), values.max()
    if mx == mn:
        return np.ones_like(values, dtype=float)
    return (values - mn) / (mx - mn)

def balanced_clustering_score(labels):
    """Compute a clustering score based on the balance of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) <= 1:
        return 0.0
    proportions = counts / counts.sum()
    entropy_val = -np.sum(proportions * np.log(proportions))
    normalized_entropy = entropy_val / np.log(len(unique))
    return normalized_entropy

def draw_gradient_edge(ax, pos, n1, n2, color1, color2, n_points=100, lw=2, alpha=1.0):
    """Draw an edge with a color gradient between two nodes."""
    x1, y1 = pos[n1]
    x2, y2 = pos[n2]
    x = np.linspace(x1, x2, n_points)
    y = np.linspace(y1, y2, n_points)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.linspace(0, 1, n_points - 1)[:, None]
    color1 = np.array(color1)
    color2 = np.array(color2)
    seg_colors = (1 - colors) * color1 + colors * color2
    lc = LineCollection(segments, colors=seg_colors, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)

def analyze_and_visualize_similarity_matrix(similarity_matrix, labels, output_graph):
    """Analyze the similarity matrix to find optimal clustering and create visualizations."""
    if not np.allclose(similarity_matrix, similarity_matrix.T, rtol=1e-5, atol=1e-8):
        print("Warning: Similarity matrix is not symmetric. Symmetrizing...")
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

    # Normalize matrix to [0,1]
    similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())

    results = []
    for n_clusters in tqdm(range(1, min(10, similarity_matrix.shape[0] - 1))):
        for rs in range(1000):
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=rs,
                assign_labels='kmeans'
            ).fit(similarity_matrix)
            if len(set(clustering.labels_)) > 1:
                dist = 1.0 - similarity_matrix
                np.fill_diagonal(dist, 0)
                bal = balanced_clustering_score(clustering.labels_)
                sil = silhouette_score(dist, clustering.labels_, metric='precomputed')
                cal = calinski_harabasz_score(dist, clustering.labels_)
                dav = davies_bouldin_score(dist, clustering.labels_)
                results.append((n_clusters, rs, clustering.labels_, bal, sil, cal, dav))

    if not results:
        raise ValueError("No valid clustering found.")

    n_cluster_list, rs_list, labels_list, bal_scores, sil_scores, cal_scores, dav_scores = zip(*results)
    norm_sil = normalize_array(np.array(sil_scores))
    norm_cal = normalize_array(np.array(cal_scores))
    norm_dav = normalize_array(np.array(dav_scores))
    mean_norm = norm_sil + norm_cal - norm_dav
    best_idx = np.argmax(mean_norm)
    cluster_labels = labels_list[best_idx]
    print("Selected n_clusters:", n_cluster_list[best_idx])
    print("Selected random state:", rs_list[best_idx])
    print("Best mean normalized score:", mean_norm[best_idx])

    # Build graph for visualization.
    G = nx.Graph()
    for i, label in enumerate(labels):
        G.add_node(i, label=label, cluster=cluster_labels[i])
    edge_data = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if similarity_matrix[i, j] > 0:
                G.add_edge(i, j, weight=similarity_matrix[i, j])
                edge_data.append((i, j, similarity_matrix[i, j]))

    # Create figure with heatmap and network graph.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(45, 20))

    # Heatmap: sort rows/cols based on clusters
    idx = []
    for cluster_id in sorted(set(cluster_labels)):
        cluster_indices = [i for i, cl in enumerate(cluster_labels) if cl == cluster_id]
        if len(cluster_indices) > 1:
            submatrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            linkage = hierarchy.linkage(1 - submatrix, method='ward')
            order = hierarchy.leaves_list(linkage)
            sorted_cluster_indices = [cluster_indices[i] for i in order]
            idx.extend(sorted_cluster_indices)
        else:
            idx.extend(cluster_indices)

    sorted_matrix = similarity_matrix[np.ix_(idx, idx)]
    sorted_labels = [labels[i] for i in idx]
    sns.heatmap(sorted_matrix * 100.0, ax=ax1, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=sorted_labels, yticklabels=sorted_labels, annot_kws={"size": 6})
    for label in ax1.get_xticklabels():
        label.set_horizontalalignment('right')
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    ax1.tick_params(axis='y', rotation=0, labelsize=12)
    ax1.set_title("Code Similarity Matrix (Sorted)", fontsize=16)

    # Draw cluster boundaries.
    cluster_sizes = np.bincount(np.array(cluster_labels)[idx])
    boundaries = np.cumsum(cluster_sizes)[:-1]
    for boundary in boundaries:
        ax1.axhline(y=boundary, color='red', linestyle='-', linewidth=4)
        ax1.axvline(x=boundary, color='red', linestyle='-', linewidth=4)


    G_layout = G.copy()
    # Modify edge weights based on cluster membership
    for u, v in G_layout.edges():
        # Check if nodes belong to the same cluster
        if G_layout.nodes[u]['cluster'] == G_layout.nodes[v]['cluster']:
            # Reduce distance (increase attraction) for nodes in same cluster
            # Original weight is between 0-1, use a scaling factor to emphasize cluster relationships
            G_layout[u][v]['weight'] = G_layout[u][v]['weight'] * 2.5  # Amplify intra-cluster edge weights

    # Network graph visualization using spring layout.
    pos = nx.spring_layout(G_layout, weight='weight', k=0.4, iterations=100, seed=42)

    cluster_colors = [
        (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e orange
        (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728 red
        (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd purple
        (0.238, 0.544,  0.789  ),                                          # 3d8bc9 light blue
        (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c green
        (0.7372549019607844,  0.7411764705882353,  0.13333333333333333 ),  # bcbd22 yellow
        (0.4980392156862745,  0.4980392156862745,  0.4980392156862745  ),  # 7f7f7f grey
        (0.09019607843137255, 0.7450980392156863,  0.8117647058823529),    # 17becf türkis
        (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4 blue
        (0.8901960784313725,  0.4666666666666667,  0.7607843137254902  ),  # e377c2 pink
        (0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ),  # 8c564b brown
    ]

    node_colors = [cluster_colors[cl % len(cluster_colors)] for cl in cluster_labels]
    for n1, n2 in G.edges():
        alpha = G[n1][n2]['weight'] ** 5.0
        width = max(0.8, alpha * 15.0)
        color1 = node_colors[n1]
        color2 = node_colors[n2]
        draw_gradient_edge(ax2, pos, n1, n2, color2, color1, n_points=200, lw=width, alpha=alpha)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax2, node_size=2000)
    nx.draw_networkx_labels(G, pos, labels={i: label for i, label in enumerate(labels)}, font_size=26, ax=ax2)
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(output_graph, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_graph}")

def main():
    parser = argparse.ArgumentParser(description="Visualize similarity analysis from pre-computed embeddings.")
    parser.add_argument("embeddings_file", type=str, help="Path to the .npz file containing embeddings and labels")
    parser.add_argument("--output-graph", type=str, required=True, help="Output file for the visualization (e.g., output.png)")
    args = parser.parse_args()

    data = np.load(args.embeddings_file, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"].tolist()

    similarity_matrix = cosine_similarity(embeddings)
    rankings = calculate_similarity_rankings(similarity_matrix, labels)
    print("Files ranked by average similarity:")
    for i, info in enumerate(rankings):
        print(f"{i+1}. {info['label']} (Avg similarity: {info['avg_similarity']:.4f})")

    analyze_and_visualize_similarity_matrix(similarity_matrix, labels, args.output_graph)

if __name__ == "__main__":
    main()

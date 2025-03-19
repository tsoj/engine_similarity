#!/usr/bin/env python3
"""
Script to clone GitHub repositories, find files matching search patterns, and analyze their similarity.
Uses a full repository clone approach to ensure all relevant files are found regardless of path.
Includes cluster analysis and graph visualization of similarities.
"""

import os
import shutil
import tempfile
import subprocess
import glob
from pathlib import Path
import argparse
import numpy as np
from transformers.file_utils import get_full_repo_name
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import csv
import re
from typing import List, Dict, Tuple, Optional
import networkx as nx
import networkx.algorithms.community as nx_comm
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
import matplotlib.colors as mcolors
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS, TSNE
from matplotlib.collections import LineCollection
from tqdm import tqdm
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def load_model() -> List[SentenceTransformer]:
    model_names = [
        # # "sentence-transformers/sentence-t5-xl",
        # "sentence-transformers/sentence-t5-xxl",
        # "sentence-transformers/all-distilroberta-v1",
        # "sentence-transformers/all-MiniLM-L12-v2",
        # # # "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        # "Alibaba-NLP/gte-Qwen2-7B-instruct",
        # "Kwaipilot/OASIS-code-embedding-1.5B",
        # "Salesforce/SFR-Embedding-Code-2B_R",
        # "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        # "nomic-ai/CodeRankEmbed",
    ]
    print(f"Loading models:", model_names)
    return [SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={"torch_dtype": "float16"}) for model_name in model_names]

def read_code_file(file_path: str) -> str:
    """Read a code file and return its contents as a string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        return file.read()

def get_embeddings(
    code: str,
    models: List[SentenceTransformer],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> np.ndarray:
    """
    Get embeddings for a piece of code using an ensemble of models that may have different dimensions.

    Args:
        code (str): The code to be embedded
        models (List[SentenceTransformer]): List of sentence transformer models
        chunk_size (int): Size of each code chunk
        overlap (int): Overlap between consecutive chunks
        combination_method (str): Method to combine embeddings from different models.
            Options: "concat" (concatenation), "individual" (return dict of embeddings)
        model_weights (Optional[Dict[int, float]]): Weights for each model (by index) if using weighted methods

    Returns:
        Union[np.ndarray, Dict[int, np.ndarray]]: The final embedding vector or dictionary of embeddings
    """
    # Split code into chunks if it's large
    def chunk_code(text):
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks

    code_chunks = chunk_code(code)

    # Initialize dictionary to store embeddings from all models
    model_embeddings = {}

    # Process each model
    for i, model in enumerate(models):
        # Get embeddings for each chunk using the current model
        chunk_embeddings = model.encode(code_chunks)

        # Average the embeddings if there are multiple chunks
        if len(chunk_embeddings) > 1:
            model_embedding = np.mean(chunk_embeddings, axis=0)
        else:
            model_embedding = chunk_embeddings[0]

        model_embeddings[i] = model_embedding / np.linalg.norm(model_embedding)

    # Flatten dictionary to list in order of indices
    embedding_list = [model_embeddings[i] for i in range(len(models))]
    # Concatenate all embeddings
    combined_embedding = np.concatenate(embedding_list)
    # Return as row vector
    return combined_embedding.reshape(1, -1)

def calculate_similarity_rankings(similarity_matrix: np.ndarray, labels: List[str]) -> List[Dict]:
    """
    Calculate average similarity of each file to all other files and return a ranked list.
    Higher average similarity score means the file is more similar to others.
    """
    # Calculate average similarity for each file (excluding self-similarity)
    avg_similarities = []
    for i in range(similarity_matrix.shape[0]):
        # Get all similarity scores except self-similarity (which is always 1.0)
        other_similarities = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
        avg_similarity = np.mean(other_similarities)
        avg_similarities.append(avg_similarity)

    # Create list of files with their average similarity
    similarity_rankings = []
    for i, name in enumerate(labels):
        similarity_rankings.append({
            'label': name,
            'avg_similarity': avg_similarities[i]
        })

    # Sort by average similarity in descending order (most similar first)
    similarity_rankings.sort(key=lambda x: x['avg_similarity'], reverse=True)

    return similarity_rankings

def normalize_array(values, invert=False):
    """Normalize a numpy array to [0, 1]. Invert if lower values are better."""
    mn, mx = values.min(), values.max()
    if mx == mn:
        return np.ones_like(values, dtype=float)
    return (values - mn) / (mx - mn)

def balanced_clustering_score(labels):
    """
    Compute a clustering score based on the balance of cluster sizes.

    Parameters:
      labels (array-like): Cluster labels for each sample.

    Returns:
      normalized_entropy (float): A score in [0, 1] indicating the balance
                                  of clusters. 1 indicates perfectly balanced,
                                  and lower values indicate imbalance.
    """
    unique, counts = np.unique(labels, return_counts=True)

    # If there's only one cluster, return 0 since balance is meaningless.
    if len(unique) <= 1:
        return 0.0

    proportions = counts / counts.sum()
    entropy_val = -np.sum(proportions * np.log(proportions))
    normalized_entropy = entropy_val / np.log(len(unique))

    return normalized_entropy

def draw_gradient_edge(ax, pos, n1, n2, color1, color2, n_points=100, lw=2, alpha=1.0):
    """
    Draws an edge from node n1 to node n2 with a color gradient from color1 to color2.

    Parameters:
      ax       : The matplotlib Axes to plot on.
      pos      : A dictionary mapping nodes to their (x,y) positions.
      n1, n2   : The two nodes connected by the edge.
      color1   : RGB (or RGBA) tuple for the starting node.
      color2   : RGB (or RGBA) tuple for the ending node.
      n_points : How many points (and thus segments) to use for the gradient.
      lw       : Line width.
    """
    # Retrieve the start and end coordinates.
    x1, y1 = pos[n1]
    x2, y2 = pos[n2]

    # Create n_points points between (x1,y1) and (x2,y2)
    x = np.linspace(x1, x2, n_points)
    y = np.linspace(y1, y2, n_points)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Build an array of interpolated colors.
    # Here, we assume color1 and color2 are tuples like (r, g, b).
    # We interpolate linearly in RGB space:
    colors = np.linspace(0, 1, n_points - 1)[:, None]  # shape (n_points-1, 1)
    color1 = np.array(color1)  # e.g., (1, 0, 0) for red
    color2 = np.array(color2)  # e.g., (0, 0, 1) for blue
    seg_colors = (1 - colors) * color1 + colors * color2

    # Create and add a LineCollection for this edge with the gradient colors.
    lc = LineCollection(segments, colors=seg_colors, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)

def analyze_and_visualize_similarity_matrix(
    similarity_matrix: np.ndarray,
    labels: List[str],
    output_graph: str,
    # n_clusters = 5
):
    """
    Analyze a similarity matrix to find optimal clustering and visualize the results.

    Parameters:
    -----------
    similarity_matrix : np.ndarray
        Square matrix with similarity scores between 0.0 and 1.0
    labels : List[str]
        Labels for each node/item in the similarity matrix
    output_graph : str
        File path to save the visualization
    layout_method : str
        Method for graph layout visualization:
        - "community": Force-directed with community-aware positioning
        - "mds": Multidimensional scaling based on distances
        - "tsne": t-SNE embedding
        - "circular_grouped": Circular layout grouped by clusters
        - "spring": Standard spring layout (default NetworkX)

    Returns:
    --------
    dict
        Dictionary containing cluster assignments and analysis metrics
    """
    # Ensure matrix is symmetric
    if not np.allclose(similarity_matrix, similarity_matrix.T, rtol=1e-5, atol=1e-8):
        print("Warning: Similarity matrix is not symmetric. Symmetrizing...")
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

    # Normalize edge weights to be between 0.0 and 1.0
    min_val = np.min(similarity_matrix)
    max_val = np.max(similarity_matrix)
    similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)

    # Collect valid clustering results: we store tuple (random_state, labels, silhouette, calinski, davies)
    results = []
    for n_clusters in tqdm(range(1, min(10, similarity_matrix.shape[0] - 1))):
        for rs in range(1000):
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=rs,
                assign_labels='kmeans'
            ).fit(similarity_matrix)

            if len(np.unique(clustering.labels_)) > 1:
                # Compute distance matrix from similarity matrix
                dist = 1.0 - similarity_matrix
                np.fill_diagonal(dist, 0)
                bal = balanced_clustering_score(clustering.labels_)
                sil = silhouette_score(dist, clustering.labels_, metric='precomputed')
                cal = calinski_harabasz_score(dist, clustering.labels_)
                dav = davies_bouldin_score(dist, clustering.labels_)  # lower is better
                results.append((n_clusters, rs, clustering.labels_, bal, sil, cal, dav))
                # print(f"Random state: {rs}, balanced: {bal:.4f}, silhouette: {sil:.4f}, calinski: {cal:.4f}, davies: {dav:.4f}")

    if not results:
        raise ValueError("No valid clustering found.")

    # Unpack the collected results
    n_cluster_list, rs_list, labels_list, bal_scores, sil_scores, cal_scores, dav_scores = zip(*results)
    bal_scores = np.array(bal_scores)
    sil_scores = np.array(sil_scores)
    cal_scores = np.array(cal_scores)
    dav_scores = np.array(dav_scores)

    # # Normalize the individual scores. Note: we invert davies because lower is better.
    norm_bal = normalize_array(bal_scores)
    norm_sil = normalize_array(sil_scores)
    norm_cal = normalize_array(cal_scores)
    norm_dav = normalize_array(dav_scores)

    # Compute the mean normalized score for each valid clustering
    mean_norm = 0.0 * norm_bal + norm_sil + norm_cal - norm_dav
    best_idx = np.argmax(mean_norm)

    cluster_labels = labels_list[best_idx]
    print("Selected best n_cluster:", n_cluster_list[best_idx])
    print("Selected best random state:", rs_list[best_idx])
    print("Best mean normalized score:", mean_norm[best_idx])

    assert cluster_labels is not None

    # Create NetworkX graph for visualization
    G = nx.Graph()

    # Add nodes with attributes
    for i, label in enumerate(labels):
        G.add_node(i, label=label, cluster=cluster_labels[i])

    # Add ALL significant edges with weights for layout calculation
    # No thresholding here, to ensure connected graph for layout
    edge_data = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if similarity_matrix[i, j] > 0:  # Include all non-zero edges
                G.add_edge(i, j, weight=similarity_matrix[i, j])
                edge_data.append((i, j, similarity_matrix[i, j]))

    n = similarity_matrix.shape[0]

    # Create figure with two subplots
    figsize = (45, 20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)


    unique_clusters = np.unique(cluster_labels)

    idx = []
    # For each spectral cluster
    for cluster_id in unique_clusters:
        # Get indices of samples in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 1:
            idx.extend(cluster_indices)
            continue


        # Extract the submatrix for this cluster
        submatrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]

        # Convert similarity to distance for hierarchical clustering
        # (Assuming similarity values are between 0 and 1)
        distance_matrix = 1 - submatrix

        # Perform hierarchical clustering on this submatrix
        linkage = hierarchy.linkage(distance_matrix, method='ward')

        # Get the hierarchical clustering order
        hierarchical_order = hierarchy.leaves_list(linkage)

        # Use this order to sort the cluster indices
        sorted_cluster_indices = cluster_indices[hierarchical_order]

        # Add to our final sorted indices
        idx.extend(sorted_cluster_indices)

    sorted_matrix = similarity_matrix[idx][:, idx]
    sorted_labels = [labels[i] for i in idx]

    # Create the heatmap with increased font size for better readability
    sns.heatmap(sorted_matrix*100.0, ax=ax1, annot=True, fmt=".0f", cmap="Blues",
                    xticklabels=sorted_labels, yticklabels=sorted_labels, annot_kws={"size": 6})

    for label in ax1.get_xticklabels():
        label.set_horizontalalignment('right')
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    ax1.tick_params(axis='y', rotation=0, labelsize=12)

    ax1.set_title("Code Similarity Matrix (Sorted)", fontsize=16)

    # Draw cluster boundaries
    cluster_sizes = np.bincount(cluster_labels[idx])
    boundaries = np.cumsum(cluster_sizes)[:-1]

    for boundary in boundaries:
        ax1.axhline(y=boundary, color='red', linestyle='-', linewidth=4)
        ax1.axvline(x=boundary, color='red', linestyle='-', linewidth=4)

    # 2. Network graph visualization
    G_layout = G.copy()

    # # Modify edge weights based on cluster membership
    # for u, v in G_layout.edges():
    #     # Check if nodes belong to the same cluster
    #     if G_layout.nodes[u]['cluster'] == G_layout.nodes[v]['cluster']:
    #         # Reduce distance (increase attraction) for nodes in same cluster
    #         # Original weight is between 0-1, use a scaling factor to emphasize cluster relationships
    #         G_layout[u][v]['weight'] = G_layout[u][v]['weight'] * 2.8  # Amplify intra-cluster edge weights

    # Use spring layout with the modified weights
    # In spring layout, higher weights mean stronger springs (shorter distances)
    pos = nx.spring_layout(
        G_layout,
        weight='weight',  # Use the modified edge weights
        k=0.4,  # Optimal distance between nodes (smaller value creates tighter clusters)
        iterations=100,  # More iterations for better convergence
        seed=42  # For reproducibility
    )

    # Sort edges by weight and keep only top N%
    sorted_edges = sorted(edge_data, key=lambda x: x[2], reverse=True)

    # Create a list of colors for the clusters
    cluster_colors = [
        (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728 red
        (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c green
        (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd purple
        (0.238, 0.544,  0.789  ),  # 3d8bc9 light blue
        (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e orange
        (0.7372549019607844,  0.7411764705882353,  0.13333333333333333 ),  # bcbd22 yellow
        (0.4980392156862745,  0.4980392156862745,  0.4980392156862745  ),  # 7f7f7f grey
        (0.09019607843137255, 0.7450980392156863,  0.8117647058823529),    # 17becf türkis
        (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4 blue
        (0.8901960784313725,  0.4666666666666667,  0.7607843137254902  ),  # e377c2 pink
        (0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ),  # 8c564b brown

    ]

    node_colors = [cluster_colors[cluster_labels[node]] for node in G.nodes()]

    # Instead of drawing edges with draw_networkx_edges,
    # draw each edge with a color gradient:
    for n1, n2 in G.edges():
        alpha = np.power(G[n1][n2]['weight'], 5.0)
        width = max(0.8, alpha * 7.0)
        color1 = node_colors[n1][:3]
        color2 = node_colors[n2][:3]
        draw_gradient_edge(ax2, pos, n1, n2, color2, color1, n_points=200, lw=width, alpha=alpha)

    # Draw the graph using the filtered edges but layout from full graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax2, node_size=2000)

    # Draw labels with smaller font
    nx.draw_networkx_labels(G, pos, labels={i: label for i, label in enumerate(labels)},
                            font_size=26, ax=ax2)

    ax2.axis('off')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Clone GitHub repos, find search files, analyze similarity and visualize clusters.')
    parser.add_argument('input-dir', type=str)
    parser.add_argument('--output-graph', type=str, default=None,
                      help='Output file for the graph visualization')
    args = parser.parse_args()

    # Load the model
    model = load_model()

    # Get embeddings for each file
    print("Generating embeddings...")
    embeddings = []
    label_list = []

    for file_name in tqdm(list(os.listdir(args.input_dir))):
        if not os.path.isfile(os.path.join(args.input_dir, file_name)):
            continue

    #for file_info in tqdm(all_file_infos):
        try:
            code = read_code_file(file_name)
            embedding = get_embeddings(code, model)
            embeddings.append(embedding)

            label = os.path.splitext(os.path.basename(file_name))[0]

            for suffix in [
                "ChessEngine", "-chess-engine", "-bot", "-Chess", "Chess", "-Engine", "Engine"
            ]:
                if label.endswith(suffix) and label != "FabChess":
                    label = label[:-len(suffix)]

            label_list.append(label)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if not embeddings:
        print("Failed to generate embeddings for any files.")
        return

    # Compute similarity matrix
    all_embeddings = np.vstack(embeddings)
    similarity_matrix = cosine_similarity(all_embeddings)


    # Calculate similarity rankings
    similarity_rankings = calculate_similarity_rankings(similarity_matrix, label_list)


    # Print similarity rankings
    print("\nFiles ranked by average similarity to other files (most to least similar):")
    for i, info in enumerate(similarity_rankings):
        print(f"{i+1}. {info['label']} (Avg similarity: {info['avg_similarity']:.4f})")

    # Perform cluster analysis
    analyze_and_visualize_similarity_matrix(similarity_matrix, label_list, args.output_graph)


if __name__ == "__main__":
    main()

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


def parse_github_url(url: str) -> str:
    """Extract the repository name from a GitHub URL."""
    # Handle different GitHub URL formats
    patterns = [
        r'https?://github\.com/([^/]+/[^/]+)',
        r'git@github\.com:([^/]+/[^/]+)\.git',
        r'https?://github\.com/([^/]+/[^/]+)\.git'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If it's just the repo name format (user/repo)
    if '/' in url and url.count('/') == 1 and not url.startswith('http') and not url.startswith('git@'):
        return url

    raise ValueError(f"Could not parse GitHub repository from URL: {url}")

def get_repo_name_without_username(full_repo_name: str) -> str:
    """Extract just the repository name without the username."""
    if '/' in full_repo_name:
        return full_repo_name.split('/')[-1]
    return full_repo_name

def clone_repository(repo_url: str, temp_dir: str, depth: int = 1) -> Optional[str]:
    """
    Clone a GitHub repository to a temporary directory.
    Returns the path to the cloned repository or None if cloning failed.
    """
    repo_name = parse_github_url(repo_url)
    repo_dir = os.path.join(temp_dir, repo_name.replace('/', '_'))

    # Ensure the URL has the correct format
    if not repo_url.startswith(('http://', 'https://', 'git@')):
        repo_url = f"https://github.com/{repo_url}"

    if os.path.exists(repo_dir):
        print(f"Repo {repo_url} already downloaded to {repo_dir}")
    else:
        print(f"Cloning {repo_url} to {repo_dir}...")

        try:
            # Clone with depth=1 to only get the latest commit
            # --filter=blob:none avoids downloading binary blobs until accessed
            subprocess.run(
                ["git", "clone", "--depth", str(depth), "--filter=blob:none", repo_url, repo_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository {repo_url}: {e}")
            print(f"stderr: {e.stderr.decode()}")
            if os.path.exists(repo_dir):
                shutil.rmtree(repo_dir)
            return None

    return repo_dir

def find_search_files(repo_dir: str, repo_name: str) -> List[str]:
    """Find all files matching the search patterns in the repository, applying special rules where necessary."""
    patterns = [
        "search.*", "searches.*", "negamax.*", "mybot.*", "alphabeta.*",
        "pvs.*", "search_manager.*", "search_worker.*", "searcher.*", "chess_search.*"
    ]

    special_rules = {
        "calvin-chess-engine": "Searcher.java",
        "Lynx": "negamax.cs",
        "Prelude": "search.cpp",
        "FabChess": "alphabeta.rs",
        "motors": "caps.rs",
        "autaxx": "tryhard/search.cpp",
        "veritas": "engine.rs",
        "mess": "negamax.go",
        "Leorik": "IterativeSearch.cs",
        "peacekeeper": "main.cpp",
        "PedanticRF": "BasicSearch.cs",
        "hactar": "search/mod.rs",
        "cinder": "search/engine.rs"
    }

    if repo_name in special_rules:
        patterns = [special_rules[repo_name]] + patterns

    search_files = []
    for pattern in patterns:
        found_files = [str(path) for path in Path(repo_dir).rglob(pattern, case_sensitive=False)]
        search_files.extend(f for f in found_files if not f.lower().endswith('.html') and f not in search_files)

    # Filter out headers if a corresponding implementation exists
    preferred_files, header_files = [], {}

    for file in search_files:
        lower_file = file.lower()
        if lower_file.endswith(('.h', '.hpp')):
            base_name = os.path.splitext(os.path.basename(file))[0]
            header_files[base_name] = file
        else:
            preferred_files.append(file)

    for base_name, header in header_files.items():
        has_impl = any(
            file.lower().endswith((f"{base_name}.c", f"{base_name}.cc", f"{base_name}.cpp"))
            for file in preferred_files
        )
        if not has_impl:
            preferred_files.append(header)

    return preferred_files

def load_model() -> List[SentenceTransformer]:
    model_names = [
        # "sentence-transformers/sentence-t5-xl",
        "sentence-transformers/sentence-t5-xxl",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/all-MiniLM-L12-v2",
        # # "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "Kwaipilot/OASIS-code-embedding-1.5B",
        "Salesforce/SFR-Embedding-Code-2B_R",
        "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        "nomic-ai/CodeRankEmbed",
    ]
    print(f"Loading models:", model_names)
    return [SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={"torch_dtype": "float16"}) for model_name in model_names]

def read_code_file(file_path: str) -> str:
    """Read a code file and return its contents as a string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        return file.read()

def get_embeddings_OLD(code: str, model: SentenceTransformer, chunk_size: int = 1000, overlap: int = 200) -> np.ndarray:
    """Get embeddings for a piece of code."""
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

    # Get embeddings for each chunk
    chunk_embeddings = model.encode(code_chunks)

    # Average the embeddings if there are multiple chunks
    if len(chunk_embeddings) > 1:
        return np.mean(chunk_embeddings, axis=0).reshape(1, -1)
    else:
        return chunk_embeddings.reshape(1, -1)


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

        model_embeddings[i] = model_embedding

    # Flatten dictionary to list in order of indices
    embedding_list = [model_embeddings[i] for i in range(len(models))]
    # Concatenate all embeddings
    combined_embedding = np.concatenate(embedding_list)
    # Return as row vector
    return combined_embedding.reshape(1, -1)


def reorder_by_similarity(similarity_matrix: np.ndarray) -> np.ndarray:
    """Reorder the matrix so similar files are close to each other."""
    # Compute distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix

    # Perform hierarchical clustering
    linkage = hierarchy.linkage(distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)], method='average')
    # Create the dendrogram from the linkage matrix Z
    plt.figure(figsize=(10, 5))
    dendrogram(linkage)

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or Cluster ID')
    plt.ylabel('Distance')

    plt.savefig("dendrogram.png", dpi=280, bbox_inches='tight')
    print(f"Saved matrix visualization to {"dendrogram.png"}")
    plt.close()

    # Get the leaf ordering
    ordering = hierarchy.leaves_list(linkage)

    return ordering

def plot_similarity_matrix(similarity_matrix: np.ndarray, file_infos: List[Dict], output_file: str = None):
    """Plot the similarity matrix as a heatmap."""
    # Reorder matrix
    order = reorder_by_similarity(similarity_matrix)

    sorted_matrix = similarity_matrix[order][:, order]
    sorted_infos = [file_infos[i] for i in order]

    # Create labels that include only repo names without usernames
    labels = [get_repo_name_without_username(info['repo']) for info in sorted_infos]

    # Create the plot with dynamic size based on matrix size
    n = len(labels)
    # Scale figure size proportionally to the number of repos
    figsize = (max(12, n * 0.4), max(10, n * 0.3))
    plt.figure(figsize=figsize)

    # Create the heatmap with increased font size for better readability
    ax = sns.heatmap(sorted_matrix, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": max(8, 12 - 0.1*n)})

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=max(8, 12 - 0.1*n))
    plt.yticks(rotation=0, fontsize=max(8, 12 - 0.1*n))

    plt.title("Code Similarity Matrix (Sorted)", fontsize=max(12, 14 - 0.05*n))
    plt.tight_layout()

    # Save if output file is specified
    if output_file:
        # Increase DPI for higher resolution
        plt.savefig(output_file, dpi=280, bbox_inches='tight')
        print(f"Saved matrix visualization to {output_file}")
        plt.close()
    else:
        try:
            plt.show()
        except:
            print("Could not display plot (likely running in a non-GUI environment)")

    return order, sorted_infos

def save_similarity_csv(similarity_matrix: np.ndarray, file_infos: List[Dict], output_file: str):
    """Save the similarity matrix to a CSV file."""
    order = reorder_by_similarity(similarity_matrix)

    sorted_matrix = similarity_matrix[order][:, order]
    sorted_infos = [file_infos[i] for i in order]

    # Create labels with repo names (without username) and file paths
    labels = [f"{get_repo_name_without_username(info['repo'])} - {info['rel_path']}" for info in sorted_infos]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(['File'] + [get_repo_name_without_username(info['repo']) for info in sorted_infos])

        # Write data rows
        for i, label in enumerate(labels):
            writer.writerow([label] + sorted_matrix[i].tolist())

    print(f"Saved similarity matrix to {output_file}")

def calculate_similarity_rankings(similarity_matrix: np.ndarray, file_infos: List[Dict]) -> List[Dict]:
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
    for i, info in enumerate(file_infos):
        similarity_rankings.append({
            'repo': info['repo'],
            'rel_path': info['rel_path'],
            'avg_similarity': avg_similarities[i]
        })

    # Sort by average similarity in descending order (most similar first)
    similarity_rankings.sort(key=lambda x: x['avg_similarity'], reverse=True)

    return similarity_rankings

def analyze_and_visualize_clusters_OLD(similarity_matrix, labels, output_graph, threshold=0.5, n_clusters=5, font_size=10, layout_type="cluster_mds"):
    # 1. Create a NetworkX graph from the similarity matrix
    G = nx.from_numpy_array(similarity_matrix)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Add node labels
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label

    # 2. Perform spectral clustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                           assign_labels='discretize', random_state=42)
    cluster_labels = sc.fit_predict(similarity_matrix)

    # Add cluster information to graph nodes
    for i, cluster_id in enumerate(cluster_labels):
        G.nodes[i]['cluster'] = int(cluster_id)

    # 3. Create layout based on selected method
    if layout_type == "mds":
        # Standard MDS based on distances
        distance_matrix = 1 - similarity_matrix
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        node_positions = mds.fit_transform(distance_matrix)
        pos = {i: (node_positions[i, 0], node_positions[i, 1]) for i in range(len(labels))}

    elif layout_type == "cluster_mds":
        # MDS within clusters, then arrange clusters
        pos = {}
        cluster_centers = []

        # First, position nodes within each cluster using MDS
        for cluster_id in range(n_clusters):
            # Get indices of nodes in this cluster
            cluster_indices = [i for i, c in enumerate(cluster_labels) if c == cluster_id]

            if len(cluster_indices) > 1:
                # Extract submatrix for this cluster
                sub_matrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
                sub_dist = 1 - sub_matrix

                # Apply MDS to position nodes within this cluster
                sub_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                sub_positions = sub_mds.fit_transform(sub_dist)

                # Store positions
                for idx, pos_idx in enumerate(cluster_indices):
                    pos[pos_idx] = (sub_positions[idx, 0], sub_positions[idx, 1])

                # Calculate cluster center
                center_x = np.mean(sub_positions[:, 0])
                center_y = np.mean(sub_positions[:, 1])
                cluster_centers.append((center_x, center_y, cluster_id))

            elif len(cluster_indices) == 1:
                # Single node in cluster, just place it at origin for now
                pos[cluster_indices[0]] = (0, 0)
                cluster_centers.append((0, 0, cluster_id))

        # Arrange clusters in a circle
        if cluster_centers:
            # Calculate positions for cluster centers in a circle
            theta = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
            radius = 5  # Arbitrary radius for the circle
            circle_x = radius * np.cos(theta)
            circle_y = radius * np.sin(theta)

            # Map each original cluster center to its new position
            center_mapping = {}
            for i, (_, _, cluster_id) in enumerate(cluster_centers):
                center_mapping[cluster_id] = (circle_x[i], circle_y[i])

            # Calculate the offset for each cluster
            for node, (x, y) in list(pos.items()):
                cluster_id = cluster_labels[node]
                new_center_x, new_center_y = center_mapping[cluster_id]
                old_center_x, old_center_y, _ = next((c for c in cluster_centers if c[2] == cluster_id), (0, 0, 0))

                # Move the node to the new center plus its relative position in the original layout
                pos[node] = (new_center_x + (x - old_center_x), new_center_y + (y - old_center_y))

    elif layout_type == "spectral":
        # Use NetworkX's spectral layout which emphasizes graph structure
        pos = nx.spectral_layout(G)

    elif layout_type == "kamada_kawai":
        # Kamada-Kawai layout tries to position nodes at distances corresponding to their path lengths
        pos = nx.kamada_kawai_layout(G, weight='weight')

    elif layout_type == "fruchterman_reingold":
        # Force-directed layout with cluster-based initial positions
        # First create initial positions based on clusters
        init_pos = {}
        for i, cluster_id in enumerate(cluster_labels):
            angle = 2 * np.pi * cluster_id / n_clusters
            # Nodes in same cluster start close to each other
            jitter = 0.1 * np.random.rand()
            init_pos[i] = (np.cos(angle) + jitter, np.sin(angle) + jitter)

        # Then apply force-directed algorithm using the initial positions
        pos = nx.spring_layout(G, pos=init_pos, weight='weight', k=1/np.sqrt(len(G)),
                              iterations=50, seed=42)

    else:
        # Default to spring layout
        pos = nx.spring_layout(G, weight='weight', seed=42)

    # 4. Visualize the clustered graph
    plt.figure(figsize=(12, 10))

    # Draw nodes colored by cluster
    for cluster_id in range(n_clusters):
        node_list = [node for node in G.nodes() if G.nodes[node]['cluster'] == cluster_id]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                              node_color=f'C{cluster_id}',
                              node_size=100,
                              alpha=0.8,
                              label=f'Cluster {cluster_id}')

    # Filter edges by threshold to reduce clutter
    edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]

    # Normalize edge weights to [0.1, 1.0] range for alpha values
    if edges:
        edge_weights = np.array([G[u][v]['weight'] for u, v in edges])
        if len(edge_weights) > 1 and np.max(edge_weights) != np.min(edge_weights):
            normalized_weights = (edge_weights - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights))
            alpha_values = 0.1 + 0.9 * normalized_weights
        else:
            alpha_values = [0.5] * len(edges)

        nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.0,
                              alpha=alpha_values,
                              edge_color='gray')

    # Draw labels with increased font size
    nx.draw_networkx_labels(G, pos, labels={i: label for i, label in enumerate(labels)},
                           font_size=font_size, font_color='black')

    plt.title(f'Graph Clusters (similarity threshold > {threshold}, {layout_type} layout)')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()

    # Save if output file is specified
    if output_graph:
        # Increase DPI for higher resolution
        plt.savefig(output_graph, dpi=280, bbox_inches='tight')
        print(f"Saved graph visualization to {output_graph}")

        plt.close()
    else:
        plt.show()

    return G, cluster_labels, pos

def create_similarity_graph_OLD(similarity_matrix: np.ndarray, file_infos: List[Dict],
                           threshold: float = 0.5, cluster_labels: List[int] = None,
                           output_file: str = None):
    """
    Create a graph visualization where:
    - Nodes are files
    - Edges are drawn between files with similarity above the threshold
    - Nodes are colored by cluster if cluster_labels are provided
    """
    import networkx as nx

    # Create a graph
    G = nx.Graph()

    # Add nodes with repository names
    for i, info in enumerate(file_infos):
        repo_name = get_repo_name_without_username(info['repo'])
        cluster = cluster_labels[i] if cluster_labels is not None else 0
        G.add_node(i, name=repo_name, cluster=cluster, file=info['rel_path'])

    # Add edges for similarities above threshold
    for i in range(similarity_matrix.shape[0]):
        for j in range(i+1, similarity_matrix.shape[0]):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                G.add_edge(i, j, weight=similarity, width=similarity * 3)

    # Check if we have any edges
    if len(G.edges) == 0:
        print(f"No edges with similarity above {threshold}. Lowering threshold...")
        # Find the highest similarity and use a slightly lower threshold
        max_sim = 0
        for i in range(similarity_matrix.shape[0]):
            for j in range(i+1, similarity_matrix.shape[0]):
                max_sim = max(max_sim, similarity_matrix[i, j])

        new_threshold = max(0.1, max_sim * 0.8)  # Use 80% of max similarity as threshold
        print(f"New threshold: {new_threshold:.2f}")

        for i in range(similarity_matrix.shape[0]):
            for j in range(i+1, similarity_matrix.shape[0]):
                similarity = similarity_matrix[i, j]
                if similarity >= new_threshold:
                    G.add_edge(i, j, weight=similarity, width=similarity * 3)

    # Prepare the visualization
    plt.figure(figsize=(14, 10))

    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes)), seed=42)

    # Get node colors based on cluster
    if cluster_labels is not None:
        n_clusters = len(set(cluster_labels))
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        node_colors = [cmap(G.nodes[i]['cluster']) for i in G.nodes]
    else:
        node_colors = 'skyblue'

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)

    # Draw the edges with width proportional to similarity
    edges = G.edges(data=True)
    edge_widths = [d['width'] for u, v, d in edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')

    # Add labels
    labels = {i: G.nodes[i]['name'] for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    # Create legend for clusters
    if cluster_labels is not None:
        unique_clusters = sorted(set(cluster_labels))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=cmap(c), markersize=10,
                                    label=f'Cluster {c+1}')
                            for c in unique_clusters]
        plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Repository Similarity Graph")
    plt.axis('off')

    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved graph visualization to {output_file}")

    try:
        plt.show()
    except:
        print("Could not display plot (likely running in a non-GUI environment)")

    # return G

def analyze_and_visualize_similarity_matrix_OLD(
    similarity_matrix: np.ndarray,
    labels: List[str],
    output_graph: str):
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

    Returns:
    --------
    dict
        Dictionary containing cluster assignments and analysis metrics
    """
    # Ensure matrix is symmetric
    if not np.allclose(similarity_matrix, similarity_matrix.T, rtol=1e-5, atol=1e-8):
        print("Warning: Similarity matrix is not symmetric. Symmetrizing...")
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2


    min_val = np.min(similarity_matrix)
    max_val = np.max(similarity_matrix)
    similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)

    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix

    # Find optimal number of clusters using silhouette score
    best_n_clusters = 2  # Default starting point
    best_score = -1

    # Try different cluster counts from 2 to min(10, n-1)
    max_clusters = min(10, len(labels) - 1) if len(labels) > 2 else 2
    silhouette_scores = []

    # for n_clusters in range(2, max_clusters + 1):
    #     clustering = AgglomerativeClustering(
    #         n_clusters=n_clusters,
    #         metric='precomputed',
    #         linkage='average'
    #     ).fit(distance_matrix)

    #     if len(set(clustering.labels_)) > 1:  # Ensure we have at least 2 clusters
    #         score = silhouette_score(distance_matrix, clustering.labels_, metric='precomputed')
    #         silhouette_scores.append(score)

    #         if score > best_score:
    #             best_score = score
    #             best_n_clusters = n_clusters
    best_n_clusters = 8
    # Apply the best clustering
    clustering = SpectralClustering(
        n_clusters=best_n_clusters,
        metric='precomputed',
        linkage='average'
    ).fit(distance_matrix)

    cluster_labels = clustering.labels_

    # Create NetworkX graph for visualization
    G = nx.Graph()

    # Add nodes with attributes
    for i, label in enumerate(labels):
        G.add_node(i, label=label, cluster=cluster_labels[i])

    # Add edges with weights (only if similarity is above threshold)
    threshold = 0.3  # Adjust based on your data
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 1. Heatmap visualization with hierarchical clustering
    # Sort by cluster
    idx = np.argsort(cluster_labels)
    sorted_matrix = similarity_matrix[idx][:, idx]
    sorted_labels = [labels[i] for i in idx]

    # Custom colormap: white to blue
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#1f77b4"])

    # Plot heatmap
    sns.heatmap(sorted_matrix, xticklabels=sorted_labels, yticklabels=sorted_labels,
                cmap=cmap, vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'Similarity'})
    ax1.set_title(f'Similarity Matrix (Sorted by {best_n_clusters} Clusters)')

    # Draw cluster boundaries
    cluster_sizes = np.bincount(cluster_labels[idx])
    boundaries = np.cumsum(cluster_sizes)[:-1]

    for boundary in boundaries:
        ax1.axhline(y=boundary, color='red', linestyle='-', linewidth=1)
        ax1.axvline(x=boundary, color='red', linestyle='-', linewidth=1)

    # 2. Network graph visualization
    pos = nx.spring_layout(G, seed=42)  # For reproducible layout

    # Create a list of colors for the clusters
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, best_n_clusters))
    node_colors = [cluster_colors[cluster_labels[node]] for node in G.nodes()]

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax2, node_size=300, alpha=0.8)

    # Draw edges with width based on weight
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, ax=ax2)

    # Draw labels with smaller font
    nx.draw_networkx_labels(G, pos, labels={i: label for i, label in enumerate(labels)},
                            font_size=8, ax=ax2)

    # Add a legend for clusters
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cluster_colors[i], markersize=10,
                       label=f'Cluster {i+1}')
                       for i in range(best_n_clusters)]
    ax2.legend(handles=legend_elements, loc='upper right')

    ax2.set_title(f'Network Graph (Optimal Clusters: {best_n_clusters}, Silhouette Score: {best_score:.3f})')
    ax2.axis('off')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()

    # Print analysis info
    print(f"Optimal number of clusters: {best_n_clusters}")
    print(f"Silhouette score: {best_score:.3f}")

    # Return analysis results
    return {
        'n_clusters': best_n_clusters,
        'silhouette_score': best_score,
        'cluster_assignments': {labels[i]: cluster_labels[i] for i in range(len(labels))},
        'cluster_sizes': {i: np.sum(cluster_labels == i) for i in range(best_n_clusters)}
    }

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
    top_edges_percent: float = 1.0
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

    # Convert similarity to distance (1 - similarity)
    # distance_matrix = 1 - similarity_matrix

    # Find optimal number of clusters using silhouette score
    n_clusters = 4  # Default starting point
    # random_state = -1

    # Try different cluster counts from 2 to min(10, n-1)
    # max_clusters = min(10, len(labels) - 1) if len(labels) > 2 else 2
    # silhouette_scores = []

    # for n_clusters in range(2, max_clusters + 1):
    #     clustering = AgglomerativeClustering(
    #         n_clusters=n_clusters,
    #         metric='precomputed',
    #         linkage='average'
    #     ).fit(distance_matrix)

    #     if len(set(clustering.labels_)) > 1:  # Ensure we have at least 2 clusters
    #         score = silhouette_score(distance_matrix, clustering.labels_, metric='precomputed')
    #         silhouette_scores.append(score)

    #         if score > best_score:
    #             best_score = score
    #             best_n_clusters = n_clusters

    # Apply the best clustering
    # clustering = AgglomerativeClustering(
    #     n_clusters=best_n_clusters,
    #     metric='precomputed',
    #     linkage='average'
    # ).fit(distance_matrix)
    #
    cluster_labels = None
    best_score = -1
    for random_state in range(0, 100):
        clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=random_state,
                assign_labels='kmeans'
            ).fit(similarity_matrix)


        if len(set(clustering.labels_)) > 1:  # Ensure we have at least 2 clusters
            distance_matrix = 1.0 - similarity_matrix
            np.fill_diagonal(distance_matrix, 0)
            score = silhouette_score(distance_matrix, clustering.labels_, metric='precomputed')
            print(f"Random number: {random_state}, score: {score}")
            if score > best_score:
                best_score = score
                cluster_labels = clustering.labels_

    print("Selected best_score:", best_score)
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
    figsize = (2*max(12, n * 0.4), max(10, n * 0.3))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Heatmap visualization with hierarchical clustering
    # Sort by cluster
    idx = np.argsort(cluster_labels)
    sorted_matrix = similarity_matrix[idx][:, idx]
    sorted_labels = [labels[i] for i in idx]

    # # Custom colormap: white to blue
    # cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#1f77b4"])

    # # Plot heatmap
    # sns.set(font_scale=0.5)
    # sns.heatmap(sorted_matrix, xticklabels=sorted_labels, yticklabels=sorted_labels,
    #             cmap=cmap, vmin=0, vmax=1, , cbar_kws={'label': 'Similarity'})
    # ax1.set_title(f'Similarity Matrix (Sorted by {n_clusters} Clusters)')


    # Create the heatmap with increased font size for better readability
    sns.heatmap(sorted_matrix, ax=ax1, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": max(8, 12 - 0.1*len(sorted_labels))})

    # # Rotate x-axis labels for better readability
    # ax1.xticks(rotation=45, ha='right', fontsize=max(8, 12 - 0.1*n))
    # ax1.yticks(rotation=0, fontsize=max(8, 12 - 0.1*n))
    # Set the x and y tick parameters for ax1
    ax1.tick_params(axis='x', rotation=45, labelsize=max(8, 12 - 0.1 * n))
    ax1.tick_params(axis='y', rotation=0, labelsize=max(8, 12 - 0.1 * n))

    ax1.set_title("Code Similarity Matrix (Sorted)", fontsize=max(12, 14 - 0.05*n))

    # Draw cluster boundaries
    cluster_sizes = np.bincount(cluster_labels[idx])
    boundaries = np.cumsum(cluster_sizes)[:-1]

    for boundary in boundaries:
        ax1.axhline(y=boundary, color='red', linestyle='-', linewidth=1)
        ax1.axvline(x=boundary, color='red', linestyle='-', linewidth=1)

    # 2. Network graph visualization
    G_layout = G.copy()

    # Modify edge weights based on cluster membership
    for u, v in G_layout.edges():
        # Check if nodes belong to the same cluster
        if G_layout.nodes[u]['cluster'] == G_layout.nodes[v]['cluster']:
            # Reduce distance (increase attraction) for nodes in same cluster
            # Original weight is between 0-1, use a scaling factor to emphasize cluster relationships
            G_layout[u][v]['weight'] = G_layout[u][v]['weight'] * 2.5  # Amplify intra-cluster edge weights

    # Use spring layout with the modified weights
    # In spring layout, higher weights mean stronger springs (shorter distances)
    pos = nx.spring_layout(
        G_layout,
        weight='weight',  # Use the modified edge weights
        k=0.35,  # Optimal distance between nodes (smaller value creates tighter clusters)
        iterations=100,  # More iterations for better convergence
        seed=42  # For reproducibility
    )



    G_viz = nx.Graph()

    # Copy all nodes with their attributes
    for node, attr in G.nodes(data=True):
        G_viz.add_node(node, **attr)

    # Sort edges by weight and keep only top N%
    sorted_edges = sorted(edge_data, key=lambda x: x[2], reverse=True)
    num_edges_to_keep = int(len(sorted_edges) * top_edges_percent)

    # Add only the top edges to visualization graph
    for i, j, weight in sorted_edges[:num_edges_to_keep]:
        G_viz.add_edge(i, j, weight=weight)

    # Create a list of colors for the clusters
    cluster_colors = [
        (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728 red
        (0.4980392156862745,  0.4980392156862745,  0.4980392156862745  ),  # 7f7f7f grey
        (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c green
        (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd purple
        (0.09019607843137255, 0.7450980392156863,  0.8117647058823529),    # 17becf türkis
        (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e orange
        (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4 blue
        (0.8901960784313725,  0.4666666666666667,  0.7607843137254902  ),  # e377c2 pink
        (0.7372549019607844,  0.7411764705882353,  0.13333333333333333 ),  # bcbd22 yellow
        (0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ),  # 8c564b brown

    ]
    #plt.cm.Dark2(np.linspace(0, 1, best_n_clusters))
    # print(cluster_colors)
    # assert False
    node_colors = [cluster_colors[cluster_labels[node]] for node in G_viz.nodes()]

    # Instead of drawing edges with draw_networkx_edges,
    # draw each edge with a color gradient:
    for n1, n2 in G_viz.edges():
        alpha = np.power(G_viz[n1][n2]['weight'], 5.2)
        width = max(0.5, alpha * 4.0)
        color1 = node_colors[n1][:3]
        color2 = node_colors[n2][:3]
        draw_gradient_edge(ax2, pos, n1, n2, color2, color1, n_points=200, lw=width, alpha=alpha)

    # Draw the graph using the filtered edges but layout from full graph
    nx.draw_networkx_nodes(G_viz, pos, node_color=node_colors, ax=ax2, node_size=300)

    # Draw labels with smaller font
    nx.draw_networkx_labels(G, pos, labels={i: label for i, label in enumerate(labels)},
                            font_size=10, ax=ax2)

    # Add a legend for clusters
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cluster_colors[i], markersize=10,
                       label=f'Cluster {i+1}')
                       for i in range(n_clusters)]
    ax2.legend(handles=legend_elements, loc='upper right')

    ax2.axis('off')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()



def analyze_and_visualize_similarity_matrix_NEW(
    similarity_matrix: np.ndarray,
    labels: List[str],
    output_graph,
    n_clusters: int = 8,
    edge_percentage: float = 0.5,  # Keep top 30% of edges by default
):

    # g = nx.from_numpy_array(similarity_matrix)

    # # Remove self-loops
    # g.remove_edges_from(nx.selfloop_edges(g))


    # Normalize edge weights to be between 0.0 and 1.0
    min_val = np.min(similarity_matrix)
    max_val = np.max(similarity_matrix)
    similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)


    # 2. Perform spectral clustering
    sc = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed',
        assign_labels='discretize', random_state=42
    )
    cluster_labels = sc.fit_predict(similarity_matrix)


    g = nx.Graph()

    # Add nodes with labels and cluster assignments
    for i, (label, cluster) in enumerate(zip(labels, cluster_labels)):
        g.add_node(i, label=label, cluster=int(cluster))

    # Modify edge weights based on cluster membership
    for u, v in g.edges():
        # Check if nodes belong to the same cluster
        if g.nodes[u]['cluster'] == g.nodes[v]['cluster']:
            # Reduce distance (increase attraction) for nodes in same cluster
            # Original weight is between 0-1, use a scaling factor to emphasize cluster relationships
            g[u][v]['weight'] = g[u][v]['weight'] / 4.0  # Amplify intra-cluster edge weights

    # Extract all edge weights from upper triangle (undirected graph)
    edge_weights = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(i + 1, similarity_matrix.shape[1]):
            edge_weights.append((i, j, similarity_matrix[i, j]))

    # Sort by weight descending
    edge_weights.sort(key=lambda x: x[2], reverse=True)

    # Calculate how many edges to keep
    total_possible_edges = len(edge_weights)
    edges_to_keep = int(total_possible_edges * edge_percentage)

    # Add only the top percentage of edges
    for i, j, weight in edge_weights[:edges_to_keep]:
        g.add_edge(i, j, weight=weight)

    # Add node labels
    for i, label in enumerate(labels):
        g.nodes[i]['label'] = label

    node_to_community = {i: c for i, c in enumerate(cluster_labels)}
    print(node_to_community)

    cluster_cmap = matplotlib.colormaps['tab10']
    node_color = {i: cluster_cmap(cluster_labels[i]) for i in range(len(labels))}


    node_labels = {i: label for i, label in enumerate(labels)}


    edge_alpha = {(i, j): max(0.1, float(g.edges[i, j]['weight'] - edge_weights[-1][2])*0.8) for i, j in g.edges()}
    edge_width = {(i, j): v * 3.0 for (i, j), v in edge_alpha.items()}

    Graph(g,
          node_color=node_color, node_edge_width=0,
          node_labels=node_labels,
          node_label_fontdict={'size': 8, 'weight': 'bold'},
          node_layout='spring', node_layout_kwargs=dict(node_to_community=node_to_community),
          edge_layout='curved', edge_layout_kwargs=dict(k=2000),
          edge_width=edge_width,
          edge_alpha=edge_alpha,
    )

    # plt.show()


    # Save if output file is specified
    if output_graph:
        plt.savefig(output_graph, dpi=400, bbox_inches='tight')
        print(f"Saved graph visualization to {output_graph}")

        plt.close()
    else:
        plt.show()



def main():
    parser = argparse.ArgumentParser(description='Clone GitHub repos, find search files, analyze similarity and visualize clusters.')
    parser.add_argument('--repos', type=str, nargs='+', required=True,
                      help='List of GitHub repository URLs or {owner}/{repo} names')
    parser.add_argument('--temp-dir', type=str, default=None,
                      help='Custom temporary directory to use (default is system temp dir)')
    parser.add_argument('--output-plot', type=str, default=None,
                      help='Output file for the heatmap visualization')
    parser.add_argument('--output-csv', type=str, default=None,
                      help='Output file for the CSV data')
    parser.add_argument('--output-graph', type=str, default=None,
                      help='Output file for the graph visualization')
    parser.add_argument('--keep-clones', action='store_true',
                      help='Keep cloned repositories after analysis')
    parser.add_argument('--chunk-size', type=int, default=1000,
                      help='Size of chunks for processing large files')
    parser.add_argument('--similarity-threshold', type=float, default=0.5,
                      help='Threshold for drawing edges in the similarity graph')
    parser.add_argument('--n-clusters', type=int, default=None,
                      help='Number of clusters for analysis (auto-detected if not specified)')
    args = parser.parse_args()

    # Create temporary directory if not specified
    temp_dir = args.temp_dir
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="github_similarity_")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    print(f"Using temporary directory: {temp_dir}")

    try:
        # Clone repositories
        repo_dirs = []
        repo_urls = []
        for repo_url in args.repos:
            repo_dir = clone_repository(repo_url, temp_dir)
            if repo_dir:
                repo_dirs.append(repo_dir)
                repo_urls.append(repo_url)

        if not repo_dirs:
            print("Failed to clone any repositories.")
            return

        # Find search files in each repository
        all_file_infos = []

        for i, repo_dir in enumerate(repo_dirs):
            repo_base = os.path.basename(repo_dir)
            repo_name = parse_github_url(repo_urls[i])
            search_files = find_search_files(repo_dir, get_repo_name_without_username(repo_name))

            if search_files:
                for i, file_path in enumerate(search_files):
                    rel_path = Path(file_path).relative_to(repo_dir).as_posix()
                    if i == 0:
                        print(f"Using {rel_path} in {repo_name}")
                        all_file_infos.append({
                            'abs_path': file_path,
                            'rel_path': rel_path,
                            'repo': repo_name,
                            'repo_dir': repo_dir
                        })
                    else:
                        print(f"Skipping {rel_path} in {repo_name}")

            else:
                print(f"No matching files found in {repo_name}")

        if not all_file_infos:
            print("No matching files found in any repository.")
            return

        # Load the model
        model = load_model()

        # Get embeddings for each file
        print("Generating embeddings...")
        embeddings = []
        valid_file_infos = []

        for file_info in tqdm(all_file_infos):
            try:
                code = read_code_file(file_info['abs_path'])
                embedding = get_embeddings(code, model, chunk_size=args.chunk_size)
                embeddings.append(embedding)
                valid_file_infos.append(file_info)
            except Exception as e:
                print(f"Error processing {file_info['abs_path']}: {e}")

        if not embeddings:
            print("Failed to generate embeddings for any files.")
            return

        # Compute similarity matrix
        all_embeddings = np.vstack(embeddings)
        similarity_matrix = cosine_similarity(all_embeddings)

        # Calculate similarity rankings
        similarity_rankings = calculate_similarity_rankings(similarity_matrix, valid_file_infos)

        # Print similarity rankings
        print("\nFiles ranked by average similarity to other files (most to least similar):")
        for i, info in enumerate(similarity_rankings):
            repo_name = get_repo_name_without_username(info['repo'])
            print(f"{i+1}. {repo_name} - {info['rel_path']} (Avg similarity: {info['avg_similarity']:.4f})")

        # Visualize and export results
        try:
            # Plot similarity matrix
            order, sorted_infos = plot_similarity_matrix(
                similarity_matrix, valid_file_infos, args.output_plot
            )

        except ImportError as e:
            print(f"Visualization error: {e}")

        # Perform cluster analysis
        labels = [get_repo_name_without_username(info["repo"]) for info in valid_file_infos]
        analyze_and_visualize_similarity_matrix(similarity_matrix, labels, args.output_graph)

        # Save to CSV if requested
        if args.output_csv:
            save_similarity_csv(similarity_matrix, valid_file_infos, args.output_csv)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary directory if not keeping clones
        if not args.keep_clones and temp_dir != args.temp_dir:
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

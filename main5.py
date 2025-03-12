#!/usr/bin/env python3
"""
Script to clone GitHub repositories, find files matching search patterns, and analyze their similarity.
Uses a full repository clone approach to ensure all relevant files are found regardless of path.
"""

import os
import shutil
import tempfile
import subprocess
import glob
from pathlib import Path
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re
from typing import List, Dict, Tuple, Optional

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
        return repo_dir
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository {repo_url}: {e}")
        print(f"stderr: {e.stderr.decode()}")
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        return None

def find_search_files(repo_dir: str) -> List[str]:
    """Find all files matching the search patterns in the repository."""
    # File patterns to search for (case insensitive)
    patterns = [
        "search.*", "searches.*", "negamax.*", "mybot.*", "alphabeta.*",
        "pvs.*", "search_manager.*", "search_worker.*", "searcher.*"
    ]

    search_files = []

    # Use find command for more efficiency on Unix-like systems
    try:
        for pattern in patterns:
            # Use -iname for case-insensitive matching
            result = subprocess.run(
                ["find", repo_dir, "-type", "f", "-iname", pattern, "-not", "-path", "*/\.*"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            found_files = result.stdout.strip().split('\n')
            # Filter out empty strings and HTML files
            for f in found_files:
                if f and not f.lower().endswith('.html'):
                    search_files.append(f)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to Python's glob if find command fails
        print("Using Python's glob for file search (this might be slower)...")
        for pattern in patterns:
            # Use case-insensitive glob in Python
            found_files = glob.glob(os.path.join(repo_dir, "**", pattern), recursive=True, case_sensitive=False)
            for f in found_files:
                if not f.lower().endswith('.html'):
                    search_files.append(f)

    # On Windows, try both approaches
    if not search_files and os.name == 'nt':
        for pattern in patterns:
            found_files = glob.glob(os.path.join(repo_dir, "**", pattern), recursive=True, case_sensitive=False)
            for f in found_files:
                if not f.lower().endswith('.html'):
                    search_files.append(f)

    # Filter out header files if there's a corresponding implementation file
    preferred_files = []
    header_files = []

    for file in search_files:
        lower_file = file.lower()
        if lower_file.endswith(('.h', '.hpp')):
            header_files.append(file)
        else:
            preferred_files.append(file)

    # For each header file, check if there's a corresponding implementation file
    for header in header_files:
        base_name = os.path.splitext(header)[0]
        has_implementation = False

        # Check if there's a corresponding implementation file
        for impl_ext in ['.cpp', '.cc', '.c']:
            potential_impl = f"{base_name}{impl_ext}"
            if potential_impl in preferred_files or potential_impl.lower() in [f.lower() for f in preferred_files]:
                has_implementation = True
                break

        # If there's no implementation file, keep the header
        if not has_implementation:
            preferred_files.append(header)

    return preferred_files

def load_model() -> SentenceTransformer:
    """Load the all-MiniLM-L6-v2 model."""
    print("Loading all-MiniLM-L6-v2 model...")
    return SentenceTransformer('all-MiniLM-L6-v2')

def read_code_file(file_path: str) -> str:
    """Read a code file and return its contents as a string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        return file.read()

def get_embeddings(code: str, model: SentenceTransformer, chunk_size: int = 1000, overlap: int = 200) -> np.ndarray:
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

def reorder_by_similarity(similarity_matrix: np.ndarray) -> np.ndarray:
    """Reorder the matrix so similar files are close to each other."""
    from scipy.cluster import hierarchy

    # Compute distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix

    # Perform hierarchical clustering
    linkage = hierarchy.linkage(distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)], method='average')

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

def main():
    parser = argparse.ArgumentParser(description='Clone GitHub repos, find search and related files, and analyze similarity.')
    parser.add_argument('--repos', type=str, nargs='+', required=True,
                      help='List of GitHub repository URLs or {owner}/{repo} names')
    parser.add_argument('--temp-dir', type=str, default=None,
                      help='Custom temporary directory to use (default is system temp dir)')
    parser.add_argument('--output-plot', type=str, default=None,
                      help='Output file for the heatmap visualization')
    parser.add_argument('--output-csv', type=str, default=None,
                      help='Output file for the CSV data')
    parser.add_argument('--keep-clones', action='store_true',
                      help='Keep cloned repositories after analysis')
    parser.add_argument('--chunk-size', type=int, default=1000,
                      help='Size of chunks for processing large files')
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
            search_files = find_search_files(repo_dir)

            repo_name = parse_github_url(repo_urls[i])

            if search_files:
                for file_path in search_files:
                    rel_path = Path(file_path).relative_to(repo_dir).as_posix()
                    print(f"Found {rel_path} in {repo_name}")
                    all_file_infos.append({
                        'abs_path': file_path,
                        'rel_path': rel_path,
                        'repo': repo_name,
                        'repo_dir': repo_dir
                    })
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

        for file_info in all_file_infos:
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

            # Print the sorted order with repository names
            print("\nFiles ordered by similarity:")
            for i, info in enumerate(sorted_infos):
                repo_name = get_repo_name_without_username(info['repo'])
                print(f"{i+1}. {repo_name} - {info['rel_path']}")

        except ImportError as e:
            print(f"Visualization error: {e}")

            # Still output the ordering
            order = reorder_by_similarity(similarity_matrix)
            print("\nFiles ordered by similarity:")
            for i, idx in enumerate(order):
                info = valid_file_infos[idx]
                repo_name = get_repo_name_without_username(info['repo'])
                print(f"{i+1}. {repo_name} - {info['rel_path']}")

        # Save to CSV if requested
        if args.output_csv:
            save_similarity_csv(similarity_matrix, valid_file_infos, args.output_csv)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up temporary directory if not keeping clones
        if not args.keep_clones and temp_dir != args.temp_dir:
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

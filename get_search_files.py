
import os
import tempfile
from pathlib import Path
import shutil
import argparse
import subprocess
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
        "cinder": "search/engine.rs",
        "4ku": "main.cpp",
        "Fruit-2.1": "search_full.cpp"
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

def read_code_file(file_path: str) -> str:
    """Read a code file and return its contents as a string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        return file.read()

def main():
    parser = argparse.ArgumentParser(description='Clone GitHub repos, find search files, analyze similarity and visualize clusters.')
    parser.add_argument('--repos', type=str, nargs='+', required=True,
                      help='List of GitHub repository URLs or {owner}/{repo} names')
    parser.add_argument('--out-dir', type=str, required=True,
                      help='List of GitHub repository URLs or {owner}/{repo} names')
    parser.add_argument('--temp-dir', type=str, default=None,
                      help='Custom temporary directory to use (default is system temp dir)')
    args = parser.parse_args()

    # Create temporary directory if not specified
    temp_dir = args.temp_dir
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="github_similarity_")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    print(f"Using temporary directory: {temp_dir}")

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


    os.makedirs(args.out_dir, exist_ok=True)
    for info in all_file_infos:
        shutil.copy(
            info["abs_path"],
            args.out_dir + "/" +  get_repo_name_without_username(info["repo"]) + "." + info["abs_path"].split('.')[-1]
        )


if __name__ == "__main__":
    main()

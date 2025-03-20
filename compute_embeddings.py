"""
Script to compute embeddings for code files in an input directory and save the results.
Usage:
    python compute_embeddings.py input_dir output_file.npz
The output file (in .npz format) contains two arrays:
    - embeddings: a numpy array with one row per file (concatenated embeddings)
    - labels: a list of labels (file names modified as needed)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_model():
    model_names = [
        "sentence-transformers/sentence-t5-xxl",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "Kwaipilot/OASIS-code-embedding-1.5B",
        "Salesforce/SFR-Embedding-Code-2B_R",
        "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        "nomic-ai/CodeRankEmbed",
    ]
    print("Loading models:", model_names)
    return [SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={"torch_dtype": "float16"}) for model_name in model_names]

def read_code_file(file_path: str) -> str:
    """Read a code file and return its contents as a string."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        return file.read()

def get_embeddings(code: str, models, chunk_size: int = 1000, overlap: int = 200) -> np.ndarray:
    """Get a combined embedding for the given code using an ensemble of models."""
    # Split code into chunks if needed.
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
    model_embeddings = {}
    for i, model in enumerate(models):
        chunk_embeddings = model.encode(code_chunks)
        # Average over chunks if necessary.
        model_embedding = np.mean(chunk_embeddings, axis=0) if len(chunk_embeddings) > 1 else chunk_embeddings[0]
        # Normalize
        model_embeddings[i] = model_embedding / np.linalg.norm(model_embedding)
    embedding_list = [model_embeddings[i] for i in range(len(models))]
    combined_embedding = np.concatenate(embedding_list)
    return combined_embedding.reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description="Compute embeddings for code files and save to a file.")
    parser.add_argument("input_dir", type=str, help="Directory containing code files")
    parser.add_argument("output_file", type=str, help="Output file (.npz) to save embeddings and labels")
    args = parser.parse_args()

    models = load_model()

    embeddings = []
    labels = []
    file_list = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    print("Computing embeddings for files...")

    for file_name in tqdm(file_list):
        file_path = os.path.join(args.input_dir, file_name)
        try:
            code = read_code_file(file_path)
            embedding = get_embeddings(code, models)
            embeddings.append(embedding)
            label = os.path.splitext(os.path.basename(file_path))[0]
            # Optionally strip common suffixes
            for suffix in ["ChessEngine", "-chess-engine", "-bot", "-Chess", "Chess", "-Engine", "Engine"]:
                if label.endswith(suffix) and label != "FabChess":
                    label = label[:-len(suffix)]
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not embeddings:
        print("No embeddings generated.")
        return

    all_embeddings = np.vstack(embeddings)
    np.savez(args.output_file, embeddings=all_embeddings, labels=labels)
    print(f"Saved embeddings and labels to {args.output_file}")

if __name__ == "__main__":
    main()

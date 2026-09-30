import json
import pathlib

import faiss
import numpy as np

repo_root = pathlib.Path(__file__).resolve().parents[1]
processed_dir = repo_root / 'data' / 'processed'

manifest_path = processed_dir / 'embedding_manifest.json'
image_ids_path = processed_dir / 'image_ids.npy'

n_check = 1000
seed = 0

def load_embeddings(model_key, expected_n, expected_dim):
    embeddings = np.load(processed_dir / f"{model_key}_image_embeddings.npy")
    assert embeddings.shape == (expected_n, expected_dim), f"{model_key}: expected shape {(expected_n, expected_dim)}, got {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"{model_key}: expected float32, got {embeddings.dtype}"

    print(f"{model_key}: loaded embeddings, shape = {embeddings.shape}")
    return embeddings

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    assert index.ntotal == len(embeddings), f"index has {index.ntotal} vectors, expected {len(embeddings)}"

    return index

def check_self_retrieval(index, embeddings, model_key):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(embeddings), size=n_check, replace=False)

    scores, positions = index.search(embeddings[rows], 2)

    found = (positions == rows[:, None]).any(axis=1)
    n_missed = int((~found).sum())

    top_scores = scores[:, 0]
    print(f"{model_key}: self retrieval n checked = {n_check}, n missed = {n_missed}, top score range = [{top_scores.min():.4f}, {top_scores.max():.4f}]")

    if n_missed:
        raise ValueError(f"{model_key}: {n_missed} rows did not retrieve themselves, index or row order is broken")

if __name__ == '__main__':
    with open(manifest_path) as f:
        manifest = json.load(f)

    image_ids = np.load(image_ids_path)

    for model_key, info in manifest.items():
        assert len(image_ids) == info['n_images'], f"{model_key}: image_ids.npy has {len(image_ids)} rows, manifest says {info['n_images']}"

        embeddings = load_embeddings(model_key, info['n_images'], info['dim'])
        index = build_index(embeddings)
        check_self_retrieval(index, embeddings, model_key)

        out_path = processed_dir / f"{model_key}.faiss"
        faiss.write_index(index, str(out_path))
        print(f"{model_key}: index n = {index.ntotal}, saved to {out_path}")
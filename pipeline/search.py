import pathlib

import faiss
import numpy as np
import pandas as pd

repo_root = pathlib.Path(__file__).resolve().parents[1]
processed_dir = repo_root / 'data' / 'processed'

image_ids_path = processed_dir / 'image_ids.npy'

def load_index(model_key):
    index = faiss.read_index(str(processed_dir / f"{model_key}.faiss"))
    image_ids = np.load(image_ids_path)

    assert index.ntotal == len(image_ids), f"{model_key}: index has {index.ntotal} vectors, image_ids.npy has {len(image_ids)} rows"

    print(f"{model_key}: index n = {index.ntotal}")

    return {
        'model_key': model_key,
        'index': index,
        'image_ids': image_ids,
    }

def search(searcher, query_embedding, k=10):
    # query_embedding is shape (1, dim), float32, unit norm, made by encode_text in a separate
    # process with the same model_key as the index (torch and faiss each bundle libomp)
    scores, positions = searcher['index'].search(query_embedding, k)

    result_ids = searcher['image_ids'][positions[0]]

    return pd.DataFrame({
        'rank': np.arange(1, k+1),
        'image_id': result_ids,
        'score': scores[0],
    })

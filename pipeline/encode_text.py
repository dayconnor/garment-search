import json
import pathlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, CLIPModel

repo_root = pathlib.Path(__file__).resolve().parents[1]
processed_dir = repo_root / 'data' / 'processed'

manifest_path = processed_dir / 'embedding_manifest.json'
image_ids_path = processed_dir / 'image_ids.npy'

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')

def load_text_encoder(model_key):
    with open(manifest_path) as f:
        manifest = json.load(f)

    info = manifest[model_key]
    device = pick_device()

    model = CLIPModel.from_pretrained(info['repo_id'], revision=info['revision']).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(info['repo_id'], revision=info['revision'])

    print(f"{model_key}: revision = {info['revision']}, device = {device}")

    return {
        'model_key': model_key,
        'model': model,
        'processor': processor,
        'device': device,
    }

def embed_text(encoder, queries):
    inputs = encoder['processor'](text=queries, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(encoder['device'])
    attention_mask = inputs['attention_mask'].to(encoder['device'])

    with torch.inference_mode():
        outputs = encoder['model'].get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    embeddings = outputs.pooler_output
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    return embeddings.cpu().numpy().astype(np.float32)

if __name__ == '__main__':
    instances_df = pd.read_csv(processed_dir / 'instances.csv')
    image_ids = np.load(image_ids_path)
    smoke_queries = {'glasses': 13, 'hat': 14, 'tie': 16, 'glove': 17}

    for model_key in ['fashion_clip', 'clip_vit_b32']:
        encoder = load_text_encoder(model_key)
        image_embeddings = np.load(processed_dir / f"{model_key}_image_embeddings.npy")
        query_embeddings = embed_text(encoder, list(smoke_queries))

        scores = query_embeddings @ image_embeddings.T

        for i, (query, category_id) in enumerate(smoke_queries.items()):
            top_rows = np.argsort(-scores[i])[:10]
            top_ids = image_ids[top_rows]

            category_image_ids = instances_df.loc[instances_df['category_id'] == category_id, 'image_id'].unique()
            n_hits = int(np.isin(top_ids, category_image_ids).sum())

            print(f"{model_key}: '{query}' -> {n_hits} / 10 top images contain category {category_id}, top score = {scores[i, top_rows[0]]:.4f}")

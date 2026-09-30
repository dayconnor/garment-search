import json
import pathlib

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from PIL import Image
from transformers import AutoProcessor, CLIPModel

repo_root = pathlib.Path(__file__).resolve().parents[1]
raw_dir = repo_root / 'data' / 'raw'
processed_dir = repo_root / 'data' / 'processed'

train_image_dir = raw_dir / 'train'
images_path = processed_dir / 'images.csv'

models = {
    'fashion_clip': 'patrickjohncyh/fashion-clip',
    'clip_vit_b32': 'openai/clip-vit-base-patch32',
}

batch_size = 64

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    
    return torch.device('cpu')

def load_model(repo_id, device):
    revision = HfApi().model_info(repo_id).sha
    model = CLIPModel.from_pretrained(repo_id, revision=revision).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(repo_id, revision=revision)
    
    print(f"{repo_id}: revision = {revision}, device = {device}, dim = {model.config.projection_dim}")
    return model, processor, revision

def load_batch(file_names):
    images = []

    for file_name in file_names:
        with Image.open(train_image_dir / file_name) as image:
            images.append(image.convert('RGB'))

    return images

def embed_images(model, processor, file_names, device):
    batches = []
    
    for start in range(0, len(file_names), batch_size):
        batch_names = file_names[start:start + batch_size]
        images = load_batch(batch_names)
        
        inputs = processor(images=images, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.inference_mode():
            outputs = model.get_image_features(pixel_values=pixel_values)
        
        embeddings = outputs.pooler_output

        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        batches.append(embeddings.cpu().numpy().astype(np.float32))
        
        if (start // batch_size) % 50 == 0:
            print(f" {start + len(batch_names)} / {len(file_names)} images embedded")
            
    return np.vstack(batches)

def check_embeddings(embeddings, n_images, dim):
    assert embeddings.shape == (n_images, dim), f"expected shape {(n_images, dim)}, got {embeddings.shape}"
    assert not np.isnan(embeddings).any(), 'embeddings contain NaN'

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1, atol=1e-3), f"norms not unit length, min = {norms.min():.4f}, max = {norms.max():.4f}"

    print(f"embeddings: shape = {embeddings.shape}, norm range = [{norms.min():.4f}, {norms.max():.4f}]")

if __name__ == '__main__':
    images_df = pd.read_csv(images_path).sort_values('image_id').reset_index(drop=True)
    file_names = images_df['file_name'].tolist()
    image_ids = images_df['image_id'].to_numpy()
    np.save(processed_dir / 'image_ids.npy', image_ids)
    print(f"images: n = {len(file_names)}, row order saved to image_ids.npy")
    
    device = pick_device()
    manifest = {}
    
    for model_key, repo_id in models.items():
        model, processor, revision = load_model(repo_id, device)
        embeddings = embed_images(model, processor, file_names, device)
        check_embeddings(embeddings, len(file_names), model.config.projection_dim)
        
        out_path = processed_dir / f"{model_key}_image_embeddings.npy"
        np.save(out_path, embeddings)
        print(f"{model_key}: shape = {embeddings.shape}, saved to {out_path}")
        
        manifest[model_key] = {
            'repo_id': repo_id,
            'revision': revision,
            'n_images': int(embeddings.shape[0]),
            'dim': int(embeddings.shape[1]),
        }
    
    manifest_path = processed_dir / 'embedding_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"manifest saved to {manifest_path}")
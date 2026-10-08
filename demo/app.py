import atexit
import json
import pathlib
import subprocess
import sys
import threading

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / 'pipeline'))

from search import load_index, search

processed_dir = repo_root / 'data' / 'processed'
train_image_dir = repo_root / 'data' / 'raw' / 'train'

manifest_path = processed_dir / 'embedding_manifest.json'
image_ids_path = processed_dir / 'image_ids.npy'
images_path = processed_dir / 'images.csv'
worker_path = pathlib.Path(__file__).resolve().parent / 'embed_worker.py'

model_labels = {
    'fashion_clip': 'FashionCLIP',
    'clip_vit_b32': 'CLIP ViT-B/32',
}

presets = [
    'cardigan',
    'striped shirt',
    'baggy jeans',
    'oversized blazer',
    'blazer over a white t-shirt',
    'long coat with a scarf',
    'monochrome black outfit',
    'goth outfit',
]

note = """
Text-to-image search over 45,622 Fashionpedia images with exact FAISS cosine search. Each query
runs against both models, FashionCLIP on the left and CLIP ViT-B/32 on the right.

On the pre-registered query set, FashionCLIP reached a capped recall@10 of 0.612 against 0.488 for
CLIP. Both models failed on compositional queries and on fit words like "baggy" and "oversized".
Queries typed here are live and are not part of that evaluation.

[Full write-up](https://connor.day/projects/garment-search)
"""

missing_image = Image.new('RGB', (256, 256), (220, 220, 220))
worker_lock = threading.Lock()

def check_artifacts():
    embedding_paths = [processed_dir / f"{model_key}_image_embeddings.npy" for model_key in model_labels]
    index_paths = [processed_dir / f"{model_key}.faiss" for model_key in model_labels]
    required_paths = [images_path, manifest_path, image_ids_path, *index_paths]

    missing = [path for path in required_paths if not path.exists()]
    if not missing:
        return

    needs_embedding = not all(path.exists() for path in [manifest_path, image_ids_path, *embedding_paths])
    needs_index = needs_embedding or not all(path.exists() for path in index_paths)

    commands = []
    if not images_path.exists():
        commands.append('python pipeline/02_build_dataset.py')
    if needs_embedding:
        commands.append('python pipeline/03_embed_images.py')
    if needs_index:
        commands.append('python pipeline/04_build_index.py')

    print('missing files the demo needs:')
    for path in missing:
        print(f"  {path.relative_to(repo_root)}")

    print('build them from the repo root with:')
    for command in commands:
        print(f"  {command}")

    if needs_embedding:
        print('03_embed_images.py needs the images in data/raw/train, from pipeline/01_download.py')

    sys.exit(1)

def load_manifest():
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"could not parse {manifest_path.relative_to(repo_root)}: {e}")

    for model_key in model_labels:
        info = manifest.get(model_key, {})
        if not info.get('repo_id') or not info.get('revision'):
            sys.exit(f"{manifest_path.relative_to(repo_root)} has no repo_id or revision for {model_key}, rerun python pipeline/03_embed_images.py")

    return manifest

def load_file_names():
    images_df = pd.read_csv(images_path, usecols=['image_id', 'file_name'])
    assert images_df['image_id'].is_unique, 'images.csv has duplicate image_id rows'

    return images_df.set_index('image_id')['file_name']

def start_worker():
    print('loading both text encoders in a separate process, the first load can take a minute')

    # stderr is inherited, so a load failure in the worker prints its own message to this terminal
    worker = subprocess.Popen([sys.executable, str(worker_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    if worker.stdout.readline().strip() != 'ready':
        worker.wait()
        sys.exit(f"text encoder worker exited with code {worker.returncode} before it was ready, see the error above")

    atexit.register(worker.terminate)
    return worker

def embed_query(query):
    with worker_lock:
        worker.stdin.write(json.dumps(query) + '\n')
        worker.stdin.flush()
        line = worker.stdout.readline()

    if not line:
        raise gr.Error('The text encoder worker stopped. Restart the app.')

    reply = json.loads(line)
    if 'error' in reply:
        raise gr.Error(f"Could not embed the query: {reply['error']}")

    return {model_key: np.array([embedding], dtype=np.float32) for model_key, embedding in reply.items()}

def check_query_embeddings(query_embeddings, manifest):
    for model_key, embedding in query_embeddings.items():
        dim = manifest[model_key]['dim']
        assert embedding.shape == (1, dim), f"{model_key}: expected query shape {(1, dim)}, got {embedding.shape}"
        assert searchers[model_key]['index'].d == dim, f"{model_key}: index dim {searchers[model_key]['index'].d}, manifest dim {dim}"

        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1, atol=1e-3), f"{model_key}: query norm {norm:.4f}, expected 1"

def build_gallery(results_df):
    items = []

    for row in results_df.itertuples():
        caption = f"#{row.rank}  cos {row.score:.3f}"
        file_name = file_names.get(row.image_id)

        if file_name is not None and (train_image_dir / file_name).exists():
            items.append((str(train_image_dir / file_name), caption))
        else:
            items.append((missing_image, f"{caption}  image_id {row.image_id}: file missing"))

    return items

def run_search(query, k):
    query = query.strip()
    if not query:
        raise gr.Error('Type a query or pick a preset.')

    query_embeddings = embed_query(query)

    galleries = []
    for model_key in model_labels:
        results_df = search(searchers[model_key], query_embeddings[model_key], k=int(k))
        galleries.append(build_gallery(results_df))

    return tuple(galleries)

def build_demo():
    with gr.Blocks(title='Garment Search') as demo:
        gr.Markdown('# Garment Search')
        gr.Markdown(note)

        with gr.Row():
            query_box = gr.Textbox(label='Query', placeholder='oversized blazer', scale=4)
            k_slider = gr.Slider(minimum=1, maximum=12, value=8, step=1, label='k', scale=1)

        search_button = gr.Button('Search', variant='primary')

        with gr.Row():
            preset_buttons = [gr.Button(preset, size='sm') for preset in presets]

        with gr.Row():
            galleries = [gr.Gallery(label=label, columns=4, object_fit='contain', height='auto') for label in model_labels.values()]

        inputs = [query_box, k_slider]

        search_button.click(run_search, inputs=inputs, outputs=galleries)
        query_box.submit(run_search, inputs=inputs, outputs=galleries)

        # a preset fills the text box with its own label, then runs the same search
        for button in preset_buttons:
            button.click(lambda preset: preset, inputs=button, outputs=query_box).then(run_search, inputs=inputs, outputs=galleries)

    return demo

if __name__ == '__main__':
    check_artifacts()
    manifest = load_manifest()

    file_names = load_file_names()
    searchers = {model_key: load_index(model_key) for model_key in model_labels}

    worker = start_worker()
    check_query_embeddings(embed_query(presets[0]), manifest)
    print('text encoders ready')

    # local only: the images belong to their original sources, so the server is never shared
    build_demo().launch(server_name='127.0.0.1', allowed_paths=[str(train_image_dir)])

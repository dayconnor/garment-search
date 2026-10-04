import pathlib
import re

import numpy as np
import pandas as pd

from encode_text import embed_text, load_text_encoder

repo_root = pathlib.Path(__file__).resolve().parents[1]
processed_dir = repo_root / 'data' / 'processed'

protocol_path = repo_root / 'docs' / 'eval_protocol.md'
queries_path = processed_dir / 'queries.csv'

model_keys = ['fashion_clip', 'clip_vit_b32']
expected_tier_counts = {'A': 5, 'B': 7, 'C': 4, 'D': 4}

def load_queries():
    text = protocol_path.read_text()
    matches = re.findall(r'^\*\*([ABCD]\d)\. `([^`]+)`\*\*', text, flags=re.M)
    queries_df = pd.DataFrame(matches, columns=['query_id', 'query'])
    queries_df['tier'] = queries_df['query_id'].str[0]

    assert queries_df['query_id'].is_unique, 'duplicate query_id in protocol'

    tier_counts = queries_df['tier'].value_counts().to_dict()
    assert tier_counts == expected_tier_counts, f"expected tier counts {expected_tier_counts}, got {tier_counts}"

    print(f"queries: n = {len(queries_df)}, tiers = {tier_counts}")

    return queries_df[['query_id', 'tier', 'query']]

def check_query_embeddings(encoder, queries, embeddings):
    dim = encoder['model'].config.projection_dim

    assert embeddings.shape == (len(queries), dim), f"expected shape {(len(queries), dim)}, got {embeddings.shape}"
    assert not np.isnan(embeddings).any(), 'embeddings contain NaN'

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1, atol=1e-3), f"norms not unit length, min = {norms.min():.4f}, max = {norms.max():.4f}"

    # the first query embedded alone must match its row in the padded batch
    single = embed_text(encoder, queries[:1])
    max_diff = np.abs(single[0] - embeddings[0]).max()
    assert max_diff < 1e-4, f"batched and single embeddings differ by {max_diff:.2e}, padding is leaking into the embedding"

    print(f"{encoder['model_key']}: query embeddings shape = {embeddings.shape}, norm range = [{norms.min():.4f}, {norms.max():.4f}], batch vs single max diff = {max_diff:.2e}")

if __name__ == '__main__':
    queries_df = load_queries()
    queries_df.to_csv(queries_path, index=False)
    print(f"queries saved to {queries_path}")

    # raw query strings, no prompt template (protocol section 5)
    queries = queries_df['query'].tolist()

    for model_key in model_keys:
        encoder = load_text_encoder(model_key)
        embeddings = embed_text(encoder, queries)
        check_query_embeddings(encoder, queries, embeddings)

        out_path = processed_dir / f"{model_key}_query_embeddings.npy"
        np.save(out_path, embeddings)
        print(f"{model_key}: saved to {out_path}")
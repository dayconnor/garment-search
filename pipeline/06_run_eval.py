import json
import pathlib

import numpy as np
import pandas as pd

from search import load_index, search

repo_root = pathlib.Path(__file__).resolve().parents[1]
processed_dir = repo_root / 'data' / 'processed'
results_dir = repo_root / 'results'

manifest_path = processed_dir / 'embedding_manifest.json'
queries_path = processed_dir / 'queries.csv'
instances_path = processed_dir / 'instances.csv'
attributes_path = processed_dir / 'instance_attributes.csv'

model_keys = ['fashion_clip', 'clip_vit_b32']
ks = [1, 5, 10, 20]

predicates = {
    'A1': [{'category': 3}],
    'A2': [{'category': 11}],
    'A3': [{'category': 7}],
    'A4': [{'category': 25}],
    'A5': [{'category': 26}],
    'B1': [{'category': 10}, {'category': 31, 'attributes': [{156}]}],
    'B2': [{'category': 0, 'attributes': [{328}]}],
    'B3': [{'category': 6, 'attributes': [{141}]}],
    'B4': [{'category': 9, 'attributes': [{289}]}],
    'B5': [{'category': 10, 'attributes': [{325}]}],
    'B6': [{'category': 6, 'attributes': [{36}, {297, 300, 298}]}],
    'B7': [{'category': 6, 'attributes': [{36}, {131, 132, 137, 138}]}],
    'C1': [{'category': 6, 'attributes': [{36}, {135}]}, {'category': 23}],
    'C2': [{'category': 4, 'attributes': [{17}]}, {'category': 1}],
    'C3': [{'category': 9, 'attributes': [{152, 154, 155}]}, {'category': 25}],
    'C4': [{'category': 4, 'attributes': [{17}, {138, 137}]}],
}

def match_condition(instances_df, attributes_df, condition):
    matched_ids = instances_df.loc[instances_df['category_id'] == condition['category'], 'instance_id']
    
    for group in condition.get('attributes', []):
        group_ids = attributes_df.loc[attributes_df['attribute_id'].isin(group), 'instance_id']
        matched_ids = matched_ids[matched_ids.isin(group_ids)]

    return set(instances_df.loc[instances_df['instance_id'].isin(matched_ids), 'image_id'])

def relevant_images(instances_df, attributes_df, conditions):
    image_sets = [match_condition(instances_df, attributes_df, condition) for condition in conditions]

    return set.intersection(*image_sets)

def score_ranking(ranked_ids, relevant):
    # np.isin treats a python set as one object, so pass a list
    hits = np.isin(ranked_ids, list(relevant))
    row = {'n_relevant': len(relevant)}

    for k in ks:
        n_hits = int(hits[:k].sum())
        row[f"capped_recall@{k}"] = n_hits / min(len(relevant), k)
        row[f"recall@{k}"] = n_hits / len(relevant)

    row['first_relevant_rank'] = int(np.argmax(hits)) + 1
    row['reciprocal_rank'] = 1 / row['first_relevant_rank']

    return row

if __name__ == '__main__':
    with open(manifest_path) as f:
        manifest = json.load(f)

    queries_df = pd.read_csv(queries_path)
    instances_df = pd.read_csv(instances_path)
    attributes_df = pd.read_csv(attributes_path)

    gradeable_ids = set(queries_df.loc[queries_df['tier'] != 'D', 'query_id'])
    assert set(predicates) == gradeable_ids, f"predicates {sorted(predicates)} do not match Tier A/B/C {sorted(gradeable_ids)}"

    relevant = {}
    for query_id, conditions in predicates.items():
        relevant[query_id] = relevant_images(instances_df, attributes_df, conditions)
        assert len(relevant[query_id]) > 0, f"{query_id}: predicate matches no images"
        print(f"{query_id}: |R| = {len(relevant[query_id])}")

    ranking_dfs = []
    metric_rows = []

    for model_key in model_keys:
        searcher = load_index(model_key)
        query_embeddings = np.load(processed_dir / f"{model_key}_query_embeddings.npy")
        assert len(query_embeddings) == len(queries_df), f"{model_key}: {len(query_embeddings)} embeddings, {len(queries_df)} queries"

        for i, query in queries_df.iterrows():
            # full exact ranking of every indexed image, so first_relevant_rank is always defined
            ranked_df = search(searcher, query_embeddings[i:i+1], k=searcher['index'].ntotal)
            rankings = {'unfiltered': ranked_df}

            if query['tier'] == 'B':
                # protocol section 5: hard pre-filter on the query's category, reported separately.
                # dropping rows from an exact full ranking gives the same order as exact search on the subset
                category_id = predicates[query['query_id']][0]['category']
                category_image_ids = instances_df.loc[instances_df['category_id'] == category_id, 'image_id']

                filtered_df = ranked_df[ranked_df['image_id'].isin(category_image_ids)].copy()
                filtered_df['rank'] = np.arange(1, len(filtered_df)+1)
                rankings['category_filter'] = filtered_df

            for condition, df in rankings.items():
                labels = {'model_key': model_key, 'revision': manifest[model_key]['revision'], 'condition': condition, 'query_id': query['query_id'], 'tier': query['tier']}
                ranking_dfs.append(df.head(max(ks)).assign(**labels))

                if query['query_id'] in relevant:
                    metric_rows.append({**labels, **score_ranking(df['image_id'].to_numpy(), relevant[query['query_id']])})

    results_dir.mkdir(exist_ok=True)
    rankings_df = pd.concat(ranking_dfs, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    rankings_df.to_csv(results_dir / 'rankings.csv', index=False)
    metrics_df.to_csv(results_dir / 'metrics.csv', index=False)
    print(f"rankings: n = {len(rankings_df)}, metrics: n = {len(metrics_df)}, saved to {results_dir}")

    print(metrics_df.groupby(['model_key', 'condition', 'tier'])[['capped_recall@10', 'reciprocal_rank']].mean().round(3))

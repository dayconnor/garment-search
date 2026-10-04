import pathlib

import pandas as pd

repo_root = pathlib.Path(__file__).resolve().parents[1]
processed_dir = repo_root / 'data' / 'processed'

rankings_path = repo_root / 'results' / 'rankings.csv'
queries_path = processed_dir / 'queries.csv'
images_path = processed_dir / 'images.csv'
judgments_path = repo_root / 'docs' / 'relevance_judgments.csv'

seed = 20261004
judged_k = 10

if __name__ == '__main__':
    if judgments_path.exists():
        existing_df = pd.read_csv(judgments_path)
        assert existing_df['score'].isna().all(), f"{judgments_path} already has scores, refusing to overwrite"
    
    rankings_df = pd.read_csv(rankings_path)
    top_df = rankings_df[rankings_df['rank'] <= judged_k]
    
    # pool both models and both conditions: each (query, image) is judged once,
    # and the judge can't tell which model or rank returned it
    pairs_df = top_df[['query_id', 'image_id']].drop_duplicates()
    
    # shuffle with a fixed seed, then regroup by query; the stable sort keeps the shuffled order within each query
    pairs_df = pairs_df.sample(frac=1, random_state=seed).sort_values('query_id', kind='stable')
    pairs_df['judge_order'] = pairs_df.groupby('query_id').cumcount() + 1
    
    queries_df = pd.read_csv(queries_path)
    images_df = pd.read_csv(images_path)
    sheet_df = pairs_df.merge(queries_df[['query_id', 'query']], on='query_id').merge(images_df[['image_id', 'file_name']], on='image_id')
    
    sheet_df['score'] = pd.NA
    sheet_df['reason'] = pd.NA
    sheet_df = sheet_df[['query_id', 'query', 'judge_order', 'image_id', 'file_name', 'score', 'reason']]
    
    assert not sheet_df.duplicated(['query_id', 'image_id']).any(), 'duplicate (query_id, image_id) pair'
    assert sheet_df['query_id'].nunique() == len(queries_df), f"expected {len(queries_df)} queries, got {sheet_df['query_id'].nunique()}"
    
    sheet_df.to_csv(judgments_path, index=False)
    print(f"judging sheet: n = {len(sheet_df)} pairs across {sheet_df['query_id'].nunique()} queries, saved to {judgments_path}")
    print(sheet_df.groupby('query_id').size().to_dict())

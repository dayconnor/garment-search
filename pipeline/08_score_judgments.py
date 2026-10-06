import importlib
import pathlib

import pandas as pd

# 06 starts with a digit, so it can't be imported with an import statement.
# its main block is guarded, so importing it doesn't rerun the eval
run_eval = importlib.import_module('06_run_eval')

repo_root = pathlib.Path(__file__).resolve().parents[1]
processed_dir = repo_root / 'data' / 'processed'
results_dir = repo_root / 'results'

rankings_path = results_dir / 'rankings.csv'
metrics_path = results_dir / 'metrics.csv'
judgments_path = repo_root / 'docs' / 'relevance_judgments.csv'
reason_codes_path = repo_root / 'docs' / 'reason_codes.csv'
instances_path = processed_dir / 'instances.csv'
attributes_path = processed_dir / 'instance_attributes.csv'

judged_k = 10
scores = [0, 1, 2]
group_cols = ['model_key', 'condition', 'query_id', 'tier']

def load_judged_rankings(judgments_df):
    rankings_df = pd.read_csv(rankings_path)
    # rankings.csv calls the cosine similarity 'score', which would collide with the judged score
    top_df = rankings_df[rankings_df['rank'] <= judged_k].rename(columns={'score': 'similarity'})

    # many ranking rows share one judgment, and every ranking row must match exactly one
    judged_df = top_df.merge(judgments_df[['query_id', 'image_id', 'score', 'reason']], on=['query_id', 'image_id'], how='left', validate='m:1')
    assert judged_df['score'].notna().all(), f"{judged_df['score'].isna().sum()} ranked pairs have no judgment"

    # precision is a mean, which only equals hits / 10 if every group has exactly 10 rows
    group_sizes = judged_df.groupby(group_cols).size()
    assert (group_sizes == judged_k).all(), f"groups without {judged_k} rows: {group_sizes[group_sizes != judged_k].to_dict()}"

    print(f"judged rankings: n = {len(judged_df)}, groups = {len(group_sizes)}")

    return judged_df

def score_precision(judged_df):
    judged_df = judged_df.assign(strict_hit=judged_df['score'] == 2, lenient_hit=judged_df['score'] >= 1)

    # protocol section 3.2: strict counts only 2s, lenient counts 2s and 1s, both reported
    return judged_df.groupby(group_cols, as_index=False).agg(**{
        f"strict_precision@{judged_k}": ('strict_hit', 'mean'),
        f"lenient_precision@{judged_k}": ('lenient_hit', 'mean'),
    })

def add_sparsity_gap(precision_df, metrics_df):
    key_cols = ['model_key', 'condition', 'query_id']
    recall_cols = ['n_relevant', f"capped_recall@{judged_k}"]

    # metrics.csv has no Tier D rows, so the left merge leaves their recall empty
    gap_df = precision_df.merge(metrics_df[key_cols + recall_cols], on=key_cols, how='left', validate='1:1')
    assert gap_df.loc[gap_df['tier'] != 'D', f"capped_recall@{judged_k}"].notna().all(), 'Tier A/B/C row missing from metrics.csv'

    # protocol section 7.2 says "manual precision@10" without picking strict or lenient, so both gaps are reported
    for kind in ['strict', 'lenient']:
        gap_df[f"{kind}_gap"] = gap_df[f"{kind}_precision@{judged_k}"] - gap_df[f"capped_recall@{judged_k}"]

    return gap_df

def build_crosstab(judgments_df):
    instances_df = pd.read_csv(instances_path)
    attributes_df = pd.read_csv(attributes_path)

    # the same predicates that produced metrics.csv, imported rather than copied so they can't drift
    relevant = {query_id: run_eval.relevant_images(instances_df, attributes_df, conditions) for query_id, conditions in run_eval.predicates.items()}

    # unique judged pairs, not ranking rows: this is about the predicate, so each (query, image) counts once
    pairs_df = judgments_df[judgments_df['query_id'].isin(relevant)].copy()
    pairs_df['tier'] = pairs_df['query_id'].str[0]
    pairs_df['is_relevant'] = [image_id in relevant[query_id] for query_id, image_id in zip(pairs_df['query_id'], pairs_df['image_id'])]

    crosstab_df = pd.crosstab([pairs_df['tier'], pairs_df['query_id'], pairs_df['is_relevant']], pairs_df['score'])
    crosstab_df = crosstab_df.reindex(columns=scores, fill_value=0).add_prefix('score_').reset_index()

    overall_df = pd.crosstab(pairs_df['is_relevant'], pairs_df['score'], margins=True)
    print(f"crosstab: n = {len(pairs_df)} judged Tier A/B/C pairs")

    return crosstab_df, overall_df

def count_reasons(judgments_df):
    # normalize case and whitespace before grouping
    reason_df = judgments_df.assign(reason=judgments_df['reason'].str.lower().str.strip())

    # post-hoc coding of the free-text reasons, written after scoring and without changing any score.
    # the judgment log stays as written; reasons with no code row are already in the standard wording
    codes_df = pd.read_csv(reason_codes_path, dtype={'score': 'Int64', 'reason': 'string'})
    reason_df = reason_df.merge(codes_df, on=['query_id', 'score', 'reason'], how='left', validate='m:1', indicator=True)

    # a code row that matches nothing means a score or reason changed after coding
    stale = codes_df.merge(reason_df.loc[reason_df['_merge'] == 'both', ['query_id', 'score', 'reason']].drop_duplicates(), how='left', indicator='matched')
    assert (stale['matched'] == 'both').all(), f"stale reason codes: {stale.loc[stale['matched'] != 'both', ['query_id', 'reason']].values.tolist()}"

    reason_df['reason_code'] = reason_df['reason_code'].fillna(reason_df['reason'])
    print(f"reasons: n = {len(reason_df)}, coded from {reason_codes_path.name} = {(reason_df['_merge'] == 'both').sum()}")

    return (reason_df.groupby(['query_id', 'score', 'reason_code']).size().reset_index(name='n')
            .sort_values(['query_id', 'score', 'n'], ascending=[True, False, False]))

if __name__ == '__main__':
    judgments_df = pd.read_csv(judgments_path, dtype={'score': 'Int64', 'reason': 'string'})
    assert judgments_df['score'].notna().all(), f"{judgments_df['score'].isna().sum()} pairs not judged yet"
    assert judgments_df['score'].isin(scores).all(), f"scores outside {scores}"

    judged_df = load_judged_rankings(judgments_df)
    precision_df = add_sparsity_gap(score_precision(judged_df), pd.read_csv(metrics_path))
    crosstab_df, overall_df = build_crosstab(judgments_df)
    reason_counts_df = count_reasons(judgments_df)

    precision_df.to_csv(results_dir / 'precision.csv', index=False)
    crosstab_df.to_csv(results_dir / 'judgment_crosstab.csv', index=False)
    reason_counts_df.to_csv(results_dir / 'reason_counts.csv', index=False)
    print(f"precision: n = {len(precision_df)}, crosstab: n = {len(crosstab_df)}, reasons: n = {len(reason_counts_df)}, saved to {results_dir}")

    precision_cols = [f"strict_precision@{judged_k}", f"lenient_precision@{judged_k}"]
    print('\nmanual precision by tier:')
    print(precision_df.groupby(['model_key', 'condition', 'tier'])[precision_cols].mean().round(3))

    print('\nmanual precision, all 20 queries, unfiltered:')
    print(precision_df[precision_df['condition'] == 'unfiltered'].groupby('model_key')[precision_cols].mean().round(3))

    gap_cols = ['model_key', 'query_id', f"capped_recall@{judged_k}", f"strict_precision@{judged_k}", 'strict_gap', 'lenient_gap']
    print('\nlabel sparsity gap (protocol section 7.2), unfiltered Tier A/B/C:')
    print(precision_df.loc[(precision_df['condition'] == 'unfiltered') & (precision_df['tier'] != 'D'), gap_cols].round(3).to_string(index=False))

    print('\nmetadata relevance vs manual score, Tier A/B/C pairs:')
    print(overall_df)

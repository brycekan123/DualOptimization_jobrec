#!/usr/bin/env python3


import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(description='Evaluate Stage 1A model')
parser.add_argument('--hf_token', type=str, required=True,
                    help='HuggingFace API token for model access')
args = parser.parse_args()

HF_TOKEN = args.hf_token
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = os.path.expanduser("~/llama_cache")

DATASET_DIR = os.path.join("..", "dataset")
QUAL_TEST_CSV = os.path.join(DATASET_DIR, "qual_test.csv")
USER_FILE = os.path.join(DATASET_DIR, "Final_users.csv")
ITEM_FILE = os.path.join(DATASET_DIR, "Final_items.csv")


NEGATIVES_PER_SAMPLE = 49  
MAX_SEQ_LENGTH = 2000  
MAX_NEW_TOKENS = 5  
NUM_RUNS = 3  

# Output directory
OUTPUT_DIR = "baseline_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

user_df = pd.read_csv(USER_FILE)
user_df = user_df[user_df['student_id'] != 'student_id']
user_df = user_df[user_df['student_id'] != 'user_id:token']
user_dict = user_df.set_index('student_id').to_dict('index')

item_df = pd.read_csv(ITEM_FILE)
item_df = item_df[item_df['job_id'] != 'job_id']
item_df = item_df[item_df['job_id'] != 'item_id:token']
item_dict = item_df.set_index('job_id').to_dict('index')

print(f"✓ Loaded {len(user_dict):,} users")
print(f"✓ Loaded {len(item_dict):,} items")

qual_test_df = pd.read_csv(QUAL_TEST_CSV)

print(f"\n✓ Qual test data: {len(qual_test_df):,} samples")
print(f"  Columns: {qual_test_df.columns.tolist()}")

# Basic statistics
n_users = qual_test_df['user'].nunique()
n_jobs = qual_test_df['item'].nunique()
n_positives = (qual_test_df['label_qual'] == 1).sum()
n_negatives = (qual_test_df['label_qual'] == 0).sum()

print(f"\n Dataset statistics:")
print(f"  Unique users: {n_users:,}")
print(f"  Unique jobs: {n_jobs:,}")
print(f"  Positive labels (label_qual=1): {n_positives:,}")
print(f"  Negative labels (label_qual=0): {n_negatives:,}")
print(f"  Pos/Neg ratio: 1:{n_negatives/n_positives:.1f}")


print("\n" + "=" * 100)
print("PREPARING BATCHES")
print("=" * 100)

def prepare_qual_batches(df, max_negatives=49):
    """
    Prepare job-centric batches for qualification data.
    Each batch: 1 job, 1 positive user + N negative users
    Uses 'label_qual' column.
    
    EXACT MATCH to Stage 2 working version.
    """
    batches = []
    job_groups = df.groupby('item')
    
    for job_id, group in job_groups:
        positives = group[group['label_qual'] == 1]
        negatives = group[group['label_qual'] == 0]
        
        if len(positives) > 0:
            if len(negatives) > max_negatives:
                negatives = negatives.sample(n=max_negatives, random_state=42)
            
            for pos_idx in range(len(positives)):
                pos_user = positives.iloc[pos_idx]['user']
                neg_users = negatives['user'].tolist()
                
                batches.append({
                    'job_id': job_id,
                    'positive_user': pos_user,
                    'negative_users': neg_users
                })
    
    return batches

qual_test_batches = prepare_qual_batches(qual_test_df, NEGATIVES_PER_SAMPLE)

print(f"\n✓ Job-centric QUAL batches (1 job, 1 pos user + N neg users each):")
print(f"  Test batches: {len(qual_test_batches):,}")

job_ids = [b['job_id'] for b in qual_test_batches]
print(f"  Unique jobs: {len(set(job_ids))}")
print(f"  Batches per job (avg): {len(qual_test_batches) / len(set(job_ids)):.1f}")


print("\n" + "=" * 100)
print("BATCH VERIFICATION - First 2 Batches")
print("=" * 100)

for batch_idx in range(min(2, len(qual_test_batches))):
    batch = qual_test_batches[batch_idx]
    job_id = batch['job_id']
    positive_user = batch['positive_user']
    negative_users = batch['negative_users']
    
    print(f"\n{'='*80}")
    print(f"BATCH {batch_idx + 1}")
    print(f"{'='*80}")
    print(f"Job ID: {job_id}")
    print(f"Positive User: {positive_user}")
    print(f"Number of Negative Users: {len(negative_users)}")
    print(f"First 5 Negative Users: {negative_users[:5]}")
    
    # Verify this matches the data
    job_data = qual_test_df[qual_test_df['item'] == job_id]
    pos_in_data = job_data[(job_data['user'] == positive_user) & (job_data['label_qual'] == 1)]
    
    print(f"\n✓ Verification against qual_test_df:")
    print(f"  Total applicants for this job: {len(job_data)}")
    print(f"  Positives (label_qual=1): {(job_data['label_qual'] == 1).sum()}")
    print(f"  Negatives (label_qual=0): {(job_data['label_qual'] == 0).sum()}")
    print(f"  Positive user in data: {len(pos_in_data) > 0}")
    
    if len(pos_in_data) > 0:
        print(f"  ✓ Positive user {positive_user} has label_qual=1 for job {job_id}")
    else:
        print(f"  ❌ WARNING: Positive user not found with label_qual=1!")
    
    neg_check = []
    for neg_user in negative_users[:5]:  # Check first 5
        neg_in_data = job_data[(job_data['user'] == neg_user) & (job_data['label_qual'] == 0)]
        neg_check.append(len(neg_in_data) > 0)
    
    if all(neg_check):
        print(f"  ✓ All sampled negative users have label_qual=0 for job {job_id}")
    else:
        print(f"  ❌ WARNING: Some negative users don't have label_qual=0!")
    
    print(f"\n  Batch structure:")
    print(f"    - All users to rank: [positive_user] + negative_users")
    print(f"    - Total users in batch: {1 + len(negative_users)}")
    print(f"    - Expected rank range: 1 to {1 + len(negative_users)}")

print(f"\n{'='*80}")
print("VERIFICATION COMPLETE")
print(f"{'='*80}")


def format_features(features_dict):
    """Convert features dict to readable string."""
    lines = []
    for key, value in features_dict.items():
        nice_key = key.replace('_', ' ').title()
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        lines.append(f"  - {nice_key}: {value_str}")
    return "\n".join(lines)

def build_ranking_prompt(user_id, item_id):
    """Build prompt with explicit numeric output instructions."""
    user_features = user_dict.get(user_id, {})
    user_info = format_features(user_features) if user_features else "Unknown user"
    
    item_features = item_dict.get(item_id, {})
    item_info = format_features(item_features) if item_features else "Unknown item"
    
    prompt = f"""Rate the interaction likelihood between user and item as a single decimal number.

User ID: {user_id}
User Features:
{user_info}

Item ID: {item_id}
Item Features:
{item_info}

Task: Predict the likelihood that this user will interact with this item.
Output format: single number between 0.0 and 1.0
Example valid outputs: 0.23, 0.8, 0.05, 1.0
Do not write anything else.

Rating:"""
    
    return prompt


def extract_score(response):
    """Extract numeric score from text response - STRICT numeric only."""
    response = response.strip()
    
    # Try direct float conversion first
    try:
        score = float(response)
        if 0.0 <= score <= 1.0:
            return score
    except ValueError:
        pass
    
    # Look for any number between 0 and 1
    pattern = r'\b(0?\.\d+|[01]\.?\d*)\b'
    matches = re.findall(pattern, response)
    
    for match in matches:
        try:
            score = float(match)
            if 0.0 <= score <= 1.0:
                return score
        except ValueError:
            continue
    
    # No numeric score found - return None and skip sample
    return None

print("\n" + "=" * 100)
print("LOADING MODEL")
print("=" * 100)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    cache_dir=CACHE_DIR
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("✓ Loaded tokenizer")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("✓ Loaded model (vanilla Llama-3-8B-Instruct)")

def score_user_job(user_id, job_id):
    """Score a single user-job pair by generating text."""
    prompt = build_ranking_prompt(user_id, job_id)
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    ).to(device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    score = extract_score(response)
    return score, response


def calculate_ranking_metrics(ranks):
    """Calculate Recall@k, NDCG@k, Precision@k from list of ranks."""
    metrics = {}
    
    # Recall@k
    for k in [1, 3, 5]:
        recall = sum(1 for r in ranks if r <= k) / len(ranks)
        metrics[f'recall@{k}'] = recall
    
    # Precision@k (same as recall for single positive)
    for k in [1, 3, 5]:
        precision = sum(1 for r in ranks if r <= k) / len(ranks)
        metrics[f'precision@{k}'] = precision
    
    # NDCG@k
    for k in [1, 3, 5]:
        ndcg_scores = []
        for rank in ranks:
            if rank > k:
                ndcg_scores.append(0.0)
            else:
                dcg = 1.0 / np.log2(rank + 1)
                idcg = 1.0 / np.log2(2)
                ndcg_scores.append(dcg / idcg)
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
    
    metrics['avg_rank'] = np.mean(ranks)
    
    return metrics


def evaluate_baseline(batches, desc="Evaluating", print_batches=False):
    """Evaluate baseline model on batches."""
    model.eval()
    
    all_ranks = []
    failed_parses = 0
    successful_scores = []
    all_scores_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=desc, ncols=100)):
            job_id = batch['job_id']
            positive_user = batch['positive_user']
            negative_users = batch['negative_users']
            
            all_users = [positive_user] + negative_users
            
            scores = []
            responses = []
            
            for user_id in all_users:
                score, response = score_user_job(user_id, job_id)
                scores.append(score)
                responses.append(response)
                
                if score is not None:
                    successful_scores.append(score)
            
            valid_scores = [s for s in scores if s is not None]
            
            if len(valid_scores) < len(scores):
                failed_parses += (len(scores) - len(valid_scores))
            
            if scores[0] is not None:
                scores_filled = [s if s is not None else 0.0 for s in scores]
                pos_rank = sum(1 for s in scores_filled[1:] if s > scores_filled[0]) + 1
                all_ranks.append(pos_rank)
                scores_array = np.array([s for s in scores[1:] if s is not None])
                all_scores_data.append({
                    'batch_idx': batch_idx,
                    'job_id': job_id,
                    'positive_user': positive_user,
                    'positive_score': scores[0],
                    'negative_scores_mean': float(scores_array.mean()) if len(scores_array) > 0 else 0.0,
                    'negative_scores_std': float(scores_array.std()) if len(scores_array) > 0 else 0.0,
                    'negative_scores_min': float(scores_array.min()) if len(scores_array) > 0 else 0.0,
                    'negative_scores_max': float(scores_array.max()) if len(scores_array) > 0 else 0.0,
                    'rank': pos_rank,
                    'valid_negatives': len(scores_array)
                })
                
                # Print first 10 batches with detailed info (only if print_batches=True)
                if print_batches and batch_idx < 10:
                    print(f"\n  Batch {batch_idx + 1}:")
                    print(f"    Job: {job_id}")
                    print(f"    Positive user: {positive_user}")
                    print(f"    Positive score: {scores[0]:.4f}")
                    print(f"    Positive response: '{responses[0]}'")
                    valid_neg_scores = [s for s in scores[1:] if s is not None]
                    print(f"    Valid negative scores: {len(valid_neg_scores)}/{len(scores[1:])}")
                    if valid_neg_scores:
                        print(f"    Negative score range: [{min(valid_neg_scores):.4f} - {max(valid_neg_scores):.4f}]")
                        print(f"    Negative scores mean: {np.mean(valid_neg_scores):.4f}")
                    print(f"    Rank: {pos_rank}/{len(all_users)}")
            else:
                # If positive score failed, treat as worst rank
                all_ranks.append(len(all_users))
                failed_parses += 1
                
                if print_batches and batch_idx < 10:
                    print(f"\n  Batch {batch_idx + 1}:")
                    print(f"    Job: {job_id}")
                    print(f"    Positive user: {positive_user}")
                    print(f"    Positive score: FAILED TO PARSE")
                    print(f"    Positive response: '{responses[0]}'")
                    print(f"    Rank: {len(all_users)}/{len(all_users)} (worst)")
            
            # Periodic cleanup
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
    
    # Calculate metrics
    if all_ranks:
        metrics = calculate_ranking_metrics(all_ranks)
    else:
        metrics = {f'recall@{k}': 0.0 for k in [1, 3, 5]}
        metrics.update({f'precision@{k}': 0.0 for k in [1, 3, 5]})
        metrics.update({f'ndcg@{k}': 0.0 for k in [1, 3, 5]})
        metrics['avg_rank'] = 0.0
    
    total_scores_attempted = len(batches) * (1 + NEGATIVES_PER_SAMPLE)
    successful_parses = total_scores_attempted - failed_parses
    parse_rate = successful_parses / total_scores_attempted if total_scores_attempted > 0 else 0
    
    metrics['failed_parses'] = failed_parses
    metrics['successful_parses'] = successful_parses
    metrics['parse_success_rate'] = parse_rate
    metrics['total_batches'] = len(batches)
    
    return metrics, successful_scores, all_scores_data


print("\n" + "=" * 100)
print(f"EVALUATION - RUNNING {NUM_RUNS} TIMES")
print("=" * 100)
print(f"\n Running {NUM_RUNS} times and averaging results...")
print(f" Using {len(qual_test_batches)} test batches")

all_runs_metrics = []
all_runs_scores_data = []

for run_idx in range(NUM_RUNS):
    print(f"\n{'='*100}")
    print(f"RUN {run_idx + 1}/{NUM_RUNS}")
    print('='*100)
    
    if run_idx == 0:
        metrics, scores, scores_data = evaluate_baseline(qual_test_batches, desc=f"Run {run_idx+1}", print_batches=True)
    else:
        metrics, scores, scores_data = evaluate_baseline(qual_test_batches, desc=f"Run {run_idx+1}", print_batches=False)
    
    all_runs_metrics.append(metrics)
    all_runs_scores_data.append(scores_data)

    print(f"\n Run {run_idx + 1} Results:")
    print(f"  Recall@1:  {metrics['recall@1']:.4f}")
    print(f"  Recall@3:  {metrics['recall@3']:.4f}")
    print(f"  Recall@5:  {metrics['recall@5']:.4f}")
    print(f"  NDCG@1:    {metrics['ndcg@1']:.4f}")
    print(f"  NDCG@3:    {metrics['ndcg@3']:.4f}")
    print(f"  NDCG@5:    {metrics['ndcg@5']:.4f}")
    print(f"  Avg Rank:  {metrics['avg_rank']:.2f}")
    print(f"  Parse Rate: {metrics['parse_success_rate']:.1%}")


print("\n" + "=" * 100)
print(f"AVERAGED RESULTS ACROSS {NUM_RUNS} RUNS")
print("=" * 100)

# Calculate mean and std for each metric
metric_keys = ['recall@1', 'recall@3', 'recall@5', 'precision@1', 'precision@3', 
               'precision@5', 'ndcg@1', 'ndcg@3', 'ndcg@5', 'avg_rank', 'parse_success_rate']

averaged_metrics = {}
for key in metric_keys:
    values = [run_metrics[key] for run_metrics in all_runs_metrics]
    averaged_metrics[f'{key}_mean'] = np.mean(values)
    averaged_metrics[f'{key}_std'] = np.std(values)

print(f"\n BASELINE PERFORMANCE (mean ± std):")
print(f"\n  Recall Metrics:")
print(f"    Recall@1:  {averaged_metrics['recall@1_mean']:.4f} ± {averaged_metrics['recall@1_std']:.4f}")
print(f"    Recall@3:  {averaged_metrics['recall@3_mean']:.4f} ± {averaged_metrics['recall@3_std']:.4f}")
print(f"    Recall@5:  {averaged_metrics['recall@5_mean']:.4f} ± {averaged_metrics['recall@5_std']:.4f}")
print(f"\n  NDCG Metrics:")
print(f"    NDCG@1:    {averaged_metrics['ndcg@1_mean']:.4f} ± {averaged_metrics['ndcg@1_std']:.4f}")
print(f"    NDCG@3:    {averaged_metrics['ndcg@3_mean']:.4f} ± {averaged_metrics['ndcg@3_std']:.4f}")
print(f"    NDCG@5:    {averaged_metrics['ndcg@5_mean']:.4f} ± {averaged_metrics['ndcg@5_std']:.4f}")
print(f"\n  Other Metrics:")
print(f"    Avg Rank:  {averaged_metrics['avg_rank_mean']:.2f} ± {averaged_metrics['avg_rank_std']:.2f}")
print(f"    Parse Rate: {averaged_metrics['parse_success_rate_mean']:.1%}")


results = {
    'model': 'baseline_llama3_8b_instruct',
    'dataset': QUAL_TEST_CSV,
    'config': {
        'negatives_per_sample': NEGATIVES_PER_SAMPLE,
        'num_test_batches': len(qual_test_batches),
        'num_runs': NUM_RUNS
    },
    'individual_runs': [
        {
            'run': i + 1,
            'metrics': {k: float(v) for k, v in all_runs_metrics[i].items()}
        }
        for i in range(NUM_RUNS)
    ],
    'averaged_metrics': {k: float(v) for k, v in averaged_metrics.items()}
}

json_file = os.path.join(OUTPUT_DIR, "baseline_job_ranking.json")
with open(json_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Complete results: {json_file}")

# Save averaged metrics as CSV
averaged_metrics_list = []
for key in metric_keys:
    averaged_metrics_list.append({
        'metric': key,
        'mean': averaged_metrics[f'{key}_mean'],
        'std': averaged_metrics[f'{key}_std']
    })

averaged_df = pd.DataFrame(averaged_metrics_list)
averaged_csv = os.path.join(OUTPUT_DIR, "baseline_averaged_metrics.csv")
averaged_df.to_csv(averaged_csv, index=False)
print(f"✓ Averaged metrics: {averaged_csv}")

# Save individual run metrics as CSV
individual_runs_data = []
for i in range(NUM_RUNS):
    for key in metric_keys:
        individual_runs_data.append({
            'run': i + 1,
            'metric': key,
            'value': all_runs_metrics[i][key]
        })

individual_df = pd.DataFrame(individual_runs_data)
individual_csv = os.path.join(OUTPUT_DIR, "baseline_individual_runs.csv")
individual_df.to_csv(individual_csv, index=False)
print(f"✓ Individual runs: {individual_csv}")

if all_runs_scores_data[0]:
    scores_df = pd.DataFrame(all_runs_scores_data[0])
    scores_csv = os.path.join(OUTPUT_DIR, "baseline_scores_run1.csv")
    scores_df.to_csv(scores_csv, index=False)
    print(f"✓ Scores from Run 1: {scores_csv} ({len(scores_df)} batches)")

print("\n" + "=" * 100)
print("BASELINE EVALUATION COMPLETE! ")
print("=" * 100)
print(f"\n Summary:")
print(f"  - Tested on {len(qual_test_batches)} batches from qual_test.csv")
print(f"  - Ran {NUM_RUNS} times and averaged results")
print(f"\n  Final Metrics (mean ± std):")
print(f"    Recall@1: {averaged_metrics['recall@1_mean']:.4f} ± {averaged_metrics['recall@1_std']:.4f}")
print(f"    Recall@3: {averaged_metrics['recall@3_mean']:.4f} ± {averaged_metrics['recall@3_std']:.4f}")
print(f"    Recall@5: {averaged_metrics['recall@5_mean']:.4f} ± {averaged_metrics['recall@5_std']:.4f}")
print(f"    NDCG@1:   {averaged_metrics['ndcg@1_mean']:.4f} ± {averaged_metrics['ndcg@1_std']:.4f}")
print(f"    NDCG@3:   {averaged_metrics['ndcg@3_mean']:.4f} ± {averaged_metrics['ndcg@3_std']:.4f}")
print(f"    NDCG@5:   {averaged_metrics['ndcg@5_mean']:.4f} ± {averaged_metrics['ndcg@5_std']:.4f}")
print(f"    Avg Rank: {averaged_metrics['avg_rank_mean']:.2f} ± {averaged_metrics['avg_rank_std']:.2f}")
print(f"\n  - Results saved to: {OUTPUT_DIR}/")
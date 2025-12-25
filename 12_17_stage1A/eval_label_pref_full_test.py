#!/usr/bin/env python3
"""
Stage 1A: Full label_pref Test Evaluation
=========================================
Load final LoRA checkpoint + pref_head and evaluate on the full label_pref test set.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import numpy as np
from tqdm import tqdm

# ========================================
# CONFIG
# ========================================
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = ""
CACHE_DIR = os.path.expanduser("~/llama_cache")

CHECKPOINT_DIR = "./stage1a_output/run_20251217_232520"
TEST_CSV = f"../pipeline_output/stage1a_data/test.csv"
USER_FILE = "../pipeline_output/Final_users.csv"
ITEM_FILE = "../pipeline_output/Final_items.csv"

NEGATIVES_PER_USER = 49
MAX_SEQ_LENGTH = 1000
PROCESS_CHUNK_SIZE = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========================================
# LOAD DATA
# ========================================
user_df = pd.read_csv(USER_FILE)
user_df = user_df[user_df['student_id'].isin(user_df['student_id'])]
user_dict = user_df.set_index('student_id').to_dict('index')

item_df = pd.read_csv(ITEM_FILE)
item_df = item_df[item_df['job_id'].isin(item_df['job_id'])]
item_dict = item_df.set_index('job_id').to_dict('index')

test_df = pd.read_csv(TEST_CSV)

print("\n📊 TEST DATA STATS:")
num_samples = len(test_df)
num_users = test_df['user'].nunique()
num_positives = (test_df['label_pref'] == 1).sum()
print(f"  Total samples: {num_samples:,}")
print(f"  Unique users: {num_users:,}")
print(f"  Positive interactions: {num_positives:,}")

# ========================================
# PREPARE FULL TEST BATCHES
# ========================================
def prepare_pref_batches(df, negatives_per_user):
    batches = []
    user_groups = df.groupby('user')
    for user_id, group in user_groups:
        positives = group[group['label_pref'] == 1]
        negatives = group[group['label_pref'] == 0]
        if len(positives) > 0 and len(negatives) >= negatives_per_user:
            for pos_idx in range(len(positives)):
                pos_sample = positives.iloc[pos_idx]
                start_idx = (pos_idx * negatives_per_user) % len(negatives)
                if start_idx + negatives_per_user <= len(negatives):
                    neg_samples = negatives.iloc[start_idx:start_idx + negatives_per_user]
                else:
                    needed = negatives_per_user - (len(negatives) - start_idx)
                    neg_samples = pd.concat([negatives.iloc[start_idx:], negatives.iloc[:needed]])
                batches.append({
                    'user_id': user_id,
                    'positive_item': pos_sample['item'],
                    'negative_items': neg_samples['item'].tolist()[:negatives_per_user]
                })
    return batches

test_batches = prepare_pref_batches(test_df, NEGATIVES_PER_USER)
print(f"  Total test batches: {len(test_batches):,}")

# ========================================
# PROMPT BUILDER
# ========================================
def format_features(features_dict):
    lines = []
    for key, value in features_dict.items():
        nice_key = key.replace('_', ' ').title()
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        lines.append(f"  - {nice_key}: {value_str}")
    return "\n".join(lines)

def build_ranking_prompt(user_id, item_id):
    user_features = user_dict.get(user_id, {})
    user_info = format_features(user_features) if user_features else "Unknown user"
    item_features = item_dict.get(item_id, {})
    item_info = format_features(item_features) if item_features else "Unknown item"
    prompt = f"""Rate the interaction likelihood between user and item.

User ID: {user_id}
User Features:
{user_info}

Item ID: {item_id}
Item Features:
{item_info}

Interaction Likelihood:"""
    return prompt

# ========================================
# MODEL LOAD
# ========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=CACHE_DIR
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
base_model = prepare_model_for_kbit_training(base_model)

# Load LoRA adapters
lora_path = os.path.join(CHECKPOINT_DIR, "lora_adapters_final")
model = PeftModel.from_pretrained(base_model, lora_path, device_map="auto")

# Load pref_head
hidden_size = model.config.hidden_size
pref_head = nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(device)

pref_head.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "pref_head_final.pt")))
pref_head.eval()
model.eval()

# ========================================
# UTILITIES
# ========================================
def pool_hidden_states(hidden_states, attention_mask):
    batch_size = hidden_states.shape[0]
    pooled = []
    for i in range(batch_size):
        last_idx = attention_mask[i].nonzero()[-1].item()
        pooled.append(hidden_states[i, last_idx, :])
    return torch.stack(pooled)

def info_nce_loss(scores):
    positive_score = scores[0]
    numerator = torch.exp(positive_score)
    denominator = torch.sum(torch.exp(scores))
    return -torch.log(numerator / denominator)

def calculate_metrics(ranks):
    metrics = {}
    for k in [1,3,5]:
        metrics[f'recall@{k}'] = sum(1 for r in ranks if r <= k) / len(ranks)
    for k in [1,3,5]:
        ndcg_scores = []
        for rank in ranks:
            if rank > k:
                ndcg_scores.append(0.0)
            else:
                dcg = 1.0 / np.log2(rank + 1)
                idcg = 1.0 / np.log2(2)
                ndcg_scores.append(dcg/idcg)
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
    metrics['avg_rank'] = np.mean(ranks)
    return metrics

# ========================================
# EVALUATION
# ========================================
all_ranks = []
all_losses = []

with torch.no_grad():
    for batch in tqdm(test_batches, desc="Evaluating Test Batches", ncols=100):
        user_id = batch['user_id']
        all_items = [batch['positive_item']] + batch['negative_items']
        embeddings_list = []
        for i in range(0, len(all_items), PROCESS_CHUNK_SIZE):
            chunk_items = all_items[i:i+PROCESS_CHUNK_SIZE]
            prompts = [build_ranking_prompt(user_id, item) for item in chunk_items]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            embeddings = pool_hidden_states(hidden, inputs['attention_mask'])
            embeddings_list.append(embeddings)
            del inputs, outputs, hidden
        batch_embeddings = torch.cat(embeddings_list, dim=0).float()
        scores = pref_head(batch_embeddings).squeeze()
        rank = (scores[0] < scores[1:]).sum().item() + 1
        loss = info_nce_loss(scores)
        all_ranks.append(rank)
        all_losses.append(loss.item())
        del embeddings_list, batch_embeddings, scores, loss

metrics = calculate_metrics(all_ranks)
metrics['loss'] = np.mean(all_losses)

print("\n📊 FINAL EVALUATION - Label_PREF Test")
print(f"  Loss: {metrics['loss']:.4f}")
print(f"  Recall@1: {metrics['recall@1']:.3f} ({metrics['recall@1']*100:.1f}%)")
print(f"  Recall@3: {metrics['recall@3']:.3f} ({metrics['recall@3']*100:.1f}%)")
print(f"  Recall@5: {metrics['recall@5']:.3f} ({metrics['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {metrics['ndcg@1']:.3f}")
print(f"  NDCG@3: {metrics['ndcg@3']:.3f}")
print(f"  NDCG@5: {metrics['ndcg@5']:.3f}")
print(f"  Avg Rank: {metrics['avg_rank']:.1f}")
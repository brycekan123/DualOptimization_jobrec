#!/usr/bin/env python3

import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Evaluate Stage 1A model on qual_test')
parser.add_argument('--hf_token', type=str, required=True,
                    help='HuggingFace API token for model access')
args = parser.parse_args()

HF_TOKEN = args.hf_token
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = os.path.expanduser("~/llama_cache")

CHECKPOINT_PATH = "stage1a_output/run_20251217_232520/checkpoint_epoch3"
HEAD_TYPE = "pref" 

DATA_DIR = "stage1a_data"
QUAL_TEST_CSV = f"../pipeline_output/{DATA_DIR}/qual_test.csv"
USER_FILE = "../pipeline_output/Final_users.csv"
ITEM_FILE = "../pipeline_output/Final_items.csv"


MAX_SEQ_LENGTH = 1536
PROCESS_CHUNK_SIZE = 25

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 100)
print(f"EVALUATING {HEAD_TYPE.upper()}_HEAD ON QUAL_TEST")
print("=" * 100)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Device: {device}")


user_df = pd.read_csv(USER_FILE)
user_df = user_df[user_df['student_id'] != 'student_id']
user_df = user_df[user_df['student_id'] != 'user_id:token']
user_dict = user_df.set_index('student_id').to_dict('index')

item_df = pd.read_csv(ITEM_FILE)
item_df = item_df[item_df['job_id'] != 'job_id']
item_df = item_df[item_df['job_id'] != 'item_id:token']
item_dict = item_df.set_index('job_id').to_dict('index')

print(f"✓ Loaded {len(user_dict):,} users, {len(item_dict):,} items")

qual_test_df = pd.read_csv(QUAL_TEST_CSV)
print(f"✓ Qual test: {len(qual_test_df):,} samples, {qual_test_df['item'].nunique():,} jobs")


print("\n" + "=" * 100)
print("PREPARING BATCHES")
print("=" * 100)

def prepare_qual_batches(df, max_negatives=49):
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

batches = prepare_qual_batches(qual_test_df)

print(f"✓ Created {len(batches):,} batches (evaluating ALL)")

# Compute batch statistics
batch_sizes = [1 + len(b['negative_users']) for b in batches]
unique_jobs = set(b['job_id'] for b in batches)
all_positive_users = set(b['positive_user'] for b in batches)
all_negative_users = set()
for b in batches:
    all_negative_users.update(b['negative_users'])
total_unique_users = all_positive_users | all_negative_users

print(f"\n📊 Batch Statistics:")
print(f"  Total batches: {len(batches):,}")
print(f"  Unique jobs: {len(unique_jobs):,}")
print(f"  Batch size (1 pos + negatives):")
print(f"    Mean: {np.mean(batch_sizes):.1f}")
print(f"    Median: {np.median(batch_sizes):.1f}")
print(f"    Min: {np.min(batch_sizes)}")
print(f"    Max: {np.max(batch_sizes)}")
print(f"  Unique users:")
print(f"    Total: {len(total_unique_users):,}")
print(f"    Positive users: {len(all_positive_users):,}")
print(f"    Negative users: {len(all_negative_users):,}")


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

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("✓ Loaded base model (8-bit)")

base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    inference_mode=False,
)

model = get_peft_model(base_model, lora_config)

# Load LoRA weights
from safetensors.torch import load_file
lora_path = os.path.join(CHECKPOINT_PATH, "lora_adapters")
adapter_weights_path = os.path.join(lora_path, "adapter_model.safetensors")
adapter_state_dict = load_file(adapter_weights_path)
model.load_state_dict(adapter_state_dict, strict=False)
print(f"✓ Loaded LoRA weights")

# Load head
hidden_size = model.config.hidden_size
ranking_head = nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(device)

head_filename = f"{HEAD_TYPE}_head_final.pt" if os.path.exists(os.path.join(CHECKPOINT_PATH, f"{HEAD_TYPE}_head_final.pt")) else f"{HEAD_TYPE}_head.pt"
head_path = os.path.join(CHECKPOINT_PATH, head_filename)
ranking_head.load_state_dict(torch.load(head_path, map_location=device))
print(f"✓ Loaded {HEAD_TYPE}_head")

# ========================================
# EVALUATION
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
    loss = -torch.log(numerator / denominator)
    return loss

def calculate_metrics(ranks):
    metrics = {}
    
    for k in [1, 3, 5]:
        recall = sum(1 for r in ranks if r <= k) / len(ranks)
        metrics[f'recall@{k}'] = recall
    
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

print("\n" + "=" * 100)
print("EVALUATING")
print("=" * 100)

model.eval()
ranking_head.eval()

all_ranks = []
all_losses = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating", ncols=100)):
        job_id = batch['job_id']
        all_users = [batch['positive_user']] + batch['negative_users']
        
        user_chunks = [all_users[i:i+PROCESS_CHUNK_SIZE] 
                      for i in range(0, len(all_users), PROCESS_CHUNK_SIZE)]
        all_embeddings = []
        
        for chunk_users in user_chunks:
            chunk_prompts = [build_ranking_prompt(user, job_id) for user in chunk_users]
            chunk_inputs = tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(device)
            
            chunk_outputs = model(
                input_ids=chunk_inputs['input_ids'],
                attention_mask=chunk_inputs['attention_mask'],
                output_hidden_states=True
            )
            
            chunk_hidden = chunk_outputs.hidden_states[-1]
            chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
            all_embeddings.append(chunk_embeddings)
            
            del chunk_inputs, chunk_outputs, chunk_hidden
        
        embeddings_batch = torch.cat(all_embeddings, dim=0)
        embeddings_batch = embeddings_batch.float()
        scores = ranking_head(embeddings_batch).squeeze()
        
        rank = (scores[0] < scores[1:]).sum().item() + 1
        loss = info_nce_loss(scores)
        
        all_ranks.append(rank)
        all_losses.append(loss.item())
        
        del all_embeddings, chunk_embeddings, embeddings_batch, scores, loss
        
        if (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()

metrics = calculate_metrics(all_ranks)
metrics['loss'] = np.mean(all_losses)


print("\n" + "=" * 100)
print(f"RESULTS - {HEAD_TYPE.upper()}_HEAD ON QUAL_TEST")
print("=" * 100)

print(f"\n📊 Metrics:")
print(f"  Loss: {metrics['loss']:.4f}")
print(f"  Recall@1: {metrics['recall@1']:.3f} ({metrics['recall@1']*100:.1f}%)")
print(f"  Recall@3: {metrics['recall@3']:.3f} ({metrics['recall@3']*100:.1f}%)")
print(f"  Recall@5: {metrics['recall@5']:.3f} ({metrics['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {metrics['ndcg@1']:.3f}")
print(f"  NDCG@3: {metrics['ndcg@3']:.3f}")
print(f"  NDCG@5: {metrics['ndcg@5']:.3f}")
print(f"  Avg Rank: {metrics['avg_rank']:.1f}")


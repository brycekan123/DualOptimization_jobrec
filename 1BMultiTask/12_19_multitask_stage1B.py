#!/usr/bin/env python3

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description='Train Stage 1B multi-task model')
parser.add_argument('--pipeline_output', type=str, required=True,
                    help='Path to pipeline_output directory (e.g., ../pipeline_output)')
parser.add_argument('--data_dir', type=str, default='stage1a_data',
                    help='Name of the data subdirectory (default: stage1a_data)')
parser.add_argument('--hf_token', type=str, required=True,
                    help='HuggingFace API token for model access')
args = parser.parse_args()

HF_TOKEN = args.hf_token
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = os.path.expanduser("~/llama_cache")

PIPELINE_OUTPUT = args.pipeline_output
DATA_DIR = args.data_dir

# CHANGE 1: Use pre-filtered pref_68_batches_train.csv instead of train.csv
PREF_TRAIN_CSV = os.path.join(PIPELINE_OUTPUT, DATA_DIR, "pref_68_batches_train.csv")
PREF_TEST_CSV = os.path.join(PIPELINE_OUTPUT, DATA_DIR, "pref_68_batches_test.csv")
QUAL_TRAIN_CSV = os.path.join(PIPELINE_OUTPUT, DATA_DIR, "qual_train.csv")
QUAL_TEST_CSV = os.path.join(PIPELINE_OUTPUT, DATA_DIR, "qual_test.csv")
USER_FILE = os.path.join(PIPELINE_OUTPUT, "Final_users.csv")
ITEM_FILE = os.path.join(PIPELINE_OUTPUT, "Final_items.csv")

NEGATIVES_PER_USER = 49
MAX_SEQ_LENGTH = 1000
PROCESS_CHUNK_SIZE = 5 
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3

# NUM_TRAIN_BATCHES removed - we'll use all batches from the pre-filtered file
NUM_EVAL_PREF_BATCHES = 30  

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

OUTPUT_DIR = f"stage1b_multitask_output/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 100)
print("STAGE 1B: MULTI-TASK TRAINING (PREFERENCE + QUALITY)")
print("=" * 100)


def print_gpu_usage(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU {tag}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


print("\n" + "=" * 100)
print("LOADING DATA")
print("=" * 100)

# Load metadata
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

# Load train/test data - NOTE: pref_train_df now loads from pref_68_batches_train.csv
pref_train_df = pd.read_csv(PREF_TRAIN_CSV)
pref_test_df = pd.read_csv(PREF_TEST_CSV)
qual_train_df = pd.read_csv(QUAL_TRAIN_CSV)
qual_test_df = pd.read_csv(QUAL_TEST_CSV)

print(f"\n✓ Label_pref data (PRE-FILTERED to 68 batches):")
print(f"  Train: {len(pref_train_df):,} samples, {pref_train_df['user'].nunique():,} users")
print(f"  Test: {len(pref_test_df):,} samples, {pref_test_df['user'].nunique():,} users")

print(f"\n✓ Label_qual data:")
print(f"  Train: {len(qual_train_df):,} samples, {qual_train_df['item'].nunique():,} jobs")
print(f"  Test: {len(qual_test_df):,} samples, {qual_test_df['item'].nunique():,} jobs")


print("\n" + "=" * 100)
print("PREPARING BATCHES")
print("=" * 100)

def prepare_pref_batches(df, negatives_per_user):
    batches = []
    user_groups = df.groupby('user')
    
    for user_id, group in user_groups:
        positives = group[group['label_pref'] == 1]
        negatives = group[group['label_pref'] == 0]
        
        if len(positives) > 0 and len(negatives) >= negatives_per_user:
            for pos_idx in range(len(positives)):
                pos_sample = positives.iloc[pos_idx]
                
                # Rotate through negatives
                start_idx = (pos_idx * negatives_per_user) % len(negatives)
                
                if start_idx + negatives_per_user <= len(negatives):
                    neg_samples = negatives.iloc[start_idx:start_idx + negatives_per_user]
                else:
                    needed = negatives_per_user - (len(negatives) - start_idx)
                    neg_samples = pd.concat([
                        negatives.iloc[start_idx:],
                        negatives.iloc[:needed]
                    ])
                
                if len(neg_samples) >= negatives_per_user:
                    batches.append({
                        'user_id': user_id,
                        'positive_item': pos_sample['item'],
                        'negative_items': neg_samples['item'].tolist()[:negatives_per_user]
                    })
    
    return batches

def prepare_qual_batches(df, max_negatives=49):
    batches = []
    job_groups = df.groupby('item')
    
    for job_id, group in job_groups:
        positives = group[group['label_qual'] == 1]
        negatives = group[group['label_qual'] == 0]
        
        if len(positives) > 0:
            # Sample negatives if too many
            if len(negatives) > max_negatives:
                negatives = negatives.sample(n=max_negatives, random_state=42)
            
            # Create one batch per positive
            for pos_idx in range(len(positives)):
                pos_user = positives.iloc[pos_idx]['user']
                neg_users = negatives['user'].tolist()
                
                batches.append({
                    'job_id': job_id,
                    'positive_user': pos_user,
                    'negative_users': neg_users
                })
    
    return batches

# CHANGE 2: No random sampling - use all batches from pre-filtered file
pref_train_batches = prepare_pref_batches(pref_train_df, NEGATIVES_PER_USER)
pref_test_batches_all = prepare_pref_batches(pref_test_df, NEGATIVES_PER_USER)
qual_train_batches = prepare_qual_batches(qual_train_df)
qual_test_batches_all = prepare_qual_batches(qual_test_df)

print(f"\n Analysis of pref training batches:")
pref_users = [b['user_id'] for b in pref_train_batches]
print(f"  Total batches: {len(pref_train_batches)}")
print(f"  Unique users: {len(set(pref_users))}")
print(f"  Most frequent user appears: {max([pref_users.count(u) for u in set(pref_users)])} times")
if len(set(pref_users)) < 10:
    print(f"  ⚠️  WARNING: Only {len(set(pref_users))} unique users - data may not be diverse!")
else:
    print(f"  ✓ Data looks diverse")

print(f"\n✓ Multi-task training batches:")
print(f"  Pref train: {len(pref_train_batches)} (all batches from pre-filtered file)")
print(f"  Qual train: {len(qual_train_batches)} (all batches)")
print(f"  Batches will be paired by index")

pref_eval_batches = pref_test_batches_all[:NUM_EVAL_PREF_BATCHES]
qual_eval_batches = qual_test_batches_all  # Use all (~30)

print(f"\n✓ Epoch evaluation batches:")
print(f"  Pref test: {len(pref_eval_batches)} (first 30)")
print(f"  Qual test: {len(qual_eval_batches)} (all batches)")

print(f"\n✓ Final evaluation batches:")
print(f"  Pref test: {len(pref_test_batches_all)} (all batches)")
print(f"  Qual test: {len(qual_test_batches_all)} (all batches)")

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
    """Build prompt for user-item ranking."""
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


print("\n" + "=" * 100)
print("LOADING MODEL (TRAINING FROM SCRATCH)")
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
print("✓ Prepared for k-bit training")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    inference_mode=False,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Create both heads
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

qual_head = nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(device)

print(f"\n✓ Created heads:")
print(f"  pref_head: {sum(p.numel() for p in pref_head.parameters()):,} params")
print(f"  qual_head: {sum(p.numel() for p in qual_head.parameters()):,} params")

# Create learned task weights (initialized to 0.0 → softmax gives 0.5/0.5)
w_pref_raw = nn.Parameter(torch.tensor(0.0))
w_qual_raw = nn.Parameter(torch.tensor(0.0))

print(f"\n✓ Initialized learned task weights:")
print(f"  w_pref_raw: {w_pref_raw.item():.5f}")
print(f"  w_qual_raw: {w_qual_raw.item():.5f}")
weights_init = F.softmax(torch.stack([w_pref_raw, w_qual_raw]), dim=0)
print(f"  Initial weights after softmax: w_pref={weights_init[0].item():.5f}, w_qual={weights_init[1].item():.5f}")

print_gpu_usage("After model loading")


def pool_hidden_states(hidden_states, attention_mask):
    """Extract last non-padding token (same as Stage 1A)."""
    batch_size = hidden_states.shape[0]
    pooled = []
    for i in range(batch_size):
        last_idx = attention_mask[i].nonzero()[-1].item()
        pooled.append(hidden_states[i, last_idx, :])
    return torch.stack(pooled)

def info_nce_loss(scores):
    """InfoNCE loss: -log(exp(s_pos) / sum(exp(all)))"""
    positive_score = scores[0]
    numerator = torch.exp(positive_score)
    denominator = torch.sum(torch.exp(scores))
    loss = -torch.log(numerator / denominator)
    return loss

def calculate_metrics(ranks):
    """Calculate Recall@k and NDCG@k from list of ranks."""
    metrics = {}
    
    # Recall@k
    for k in [1, 3, 5]:
        recall = sum(1 for r in ranks if r <= k) / len(ranks)
        metrics[f'recall@{k}'] = recall
    
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

def evaluate_pref(batches, model, pref_head, desc="Evaluating Pref"):
    model.eval()
    pref_head.eval()
    
    all_ranks = []
    all_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=desc, ncols=100)):
            user_id = batch['user_id']
            all_items = [batch['positive_item']] + batch['negative_items']
            
            # Process in chunks (Stage 1A pattern)
            item_chunks = [all_items[i:i+PROCESS_CHUNK_SIZE] 
                          for i in range(0, len(all_items), PROCESS_CHUNK_SIZE)]
            all_embeddings = []
            
            for chunk_items in item_chunks:
                chunk_prompts = [build_ranking_prompt(user_id, item) for item in chunk_items]
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
                
                # Extract ONLY last layer hidden states
                chunk_hidden = chunk_outputs.hidden_states[-1]
                del chunk_outputs  # Free memory immediately
                
                chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
                all_embeddings.append(chunk_embeddings)
                
                del chunk_inputs, chunk_hidden
            
            embeddings_batch = torch.cat(all_embeddings, dim=0)
            embeddings_batch = embeddings_batch.float()
            scores = pref_head(embeddings_batch).squeeze()
            
            # Calculate rank (positive is at index 0)
            rank = (scores[0] < scores[1:]).sum().item() + 1
            loss = info_nce_loss(scores)
            
            all_ranks.append(rank)
            all_losses.append(loss.item())
            
            del all_embeddings, chunk_embeddings, embeddings_batch, scores, loss
            
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
    
    metrics = calculate_metrics(all_ranks)
    metrics['loss'] = np.mean(all_losses)
    
    return metrics

def evaluate_qual(batches, model, qual_head, desc="Evaluating Qual"):
    """Evaluate on label_qual batches (job-centric)."""
    model.eval()
    qual_head.eval()
    
    all_ranks = []
    all_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=desc, ncols=100)):
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
                
                # Extract ONLY last layer hidden states
                chunk_hidden = chunk_outputs.hidden_states[-1]
                del chunk_outputs  
                
                chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
                all_embeddings.append(chunk_embeddings)
                
                del chunk_inputs, chunk_hidden
            
            embeddings_batch = torch.cat(all_embeddings, dim=0)
            embeddings_batch = embeddings_batch.float()
            scores = qual_head(embeddings_batch).squeeze()
            
            # Calculate rank (positive is at index 0)
            rank = (scores[0] < scores[1:]).sum().item() + 1
            loss = info_nce_loss(scores)
            
            all_ranks.append(rank)
            all_losses.append(loss.item())
            
            del all_embeddings, chunk_embeddings, embeddings_batch, scores, loss
            
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
    
    metrics = calculate_metrics(all_ranks)
    metrics['loss'] = np.mean(all_losses)
    
    return metrics

# ========================================
# TRAINING LOOP
# ========================================
print("\n" + "=" * 100)
print("TRAINING")
print("=" * 100)

# Optimizer: LoRA + both heads + learned weights
optimizer = torch.optim.AdamW(
    list(model.parameters()) + 
    list(pref_head.parameters()) + 
    list(qual_head.parameters()) +
    [w_pref_raw, w_qual_raw],
    lr=LEARNING_RATE
)

print(f"\nConfiguration:")
print(f"  Train batches: {len(pref_train_batches)} pref + {len(qual_train_batches)} qual")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Trainable: LoRA + pref_head + qual_head + task weights")

print_gpu_usage("Before training")

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*80}")
    
    model.train()
    pref_head.train()
    qual_head.train()
    
    epoch_loss_pref = 0
    epoch_loss_qual = 0
    epoch_loss_total = 0
    
    pbar = tqdm(range(len(pref_train_batches)), desc=f"Epoch {epoch+1}", ncols=100)
    
    # Track first batch for diagnostics
    first_batch = True
    
    for batch_idx in range(len(pref_train_batches)):
        pref_batch = pref_train_batches[batch_idx]
        qual_batch = qual_train_batches[batch_idx]
        
        if batch_idx == 0:
            print("\n" + "="*80)
            print("DETAILED MEMORY TRACKING - FIRST BATCH")
            print("="*80)
            print_gpu_usage("Start of batch 0")
        
        # ============================================
        # COMPUTE WEIGHTS ONCE (BEFORE BOTH FORWARDS)
        # ============================================
        optimizer.zero_grad()  # Clear gradients once at start
        
        weights = F.softmax(torch.stack([w_pref_raw, w_qual_raw]), dim=0)
        w_pref = weights[0]
        w_qual = weights[1]
        
        if batch_idx == 0:
            print(f"\nWeights: w_pref={w_pref.item():.5f}, w_qual={w_qual.item():.5f}")
        
        # ============================================
        # FORWARD PASS 1: PREF BATCH
        # ============================================
        if batch_idx == 0:
            print("\n[PREF FORWARD PASS]")
        
        user_id = pref_batch['user_id']
        all_items = [pref_batch['positive_item']] + pref_batch['negative_items']
        
        if batch_idx == 0:
            print(f"  Pref batch size: 1 pos + {len(pref_batch['negative_items'])} neg = {len(all_items)} total")
        
        # Process in chunks (Stage 1A pattern)
        item_chunks = [all_items[i:i+PROCESS_CHUNK_SIZE] 
                      for i in range(0, len(all_items), PROCESS_CHUNK_SIZE)]
        
        if batch_idx == 0:
            print(f"  Split into {len(item_chunks)} chunks of size {PROCESS_CHUNK_SIZE}")
        
        all_embeddings_pref = []
        
        for chunk_idx, chunk_items in enumerate(item_chunks):
            if batch_idx == 0:
                print(f"\n  Chunk {chunk_idx+1}/{len(item_chunks)}: {len(chunk_items)} items")
                print_gpu_usage(f"  Before chunk {chunk_idx+1}")
            
            chunk_prompts = [build_ranking_prompt(user_id, item) for item in chunk_items]
            chunk_inputs = tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(device)
            
            if batch_idx == 0:
                print(f"  Input shape: {chunk_inputs['input_ids'].shape}")
                print_gpu_usage(f"  After tokenization")
            
            chunk_outputs = model(
                input_ids=chunk_inputs['input_ids'],
                attention_mask=chunk_inputs['attention_mask'],
                output_hidden_states=True
            )
            
            if batch_idx == 0:
                print_gpu_usage(f"  After model forward")
            
            # Extract ONLY last layer hidden states
            chunk_hidden = chunk_outputs.hidden_states[-1]
            del chunk_outputs  # Free memory immediately
            
            chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
            all_embeddings_pref.append(chunk_embeddings)
            
            del chunk_inputs, chunk_hidden
            
            if batch_idx == 0:
                print_gpu_usage(f"  After cleanup")
        
        embeddings_batch_pref = torch.cat(all_embeddings_pref, dim=0)
        embeddings_batch_pref = embeddings_batch_pref.float()
        scores_pref = pref_head(embeddings_batch_pref).squeeze()
        
        if batch_idx == 0:
            print(f"\n  Pref scores shape: {scores_pref.shape}")
            print_gpu_usage("  After pref_head")
        
        # Compute pref loss
        loss_pref = info_nce_loss(scores_pref)
        
        if batch_idx == 0:
            print(f"  Pref loss: {loss_pref.item():.4f}")
            print_gpu_usage("  After pref loss computation")
        
        # ============================================
        # IMMEDIATE BACKWARD: PREF (FREE GRAPH NOW!)
        # ============================================
        loss_pref_weighted = w_pref * loss_pref
        loss_pref_weighted.backward(retain_graph=True)  # Retain weights graph for second backward
        
        if batch_idx == 0:
            print_gpu_usage("  After pref backward (retain_graph=True)")
        
        # Store loss value for logging (before deletion)
        loss_pref_value = loss_pref.item()
        
        # Free pref computation graph
        del all_embeddings_pref, chunk_embeddings, embeddings_batch_pref, scores_pref
        del loss_pref, loss_pref_weighted
        
        if batch_idx == 0:
            print_gpu_usage("  After pref cleanup")
            torch.cuda.empty_cache()
            print_gpu_usage("  After cache clear")
        
        # ============================================
        # FORWARD PASS 2: QUAL BATCH
        # ============================================
        if batch_idx == 0:
            print("\n[QUAL FORWARD PASS]")
        
        job_id = qual_batch['job_id']
        all_users = [qual_batch['positive_user']] + qual_batch['negative_users']
        
        if batch_idx == 0:
            print(f"  Qual batch size: 1 pos + {len(qual_batch['negative_users'])} neg = {len(all_users)} total")
        
        # Process in chunks (Stage 1A pattern)
        user_chunks = [all_users[i:i+PROCESS_CHUNK_SIZE] 
                      for i in range(0, len(all_users), PROCESS_CHUNK_SIZE)]
        
        if batch_idx == 0:
            print(f"  Split into {len(user_chunks)} chunks of size {PROCESS_CHUNK_SIZE}")
        
        all_embeddings_qual = []
        
        for chunk_idx, chunk_users in enumerate(user_chunks):
            if batch_idx == 0:
                print(f"\n  Chunk {chunk_idx+1}/{len(user_chunks)}: {len(chunk_users)} users")
                print_gpu_usage(f"  Before chunk {chunk_idx+1}")
            
            chunk_prompts = [build_ranking_prompt(user, job_id) for user in chunk_users]
            chunk_inputs = tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(device)
            
            if batch_idx == 0:
                print(f"  Input shape: {chunk_inputs['input_ids'].shape}")
                print_gpu_usage(f"  After tokenization")
            
            chunk_outputs = model(
                input_ids=chunk_inputs['input_ids'],
                attention_mask=chunk_inputs['attention_mask'],
                output_hidden_states=True
            )
            
            if batch_idx == 0:
                print_gpu_usage(f"  After model forward")
            
            # Extract ONLY last layer hidden states
            chunk_hidden = chunk_outputs.hidden_states[-1]
            del chunk_outputs  # Free memory immediately
            
            chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
            all_embeddings_qual.append(chunk_embeddings)
            
            del chunk_inputs, chunk_hidden
            
            if batch_idx == 0:
                print_gpu_usage(f"  After cleanup")
        
        embeddings_batch_qual = torch.cat(all_embeddings_qual, dim=0)
        embeddings_batch_qual = embeddings_batch_qual.float()
        scores_qual = qual_head(embeddings_batch_qual).squeeze()
        
        if batch_idx == 0:
            print(f"\n  Qual scores shape: {scores_qual.shape}")
            print_gpu_usage("  After qual_head")
        
        # Compute qual loss
        loss_qual = info_nce_loss(scores_qual)
        
        if batch_idx == 0:
            print(f"  Qual loss: {loss_qual.item():.4f}")
            print_gpu_usage("  After qual loss computation")
        
        # ============================================
        # IMMEDIATE BACKWARD: QUAL (ACCUMULATE GRADIENTS)
        # ============================================
        loss_qual_weighted = w_qual * loss_qual
        loss_qual_weighted.backward()  # Accumulate with pref gradients
        
        if batch_idx == 0:
            print_gpu_usage("  After qual backward")
        
        # Store loss value for logging (before deletion)
        loss_qual_value = loss_qual.item()
        
        # Free qual computation graph
        del all_embeddings_qual, chunk_embeddings, embeddings_batch_qual, scores_qual
        del loss_qual, loss_qual_weighted
        
        if batch_idx == 0:
            print_gpu_usage("  After qual cleanup")
            torch.cuda.empty_cache()
            print_gpu_usage("  After cache clear")
        
        # Gradients already accumulated from both backwards
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(pref_head.parameters()) + 
            list(qual_head.parameters()) + [w_pref_raw, w_qual_raw],
            max_norm=1.0
        )
        
        if batch_idx == 0:
            print("\n[GRADIENT CLIPPING & OPTIMIZER STEP]")
            print_gpu_usage("  Before optimizer.step()")
        
        # Update with accumulated gradients from both tasks
        optimizer.step()
        
        if batch_idx == 0:
            print_gpu_usage("  After optimizer.step()")
        
        # Track losses (for logging - using stored values)
        loss_total = loss_pref_value + loss_qual_value
        epoch_loss_pref += loss_pref_value
        epoch_loss_qual += loss_qual_value
        epoch_loss_total += loss_total
        
        # Diagnostics on first batch (AFTER backward to check gradients)
        if first_batch:
            print("\n FIRST BATCH DIAGNOSTICS:")
            print(f"  loss_pref: {loss_pref_value:.4f}")
            print(f"  loss_qual: {loss_qual_value:.4f}")
            print(f"  w_pref: {w_pref.item():.5f}")
            print(f"  w_qual: {w_qual.item():.5f}")
            print(f"  combined loss: {loss_total:.4f}")
            print(f"  (Using IMMEDIATE backwards to save memory)")
            
            # Check gradients
            pref_grad = torch.nn.utils.clip_grad_norm_(pref_head.parameters(), max_norm=float('inf'))
            qual_grad = torch.nn.utils.clip_grad_norm_(qual_head.parameters(), max_norm=float('inf'))
            print(f"  pref_head grad norm: {pref_grad:.6f}")
            print(f"  qual_head grad norm: {qual_grad:.6f}")
            
            lora_grads = []
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    if param.grad is not None:
                        lora_grads.append(param.grad.norm().item())
            if lora_grads:
                print(f"  LoRA grad norm (avg): {np.mean(lora_grads):.6f}")
            
            print(f"  w_pref_raw grad: {w_pref_raw.grad.item() if w_pref_raw.grad is not None else 'None'}")
            print(f"  w_qual_raw grad: {w_qual_raw.grad.item() if w_qual_raw.grad is not None else 'None'}")
            print("="*80)
            print("END OF DETAILED MEMORY TRACKING")
            print("="*80 + "\n")
            
            first_batch = False
        
        pbar.update(1)
        pbar.set_postfix({
            'l_p': f'{loss_pref_value:.2f}',
            'l_q': f'{loss_qual_value:.2f}',
            'w_p': f'{w_pref.item():.5f}',
            'w_q': f'{w_qual.item():.5f}'
        })
        
        del weights, w_pref, w_qual
        
        # Aggressive memory cleanup every batch
        torch.cuda.empty_cache()
    
    pbar.close()
    
    # Print learned weights at end of epoch
    weights_epoch = F.softmax(torch.stack([w_pref_raw, w_qual_raw]), dim=0)
    print(f"\n Epoch {epoch+1} Summary:")
    print(f"  Avg loss_pref: {epoch_loss_pref/len(pref_train_batches):.4f}")
    print(f"  Avg loss_qual: {epoch_loss_qual/len(qual_train_batches):.4f}")
    print(f"  Avg loss_total: {epoch_loss_total/len(pref_train_batches):.4f}")
    print(f"  Learned weights: w_pref={weights_epoch[0].item():.5f}, w_qual={weights_epoch[1].item():.5f}")
    print(f"  Raw values: w_pref_raw={w_pref_raw.item():.5f}, w_qual_raw={w_qual_raw.item():.5f}")
    
    # Epoch evaluation (quick check)
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch+1} EVALUATION (QUICK CHECK)")
    print(f"{'='*80}")
    
    # Main evaluations
    print("\n Evaluating pref_head on 30 pref test batches...")
    pref_metrics = evaluate_pref(pref_eval_batches, model, pref_head, desc=f"Epoch {epoch+1} Pref")
    print(f"  Recall@1: {pref_metrics['recall@1']:.3f} | Recall@5: {pref_metrics['recall@5']:.3f}")
    
    print("\n Evaluating qual_head on all qual test batches...")
    qual_metrics = evaluate_qual(qual_eval_batches, model, qual_head, desc=f"Epoch {epoch+1} Qual")
    print(f"  Recall@1: {qual_metrics['recall@1']:.3f} | Recall@5: {qual_metrics['recall@5']:.3f}")
    
    print("\n Cross-eval: qual_head on pref test (expect BAD)...")
    qual_on_pref = evaluate_pref(pref_eval_batches, model, qual_head, desc=f"Epoch {epoch+1} Qual→Pref")
    print(f"  Recall@1: {qual_on_pref['recall@1']:.3f} | Recall@5: {qual_on_pref['recall@5']:.3f}")
    
    print("\n Cross-eval: pref_head on qual test (expect BAD)...")
    pref_on_qual = evaluate_qual(qual_eval_batches, model, pref_head, desc=f"Epoch {epoch+1} Pref→Qual")
    print(f"  Recall@1: {pref_on_qual['recall@1']:.3f} | Recall@5: {pref_on_qual['recall@5']:.3f}")
    
    print(f"\n Epoch {epoch+1} Summary:")
    print(f"  Main: pref_head→pref R@1={pref_metrics['recall@1']:.3f}, qual_head→qual R@1={qual_metrics['recall@1']:.3f}")
    print(f"  Cross: qual_head→pref R@1={qual_on_pref['recall@1']:.3f}, pref_head→qual R@1={pref_on_qual['recall@1']:.3f}")
    
    # Save checkpoint
    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.save_pretrained(os.path.join(checkpoint_dir, "lora_adapters"))
    torch.save(pref_head.state_dict(), os.path.join(checkpoint_dir, "pref_head.pt"))
    torch.save(qual_head.state_dict(), os.path.join(checkpoint_dir, "qual_head.pt"))
    torch.save({
        'w_pref_raw': w_pref_raw,
        'w_qual_raw': w_qual_raw
    }, os.path.join(checkpoint_dir, "learned_weights.pt"))
    
    print(f"\n✓ Saved checkpoint: {checkpoint_dir}")

# ========================================
# FINAL EVALUATION
# ========================================
print("\n" + "=" * 100)
print("FINAL EVALUATION (ALL TEST BATCHES)")
print("=" * 100)

# Final learned weights
weights_final = F.softmax(torch.stack([w_pref_raw, w_qual_raw]), dim=0)
print(f"\n Final learned weights:")
print(f"  w_pref: {weights_final[0].item():.5f}")
print(f"  w_qual: {weights_final[1].item():.5f}")
print(f"  Raw values: w_pref_raw={w_pref_raw.item():.5f}, w_qual_raw={w_qual_raw.item():.5f}")

# Main evaluations
print("\n" + "=" * 100)
print("MAIN EVALUATIONS ")
print("=" * 100)

print("\n Evaluating pref_head on ALL label_pref test batches...")
pref_metrics_final = evaluate_pref(pref_test_batches_all, model, pref_head, desc="Final Pref Eval")

print(f"\n FINAL - pref_head on label_PREF test:")
print(f"  Loss: {pref_metrics_final['loss']:.4f}")
print(f"  Recall@1: {pref_metrics_final['recall@1']:.3f} ({pref_metrics_final['recall@1']*100:.1f}%)")
print(f"  Recall@3: {pref_metrics_final['recall@3']:.3f} ({pref_metrics_final['recall@3']*100:.1f}%)")
print(f"  Recall@5: {pref_metrics_final['recall@5']:.3f} ({pref_metrics_final['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {pref_metrics_final['ndcg@1']:.3f}")
print(f"  NDCG@3: {pref_metrics_final['ndcg@3']:.3f}")
print(f"  NDCG@5: {pref_metrics_final['ndcg@5']:.3f}")
print(f"  Avg Rank: {pref_metrics_final['avg_rank']:.1f}/50")

print("\n Evaluating qual_head on ALL label_qual test batches...")
qual_metrics_final = evaluate_qual(qual_test_batches_all, model, qual_head, desc="Final Qual Eval")

print(f"\n FINAL - qual_head on label_QUAL test:")
print(f"  Loss: {qual_metrics_final['loss']:.4f}")
print(f"  Recall@1: {qual_metrics_final['recall@1']:.3f} ({qual_metrics_final['recall@1']*100:.1f}%)")
print(f"  Recall@3: {qual_metrics_final['recall@3']:.3f} ({qual_metrics_final['recall@3']*100:.1f}%)")
print(f"  Recall@5: {qual_metrics_final['recall@5']:.3f} ({qual_metrics_final['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {qual_metrics_final['ndcg@1']:.3f}")
print(f"  NDCG@3: {qual_metrics_final['ndcg@3']:.3f}")
print(f"  NDCG@5: {qual_metrics_final['ndcg@5']:.3f}")
print(f"  Avg Rank: {qual_metrics_final['avg_rank']:.1f}")

# Cross-evaluations (expect bad - check specialization)
print("\n" + "=" * 100)
print("CROSS-EVALUATIONS (wrong head on wrong task - expect BAD)")
print("=" * 100)

print("\n Cross-eval: qual_head on ALL label_pref test batches...")
qual_on_pref_final = evaluate_pref(pref_test_batches_all, model, qual_head, desc="Final Qual→Pref")

print(f"\n FINAL - qual_head on label_PREF test (expect BAD):")
print(f"  Loss: {qual_on_pref_final['loss']:.4f}")
print(f"  Recall@1: {qual_on_pref_final['recall@1']:.3f} ({qual_on_pref_final['recall@1']*100:.1f}%)")
print(f"  Recall@3: {qual_on_pref_final['recall@3']:.3f} ({qual_on_pref_final['recall@3']*100:.1f}%)")
print(f"  Recall@5: {qual_on_pref_final['recall@5']:.3f} ({qual_on_pref_final['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {qual_on_pref_final['ndcg@1']:.3f}")
print(f"  Avg Rank: {qual_on_pref_final['avg_rank']:.1f}/50")

print("\n Cross-eval: pref_head on ALL label_qual test batches...")
pref_on_qual_final = evaluate_qual(qual_test_batches_all, model, pref_head, desc="Final Pref→Qual")

print(f"\n FINAL - pref_head on label_QUAL test (expect BAD):")
print(f"  Loss: {pref_on_qual_final['loss']:.4f}")
print(f"  Recall@1: {pref_on_qual_final['recall@1']:.3f} ({pref_on_qual_final['recall@1']*100:.1f}%)")
print(f"  Recall@3: {pref_on_qual_final['recall@3']:.3f} ({pref_on_qual_final['recall@3']*100:.1f}%)")
print(f"  Recall@5: {pref_on_qual_final['recall@5']:.3f} ({pref_on_qual_final['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {pref_on_qual_final['ndcg@1']:.3f}")
print(f"  Avg Rank: {pref_on_qual_final['avg_rank']:.1f}")

# Summary
print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)
print(f"\nMain Performance:")
print(f"  pref_head → pref_test: Recall@1={pref_metrics_final['recall@1']:.3f}")
print(f"  qual_head → qual_test: Recall@1={qual_metrics_final['recall@1']:.3f}")
print(f"\nCross-Evaluation (specialization check):")
print(f"  qual_head → pref_test: Recall@1={qual_on_pref_final['recall@1']:.3f} (should be BAD)")
print(f"  pref_head → qual_test: Recall@1={pref_on_qual_final['recall@1']:.3f} (should be BAD)")
print(f"\nLearned Weights:")
print(f"  w_pref: {weights_final[0].item():.5f}")
print(f"  w_qual: {weights_final[1].item():.5f}")

print("\n" + "=" * 100)
print("TRAINING COMPLETE!")
print("=" * 100)

print("\n Saving final model...")

lora_path = os.path.join(OUTPUT_DIR, "lora_adapters_final")
model.save_pretrained(lora_path)
print(f"✓ Saved LoRA adapters: {lora_path}")

pref_head_path = os.path.join(OUTPUT_DIR, "pref_head_final.pt")
torch.save(pref_head.state_dict(), pref_head_path)
print(f"✓ Saved pref_head: {pref_head_path}")

qual_head_path = os.path.join(OUTPUT_DIR, "qual_head_final.pt")
torch.save(qual_head.state_dict(), qual_head_path)
print(f"✓ Saved qual_head: {qual_head_path}")

weights_path = os.path.join(OUTPUT_DIR, "learned_weights_final.pt")
torch.save({
    'w_pref_raw': w_pref_raw,
    'w_qual_raw': w_qual_raw,
    'w_pref': weights_final[0],
    'w_qual': weights_final[1]
}, weights_path)
print(f"✓ Saved learned weights: {weights_path}")

tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
print(f"✓ Saved tokenizer")

print(f"\n All files saved to: {OUTPUT_DIR}")

print("\n" + "=" * 100)
print("STAGE 1B MULTI-TASK TRAINING COMPLETE!")
print("=" * 100)

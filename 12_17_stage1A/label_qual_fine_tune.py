#!/usr/bin/env python3
"""
Stage 1B: Quality Expert Training
==================================
Train LLaMA encoder + LoRA + MLP qual_head on label_qual (with hard negatives)

Data Files Used:
- stage1a_data/qual_train.csv → Training data with label_qual column (job-centric)
- stage1a_data/qual_test.csv → Test data with label_qual column (job-centric)  
- stage1a_data/test.csv → Cross-eval data with label_pref column (user-centric)

Data Structure:
- 1 job per batch
- 1 positive (label_qual=1) + N hard negatives (label_qual=0, variable N)

Training:
- Start from BASE LLaMA model (not from Stage 1A checkpoint!)
- Finetune LoRA adapters + MLP qual_head on label_qual using InfoNCE

Eval:
- Recall@1,3,5 and NDCG@1,3,5 on label_qual test set
- Cross-eval: Recall@1,3,5 and NDCG@1,3,5 on ENTIRE label_pref test set
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ========================================
# CONFIGURATION
# ========================================
HF_TOKEN = ""
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = os.path.expanduser("~/llama_cache")

# Data paths
# Stage 1B uses qual_*.csv files for label_qual training
# Cross-evaluates on test.csv which has label_pref data
DATA_DIR = "stage1a_data"
TRAIN_CSV = f"../pipeline_output/{DATA_DIR}/qual_train.csv"  # Use qual_train for label_qual
TEST_CSV = f"../pipeline_output/{DATA_DIR}/qual_test.csv"    # Use qual_test for label_qual
PREF_TEST_CSV = f"../pipeline_output/{DATA_DIR}/test.csv"     # Use test.csv for cross-eval (has label_pref)
USER_FILE = "../pipeline_output/Final_users.csv"
ITEM_FILE = "../pipeline_output/Final_items.csv"
MAX_SEQ_LENGTH = 1000
PROCESS_CHUNK_SIZE = 10
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3

# NO SAMPLING - Use all qual batches!
# Using qual_train.csv and qual_test.csv which have label_qual data

# LoRA settings
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Output directory
OUTPUT_DIR = f"stage1b_output/run_fine_tune_label_qual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# DEVICE SETUP
# ========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 100)
print("STAGE 1B: QUALITY EXPERT TRAINING")
print("=" * 100)
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ========================================
# LOAD DATA
# ========================================
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

# Load train/test data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Check what columns we have
print(f"\n🔍 Checking columns in data files:")
print(f"  Train columns: {list(train_df.columns)}")
print(f"  Test columns: {list(test_df.columns)}")

# Determine which label column to use
if 'label_qual' in train_df.columns:
    label_col = 'label_qual'
    print(f"  ✓ Using 'label_qual' column (quality labels)")
elif 'label_pref' in train_df.columns:
    label_col = 'label_pref'
    print(f"  ⚠️  WARNING: Using 'label_pref' column - you may need different data for Stage 1B!")
    print(f"  Expected: label_qual column for quality expert training")
else:
    raise ValueError(f"No label column found! Available columns: {list(train_df.columns)}")

print(f"\n✓ Label_qual data (with hard negatives):")
print(f"  Train (qual_train.csv): {len(train_df):,} samples, {train_df['item'].nunique():,} jobs")
print(f"  Test (qual_test.csv): {len(test_df):,} samples, {test_df['item'].nunique():,} jobs")
print(f"  Train positives: {(train_df[label_col]==1).sum():,}")
print(f"  Test positives: {(test_df[label_col]==1).sum():,}")

# Load pref test for cross-eval (ENTIRE set!)
pref_test_df = pd.read_csv(PREF_TEST_CSV)
print(f"\n✓ Label_pref test (test.csv - for cross-eval - ENTIRE set):")
print(f"  Samples: {len(pref_test_df):,}")
print(f"  Users: {pref_test_df['user'].nunique():,}")
print(f"  Positives: {(pref_test_df['label_pref']==1).sum():,}")

# ========================================
# PREPARE BATCHES
# ========================================
print("\n" + "=" * 100)
print("PREPARING BATCHES")
print("=" * 100)

def prepare_qual_batches(df, label_col, min_negatives=1):
    """
    Prepare batches for label_qual (job-centric).
    Each batch: 1 job with 1 positive + N negatives (variable N)
    """
    batches = []
    job_groups = df.groupby('item')
    
    for job_id, group in job_groups:
        positives = group[group[label_col] == 1]
        negatives = group[group[label_col] == 0]
        
        if len(positives) > 0 and len(negatives) >= min_negatives:
            # Create one batch per positive
            for pos_idx in range(len(positives)):
                pos_user = positives.iloc[pos_idx]['user']
                neg_users = negatives['user'].tolist()
                
                batches.append({
                    'job_id': job_id,
                    'positive_user': pos_user,
                    'negative_users': neg_users,
                    'num_negatives': len(neg_users)
                })
    
    return batches

def prepare_pref_batches(df, negatives_per_user=49):
    """
    Prepare batches for label_pref (user-centric).
    Each batch: 1 user with 1 positive + up to 49 negatives
    """
    batches = []
    user_groups = df.groupby('user')
    
    for user_id, group in user_groups:
        positives = group[group['label_pref'] == 1]
        negatives = group[group['label_pref'] == 0]
        
        if len(positives) > 0 and len(negatives) >= 1:
            for pos_idx in range(len(positives)):
                pos_sample = positives.iloc[pos_idx]
                
                # Take up to negatives_per_user negatives
                neg_items = negatives['item'].tolist()[:negatives_per_user]
                
                batches.append({
                    'user_id': user_id,
                    'positive_item': pos_sample['item'],
                    'negative_items': neg_items,
                    'num_negatives': len(neg_items)
                })
    
    return batches

# NO SAMPLING - Use ALL batches
train_batches = prepare_qual_batches(train_df, label_col)
test_batches = prepare_qual_batches(test_df, label_col)
pref_test_batches = prepare_pref_batches(pref_test_df)  # ENTIRE set!

print(f"✓ Label_qual batches (job-centric, ALL batches):")
print(f"  Train: {len(train_batches):,} batches")
print(f"  Test: {len(test_batches):,} batches")

# Check variable negatives
train_neg_counts = [b['num_negatives'] for b in train_batches]
print(f"  Train negatives per job: min={min(train_neg_counts)}, max={max(train_neg_counts)}, avg={np.mean(train_neg_counts):.1f}")

test_neg_counts = [b['num_negatives'] for b in test_batches]
print(f"  Test negatives per job: min={min(test_neg_counts)}, max={max(test_neg_counts)}, avg={np.mean(test_neg_counts):.1f}")

print(f"\n✓ Label_pref batches (user-centric for cross-eval, ENTIRE set):")
print(f"  Test: {len(pref_test_batches):,} batches")
pref_neg_counts = [b['num_negatives'] for b in pref_test_batches]
print(f"  Negatives per user: min={min(pref_neg_counts)}, max={max(pref_neg_counts)}, avg={np.mean(pref_neg_counts):.1f}")

# ========================================
# PROMPT BUILDER
# ========================================
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

# ========================================
# LOAD MODEL (FRESH FROM BASE!)
# ========================================
print("\n" + "=" * 100)
print("LOADING MODEL (FRESH FROM BASE LLAMA - NOT FROM STAGE 1A!)")
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
print("  ⚠️  Starting from FRESH base LLaMA (NOT from Stage 1A checkpoint)")

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

hidden_size = model.config.hidden_size
qual_head = nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(device)

print(f"\n✓ Quality head (randomly initialized): {sum(p.numel() for p in qual_head.parameters()):,} params")

# ========================================
# TRAINING UTILITIES
# ========================================
def pool_hidden_states(hidden_states, attention_mask):
    """Extract last non-padding token."""
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
            
            # Process in chunks
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

def evaluate_pref(batches, model, qual_head, desc="Evaluating Pref"):
    """Evaluate on label_pref batches (user-centric) - cross-evaluation."""
    model.eval()
    qual_head.eval()
    
    all_ranks = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=desc, ncols=100)):
            user_id = batch['user_id']
            all_items = [batch['positive_item']] + batch['negative_items']
            
            # Process in chunks
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
                
                chunk_hidden = chunk_outputs.hidden_states[-1]
                chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
                all_embeddings.append(chunk_embeddings)
                
                del chunk_inputs, chunk_outputs, chunk_hidden
            
            embeddings_batch = torch.cat(all_embeddings, dim=0)
            embeddings_batch = embeddings_batch.float()
            scores = qual_head(embeddings_batch).squeeze()
            
            # Calculate rank (positive is at index 0)
            rank = (scores[0] < scores[1:]).sum().item() + 1
            all_ranks.append(rank)
            
            del all_embeddings, chunk_embeddings, embeddings_batch, scores
            
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
    
    metrics = calculate_metrics(all_ranks)
    
    return metrics

# ========================================
# BASELINE EVALUATION
# ========================================
print("\n" + "=" * 100)
print("BASELINE EVALUATION (SKIPPED FOR NOW)")
print("=" * 100)
print("✓ Skipping baseline eval to start training immediately")

print("\n🧹 Cleaning memory before training...")
torch.cuda.empty_cache()

# ========================================
# TRAINING LOOP
# ========================================
print("\n" + "=" * 100)
print("TRAINING")
print("=" * 100)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(qual_head.parameters()),
    lr=LEARNING_RATE
)

print(f"\nConfiguration:")
print(f"  Train batches: {len(train_batches):,} (ALL batches)")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")

# Track metrics across epochs
all_epoch_metrics = []

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*80}")
    
    model.train()
    qual_head.train()
    
    epoch_loss = 0
    pbar = tqdm(range(len(train_batches)), desc=f"Epoch {epoch+1}", ncols=100)
    
    first_batch = True
    
    for batch_idx, batch in enumerate(train_batches):
        job_id = batch['job_id']
        all_users = [batch['positive_user']] + batch['negative_users']
        
        # Process in chunks
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
        scores = qual_head(embeddings_batch).squeeze()
        
        # InfoNCE loss
        loss = info_nce_loss(scores)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(qual_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        # Check gradients on first batch
        if first_batch:
            print("\n🔍 GRADIENT CHECK (First Batch):")
            
            mlp_grad_norm = torch.nn.utils.clip_grad_norm_(qual_head.parameters(), max_norm=float('inf'))
            print(f"  MLP head gradient norm: {mlp_grad_norm:.6f}")
            
            lora_grads = []
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    if param.grad is not None:
                        lora_grads.append(param.grad.norm().item())
            
            if lora_grads:
                print(f"  LoRA gradient norm (avg): {np.mean(lora_grads):.6f}")
                print(f"  LoRA params with gradients: {len(lora_grads)}")
            else:
                print("  ⚠️  WARNING: No LoRA gradients found!")
            
            print(f"  Loss: {loss.item():.4f}\n")
            first_batch = False
        
        epoch_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        del all_embeddings, chunk_embeddings, embeddings_batch, scores, loss
        
        if (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
    
    avg_epoch_loss = epoch_loss / len(train_batches)
    print(f"\n📊 Epoch {epoch+1} Training Complete:")
    print(f"  Average Loss: {avg_epoch_loss:.4f}")
    
    # Evaluate on qual test set (ALL batches)
    print(f"\n🔍 Evaluating on qual test set ({len(test_batches)} batches)...")
    epoch_qual_metrics = evaluate_qual(test_batches, model, qual_head, desc=f"Epoch {epoch+1} Qual Eval")
    
    print(f"\n📊 Epoch {epoch+1} - Label_QUAL Test Results:")
    print(f"  Loss: {epoch_qual_metrics['loss']:.4f}")
    print(f"  Recall@1: {epoch_qual_metrics['recall@1']:.3f} ({epoch_qual_metrics['recall@1']*100:.1f}%)")
    print(f"  Recall@3: {epoch_qual_metrics['recall@3']:.3f} ({epoch_qual_metrics['recall@3']*100:.1f}%)")
    print(f"  Recall@5: {epoch_qual_metrics['recall@5']:.3f} ({epoch_qual_metrics['recall@5']*100:.1f}%)")
    print(f"  NDCG@1: {epoch_qual_metrics['ndcg@1']:.3f}")
    print(f"  NDCG@3: {epoch_qual_metrics['ndcg@3']:.3f}")
    print(f"  NDCG@5: {epoch_qual_metrics['ndcg@5']:.3f}")
    print(f"  Avg Rank: {epoch_qual_metrics['avg_rank']:.1f}")
    
    # Track metrics
    all_epoch_metrics.append({
        'epoch': epoch + 1,
        'train_loss': avg_epoch_loss,
        'qual_test': epoch_qual_metrics
    })
    
    # Save checkpoint
    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.save_pretrained(os.path.join(checkpoint_dir, "lora_adapters"))
    torch.save(qual_head.state_dict(), os.path.join(checkpoint_dir, "qual_head.pt"))
    
    print(f"\n✓ Saved checkpoint: {checkpoint_dir}")

# ========================================
# FINAL EVALUATION
# ========================================
print("\n" + "=" * 100)
print("FINAL EVALUATION")
print("=" * 100)

# Print training progression summary
print("\n📈 TRAINING PROGRESSION SUMMARY:")
print("=" * 80)
print(f"{'Epoch':<8} {'Train Loss':<12} {'Test Loss':<12} {'Recall@1':<12} {'Recall@5':<12} {'Avg Rank':<10}")
print("-" * 80)
for epoch_metrics in all_epoch_metrics:
    epoch = epoch_metrics['epoch']
    train_loss = epoch_metrics['train_loss']
    test_loss = epoch_metrics['qual_test']['loss']
    recall1 = epoch_metrics['qual_test']['recall@1']
    recall5 = epoch_metrics['qual_test']['recall@5']
    avg_rank = epoch_metrics['qual_test']['avg_rank']
    print(f"{epoch:<8} {train_loss:<12.4f} {test_loss:<12.4f} {recall1:<12.3f} {recall5:<12.3f} {avg_rank:<10.1f}")
print("=" * 80)

# Eval on qual test (ALL batches) - already done at end of last epoch
print(f"\n🔍 Final qual test results (from Epoch {NUM_EPOCHS})...")
qual_metrics = all_epoch_metrics[-1]['qual_test']

print(f"\n📊 FINAL - Label_QUAL Test (ALL {len(test_batches)} batches):")
print(f"  Loss: {qual_metrics['loss']:.4f}")
print(f"  Recall@1: {qual_metrics['recall@1']:.3f} ({qual_metrics['recall@1']*100:.1f}%)")
print(f"  Recall@3: {qual_metrics['recall@3']:.3f} ({qual_metrics['recall@3']*100:.1f}%)")
print(f"  Recall@5: {qual_metrics['recall@5']:.3f} ({qual_metrics['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {qual_metrics['ndcg@1']:.3f}")
print(f"  NDCG@3: {qual_metrics['ndcg@3']:.3f}")
print(f"  NDCG@5: {qual_metrics['ndcg@5']:.3f}")
print(f"  Avg Rank: {qual_metrics['avg_rank']:.1f}")

# Cross-eval on ENTIRE pref test
print(f"\n🔍 Cross-evaluating on ENTIRE label_pref test ({len(pref_test_batches):,} batches)...")
pref_metrics = evaluate_pref(pref_test_batches, model, qual_head, desc="Final Pref Eval (FULL)")

print(f"\n📊 FINAL - Label_PREF Test (cross-eval, FULL {len(pref_test_batches):,} batches):")
print(f"  Recall@1: {pref_metrics['recall@1']:.3f} ({pref_metrics['recall@1']*100:.1f}%)")
print(f"  Recall@3: {pref_metrics['recall@3']:.3f} ({pref_metrics['recall@3']*100:.1f}%)")
print(f"  Recall@5: {pref_metrics['recall@5']:.3f} ({pref_metrics['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {pref_metrics['ndcg@1']:.3f}")
print(f"  NDCG@3: {pref_metrics['ndcg@3']:.3f}")
print(f"  NDCG@5: {pref_metrics['ndcg@5']:.3f}")
print(f"  Avg Rank: {pref_metrics['avg_rank']:.1f}")

print("\n" + "=" * 100)
print("TRAINING COMPLETE!")
print("=" * 100)

# Save final model
print("\n📦 Saving final model (for Stage 2)...")

lora_path = os.path.join(OUTPUT_DIR, "lora_adapters_final")
model.save_pretrained(lora_path)
print(f"✓ Saved LoRA adapters: {lora_path}")

qual_head_path = os.path.join(OUTPUT_DIR, "qual_head_final.pt")
torch.save(qual_head.state_dict(), qual_head_path)
print(f"✓ Saved qual_head: {qual_head_path}")

tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
print(f"✓ Saved tokenizer")

print(f"\n✅ All files saved to: {OUTPUT_DIR}")
print(f"🎯 Use this checkpoint for Stage 2")

print("\n" + "=" * 100)
print("STAGE 1B COMPLETE!")
print("=" * 100)
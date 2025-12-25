#!/usr/bin/env python3
"""
Stage 1A: Preference Expert Training
=====================================
Train LLaMA encoder + LoRA + MLP pref_head on label_pref (with hard negatives)

Data Structure:
- 1 user per batch
- 1 positive (label_pref=1) + 49 hard negatives (label_pref=0, scores in [10,14))

Training:
- Finetune LoRA adapters + MLP pref_head on label_pref using InfoNCE

Eval:
- Recall@1,3,5 and NDCG@1,3,5 on label_pref test set
- Cross-eval: Recall@1,3,5 and NDCG@1,3,5 on label_qual test set (expect worse)
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
DATA_DIR = "stage1a_data"
TRAIN_CSV = f"../pipeline_output/{DATA_DIR}/train.csv"
TEST_CSV = f"../pipeline_output/{DATA_DIR}/test.csv"
VAL_CSV = f"../pipeline_output/{DATA_DIR}/val.csv"
QUAL_TEST_CSV = f"../pipeline_output/{DATA_DIR}/qual_test.csv"  # For cross-eval
USER_FILE = "../pipeline_output/Final_users.csv"
ITEM_FILE = "../pipeline_output/Final_items.csv"
NEGATIVES_PER_USER = 49
MAX_SEQ_LENGTH = 1000
PROCESS_CHUNK_SIZE = 10
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3

# Batch sampling (for faster training/testing)
NUM_TRAIN_BATCHES = 250
NUM_TEST_BATCHES = 250
NUM_QUAL_TEST_BATCHES = 50  # Keep as-is, ~19 available

# LoRA settings
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Output directory
OUTPUT_DIR = f"stage1a_output/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# DEVICE SETUP
# ========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 100)
print("STAGE 1A: PREFERENCE EXPERT TRAINING")
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

# Load train/test/val data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
val_df = pd.read_csv(VAL_CSV)

print(f"\n✓ Label_pref data (with hard negatives):")
print(f"  Train: {len(train_df):,} samples, {train_df['user'].nunique():,} users")
print(f"  Test: {len(test_df):,} samples, {test_df['user'].nunique():,} users")
print(f"  Val: {len(val_df):,} samples, {val_df['user'].nunique():,} users")
print(f"  Train positives: {(train_df['label_pref']==1).sum():,}")
print(f"  Test positives: {(test_df['label_pref']==1).sum():,}")

# Load qual test for cross-eval
qual_test_df = pd.read_csv(QUAL_TEST_CSV)
print(f"\n✓ Label_qual test (for cross-eval):")
print(f"  Samples: {len(qual_test_df):,}")
print(f"  Jobs: {qual_test_df['item'].nunique():,}")
print(f"  Positives: {(qual_test_df['label_qual']==1).sum():,}")

# ========================================
# PREPARE BATCHES
# ========================================
print("\n" + "=" * 100)
print("PREPARING BATCHES")
print("=" * 100)

def prepare_pref_batches(df, negatives_per_user):
    """
    Prepare batches for label_pref (user-centric).
    Each batch: 1 positive + 49 hard negatives
    """
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
    """
    Prepare batches for label_qual (job-centric).
    Each batch: 1 job with all its applicants
    """
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

train_batches = prepare_pref_batches(train_df, NEGATIVES_PER_USER)
test_batches = prepare_pref_batches(test_df, NEGATIVES_PER_USER)
val_batches = prepare_pref_batches(val_df, NEGATIVES_PER_USER)
qual_test_batches = prepare_qual_batches(qual_test_df)

# Sample to configured batch sizes for faster training
np.random.seed(42)
train_batches = [train_batches[i] for i in np.random.choice(len(train_batches), size=min(NUM_TRAIN_BATCHES, len(train_batches)), replace=False)]
test_batches = [test_batches[i] for i in np.random.choice(len(test_batches), size=min(NUM_TEST_BATCHES, len(test_batches)), replace=False)]
qual_test_batches = [qual_test_batches[i] for i in np.random.choice(len(qual_test_batches), size=min(NUM_QUAL_TEST_BATCHES, len(qual_test_batches)), replace=False)]

print(f"✓ Label_pref batches (user-centric, sampled):")
print(f"  Train: {len(train_batches):,}")
print(f"  Test: {len(test_batches):,}")
print(f"  Val: {len(val_batches):,}")
print(f"\n✓ Label_qual batches (job-centric for cross-eval, sampled):")
print(f"  Test: {len(qual_test_batches):,}")

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
# LOAD MODEL
# ========================================
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
pref_head = nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(device)

# Keep fp32 for proper gradient computation
print(f"\n✓ Preference head: {sum(p.numel() for p in pref_head.parameters()):,} params")

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

def evaluate_pref(batches, model, pref_head, desc="Evaluating"):
    """Evaluate on label_pref batches (user-centric)."""
    model.eval()
    pref_head.eval()
    
    all_ranks = []
    all_losses = []
    
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
            # Convert to fp32 for MLP head
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

def evaluate_qual(batches, model, pref_head, desc="Evaluating Qual"):
    """Evaluate on label_qual batches (job-centric) - cross-evaluation."""
    model.eval()
    pref_head.eval()
    
    all_ranks = []
    
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
            # Convert to fp32 for MLP head
            embeddings_batch = embeddings_batch.float()
            scores = pref_head(embeddings_batch).squeeze()
            
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

# print("\n🔍 Evaluating on label_pref test...")
# baseline_pref = evaluate_pref(test_batches, model, pref_head, desc="Baseline Pref")

# print(f"\n📊 Baseline - Label_PREF Test:")
# print(f"  Loss: {baseline_pref['loss']:.4f}")
# print(f"  Recall@1: {baseline_pref['recall@1']:.3f} ({baseline_pref['recall@1']*100:.1f}%)")
# print(f"  Recall@3: {baseline_pref['recall@3']:.3f} ({baseline_pref['recall@3']*100:.1f}%)")
# print(f"  Recall@5: {baseline_pref['recall@5']:.3f} ({baseline_pref['recall@5']*100:.1f}%)")
# print(f"  NDCG@1: {baseline_pref['ndcg@1']:.3f}")
# print(f"  NDCG@3: {baseline_pref['ndcg@3']:.3f}")
# print(f"  NDCG@5: {baseline_pref['ndcg@5']:.3f}")
# print(f"  Avg Rank: {baseline_pref['avg_rank']:.1f}/50")

# print("\n🔍 Cross-evaluating on label_qual test...")
# baseline_qual = evaluate_qual(qual_test_batches, model, pref_head, desc="Baseline Qual")

# print(f"\n📊 Baseline - Label_QUAL Test (cross-eval, expect bad):")
# print(f"  Recall@1: {baseline_qual['recall@1']:.3f} ({baseline_qual['recall@1']*100:.1f}%)")
# print(f"  Recall@3: {baseline_qual['recall@3']:.3f} ({baseline_qual['recall@3']*100:.1f}%)")
# print(f"  Recall@5: {baseline_qual['recall@5']:.3f} ({baseline_qual['recall@5']*100:.1f}%)")
# print(f"  NDCG@1: {baseline_qual['ndcg@1']:.3f}")
# print(f"  NDCG@3: {baseline_qual['ndcg@3']:.3f}")
# print(f"  NDCG@5: {baseline_qual['ndcg@5']:.3f}")
# print(f"  Avg Rank: {baseline_qual['avg_rank']:.1f}")

print("✓ Skipping baseline eval to start training immediately")

print("\n🧹 Cleaning memory before training...")
# ========================================
# GPU MEMORY CHECK & CLEAR BEFORE TRAINING
# ========================================
def print_gpu_usage(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU USAGE {tag}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
                capture_output=True, text=True
            )
            for i, line in enumerate(result.stdout.strip().split("\n")):
                used, total = map(int, line.split(","))
                print(f"  GPU {i}: {used/1024:.2f} GB / {total/1024:.2f} GB")
        except Exception as e:
            print(f"  ⚠️ Could not run nvidia-smi: {e}")

print("\n🧪 GPU memory BEFORE training:")
torch.cuda.empty_cache()
print_gpu_usage("before epoch 1")

# ========================================
# TRAINING LOOP
# ========================================
print("\n" + "=" * 100)
print("TRAINING")
print("=" * 100)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(pref_head.parameters()),
    lr=LEARNING_RATE
)

print(f"\nConfiguration:")
print(f"  Train batches: {len(train_batches):,}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*80}")
    
    model.train()
    pref_head.train()
    
    epoch_loss = 0
    pbar = tqdm(range(len(train_batches)), desc=f"Epoch {epoch+1}", ncols=100)
    
    # For gradient checking on first batch
    first_batch = True
    
    for batch_idx, batch in enumerate(train_batches):
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
        # Convert to fp32 for MLP head
        embeddings_batch = embeddings_batch.float()
        scores = pref_head(embeddings_batch).squeeze()
        
        # InfoNCE loss
        loss = info_nce_loss(scores)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(pref_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        # Check gradients on first batch
        if first_batch:
            print("\n🔍 GRADIENT CHECK (First Batch):")
            
            # Check MLP head gradients
            mlp_grad_norm = torch.nn.utils.clip_grad_norm_(pref_head.parameters(), max_norm=float('inf'))
            print(f"  MLP head gradient norm: {mlp_grad_norm:.6f}")
            
            # Check LoRA gradients
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
    
    # Save checkpoint
    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.save_pretrained(os.path.join(checkpoint_dir, "lora_adapters"))
    torch.save(pref_head.state_dict(), os.path.join(checkpoint_dir, "pref_head.pt"))
    
    print(f"✓ Saved checkpoint: {checkpoint_dir}")

# ========================================
# FINAL EVALUATION
# ========================================
print("\n" + "=" * 100)
print("FINAL EVALUATION")
print("=" * 100)

# Eval on pref test
print("\n🔍 Evaluating on label_pref test...")
pref_metrics = evaluate_pref(test_batches, model, pref_head, desc="Final Pref Eval")

print(f"\n📊 FINAL - Label_PREF Test:")
print(f"  Loss: {pref_metrics['loss']:.4f}")
print(f"  Recall@1: {pref_metrics['recall@1']:.3f} ({pref_metrics['recall@1']*100:.1f}%)")
print(f"  Recall@3: {pref_metrics['recall@3']:.3f} ({pref_metrics['recall@3']*100:.1f}%)")
print(f"  Recall@5: {pref_metrics['recall@5']:.3f} ({pref_metrics['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {pref_metrics['ndcg@1']:.3f}")
print(f"  NDCG@3: {pref_metrics['ndcg@3']:.3f}")
print(f"  NDCG@5: {pref_metrics['ndcg@5']:.3f}")
print(f"  Avg Rank: {pref_metrics['avg_rank']:.1f}/50")

# Cross-eval on qual test
print("\n🔍 Cross-evaluating on label_qual test...")
qual_metrics = evaluate_qual(qual_test_batches, model, pref_head, desc="Final Qual Eval")

print(f"\n📊 FINAL - Label_QUAL Test (cross-eval):")
print(f"  Recall@1: {qual_metrics['recall@1']:.3f} ({qual_metrics['recall@1']*100:.1f}%)")
print(f"  Recall@3: {qual_metrics['recall@3']:.3f} ({qual_metrics['recall@3']*100:.1f}%)")
print(f"  Recall@5: {qual_metrics['recall@5']:.3f} ({qual_metrics['recall@5']*100:.1f}%)")
print(f"  NDCG@1: {qual_metrics['ndcg@1']:.3f}")
print(f"  NDCG@3: {qual_metrics['ndcg@3']:.3f}")
print(f"  NDCG@5: {qual_metrics['ndcg@5']:.3f}")
print(f"  Avg Rank: {qual_metrics['avg_rank']:.1f}")

print("\n" + "=" * 100)
print("TRAINING COMPLETE!")
print("=" * 100)

# Save final model
print("\n📦 Saving final model (for Stage 1B)...")

lora_path = os.path.join(OUTPUT_DIR, "lora_adapters_final")
model.save_pretrained(lora_path)
print(f"✓ Saved LoRA adapters: {lora_path}")

pref_head_path = os.path.join(OUTPUT_DIR, "pref_head_final.pt")
torch.save(pref_head.state_dict(), pref_head_path)
print(f"✓ Saved pref_head: {pref_head_path}")

tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
print(f"✓ Saved tokenizer")

print(f"\n✅ All files saved to: {OUTPUT_DIR}")
print(f"🎯 Use this checkpoint for Stage 1B")

print("\n" + "=" * 100)
print("STAGE 1A COMPLETE!")
print("=" * 100)


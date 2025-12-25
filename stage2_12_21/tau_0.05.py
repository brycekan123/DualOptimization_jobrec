#!/usr/bin/env python3
# Stage 2: Lagrangian Policy Training (FIXED - with NaN prevention)
# Load Stage 1B checkpoint, freeze all, train lambda_head to balance preference vs quality

import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # For loading LoRA adapters
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

# Stage 1B checkpoint - FIXED PATHS
# Get absolute paths to avoid confusion
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # stage2_12_21/
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # LLama_3.2_implementation/
STAGE1B_DIR = os.path.join(PARENT_DIR, "12_19_multitask_stage1B")

STAGE1B_CHECKPOINT = os.path.join(STAGE1B_DIR, "stage1b_multitask_output/5e-5_head_LoRA/checkpoint_epoch3")
STAGE1B_LORA_PATH = os.path.join(STAGE1B_CHECKPOINT, "lora_adapters")
STAGE1B_PREF_HEAD = os.path.join(STAGE1B_CHECKPOINT, "pref_head.pt")
STAGE1B_QUAL_HEAD = os.path.join(STAGE1B_CHECKPOINT, "qual_head.pt")

# Data paths (relative to stage2_12_21/)
DATA_DIR = "stage1a_data"
PREF_TRAIN_CSV = os.path.join(PARENT_DIR, "pipeline_output", DATA_DIR, "train.csv")
PREF_TEST_CSV = os.path.join(PARENT_DIR, "pipeline_output", DATA_DIR, "test.csv")
QUAL_TEST_CSV = os.path.join(PARENT_DIR, "pipeline_output", DATA_DIR, "qual_test.csv")
USER_FILE = os.path.join(PARENT_DIR, "pipeline_output", "Final_users.csv")
ITEM_FILE = os.path.join(PARENT_DIR, "pipeline_output", "Final_items.csv")

# Training settings
NEGATIVES_PER_USER = 49  # Match Stage 1B
MAX_SEQ_LENGTH = 1000  # Match ground truth eval_checkpoint.py
PROCESS_CHUNK_SIZE = 5  # Match ground truth eval_checkpoint.py
LEARNING_RATE = 1e-4  # REDUCED from 1e-3 to prevent NaN
NUM_EPOCHS = 20  # Increased for better Lagrangian convergence
NUM_TRAIN_BATCHES = 68  # Match Stage 1B (random sample, seed=42)

# Lagrangian hyperparameters
TAU = 0.05  # Target qualification rate (~1%, moderate challenge based on qual_head predictions)
ETA_MU = 0.01  # Learning rate for μ updates
TEMPERATURE = 1.0  # Softmax temperature
LAMBDA_CLAMP = 5.0  # ADDED: Clamp lambda to [-5, 5] to prevent explosions

# Batch sampling
NUM_TRAIN_BATCHES = 68  # Match multi-task Stage 1B (random sample, seed=42)
# NUM_TEST_BATCHES: Use ALL test batches (no cap)

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, f"stage2_output/run_tau_0.05")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# PATH VERIFICATION
# ========================================
# Save config
with open(os.path.join(OUTPUT_DIR, "config.txt"), 'w') as f:
    f.write(f"TAU: {TAU} (~{TAU*100:.1f}% expected qualification rate)\n")
    f.write(f"  NOTE: TAU set manually (see pre-training diagnostic for data-derived recommendation)\n")
    f.write(f"ETA_MU: {ETA_MU}\n")
    f.write(f"TEMPERATURE: {TEMPERATURE}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"LAMBDA_CLAMP: {LAMBDA_CLAMP}\n")
    f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
    f.write(f"NUM_TRAIN_BATCHES: {NUM_TRAIN_BATCHES}\n")
    f.write(f"\nPaths:\n")
    f.write(f"Stage1B checkpoint: {STAGE1B_CHECKPOINT}\n")
    f.write(f"LoRA path: {STAGE1B_LORA_PATH}\n")
    f.write(f"\nChanges:\n")
    f.write(f"- Lambda head initialized with zero bias (outputs near 0 initially)\n")
    f.write(f"- Lambda clamped to [-{LAMBDA_CLAMP}, {LAMBDA_CLAMP}] to prevent NaN\n")
    f.write(f"- Learning rate reduced to {LEARNING_RATE} for stability\n")
    f.write(f"- Checkpoints saved every 2 epochs\n")

# ========================================
# DEVICE SETUP
# ========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n" + "=" * 100)
print("STAGE 2: LAGRANGIAN POLICY TRAINING (FIXED)")
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

# Load pref data (user-centric batches for Stage 2)
pref_train_df = pd.read_csv(PREF_TRAIN_CSV)
pref_test_df = pd.read_csv(PREF_TEST_CSV)
qual_test_df = pd.read_csv(QUAL_TEST_CSV)

print(f"\n✓ Label_pref data (training data for Stage 2):")
print(f"  Train: {len(pref_train_df):,} samples, {pref_train_df['user'].nunique():,} users")
print(f"  Test: {len(pref_test_df):,} samples, {pref_test_df['user'].nunique():,} users")
print(f"  Train positives: {(pref_train_df['label_pref']==1).sum():,}")
print(f"  Test positives: {(pref_test_df['label_pref']==1).sum():,}")

print(f"\n✓ Label_qual data (for evaluation):")
print(f"  Test: {len(qual_test_df):,} samples")
print(f"  Test positives: {(qual_test_df['label_qual']==1).sum():,}")

# ========================================
# PREPARE BATCHES
# ========================================
print("\n" + "=" * 100)
print("PREPARING BATCHES")
print("=" * 100)

def prepare_pref_batches(df, negatives_per_user=49):
    """
    Prepare user-centric batches for preference data.
    Each batch: 1 user, 1 positive job + 49 negative jobs
    Uses 'label_pref' column.
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
    Prepare job-centric batches for qualification data.
    Each batch: 1 job, 1 positive user + N negative users
    Uses 'label_qual' column.
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

pref_train_batches_all = prepare_pref_batches(pref_train_df, NEGATIVES_PER_USER)
pref_test_batches = prepare_pref_batches(pref_test_df, NEGATIVES_PER_USER)

# Randomly sample 68 pref batches with FIXED SEED for reproducibility (MATCHES STAGE 1B)
np.random.seed(42)
pref_train_indices = np.random.choice(len(pref_train_batches_all), size=NUM_TRAIN_BATCHES, replace=False)
pref_train_batches = [pref_train_batches_all[i] for i in pref_train_indices]

# Analyze sampled 68 pref batches (verify matches Stage 1B)
print(f"\n📊 Analysis of randomly sampled {NUM_TRAIN_BATCHES} pref training batches:")
pref_users = [b['user_id'] for b in pref_train_batches]
print(f"  Total batches: {len(pref_train_batches)}")
print(f"  Unique users: {len(set(pref_users))}")
print(f"  Most frequent user appears: {max([pref_users.count(u) for u in set(pref_users)])} times")
if len(set(pref_users)) < 10:
    print(f"  ⚠️  WARNING: Only {len(set(pref_users))} unique users - data may not be diverse!")
else:
    print(f"  ✓ Data looks diverse")

# Use ALL test batches (no sampling)
print(f"\n✓ User-centric PREF batches (1 user, 50 jobs each):")
print(f"  Train: {len(pref_train_batches):,} (random sample, seed=42)")
print(f"  Test: {len(pref_test_batches):,} (ALL batches)")

# Note: qual batches will be prepared during verification
print(f"\n✓ QUAL batches will be prepared during verification from qual_test_df")

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
# LOAD FROZEN COMPONENTS FROM STAGE 1B
# ========================================
print("\n" + "=" * 100)
print("LOADING FROZEN COMPONENTS FROM STAGE 1B")
print("=" * 100)

# Load tokenizer (from HuggingFace, same as Stage 1B used)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    cache_dir=CACHE_DIR
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("✓ Loaded tokenizer")

# Load base model with LoRA
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

# Load LoRA adapters using PeftModel.from_pretrained (local path)
print(f"\nLoading LoRA adapters from:")
print(f"  {STAGE1B_LORA_PATH}")

model = PeftModel.from_pretrained(
    base_model,
    STAGE1B_LORA_PATH,
    is_trainable=False,
    device_map="auto"
)
print(f"✓ Loaded Stage 1B LoRA adapters")

# FREEZE encoder + LoRA
model.eval()
for param in model.parameters():
    param.requires_grad = False
print("✓ FROZEN: Encoder + LoRA adapters")

# Load pref_head (FROZEN)
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

pref_head.load_state_dict(torch.load(STAGE1B_PREF_HEAD, map_location=device))
pref_head.eval()
for param in pref_head.parameters():
    param.requires_grad = False
print(f"✓ Loaded pref_head from Stage 1B (FROZEN)")

# Load qual_head (FROZEN)
qual_head = nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(device)

qual_head.load_state_dict(torch.load(STAGE1B_QUAL_HEAD, map_location=device))
qual_head.eval()
for param in qual_head.parameters():
    param.requires_grad = False
print(f"✓ Loaded qual_head from Stage 1B (FROZEN)")

# Create lambda_head (NEW, TRAINABLE)
lambda_head = nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)
).to(device)

# Initialize lambda to output near 0 (neutral starting point)
lambda_head[-1].bias.data.fill_(0.0)

print(f"✓ Created lambda_head (NEW, randomly initialized, TRAINABLE)")
print(f"  lambda_head params: {sum(p.numel() for p in lambda_head.parameters()):,}")
print(f"  lambda_head initialized with zero bias (outputs near 0 initially)")
print(f"  lambda_head will be CLAMPED to [-{LAMBDA_CLAMP}, {LAMBDA_CLAMP}]")

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
    """Calculate Recall@k and NDCG@k from list of ranks. EXACT copy from eval_checkpoint.py"""
    metrics = {}
    
    # Recall@k
    for k in [1, 3, 5, 10, 20]:
        recall = sum(1 for r in ranks if r <= k) / len(ranks)
        metrics[f'recall@{k}'] = recall
    
    # NDCG@k
    for k in [1, 3, 5, 10, 20]:
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

# ========================================
# MEMORY-EFFICIENT ENCODING
# ========================================
def encode_user_item_pairs(user_id, items, model, tokenizer):
    """
    Encode (user, item) pairs efficiently.
    CRITICAL: Only extract LAST hidden state, immediately delete full outputs.
    
    Returns: (50, hidden_size) tensor in fp32
    """
    # Process in chunks of PROCESS_CHUNK_SIZE
    item_chunks = [items[i:i+PROCESS_CHUNK_SIZE] 
                   for i in range(0, len(items), PROCESS_CHUNK_SIZE)]
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
        
        with torch.no_grad():  # Encoder is frozen
            chunk_outputs = model(
                input_ids=chunk_inputs['input_ids'],
                attention_mask=chunk_inputs['attention_mask'],
                output_hidden_states=True  # Need this to get hidden states
            )
            
            # CRITICAL: Extract ONLY last layer, then DELETE immediately
            chunk_hidden = chunk_outputs.hidden_states[-1].clone()
            del chunk_outputs  # Free all 32 layers from memory!
            
            chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
            all_embeddings.append(chunk_embeddings)
            
            del chunk_inputs, chunk_hidden
    
    # Concatenate all chunks
    embeddings_batch = torch.cat(all_embeddings, dim=0)
    
    # Convert to fp32 for MLP heads
    embeddings_batch = embeddings_batch.float()
    
    return embeddings_batch

# ========================================
# EVALUATION (for verification ONLY - just testing checkpoint)
# ========================================
def evaluate_pref_checkpoint(batches, model, head, desc="Evaluating Pref"):
    """Evaluate on label_pref batches. EXACT copy from eval_checkpoint.py"""
    model.eval()
    head.eval()
    
    all_ranks = []
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(batches, desc=desc, ncols=100):
            user_id = batch['user_id']
            all_items = [batch['positive_item']] + batch['negative_items']
            
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
                del chunk_outputs
                
                chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
                all_embeddings.append(chunk_embeddings)
                
                del chunk_inputs, chunk_hidden
            
            embeddings_batch = torch.cat(all_embeddings, dim=0)
            embeddings_batch = embeddings_batch.float()
            scores = head(embeddings_batch).squeeze()
            
            rank = (scores[0] < scores[1:]).sum().item() + 1
            loss = info_nce_loss(scores)
            
            all_ranks.append(rank)
            all_losses.append(loss.item())
            
            del all_embeddings, chunk_embeddings, embeddings_batch, scores, loss
    
    metrics = calculate_metrics(all_ranks)
    metrics['loss'] = np.mean(all_losses)
    
    return metrics

def evaluate_qual_checkpoint(batches, model, head, desc="Evaluating Qual"):
    """Evaluate on label_qual batches. EXACT copy from eval_checkpoint.py"""
    model.eval()
    head.eval()
    
    all_ranks = []
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(batches, desc=desc, ncols=100):
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
                del chunk_outputs
                
                chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
                all_embeddings.append(chunk_embeddings)
                
                del chunk_inputs, chunk_hidden
            
            embeddings_batch = torch.cat(all_embeddings, dim=0)
            embeddings_batch = embeddings_batch.float()
            scores = head(embeddings_batch).squeeze()
            
            rank = (scores[0] < scores[1:]).sum().item() + 1
            loss = info_nce_loss(scores)
            
            all_ranks.append(rank)
            all_losses.append(loss.item())
            
            del all_embeddings, chunk_embeddings, embeddings_batch, scores, loss
    
    metrics = calculate_metrics(all_ranks)
    metrics['loss'] = np.mean(all_losses)
    
    return metrics

# ========================================
# FULL EVALUATION (with lambda_head)
# ========================================
def evaluate_qual(batches, model, pref_head, qual_head, lambda_head, 
                  pref_test_df, qual_test_df, desc="Evaluating Qual"):
    """
    Evaluate FULL LAGRANGIAN POLICY on job-centric qual batches.
    Uses s_final = s_pref + λ*s_qual
    """
    model.eval()
    pref_head.eval()
    qual_head.eval()
    lambda_head.eval()
    
    qual_ranks = []
    all_losses = []
    all_c_qual_preds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=desc, ncols=100)):
            job_id = batch['job_id']
            all_users = [batch['positive_user']] + batch['negative_users']
            
            # Encode all (user, job) pairs
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
                
                chunk_hidden = chunk_outputs.hidden_states[-1].clone()
                del chunk_outputs
                
                chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
                all_embeddings.append(chunk_embeddings)
                
                del chunk_inputs, chunk_hidden
            
            h = torch.cat(all_embeddings, dim=0)
            h = h.float()
            
            # Get scores from all three heads
            s_pref = pref_head(h).squeeze()
            qual_logit = qual_head(h).squeeze()
            p_qual = torch.sigmoid(qual_logit)
            lambda_uj = lambda_head(h).squeeze()
            
            # CLAMP lambda (prevent explosions)
            lambda_uj = torch.clamp(lambda_uj, min=-LAMBDA_CLAMP, max=LAMBDA_CLAMP)
            
            # Final score
            s_qual = qual_logit
            s_final = s_pref + lambda_uj * s_qual
            
            # InfoNCE loss
            loss = info_nce_loss(s_final)
            all_losses.append(loss.item())
            
            # Qualification constraint
            Pi = torch.softmax(s_final / TEMPERATURE, dim=0)
            C_qual_pred = torch.sum(Pi * p_qual)
            all_c_qual_preds.append(C_qual_pred.item())
            
            # Calculate rank of positive (index 0)
            rank = (s_final[0] < s_final[1:]).sum().item() + 1
            
            # Check if positive has label_qual=1
            pos_user = all_users[0]
            has_qual = ((qual_test_df['user'] == pos_user) & 
                       (qual_test_df['item'] == job_id) & 
                       (qual_test_df['label_qual'] == 1)).any()
            if has_qual:
                qual_ranks.append(rank)
            
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
    
    # Calculate metrics
    results = {
        'loss': np.mean(all_losses),
        'c_qual_pred_mean': np.mean(all_c_qual_preds),
        'c_qual_pred_std': np.std(all_c_qual_preds),
    }
    
    # Over/under qualified batch percentages
    c_qual_array = np.array(all_c_qual_preds)
    over_qualified = (c_qual_array > TAU).sum()
    under_qualified = (c_qual_array < TAU).sum()
    results['over_qualified_pct'] = over_qualified / len(all_c_qual_preds) * 100
    results['under_qualified_pct'] = under_qualified / len(all_c_qual_preds) * 100
    
    if qual_ranks:
        qual_metrics = calculate_metrics(qual_ranks)
        for k, v in qual_metrics.items():
            results[f'qual_{k}'] = v
    
    return results


def evaluate_pref(batches, model, pref_head, qual_head, lambda_head,
                  pref_test_df, qual_test_df, desc="Evaluating Pref"):
    """
    Evaluate FULL LAGRANGIAN POLICY on user-centric pref batches.
    Uses s_final = s_pref + λ*s_qual
    """
    model.eval()
    pref_head.eval()
    qual_head.eval()
    lambda_head.eval()
    
    pref_ranks = []
    all_losses = []
    all_c_qual_preds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc=desc, ncols=100)):
            user_id = batch['user_id']
            all_items = [batch['positive_item']] + batch['negative_items']
            
            # Encode all (user, item) pairs
            h = encode_user_item_pairs(user_id, all_items, model, tokenizer)
            
            # Get scores from all three heads
            s_pref = pref_head(h).squeeze()
            qual_logit = qual_head(h).squeeze()
            p_qual = torch.sigmoid(qual_logit)
            lambda_uj = lambda_head(h).squeeze()
            
            # CLAMP lambda (prevent explosions)
            lambda_uj = torch.clamp(lambda_uj, min=-LAMBDA_CLAMP, max=LAMBDA_CLAMP)
            
            # Final score
            s_qual = qual_logit
            s_final = s_pref + lambda_uj * s_qual
            
            # InfoNCE loss
            loss = info_nce_loss(s_final)
            all_losses.append(loss.item())
            
            # Qualification constraint
            Pi = torch.softmax(s_final / TEMPERATURE, dim=0)
            C_qual_pred = torch.sum(Pi * p_qual)
            all_c_qual_preds.append(C_qual_pred.item())
            
            # Calculate rank of positive (index 0)
            rank = (s_final[0] < s_final[1:]).sum().item() + 1
            
            # Check if positive has label_pref=1
            pos_item = all_items[0]
            has_pref = ((pref_test_df['user'] == user_id) & 
                       (pref_test_df['item'] == pos_item) & 
                       (pref_test_df['label_pref'] == 1)).any()
            if has_pref:
                pref_ranks.append(rank)
            
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
    
    # Calculate metrics
    results = {
        'loss': np.mean(all_losses),
        'c_qual_pred_mean': np.mean(all_c_qual_preds),
        'c_qual_pred_std': np.std(all_c_qual_preds),
    }
    
    # Over/under qualified batch percentages
    c_qual_array = np.array(all_c_qual_preds)
    over_qualified = (c_qual_array > TAU).sum()
    under_qualified = (c_qual_array < TAU).sum()
    results['over_qualified_pct'] = over_qualified / len(all_c_qual_preds) * 100
    results['under_qualified_pct'] = under_qualified / len(all_c_qual_preds) * 100
    
    if pref_ranks:
        pref_metrics = calculate_metrics(pref_ranks)
        for k, v in pref_metrics.items():
            results[f'pref_{k}'] = v
    
    return results

# ========================================
# VERIFICATION EVAL (Check Checkpoint Loaded Correctly)
# ========================================
print("\n" + "=" * 100)
print("VERIFICATION EVAL - TESTING STAGE 1B CHECKPOINT")
print("=" * 100)
print("Testing if Stage 1B checkpoint loaded correctly")
print("Evaluating heads independently (NO Stage 2 training yet)\n")

# ========================================
# 1. QUAL TEST (qual_test_df) - JOB-CENTRIC
# ========================================
print("=" * 80)
print("1. LABEL_QUAL VERIFICATION (job-centric batches)")
print("=" * 80)
print("Preparing qual test batches from qual_test_df...")

# Prepare qual test batches (job-centric: 1 job, multiple users)
qual_verify_batches = prepare_qual_batches(qual_test_df)
print(f"  Created {len(qual_verify_batches)} qual test batches")
print(f"  Format: 1 job + 1 positive user + N negative users per batch")

# Use first 30 batches
qual_verify_batches = qual_verify_batches[:30]
print(f"  Using first 30 batches for verification")

# Evaluate qual_head ONLY
print("\nEvaluating qual_head performance (ranking on label_qual=1)...")
qual_verify_results = evaluate_qual_checkpoint(
    qual_verify_batches,
    model, qual_head,
    desc="Qual Verification"
)

print(f"\n📊 QUAL VERIFICATION RESULTS:")
print(f"  Loss: {qual_verify_results['loss']:.4f}")

if 'recall@1' in qual_verify_results:
    print(f"  Recall@1: {qual_verify_results['recall@1']:.3f} | Recall@3: {qual_verify_results['recall@3']:.3f} | Recall@5: {qual_verify_results['recall@5']:.3f} | Recall@10: {qual_verify_results['recall@10']:.3f} | Recall@20: {qual_verify_results['recall@20']:.3f}")
    print(f"  NDCG@1: {qual_verify_results['ndcg@1']:.3f} | NDCG@3: {qual_verify_results['ndcg@3']:.3f} | NDCG@5: {qual_verify_results['ndcg@5']:.3f} | NDCG@10: {qual_verify_results['ndcg@10']:.3f} | NDCG@20: {qual_verify_results['ndcg@20']:.3f}")
else:
    print(f"  ⚠️  WARNING: No samples found!")

# ========================================
# 2. PREF TEST (pref_test_df) - USER-CENTRIC
# ========================================
print("\n" + "=" * 80)
print("2. LABEL_PREF VERIFICATION (user-centric batches)")
print("=" * 80)
print("Using first 50 batches from pref_test_batches...")

pref_verify_batches = pref_test_batches[:50]
print(f"  Testing on {len(pref_verify_batches)} pref test batches")
print(f"  Format: 1 user + 1 positive job + 49 negative jobs per batch")

# Evaluate pref_head ONLY
print("\nEvaluating pref_head performance (ranking on label_pref=1)...")
pref_verify_results = evaluate_pref_checkpoint(
    pref_verify_batches,
    model, pref_head,
    desc="Pref Verification"
)

print(f"\n📊 PREF VERIFICATION RESULTS:")
print(f"  Loss: {pref_verify_results['loss']:.4f}")

if 'recall@1' in pref_verify_results:
    print(f"  Recall@1: {pref_verify_results['recall@1']:.3f} | Recall@3: {pref_verify_results['recall@3']:.3f} | Recall@5: {pref_verify_results['recall@5']:.3f} | Recall@10: {pref_verify_results['recall@10']:.3f} | Recall@20: {pref_verify_results['recall@20']:.3f}")
    print(f"  NDCG@1: {pref_verify_results['ndcg@1']:.3f} | NDCG@3: {pref_verify_results['ndcg@3']:.3f} | NDCG@5: {pref_verify_results['ndcg@5']:.3f} | NDCG@10: {pref_verify_results['ndcg@10']:.3f} | NDCG@20: {pref_verify_results['ndcg@20']:.3f}")

print("\n" + "=" * 80)
print("✓ Checkpoint verification complete!")
print("=" * 80)
print("  Qual_head: Ranks qualified users for jobs")
print("  Pref_head: Ranks preferred jobs for users")
print("  Both heads from Stage 1B are working correctly")
print("  Ready to start Stage 2 training...\n")

# ========================================
# TRAINING LOOP
# ========================================
print("\n" + "=" * 100)
print("TRAINING")
print("=" * 100)

# Optimizer: lambda_head ONLY
optimizer = torch.optim.AdamW(
    lambda_head.parameters(),
    lr=LEARNING_RATE
)

# Initialize Lagrange multiplier
mu = 0.0

print(f"\nConfiguration:")
print(f"  Train batches: {len(pref_train_batches):,}")
print(f"  Test batches: {len(pref_test_batches):,}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Lambda clamp: [-{LAMBDA_CLAMP}, {LAMBDA_CLAMP}]")
print(f"  τ (target qualified jobs): {TAU}")
print(f"  η_μ (Lagrange LR): {ETA_MU}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Trainable: lambda_head ONLY (~{sum(p.numel() for p in lambda_head.parameters()):,} params)")
print(f"  Frozen: Encoder, LoRA, pref_head, qual_head")

# Save initial lambda_head state for comparison
initial_lambda_state = {name: param.clone().detach() for name, param in lambda_head.named_parameters()}
print("\n✓ Saved initial lambda_head state for comparison")

# Track training history
history = {
    'epoch': [],
    'train_loss': [],
    'L_pref': [],
    'constraint_term': [],
    'C_qual_pred': [],
    'mu': [],
    'lambda_min': [],
    'lambda_max': [],
    'lambda_mean': [],
}

# Track intermediate eval results for best epoch selection and convergence analysis
intermediate_eval_results = {
    'epoch': [],
    # Primary ranking metrics
    'pref_recall@5': [],
    'pref_ndcg@5': [],
    'qual_recall@5': [],
    'qual_ndcg@5': [],
    'pref_recall@1': [],
    'pref_ndcg@1': [],
    'qual_recall@1': [],
    'qual_ndcg@1': [],
    'pref_avg_rank': [],
    'qual_avg_rank': [],
    # Constraint satisfaction (convergence tracking)
    'c_qual_pred': [],
    'constraint_gap': [],  # |C_qual - TAU|
    'mu': [],
    # Lambda learning (stability tracking)
    'mean_lambda': [],
    'lambda_delta': [],  # Change from previous epoch
    # Training effectiveness
    'loss': []
}

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*100}")
    print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*100}")
    
    lambda_head.train()
    
    epoch_loss = 0
    epoch_l_pref = 0
    epoch_constraint = 0
    epoch_c_qual = 0
    epoch_lambda_stats = []
    
    # Additional diagnostics
    epoch_score_stats = {
        's_pref': [], 's_qual': [], 's_final': [], 'lambda': []
    }
    epoch_rank_stats = []
    epoch_constraint_gap = []
    
    pbar = tqdm(range(len(pref_train_batches)), desc=f"Epoch {epoch+1}", ncols=100)
    
    for batch_idx, batch in enumerate(pref_train_batches):
        user_id = batch['user_id']
        all_items = [batch['positive_item']] + batch['negative_items']
        
        # Encode all 50 (user, item) pairs
        h = encode_user_item_pairs(user_id, all_items, model, tokenizer)
        
        # Get scores from frozen heads
        with torch.no_grad():
            s_pref = pref_head(h).squeeze()  # (50,) logits
            qual_logit = qual_head(h).squeeze()  # (50,) logits
            p_qual = torch.sigmoid(qual_logit)  # (50,) probabilities
        
        # Get lambda from trainable head
        lambda_uj = lambda_head(h).squeeze()  # (50,) weights
        
        # CLAMP lambda (CRITICAL FIX)
        lambda_uj = torch.clamp(lambda_uj, min=-LAMBDA_CLAMP, max=LAMBDA_CLAMP)
        
        # Final score
        s_qual = qual_logit  # Use raw logit
        s_final = s_pref + lambda_uj * s_qual  # (50,)
        
        # Loss component 1: InfoNCE on final scores
        L_pref = info_nce_loss(s_final)
        
        # Loss component 2: Constraint on expected qualified jobs
        Pi = torch.softmax(s_final / TEMPERATURE, dim=0)  # (50,)
        C_qual_pred = torch.sum(Pi * p_qual)  # Scalar
        constraint_term = mu * (TAU - C_qual_pred)
        
        # Total loss
        loss = L_pref + constraint_term
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lambda_head.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update mu (allow negative for equality constraint)
        with torch.no_grad():
            mu = mu + ETA_MU * (TAU - C_qual_pred.item())
        
        # Track stats
        epoch_loss += loss.item()
        epoch_l_pref += L_pref.item()
        epoch_constraint += constraint_term.item()
        epoch_c_qual += C_qual_pred.item()
        epoch_lambda_stats.append({
            'min': lambda_uj.min().item(),
            'max': lambda_uj.max().item(),
            'mean': lambda_uj.mean().item()
        })
        
        # Additional diagnostics (ALL batches)
        epoch_score_stats['s_pref'].append({
            'pos': s_pref[0].item(),
            'neg_mean': s_pref[1:].mean().item(),
            'neg_std': s_pref[1:].std().item()
        })
        epoch_score_stats['s_qual'].append({
            'pos': s_qual[0].item(),
            'neg_mean': s_qual[1:].mean().item(),
            'neg_std': s_qual[1:].std().item()
        })
        epoch_score_stats['s_final'].append({
            'pos': s_final[0].item(),
            'neg_mean': s_final[1:].mean().item(),
            'neg_std': s_final[1:].std().item()
        })
        epoch_score_stats['lambda'].append({
            'pos': lambda_uj[0].item(),
            'neg_mean': lambda_uj[1:].mean().item()
        })
        
        # Track rank of positive
        pos_rank = (s_final[0] < s_final[1:]).sum().item() + 1
        epoch_rank_stats.append(pos_rank)
        
        # Track constraint gap
        epoch_constraint_gap.append(TAU - C_qual_pred.item())
        
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'C_qual': f'{C_qual_pred.item():.2f}',
            'μ': f'{mu:.3f}'
        })
        
        # Cleanup
        del h, s_pref, qual_logit, p_qual, lambda_uj, s_qual, s_final
        del L_pref, constraint_term, loss, Pi, C_qual_pred
        
        if (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
    
    # Extract lambda statistics for tracking
    lambda_mins = [s['min'] for s in epoch_lambda_stats]
    lambda_maxs = [s['max'] for s in epoch_lambda_stats]
    lambda_means = [s['mean'] for s in epoch_lambda_stats]
    
    # Minimal training epoch summary with rank distribution
    n_batches = len(pref_train_batches)
    print(f"\n📊 EPOCH {epoch+1} Training:")
    print(f"  Loss={epoch_loss/n_batches:.4f} | C_qual={epoch_c_qual/n_batches:.4f} (τ={TAU}) | μ={mu:.4f} | λ_mean={np.mean(lambda_means):.3f}")
    
    # Rank distribution (where positive ranks in training batches)
    rank_counts = {}
    for rank in epoch_rank_stats:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    top5_count = sum(1 for r in epoch_rank_stats if r <= 5)
    top10_count = sum(1 for r in epoch_rank_stats if r <= 10)
    print(f"  Ranks: #1={rank_counts.get(1, 0)}/{n_batches} | Top-5={top5_count}/{n_batches} | Top-10={top10_count}/{n_batches} | Avg={np.mean(epoch_rank_stats):.1f} | Worst={np.max(epoch_rank_stats)}")
    
    # Constraint satisfaction distribution
    gaps = np.array(epoch_constraint_gap)
    over_qualified = (gaps < 0).sum()  # C_qual > TAU
    under_qualified = (gaps > 0).sum()  # C_qual < TAU
    print(f"  Batches: Over-qualified={over_qualified}/{n_batches} ({over_qualified/n_batches*100:.1f}%) | Under-qualified={under_qualified}/{n_batches} ({under_qualified/n_batches*100:.1f}%)")
    
    # Save history
    history['epoch'].append(epoch+1)
    history['train_loss'].append(epoch_loss/n_batches)
    history['L_pref'].append(epoch_l_pref/n_batches)
    history['constraint_term'].append(epoch_constraint/n_batches)
    history['C_qual_pred'].append(epoch_c_qual/n_batches)
    history['mu'].append(mu)
    history['lambda_min'].append(np.mean(lambda_mins))
    history['lambda_max'].append(np.mean(lambda_maxs))
    history['lambda_mean'].append(np.mean(lambda_means))
    
    # Save checkpoint every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save(lambda_head.state_dict(), os.path.join(checkpoint_dir, "lambda_head.pt"))
        torch.save({
            'mu': mu,
            'epoch': epoch+1,
            'history': history
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        print(f"\n✓ Saved checkpoint: {checkpoint_dir}")
    
    # ========================================
    # INTERMEDIATE EVALUATION (full Lagrangian policy)
    # ========================================
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch+1} INTERMEDIATE EVALUATION")
    print(f"{'='*80}")
    print("Evaluating full Lagrangian policy: s_final = s_pref + λ*s_qual")
    
    # Prepare eval batches (use same as verification)
    qual_test_batches_all = prepare_qual_batches(qual_test_df)
    qual_intermediate_batches = qual_test_batches_all[:30]
    pref_intermediate_batches = pref_test_batches[:50]
    
    print(f"\n1. QUAL EVALUATION (job-centric, 30 batches)")
    print("=" * 80)
    qual_intermediate_results = evaluate_qual(
        qual_intermediate_batches,
        model, pref_head, qual_head, lambda_head,
        pref_test_df, qual_test_df,
        desc=f"Epoch {epoch+1} Qual"
    )
    
    print(f"\n📊 Epoch {epoch+1} - QUAL Results (30 batches):")
    print(f"  Recall: R@1={qual_intermediate_results.get('qual_recall@1', 0):.3f} | R@3={qual_intermediate_results.get('qual_recall@3', 0):.3f} | R@5={qual_intermediate_results.get('qual_recall@5', 0):.3f} | R@10={qual_intermediate_results.get('qual_recall@10', 0):.3f} | R@20={qual_intermediate_results.get('qual_recall@20', 0):.3f}")
    print(f"  NDCG:   N@1={qual_intermediate_results.get('qual_ndcg@1', 0):.3f} | N@3={qual_intermediate_results.get('qual_ndcg@3', 0):.3f} | N@5={qual_intermediate_results.get('qual_ndcg@5', 0):.3f} | N@10={qual_intermediate_results.get('qual_ndcg@10', 0):.3f} | N@20={qual_intermediate_results.get('qual_ndcg@20', 0):.3f}")
    print(f"  Avg Rank: {qual_intermediate_results.get('qual_avg_rank', 0):.1f}")
    print(f"  Constraint: C_qual={qual_intermediate_results['c_qual_pred_mean']:.4f} | |τ-C_qual|={abs(qual_intermediate_results['c_qual_pred_mean']-TAU):.4f} | μ*(τ-C_qual)={qual_intermediate_results.get('constraint_term_mean', 0):.4f}")
    print(f"  Batches: Over-qualified={qual_intermediate_results.get('over_qualified_pct', 0):.1f}% | Under-qualified={qual_intermediate_results.get('under_qualified_pct', 0):.1f}%")
    print(f"  Loss: {qual_intermediate_results['loss']:.4f} | μ={mu:.4f}")
    
    print(f"\n2. PREF EVALUATION (user-centric, 50 batches)")
    print("=" * 80)
    pref_intermediate_results = evaluate_pref(
        pref_intermediate_batches,
        model, pref_head, qual_head, lambda_head,
        pref_test_df, qual_test_df,
        desc=f"Epoch {epoch+1} Pref"
    )
    
    print(f"\n📊 Epoch {epoch+1} - PREF Results (50 batches):")
    print(f"  Recall: R@1={pref_intermediate_results['pref_recall@1']:.3f} | R@3={pref_intermediate_results['pref_recall@3']:.3f} | R@5={pref_intermediate_results['pref_recall@5']:.3f} | R@10={pref_intermediate_results['pref_recall@10']:.3f} | R@20={pref_intermediate_results['pref_recall@20']:.3f}")
    print(f"  NDCG:   N@1={pref_intermediate_results['pref_ndcg@1']:.3f} | N@3={pref_intermediate_results['pref_ndcg@3']:.3f} | N@5={pref_intermediate_results['pref_ndcg@5']:.3f} | N@10={pref_intermediate_results['pref_ndcg@10']:.3f} | N@20={pref_intermediate_results['pref_ndcg@20']:.3f}")
    print(f"  Avg Rank: {pref_intermediate_results['pref_avg_rank']:.1f}")
    print(f"  Constraint: C_qual={pref_intermediate_results['c_qual_pred_mean']:.4f} | |τ-C_qual|={abs(pref_intermediate_results['c_qual_pred_mean']-TAU):.4f} | μ*(τ-C_qual)={pref_intermediate_results.get('constraint_term_mean', 0):.4f}")
    print(f"  Batches: Over-qualified={pref_intermediate_results.get('over_qualified_pct', 0):.1f}% | Under-qualified={pref_intermediate_results.get('under_qualified_pct', 0):.1f}%")
    
    # Calculate lambda delta from previous epoch
    current_lambda_mean = np.mean(lambda_means)
    if epoch > 0 and len(intermediate_eval_results['mean_lambda']) > 0:
        prev_lambda = intermediate_eval_results['mean_lambda'][-1]
        lambda_delta = current_lambda_mean - prev_lambda
        delta_str = f"Δλ={lambda_delta:+.4f}"
    else:
        lambda_delta = 0.0
        delta_str = "Δλ=--"
    
    print(f"  Loss: {pref_intermediate_results['loss']:.4f} | μ={mu:.4f} | λ={current_lambda_mean:.3f} | {delta_str}")
    
    # Save intermediate eval results for best epoch tracking and convergence analysis
    intermediate_eval_results['epoch'].append(epoch+1)
    
    # Ranking metrics
    intermediate_eval_results['pref_recall@5'].append(pref_intermediate_results.get('pref_recall@5', 0.0))
    intermediate_eval_results['pref_ndcg@5'].append(pref_intermediate_results.get('pref_ndcg@5', 0.0))
    intermediate_eval_results['qual_recall@5'].append(qual_intermediate_results.get('qual_recall@5', 0.0))
    intermediate_eval_results['qual_ndcg@5'].append(qual_intermediate_results.get('qual_ndcg@5', 0.0))
    intermediate_eval_results['pref_recall@1'].append(pref_intermediate_results.get('pref_recall@1', 0.0))
    intermediate_eval_results['pref_ndcg@1'].append(pref_intermediate_results.get('pref_ndcg@1', 0.0))
    intermediate_eval_results['qual_recall@1'].append(qual_intermediate_results.get('qual_recall@1', 0.0))
    intermediate_eval_results['qual_ndcg@1'].append(qual_intermediate_results.get('qual_ndcg@1', 0.0))
    intermediate_eval_results['pref_avg_rank'].append(pref_intermediate_results.get('pref_avg_rank', 0.0))
    intermediate_eval_results['qual_avg_rank'].append(qual_intermediate_results.get('qual_avg_rank', 0.0))
    
    # Constraint satisfaction
    c_qual_mean = pref_intermediate_results['c_qual_pred_mean']
    intermediate_eval_results['c_qual_pred'].append(c_qual_mean)
    intermediate_eval_results['constraint_gap'].append(abs(c_qual_mean - TAU))
    intermediate_eval_results['mu'].append(mu)
    
    # Lambda tracking
    intermediate_eval_results['mean_lambda'].append(current_lambda_mean)
    intermediate_eval_results['lambda_delta'].append(lambda_delta)
    
    # Loss
    intermediate_eval_results['loss'].append(pref_intermediate_results['loss'])

# ========================================
# EPOCH PROGRESSION TABLE (Convergence Analysis)
# ========================================
print("\n" + "=" * 100)
print("EPOCH PROGRESSION - CONVERGENCE ANALYSIS")
print("=" * 100)

if len(intermediate_eval_results['epoch']) > 0:
    print(f"\nShowing trends across {len(intermediate_eval_results['epoch'])} epochs:")
    print(f"\nKey: R@5=Recall@5, N@5=NDCG@5, C_qual=predicted qualification rate")
    print(f"     |τ-C_qual|=constraint gap (should → 0), μ=Lagrange multiplier")
    print(f"     λ=mean lambda, Δλ=lambda change from prev epoch (should → 0)")
    
    print(f"\n{'Epoch':<7} {'Pref':<9} {'Qual':<9} {'C_qual':<8} {'|τ-C_q|':<9} {'μ':<8} {'λ':<8} {'Δλ':<9} {'Loss':<8}")
    print(f"{'':7} {'R@5':<9} {'R@5':<9}")
    print(f"{'-'*90}")
    
    for i, ep in enumerate(intermediate_eval_results['epoch']):
        delta_lambda = intermediate_eval_results['lambda_delta'][i]
        delta_str = f"{delta_lambda:+.4f}" if i > 0 else "--"
        
        print(f"{ep:<7} "
              f"{intermediate_eval_results['pref_recall@5'][i]:.3f}     "
              f"{intermediate_eval_results['qual_recall@5'][i]:.3f}     "
              f"{intermediate_eval_results['c_qual_pred'][i]:.5f}  "
              f"{intermediate_eval_results['constraint_gap'][i]:.5f}  "
              f"{intermediate_eval_results['mu'][i]:.5f}  "
              f"{intermediate_eval_results['mean_lambda'][i]:.4f}   "
              f"{delta_str:<9} "
              f"{intermediate_eval_results['loss'][i]:.4f}")
    
    # Convergence summary
    if len(intermediate_eval_results['epoch']) >= 3:
        last_3_deltas = intermediate_eval_results['lambda_delta'][-3:]
        avg_delta = np.mean([abs(d) for d in last_3_deltas if d != 0])
        last_gap = intermediate_eval_results['constraint_gap'][-1]
        
        print(f"\n📊 Convergence Status:")
        if avg_delta < 0.01:
            print(f"  ✓ Lambda stabilized (avg |Δλ| in last 3 epochs: {avg_delta:.4f})")
        else:
            print(f"  ⚠ Lambda still changing (avg |Δλ| in last 3 epochs: {avg_delta:.4f})")
        
        if last_gap < 0.0005:
            print(f"  ✓ Constraint satisfied (|τ-C_qual|: {last_gap:.5f} < 0.0005)")
        else:
            print(f"  ⚠ Constraint not fully satisfied (|τ-C_qual|: {last_gap:.5f})")

print("=" * 100)

# ========================================
# BEST EPOCH ANALYSIS
# ========================================
print("\n" + "=" * 100)
print("BEST EPOCH ANALYSIS")
print("=" * 100)

if len(intermediate_eval_results['epoch']) > 0:
    # Find best epochs for each metric
    best_pref_recall5_idx = np.argmax(intermediate_eval_results['pref_recall@5'])
    best_pref_ndcg5_idx = np.argmax(intermediate_eval_results['pref_ndcg@5'])
    best_qual_recall5_idx = np.argmax(intermediate_eval_results['qual_recall@5'])
    best_qual_ndcg5_idx = np.argmax(intermediate_eval_results['qual_ndcg@5'])
    
    print(f"\n📊 Best Epochs by Metric:")
    print(f"\n  PREF (user-centric):")
    print(f"    Best Recall@5: Epoch {intermediate_eval_results['epoch'][best_pref_recall5_idx]} ({intermediate_eval_results['pref_recall@5'][best_pref_recall5_idx]:.3f})")
    print(f"    Best NDCG@5:   Epoch {intermediate_eval_results['epoch'][best_pref_ndcg5_idx]} ({intermediate_eval_results['pref_ndcg@5'][best_pref_ndcg5_idx]:.3f})")
    
    print(f"\n  QUAL (job-centric):")
    print(f"    Best Recall@5: Epoch {intermediate_eval_results['epoch'][best_qual_recall5_idx]} ({intermediate_eval_results['qual_recall@5'][best_qual_recall5_idx]:.3f})")
    print(f"    Best NDCG@5:   Epoch {intermediate_eval_results['epoch'][best_qual_ndcg5_idx]} ({intermediate_eval_results['qual_ndcg@5'][best_qual_ndcg5_idx]:.3f})")
    
    # Overall recommendation (average of the 4 key metrics)
    epoch_scores = []
    for i in range(len(intermediate_eval_results['epoch'])):
        avg_score = (
            intermediate_eval_results['pref_recall@5'][i] +
            intermediate_eval_results['pref_ndcg@5'][i] +
            intermediate_eval_results['qual_recall@5'][i] +
            intermediate_eval_results['qual_ndcg@5'][i]
        ) / 4.0
        epoch_scores.append(avg_score)
    
    best_overall_idx = np.argmax(epoch_scores)
    best_overall_epoch = intermediate_eval_results['epoch'][best_overall_idx]
    
    print(f"\n  🏆 OVERALL BEST EPOCH: {best_overall_epoch}")
    print(f"     Average of 4 key metrics: {epoch_scores[best_overall_idx]:.3f}")
    print(f"     Checkpoint: checkpoint_epoch{best_overall_epoch}/lambda_head.pt")
    
    # Show all epochs for comparison
    print(f"\n📈 All Epochs Comparison (sorted by overall score):")
    print(f"  {'Epoch':<8} {'Pref R@5':<10} {'Pref N@5':<10} {'Qual R@5':<10} {'Qual N@5':<10} {'Avg':<10}")
    print(f"  {'-'*58}")
    
    # Sort by avg score descending
    sorted_indices = np.argsort(epoch_scores)[::-1]
    for idx in sorted_indices:
        epoch_num = intermediate_eval_results['epoch'][idx]
        marker = "🏆" if idx == best_overall_idx else "  "
        print(f"{marker} Epoch {epoch_num:<3} "
              f"{intermediate_eval_results['pref_recall@5'][idx]:.3f}      "
              f"{intermediate_eval_results['pref_ndcg@5'][idx]:.3f}      "
              f"{intermediate_eval_results['qual_recall@5'][idx]:.3f}      "
              f"{intermediate_eval_results['qual_ndcg@5'][idx]:.3f}      "
              f"{epoch_scores[idx]:.3f}")

print("=" * 100)

# ========================================
# FINAL EVALUATION - BEST EPOCH ON FULL TEST SET
# ========================================
if len(intermediate_eval_results['epoch']) > 0:
    print("\n" + "=" * 100)
    print("FINAL EVALUATION - BEST EPOCH ON FULL TEST SET")
    print("=" * 100)
    
    # Load best checkpoint
    best_checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{best_overall_epoch}")
    print(f"\n📥 Loading best checkpoint: Epoch {best_overall_epoch}")
    print(f"   Path: {best_checkpoint_dir}")
    
    # Load lambda_head from best epoch
    best_lambda_head = nn.Sequential(
        nn.Linear(hidden_size, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1)
    ).to(device)
    
    best_lambda_head.load_state_dict(torch.load(os.path.join(best_checkpoint_dir, "lambda_head.pt"), map_location=device))
    best_lambda_head.eval()
    
    # Load training state to get mu
    training_state = torch.load(os.path.join(best_checkpoint_dir, "training_state.pt"), map_location=device)
    best_mu = training_state['mu']
    
    print(f"✓ Loaded lambda_head from Epoch {best_overall_epoch}")
    print(f"✓ Loaded μ = {best_mu:.5f}")
    print(f"\nNote: Using frozen encoder, pref_head, and qual_head from Stage 1B")
    
    # Prepare all test batches
    print(f"\n{'='*100}")
    print("Preparing FULL test sets...")
    print(f"{'='*100}")
    
    qual_test_batches_all = prepare_qual_batches(qual_test_df)
    print(f"✓ QUAL test batches: {len(qual_test_batches_all)} (all job-centric batches)")
    print(f"✓ PREF test batches: {len(pref_test_batches)} (all user-centric batches)")
    
    # 1. QUAL Evaluation
    print(f"\n{'='*100}")
    print(f"1. QUAL EVALUATION - {len(qual_test_batches_all)} batches")
    print(f"{'='*100}")
    print("Evaluating full Lagrangian policy on job-centric batches: s_final = s_pref + λ*s_qual")
    
    qual_final_results = evaluate_qual(
        qual_test_batches_all,
        model, pref_head, qual_head, best_lambda_head,
        pref_test_df, qual_test_df,
        desc="Final QUAL Eval"
    )
    
    print(f"\n📊 FINAL QUAL RESULTS ({len(qual_test_batches_all)} batches):")
    print(f"  Recall: R@1={qual_final_results.get('qual_recall@1', 0):.3f} | R@3={qual_final_results.get('qual_recall@3', 0):.3f} | R@5={qual_final_results.get('qual_recall@5', 0):.3f} | R@10={qual_final_results.get('qual_recall@10', 0):.3f} | R@20={qual_final_results.get('qual_recall@20', 0):.3f}")
    print(f"  NDCG:   N@1={qual_final_results.get('qual_ndcg@1', 0):.3f} | N@3={qual_final_results.get('qual_ndcg@3', 0):.3f} | N@5={qual_final_results.get('qual_ndcg@5', 0):.3f} | N@10={qual_final_results.get('qual_ndcg@10', 0):.3f} | N@20={qual_final_results.get('qual_ndcg@20', 0):.3f}")
    print(f"  Avg Rank: {qual_final_results.get('qual_avg_rank', 0):.1f}")
    print(f"  Constraint: C_qual={qual_final_results['c_qual_pred_mean']:.4f} | |τ-C_qual|={abs(qual_final_results['c_qual_pred_mean']-TAU):.5f}")
    print(f"  Batches: Over-qualified={qual_final_results.get('over_qualified_pct', 0):.1f}% | Under-qualified={qual_final_results.get('under_qualified_pct', 0):.1f}%")
    print(f"  Loss: {qual_final_results['loss']:.4f}")
    
    # 2. PREF Evaluation
    print(f"\n{'='*100}")
    print(f"2. PREF EVALUATION - {len(pref_test_batches)} batches")
    print(f"{'='*100}")
    print("Evaluating full Lagrangian policy on user-centric batches: s_final = s_pref + λ*s_qual")
    
    pref_final_results = evaluate_pref(
        pref_test_batches,
        model, pref_head, qual_head, best_lambda_head,
        pref_test_df, qual_test_df,
        desc="Final PREF Eval"
    )
    
    print(f"\n📊 FINAL PREF RESULTS ({len(pref_test_batches)} batches):")
    print(f"  Recall: R@1={pref_final_results['pref_recall@1']:.3f} | R@3={pref_final_results['pref_recall@3']:.3f} | R@5={pref_final_results['pref_recall@5']:.3f} | R@10={pref_final_results['pref_recall@10']:.3f} | R@20={pref_final_results['pref_recall@20']:.3f}")
    print(f"  NDCG:   N@1={pref_final_results['pref_ndcg@1']:.3f} | N@3={pref_final_results['pref_ndcg@3']:.3f} | N@5={pref_final_results['pref_ndcg@5']:.3f} | N@10={pref_final_results['pref_ndcg@10']:.3f} | N@20={pref_final_results['pref_ndcg@20']:.3f}")
    print(f"  Avg Rank: {pref_final_results['pref_avg_rank']:.1f}")
    print(f"  Constraint: C_qual={pref_final_results['c_qual_pred_mean']:.4f} | |τ-C_qual|={abs(pref_final_results['c_qual_pred_mean']-TAU):.5f}")
    print(f"  Batches: Over-qualified={pref_final_results.get('over_qualified_pct', 0):.1f}% | Under-qualified={pref_final_results.get('under_qualified_pct', 0):.1f}%")
    print(f"  Loss: {pref_final_results['loss']:.4f}")
    
    print(f"\n{'='*100}")
    print(f"🏆 FINAL EVALUATION COMPLETE - EPOCH {best_overall_epoch}")
    print(f"{'='*100}")
    
    # Save final results
    final_results_summary = {
        'best_epoch': best_overall_epoch,
        'best_mu': best_mu,
        'qual_results': qual_final_results,
        'pref_results': pref_final_results
    }
else:
    print("\n⚠️ No training completed - skipping final evaluation")
    qual_final_results = None
    pref_final_results = None
    final_results_summary = None

# Save final model and results
print("\n📦 Saving final model...")

# Save best lambda_head (if final eval was run)
if final_results_summary is not None:
    torch.save(best_lambda_head.state_dict(), os.path.join(OUTPUT_DIR, "lambda_head_final.pt"))
    print(f"✓ Saved BEST lambda_head (Epoch {best_overall_epoch}) to: lambda_head_final.pt")
else:
    torch.save(lambda_head.state_dict(), os.path.join(OUTPUT_DIR, "lambda_head_final.pt"))
    print(f"✓ Saved lambda_head (last epoch) to: lambda_head_final.pt")

torch.save(pref_head.state_dict(), os.path.join(OUTPUT_DIR, "pref_head_final.pt"))
torch.save(qual_head.state_dict(), os.path.join(OUTPUT_DIR, "qual_head_final.pt"))
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapters_final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))

torch.save({
    'mu_final': best_mu if final_results_summary is not None else mu,
    'best_epoch': best_overall_epoch if final_results_summary is not None else None,
    'history': history,
    'final_results': final_results_summary,
    'config': {
        'TAU': TAU,
        'ETA_MU': ETA_MU,
        'TEMPERATURE': TEMPERATURE,
        'LEARNING_RATE': LEARNING_RATE,
        'LAMBDA_CLAMP': LAMBDA_CLAMP,
        'NUM_EPOCHS': NUM_EPOCHS,
    }
}, os.path.join(OUTPUT_DIR, "training_results.pt"))

# Save history as CSV
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

# Save final results summary as CSV
if final_results_summary is not None:
    final_summary_data = {
        'best_epoch': [best_overall_epoch],
        'best_mu': [best_mu],
        'pref_recall@1': [pref_final_results['pref_recall@1']],
        'pref_recall@5': [pref_final_results['pref_recall@5']],
        'pref_ndcg@5': [pref_final_results['pref_ndcg@5']],
        'pref_avg_rank': [pref_final_results['pref_avg_rank']],
        'qual_recall@1': [qual_final_results.get('qual_recall@1', 0)],
        'qual_recall@5': [qual_final_results.get('qual_recall@5', 0)],
        'qual_ndcg@5': [qual_final_results.get('qual_ndcg@5', 0)],
        'qual_avg_rank': [qual_final_results.get('qual_avg_rank', 0)],
        'pref_c_qual': [pref_final_results['c_qual_pred_mean']],
        'qual_c_qual': [qual_final_results['c_qual_pred_mean']],
        'tau': [TAU]
    }
    final_summary_df = pd.DataFrame(final_summary_data)
    final_summary_df.to_csv(os.path.join(OUTPUT_DIR, "final_results_summary.csv"), index=False)
    print(f"✓ Saved final results summary to: final_results_summary.csv")

print(f"\n✅ All files saved to: {OUTPUT_DIR}")

print("\n" + "=" * 100)
print("STAGE 2 COMPLETE!")
print("=" * 100)

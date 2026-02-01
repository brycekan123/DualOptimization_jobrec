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
from datetime import datetime
import argparse
parser = argparse.ArgumentParser(description='Train Stage 1A LLM-based recommender model')
parser.add_argument('--dataset', type=str, required=True,
                    help='Path to dataset directory (e.g., ../dataset)')
parser.add_argument('--hf_token', type=str, required=True,
                    help='HuggingFace API token for model access')
args = parser.parse_args()

HF_TOKEN = args.hf_token
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = os.path.expanduser("~/llama_cache")

DATASET_DIR = args.dataset

# All CSVs are now directly in the dataset directory
TRAIN_CSV = os.path.join(DATASET_DIR, "pref_68_batches_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "pref_68_batches_test.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
QUAL_TEST_CSV = os.path.join(DATASET_DIR, "qual_test.csv")  # For cross-eval
USER_FILE = os.path.join(DATASET_DIR, "Final_users.csv")
ITEM_FILE = os.path.join(DATASET_DIR, "Final_items.csv")

NEGATIVES_PER_USER = 49
MAX_SEQ_LENGTH = 1000
PROCESS_CHUNK_SIZE = 10
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
NUM_QUAL_TEST_BATCHES = 50  

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

OUTPUT_DIR = f"stage1a_output/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

# Sample only qual_test_batches to configured size
np.random.seed(42)
qual_test_batches = [qual_test_batches[i] for i in np.random.choice(len(qual_test_batches), size=min(NUM_QUAL_TEST_BATCHES, len(qual_test_batches)), replace=False)]

print(f"✓ Label_pref batches:")
print(f"  Train: {len(train_batches):,}")
print(f"  Test: {len(test_batches):,}")
print(f"  Val: {len(val_batches):,}")
print(f"\n✓ Label_qual batches:")
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
            embeddings_batch = embeddings_batch.float()
            scores = pref_head(embeddings_batch).squeeze()
            
            rank = (scores[0] < scores[1:]).sum().item() + 1
            all_ranks.append(rank)
            
            del all_embeddings, chunk_embeddings, embeddings_batch, scores
            
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
    
    metrics = calculate_metrics(all_ranks)
    return metrics

print("\n Cleaning memory before training...")
torch.cuda.empty_cache()

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
    
    for batch_idx, batch in enumerate(train_batches):
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
            chunk_embeddings = pool_hidden_states(chunk_hidden, chunk_inputs['attention_mask'])
            all_embeddings.append(chunk_embeddings)
            
            del chunk_inputs, chunk_outputs, chunk_hidden
        
        embeddings_batch = torch.cat(all_embeddings, dim=0)
        embeddings_batch = embeddings_batch.float()
        scores = pref_head(embeddings_batch).squeeze()
        loss = info_nce_loss(scores)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(pref_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()
                
        epoch_loss += loss.item()
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        del all_embeddings, chunk_embeddings, embeddings_batch, scores, loss
        
        if (batch_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    pbar.close()
 
    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.save_pretrained(os.path.join(checkpoint_dir, "lora_adapters"))
    torch.save(pref_head.state_dict(), os.path.join(checkpoint_dir, "pref_head.pt"))
    
    print(f" Saved checkpoint: {checkpoint_dir}")

print("\n" + "=" * 100)
print("STAGE 1A COMPLETE!")
print("=" * 100)
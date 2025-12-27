import os
import numpy as np
import pandas as pd
import torch
import shutil

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataloader import TrainDataLoader
from recbole.data.interaction import Interaction
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from recbole.utils import FeatureType, FeatureSource


# ============================================================
# 0. PATHS / NAMES
# ============================================================

"""
USER-CENTRIC TRAINING FOR label_pref WITH DCN-V2:

This script trains DCN-V2 in a USER-CENTRIC manner for job-applicant matching.
The goal: For each user (applicant), rank preferred jobs higher.

KEY DIFFERENCE FROM JOB-CENTRIC:
- User-centric (label_pref): For each user, recommend jobs they'd prefer
  Format: (user, job) with label ∈ {0, 1}
  
- No swapping needed - users are users, items are jobs (natural RecBole format)

DCN-V2 APPROACH:
- DCN-V2 is a CTR model that needs explicit labels (0/1)
- We use BOTH positives (label=1) AND negatives (label=0) for training
- No need for custom negative sampling - we train on labeled pairs directly
"""

# Your original labeled dataset (with label=0,1 and hard negatives)
orig_ds = "hard_negative_label_pref"
orig_dir = f"dataset/{orig_ds}"

# New dataset for DCN-V2 (explicit labels)
dcnv2_ds = "dcnv2_pref"
dcnv2_dir = f"dataset/{dcnv2_ds}"
os.makedirs(dcnv2_dir, exist_ok=True)

# Full label files
full_train_path = f"{orig_dir}/{orig_ds}.train.inter"
full_test_path = f"{orig_dir}/{orig_ds}.test.inter"

# Training settings
NUM_TRAIN_BATCHES = 68   # Sample 68 batches from training data
NEGATIVES_PER_USER = 49  # 49 negatives per positive (for reference)
RANDOM_SEED = 42         # Fixed seed for reproducibility


# ============================================================
# 1. Build DCN-V2 dataset: keep natural user/item, keep all labels
# ============================================================
def get_label_col(df):
    """Return the correct label column name in df."""
    label_candidates = {"label", "label_pref", "label_qual"}
    for c in df.columns:
        base = c.split(":")[0]
        if base in label_candidates:
            return c
    raise ValueError(f"Label column not found. Columns = {list(df.columns)}")

def get_user_item_cols(df):
    """Return (user_col, item_col) from a df."""
    user_candidates = {"user", "user_id"}
    item_candidates = {"item", "item_id"}

    user_col = None
    item_col = None

    for c in df.columns:
        base = c.split(":")[0]
        if user_col is None and base in user_candidates:
            user_col = c
        if item_col is None and base in item_candidates:
            item_col = c

    if user_col is None or item_col is None:
        raise ValueError(f"Could not find user/item columns. Columns = {list(df.columns)}")
    return user_col, item_col

def build_dcnv2_dataset():
    """
    USER-CENTRIC DCN-V2:
      - NO swapping needed (user=applicant, item=job is natural)
      - SAMPLE 68 training batches (matching SimpleX approach)
      - Keep ALL label=0 and label=1 rows for the sampled batches
      - Convert label to float format for CTR prediction
    """
    os.makedirs(dcnv2_dir, exist_ok=True)
    print(f"✅ Created dataset directory: {dcnv2_dir}")
    
    # --- Copy .user/.item features (NO SWAPPING for user-centric) ---
    print("\n📊 Copying feature files (user-centric, no swapping)...")
    
    for feature_type in ['user', 'item']:
        src_file = f"{orig_dir}/{orig_ds}.{feature_type}"
        dst_file = f"{dcnv2_dir}/{dcnv2_ds}.{feature_type}"
        
        if os.path.exists(src_file):
            with open(src_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                header_line = lines[0].strip()
                print(f"\n  Processing {feature_type} file:")
                print(f"     Original header: {header_line[:200]}...")
                
                id_field_original = header_line.split('\t')[0].split(':')[0]
                print(f"     Original ID field: {id_field_original}")
                
                # Fix field names if needed
                if id_field_original == f'{feature_type}_id':
                    lines[0] = lines[0].replace(f'{feature_type}_id:token', f'{feature_type}:token')
                elif id_field_original == feature_type:
                    pass  # Already correct
                
                header_line = lines[0].strip()
                print(f"     ✅ Final header: {header_line[:200]}...")
            
            with open(dst_file, 'w') as f:
                f.writelines(lines)
            
            print(f"  ✅ Copied: {src_file} → {dst_file}")
            print(f"     File has {len(lines)} lines (including header)")
        else:
            print(f"  ⚠️  No {feature_type} feature file found at {src_file}")
    
    # --- Process TRAIN: Sample 68 batches, keep all labels for those batches ---
    print("\n📊 Building training data (USER-CENTRIC, sampling 68 batches)...")
    train_src = f"{orig_dir}/{orig_ds}.train.inter"
    if not os.path.exists(train_src):
        raise FileNotFoundError(f"Missing source file: {train_src}")
    
    train_df = pd.read_csv(train_src, sep=",")
    user_col, item_col = get_user_item_cols(train_df)
    label_col = get_label_col(train_df)

    # Count negatives per user to find valid users
    print(f"\n📊 Analyzing user negative counts...")
    negatives = train_df[train_df[label_col] == 0]
    user_neg_counts = negatives.groupby(user_col).size()
    
    print(f"  Total users in training: {train_df[user_col].nunique()}")
    print(f"  Users with ≥{NEGATIVES_PER_USER} negatives: {(user_neg_counts >= NEGATIVES_PER_USER).sum()}")
    
    # Sample 68 batches: each batch = 1 user with 1 positive
    print(f"\n📊 Sampling {NUM_TRAIN_BATCHES} training batches...")
    positives = train_df[train_df[label_col] == 1]
    
    # Group by user to ensure we sample complete batches
    user_groups = positives.groupby(user_col)
    all_batches = []
    
    for user_id, group in user_groups:
        for idx in range(len(group)):
            all_batches.append(group.iloc[idx])
    
    print(f"  Available batches from all users: {len(all_batches)}")
    
    # Sample 68 batches with fixed seed
    np.random.seed(RANDOM_SEED)
    if len(all_batches) < NUM_TRAIN_BATCHES:
        print(f"⚠️  Warning: Only {len(all_batches)} batches available, requested {NUM_TRAIN_BATCHES}")
        sampled_batches = all_batches
    else:
        sampled_indices = np.random.choice(len(all_batches), size=NUM_TRAIN_BATCHES, replace=False)
        sampled_batches = [all_batches[i] for i in sampled_indices]
    
    sampled_positives = pd.DataFrame(sampled_batches)
    unique_users = sampled_positives[user_col].nunique()
    print(f"  ✅ Sampled {len(sampled_positives)} positives from {unique_users} unique users")
    
    # For each sampled positive, include up to 49 labeled negatives (same user)
    print(f"\n📊 Adding labeled negatives for each sampled positive...")
    train_rows = []
    users_with_few_negs = 0
    
    for _, pos_row in sampled_positives.iterrows():
        user_id = pos_row[user_col]
        
        # Add the positive
        train_rows.append({
            user_col: user_id,
            item_col: pos_row[item_col],
            label_col: 1
        })
        
        # Get negatives for this user
        user_negs = negatives[negatives[user_col] == user_id]
        
        if len(user_negs) < NEGATIVES_PER_USER:
            users_with_few_negs += 1
        
        # Sample up to 49 negatives (or all available if fewer)
        if len(user_negs) > NEGATIVES_PER_USER:
            user_negs_sampled = user_negs.sample(n=NEGATIVES_PER_USER, random_state=RANDOM_SEED)
        else:
            user_negs_sampled = user_negs
        
        # Add the negatives
        for _, neg_row in user_negs_sampled.iterrows():
            train_rows.append({
                user_col: user_id,
                item_col: neg_row[item_col],
                label_col: 0
            })
    
    if users_with_few_negs > 0:
        print(f"  ⚠️  Note: {users_with_few_negs} users had <{NEGATIVES_PER_USER} negatives (used all available)")
    
    train_with_labels = pd.DataFrame(train_rows)
    train_with_labels.columns = ["user:token", "item:token", "label:float"]
    
    out_path = f"{dcnv2_dir}/{dcnv2_ds}.train.inter"
    train_with_labels.to_csv(out_path, index=False, sep='\t')
    
    n_pos = int((train_with_labels["label:float"] == 1).sum())
    n_neg = int((train_with_labels["label:float"] == 0).sum())
    print(f"\n✅ Wrote training file: {out_path} ({len(train_with_labels)} rows)")
    print(f"   Positives (label=1): {n_pos} (sampled 68)")
    print(f"   Negatives (label=0): {n_neg} (up to 49 per positive)")
    print(f"   Unique users: {train_with_labels['user:token'].nunique()}")
    print(f"   Unique items: {train_with_labels['item:token'].nunique()}")
    print(f"   ✅ This matches SimpleX's 68-batch setup!")
    
    # VERIFICATION: Check that we have exactly 68 positives
    if n_pos != NUM_TRAIN_BATCHES:
        print(f"  ⚠️  WARNING: Expected {NUM_TRAIN_BATCHES} positives but got {n_pos}!")
    else:
        print(f"  ✅ VERIFIED: Exactly {NUM_TRAIN_BATCHES} positive samples as expected")
    
    # --- Process VAL (if exists) and TEST ---
    for split in ["val", "test"]:
        src = f"{orig_dir}/{orig_ds}.{split}.inter"
        out_path = f"{dcnv2_dir}/{dcnv2_ds}.{split}.inter"

        if not os.path.exists(src):
            if split == "val":
                print(f"⚠️  No original {split} file found. Creating empty {out_path}.")
                with open(out_path, "w") as f:
                    f.write("user:token\titem:token\tlabel:float\n")
                continue
            else:
                print(f"⚠️  Warning: Missing {split} file at {src}, skipping")
                continue

        df = pd.read_csv(src, sep=",")
        user_col_split, item_col_split = get_user_item_cols(df)
        label_col_split = get_label_col(df)

        # NO SWAPPING: keep natural order, keep all labels
        split_out = df[[user_col_split, item_col_split, label_col_split]].copy()
        split_out.columns = ["user:token", "item:token", "label:float"]
        split_out.to_csv(out_path, index=False, sep='\t')
        
        n_pos = int((split_out["label:float"] == 1).sum())
        n_neg = int((split_out["label:float"] == 0).sum())
        print(f"✅ Wrote {split} file: {out_path} ({len(split_out)} rows)")
        print(f"   Positives (label=1): {n_pos}, Negatives (label=0): {n_neg}")


# ============================================================
# 2. Main: build data, train DCN-V2
# ============================================================

if __name__ == "__main__":
    # 1) Build DCN-V2 dataset
    build_dcnv2_dataset()

    # 2) Config for DCN-V2
    config_dict = {
        'data_path': 'dataset',
        'USER_ID_FIELD': 'user',
        'ITEM_ID_FIELD': 'item',
        'LABEL_FIELD': 'label',
        
        'load_col': {
            'inter': ['user', 'item', 'label'],
            # Don't auto-load features - we'll inject them manually
        },
        'field_separator': '\t',
        'benchmark_filename': ['train', 'val', 'test'],

        # DCN-V2 specific: no negative sampling (we have explicit labels)
        'neg_sampling': None,

        'epochs': 50,  # Reduced to 10 for faster training
        'train_batch_size': 256,
        'eval_batch_size': 256,
        'learning_rate': 5e-4,  # Reduced from 1e-3 for small dataset
        'stopping_step': 20,  # Reduced from 100 for faster stopping
        'eval_step': 5,  # Evaluate more frequently
        'weight_decay': 1e-6,  # Add regularization

        'metrics': ['AUC', 'LogLoss'],
        'valid_metric': 'AUC',

        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'RO',
            'mode': 'labeled',  # Changed from 'full' to 'labeled' for CTR models
        },

        'device': 'cuda',
        'show_progress': True,
        'checkpoint_dir': 'saved',
    }

    config = Config(model='DCNV2', dataset=dcnv2_ds, config_dict=config_dict)

    # 3) Create dataset and prepare data
    init_seed(config['seed'], reproducibility=True)
    init_logger(config)

    dataset = create_dataset(config)
    print(dataset)
    
    # MANUAL FEATURE LOADING
    print("\n📊 Manually loading user and item features...")
    print("="*80)
    
    # Load user features (applicants)
    user_feat_file = f"{dcnv2_dir}/{dcnv2_ds}.user"
    if os.path.exists(user_feat_file):
        user_feat_df = pd.read_csv(user_feat_file, sep='\t')
        print(f"✅ Loaded user features: {len(user_feat_df)} users, {len(user_feat_df.columns)} columns")
        
        # Get the user ID column
        user_id_col = [c for c in user_feat_df.columns if c.startswith('user')][0]
        user_tokens = user_feat_df[user_id_col].astype(str)
        
        # Map to internal IDs
        user_token2id = dataset.field2token_id[dataset.uid_field]
        num_internal_users = dataset.user_num
        
        print(f"   RecBole user token2id mapping has {len(user_token2id)} entries")
        print(f"   RecBole says dataset has {num_internal_users} users")
        
        # Create mapping from original index to internal ID
        token_to_internal = user_tokens.map(user_token2id)
        valid_mask = token_to_internal.notna()
        
        print(f"   Found {valid_mask.sum()}/{len(user_tokens)} users in dataset")
        
        # Build feature dictionary - vectorized
        user_feat_dict = {}
        for col in user_feat_df.columns:
            if col == user_id_col:
                continue
                
            field_name, field_type = col.split(':')
            
            if field_name not in dataset.field2type:
                if field_type == 'token':
                    dataset.field2type[field_name] = FeatureType.TOKEN
                    dataset.field2seqlen[field_name] = 1
                    dataset.field2source[field_name] = FeatureSource.USER
                elif field_type == 'float':
                    dataset.field2type[field_name] = FeatureType.FLOAT
                    dataset.field2source[field_name] = FeatureSource.USER
            
            # Initialize with zeros
            if field_type == 'token':
                feat_tensor = torch.zeros(num_internal_users, dtype=torch.long)
            elif field_type == 'float':
                feat_tensor = torch.zeros(num_internal_users, dtype=torch.float)
                # Fill in values for valid users (vectorized)
                valid_internal_ids = token_to_internal[valid_mask].astype(int).values
                valid_values = user_feat_df.loc[valid_mask, col].fillna(0).values
                feat_tensor[valid_internal_ids] = torch.FloatTensor(valid_values)
            
            user_feat_dict[field_name] = feat_tensor
        
        # Add user ID field
        user_feat_dict[dataset.uid_field] = torch.arange(num_internal_users, dtype=torch.long)
        
        dataset.user_feat = Interaction(user_feat_dict)
        print(f"   ✅ Injected {len(user_feat_dict)} user feature fields")
        print(f"   ✅ Feature tensors sized for {num_internal_users} internal user IDs")
    
    # Load item features (jobs)
    item_feat_file = f"{dcnv2_dir}/{dcnv2_ds}.item"
    if os.path.exists(item_feat_file):
        item_feat_df = pd.read_csv(item_feat_file, sep='\t')
        print(f"✅ Loaded item features: {len(item_feat_df)} items, {len(item_feat_df.columns)} columns")
        
        # Get the item ID column
        item_id_col = [c for c in item_feat_df.columns if c.startswith('item')][0]
        item_tokens = item_feat_df[item_id_col].astype(str)
        
        # Map to internal IDs
        item_token2id = dataset.field2token_id[dataset.iid_field]
        num_internal_items = dataset.item_num
        
        print(f"   RecBole item token2id mapping has {len(item_token2id)} entries")
        print(f"   RecBole says dataset has {num_internal_items} items")
        
        # Create mapping from original index to internal ID
        token_to_internal = item_tokens.map(item_token2id)
        valid_mask = token_to_internal.notna()
        
        print(f"   Found {valid_mask.sum()}/{len(item_tokens)} items in dataset")
        
        # Build feature dictionary - vectorized
        item_feat_dict = {}
        for col in item_feat_df.columns:
            if col == item_id_col:
                continue
                
            field_name, field_type = col.split(':')
            
            if field_name not in dataset.field2type:
                if field_type == 'token':
                    dataset.field2type[field_name] = FeatureType.TOKEN
                    dataset.field2seqlen[field_name] = 1
                    dataset.field2source[field_name] = FeatureSource.ITEM
                elif field_type == 'float':
                    dataset.field2type[field_name] = FeatureType.FLOAT
                    dataset.field2source[field_name] = FeatureSource.ITEM
            
            # Initialize with zeros
            if field_type == 'token':
                feat_tensor = torch.zeros(num_internal_items, dtype=torch.long)
            elif field_type == 'float':
                feat_tensor = torch.zeros(num_internal_items, dtype=torch.float)
                # Fill in values for valid items (vectorized)
                valid_internal_ids = token_to_internal[valid_mask].astype(int).values
                valid_values = item_feat_df.loc[valid_mask, col].fillna(0).values
                feat_tensor[valid_internal_ids] = torch.FloatTensor(valid_values)
            
            item_feat_dict[field_name] = feat_tensor
        
        # Add item ID field
        item_feat_dict[dataset.iid_field] = torch.arange(num_internal_items, dtype=torch.long)
        
        dataset.item_feat = Interaction(item_feat_dict)
        print(f"   ✅ Injected {len(item_feat_dict)} item feature fields")
        print(f"   ✅ Feature tensors sized for {num_internal_items} internal item IDs")
    
    print("="*80)

    # CRITICAL VERIFICATION: Check that features are actually in the dataset
    print("\n" + "="*80)
    print("FEATURE VERIFICATION - CHECKING DATASET FIELDS")
    print("="*80)
    print(f"Dataset field2type keys: {list(dataset.field2type.keys())[:20]}...")
    print(f"Total fields registered: {len(dataset.field2type)}")
    
    user_fields = [f for f, src in dataset.field2source.items() if src == FeatureSource.USER]
    item_fields = [f for f, src in dataset.field2source.items() if src == FeatureSource.ITEM]
    
    print(f"\n✅ User fields (applicants): {len(user_fields)} fields")
    print(f"   First 10: {user_fields[:10]}")
    print(f"\n✅ Item fields (jobs): {len(item_fields)} fields")
    print(f"   First 10: {item_fields[:10]}")
    
    if dataset.user_feat is not None:
        print(f"\n✅ User features loaded: {len(dataset.user_feat)} rows")
        print(f"   Fields in user_feat: {list(dataset.user_feat.interaction.keys())[:10]}...")
    else:
        print(f"\n⚠️  WARNING: dataset.user_feat is None!")
    
    if dataset.item_feat is not None:
        print(f"\n✅ Item features loaded: {len(dataset.item_feat)} rows")
        print(f"   Fields in item_feat: {list(dataset.item_feat.interaction.keys())[:10]}...")
    else:
        print(f"\n⚠️  WARNING: dataset.item_feat is None!")
    
    print("="*80)

    # Standard RecBole data preparation (no custom dataloader needed)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    print("\n" + "="*80)
    print("TRAINING DATA VERIFICATION (USER-CENTRIC DCN-V2)")
    print("="*80)
    print(f"Total interactions in train dataset: {len(train_data.dataset)}")
    print(f"Number of batches: {len(train_data)}")
    print(f"Batch size: {config['train_batch_size']}")
    print(f"Expected samples per epoch: ~{len(train_data) * config['train_batch_size']}")
    
    # Show actual training data statistics
    train_inter = train_data.dataset.inter_feat
    if 'label' in train_inter.interaction:
        labels = train_inter['label'].cpu().numpy()
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        print(f"\nActual training data:")
        print(f"  Positives (label=1): {n_pos}")
        print(f"  Negatives (label=0): {n_neg}")
        print(f"  Total: {len(labels)}")
        print(f"  Positive rate: {n_pos/len(labels)*100:.2f}%")
    
    print(f"\nFormat: Each sample is (user, item, label) where label ∈ {{0, 1}}")
    print(f"Training uses ALL labeled pairs (no negative sampling needed)")
    
    # Verify features are in the batch
    print(f"\n✅ Verifying features in training batches:")
    sample_batch = next(iter(train_data))
    batch_fields = list(sample_batch.interaction.keys())
    print(f"   Fields in batch: {len(batch_fields)} fields")
    print(f"   First 20 fields: {batch_fields[:20]}")
    
    user_fields_in_batch = [f for f in batch_fields if f in user_fields]
    item_fields_in_batch = [f for f in batch_fields if f in item_fields]
    print(f"   User fields in batch: {len(user_fields_in_batch)}")
    print(f"   Item fields in batch: {len(item_fields_in_batch)}")
    
    if len(user_fields_in_batch) == 0:
        print(f"   ⚠️  WARNING: No user features in training batch!")
    if len(item_fields_in_batch) == 0:
        print(f"   ⚠️  WARNING: No item features in training batch!")
    
    # Show sample values
    print(f"\n📊 Sample feature values from first batch:")
    for field in batch_fields[:10]:
        if field in sample_batch.interaction:
            vals = sample_batch.interaction[field]
            if vals.dim() == 1:
                print(f"   {field}: shape={vals.shape}, sample values={vals[:3].tolist()}")
            else:
                print(f"   {field}: shape={vals.shape}")
    
    print("="*80)
    
    # Model + Trainer
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    
    # CRITICAL VERIFICATION: Check model's embedding dimensions
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE VERIFICATION")
    print("="*80)
    print(f"Model type: {type(model).__name__}")
    
    # Check if model has embedding layers
    if hasattr(model, 'embed_input_fields'):
        print(f"✅ Model embed_input_fields: {model.embed_input_fields}")
    if hasattr(model, 'embedding_size'):
        print(f"✅ Model embedding_size: {model.embedding_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Check embedding layers specifically
    embedding_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)]
    print(f"\n✅ Embedding layers found: {len(embedding_layers)}")
    if len(embedding_layers) > 0:
        print(f"   First 10 embedding layers: {embedding_layers[:10]}")
        # Show size of first few embeddings
        for name in embedding_layers[:5]:
            layer = dict(model.named_modules())[name]
            print(f"      {name}: {layer.num_embeddings} × {layer.embedding_dim}")
    else:
        print(f"   ⚠️  WARNING: No embedding layers found!")
    
    # Check if model is using features
    print(f"\n✅ Checking if model will use features:")
    sample_batch = next(iter(train_data))
    print(f"   Sample batch keys: {list(sample_batch.interaction.keys())}")
    print(f"   Batch size: {len(sample_batch)}")
    
    # Try a forward pass to see what happens
    model.eval()
    with torch.no_grad():
        try:
            output = model.predict(sample_batch)
            print(f"   ✅ Forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"   ⚠️  Forward pass failed: {e}")
    model.train()
    
    print("="*80)
    
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    # Train with validation
    best_valid_score, best_valid_result = trainer.fit(
        train_data, 
        valid_data=valid_data, 
        show_progress=config['show_progress']
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Save model
    model_save_path = f"saved/DCNV2-{dcnv2_ds}-labeled.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'dataset_name': dcnv2_ds,
    }, model_save_path)
    print(f"💾 Saved trained model to: {model_save_path}")
print("\n" + "="*80)
print("EVALUATING ON TEST SET (USER-CENTRIC, label_pref)")
print("="*80)

test_label_path = full_test_path
test_df = pd.read_csv(test_label_path, sep=",")
test_user_col, test_item_col = get_user_item_cols(test_df)
test_label_col = get_label_col(test_df)

print(f"Using labeled test file: {test_label_path}")
print(f"\nNOTE: USER-CENTRIC format (no swapping):")
print(f"  - user_col = APPLICANTS (users)")
print(f"  - item_col = JOBS (items)")
print(f"  - Test evaluation uses first {NEGATIVES_PER_USER} negatives per user (DETERMINISTIC)")

model.eval()

user_token2id = dataset.field2token_id[dataset.uid_field]
item_token2id = dataset.field2token_id[dataset.iid_field]
user_field = dataset.uid_field
item_field = dataset.iid_field

all_ranks = []
evaluated_users = 0
skipped_users = []

total_test_negatives = 0
total_test_positives = 0

# Group by USER (user column in test file)
user_groups = test_df.groupby(test_user_col)

print(f"\n📊 Test data structure from {test_label_path}:")
print(f"   Total users in test file: {test_df[test_user_col].nunique()}")
print(f"   Using first {NEGATIVES_PER_USER} negatives per user (DETERMINISTIC SLICING)")
print(f"   Format: Each user evaluated with 1+ positives + up to {NEGATIVES_PER_USER} negatives")

with torch.no_grad():
    for user_tok, group in user_groups:
        items_tok = group[test_item_col].astype(str).values
        labels = group[test_label_col].values

        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        if pos_mask.sum() == 0:
            skipped_users.append((user_tok, "no_positive_items"))
            continue

        if str(user_tok) not in user_token2id:
            skipped_users.append((user_tok, "user_not_in_mapping"))
            continue

        # FIXED: USE SLICING INSTEAD OF RANDOM SAMPLING (deterministic)
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]
        
        # Take first 49 negatives (or all if fewer) - DETERMINISTIC
        neg_indices_selected = neg_indices[:NEGATIVES_PER_USER]
        
        # Combine: all positives + first 49 negatives
        selected_indices = np.concatenate([pos_indices, neg_indices_selected])
        items_tok_sampled = items_tok[selected_indices]
        labels_sampled = labels[selected_indices]

        item_ids = []
        valid_indices = []
        for idx, item_tok in enumerate(items_tok_sampled):
            if item_tok in item_token2id:
                item_ids.append(item_token2id[item_tok])
                valid_indices.append(idx)

        if len(item_ids) == 0:
            skipped_users.append((user_tok, "no_items_in_mapping"))
            continue

        labels_valid = labels_sampled[valid_indices]
        pos_mask_valid = (labels_valid == 1)
        neg_mask_valid = (labels_valid == 0)

        if pos_mask_valid.sum() == 0:
            skipped_users.append((user_tok, "no_positive_items_after_filtering"))
            continue

        total_test_positives += pos_mask_valid.sum()
        total_test_negatives += neg_mask_valid.sum()

        user_id = user_token2id[str(user_tok)]
        user_ids = [user_id] * len(item_ids)

        interaction_dict = {
            user_field: torch.LongTensor(user_ids).to(config['device']),
            item_field: torch.LongTensor(item_ids).to(config['device']),
        }
        interaction = Interaction(interaction_dict)

        scores = model.predict(interaction).cpu().numpy()

        for idx_in_batch, is_pos in enumerate(pos_mask_valid):
            if not is_pos:
                continue
            pos_score = scores[idx_in_batch]
            rank = (scores > pos_score).sum() + 1
            all_ranks.append(rank)

        evaluated_users += 1
        
        # Print first evaluated user
        if evaluated_users == 1:
            print(f"\n✅ First evaluated user: {user_tok}")
            print(f"   Positives (label=1): {pos_mask_valid.sum()}")
            print(f"   Negatives (label=0): {neg_mask_valid.sum()} (first {NEGATIVES_PER_USER} via slicing)")
            print(f"   This confirms test uses deterministic slicing, not random sampling")

print(f"\n" + "="*80)
print("TEST EVALUATION RESULTS")
print("="*80)
print(f"✅ User-centric test evaluation complete:")
print(f"   Evaluated users: {evaluated_users}")
print(f"   Skipped users: {len(skipped_users)}")
print(f"   Total positive items evaluated: {len(all_ranks)}")
print(f"   Total test positives used (label=1): {total_test_positives}")
print(f"   Total test negatives used (label=0): {total_test_negatives} (first {NEGATIVES_PER_USER} per user via slicing)")

# VERIFICATION
print(f"\n🔍 TEST DATA VERIFICATION:")
print(f"   Expected test positives: 1402 (from test file)")
print(f"   Actual positives evaluated: {len(all_ranks)}")
if len(all_ranks) == 1402:
    print(f"   ✅ MATCH! Evaluated all 1402 test positives")
else:
    print(f"   ⚠️  Difference: {1402 - len(all_ranks)} positives not evaluated")

print(f"\n   Each user evaluated with up to {NEGATIVES_PER_USER} negatives (first 49 via slicing)")
print(f"   Average negatives per user: {total_test_negatives/evaluated_users:.1f}")
print(f"   ✅ Test setup: 1 positive + first {NEGATIVES_PER_USER} negatives per user (DETERMINISTIC)")
print("="*80)

all_ranks = np.array(all_ranks)

def recall_at_k(ranks, k):
    return np.mean(ranks <= k)

def ndcg_at_k(ranks, k):
    ndcg_scores = []
    for r in ranks:
        if r <= k:
            ndcg_scores.append(1.0 / np.log2(r + 1))
        else:
            ndcg_scores.append(0.0)
    return np.mean(ndcg_scores)

print(f"\nTest Results (user-centric, DCN-V2):")
for k in [1, 3, 5]:
    recall = recall_at_k(all_ranks, k)
    ndcg = ndcg_at_k(all_ranks, k)
    print(f"  Recall@{k}: {recall:.4f}")
    print(f"  NDCG@{k}:   {ndcg:.4f}")
print("="*80)
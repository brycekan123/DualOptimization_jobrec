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
USER-CENTRIC TRAINING FOR label_pref WITH DeepFM:

This script trains DeepFM in a USER-CENTRIC manner for user-job matching.
The goal: For each user, rank preferred jobs higher.

KEY DIFFERENCE FROM SimpleX:
- SimpleX: Uses positive-only data + negative sampling
- DeepFM: CTR model that trains on EXPLICIT labels (both 0 and 1)

DeepFM trains directly on all labeled data:
- label=1: preferred jobs for users
- label=0: non-preferred jobs for users (hard negatives)

Files created:
- .inter files have: user:token, item:token, label:float
- Training: DeepFM learns from all labeled pairs (no additional sampling needed)
"""

# Your original labeled dataset (with label=0,1 and hard negatives)
orig_ds = "hard_negative_label_pref"
orig_dir = f"dataset/{orig_ds}"

# New dataset for DeepFM (with explicit labels)
deepfm_ds = "deepfm_pref"
deepfm_dir = f"dataset/{deepfm_ds}"
os.makedirs(deepfm_dir, exist_ok=True)

# Training settings
RANDOM_SEED = 42         # Fixed seed for reproducibility


# ============================================================
# 1. Build DeepFM-style dataset: ALL labeled data
# ============================================================

def build_deepfm_dataset():
    """
    USER-CENTRIC for DeepFM:
      - Sample 68 positives (same as SimpleX)
      - For each positive, include up to 49 labeled negatives (same user)
      - DeepFM trains on these explicit labels
    """
    os.makedirs(deepfm_dir, exist_ok=True)
    print(f"✅ Created dataset directory: {deepfm_dir}")
    
    # --- Copy .user/.item features ---
    print("\n📊 Copying feature files for user-centric training...")
    for feature_type in ['user', 'item']:
        src_file = f"{orig_dir}/{orig_ds}.{feature_type}"
        dst_file = f"{deepfm_dir}/{deepfm_ds}.{feature_type}"
        
        if os.path.exists(src_file):
            with open(src_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                header_line = lines[0].strip()
                print(f"\n  Processing {feature_type} file:")
                print(f"     Original header: {header_line[:200]}...")
                
                id_field_original = header_line.split('\t')[0].split(':')[0]
                print(f"     Original ID field: {id_field_original}")
                
                if id_field_original == f'{feature_type}_id':
                    lines[0] = lines[0].replace(f'{feature_type}_id:token', f'{feature_type}:token')
                elif id_field_original == feature_type:
                    print(f"     ✅ Header already correct")
                
                header_line = lines[0].strip()
                print(f"     ✅ Final header: {header_line[:200]}...")
            
            with open(dst_file, 'w') as f:
                f.writelines(lines)
            
            print(f"  ✅ Copied: {src_file} → {dst_file}")
            print(f"     File has {len(lines)} lines (including header)")
        else:
            print(f"  ⚠️  No {feature_type} feature file found at {src_file}")
    
    # --- TRAIN: Sample 68 batches + their labeled negatives ---
    print("\n📊 Building training data (USER-CENTRIC, 68 sampled batches for DeepFM)...")
    train_src = f"{orig_dir}/{orig_ds}.train.inter"
    if not os.path.exists(train_src):
        raise FileNotFoundError(f"Missing source file: {train_src}")
    
    train_df = pd.read_csv(train_src, sep=",")
    user_col = next(c for c in train_df.columns if c.split(":")[0] == "user")
    item_col = next(c for c in train_df.columns if c.split(":")[0] == "item")
    label_col = next(c for c in train_df.columns if c.split(":")[0] == "label")

    # Get all negatives
    print("\n📊 Analyzing user negative counts...")
    negatives = train_df[train_df[label_col] == 0]
    user_neg_counts = negatives.groupby(user_col).size()
    
    NEGATIVES_PER_USER = 49
    NUM_TRAIN_BATCHES = 68
    
    print(f"  Total users in training: {train_df[user_col].nunique()}")
    print(f"  Users with ≥{NEGATIVES_PER_USER} negatives: {(user_neg_counts >= NEGATIVES_PER_USER).sum()}")
    
    # Sample 68 positives from ALL users (don't filter by negative count)
    print(f"\n📊 Sampling {NUM_TRAIN_BATCHES} training batches...")
    positives = train_df[train_df[label_col] == 1]
    
    # Group by user to get all available batches
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

    out_path = f"{deepfm_dir}/{deepfm_ds}.train.inter"
    train_with_labels.to_csv(out_path, index=False, sep='\t')

    n_pos = int((train_with_labels["label:float"] == 1).sum())
    n_neg = int((train_with_labels["label:float"] == 0).sum())
    print(f"\n✅ Wrote training file: {out_path} ({len(train_with_labels)} rows)")
    print(f"   Positives (label=1): {n_pos} (sampled 68)")
    print(f"   Negatives (label=0): {n_neg} (up to 49 per positive)")
    print(f"   Unique users: {train_with_labels['user:token'].nunique()}")
    print(f"   Unique items: {train_with_labels['item:token'].nunique()}")
    print(f"   ✅ This matches SimpleX's 68-batch setup!")
    
    # --- VAL (create empty to avoid RecBole crash) ---
    val_out = f"{deepfm_dir}/{deepfm_ds}.val.inter"
    with open(val_out, "w") as f:
        f.write("user:token\titem:token\tlabel:float\n")
    print(f"✅ Created empty val file: {val_out}")
    
    # --- TEST: ALL labeled data (pos + neg) ---
    test_src = f"{orig_dir}/{orig_ds}.test.inter"
    if os.path.exists(test_src):
        test_df = pd.read_csv(test_src, sep=",")
        t_user_col = next(c for c in test_df.columns if c.split(":")[0] == "user")
        t_item_col = next(c for c in test_df.columns if c.split(":")[0] == "item")
        t_label_col = next(c for c in test_df.columns if c.split(":")[0] == "label")

        test_with_labels = test_df[[t_user_col, t_item_col, t_label_col]].copy()
        test_with_labels.columns = ["user:token", "item:token", "label:float"]
        
        test_out = f"{deepfm_dir}/{deepfm_ds}.test.inter"
        test_with_labels.to_csv(test_out, index=False, sep='\t')
        
        n_pos_test = int((test_with_labels["label:float"] == 1).sum())
        n_neg_test = int((test_with_labels["label:float"] == 0).sum())
        print(f"✅ Wrote test file: {test_out} ({len(test_with_labels)} rows)")
        print(f"   Positives (label=1): {n_pos_test}")
        print(f"   Negatives (label=0): {n_neg_test}")
    else:
        print(f"⚠️  No test file found at {test_src}")


# ============================================================
# 4. Main: build data, train DeepFM with labeled data
# ============================================================

if __name__ == "__main__":
    # 1) Build dataset for DeepFM
    build_deepfm_dataset()

    # 2) Config for DeepFM
    config_dict = {
        'data_path': 'dataset',
        'USER_ID_FIELD': 'user',
        'ITEM_ID_FIELD': 'item',
        'LABEL_FIELD': 'label',  # DeepFM needs this
        
        'load_col': {
            'inter': ['user', 'item', 'label'],
        },
        'field_separator': '\t',
        'benchmark_filename': ['train', 'val', 'test'],

        # DeepFM doesn't use negative sampling - trains on explicit labels
        'neg_sampling': None,

        'epochs': 50,
        'train_batch_size': 256,
        'eval_batch_size': 256,
        'learning_rate': 1e-3,
        'stopping_step': 100,
        'eval_step': 100,

        'metrics': ['AUC', 'LogLoss'],
        'valid_metric': 'AUC',

        'device': 'cuda',
        'show_progress': True,
        'checkpoint_dir': 'saved',
    }

    config = Config(model='xDeepFM', dataset=deepfm_ds, config_dict=config_dict)

    # 3) Create dataset
    init_seed(config['seed'], reproducibility=True)
    init_logger(config)

    dataset = create_dataset(config)
    print(dataset)
    
    # MANUAL FEATURE LOADING
    print("\n📊 Manually loading user and item features...")
    print("="*80)
    
    # Load user features
    user_feat_file = f"{deepfm_dir}/{deepfm_ds}.user"
    if os.path.exists(user_feat_file):
        user_feat_df = pd.read_csv(user_feat_file, sep='\t')
        print(f"✅ Loaded user features: {len(user_feat_df)} users, {len(user_feat_df.columns)} columns")
        
        # Get the ID column (first column in the file)
        user_id_col = user_feat_df.columns[0]  # e.g., 'user:token' or 'user_id:token'
        print(f"   User ID column in file: {user_id_col}")
        
        user_feat_dict = {}
        for col in user_feat_df.columns:
            if col == user_id_col:
                continue  # Skip the ID column itself
                
            field_name, field_type = col.split(':')
            
            if field_name not in dataset.field2type:
                if field_type == 'token':
                    dataset.field2type[field_name] = FeatureType.TOKEN
                    dataset.field2seqlen[field_name] = 1
                    dataset.field2source[field_name] = FeatureSource.USER
                elif field_type == 'float':
                    dataset.field2type[field_name] = FeatureType.FLOAT
                    dataset.field2source[field_name] = FeatureSource.USER
            
            # CRITICAL FIX: Create tensor sized to dataset.user_num, not len(user_feat_df)
            if field_type == 'token':
                # Initialize with 0 (default/unknown token)
                feat_tensor = torch.zeros(dataset.user_num, dtype=torch.long)
                # Map each user token to its internal ID and set the feature value
                for idx, user_token in enumerate(user_feat_df[user_id_col]):
                    if str(user_token) in dataset.field2token_id[dataset.uid_field]:
                        internal_id = dataset.field2token_id[dataset.uid_field][str(user_token)]
                        feat_value = dataset.field2token_id.get(field_name, {}).get(str(user_feat_df[col].iloc[idx]), 0)
                        feat_tensor[internal_id] = feat_value
                user_feat_dict[field_name] = feat_tensor
            elif field_type == 'float':
                # Initialize with 0.0 (default value)
                feat_tensor = torch.zeros(dataset.user_num, dtype=torch.float)
                # Map each user token to its internal ID and set the feature value
                for idx, user_token in enumerate(user_feat_df[user_id_col]):
                    if str(user_token) in dataset.field2token_id[dataset.uid_field]:
                        internal_id = dataset.field2token_id[dataset.uid_field][str(user_token)]
                        feat_value = float(user_feat_df[col].iloc[idx]) if pd.notna(user_feat_df[col].iloc[idx]) else 0.0
                        feat_tensor[internal_id] = feat_value
                user_feat_dict[field_name] = feat_tensor
        
        dataset.user_feat = Interaction(user_feat_dict)
        print(f"   ✅ Injected {len(user_feat_dict)} user feature fields")
        print(f"   Feature tensor size: {dataset.user_num} (matches dataset.user_num)")
    
    # Load item features
    item_feat_file = f"{deepfm_dir}/{deepfm_ds}.item"
    if os.path.exists(item_feat_file):
        item_feat_df = pd.read_csv(item_feat_file, sep='\t')
        print(f"✅ Loaded item features: {len(item_feat_df)} items, {len(item_feat_df.columns)} columns")
        
        # Get the ID column (first column in the file)
        item_id_col = item_feat_df.columns[0]  # e.g., 'item:token' or 'item_id:token'
        print(f"   Item ID column in file: {item_id_col}")
        
        item_feat_dict = {}
        for col in item_feat_df.columns:
            if col == item_id_col:
                continue  # Skip the ID column itself
                
            field_name, field_type = col.split(':')
            
            if field_name not in dataset.field2type:
                if field_type == 'token':
                    dataset.field2type[field_name] = FeatureType.TOKEN
                    dataset.field2seqlen[field_name] = 1
                    dataset.field2source[field_name] = FeatureSource.ITEM
                elif field_type == 'float':
                    dataset.field2type[field_name] = FeatureType.FLOAT
                    dataset.field2source[field_name] = FeatureSource.ITEM
            
            # CRITICAL FIX: Create tensor sized to dataset.item_num, not len(item_feat_df)
            if field_type == 'token':
                # Initialize with 0 (default/unknown token)
                feat_tensor = torch.zeros(dataset.item_num, dtype=torch.long)
                # Map each item token to its internal ID and set the feature value
                for idx, item_token in enumerate(item_feat_df[item_id_col]):
                    if str(item_token) in dataset.field2token_id[dataset.iid_field]:
                        internal_id = dataset.field2token_id[dataset.iid_field][str(item_token)]
                        feat_value = dataset.field2token_id.get(field_name, {}).get(str(item_feat_df[col].iloc[idx]), 0)
                        feat_tensor[internal_id] = feat_value
                item_feat_dict[field_name] = feat_tensor
            elif field_type == 'float':
                # Initialize with 0.0 (default value)
                feat_tensor = torch.zeros(dataset.item_num, dtype=torch.float)
                # Map each item token to its internal ID and set the feature value
                for idx, item_token in enumerate(item_feat_df[item_id_col]):
                    if str(item_token) in dataset.field2token_id[dataset.iid_field]:
                        internal_id = dataset.field2token_id[dataset.iid_field][str(item_token)]
                        feat_value = float(item_feat_df[col].iloc[idx]) if pd.notna(item_feat_df[col].iloc[idx]) else 0.0
                        feat_tensor[internal_id] = feat_value
                item_feat_dict[field_name] = feat_tensor
        
        dataset.item_feat = Interaction(item_feat_dict)
        print(f"   ✅ Injected {len(item_feat_dict)} item feature fields")
        print(f"   Feature tensor size: {dataset.item_num} (matches dataset.item_num)")
    
    print("="*80)

    # Use RecBole's default data preparation (DeepFM doesn't need custom dataloader)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    print("\n" + "="*80)
    print("TRAINING DATA VERIFICATION (USER-CENTRIC, DeepFM)")
    print("="*80)
    print(f"Total interactions in train: {len(train_data.dataset)}")
    print(f"\n✅ DeepFM trained on SAMPLED 68 batches (same as SimpleX):")
    print(f"   - 68 sampled positives (same users/items as SimpleX)")
    print(f"   - Up to 49 labeled negatives per positive")
    print(f"   - Expected: ~68 + (68 × 49) = ~3,400 interactions")

    # Verify labels are present and mixed (0 and 1)
    if 'label' in train_data.dataset.inter_feat:
        labels = train_data.dataset.inter_feat['label'].cpu().numpy()
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        
        # Get unique users in training data
        train_user_ids = train_data.dataset.inter_feat[train_data.dataset.uid_field].cpu().numpy()
        unique_train_users = np.unique(train_user_ids)
        
        print(f"\n✅ Label distribution in training:")
        print(f"   Positives (label=1): {n_pos} (expected: 68)")
        print(f"   Negatives (label=0): {n_neg} (expected: ~3,332)")
        print(f"   Total: {n_pos + n_neg}")
        print(f"\n✅ Unique users in TRAINING data: {len(unique_train_users)} (expected: 62)")
        
        # Show sample of actual training data
        print(f"\n📋 Sample of training data (first 5 rows):")
        for i in range(min(5, len(train_data.dataset))):
            user_id = train_data.dataset.inter_feat[train_data.dataset.uid_field][i].item()
            item_id = train_data.dataset.inter_feat[train_data.dataset.iid_field][i].item()
            label = train_data.dataset.inter_feat['label'][i].item()
            
            # Convert to tokens
            user_tok = train_data.dataset.id2token(train_data.dataset.uid_field, user_id)
            item_tok = train_data.dataset.id2token(train_data.dataset.iid_field, item_id)
            
            print(f"   {i+1}. User {user_tok}, Item {item_tok}, Label {label}")
        
        print(f"\n   ✅ Training on YOUR sampled 68 batches (label=0 = your hard negatives)")
        print(f"\n   ℹ️  NOTE: RecBole's dataset stats above show TOTAL (train+test combined)")
        print(f"             Only TRAINING data ({len(train_data.dataset)} rows, {len(unique_train_users)} users) is used for training!")
    else:
        print(f"   ⚠️  No label field found in training data!")
    print("="*80)
    
    # Model + Trainer
    model_cls = get_model(config['model'])
    model = model_cls(config, train_data.dataset).to(config['device'])

    trainer_cls = get_trainer(config['MODEL_TYPE'], config['model'])
    trainer = trainer_cls(config, model)

    print("\n" + "="*80)
    print("STARTING TRAINING (DeepFM)")
    print("="*80)
    
    # Train
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=valid_data, show_progress=config['show_progress'])
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Save model
    model_save_path = f"saved/DeepFM-{deepfm_ds}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'dataset_name': deepfm_ds,
    }, model_save_path)
    print(f"💾 Saved trained model to: {model_save_path}")

    # =====================================================================
    # USER-CENTRIC EVALUATION ON TEST SET
    # =====================================================================
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (USER-CENTRIC, DeepFM)")
    print("="*80)

    test_label_path = f"{orig_dir}/{orig_ds}.test.inter"
    test_df = pd.read_csv(test_label_path, sep=",")
    test_user_col = next(c for c in test_df.columns if c.split(":")[0] == "user")
    test_item_col = next(c for c in test_df.columns if c.split(":")[0] == "item")
    test_label_col = next(c for c in test_df.columns if c.split(":")[0] == "label")

    print(f"Using labeled test file: {test_label_path}")

    model.eval()

    user_token2id = dataset.field2token_id[dataset.uid_field]
    item_token2id = dataset.field2token_id[dataset.iid_field]
    user_field = dataset.uid_field
    item_field = dataset.iid_field

    all_ranks = []
    evaluated_users = 0
    skipped_users = []

    # Group by USER (user column in test file)
    user_groups = test_df.groupby(test_user_col)
    
    print(f"\n📊 Test data structure:")
    print(f"   Total users in test file: {test_df[test_user_col].nunique()}")
    print(f"   ⚠️  Capping negatives at 49 per user for fair comparison")

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

            # Cap negatives at 49
            neg_indices = np.where(neg_mask)[0]
            if len(neg_indices) > 49:
                np.random.seed(42 + user_token2id[str(user_tok)])  # Deterministic per user
                neg_indices = np.random.choice(neg_indices, size=49, replace=False)
            
            # Combine pos + capped neg indices
            pos_indices = np.where(pos_mask)[0]
            valid_indices_mask = np.concatenate([pos_indices, neg_indices])
            
            items_tok_filtered = items_tok[valid_indices_mask]
            labels_filtered = labels[valid_indices_mask]
            
            item_ids = []
            valid_indices = []
            for idx, item_tok in enumerate(items_tok_filtered):
                if item_tok in item_token2id:
                    item_ids.append(item_token2id[item_tok])
                    valid_indices.append(idx)

            if len(item_ids) == 0:
                skipped_users.append((user_tok, "no_items_in_mapping"))
                continue

            labels_valid = labels_filtered[valid_indices]
            pos_mask_valid = (labels_valid == 1)

            if pos_mask_valid.sum() == 0:
                skipped_users.append((user_tok, "no_positive_items_after_filtering"))
                continue

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
            
            # Print details for first evaluated user
            if evaluated_users == 1:
                print(f"\n✅ First evaluated user: {user_tok}")
                n_pos = pos_mask_valid.sum()
                n_neg = (labels_valid == 0).sum()
                print(f"   Positives (label=1): {n_pos}")
                print(f"   Negatives (label=0): {n_neg} (capped at 49)")
                print(f"   Total candidates: {len(item_ids)}")
                print(f"   ✅ Using YOUR labeled negatives from hard_negative_label_pref.test.inter")
                
                print(f"\n   Sample candidates for this user:")
                for idx in range(min(5, len(item_ids))):
                    item_id = item_ids[idx]
                    item_tok = dataset.id2token(item_field, item_id)
                    label = labels_valid[idx]
                    score = scores[idx]
                    print(f"     Item {item_tok}: label={int(label)}, score={score:.4f}")

    print(f"\n" + "="*80)
    print("TEST EVALUATION SUMMARY")
    print("="*80)
    print(f"   Evaluated users: {evaluated_users}")
    print(f"   Skipped users: {len(skipped_users)}")
    print(f"   Total positives ranked: {len(all_ranks)}")
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

    print(f"\nTest Results (user-centric, DeepFM):")
    for k in [1, 3, 5]:
        recall = recall_at_k(all_ranks, k)
        ndcg = ndcg_at_k(all_ranks, k)
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  NDCG@{k}:   {ndcg:.4f}")
    print("="*80)
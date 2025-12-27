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
JOB-CENTRIC TRAINING FOR label_qual WITH DCN-V2:

This script trains DCN-V2 in a JOB-CENTRIC manner for job-applicant matching.
The goal: For each job, rank qualified applicants higher.

FIXED VERSION - MATCHES xDeepFM EXACTLY:
- Sample 68 training batches (positives)
- Add up to 49 labeled negatives per positive
- Cap test negatives at 49 per job for fair evaluation
"""

# Your original labeled dataset (with label=0,1 and hard negatives)
orig_ds = "hard_negative_label_qual"
orig_dir = f"dataset/{orig_ds}"

# New dataset for DCN-V2 (explicit labels)
dcnv2_ds = "dcnv2_qual"
dcnv2_dir = f"dataset/{dcnv2_ds}"
os.makedirs(dcnv2_dir, exist_ok=True)

# Full label files
full_train_path = f"{orig_dir}/{orig_ds}.train.inter"
full_test_path = f"{orig_dir}/{orig_ds}.test.inter"

# Training settings - MATCH xDeepFM
NEGATIVES_PER_JOB = 49
NUM_TRAIN_BATCHES = 68
RANDOM_SEED = 42


# ============================================================
# 1. Build DCN-V2 dataset: SAME AS xDeepFM
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
    JOB-CENTRIC DCN-V2 - EXACT SAME LOGIC AS xDeepFM:
      - Sample 68 positives (same approach as xDeepFM)
      - For each positive, include up to 49 labeled negatives (same job)
      - DCN-V2 trains on these explicit labels
    """
    os.makedirs(dcnv2_dir, exist_ok=True)
    print(f"✅ Created dataset directory: {dcnv2_dir}")
    
    # --- SWAP .user/.item features ---
    print("\n📊 Copying and SWAPPING feature files for job-centric training...")
    swap_mapping = {
        'user': 'item',  # Original user features → become item features (item=user)
        'item': 'user',  # Original item features → become user features (user=job)
    }
    
    for orig_type, dest_type in swap_mapping.items():
        src_file = f"{orig_dir}/{orig_ds}.{orig_type}"
        dst_file = f"{dcnv2_dir}/{dcnv2_ds}.{dest_type}"
        
        if os.path.exists(src_file):
            with open(src_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                header_line = lines[0].strip()
                print(f"\n  Processing {orig_type} file → {dest_type} file:")
                print(f"     Original header: {header_line[:200]}...")
                
                id_field_original = header_line.split('\t')[0].split(':')[0]
                print(f"     Original ID field: {id_field_original}")
                
                if id_field_original == f'{orig_type}_id':
                    lines[0] = lines[0].replace(f'{orig_type}_id:token', f'{dest_type}:token')
                elif id_field_original == orig_type:
                    lines[0] = lines[0].replace(f'{orig_type}:token', f'{dest_type}:token')
                
                header_line = lines[0].strip()
                print(f"     ✅ New header: {header_line[:200]}...")
                print(f"     ✅ {orig_type} features → {dest_type} features (swapped)")
            
            with open(dst_file, 'w') as f:
                f.writelines(lines)
            
            print(f"  ✅ Copied: {src_file} → {dst_file}")
            print(f"     File has {len(lines)} lines (including header)")
        else:
            print(f"  ⚠️  No {orig_type} feature file found at {src_file}")
    
    # --- TRAIN: Sample 68 batches + their labeled negatives (EXACT xDeepFM LOGIC) ---
    print("\n📊 Building training data (JOB-CENTRIC, 68 sampled batches for DCN-V2)...")
    train_src = f"{orig_dir}/{orig_ds}.train.inter"
    if not os.path.exists(train_src):
        raise FileNotFoundError(f"Missing source file: {train_src}")
    
    train_df = pd.read_csv(train_src, sep=",")
    user_col, item_col = get_user_item_cols(train_df)
    label_col = get_label_col(train_df)

    # Get all negatives
    print("\n📊 Analyzing job negative counts...")
    negatives = train_df[train_df[label_col] == 0]
    job_neg_counts = negatives.groupby(item_col).size()
    
    print(f"  Total jobs in training: {train_df[item_col].nunique()}")
    print(f"  Jobs with ≥{NEGATIVES_PER_JOB} negatives: {(job_neg_counts >= NEGATIVES_PER_JOB).sum()}")
    
    # Sample 68 positives from ALL jobs (EXACT xDeepFM LOGIC)
    print(f"\n📊 Sampling {NUM_TRAIN_BATCHES} training batches...")
    positives = train_df[train_df[label_col] == 1]
    
    # Group by job to get all available batches
    job_groups = positives.groupby(item_col)
    all_batches = []
    
    for job_id, group in job_groups:
        for idx in range(len(group)):
            all_batches.append(group.iloc[idx])
    
    print(f"  Available batches from all jobs: {len(all_batches)}")
    
    # Sample 68 batches with fixed seed (EXACT xDeepFM LOGIC)
    np.random.seed(RANDOM_SEED)
    if len(all_batches) < NUM_TRAIN_BATCHES:
        print(f"⚠️  Warning: Only {len(all_batches)} batches available, requested {NUM_TRAIN_BATCHES}")
        sampled_batches = all_batches
    else:
        sampled_indices = np.random.choice(len(all_batches), size=NUM_TRAIN_BATCHES, replace=False)
        sampled_batches = [all_batches[i] for i in sampled_indices]
    
    sampled_positives = pd.DataFrame(sampled_batches)
    unique_jobs = sampled_positives[item_col].nunique()
    print(f"  ✅ Sampled {len(sampled_positives)} positives from {unique_jobs} unique jobs")
    
    # For each sampled positive, include up to 49 labeled negatives (EXACT xDeepFM LOGIC)
    print(f"\n📊 Adding labeled negatives for each sampled positive...")
    train_rows = []
    jobs_with_few_negs = 0
    
    for _, pos_row in sampled_positives.iterrows():
        job_id = pos_row[item_col]
        user_id = pos_row[user_col]
        
        # Add the positive (SWAPPED: job, applicant, label)
        train_rows.append({
            item_col: job_id,
            user_col: user_id,
            label_col: 1
        })
        
        # Get negatives for this job
        job_negs = negatives[negatives[item_col] == job_id]
        
        if len(job_negs) < NEGATIVES_PER_JOB:
            jobs_with_few_negs += 1
        
        # Sample up to 49 negatives (or all available if fewer)
        if len(job_negs) > NEGATIVES_PER_JOB:
            job_negs_sampled = job_negs.sample(n=NEGATIVES_PER_JOB, random_state=RANDOM_SEED)
        else:
            job_negs_sampled = job_negs
        
        # Add the negatives (SWAPPED: job, applicant, label)
        for _, neg_row in job_negs_sampled.iterrows():
            train_rows.append({
                item_col: job_id,
                user_col: neg_row[user_col],
                label_col: 0
            })
    
    if jobs_with_few_negs > 0:
        print(f"  ⚠️  Note: {jobs_with_few_negs} jobs had <{NEGATIVES_PER_JOB} negatives (used all available)")
    
    train_with_labels = pd.DataFrame(train_rows)
    # SWAP columns: job becomes user, applicant becomes item
    train_with_labels = train_with_labels[[item_col, user_col, label_col]]
    train_with_labels.columns = ["user:token", "item:token", "label:float"]

    out_path = f"{dcnv2_dir}/{dcnv2_ds}.train.inter"
    train_with_labels.to_csv(out_path, index=False, sep='\t')

    n_pos = int((train_with_labels["label:float"] == 1).sum())
    n_neg = int((train_with_labels["label:float"] == 0).sum())
    print(f"\n✅ Wrote training file: {out_path} ({len(train_with_labels)} rows)")
    print(f"   Positives (label=1): {n_pos} (sampled 68)")
    print(f"   Negatives (label=0): {n_neg} (up to 49 per positive)")
    print(f"   Unique jobs: {train_with_labels['user:token'].nunique()}")
    print(f"   Unique applicants: {train_with_labels['item:token'].nunique()}")
    print(f"   ✅ This matches xDeepFM's 68-batch setup!")
    
    # VERIFICATION
    if n_pos != NUM_TRAIN_BATCHES:
        print(f"  ⚠️  WARNING: Expected {NUM_TRAIN_BATCHES} positives but got {n_pos}!")
    else:
        print(f"  ✅ VERIFIED: Exactly {NUM_TRAIN_BATCHES} positive samples as expected")
    
    # --- VAL (create empty to avoid RecBole crash) ---
    val_out = f"{dcnv2_dir}/{dcnv2_ds}.val.inter"
    with open(val_out, "w") as f:
        f.write("user:token\titem:token\tlabel:float\n")
    print(f"✅ Created empty val file: {val_out}")
    
    # --- TEST: ALL labeled data (pos + neg), SWAPPED ---
    test_src = f"{orig_dir}/{orig_ds}.test.inter"
    if os.path.exists(test_src):
        test_df = pd.read_csv(test_src, sep=",")
        t_user_col, t_item_col = get_user_item_cols(test_df)
        t_label_col = get_label_col(test_df)

        test_swapped = test_df[[t_item_col, t_user_col, t_label_col]].copy()
        test_swapped.columns = ["user:token", "item:token", "label:float"]
        
        test_out = f"{dcnv2_dir}/{dcnv2_ds}.test.inter"
        test_swapped.to_csv(test_out, index=False, sep='\t')
        
        n_pos_test = int((test_swapped["label:float"] == 1).sum())
        n_neg_test = int((test_swapped["label:float"] == 0).sum())
        print(f"✅ Wrote test file: {test_out} ({len(test_swapped)} rows)")
        print(f"   Positives (label=1): {n_pos_test}")
        print(f"   Negatives (label=0): {n_neg_test}")
    else:
        print(f"⚠️  No test file found at {test_src}")


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
        },
        'field_separator': '\t',
        'benchmark_filename': ['train', 'val', 'test'],

        # DCN-V2 specific: no negative sampling (we have explicit labels)
        'neg_sampling': None,

        'epochs': 50,
        'train_batch_size': 256,
        'eval_batch_size': 256,
        'learning_rate': 1e-3,
        'stopping_step': 100,
        'eval_step': 100,

        'metrics': ['AUC', 'LogLoss'],
        'valid_metric': 'AUC',

        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'RO',
            'mode': 'labeled',
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
    
    # Load user features (jobs)
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
    
    # Load item features (applicants)
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

    # Standard RecBole data preparation
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    print("\n" + "="*80)
    print("TRAINING DATA VERIFICATION (JOB-CENTRIC DCN-V2)")
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
        print(f"  Positives (label=1): {n_pos} (expected: {NUM_TRAIN_BATCHES})")
        print(f"  Negatives (label=0): {n_neg} (expected: ~{NUM_TRAIN_BATCHES * NEGATIVES_PER_JOB})")
        print(f"  Total: {len(labels)}")
        print(f"  Positive rate: {n_pos/len(labels)*100:.2f}%")
    
    print(f"\nFormat: Each sample is (job, applicant, label) where label ∈ {{0, 1}}")
    print(f"Training uses sampled 68 positives + their labeled negatives")
    print("="*80)
    
    # Model + Trainer
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
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

    # =====================================================================
    # JOB-CENTRIC EVALUATION ON TEST SET (EXACT xDeepFM LOGIC)
    # =====================================================================
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (JOB-CENTRIC, label_qual)")
    print("="*80)

    test_label_path = full_test_path
    test_df = pd.read_csv(test_label_path, sep=",")
    test_user_col, test_item_col = get_user_item_cols(test_df)
    test_label_col = get_label_col(test_df)

    print(f"Using labeled test file: {test_label_path}")
    print(f"\nNOTE: Field mappings due to swapping:")
    print(f"  - Original file: user_col=APPLICANTS, item_col=JOBS")
    print(f"  - RecBole dataset: uid_field=JOBS, iid_field=APPLICANTS (swapped!)")
    print(f"  - Test evaluation capped at {NEGATIVES_PER_JOB} negatives per job (matching training)")

    model.eval()

    applicant_token2id = dataset.field2token_id[dataset.iid_field]
    job_token2id = dataset.field2token_id[dataset.uid_field]
    job_field = dataset.uid_field
    applicant_field = dataset.iid_field

    all_ranks = []
    evaluated_jobs = 0
    skipped_jobs = []
    
    total_test_negatives = 0
    total_test_positives = 0

    # Group by JOB (item column in original test file)
    job_groups = test_df.groupby(test_item_col)
    
    print(f"\n📊 Test data structure from {test_label_path}:")
    print(f"   Total jobs in test file: {test_df[test_item_col].nunique()}")
    print(f"   Negatives will be capped at {NEGATIVES_PER_JOB} per job (DETERMINISTIC)")
    print(f"   Format: Each job evaluated with 1+ positives + up to {NEGATIVES_PER_JOB} negatives")

    with torch.no_grad():
        for job_tok, group in job_groups:
            applicants_tok = group[test_user_col].astype(str).values
            labels = group[test_label_col].values

            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            
            if pos_mask.sum() == 0:
                skipped_jobs.append((job_tok, "no_positive_applicants"))
                continue

            if str(job_tok) not in job_token2id:
                skipped_jobs.append((job_tok, "job_not_in_mapping"))
                continue

            # FIXED: CAP NEGATIVES AT 49 (EXACT xDeepFM LOGIC)
            pos_indices = np.where(pos_mask)[0]
            neg_indices = np.where(neg_mask)[0]
            
            # Cap negatives using deterministic seed per job
            if len(neg_indices) > NEGATIVES_PER_JOB:
                np.random.seed(42 + job_token2id[str(job_tok)])
                neg_indices = np.random.choice(neg_indices, size=NEGATIVES_PER_JOB, replace=False)
            
            # Combine pos + capped neg indices
            valid_indices_mask = np.concatenate([pos_indices, neg_indices])
            applicants_tok_filtered = applicants_tok[valid_indices_mask]
            labels_filtered = labels[valid_indices_mask]
            
            # Map to internal IDs
            applicant_ids = []
            valid_indices = []
            for idx, applicant_tok in enumerate(applicants_tok_filtered):
                if applicant_tok in applicant_token2id:
                    applicant_ids.append(applicant_token2id[applicant_tok])
                    valid_indices.append(idx)

            if len(applicant_ids) == 0:
                skipped_jobs.append((job_tok, "no_applicants_in_mapping"))
                continue

            labels_valid = labels_filtered[valid_indices]
            pos_mask_valid = (labels_valid == 1)
            neg_mask_valid = (labels_valid == 0)

            if pos_mask_valid.sum() == 0:
                skipped_jobs.append((job_tok, "no_positive_applicants_after_filtering"))
                continue

            total_test_positives += pos_mask_valid.sum()
            total_test_negatives += neg_mask_valid.sum()

            job_id = job_token2id[str(job_tok)]
            job_ids = [job_id] * len(applicant_ids)

            interaction_dict = {
                job_field: torch.LongTensor(job_ids).to(config['device']),
                applicant_field: torch.LongTensor(applicant_ids).to(config['device']),
            }
            interaction = Interaction(interaction_dict)

            scores = model.predict(interaction).cpu().numpy()

            for idx_in_batch, is_pos in enumerate(pos_mask_valid):
                if not is_pos:
                    continue
                pos_score = scores[idx_in_batch]
                rank = (scores > pos_score).sum() + 1
                all_ranks.append(rank)

            evaluated_jobs += 1
            
            # Print first evaluated job
            if evaluated_jobs == 1:
                print(f"\n✅ First evaluated job: {job_tok}")
                print(f"   Positives (label=1): {pos_mask_valid.sum()}")
                print(f"   Negatives (label=0): {neg_mask_valid.sum()} (capped at {NEGATIVES_PER_JOB})")
                print(f"   This confirms test uses deterministic capping, matching xDeepFM")

    print(f"\n" + "="*80)
    print("TEST EVALUATION RESULTS")
    print("="*80)
    print(f"✅ Job-centric test evaluation complete:")
    print(f"   Evaluated jobs: {evaluated_jobs}")
    print(f"   Skipped jobs: {len(skipped_jobs)}")
    print(f"   Total positive applicants evaluated: {len(all_ranks)}")
    print(f"   Total test positives used (label=1): {total_test_positives}")
    print(f"   Total test negatives used (label=0): {total_test_negatives} (capped at {NEGATIVES_PER_JOB} per job)")
    
    print(f"\n   Each job evaluated with up to {NEGATIVES_PER_JOB} negatives (matching training)")
    print(f"   Average negatives per job: {total_test_negatives/evaluated_jobs:.1f}")
    print(f"   ✅ Test setup matches xDeepFM: 1 positive + up to 49 negatives per job")
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

    print(f"\nTest Results (job-centric, DCN-V2):")
    for k in [1, 3, 5]:
        recall = recall_at_k(all_ranks, k)
        ndcg = ndcg_at_k(all_ranks, k)
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  NDCG@{k}:   {ndcg:.4f}")
    print("="*80)
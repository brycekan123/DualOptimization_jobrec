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
JOB-CENTRIC TRAINING FOR label_qual:

This script trains SimpleX in a JOB-CENTRIC manner for job-applicant matching.
The goal: For each job, rank qualified applicants higher.

KEY DIFFERENCE FROM USER-CENTRIC:
- User-centric (label_pref): For each user, recommend jobs they'd prefer
  Format: (user, positive_job) + negative_jobs
  
- Job-centric (label_qual): For each job, recommend qualified applicants
  Format: (job, positive_user) + negative_users

IMPLEMENTATION:
We SWAP the meaning of "user" and "item" fields in RecBole:
- RecBole "user" field = JOB (the entity we make recommendations FOR)
- RecBole "item" field = USER (the entity being recommended)
- "neg_item" = negative USERS for each job

Files created:
- .inter files have: user:token=JOB, item:token=USER (swapped!)
- Negatives mapping: job_id -> [negative_user_ids]
- Training: For each (job, qualified_user), add K unqualified_users (K varies per job)
"""

# Your original labeled dataset (with label=0,1 and hard negatives)
orig_ds = "hard_negative_label_qual"
orig_dir = f"dataset/{orig_ds}"

# New dataset for SimpleX (implicit, positives only)
simplex_ds = "simplex_qual"
simplex_dir = f"dataset/{simplex_ds}"
os.makedirs(simplex_dir, exist_ok=True)

# Full label file we will use to build custom negatives
full_train_path = f"{orig_dir}/{orig_ds}.train.inter"

# Training settings
NEGATIVES_PER_USER = 49  # Max 49 negatives per positive (actual may be less)
RANDOM_SEED = 42         # Fixed seed for reproducibility


# ============================================================
# 1. Build SimpleX-style dataset: positives only
# ============================================================
def get_label_col(df):
    """
    Return the correct label column name in df.
    Accepts: label, label_pref, label_qual (any type suffix).
    """
    label_candidates = {"label", "label_pref", "label_qual"}
    for c in df.columns:
        base = c.split(":")[0]
        if base in label_candidates:
            return c
    raise ValueError(
        f"Label column not found. Columns = {list(df.columns)}"
    )

def get_user_item_cols(df):
    """
    Return (user_col, item_col) from a df that may use:
      - user:token / item:token
      - user_id:token / item_id:token
    """
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
        raise ValueError(
            f"Could not find user/item columns. Columns = {list(df.columns)}"
        )
    return user_col, item_col
def build_simplex_dataset():
    """
    JOB-CENTRIC:
      - .train.inter will contain:
          * 68 true positives (job, qualified_applicant) with label=1
          * all negative (job, applicant) pairs from qual TRAIN with label=0
          * all negative (job, applicant) pairs from qual TEST with label=0 (dummy)
        This forces RecBole to assign IDs to all applicants (train + test).
      - Our custom dataloader will ONLY treat label=1 rows as positives.
    """
    os.makedirs(simplex_dir, exist_ok=True)
    print(f"✅ Created dataset directory: {simplex_dir}")
    
    # --- SWAP .user/.item features ---
    print("\n📊 Copying and SWAPPING feature files for job-centric training...")
    swap_mapping = {
        'user': 'item',  # Original user features → become item features (item=user)
        'item': 'user',  # Original item features → become user features (user=job)
    }
    
    for orig_type, dest_type in swap_mapping.items():
        src_file = f"{orig_dir}/{orig_ds}.{orig_type}"
        dst_file = f"{simplex_dir}/{simplex_ds}.{dest_type}"
        
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
    
    # --- READ FULL QUAL TRAIN (original, user=applicant, item=job) ---
    train_src = f"{orig_dir}/{orig_ds}.train.inter"
    if not os.path.exists(train_src):
        raise FileNotFoundError(f"Missing source file: {train_src}")
    
    train_df = pd.read_csv(train_src, sep=",")
    user_col = next(c for c in train_df.columns if c.split(":")[0] == "user")  # applicant
    item_col = next(c for c in train_df.columns if c.split(":")[0] == "item")  # job
    label_col = get_label_col(train_df)

    # --- TRAIN POSITIVES: label=1, SWAP (job, applicant) ---
    print("\n📊 Building training data (JOB-CENTRIC, train positives + dummy negatives)...")
    pos_df = train_df[train_df[label_col] == 1][[item_col, user_col]].copy()
    pos_df["label:float"] = 1.0  # true positives

    # --- TRAIN NEGATIVES: label=0, SWAP (job, applicant) ---
    neg_df = train_df[train_df[label_col] == 0][[item_col, user_col]].copy()
    neg_df["label:float"] = 0.0  # dummy negatives (train)

    # Concatenate train positives + train negatives
    train_all = pd.concat([pos_df, neg_df], ignore_index=True)

    # --- ADD TEST NEGATIVES AS DUMMY ROWS (for vocab only) ---
    test_src = f"{orig_dir}/{orig_ds}.test.inter"
    if os.path.exists(test_src):
        test_df = pd.read_csv(test_src, sep=",")
        t_user_col, t_item_col = get_user_item_cols(test_df)
        t_label_col = get_label_col(test_df)

        test_neg = test_df[test_df[t_label_col] == 0][[t_item_col, t_user_col]].copy()
        test_neg["label:float"] = 0.0  # dummy negatives from TEST (NOT used as positives)
        test_neg.columns = ["user:token", "item:token", "label:float"]

        print(f"   ➕ Adding {len(test_neg)} test negatives as dummy rows to train.inter for vocab coverage")
        
        # First rename train_all to correct column names, THEN append test_neg
        train_all.columns = ["user:token", "item:token", "label:float"]
        train_all = pd.concat([train_all, test_neg], ignore_index=True)
    else:
        # Just rename for the normal case (no test_neg appended)
        train_all.columns = ["user:token", "item:token", "label:float"]

    out_path = f"{simplex_dir}/{simplex_ds}.train.inter"
    train_all.to_csv(out_path, index=False, sep='\t')

    # Debug summary
    n_true_pos = int((train_all["label:float"] == 1).sum())
    n_dummy_train_negs = int(
        ((train_all["label:float"] == 0) & (train_all["user:token"].isin(pos_df[item_col].astype(str)))).sum()
    )
    print(f"✅ Wrote training file: {out_path} ({len(train_all)} rows)")
    print(f"   True positives (label=1): {n_true_pos}")
    print(f"   Dummy negatives (label=0, train+test): {len(train_all) - n_true_pos}")
    print(f"   Unique jobs in train: {train_all['user:token'].nunique()}")
    print(f"   Unique applicants in train: {train_all['item:token'].nunique()}")
    
    # --- VAL / TEST (job-centric positives only) ---
    for split in ["val", "test"]:
        src = f"{orig_dir}/{orig_ds}.{split}.inter"
        out_path = f"{simplex_dir}/{simplex_ds}.{split}.inter"

        if not os.path.exists(src):
            if split == "val":
                print(f"⚠️  No original {split} file found at {src}. Creating empty {out_path} so RecBole doesn't crash.")
                with open(out_path, "w") as f:
                    f.write("user:token\titem:token\n")
                continue
            else:
                print(f"⚠️  Warning: Missing {split} file at {src}, skipping")
                continue

        df = pd.read_csv(src, sep=",")
        user_col_split, item_col_split = get_user_item_cols(df)
        label_col_split = get_label_col(df)

        pos = df[df[label_col_split] == 1][[item_col_split, user_col_split]].copy()  # job, applicant
        pos.columns = ["user:token", "item:token"]
        pos.to_csv(out_path, index=False, sep='\t')
        print(f"✅ Wrote {split} file: {out_path} ({len(pos)} rows, job-centric format)")
# ============================================================
# 2. Build labeled negatives mapping - JOB-CENTRIC
# ============================================================

def build_job_negatives(dataset, full_label_path):
    """
    Build mapping of job_id -> array of negative user_ids from labeled data.
    JOB-CENTRIC: For each job, find users who are labeled as negative.
    Returns dict: {internal_job_id: np.array([neg_applicant_ids])}
    
    CRITICAL: The .inter files have SWAPPED columns (user=job, item=applicant),
    but the ORIGINAL label file still has (user=applicant, item=job).
    """
    if not os.path.exists(full_label_path):
        raise FileNotFoundError(f"full_label_path not found: {full_label_path}")

    df = pd.read_csv(full_label_path, sep=",")
    user_col = next(c for c in df.columns if c.split(":")[0] == "user")  # Applicants in original file
    item_col = next(c for c in df.columns if c.split(":")[0] == "item")  # Jobs in original file
    label_col = get_label_col(df)

    # SWAPPED: In the dataset, "user" field = jobs, "item" field = applicants
    job_field = dataset.uid_field  # Jobs are in "user" field
    applicant_field = dataset.iid_field  # Applicants are in "item" field
    
    # Get token2id mappings - SWAPPED!
    applicant_token2id = dataset.field2token_id[applicant_field]  # Applicants mapped via "item" field
    job_token2id = dataset.field2token_id[job_field]  # Jobs mapped via "user" field

    # JOB-CENTRIC: Map job -> negative applicants
    job_neg = {}
    skipped = 0

    for applicant_raw, job_raw, y in zip(df[user_col], df[item_col], df[label_col]):
        if y != 0:  # Only want negatives
            continue

        applicant_tok = str(applicant_raw)
        job_tok = str(job_raw)
        
        # Check if tokens exist in mappings
        if applicant_tok not in applicant_token2id or job_tok not in job_token2id:
            skipped += 1
            continue

        applicant_id = applicant_token2id[applicant_tok]
        job_id = job_token2id[job_tok]

        # For this JOB, add this APPLICANT as a negative
        if job_id not in job_neg:
            job_neg[job_id] = set()
        job_neg[job_id].add(applicant_id)

    # Convert sets to numpy arrays
    job_neg_arr = {}
    total_negs = 0
    for job_id, applicants in job_neg.items():
        job_neg_arr[job_id] = np.array(list(applicants), dtype=np.int64)
        total_negs += len(applicants)

    avg_negs = total_negs / len(job_neg_arr) if job_neg_arr else 0
    print(f"✅ Custom negatives built for {len(job_neg_arr)} jobs (job-centric)")
    print(f"   Average negative applicants per job: {avg_negs:.1f}")
    print(f"   Total negative samples: {total_negs}")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} entries (users/items not in train set)")
    
    return job_neg_arr


# ============================================================
# 3. Custom DataLoader for SimpleX with labeled negatives
# ============================================================

class SimpleXLabeledNegDataLoader:
    """
    Job-centric dataloader for SimpleX.

    - dataset.inter_feat now contains BOTH:
        * true positives: label=1 (job, qualified applicant)
        * dummy negatives: label=0 (job, unqualified applicant)
    - We ONLY use label=1 rows as positive samples.
    - Dummy rows exist solely so RecBole assigns IDs to all applicants.
    """

    def __init__(self, config, dataset, job_negatives, num_negatives=49, batch_size=256, shuffle=True):
        self.config = config
        self.dataset = dataset
        self.job_negatives = job_negatives
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle

        # SWAPPED: jobs in uid_field, applicants in iid_field
        self.job_field = dataset.uid_field
        self.applicant_field = dataset.iid_field

        self.interactions = dataset.inter_feat
        self.n_samples_total = len(dataset)

        # Identify label field (we wrote label:float → 'label')
        self.label_field = None
        if 'label' in self.interactions:
            self.label_field = 'label'
        else:
            # fallback: treat all as positives if label missing
            self.label_field = None

        if self.label_field is not None:
            labels_np = self.interactions[self.label_field].cpu().numpy()
            self.pos_indices = np.where(labels_np == 1)[0]
        else:
            self.pos_indices = np.arange(self.n_samples_total)

        self.n_pos = len(self.pos_indices)

        # Global applicant pool (from ALL rows, pos + dummy)
        self.all_applicant_ids = np.unique(self.interactions[self.applicant_field].cpu().numpy())

        self._epoch_count = 0
        self._warned_about_sampling = False
        self._printed_first_batch = False

    def __len__(self):
        # Batches are over POSITIVE interactions only
        return (self.n_pos + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Use only positives as anchors
        indices = self.pos_indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)

        samples_using_fallback = 0
        samples_with_replacement = 0  # should stay 0
        samples_with_too_few_candidates = 0
        total_processed = 0
        
        # Track which jobs we're training on
        unique_jobs_in_epoch = set()
        jobs_with_labeled_negs = 0
        jobs_without_labeled_negs = 0

        for start_idx in range(0, self.n_pos, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_pos)
            batch_indices = indices[start_idx:end_idx]
            if len(batch_indices) == 0:
                continue

            batch_inter = self.interactions[batch_indices]
            job_ids = batch_inter[self.job_field].cpu().numpy()
            pos_applicant_ids = batch_inter[self.applicant_field].cpu().numpy()

            neg_applicants_list = []
            valid_indices = []

            for idx, job_id in enumerate(job_ids):
                pos_app_id = pos_applicant_ids[idx]
                unique_jobs_in_epoch.add(int(job_id))

                # 1) labeled negatives for this job
                neg_pool = self.job_negatives.get(int(job_id), None)
                used_fallback = False

                if neg_pool is None or len(neg_pool) == 0:
                    # Fallback to global pool
                    neg_pool = self.all_applicant_ids
                    used_fallback = True
                    jobs_without_labeled_negs += 1
                else:
                    jobs_with_labeled_negs += 1

                # Exclude the positive applicant
                candidate = neg_pool[neg_pool != pos_app_id]

                if candidate.size == 0:
                    samples_with_too_few_candidates += 1
                    continue

                k = min(self.num_negatives, candidate.size)
                if k < self.num_negatives:
                    samples_with_too_few_candidates += 1

                # Sample WITHOUT replacement from candidate pool
                sampled_negs = np.random.choice(candidate, size=k, replace=False)

                if used_fallback:
                    samples_using_fallback += 1

                neg_applicants_list.append(sampled_negs)
                valid_indices.append(idx)

            if len(neg_applicants_list) == 0:
                continue

            job_ids_filtered = job_ids[valid_indices]
            pos_applicant_ids_filtered = pos_applicant_ids[valid_indices]

            # Pad negatives to max length in batch for rectangular tensor shape
            max_k = max(len(negs) for negs in neg_applicants_list)
            padded_negs = []
            for negs in neg_applicants_list:
                if len(negs) < max_k:
                    pad_size = max_k - len(negs)
                    pad = np.random.choice(negs, size=pad_size, replace=True)
                    negs = np.concatenate([negs, pad])
                padded_negs.append(negs)

            neg_applicants_array = np.stack(padded_negs, axis=0)

            total_processed += len(job_ids_filtered)

            interaction = Interaction({
                self.job_field: torch.LongTensor(job_ids_filtered).to(self.config['device']),
                self.applicant_field: torch.LongTensor(pos_applicant_ids_filtered).to(self.config['device']),
                'neg_item': torch.LongTensor(neg_applicants_array).to(self.config['device']),
            })

            if not self._printed_first_batch:
                print(f"\n✅ First batch created (JOB-CENTRIC):")
                print(f"   Batch size (pos samples): {len(interaction[self.job_field])}")
                print(f"   Negatives per job (max in this batch): {neg_applicants_array.shape[1]}")
                print(f"   Using dummy negatives approach (label=0 rows ensure all applicants get IDs)")
                
                # Show first job's details
                first_job_id = int(job_ids_filtered[0])
                first_job_labeled_negs = self.job_negatives.get(first_job_id, None)
                if first_job_labeled_negs is not None:
                    print(f"   First job (internal ID={first_job_id}) has {len(first_job_labeled_negs)} LABELED negatives ✅")
                else:
                    print(f"   First job (internal ID={first_job_id}) has NO labeled negatives (using fallback) ⚠️")
                
                self._printed_first_batch = True

            yield interaction

        if not self._warned_about_sampling and self._epoch_count == 0:
            print(f"\n" + "="*80)
            print(f"EPOCH {self._epoch_count} TRAINING DATA VERIFICATION")
            print("="*80)
            print(f"✅ Training data statistics (JOB-CENTRIC):")
            print(f"   Total interactions in train dataset (pos+dummy): {self.n_samples_total}")
            print(f"   True positives used as anchors: {self.n_pos}")
            print(f"   Samples processed this epoch: {total_processed}")
            print(f"   Unique jobs trained on this epoch: {len(unique_jobs_in_epoch)}")
            print(f"   Jobs using LABELED negatives: {jobs_with_labeled_negs} ✅")
            print(f"   Jobs using fallback negatives: {jobs_without_labeled_negs} ⚠️")
            print(f"   Samples with <{self.num_negatives} candidates: {samples_with_too_few_candidates}")
            print("="*80 + "\n")
            self._warned_about_sampling = True

        self._epoch_count += 1


# ============================================================
# 4. Main: build data, train SimpleX with labeled negatives
# ============================================================

if __name__ == "__main__":
    # 1) Build implicit dataset for SimpleX
    build_simplex_dataset()

    # 2) Config for SimpleX
    config_dict = {
        'data_path': 'dataset',
        'USER_ID_FIELD': 'user',
        'ITEM_ID_FIELD': 'item',
        'NEG_PREFIX': 'neg_',
        
        'load_col': {
            'inter': ['user', 'item', 'label'],  # also load label:float
        },
        'field_separator': '\t',
        'benchmark_filename': ['train', 'val', 'test'],

        'train_neg_sample_args': {
            'distribution': 'uniform',
            'sample_num': NEGATIVES_PER_USER,
        },
        
        'neg_sampling': None,

        'epochs': 50,
        'train_batch_size': 68,
        'eval_batch_size': 68,
        'learning_rate': 1e-3,
        'stopping_step': 100,
        'eval_step': 100,

        'metrics': ['Recall', 'NDCG'],
        'topk': [1, 3, 5],
        'valid_metric': 'NDCG@5',

        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'RO',
            'mode': 'full',
        },

        'device': 'cuda',
        'show_progress': True,
        'checkpoint_dir': 'saved',
    }

    config = Config(model='SimpleX', dataset=simplex_ds, config_dict=config_dict)

    # 3) Create dataset and prepare data
    init_seed(config['seed'], reproducibility=True)
    init_logger(config)

    dataset = create_dataset(config)
    print(dataset)
    
    # MANUAL FEATURE LOADING
    print("\n📊 Manually loading user and item features...")
    print("="*80)
    
    # Load user features
    user_feat_file = f"{simplex_dir}/{simplex_ds}.user"
    if os.path.exists(user_feat_file):
        user_feat_df = pd.read_csv(user_feat_file, sep='\t')
        print(f"✅ Loaded user features: {len(user_feat_df)} users, {len(user_feat_df.columns)} columns")
        
        user_feat_dict = {}
        for col in user_feat_df.columns:
            field_name, field_type = col.split(':')
            
            if field_name not in dataset.field2type:
                if field_type == 'token':
                    dataset.field2type[field_name] = FeatureType.TOKEN
                    dataset.field2seqlen[field_name] = 1
                    dataset.field2source[field_name] = FeatureSource.USER
                elif field_type == 'float':
                    dataset.field2type[field_name] = FeatureType.FLOAT
                    dataset.field2source[field_name] = FeatureSource.USER
            
            if field_type == 'token':
                user_feat_dict[field_name] = torch.LongTensor([
                    dataset.field2token_id[dataset.uid_field].get(str(v), 0) 
                    for v in user_feat_df[col]
                ])
            elif field_type == 'float':
                user_feat_dict[field_name] = torch.FloatTensor(user_feat_df[col].fillna(0).values)
        
        dataset.user_feat = Interaction(user_feat_dict)
        print(f"   ✅ Injected {len(user_feat_dict)} user feature fields")
    
    # Load item features
    item_feat_file = f"{simplex_dir}/{simplex_ds}.item"
    if os.path.exists(item_feat_file):
        item_feat_df = pd.read_csv(item_feat_file, sep='\t')
        print(f"✅ Loaded item features: {len(item_feat_df)} items, {len(item_feat_df.columns)} columns")
        
        item_feat_dict = {}
        for col in item_feat_df.columns:
            field_name, field_type = col.split(':')
            
            if field_name not in dataset.field2type:
                if field_type == 'token':
                    dataset.field2type[field_name] = FeatureType.TOKEN
                    dataset.field2seqlen[field_name] = 1
                    dataset.field2source[field_name] = FeatureSource.ITEM
                elif field_type == 'float':
                    dataset.field2type[field_name] = FeatureType.FLOAT
                    dataset.field2source[field_name] = FeatureSource.ITEM
            
            if field_type == 'token':
                item_feat_dict[field_name] = torch.LongTensor([
                    dataset.field2token_id[dataset.iid_field].get(str(v), 0) 
                    for v in item_feat_df[col]
                ])
            elif field_type == 'float':
                item_feat_dict[field_name] = torch.FloatTensor(item_feat_df[col].fillna(0).values)
        
        dataset.item_feat = Interaction(item_feat_dict)
        print(f"   ✅ Injected {len(item_feat_dict)} item feature fields")
    
    print("="*80)

    try:
        train_data_default, valid_data, test_data = data_preparation(config, dataset)
    except TypeError as e:
        print(f"\n⚠️  Note: Test data preparation skipped (history fields issue)")
        from recbole.data.dataloader import TrainDataLoader
        train_data_default = TrainDataLoader(config, dataset, None, shuffle=True)
        valid_data = TrainDataLoader(config, dataset, None, shuffle=False)
        test_data = None

    # Build labeled negatives mapping - JOB-CENTRIC
    train_dataset = train_data_default.dataset
    job_negatives = build_job_negatives(train_dataset, full_train_path)

    # Create custom dataloader with labeled negatives - JOB-CENTRIC
    train_data = SimpleXLabeledNegDataLoader(
        config=config,
        dataset=train_dataset,
        job_negatives=job_negatives,
        num_negatives=NEGATIVES_PER_USER,
        batch_size=config['train_batch_size'],
        shuffle=True
    )
    
    print("\n" + "="*80)
    print("TRAINING DATA VERIFICATION (JOB-CENTRIC)")
    print("="*80)
    print(f"Total interactions in train dataset (pos+dummy): {len(train_dataset)}")
    print(f"True positives used as anchors: {len(train_data.pos_indices)}")
    print(f"Jobs with labeled negatives: {len(job_negatives)}")
    print(f"Format: Each POS sample is (job, positive_applicant) + K negative_applicants (K ≤ {NEGATIVES_PER_USER})")
    
    # Verify data source
    print(f"\n📊 Data source verification:")
    print(f"   Training positives from: {full_train_path}")
    print(f"   Expected: 68 positives (label=1) from hard_negative_label_qual.train.inter")
    print(f"   Actual loaded: {len(train_data.pos_indices)} positives")
    
    if len(train_data.pos_indices) == 68:
        print(f"   ✅ MATCH! Training on exactly 68 samples as expected")
    else:
        print(f"   ⚠️  MISMATCH! Expected 68 but got {len(train_data.pos_indices)}")
    
    # Show sample of actual jobs being trained on
    print(f"\n📋 Sample of jobs being trained on (showing first 5):")
    sample_indices = train_data.pos_indices[:5]
    for i, idx in enumerate(sample_indices):
        job_id = train_dataset.inter_feat[train_dataset.uid_field][idx].item()
        applicant_id = train_dataset.inter_feat[train_dataset.iid_field][idx].item()
        
        # Convert back to tokens
        job_token = train_dataset.id2token(train_dataset.uid_field, job_id)
        applicant_token = train_dataset.id2token(train_dataset.iid_field, applicant_id)
        
        # Check negatives
        neg_count = len(job_negatives.get(job_id, []))
        print(f"   {i+1}. Job {job_token}, Applicant {applicant_token}, Labeled negs: {neg_count}")
    
    # Show distribution of negatives per job
    neg_counts = [len(negs) for negs in job_negatives.values()]
    if neg_counts:
        print(f"\n📊 Labeled negatives distribution:")
        print(f"   Jobs with labeled negatives: {len(job_negatives)}")
        print(f"   Min negatives per job: {min(neg_counts)}")
        print(f"   Max negatives per job: {max(neg_counts)}")
        print(f"   Avg negatives per job: {sum(neg_counts)/len(neg_counts):.1f}")
        print(f"   This confirms negatives come from hard_negative_label_qual.train.inter ✅")
    
    print("="*80)
    
    # Prepare test data
    test_label_path = f"{orig_dir}/{orig_ds}.test.inter"
    test_df = pd.read_csv(test_label_path, sep=None, engine="python")

    test_user_col, test_item_col = get_user_item_cols(test_df)
    test_label_col = get_label_col(test_df)

    test_dataset = dataset  # Use main SimpleX dataset for token→id mappings

    # Model + Trainer
    model_cls = get_model(config['model'])
    model = model_cls(config, train_data.dataset).to(config['device'])

    trainer_cls = get_trainer(config['MODEL_TYPE'], config['model'])
    trainer = trainer_cls(config, model)

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    # Train without validation
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=None, show_progress=config['show_progress'])
    
    print("\n" + "="*80)
    print("MID-TRAINING VERIFICATION (After all epochs)")
    print("="*80)
    print(f"✅ Training completed on {train_data.n_pos} positive samples")
    print(f"   Expected: 68 samples from hard_negative_label_qual.train.inter")
    print(f"   Epochs completed: {train_data._epoch_count}")
    
    # Sample one batch to verify
    print("\n📊 Verifying trained data matches expected source...")
    sample_iter = iter(train_data)
    sample_batch = next(sample_iter)
    sample_job_id = int(sample_batch[train_data.job_field][0].item())
    
    # Check if this job has labeled negatives
    if sample_job_id in job_negatives:
        print(f"   ✅ Sample job (internal ID={sample_job_id}) has {len(job_negatives[sample_job_id])} LABELED negatives")
        print(f"   This confirms training used negatives from hard_negative_label_qual.train.inter")
    else:
        print(f"   ⚠️  Sample job (internal ID={sample_job_id}) has NO labeled negatives (used fallback)")
    
    print("="*80)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Save model
    model_save_path = f"saved/SimpleX-{simplex_ds}-labeled-negatives.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'dataset_name': simplex_ds,
    }, model_save_path)
    print(f"💾 Saved trained model to: {model_save_path}")

    # =====================================================================
    # JOB-CENTRIC EVALUATION ON TEST SET
    # =====================================================================
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (JOB-CENTRIC, label_qual)")
    print("="*80)

    print(f"Using labeled test file: {test_label_path}")
    print(f"\nNOTE: Field mappings due to swapping:")
    print(f"  - Original file: user_col=APPLICANTS, item_col=JOBS")
    print(f"  - RecBole dataset: uid_field=JOBS, iid_field=APPLICANTS (swapped!)")

    model.eval()

    applicant_token2id = test_dataset.field2token_id[test_dataset.iid_field]
    job_token2id = test_dataset.field2token_id[test_dataset.uid_field]
    job_field = test_dataset.uid_field
    applicant_field = test_dataset.iid_field

    all_ranks = []
    evaluated_jobs = 0
    skipped_jobs = []
    
    # Track test negatives verification
    jobs_evaluated_list = []
    total_test_negatives = 0
    total_test_positives = 0

    # Group by JOB (item column in original test file)
    job_groups = test_df.groupby(test_item_col)
    
    print(f"\n📊 Test data structure from {test_label_path}:")
    print(f"   Total jobs in test file: {test_df[test_item_col].nunique()}")
    print(f"   Expected to evaluate: 30 jobs")

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

            applicant_ids = []
            valid_indices = []
            for idx, applicant_tok in enumerate(applicants_tok):
                if applicant_tok in applicant_token2id:
                    applicant_ids.append(applicant_token2id[applicant_tok])
                    valid_indices.append(idx)

            if len(applicant_ids) == 0:
                skipped_jobs.append((job_tok, "no_applicants_in_mapping"))
                continue

            labels_valid = labels[valid_indices]
            pos_mask_valid = (labels_valid == 1)
            neg_mask_valid = (labels_valid == 0)

            if pos_mask_valid.sum() == 0:
                skipped_jobs.append((job_tok, "no_positive_applicants_after_filtering"))
                continue

            # Track this job
            jobs_evaluated_list.append(job_tok)
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
            
            # Print first evaluated job details
            if evaluated_jobs == 1:
                print(f"\n✅ First evaluated job: {job_tok}")
                print(f"   Positives (label=1): {pos_mask_valid.sum()}")
                print(f"   Negatives (label=0): {neg_mask_valid.sum()}")
                print(f"   This confirms evaluation uses negatives from {test_label_path}")

    print(f"\n" + "="*80)
    print("TEST EVALUATION VERIFICATION")
    print("="*80)
    print(f"✅ Job-centric test evaluation complete:")
    print(f"   Total jobs in test file: {test_df[test_item_col].nunique()}")
    print(f"   Expected: 30 jobs from hard_negative_label_qual.test.inter")
    print(f"   Evaluated jobs: {evaluated_jobs}")
    print(f"   Skipped jobs: {len(skipped_jobs)}")
    
    # Show which jobs were evaluated
    if len(jobs_evaluated_list) > 0:
        print(f"\n📋 First 5 jobs evaluated:")
        for i, job_tok in enumerate(jobs_evaluated_list[:5]):
            # Count positives and negatives for this job
            group = test_df[test_df[test_item_col] == job_tok]
            n_pos = (group[test_label_col] == 1).sum()
            n_neg = (group[test_label_col] == 0).sum()
            print(f"   {i+1}. Job {job_tok}: {n_pos} positive, {n_neg} negative applicants")
    
    if len(skipped_jobs) > 0:
        print(f"\n   Skipped job details:")
        for job_tok, reason in skipped_jobs:
            print(f"     - Job {job_tok}: {reason}")
    
    print(f"\n   Total positive applicants evaluated: {len(all_ranks)}")
    print(f"   Total test positives used (label=1): {total_test_positives}")
    print(f"   Total test negatives used (label=0): {total_test_negatives}")
    print(f"   This confirms test used LABELED negatives from the test file ✅")
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

    print(f"\nTest Results (job-centric):")
    for k in [1, 3, 5]:
        recall = recall_at_k(all_ranks, k)
        ndcg = ndcg_at_k(all_ranks, k)
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  NDCG@{k}:   {ndcg:.4f}")
    print("="*80)
#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.interaction import Interaction
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from recbole.utils import FeatureType, FeatureSource

# >>> ADD THIS BLOCK <<<
import scipy.sparse._dok as dok

def _dok_update(self, other):
    for (i, j), v in other.items():
        self[i, j] = v

if not hasattr(dok.dok_matrix, "_update"):
    dok.dok_matrix._update = _dok_update
    print("🔧 Patched scipy.sparse._dok.dok_matrix with ._update() for LightGCN")
# >>> END PATCH <<<

# ============================================================
# 0. CONSTANTS / PATHS
# ============================================================

"""
Goal:
- Train LightGCN on label_qual in a JOB-CENTRIC way:
    job  → RecBole "user"
    applicant → RecBole "item"

- Training:
    Only label=1 edges used to build the graph (LightGCN standard).
    BUT we override RecBole's neg-sampler with our OWN labeled
    negatives: for each job, we sample K negatives from label=0 rows.

- Evaluation:
    Custom job-centric eval using the full labeled test file:
    For each job:
      candidate applicants = {positives + all label=0 negatives for that job}
      rank the positive among these applicants.
"""

orig_ds = "hard_negative_label_qual"
orig_dir = f"dataset/{orig_ds}"

# New dataset for LightGCN
lgn_ds = "lightgcn_qual"
lgn_dir = f"dataset/{lgn_ds}"
os.makedirs(lgn_dir, exist_ok=True)

# Training config
NUM_NEGATIVES = 49
RANDOM_SEED = 42


# ============================================================
# 1. Helper functions
# ============================================================

def get_label_col(df: pd.DataFrame) -> str:
    """Return the column name that holds labels."""
    label_candidates = {"label", "label_pref", "label_qual"}
    for c in df.columns:
        base = c.split(":")[0]
        if base in label_candidates:
            return c
    raise ValueError(f"Label column not found. Columns = {list(df.columns)}")


def get_user_item_cols(df: pd.DataFrame):
    """Return (user_col, item_col) names from the dataframe."""
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


# ============================================================
# 2. Build LightGCN dataset (job-centric, positives only)
# ============================================================

def build_lightgcn_dataset():
    """
    Build dataset/lightgcn_qual/* for RecBole:

      - Copy .user / .item feature files from original dataset
      - Make .train/.val/.test.inter with:
          user:token  = JOB token   (original item_col)
          item:token  = APPLICANT token (original user_col)
          label:float = original label (0/1)

    Original hard_negative_label_qual.*.inter are:
      user = APPLICANT, item = JOB, label_qual ∈ {0,1}
    We SWAP these for job-centric training.
    """
    os.makedirs(lgn_dir, exist_ok=True)
    print(f"✅ Created dataset directory: {lgn_dir}")

    # --- Copy .user / .item features and normalize ID field names ---
    print("\n📊 Copying feature files (jobs/applicants)...")
    for feature_type in ["user", "item"]:
        src_file = f"{orig_dir}/{orig_ds}.{feature_type}"
        dst_file = f"{lgn_dir}/{lgn_ds}.{feature_type}"

        if not os.path.exists(src_file):
            print(f"  ⚠️  No {feature_type} feature file at {src_file}")
            continue

        with open(src_file, "r") as f:
            lines = f.readlines()

        if lines:
            header = lines[0].strip()
            print(f"\n  Processing {feature_type} header:")
            print(f"     Original: {header[:150]}...")
            id_field_original = header.split("\t")[0].split(":")[0]

            if id_field_original == f"{feature_type}_id":
                lines[0] = lines[0].replace(
                    f"{feature_type}_id:token", f"{feature_type}:token"
                )
                print(f"     ✅ Renamed '{feature_type}_id:token' → '{feature_type}:token'")
            elif id_field_original == feature_type:
                print(f"     ✅ ID field already '{feature_type}:token'")
            else:
                print(f"     ⚠️  Unexpected ID field: {id_field_original}")

        with open(dst_file, "w") as f:
            f.writelines(lines)

        print(f"  ✅ Copied features: {src_file} → {dst_file}")

    # --- Build .train/.val/.test.inter (job-centric, pos+neg with labels) ---
    # First pass: collect training job IDs to prevent leakage
    train_job_ids = None
    if os.path.exists(f"{orig_dir}/{orig_ds}.train.inter"):
        train_df = pd.read_csv(f"{orig_dir}/{orig_ds}.train.inter", sep=",")
        train_user_col, train_item_col = get_user_item_cols(train_df)
        train_job_ids = set(train_df[train_item_col].unique())
        print(f"\n✅ Collected {len(train_job_ids)} unique training jobs (will exclude from test)")
    
    for split in ["train", "val", "test"]:
        src = f"{orig_dir}/{orig_ds}.{split}.inter"
        dst = f"{lgn_dir}/{lgn_ds}.{split}.inter"

        if not os.path.exists(src):
            if split == "val":
                # Create empty val file with header
                print(f"⚠️  No {split} file at {src}. Creating empty {dst}")
                with open(dst, "w") as f:
                    f.write("user:token\titem:token\tlabel:float\n")
                continue
            else:
                print(f"⚠️  Missing {split} file at {src}, skipping")
                continue

        df = pd.read_csv(src, sep=",")
        user_col, item_col = get_user_item_cols(df)
        label_col = get_label_col(df)

        # Original: user = applicant, item = job.
        # Job-centric: user:token = job, item:token = applicant.
        jobcentric = df[[item_col, user_col, label_col]].copy()
        jobcentric.columns = ["user:token", "item:token", "label:float"]
        
        # CRITICAL: Verify no train/test leakage
        if split == "test" and train_job_ids is not None:
            test_jobs = set(jobcentric["user:token"].unique())
            overlap = train_job_ids & test_jobs
            if len(overlap) > 0:
                print(f"\n⚠️  ERROR: {len(overlap)} jobs appear in BOTH train and test!")
                print(f"   Overlapping jobs: {list(overlap)[:5]}...")
                raise ValueError("Train/test leakage detected!")

        jobcentric.to_csv(dst, index=False, sep="\t")
        print(
            f"✅ Wrote {split} file: {dst} "
            f"({(jobcentric['label:float']==1).sum()} positives, "
            f"{(jobcentric['label:float']==0).sum()} negatives, "
            f"{jobcentric['user:token'].nunique()} unique jobs)"
        )

    # === CRITICAL FIX: Add dummy rows for test jobs to train file ===
    # This ensures RecBole creates mappings for all jobs (train + test)
    print("\n🔧 Adding dummy interactions for test jobs AND applicants to train file...")
    
    train_dst = f"{lgn_dir}/{lgn_ds}.train.inter"
    test_dst = f"{lgn_dir}/{lgn_ds}.test.inter"
    
    if os.path.exists(train_dst) and os.path.exists(test_dst):
        train_df = pd.read_csv(train_dst, sep="\t")
        test_df = pd.read_csv(test_dst, sep="\t")
        
        # Get test jobs and applicants
        test_jobs = set(test_df["user:token"].unique())
        test_applicants = set(test_df["item:token"].unique())
        
        train_jobs = set(train_df["user:token"].unique())
        train_applicants = set(train_df["item:token"].unique())
        
        # Find dummy job/applicant for creating dummy rows
        dummy_job = train_df["user:token"].iloc[0]
        dummy_applicant = train_df["item:token"].iloc[0]
        
        # Create dummy rows
        dummy_rows = []
        
        # Dummy rows for test jobs (with a train applicant)
        for test_job in test_jobs:
            if test_job not in train_jobs:
                dummy_rows.append({
                    "user:token": test_job,
                    "item:token": dummy_applicant,
                    "label:float": 0.0  # Label 0 so it gets filtered out
                })
        
        # Dummy rows for test applicants (with a train job)
        for test_app in test_applicants:
            if test_app not in train_applicants:
                dummy_rows.append({
                    "user:token": dummy_job,
                    "item:token": test_app,
                    "label:float": 0.0  # Label 0 so it gets filtered out
                })
        
        if dummy_rows:
            dummy_df = pd.DataFrame(dummy_rows)
            train_with_dummies = pd.concat([train_df, dummy_df], ignore_index=True)
            train_with_dummies.to_csv(train_dst, index=False, sep="\t")
            
            test_jobs_added = sum(1 for r in dummy_rows if r["user:token"] in test_jobs)
            test_apps_added = sum(1 for r in dummy_rows if r["item:token"] in test_applicants)
            
            print(f"   Added {len(dummy_rows)} dummy rows total:")
            print(f"     - {test_jobs_added} for test jobs")
            print(f"     - {test_apps_added} for test applicants")
            print(f"   Total rows in train file: {len(train_with_dummies)}")
        else:
            print("   No dummy rows needed (all test entities already in train)")

    print("\n✅ LightGCN dataset build complete (job-centric, pos+neg+label)\n")
    
    # === DIAGNOSTIC: Check what's in the created files ===
    print("\n" + "="*80)
    print("FILE CONTENT DIAGNOSTIC")
    print("="*80)
    
    for split in ["train", "test"]:
        file_path = f"{lgn_dir}/{lgn_ds}.{split}.inter"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep="\t")
            unique_jobs = df["user:token"].nunique()
            unique_apps = df["item:token"].nunique()
            print(f"\n{split}.inter:")
            print(f"  Total rows: {len(df)}")
            print(f"  Unique jobs: {unique_jobs}")
            print(f"  Unique applicants: {unique_apps}")
            print(f"  Sample jobs: {sorted(df['user:token'].unique())[:10]}")
    
    print("="*80 + "\n")
    
    # === DIAGNOSTIC: Check for data leakage ===
    print("\n" + "="*80)
    print("DATA LEAKAGE DIAGNOSTIC")
    print("="*80)
    
    train_path = f"{lgn_dir}/{lgn_ds}.train.inter"
    test_path = f"{lgn_dir}/{lgn_ds}.test.inter"
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path, sep="\t")
        test_df = pd.read_csv(test_path, sep="\t")
        
        train_jobs = set(train_df["user:token"].unique())
        test_jobs = set(test_df["user:token"].unique())
        
        train_applicants = set(train_df["item:token"].unique())
        test_applicants = set(test_df["item:token"].unique())
        
        job_overlap = train_jobs & test_jobs
        applicant_overlap = train_applicants & test_applicants
        
        print(f"Train jobs: {len(train_jobs)}")
        print(f"Test jobs: {len(test_jobs)}")
        print(f"Job overlap: {len(job_overlap)}")
        if len(job_overlap) > 0:
            print(f"  ⚠️  Overlapping jobs: {sorted(list(job_overlap))[:10]}")
        
        print(f"\nTrain applicants: {len(train_applicants)}")
        print(f"Test applicants: {len(test_applicants)}")
        print(f"Applicant overlap: {len(applicant_overlap)}")
        
        # Check specific train/test edges
        train_edges = set(zip(train_df["user:token"], train_df["item:token"]))
        test_edges = set(zip(test_df["user:token"], test_df["item:token"]))
        edge_overlap = train_edges & test_edges
        print(f"\nTrain edges (job, applicant): {len(train_edges)}")
        print(f"Test edges (job, applicant): {len(test_edges)}")
        print(f"Edge overlap: {len(edge_overlap)}")
        if len(edge_overlap) > 0:
            print(f"  ⚠️  Overlapping edges: {list(edge_overlap)[:5]}")
        
        # Sample some train and test jobs
        print(f"\nSample train jobs: {sorted(list(train_jobs))[:10]}")
        print(f"Sample test jobs: {sorted(list(test_jobs))[:10]}")
        
    print("="*80 + "\n")


# ============================================================
# 3. Build job → labeled negative applicants mapping (train)
# ============================================================

def build_job_negatives(train_dataset, full_label_path):
    """
    Build mapping: job_id (internal) -> np.array(neg_applicant_ids)

    Uses full hard_negative_label_qual.train.inter:

      - For each row with label=0:
          job_tok = item_col (original job)
          applicant_tok = user_col (original applicant)
      - Map to internal ids via field2token_id from RecBole
      - Returns dict[int, np.ndarray]
    """
    if not os.path.exists(full_label_path):
        raise FileNotFoundError(full_label_path)

    df = pd.read_csv(full_label_path, sep=",")
    user_col, item_col = get_user_item_cols(df)
    label_col = get_label_col(df)

    job_field = train_dataset.uid_field     # "user" field in RecBole = JOB
    appl_field = train_dataset.iid_field    # "item" field in RecBole = APPLICANT

    job_token2id = train_dataset.field2token_id[job_field]
    appl_token2id = train_dataset.field2token_id[appl_field]

    job_neg = {}
    skipped = 0

    for _, row in df.iterrows():
        if row[label_col] != 0:
            continue

        job_tok = str(row[item_col])   # original item = job
        appl_tok = str(row[user_col])  # original user = applicant

        if job_tok not in job_token2id or appl_tok not in appl_token2id:
            skipped += 1
            continue

        j_id = job_token2id[job_tok]
        a_id = appl_token2id[appl_tok]

        if j_id not in job_neg:
            job_neg[j_id] = set()
        job_neg[j_id].add(a_id)

    job_neg_arr = {}
    total_negs = 0
    for j_id, items in job_neg.items():
        arr = np.array(list(items), dtype=np.int64)
        job_neg_arr[j_id] = arr
        total_negs += len(arr)

    avg_negs = total_negs / max(len(job_neg_arr), 1)
    print(f"✅ Labeled negatives built for {len(job_neg_arr)} jobs (train)")
    print(f"   Average negatives per job: {avg_negs:.1f}")
    print(f"   Total labeled negatives: {total_negs}")
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} rows (job/applicant not in mapping)")

    return job_neg_arr


# ============================================================
# 4. Custom DataLoader for LightGCN with labeled negatives
# ============================================================

class JobCentricLightGCNTrainDataLoader:
    """
    Custom dataloader for LightGCN (job-centric, K labeled negatives per positive).

    - RecBole "user" field = job (uid_field)
    - RecBole "item" field = positive applicant (iid_field)
    - We generate K triples per positive: (job, pos_app, neg_app)
      and feed LightGCN one negative per row as it expects.

    LightGCN loss stays unchanged; we only change what we feed into it.
    """

    def __init__(self, config, dataset, job_negatives,
                 k_per_pos=49, batch_size=256, shuffle=True):
        self.config = config
        self.dataset = dataset           # Trainer expects this
        self.job_negatives = job_negatives
        self.k_per_pos = k_per_pos
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.user_field = dataset.uid_field
        self.item_field = dataset.iid_field
        self.neg_field = f"{config['NEG_PREFIX']}{self.item_field}"  # 'neg_item'

        self.interactions = dataset.inter_feat
        self.device = config["device"]

        # ---- Precollect all positive (job, pos_app) pairs ----
        self.pos_pairs = []   # list of (job_id, pos_app_id)

        if isinstance(self.interactions, pd.DataFrame):
            job_ids = self.interactions[self.user_field].to_numpy()
            pos_ids = self.interactions[self.item_field].to_numpy()
        else:
            job_ids = self.interactions[self.user_field].cpu().numpy()
            pos_ids = self.interactions[self.item_field].cpu().numpy()

        for j_id, pos_id in zip(job_ids, pos_ids):
            j_id_int = int(j_id)
            # Only keep positives for jobs that actually have labeled negatives
            if j_id_int in self.job_negatives and len(self.job_negatives[j_id_int]) > 0:
                self.pos_pairs.append((j_id_int, int(pos_id)))

        if not self.pos_pairs:
            raise RuntimeError("No positive pairs with labeled negatives found for LightGCN training.")

        print("\n🔧 JobCentricLightGCNTrainDataLoader init:")
        print(f"   Positives with at least 1 labeled negative: {len(self.pos_pairs)}")
        print(f"   K per positive (training triples per pos): {self.k_per_pos}")

        # We'll generate triples fresh each epoch in __iter__
        self._epoch_count = 0
        self._printed_first_batch = False

    def __len__(self):
        # Rough upper bound; true number depends on k_per_pos and availability of negatives
        approx_triples = len(self.pos_pairs) * max(self.k_per_pos, 1)
        return (approx_triples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        For each epoch:
          - Re-sample up to K labeled negatives for each positive
          - Build a big list of (job, pos_app, neg_app) triples
          - Shuffle and batch them
        """
        users = []
        pos_items = []
        neg_items = []

        # Build triples
        for (j_id, pos_id) in self.pos_pairs:
            neg_candidates = self.job_negatives.get(j_id, None)
            if neg_candidates is None or len(neg_candidates) == 0:
                continue

            # If fewer than K, sample with replacement; else without
            if len(neg_candidates) >= self.k_per_pos:
                sampled = np.random.choice(neg_candidates, size=self.k_per_pos, replace=False)
            else:
                sampled = np.random.choice(neg_candidates, size=self.k_per_pos, replace=True)

            for neg_id in sampled:
                users.append(j_id)
                pos_items.append(pos_id)
                neg_items.append(int(neg_id))

        if not users:
            raise RuntimeError("No training triples constructed for this epoch (check job_negatives).")

        users = np.array(users, dtype=np.int64)
        pos_items = np.array(pos_items, dtype=np.int64)
        neg_items = np.array(neg_items, dtype=np.int64)

        # Shuffle triples
        indices = np.arange(len(users))
        if self.shuffle:
            np.random.shuffle(indices)

        users = users[indices]
        pos_items = pos_items[indices]
        neg_items = neg_items[indices]

        # Batch and yield Interaction objects
        for start in range(0, len(users), self.batch_size):
            end = min(start + self.batch_size, len(users))
            if start >= end:
                continue

            u_batch = torch.LongTensor(users[start:end]).to(self.device)
            p_batch = torch.LongTensor(pos_items[start:end]).to(self.device)
            n_batch = torch.LongTensor(neg_items[start:end]).to(self.device)

            interaction = Interaction({
                self.user_field: u_batch,
                self.item_field: p_batch,
                self.neg_field:  n_batch,
            })

            if not self._printed_first_batch:
                print("\n✅ First LightGCN training batch (job-centric, K labeled negs)")
                print(f"   Triples in batch: {len(u_batch)}")
                print(f"   Users tensor shape:      {interaction[self.user_field].shape}")
                print(f"   Pos items tensor shape:  {interaction[self.item_field].shape}")
                print(f"   Neg items tensor shape:  {interaction[self.neg_field].shape}  (should be [B])")
                self._printed_first_batch = True

            yield interaction

        self._epoch_count += 1


# ============================================================
# 5. Main: build data, train LightGCN, evaluate job-centrically
# ============================================================

if __name__ == "__main__":
    # 1) Build LightGCN dataset
    build_lightgcn_dataset()

    # 2) Config for LightGCN
    config_dict = {
        "data_path": "dataset",
        "USER_ID_FIELD": "user",
        "ITEM_ID_FIELD": "item",

        "NEG_PREFIX": "neg_",  # so neg field is 'neg_item'

        'load_col': {
            'inter': ['user', 'item', 'label'],   # <--- include label
        },
        "field_separator": "\t",

        # Load only train (which now contains dummy rows for test jobs)
        # This ensures all jobs get embeddings while keeping test data separate
        "benchmark_filename": ["train"],

        # We will override neg sampling with our custom loader
        "neg_sampling": None,
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": NUM_NEGATIVES,
        },

        "epochs": 50,
        "train_batch_size": 68,  # we'll override anyway via custom loader
        "eval_batch_size": 68,
        "learning_rate": 1e-3,
        "stopping_step": 100,
        "eval_step": 100,  # effectively disables RecBole's internal validation

        "metrics": ["Recall", "NDCG"],
        "topk": [1, 3, 5],
        "valid_metric": "NDCG@5",

        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },

        "device": "cuda",  # change to 'cpu' if no GPU
        "show_progress": True,
        "checkpoint_dir": "saved",
    }

    config = Config(
        model="LightGCN",
        dataset=lgn_ds,
        config_dict=config_dict,
    )

    # 3) Create dataset
    init_seed(config["seed"], reproducibility=True)
    init_logger(config)

    dataset = create_dataset(config)
    print(dataset)
    
    # === DIAGNOSTIC: What's in the RecBole dataset? ===
    print("\n" + "="*80)
    print("RECBOLE DATASET DIAGNOSTIC (BEFORE FILTERING)")
    print("="*80)
    
    # Check what RecBole actually loaded
    job_field = dataset.uid_field
    item_field = dataset.iid_field
    
    print(f"UID field (job): {job_field}")
    print(f"IID field (applicant): {item_field}")
    print(f"Total users in dataset: {dataset.user_num}")
    print(f"Total items in dataset: {dataset.item_num}")
    
    # Check job token mapping
    job_token2id = dataset.field2token_id[job_field]
    print(f"\nTotal jobs in token2id mapping: {len(job_token2id)}")
    print(f"Sample job tokens in mapping: {sorted(list(job_token2id.keys()))[:20]}")
    
    # Check if test jobs are in the mapping
    test_file_path = f"{orig_dir}/{orig_ds}.test.inter"
    if os.path.exists(test_file_path):
        test_check_df = pd.read_csv(test_file_path, sep=",")
        test_check_user_col, test_check_item_col = get_user_item_cols(test_check_df)
        test_check_job_tokens = test_check_df[test_check_item_col].astype(str).unique()
        
        test_jobs_in_mapping = sum(1 for tok in test_check_job_tokens if tok in job_token2id)
        print(f"\nTest jobs in RecBole mapping: {test_jobs_in_mapping} / {len(test_check_job_tokens)}")
        
        missing_test_jobs = [tok for tok in test_check_job_tokens if tok not in job_token2id]
        if missing_test_jobs:
            print(f"⚠️  Missing test jobs: {missing_test_jobs[:10]}")
    
    if isinstance(dataset.inter_feat, pd.DataFrame):
        print("\nType: DataFrame")
        print(f"Shape: {dataset.inter_feat.shape}")
        print(f"Columns: {list(dataset.inter_feat.columns)}")
        
        if job_field in dataset.inter_feat.columns:
            unique_users = dataset.inter_feat[job_field].nunique()
            print(f"Unique users (jobs) in inter_feat: {unique_users}")
    else:
        print("\nType: Interaction")
        print(f"Length: {len(dataset.inter_feat)}")
    
    print("="*80)

    print("\n" + "="*80)
    print("FILTERING INTERACTIONS TO LABEL=1 FOR LIGHTGCN GRAPH")
    print("="*80)
    
    if isinstance(dataset.inter_feat, pd.DataFrame):
        inter_df = dataset.inter_feat
        inter_dict = {}
        for col in inter_df.columns:
            vals = inter_df[col].values
            # choose tensor type
            if np.issubdtype(vals.dtype, np.integer):
                inter_dict[col] = torch.LongTensor(vals)
            else:
                inter_dict[col] = torch.FloatTensor(vals)
        dataset.inter_feat = Interaction(inter_dict)
        print("🔧 Converted dataset.inter_feat from DataFrame → Interaction")

    inter = dataset.inter_feat
    
    # ✅ FIX: Access the underlying dictionary with .interaction
    label_field = None
    if hasattr(inter, 'interaction'):
        available_fields = inter.interaction.keys()
    else:
        raise AttributeError("Interaction object has no .interaction attribute")
    
    for f in available_fields:
        if f.startswith("label"):
            label_field = f
            break

    if label_field is None:
        raise ValueError(f"No label field found in dataset.inter_feat. Fields: {list(available_fields)}")

    labels = inter[label_field]          # Tensor [num_interactions]
    pos_mask = (labels == 1)             # bool mask

    # 3) Keep only label=1 rows for the LightGCN graph
    old_num = len(inter)
    filtered = inter[pos_mask]
    new_num = len(filtered)
    
    # CRITICAL: Now we need to filter to ONLY TRAIN JOBS
    # Read the original train file to get train job tokens
    train_file_path = f"{orig_dir}/{orig_ds}.train.inter"
    train_df = pd.read_csv(train_file_path, sep=",")
    train_user_col, train_item_col = get_user_item_cols(train_df)
    train_job_tokens = set(train_df[train_item_col].astype(str).unique())
    
    print(f"\n🔧 Filtering to ONLY training jobs (prevent test leakage in graph)")
    print(f"   Training job tokens: {len(train_job_tokens)}")
    
    # Get job IDs from filtered interactions
    job_field = dataset.uid_field
    job_token2id = dataset.field2token_id[job_field]
    
    # Map train job tokens to IDs
    train_job_ids = set()
    for tok in train_job_tokens:
        if tok in job_token2id:
            train_job_ids.add(job_token2id[tok])
    
    print(f"   Training job IDs: {len(train_job_ids)}")
    
    # Filter to only train jobs
    filtered_jobs = filtered['user'].cpu().numpy()
    train_job_mask = np.array([jid in train_job_ids for jid in filtered_jobs])
    
    filtered_train_only = filtered[torch.from_numpy(train_job_mask)]
    
    print(f"   Positives before train-only filter: {len(filtered)}")
    print(f"   Positives after train-only filter: {len(filtered_train_only)}")
    
    dataset.inter_feat = filtered_train_only
    # Note: inter_num is a read-only property calculated from inter_feat length
    # So we don't need to set it manually

    print(f"   Original interactions (pos+neg, train+test): {old_num}")
    print(f"   Kept positives only (train jobs only): {len(filtered_train_only)}")
    print(f"   Dropped from graph: {old_num - len(filtered_train_only)}")
    print("="*80 + "\n")
    
    # === DIAGNOSTIC: What jobs are in the filtered dataset? ===
    print("\n" + "="*80)
    print("FILTERED DATASET DIAGNOSTIC (AFTER KEEPING ONLY POSITIVES)")
    print("="*80)
    
    filtered_job_ids = filtered_train_only['user'].cpu().numpy()
    unique_filtered_jobs = np.unique(filtered_job_ids)
    print(f"Unique jobs in filtered positives: {len(unique_filtered_jobs)}")
    print(f"Total positive edges: {len(filtered_train_only)}")
    print(f"Sample job IDs (internal): {unique_filtered_jobs[:10]}")
    
    # Map back to tokens
    job_id2token = {v: k for k, v in dataset.field2token_id[dataset.uid_field].items()}
    sample_job_tokens = [job_id2token.get(jid, f"UNKNOWN_{jid}") for jid in unique_filtered_jobs[:10]]
    print(f"Sample job tokens: {sample_job_tokens}")
    
    # CRITICAL: Check if test jobs are still in the token2id mapping
    print(f"\n🔍 Checking if test jobs are still in dataset.field2token_id:")
    test_check_job_tokens = test_check_df[test_check_item_col].astype(str).unique()
    job_token2id_after = dataset.field2token_id[dataset.uid_field]
    test_jobs_still_in_mapping = sum(1 for tok in test_check_job_tokens if tok in job_token2id_after)
    print(f"   Test jobs still in mapping: {test_jobs_still_in_mapping} / {len(test_check_job_tokens)}")
    
    if test_jobs_still_in_mapping < len(test_check_job_tokens):
        missing = [tok for tok in test_check_job_tokens if tok not in job_token2id_after]
        print(f"   ⚠️  MISSING test jobs after filtering: {missing[:10]}")
        print(f"   ⚠️  This is the problem! RecBole removed test jobs from mapping.")
    
    print("="*80 + "\n")
    
    print("\n" + "="*80)
    print("DATASET SUMMARY (LightGCN, job-centric)")
    print("="*80)
    print(f"Users (jobs):       {dataset.user_num}")
    print(f"Items (applicants): {dataset.item_num}")
    print(f"Positive edges:     {len(dataset)}")
    print("="*80)

    # 4) Build job → labeled negatives mapping from full train file
    full_train_path = f"{orig_dir}/{orig_ds}.train.inter"
    job_negatives = build_job_negatives(dataset, full_train_path)

    # === DIAGNOSTIC: What's in job_negatives? ===
    print("\n" + "="*80)
    print("JOB NEGATIVES DIAGNOSTIC")
    print("="*80)
    
    job_id2token = {v: k for k, v in dataset.field2token_id[dataset.uid_field].items()}
    
    print(f"Jobs with labeled negatives: {len(job_negatives)}")
    print(f"Sample job IDs with negatives (internal): {list(job_negatives.keys())[:10]}")
    
    sample_jobs_with_negs = [job_id2token.get(jid, f"UNKNOWN_{jid}") for jid in list(job_negatives.keys())[:10]]
    print(f"Sample job tokens with negatives: {sample_jobs_with_negs}")
    
    # Check which jobs DON'T have negatives
    all_train_jobs = np.unique(dataset.inter_feat['user'].cpu().numpy())
    jobs_with_negs = set(job_negatives.keys())
    jobs_without_negs = set(all_train_jobs) - jobs_with_negs
    
    print(f"\nJobs WITHOUT labeled negatives: {len(jobs_without_negs)}")
    if len(jobs_without_negs) > 0:
        sample_jobs_without = [job_id2token.get(jid, f"UNKNOWN_{jid}") for jid in list(jobs_without_negs)[:10]]
        print(f"Sample jobs without negatives: {sample_jobs_without}")
    
    print("="*80 + "\n")

    print("\n" + "="*80)
    print("TRAINING DATA VERIFICATION (LightGCN, job-centric + labeled negs)")
    print("="*80)
    print(f"Total positive edges in train_dataset: {len(dataset)}")
    print(f"Jobs with labeled negatives: {len(job_negatives)}")
    print("="*80)

    # 5) Create custom train dataloader
    train_data = JobCentricLightGCNTrainDataLoader(
        config=config,
        dataset=dataset,
        job_negatives=job_negatives,
        k_per_pos=49,           # <-- K labeled negatives per positive
        batch_size=256,         # or len(train_dataset), up to you
        shuffle=True,
    )

    # 6) Model + Trainer (LightGCN unchanged)
    model_cls = get_model(config["model"])
    model = model_cls(config, train_data.dataset).to(config["device"])

    trainer_cls = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_cls(config, model)

    print("\n" + "="*80)
    print("STARTING TRAINING (LightGCN + labeled job-centric negatives)")
    print("="*80)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data=None,  # disable RecBole validation
        show_progress=config["show_progress"],
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # 7) Save model
    model_save_path = f"saved/LightGCN-{lgn_ds}-labeled-negatives.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "dataset_name": lgn_ds,
        },
        model_save_path,
    )
    print(f"💾 Saved trained model to: {model_save_path}")

    # ============================================================
    # 8) Custom JOB-CENTRIC TEST EVALUATION
    # ============================================================

    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (JOB-CENTRIC, label_qual)")
    print("="*80)

    test_label_path = f"{orig_dir}/{orig_ds}.test.inter"
    test_df = pd.read_csv(test_label_path, sep=",")
    test_user_col, test_item_col = get_user_item_cols(test_df)
    test_label_col = get_label_col(test_df)

    print(f"Using labeled test file: {test_label_path}")
    print(f"\n📊 Test data from {test_label_path}:")
    print(f"   Total jobs in test file: {test_df[test_item_col].nunique()}")
    print(f"   Expected: 30 jobs")
    
    # === DIAGNOSTIC: Compare test jobs to train jobs ===
    print("\n" + "="*80)
    print("TEST vs TRAIN JOB COMPARISON")
    print("="*80)
    
    test_job_tokens = test_df[test_item_col].unique()
    print(f"Test job tokens (sample): {sorted(test_job_tokens)[:10]}")
    
    # Get train job tokens from the filtered dataset
    train_job_ids = dataset.inter_feat['user'].cpu().numpy()
    job_id2token = {v: k for k, v in dataset.field2token_id[dataset.uid_field].items()}
    train_job_tokens = set([job_id2token[int(jid)] for jid in np.unique(train_job_ids)])
    
    print(f"Train job tokens (sample): {sorted(list(train_job_tokens))[:10]}")
    
    test_jobs_set = set(test_job_tokens.astype(str))
    train_jobs_set = set(train_job_tokens)
    
    overlap = test_jobs_set & train_jobs_set
    print(f"\n⚠️  Test jobs that appear in train: {len(overlap)}")
    if len(overlap) > 0:
        print(f"   Overlapping jobs: {sorted(list(overlap))[:10]}")
    else:
        print("   ✅ No overlap - test jobs are completely separate from train jobs")
    
    # Check if test jobs are in job_negatives
    test_jobs_with_negs = 0
    for test_job_tok in test_job_tokens:
        if str(test_job_tok) in dataset.field2token_id[dataset.uid_field]:
            test_job_id = dataset.field2token_id[dataset.uid_field][str(test_job_tok)]
            if test_job_id in job_negatives:
                test_jobs_with_negs += 1
    
    print(f"\nTest jobs that have entries in job_negatives: {test_jobs_with_negs} / {len(test_job_tokens)}")
    if test_jobs_with_negs > 0:
        print("   ⚠️  WARNING: Test jobs should NOT be in job_negatives (train-only mapping)")
    
    print("="*80 + "\n")

    model.eval()

    # Note: test_dataset is same as training dataset (for token→id mapping)
    test_dataset = dataset
    applicant_token2id = test_dataset.field2token_id[test_dataset.iid_field]
    job_token2id = test_dataset.field2token_id[test_dataset.uid_field]
    job_field = test_dataset.uid_field
    applicant_field = test_dataset.iid_field

    all_ranks = []
    evaluated_jobs = 0
    skipped_jobs = []

    jobs_evaluated_list = []
    total_test_positives = 0
    total_test_negatives = 0

    job_groups = test_df.groupby(test_item_col)

    with torch.no_grad():
        for job_tok, group in job_groups:
            applicants_tok = group[test_user_col].astype(str).values
            labels = group[test_label_col].values

            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.sum() == 0:
                skipped_jobs.append((job_tok, "no_positive_applicants"))
                print(f"⚠️  Skipping job {job_tok}: has {len(group)} rows but no label=1")
                continue

            if str(job_tok) not in job_token2id:
                skipped_jobs.append((job_tok, "job_not_in_mapping"))
                continue

            applicant_ids = []
            valid_indices = []
            for idx, appl_tok in enumerate(applicants_tok):
                if appl_tok in applicant_token2id:
                    applicant_ids.append(applicant_token2id[appl_tok])
                    valid_indices.append(idx)
                else:
                    # Track which applicants are missing
                    if evaluated_jobs == 0:  # Only print for first job
                        print(f"⚠️  Job {job_tok}: applicant {appl_tok} not in mapping")

            if len(applicant_ids) == 0:
                skipped_jobs.append((job_tok, "no_applicants_in_mapping"))
                print(f"⚠️  Skipping job {job_tok}: none of {len(applicants_tok)} applicants are in mapping")
                continue

            labels_valid = labels[valid_indices]
            pos_mask_valid = labels_valid == 1
            neg_mask_valid = labels_valid == 0

            if pos_mask_valid.sum() == 0:
                skipped_jobs.append((job_tok, "no_positive_applicants_after_filtering"))
                print(f"⚠️  Skipping job {job_tok}: had {pos_mask.sum()} positives, but 0 after filtering (all positive applicants not in mapping)")
                print(f"   Original positive applicants: {applicants_tok[pos_mask][:5]}")
                continue

            jobs_evaluated_list.append(job_tok)
            total_test_positives += pos_mask_valid.sum()
            total_test_negatives += neg_mask_valid.sum()

            job_id = job_token2id[str(job_tok)]
            job_ids = [job_id] * len(applicant_ids)

            interaction_dict = {
                job_field: torch.LongTensor(job_ids).to(config["device"]),
                applicant_field: torch.LongTensor(applicant_ids).to(config["device"]),
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

            if evaluated_jobs == 1:
                print(f"\n✅ First evaluated job: {job_tok}")
                print(f"   Positives: {pos_mask_valid.sum()}, Negatives: {neg_mask_valid.sum()}")

    print(f"\n" + "="*80)
    print("TEST EVALUATION SUMMARY (JOB-CENTRIC, LightGCN)")
    print("="*80)
    print(f"   Total jobs in test: {test_df[test_item_col].nunique()}")
    print(f"   Evaluated jobs: {evaluated_jobs}")
    print(f"   Skipped jobs:   {len(skipped_jobs)}")

    if len(jobs_evaluated_list) > 0:
        print(f"\n📋 First 5 jobs evaluated:")
        for i, job_tok in enumerate(jobs_evaluated_list[:5]):
            group = test_df[test_df[test_item_col] == job_tok]
            n_pos = (group[test_label_col] == 1).sum()
            n_neg = (group[test_label_col] == 0).sum()
            print(f"   {i+1}. Job {job_tok}: {n_pos} positive, {n_neg} negative")

    if len(skipped_jobs) > 0:
        print(f"\n   Skipped jobs:")
        for job_tok, reason in skipped_jobs:
            print(f"     - {job_tok}: {reason}")

    print(f"\n   Total positive applicants evaluated: {len(all_ranks)}")
    print(f"   Test positives (label=1): {total_test_positives}")
    print(f"   Test negatives (label=0): {total_test_negatives}")
    print(f"   ✅ Used LABELED negatives from test file (per-job candidate sets)")
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

    print("\nTest Results (job-centric, LightGCN):")
    for k in [1, 3, 5]:
        r = recall_at_k(all_ranks, k)
        n = ndcg_at_k(all_ranks, k)
        print(f"  Recall@{k}: {r:.4f}")
        print(f"  NDCG@{k}:   {n:.4f}")
    print("="*80)
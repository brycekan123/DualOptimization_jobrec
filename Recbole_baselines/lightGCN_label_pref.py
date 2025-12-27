#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.interaction import Interaction
from recbole.utils import init_seed, init_logger, get_model, get_trainer

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
- Train LightGCN on label_pref in a USER-CENTRIC way:
    applicant → RecBole "user"
    job → RecBole "item"

- Training:
    Only label=1 edges used to build the graph (LightGCN standard).
    BUT we override RecBole's neg-sampler with our OWN labeled
    negatives: for each applicant, we sample K negatives from label=0 rows.

- Evaluation:
    Custom user-centric eval using the full labeled test file:
    For each applicant:
      candidate jobs = {positives + all label=0 negatives for that applicant}
      rank the positive among these jobs.
"""

orig_ds = "hard_negative_label_pref"
orig_dir = f"dataset/{orig_ds}"

# New dataset for LightGCN
lgn_ds = "lightgcn_pref"
lgn_dir = f"dataset/{lgn_ds}"
os.makedirs(lgn_dir, exist_ok=True)

# Training config
NUM_NEGATIVES = 49
NUM_TRAIN_BATCHES = 68
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
# 2. Build LightGCN dataset (user-centric, positives + negatives with labels)
# ============================================================

def build_lightgcn_dataset():
    """
    Build dataset/lightgcn_pref/* for RecBole:

      - Copy .user / .item feature files from original dataset
      - Make .train/.test.inter with:
          user:token  = APPLICANT token (original user_col)
          item:token  = JOB token (original item_col)
          label:float = original label (0/1)

    Original hard_negative_label_pref.*.inter are:
      user = APPLICANT, item = JOB, label_pref ∈ {0,1}
    We KEEP this structure (user-centric).
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

    # --- Build .train.inter (sample 68 batches from users with ≥49 negatives) ---
    train_src = f"{orig_dir}/{orig_ds}.train.inter"
    if os.path.exists(train_src):
        train_df = pd.read_csv(train_src, sep=",")
        user_col, item_col = get_user_item_cols(train_df)
        label_col = get_label_col(train_df)
        
        # Count negatives per user
        print("\n📊 Analyzing user negative counts...")
        negatives = train_df[train_df[label_col] == 0]
        user_neg_counts = negatives.groupby(user_col).size()
        
        # Filter to users with at least 49 negatives
        valid_users = user_neg_counts[user_neg_counts >= NUM_NEGATIVES].index.tolist()
        print(f"  Users with ≥{NUM_NEGATIVES} negatives: {len(valid_users)}/{len(user_neg_counts)}")
        
        # Sample 68 batches: each batch = 1 user with 1 positive
        # Only from users that have enough negatives
        print(f"\n📊 Sampling {NUM_TRAIN_BATCHES} training batches...")
        positives = train_df[train_df[label_col] == 1]
        positives = positives[positives[user_col].isin(valid_users)]
        
        # Group by user to ensure we sample complete batches
        user_groups = positives.groupby(user_col)
        all_batches = []
        
        for user_id, group in user_groups:
            for idx in range(len(group)):
                all_batches.append(group.iloc[idx])
        
        print(f"  Available batches from valid users: {len(all_batches)}")
        
        # Sample 68 batches with fixed seed
        np.random.seed(RANDOM_SEED)
        if len(all_batches) < NUM_TRAIN_BATCHES:
            print(f"⚠️  Warning: Only {len(all_batches)} batches available, requested {NUM_TRAIN_BATCHES}")
            sampled_batches = all_batches
        else:
            sampled_indices = np.random.choice(len(all_batches), size=NUM_TRAIN_BATCHES, replace=False)
            sampled_batches = [all_batches[i] for i in sampled_indices]
        
        train_pos = pd.DataFrame(sampled_batches)[[user_col, item_col]]
        
        # Analyze sampled batches
        unique_users = train_pos[user_col].nunique()
        print(f"  ✅ Sampled {len(train_pos)} batches from {unique_users} unique users")
        print(f"  Expected: 62 unique users (some users have multiple positives)")
        print(f"  All sampled users have ≥{NUM_NEGATIVES} labeled negatives")
        
        # Verify it's 62 unique users
        assert unique_users == 62, f"Expected 62 unique users, got {unique_users}"
        print(f"  ✅ Verified: {unique_users} unique users from {len(train_pos)} samples")
        
        # Also need the full train data with labels for building negatives mapping
        # Save both: sampled positives for training, full labeled data for negatives
        full_train_with_labels = train_df[[user_col, item_col, label_col]].copy()
        full_train_with_labels.columns = ["user:token", "item:token", "label:float"]
        full_train_labeled_path = f"{lgn_dir}/{lgn_ds}.train.labeled.inter"
        full_train_with_labels.to_csv(full_train_labeled_path, index=False, sep="\t")
        print(f"✅ Wrote full labeled train file: {full_train_labeled_path}")
        
        # Write training data (sampled positives only)
        dst = f"{lgn_dir}/{lgn_ds}.train.inter"
        train_pos.columns = ["user:token", "item:token"]
        train_pos.to_csv(dst, index=False, sep="\t")
        print(f"✅ Wrote training file: {dst} ({len(train_pos)} rows)")
    else:
        print(f"⚠️  Missing train file at {train_src}, skipping")
    
    # === Add dummy rows for test applicants AND jobs NOT in training ===
    # This ensures ALL test entities are in RecBole's mapping for evaluation
    print("\n🔧 Adding dummy rows for test applicants and jobs not in training...")
    
    train_dst = f"{lgn_dir}/{lgn_ds}.train.inter"
    test_src = f"{orig_dir}/{orig_ds}.test.inter"
    
    if os.path.exists(train_dst) and os.path.exists(test_src):
        train_df = pd.read_csv(train_dst, sep="\t")
        test_df = pd.read_csv(test_src, sep=",")
        
        test_user_col, test_item_col = get_user_item_cols(test_df)
        
        # Get test applicants and jobs
        test_applicants = set(test_df[test_user_col].astype(str).unique())
        test_jobs = set(test_df[test_item_col].astype(str).unique())
        
        train_applicants = set(train_df["user:token"].astype(str).unique())
        train_jobs = set(train_df["item:token"].astype(str).unique())
        
        # Find entities in test but not in train
        missing_applicants = test_applicants - train_applicants
        missing_jobs = test_jobs - train_jobs
        
        dummy_rows = []
        
        if missing_applicants:
            # Use a dummy job from training for missing applicants
            dummy_job = train_df["item:token"].iloc[0]
            for applicant in missing_applicants:
                dummy_rows.append({"user:token": applicant, "item:token": dummy_job})
        
        if missing_jobs:
            # Use a dummy applicant from training for missing jobs
            dummy_applicant = train_df["user:token"].iloc[0]
            for job in missing_jobs:
                dummy_rows.append({"user:token": dummy_applicant, "item:token": job})
        
        if dummy_rows:
            dummy_df = pd.DataFrame(dummy_rows)
            train_with_dummies = pd.concat([train_df, dummy_df], ignore_index=True)
            train_with_dummies.to_csv(train_dst, index=False, sep="\t")
            
            print(f"   Added {len(dummy_rows)} dummy rows:")
            print(f"     - {len(missing_applicants)} for test applicants")
            print(f"     - {len(missing_jobs)} for test jobs")
            print(f"   Total rows in train file: {len(train_with_dummies)}")
            print(f"   (68 real positives + {len(dummy_rows)} dummies for test entities)")
        else:
            print("   No dummy rows needed (all test entities in training)")
    
    # --- Build .test.inter (all positives) ---
    for split in ["test"]:
        src = f"{orig_dir}/{orig_ds}.{split}.inter"
        dst = f"{lgn_dir}/{lgn_ds}.{split}.inter"

        if not os.path.exists(src):
            print(f"⚠️  Missing {split} file at {src}, skipping")
            continue

        df = pd.read_csv(src, sep=",")
        user_col, item_col = get_user_item_cols(df)
        label_col = get_label_col(df)

        # Test: keep all positives
        pos = df[df[label_col] == 1][[user_col, item_col]].copy()
        pos.columns = ["user:token", "item:token"]
        pos.to_csv(dst, index=False, sep="\t")
        print(f"✅ Wrote {split} file: {dst} ({len(pos)} rows)")
        
        # Also save full labeled test for evaluation
        full_test_with_labels = df[[user_col, item_col, label_col]].copy()
        full_test_with_labels.columns = ["user:token", "item:token", "label:float"]
        full_test_labeled_path = f"{lgn_dir}/{lgn_ds}.{split}.labeled.inter"
        full_test_with_labels.to_csv(full_test_labeled_path, index=False, sep="\t")
        print(f"✅ Wrote full labeled {split} file: {full_test_labeled_path}")

    print("\n✅ LightGCN dataset build complete (68 sampled batches, ≥49 negs per user)\n")


# ============================================================
# 3. Build applicant → labeled negative jobs mapping (train)
# ============================================================

def build_applicant_negatives(train_dataset, full_label_path):
    """
    Build mapping: applicant_id (internal) -> np.array(neg_job_ids)
    Uses label=0 from training data as labeled negatives.
    Reads from .train.labeled.inter which has standardized columns.
    """
    if not os.path.exists(full_label_path):
        raise FileNotFoundError(full_label_path)

    df = pd.read_csv(full_label_path, sep="\t")  # Tab-separated now
    # Standardized column names
    user_col = "user:token"
    item_col = "item:token"
    label_col = "label:float"

    applicant_field = train_dataset.uid_field     # "user" field in RecBole = APPLICANT
    job_field = train_dataset.iid_field          # "item" field in RecBole = JOB

    applicant_token2id = train_dataset.field2token_id[applicant_field]
    job_token2id = train_dataset.field2token_id[job_field]

    applicant_neg = {}
    skipped = 0

    for _, row in df.iterrows():
        if row[label_col] != 0:
            continue

        applicant_tok = str(row[user_col])  # user = applicant
        job_tok = str(row[item_col])        # item = job

        if applicant_tok not in applicant_token2id or job_tok not in job_token2id:
            skipped += 1
            continue

        a_id = applicant_token2id[applicant_tok]
        j_id = job_token2id[job_tok]

        if a_id not in applicant_neg:
            applicant_neg[a_id] = set()
        applicant_neg[a_id].add(j_id)

    applicant_neg_arr = {}
    total_negs = 0
    for a_id, items in applicant_neg.items():
        arr = np.array(list(items), dtype=np.int64)
        applicant_neg_arr[a_id] = arr
        total_negs += len(arr)

    avg_negs = total_negs / max(len(applicant_neg_arr), 1)
    print(f"✅ Labeled negatives built for {len(applicant_neg_arr)} applicants (train)")
    print(f"   Average negatives per applicant: {avg_negs:.1f}")
    print(f"   Total labeled negatives: {total_negs}")
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} rows (applicant/job not in mapping)")

    return applicant_neg_arr


# ============================================================
# 4. Custom DataLoader for LightGCN with labeled negatives
# ============================================================

class UserCentricLightGCNTrainDataLoader:
    """
    Custom dataloader for LightGCN (user-centric, K labeled negatives per positive).

    - RecBole "user" field = applicant (uid_field)
    - RecBole "item" field = positive job (iid_field)
    - We generate K triples per positive: (applicant, pos_job, neg_job)
    - Negatives are sampled from LABELED negatives (label=0)
    """

    def __init__(self, config, dataset, applicant_negatives,
                 k_per_pos=49, batch_size=256, shuffle=True):
        self.config = config
        self.dataset = dataset
        self.applicant_negatives = applicant_negatives
        self.k_per_pos = k_per_pos
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.user_field = dataset.uid_field
        self.item_field = dataset.iid_field
        self.neg_field = f"{config['NEG_PREFIX']}{self.item_field}"

        self.interactions = dataset.inter_feat
        self.device = config["device"]

        self.pos_pairs = []

        if isinstance(self.interactions, pd.DataFrame):
            applicant_ids = self.interactions[self.user_field].to_numpy()
            pos_ids = self.interactions[self.item_field].to_numpy()
        else:
            applicant_ids = self.interactions[self.user_field].cpu().numpy()
            pos_ids = self.interactions[self.item_field].cpu().numpy()

        for a_id, pos_id in zip(applicant_ids, pos_ids):
            a_id_int = int(a_id)
            if a_id_int in self.applicant_negatives and len(self.applicant_negatives[a_id_int]) > 0:
                self.pos_pairs.append((a_id_int, int(pos_id)))

        if not self.pos_pairs:
            raise RuntimeError("No positive pairs with labeled negatives found for LightGCN training.")

        print("\n🔧 UserCentricLightGCNTrainDataLoader init:")
        print(f"   Total positives from dataset: {len(self.pos_pairs)}")
        print(f"   Expected: 68 (sampled at random seed 42)")
        print(f"   K per positive (training triples per pos): {self.k_per_pos}")
        
        assert len(self.pos_pairs) == 68, f"Expected 68 positives, got {len(self.pos_pairs)}!"
        print(f"   ✅ Verified: Exactly 68 positives for training")

        self._epoch_count = 0
        self._printed_first_batch = False

    def __len__(self):
        approx_triples = len(self.pos_pairs) * max(self.k_per_pos, 1)
        return (approx_triples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        users = []
        pos_items = []
        neg_items = []

        for (a_id, pos_id) in self.pos_pairs:
            neg_candidates = self.applicant_negatives.get(a_id, None)
            if neg_candidates is None or len(neg_candidates) == 0:
                continue

            if len(neg_candidates) >= self.k_per_pos:
                sampled = np.random.choice(neg_candidates, size=self.k_per_pos, replace=False)
            else:
                sampled = np.random.choice(neg_candidates, size=self.k_per_pos, replace=True)

            for neg_id in sampled:
                users.append(a_id)
                pos_items.append(pos_id)
                neg_items.append(int(neg_id))

        if not users:
            raise RuntimeError("No training triples constructed for this epoch.")

        users = np.array(users, dtype=np.int64)
        pos_items = np.array(pos_items, dtype=np.int64)
        neg_items = np.array(neg_items, dtype=np.int64)

        indices = np.arange(len(users))
        if self.shuffle:
            np.random.shuffle(indices)

        users = users[indices]
        pos_items = pos_items[indices]
        neg_items = neg_items[indices]

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
                print("\n✅ First LightGCN training batch (user-centric, K labeled negs)")
                print(f"   Triples in batch: {len(u_batch)}")
                print(f"   Users tensor shape:      {interaction[self.user_field].shape}")
                print(f"   Pos items tensor shape:  {interaction[self.item_field].shape}")
                print(f"   Neg items tensor shape:  {interaction[self.neg_field].shape}")
                self._printed_first_batch = True

            yield interaction

        self._epoch_count += 1


# ============================================================
# 5. Main: build data, train LightGCN, evaluate user-centrically
# ============================================================

if __name__ == "__main__":
    # 1) Build LightGCN dataset
    build_lightgcn_dataset()

    # 2) Config for LightGCN
    config_dict = {
        "data_path": "dataset",
        "USER_ID_FIELD": "user",
        "ITEM_ID_FIELD": "item",
        "NEG_PREFIX": "neg_",

        'load_col': {
            'inter': ['user', 'item'],  # Train file only has user/item (sampled positives)
        },
        "field_separator": "\t",

        # Load only train (which now contains dummy rows for test jobs/applicants)
        "benchmark_filename": ["train"],

        "neg_sampling": None,
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": NUM_NEGATIVES,
        },

        "epochs": 50,
        "train_batch_size": 256,
        "eval_batch_size": 256,
        "learning_rate": 1e-3,
        "stopping_step": 100,
        "eval_step": 100,

        "metrics": ["Recall", "NDCG"],
        "topk": [1, 3, 5],
        "valid_metric": "NDCG@5",

        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },

        "device": "cuda",
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

    print("\n" + "="*80)
    print("FILTERING TO 68 SAMPLED POSITIVES (removing dummy rows)")
    print("="*80)
    
    # Load the original 68 sampled positives to identify which ones to keep
    train_pos_file = f"{lgn_dir}/{lgn_ds}.train.inter"
    train_pos_df = pd.read_csv(train_pos_file, sep="\t")
    
    # The first 68 rows are the real sampled positives
    # (dummy rows were appended after)
    original_68_users = set(train_pos_df["user:token"].iloc[:68].astype(str).unique())
    original_68_pairs = set(
        train_pos_df.iloc[:68].apply(
            lambda row: (str(row["user:token"]), str(row["item:token"])), 
            axis=1
        )
    )
    
    print(f"Original 68 sampled pairs identified")
    print(f"From {len(original_68_users)} unique applicants")
    
    # Filter dataset.inter_feat to only include the original 68 pairs
    user_field = dataset.uid_field
    item_field = dataset.iid_field
    
    user_token2id = dataset.field2token_id[user_field]
    item_token2id = dataset.field2token_id[item_field]
    
    # Convert interaction to numpy for filtering
    if isinstance(dataset.inter_feat, pd.DataFrame):
        inter_df = dataset.inter_feat
        inter_dict = {}
        for col in inter_df.columns:
            vals = inter_df[col].values
            if np.issubdtype(vals.dtype, np.integer):
                inter_dict[col] = torch.LongTensor(vals)
            else:
                inter_dict[col] = torch.FloatTensor(vals)
        dataset.inter_feat = Interaction(inter_dict)
        print("🔧 Converted dataset.inter_feat from DataFrame → Interaction")
    
    inter = dataset.inter_feat
    user_ids = inter[user_field].cpu().numpy()
    item_ids = inter[item_field].cpu().numpy()
    
    # Build reverse mapping (id -> token)
    id2user_token = {v: k for k, v in user_token2id.items()}
    id2item_token = {v: k for k, v in item_token2id.items()}
    
    # Create mask for original 68 pairs
    keep_mask = []
    for u_id, i_id in zip(user_ids, item_ids):
        u_tok = id2user_token.get(u_id, None)
        i_tok = id2item_token.get(i_id, None)
        if u_tok and i_tok:
            pair = (str(u_tok), str(i_tok))
            keep_mask.append(pair in original_68_pairs)
        else:
            keep_mask.append(False)
    
    keep_mask = torch.tensor(keep_mask)
    filtered_inter = inter[keep_mask]
    
    dataset.inter_feat = filtered_inter
    
    print(f"Filtered from {len(inter)} to {len(filtered_inter)} interactions")
    print("="*80)

    print("\n" + "="*80)
    print("DATASET LOADED (68 sampled positives from users with ≥49 negatives)")
    print("="*80)
    print(f"Users (applicants):  {dataset.user_num}")
    print(f"Items (jobs):        {dataset.item_num}")
    print(f"Interactions:        {len(dataset)}")
    print("="*80)
    
    # CRITICAL: Verify exactly 68 interactions loaded
    assert len(dataset) == 68, f"Expected 68 interactions, got {len(dataset)}!"
    print("✅ Verified: Exactly 68 training interactions loaded\n")

    # 4) Build applicant → labeled negatives mapping
    # Use the full labeled train file (not the sampled positives file)
    full_train_labeled_path = f"{lgn_dir}/{lgn_ds}.train.labeled.inter"
    applicant_negatives = build_applicant_negatives(dataset, full_train_labeled_path)

    print("\n" + "="*80)
    print("TRAINING DATA VERIFICATION (LightGCN, user-centric + labeled negs)")
    print("="*80)
    print(f"Total positive edges in train_dataset: {len(dataset)}")
    print(f"Expected: 68 (sampled at random seed 42)")
    print(f"Applicants with labeled negatives: {len(applicant_negatives)}")
    print(f"Expected: 62 unique applicants")
    print("="*80)

    # 5) Create custom train dataloader
    train_data = UserCentricLightGCNTrainDataLoader(
        config=config,
        dataset=dataset,
        applicant_negatives=applicant_negatives,
        k_per_pos=49,
        batch_size=256,
        shuffle=True,
    )

    # 6) Model + Trainer
    model_cls = get_model(config["model"])
    model = model_cls(config, train_data.dataset).to(config["device"])

    trainer_cls = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_cls(config, model)

    print("\n" + "="*80)
    print("STARTING TRAINING (LightGCN + labeled user-centric negatives)")
    print("="*80)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data=None,
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
    # 8) Custom USER-CENTRIC TEST EVALUATION
    # ============================================================

    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (USER-CENTRIC, label_pref)")
    print("="*80)

    # Use the labeled test file
    test_label_path = f"{lgn_dir}/{lgn_ds}.test.labeled.inter"
    test_df = pd.read_csv(test_label_path, sep="\t")
    
    # Standardized column names
    test_user_col = "user:token"
    test_item_col = "item:token"
    test_label_col = "label:float"

    print(f"Using labeled test file: {test_label_path}")
    print(f"\n📊 Test data from {test_label_path}:")
    print(f"   Total applicants in test file: {test_df[test_user_col].nunique()}")

    model.eval()

    test_dataset = dataset
    job_token2id = test_dataset.field2token_id[test_dataset.iid_field]
    applicant_token2id = test_dataset.field2token_id[test_dataset.uid_field]
    applicant_field = test_dataset.uid_field
    job_field = test_dataset.iid_field

    all_ranks = []
    evaluated_applicants = 0
    skipped_applicants = []

    applicants_evaluated_list = []
    total_test_positives = 0
    total_test_negatives = 0

    applicant_groups = test_df.groupby(test_user_col)

    with torch.no_grad():
        for applicant_tok, group in applicant_groups:
            jobs_tok = group[test_item_col].astype(str).values
            labels = group[test_label_col].values

            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.sum() == 0:
                skipped_applicants.append((applicant_tok, "no_positive_jobs"))
                continue

            if str(applicant_tok) not in applicant_token2id:
                skipped_applicants.append((applicant_tok, "applicant_not_in_mapping"))
                continue

            # Cap negatives at 49
            neg_indices = np.where(neg_mask)[0]
            if len(neg_indices) > 49:
                np.random.seed(RANDOM_SEED)  # For reproducibility
                neg_indices = np.random.choice(neg_indices, size=49, replace=False)
            
            # Combine positives and capped negatives
            pos_indices = np.where(pos_mask)[0]
            selected_indices = np.concatenate([pos_indices, neg_indices])
            
            jobs_tok = jobs_tok[selected_indices]
            labels = labels[selected_indices]

            job_ids = []
            valid_indices = []
            for idx, job_tok in enumerate(jobs_tok):
                if job_tok in job_token2id:
                    job_ids.append(job_token2id[job_tok])
                    valid_indices.append(idx)

            if len(job_ids) == 0:
                skipped_applicants.append((applicant_tok, "no_jobs_in_mapping"))
                continue

            labels_valid = labels[valid_indices]
            pos_mask_valid = labels_valid == 1
            neg_mask_valid = labels_valid == 0

            if pos_mask_valid.sum() == 0:
                skipped_applicants.append((applicant_tok, "no_positive_jobs_after_filtering"))
                continue

            applicants_evaluated_list.append(applicant_tok)
            total_test_positives += pos_mask_valid.sum()
            total_test_negatives += neg_mask_valid.sum()

            applicant_id = applicant_token2id[str(applicant_tok)]
            applicant_ids = [applicant_id] * len(job_ids)

            interaction_dict = {
                applicant_field: torch.LongTensor(applicant_ids).to(config["device"]),
                job_field: torch.LongTensor(job_ids).to(config["device"]),
            }
            interaction = Interaction(interaction_dict)

            scores = model.predict(interaction).cpu().numpy()

            for idx_in_batch, is_pos in enumerate(pos_mask_valid):
                if not is_pos:
                    continue
                pos_score = scores[idx_in_batch]
                rank = (scores > pos_score).sum() + 1
                all_ranks.append(rank)

            evaluated_applicants += 1

            if evaluated_applicants == 1:
                print(f"\n✅ First evaluated applicant: {applicant_tok}")
                print(f"   Positives: {pos_mask_valid.sum()}, Negatives: {neg_mask_valid.sum()} (capped at 49)")

    print(f"\n" + "="*80)
    print("TEST EVALUATION SUMMARY (USER-CENTRIC, LightGCN)")
    print("="*80)
    print(f"   Total applicants in test: {test_df[test_user_col].nunique()}")
    print(f"   Evaluated applicants: {evaluated_applicants}")
    print(f"   Skipped applicants:   {len(skipped_applicants)}")

    if len(applicants_evaluated_list) > 0:
        print(f"\n📋 First 5 applicants evaluated:")
        for i, applicant_tok in enumerate(applicants_evaluated_list[:5]):
            group = test_df[test_df[test_user_col] == applicant_tok]
            n_pos = (group[test_label_col] == 1).sum()
            n_neg = (group[test_label_col] == 0).sum()
            print(f"   {i+1}. Applicant {applicant_tok}: {n_pos} positive, {n_neg} negative")

    if len(skipped_applicants) > 0:
        print(f"\n   Skipped applicants:")
        for applicant_tok, reason in skipped_applicants:
            print(f"     - {applicant_tok}: {reason}")

    print(f"\n   Total positive jobs evaluated: {len(all_ranks)}")
    print(f"   Test positives (label=1): {total_test_positives}")
    print(f"   Test negatives (label=0, capped at 49 per applicant): {total_test_negatives}")
    print(f"   ✅ Used LABELED negatives from test file (capped at 49 per applicant)")
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

    print("\nTest Results (user-centric, LightGCN, label_pref):")
    for k in [1, 3, 5]:
        r = recall_at_k(all_ranks, k)
        n = ndcg_at_k(all_ranks, k)
        print(f"  Recall@{k}: {r:.4f}")
        print(f"  NDCG@{k}:   {n:.4f}")
    print("="*80)
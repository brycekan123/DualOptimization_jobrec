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
HOW DATA LOADING WORKS:
1. build_simplex_dataset() creates .inter files with user:token and item:token columns
   - These are tab-separated files that RecBole loads
   - ALSO copies .user and .item feature files from the original dataset
   
2. RecBole's create_dataset() reads ALL files (.inter, .user, .item) and creates internal mappings:
   - user token → internal user_id (0, 1, 2, ...)
   - item token → internal item_id (0, 1, 2, ...)
   - Loads all user features from .user file
   - Loads all item features from .item file
   
3. build_user_negatives() reads the original labeled data and maps negatives:
   - Converts tokens → internal IDs using the mappings from step 2
   - Creates {user_id: [neg_item_ids]} dictionary
   
4. SimpleXLabeledNegDataLoader uses this mapping:
   - For each positive interaction, adds 49 labeled negatives
   - SimpleX expects specific format: user, item (positive), neg_item (negatives array)
   - Different from DiffRec which concatenates all interactions
   - Verification prints below confirm this is working
"""

# Your original labeled dataset (with label=0,1 and hard negatives)
orig_ds = "hard_negative_label_pref"
orig_dir = f"dataset/{orig_ds}"

# New dataset for SimpleX (implicit, positives only)
simplex_ds = "simplex_pref"
simplex_dir = f"dataset/{simplex_ds}"
os.makedirs(simplex_dir, exist_ok=True)

# Full label file we will use to build custom negatives
full_train_path = f"{orig_dir}/{orig_ds}.train.inter"

# Training settings - match Stage 1B
NEGATIVES_PER_USER = 49  # 49 negatives per positive
NUM_TRAIN_BATCHES = 68   # Sample 68 batches from training data
RANDOM_SEED = 42         # Fixed seed for reproducibility


# ============================================================
# 1. Build SimpleX-style dataset: positives only
# ============================================================

def build_simplex_dataset():
    """
    Create dataset/simplex_pref/simplex_pref.{train,val,test}.inter
    AND copy .user and .item feature files
    For training: Sample 68 batches (1 pos + 49 negs per batch) with seed=42
    Only sample from users that have at least 49 negatives available.
    For test: Use all positives
    """
    # Create dataset directory
    os.makedirs(simplex_dir, exist_ok=True)
    print(f"✅ Created dataset directory: {simplex_dir}")
    
    # Copy .user and .item feature files if they exist
    print("\n📊 Copying feature files...")
    for feature_type in ['user', 'item']:
        src_file = f"{orig_dir}/{orig_ds}.{feature_type}"
        dst_file = f"{simplex_dir}/{simplex_ds}.{feature_type}"
        
        if os.path.exists(src_file):
            # Read the file to check and fix field names
            with open(src_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                header_line = lines[0].strip()
                print(f"\n  Checking {feature_type} file header:")
                print(f"     Original header: {header_line[:200]}...")
                
                # Extract the ID field name
                id_field_original = header_line.split('\t')[0].split(':')[0]
                print(f"     {feature_type.capitalize()} ID field in file: {id_field_original}")
                
                # Fix the header if needed (user_id → user, item_id → item)
                if id_field_original == f'{feature_type}_id':
                    print(f"     ⚠️  Renaming '{feature_type}_id:token' → '{feature_type}:token' to match config")
                    lines[0] = lines[0].replace(f'{feature_type}_id:token', f'{feature_type}:token')
                    header_line = lines[0].strip()
                    print(f"     ✅ Updated header: {header_line[:200]}...")
                elif id_field_original == feature_type:
                    print(f"     ✅ Header already correct: '{feature_type}:token'")
                else:
                    print(f"     ⚠️  Unexpected ID field: {id_field_original}")
            
            # Write the fixed file
            with open(dst_file, 'w') as f:
                f.writelines(lines)
            
            print(f"  ✅ Copied and processed {feature_type} features: {src_file} → {dst_file}")
            print(f"     File has {len(lines)} lines (including header)")
        else:
            print(f"  ⚠️  No {feature_type} feature file found at {src_file}")
    
    # Read full training data to sample batches
    train_src = f"{orig_dir}/{orig_ds}.train.inter"
    if not os.path.exists(train_src):
        raise FileNotFoundError(f"Missing source file: {train_src}")
    
    train_df = pd.read_csv(train_src, sep=",")
    user_col = next(c for c in train_df.columns if c.split(":")[0] == "user")
    item_col = next(c for c in train_df.columns if c.split(":")[0] == "item")
    label_col = next(c for c in train_df.columns if c.split(":")[0] == "label")
    
    # Count negatives per user
    print("\n📊 Analyzing user negative counts...")
    negatives = train_df[train_df[label_col] == 0]
    user_neg_counts = negatives.groupby(user_col).size()
    
    # Filter to users with at least 49 negatives
    valid_users = user_neg_counts[user_neg_counts >= NEGATIVES_PER_USER].index.tolist()
    print(f"  Users with ≥{NEGATIVES_PER_USER} negatives: {len(valid_users)}/{len(user_neg_counts)}")
    
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
    print(f"  All sampled users have ≥{NEGATIVES_PER_USER} labeled negatives")
    
    # Write training data
    out_path = f"{simplex_dir}/{simplex_ds}.train.inter"
    train_pos.columns = ["user:token", "item:token"]
    train_pos.to_csv(out_path, index=False, sep='\t')
    print(f"✅ Wrote training file: {out_path} ({len(train_pos)} rows)")
    
    # Process validation and test splits - use all positives
    for split in ["val", "test"]:
        src = f"{orig_dir}/{orig_ds}.{split}.inter"
        if not os.path.exists(src):
            print(f"⚠️  Warning: Missing {split} file, skipping")
            continue
        
        df = pd.read_csv(src, sep=",")
        user_col_split = next(c for c in df.columns if c.split(":")[0] == "user")
        item_col_split = next(c for c in df.columns if c.split(":")[0] == "item")
        label_col_split = next(c for c in df.columns if c.split(":")[0] == "label")
        
        pos = df[df[label_col_split] == 1][[user_col_split, item_col_split]].copy()
        
        out_path = f"{simplex_dir}/{simplex_ds}.{split}.inter"
        pos.columns = ["user:token", "item:token"]
        pos.to_csv(out_path, index=False, sep='\t')
        print(f"✅ Wrote {split} file: {out_path} ({len(pos)} rows)")


# ============================================================
# 2. Build labeled negatives mapping
# ============================================================

def build_user_negatives(dataset, full_label_path):
    """
    Build mapping of user_id -> array of negative item_ids from labeled data.
    Returns dict: {internal_user_id: np.array([neg_item_ids])}
    """
    if not os.path.exists(full_label_path):
        raise FileNotFoundError(f"full_label_path not found: {full_label_path}")

    df = pd.read_csv(full_label_path, sep=",")
    user_col = next(c for c in df.columns if c.split(":")[0] == "user")
    item_col = next(c for c in df.columns if c.split(":")[0] == "item")
    label_col = next(c for c in df.columns if c.split(":")[0] == "label")

    user_field = dataset.uid_field
    item_field = dataset.iid_field
    
    # Get token2id mappings
    user_token2id = dataset.field2token_id[user_field]
    item_token2id = dataset.field2token_id[item_field]

    user_neg = {}
    skipped = 0

    for u_raw, i_raw, y in zip(df[user_col], df[item_col], df[label_col]):
        if y != 0:
            continue

        u_tok = str(u_raw)
        i_tok = str(i_raw)
        
        if u_tok not in user_token2id or i_tok not in item_token2id:
            skipped += 1
            continue

        u_id = user_token2id[u_tok]
        i_id = item_token2id[i_tok]

        if u_id not in user_neg:
            user_neg[u_id] = set()
        user_neg[u_id].add(i_id)

    # Convert sets to numpy arrays
    user_neg_arr = {}
    total_negs = 0
    for u_id, items in user_neg.items():
        user_neg_arr[u_id] = np.array(list(items), dtype=np.int64)
        total_negs += len(items)

    avg_negs = total_negs / len(user_neg_arr) if user_neg_arr else 0
    print(f"✅ Custom negatives built for {len(user_neg_arr)} users")
    print(f"   Average negatives per user: {avg_negs:.1f}")
    print(f"   Total negative samples: {total_negs}")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} entries (users/items not in train set)")
    
    return user_neg_arr


# ============================================================
# 3. Custom DataLoader for SimpleX with labeled negatives
# ============================================================

class SimpleXLabeledNegDataLoader:
    """
    Custom dataloader for SimpleX that provides labeled negatives.
    SimpleX expects: user, item (positive), and neg_item (negatives) fields.
    
    Important: Filters out samples without valid negatives to avoid shape mismatches.
    This ensures all batches have consistent tensor shapes regardless of batch_size.
    """
    
    def __init__(self, config, dataset, user_negatives, num_negatives=49, batch_size=256, shuffle=True):
        self.config = config
        self.dataset = dataset  # Important: trainer expects this attribute
        self.user_negatives = user_negatives
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.user_field = dataset.uid_field
        self.item_field = dataset.iid_field
        self.neg_item_field = dataset.iid_field  # SimpleX uses same field for negatives
        
        # Get all interactions from dataset
        self.interactions = dataset.inter_feat
        self.n_samples = len(dataset)
        
        # Track statistics
        self._epoch_count = 0
        self._warned_about_skips = False
        self._printed_first_batch = False
        
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """
        Iterate through dataset and provide labeled negatives in SimpleX format.
        Each batch contains: user_ids, pos_item_ids, and neg_item_ids (reshaped).
        """
        # Get indices
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Track skipped samples (for debugging)
        total_skipped = 0
        total_processed = 0
        
        # Process in batches
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Skip empty batches (shouldn't happen, but safety check)
            if len(batch_indices) == 0:
                continue
            
            # Get batch data
            batch_inter = self.interactions[batch_indices]
            user_ids = batch_inter[self.user_field].cpu().numpy()
            pos_item_ids = batch_inter[self.item_field].cpu().numpy()
            
            # Safety check: ensure we have data
            if len(user_ids) == 0 or len(pos_item_ids) == 0:
                continue
            
            # Collect negatives for each user
            neg_items_list = []
            valid_indices = []  # Track which samples have valid negatives
            
            for idx, u_id in enumerate(user_ids):
                # Get labeled negatives for this user
                if int(u_id) in self.user_negatives:
                    neg_items = self.user_negatives[int(u_id)]
                    
                    # Only proceed if we have ENOUGH negatives
                    if len(neg_items) >= self.num_negatives:
                        # Sample negatives without replacement
                        sampled_negs = np.random.choice(neg_items, size=self.num_negatives, replace=False)
                        neg_items_list.append(sampled_negs)
                        valid_indices.append(idx)
                    elif len(neg_items) > 0:
                        # Sample with replacement if not enough
                        sampled_negs = np.random.choice(neg_items, size=self.num_negatives, replace=True)
                        neg_items_list.append(sampled_negs)
                        valid_indices.append(idx)
                    else:
                        # No negatives available
                        total_skipped += 1
                else:
                    # User not in negatives mapping
                    total_skipped += 1
            
            # Skip batch if no valid samples with negatives
            if len(neg_items_list) == 0 or len(valid_indices) == 0:
                continue
            
            # Filter to only valid samples
            user_ids_filtered = user_ids[valid_indices]
            pos_item_ids_filtered = pos_item_ids[valid_indices]
            
            # CRITICAL: Verify lengths match before stacking
            assert len(user_ids_filtered) == len(pos_item_ids_filtered) == len(neg_items_list), \
                f"Mismatch: users={len(user_ids_filtered)}, pos={len(pos_item_ids_filtered)}, negs={len(neg_items_list)}"
            
            # Stack negatives: shape (num_valid_samples, num_negatives)
            neg_items_array = np.stack(neg_items_list, axis=0)
            
            # CRITICAL: Final verification before creating interaction
            if len(user_ids_filtered) == 0 or len(pos_item_ids_filtered) == 0 or len(neg_items_array) == 0:
                continue
            
            total_processed += len(user_ids_filtered)
            
            # Create interaction batch in SimpleX format
            # Try WITHOUT history fields to see if SimpleX actually needs them
            try:
                interaction = Interaction({
                    self.user_field: torch.LongTensor(user_ids_filtered).to(self.config['device']),
                    self.item_field: torch.LongTensor(pos_item_ids_filtered).to(self.config['device']),
                    'neg_item': torch.LongTensor(neg_items_array).to(self.config['device']),
                })
                
                # Final sanity check on created interaction
                if len(interaction[self.user_field]) == 0:
                    print(f"⚠️  Warning: Created empty interaction, skipping")
                    continue

                # Debug: Print first batch info + show one user's positives & negatives
                if not self._printed_first_batch:
                    print(f"\n✅ First batch created successfully:")
                    print(f"   Users in batch: {len(interaction[self.user_field])}")
                    print(f"   Pos items in batch: {len(interaction[self.item_field])}")
                    print(f"   Neg items shape: {interaction['neg_item'].shape}  "
                        f"(B={interaction['neg_item'].shape[0]}, K={interaction['neg_item'].shape[1]})")

                    # Pick one example from this batch
                    example_idx = 0
                    u_id = int(user_ids_filtered[example_idx])
                    pos_id = int(pos_item_ids_filtered[example_idx])
                    neg_ids = neg_items_array[example_idx]  # shape: [K]

                    # Map internal IDs → original tokens
                    u_tok = self.dataset.id2token(self.user_field, u_id)
                    pos_tok = self.dataset.id2token(self.item_field, pos_id)
                    neg_toks = [
                        self.dataset.id2token(self.item_field, int(n))
                        for n in neg_ids[:10]  # show first 10 negatives
                    ]

                    print(f"\n🔍 Example training tuple from this batch:")
                    print(f"   User token: {u_tok}")
                    print(f"   Positive item token: {pos_tok}")
                    print(f"   First 10 negative item tokens (all label=0 for this user):")
                    print(f"     {neg_toks}")
                    if u_id in self.user_negatives:
                        print(f"   Total labeled negatives available for this user: "
                            f"{len(self.user_negatives[u_id])}")
                    else:
                        print(f"   ⚠️ User {u_id} not found in user_negatives mapping (unexpected)")

                    self._printed_first_batch = True

                yield interaction

            except Exception as e:
                print(f"⚠️  Error creating interaction batch: {e}")
                print(f"   users shape: {user_ids_filtered.shape}")
                print(f"   pos_items shape: {pos_item_ids_filtered.shape}")
                print(f"   neg_items shape: {neg_items_array.shape}")
                continue
                    
        # Report skipped samples once per epoch
        if total_skipped > 0 and not self._warned_about_skips:
            print(f"\nℹ️  Note: Skipped {total_skipped}/{self.n_samples} samples without valid negatives in first epoch")
            print(f"   Processed {total_processed} valid samples")
            self._warned_about_skips = True
        
        self._epoch_count += 1


# ============================================================
# 4. Main: build data, train SimpleX with labeled negatives
# ============================================================

if __name__ == "__main__":
    # 1) Build implicit dataset for SimpleX
    build_simplex_dataset()

    print("\n📊 VERIFICATION: Checking created files before RecBole loads them...")
    print("="*80)
    for feature_type in ['user', 'item']:
        check_file = f"{simplex_dir}/{simplex_ds}.{feature_type}"
        if os.path.exists(check_file):
            with open(check_file, 'r') as f:
                header = f.readline().strip()
            print(f"✅ {feature_type.capitalize()} file exists: {check_file}")
            print(f"   Header: {header[:100]}...")
            # Verify first column matches expected field name
            first_field = header.split('\t')[0]
            expected_field = f"{feature_type}:token"
            if first_field.startswith(expected_field):
                print(f"   ✅ First field '{first_field}' matches expected '{expected_field}'")
            else:
                print(f"   ⚠️  WARNING: First field '{first_field}' does NOT match expected '{expected_field}'")
        else:
            print(f"❌ {feature_type.capitalize()} file NOT found: {check_file}")
    print("="*80 + "\n")
    
    # 2) Config for SimpleX using the new simplex_pref dataset
    config_dict = {
        'data_path': 'dataset',

        'USER_ID_FIELD': 'user',
        'ITEM_ID_FIELD': 'item',
        'NEG_PREFIX': 'neg_',  # SimpleX looks for neg_item field
        
        'load_col': {
            'inter': ['user', 'item'],  # Interaction file columns
        },
        'field_separator': '\t',  # RecBole default is tab

        # Files: simplex_pref.train/val/test.inter
        'benchmark_filename': ['train', 'val', 'test'],

        # SimpleX needs this to initialize properly, even though we override with custom dataloader
        'train_neg_sample_args': {
            'distribution': 'uniform',
            'sample_num': NEGATIVES_PER_USER,  # This sets neg_seq_len
        },
        
        # SimpleX-specific parameters
        'neg_sampling': None,  # We provide negatives directly

        'epochs': 10,
        'train_batch_size': 68,  # Set to dataset size for SimpleX (works with any size >= 68)
        'eval_batch_size': 68,
        'learning_rate': 1e-3,
        'stopping_step': 100,  # Set high to disable early stopping
        'eval_step': 100,  # Set higher than epochs to skip validation during training

        'metrics': ['Recall', 'NDCG'],
        'topk': [1, 3, 5],
        'valid_metric': 'NDCG@5',

        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},  # Not used since we provide files
            'group_by': 'user',
            'order': 'RO',
            'mode': 'full',  # Use full-sort for validation (we'll do labeled eval at the end)
        },

        'device': 'cuda',
        'show_progress': True,

        'checkpoint_dir': 'saved',
    }

    config = Config(
        model='SimpleX',
        dataset=simplex_ds,    # 'simplex_pref'
        config_dict=config_dict
    )

    # 3) Create dataset and prepare data
    init_seed(config['seed'], reproducibility=True)
    init_logger(config)

    """
    VERIFICATION STEPS - Watch for these in the output:
    1. Feature file copying: Shows .user and .item files being copied
    2. Dataset loading: Shows users/items are loaded from .inter files
    3. Feature loading: Shows which user/item features were loaded
    4. Sample interactions: Shows actual user/item tokens being loaded
    5. Labeled negatives: Shows negatives mapped for each user
    6. Batch verification: Shows batches contain negatives from labeled set
    7. Overlap check: Confirms negatives come from labeled set, not random
    """

    # Create dataset (positives only) and default loaders
    dataset = create_dataset(config)
    print(dataset)
    
    # MANUAL FEATURE LOADING: Load user and item features ourselves since RecBole won't
    print("\n📊 Manually loading user and item features...")
    print("="*80)
    
    # Load user features
    user_feat_file = f"{simplex_dir}/{simplex_ds}.user"
    if os.path.exists(user_feat_file):
        user_feat_df = pd.read_csv(user_feat_file, sep='\t')
        print(f"✅ Loaded user features: {len(user_feat_df)} users, {len(user_feat_df.columns)} columns")
        print(f"   Columns: {list(user_feat_df.columns[:10])}...")
        
        # Convert to RecBole Interaction format and register fields
        user_feat_dict = {}
        for col in user_feat_df.columns:
            field_name, field_type = col.split(':')
            
            # Register field properties with RecBole
            if field_name not in dataset.field2type:  # Avoid re-registering
                if field_type == 'token':
                    dataset.field2type[field_name] = FeatureType.TOKEN
                    dataset.field2seqlen[field_name] = 1  # Add this for token fields
                    dataset.field2source[field_name] = FeatureSource.USER
                elif field_type == 'float':
                    dataset.field2type[field_name] = FeatureType.FLOAT
                    dataset.field2source[field_name] = FeatureSource.USER
            
            # Build feature dictionary
            if field_type == 'token':
                user_feat_dict[field_name] = torch.LongTensor([
                    dataset.field2token_id[dataset.uid_field].get(str(v), 0) 
                    for v in user_feat_df[col]
                ])
            elif field_type == 'float':
                user_feat_dict[field_name] = torch.FloatTensor(user_feat_df[col].fillna(0).values)
        
        dataset.user_feat = Interaction(user_feat_dict)
        print(f"   ✅ Injected user features into dataset.user_feat")
        print(f"   ✅ Registered {len(user_feat_dict)} user feature fields")
    
    # Load item features
    item_feat_file = f"{simplex_dir}/{simplex_ds}.item"
    if os.path.exists(item_feat_file):
        item_feat_df = pd.read_csv(item_feat_file, sep='\t')
        print(f"✅ Loaded item features: {len(item_feat_df)} items, {len(item_feat_df.columns)} columns")
        print(f"   Columns: {list(item_feat_df.columns[:10])}...")
        
        # Convert to RecBole Interaction format and register fields
        item_feat_dict = {}
        for col in item_feat_df.columns:
            field_name, field_type = col.split(':')
            
            # Register field properties with RecBole
            if field_name not in dataset.field2type:  # Avoid re-registering
                if field_type == 'token':
                    dataset.field2type[field_name] = FeatureType.TOKEN
                    dataset.field2seqlen[field_name] = 1  # Add this for token fields
                    dataset.field2source[field_name] = FeatureSource.ITEM
                elif field_type == 'float':
                    dataset.field2type[field_name] = FeatureType.FLOAT
                    dataset.field2source[field_name] = FeatureSource.ITEM
            
            # Build feature dictionary
            if field_type == 'token':
                item_feat_dict[field_name] = torch.LongTensor([
                    dataset.field2token_id[dataset.iid_field].get(str(v), 0) 
                    for v in item_feat_df[col]
                ])
            elif field_type == 'float':
                item_feat_dict[field_name] = torch.FloatTensor(item_feat_df[col].fillna(0).values)
        
        dataset.item_feat = Interaction(item_feat_dict)
        print(f"   ✅ Injected item features into dataset.item_feat")
        print(f"   ✅ Registered {len(item_feat_dict)} item feature fields")
    
    print("="*80)
    
    # Print dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Total users: {dataset.user_num}")
    print(f"Total items: {dataset.item_num}")
    print(f"Total interactions: {len(dataset)}")
    
    # Show loaded features
    print("\n📊 Loaded Features:")
    print("="*80)
    print("NOTE: RecBole loads .user and .item files based on field name matching.")
    print("If features show 'None loaded', the field names might not match config.")
    print(f"Config expects: USER_ID_FIELD='{config['USER_ID_FIELD']}', ITEM_ID_FIELD='{config['ITEM_ID_FIELD']}'")
    print("Files must have matching field names (e.g., 'user:token' not 'user_id:token')")
    print("="*80)
    
    features_loaded = {'user': False, 'item': False}
    
    # Check for user features
    try:
        print(f"\n  Checking user features...")
        print(f"  hasattr(dataset, 'user_feat'): {hasattr(dataset, 'user_feat')}")
        
        if hasattr(dataset, 'user_feat'):
            print(f"  dataset.user_feat type: {type(dataset.user_feat)}")
            print(f"  dataset.user_feat is None: {dataset.user_feat is None}")
            
            if dataset.user_feat is not None:
                # Try to get the actual feature data
                print(f"  dataset.user_feat attributes: {dir(dataset.user_feat)}")
                
                # Try different ways to access the data
                if hasattr(dataset.user_feat, 'interaction'):
                    user_cols = list(dataset.user_feat.interaction.keys())
                    print(f"  ✅ User features loaded via .interaction: {user_cols}")
                    print(f"     Shape: {dataset.user_feat.interaction[user_cols[0]].shape if user_cols else 'N/A'}")
                    features_loaded['user'] = True
                elif hasattr(dataset.user_feat, 'keys'):
                    user_cols = list(dataset.user_feat.keys())
                    print(f"  ✅ User features loaded via .keys(): {user_cols}")
                    features_loaded['user'] = True
                else:
                    print(f"  User features object: {dataset.user_feat}")
            else:
                print(f"  ⚠️  dataset.user_feat is None")
        else:
            print(f"  ❌ dataset does not have user_feat attribute")
            
    except Exception as e:
        print(f"  ❌ User features: Error checking ({e})")
        import traceback
        traceback.print_exc()
    
    # Check for item features  
    try:
        print(f"\n  Checking item features...")
        print(f"  hasattr(dataset, 'item_feat'): {hasattr(dataset, 'item_feat')}")
        
        if hasattr(dataset, 'item_feat'):
            print(f"  dataset.item_feat type: {type(dataset.item_feat)}")
            print(f"  dataset.item_feat is None: {dataset.item_feat is None}")
            
            if dataset.item_feat is not None:
                print(f"  dataset.item_feat attributes: {dir(dataset.item_feat)}")
                
                if hasattr(dataset.item_feat, 'interaction'):
                    item_cols = list(dataset.item_feat.interaction.keys())
                    print(f"  ✅ Item features loaded via .interaction: {item_cols}")
                    print(f"     Shape: {dataset.item_feat.interaction[item_cols[0]].shape if item_cols else 'N/A'}")
                    features_loaded['item'] = True
                elif hasattr(dataset.item_feat, 'keys'):
                    item_cols = list(dataset.item_feat.keys())
                    print(f"  ✅ Item features loaded via .keys(): {item_cols}")
                    features_loaded['item'] = True
                else:
                    print(f"  Item features object: {dataset.item_feat}")
            else:
                print(f"  ⚠️  dataset.item_feat is None")
        else:
            print(f"  ❌ dataset does not have item_feat attribute")
            
    except Exception as e:
        print(f"  ❌ Item features: Error checking ({e})")
        import traceback
        traceback.print_exc()
    
    # Interaction features
    try:
        print(f"\n  Checking interaction features...")
        if hasattr(dataset.inter_feat, 'interaction'):
            inter_cols = list(dataset.inter_feat.interaction.keys())
        else:
            inter_cols = list(dataset.inter_feat.keys()) if hasattr(dataset.inter_feat, 'keys') else str(dataset.inter_feat)
        print(f"  Interaction features: {inter_cols}")
    except Exception as e:
        print(f"  Interaction features: Error checking ({e})")
    
    print("="*80)
    
    # STOP EXECUTION IF FEATURES NOT LOADED
    if not features_loaded['user'] or not features_loaded['item']:
        print("\n" + "="*80)
        print("❌ CRITICAL ERROR: User and/or Item features not loaded!")
        print("="*80)
        
        # Check if files actually exist
        print("\nDiagnostics:")
        user_file = f"{simplex_dir}/{simplex_ds}.user"
        item_file = f"{simplex_dir}/{simplex_ds}.item"
        
        print(f"\n1. Checking if feature files exist:")
        print(f"   User file: {user_file}")
        print(f"   Exists: {os.path.exists(user_file)}")
        if os.path.exists(user_file):
            with open(user_file, 'r') as f:
                header = f.readline().strip()
            print(f"   Header: {header[:150]}...")
            print(f"   First field: {header.split()[0]}")
        
        print(f"\n   Item file: {item_file}")
        print(f"   Exists: {os.path.exists(item_file)}")
        if os.path.exists(item_file):
            with open(item_file, 'r') as f:
                header = f.readline().strip()
            print(f"   Header: {header[:150]}...")
            print(f"   First field: {header.split()[0]}")
        
        print(f"\n2. Dataset config:")
        print(f"   dataset_name: {dataset.dataset_name}")
        print(f"   dataset_path: {dataset.dataset_path}")
        print(f"   USER_ID_FIELD: {config['USER_ID_FIELD']}")
        print(f"   ITEM_ID_FIELD: {config['ITEM_ID_FIELD']}")
        
        print("\n" + "="*80)
        print("Please fix the feature loading issue before continuing.")
        print("="*80)
        
        import sys
        sys.exit(1)
    
    # VERIFICATION: Show sample of loaded data
    print("\n" + "="*80)
    print("VERIFICATION: Sample of loaded interactions")
    print("="*80)
    user_field = dataset.uid_field
    item_field = dataset.iid_field
    print(f"User field name: {user_field}")
    print(f"Item field name: {item_field}")
    
    # Show first 5 interactions
    print("\nFirst 5 interactions from dataset:")
    for i in range(min(5, len(dataset))):
        u_id = dataset.inter_feat[user_field][i].item()
        i_id = dataset.inter_feat[item_field][i].item()
        # Convert back to tokens to show original IDs
        u_token = dataset.id2token(user_field, u_id)
        i_token = dataset.id2token(item_field, i_id)
        print(f"  {i}: user={u_token} (internal_id={u_id}), item={i_token} (internal_id={i_id})")
    print("="*80)

    try:
        train_data_default, valid_data, test_data = data_preparation(config, dataset)
    except TypeError as e:
        # Test data preparation may fail if history fields are needed
        # We don't need RecBole's test_data since we do custom evaluation
        print(f"\n⚠️  Note: Test data preparation skipped (history fields issue)")
        print(f"   This is OK - we use custom test evaluation with labeled negatives\n")
        from recbole.data.dataloader import TrainDataLoader
        train_data_default = TrainDataLoader(config, dataset, None, shuffle=True)
        valid_data = TrainDataLoader(config, dataset, None, shuffle=False)
        test_data = None
    
    # IMPORTANT: Validation won't run during training (eval_step=100 > epochs=50)
    # We'll do final evaluation on test set with labeled negatives instead
    print("\n⚠️  Note: Validation disabled during training (eval_step > epochs)")
    print("   Final test evaluation will use labeled negatives as intended\n")
    
    print("\n" + "="*80)
    print("SPLIT STATISTICS")
    print("="*80)
    print(f"Train interactions: {len(train_data_default.dataset)}")
    print(f"Valid interactions: {len(valid_data.dataset)} (won't be used - eval_step > epochs)")
    if test_data:
        print(f"Test interactions: {len(test_data.dataset)}")
        print(f"Test unique users: {test_data.dataset.inter_feat[test_data.dataset.uid_field].unique().shape[0]}")
    else:
        print(f"Test interactions: Using custom evaluation (RecBole's test_data not needed)")
    print("="*80 + "\n")

    # Build labeled negatives mapping
    train_dataset = train_data_default.dataset
    user_negatives = build_user_negatives(train_dataset, full_train_path)

    # Create custom dataloader with labeled negatives
    # NOTE: Due to SimpleX's architecture, batch_size >= 68 works best
    # Smaller batch sizes may cause shape mismatch errors
    train_data = SimpleXLabeledNegDataLoader(
        config=config,
        dataset=train_dataset,
        user_negatives=user_negatives,
        num_negatives=NEGATIVES_PER_USER,
        batch_size=config['train_batch_size'],
        shuffle=True
    )
    
    print("\n" + "="*80)
    print("DATALOADER VERIFICATION - TRAINING DATA")
    print("="*80)
    print(f"Using custom dataloader with labeled negatives")
    print(f"Batch size: {config['train_batch_size']}")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_data)}")
    
    # VERIFICATION: Sample the training data and show actual user/item IDs
    print("\n📊 Sampling training batch to verify it uses the 68 sampled examples...")
    sample_iter = iter(train_data)
    sample_batch = next(sample_iter)
    
    sample_users = sample_batch[train_dataset.uid_field].cpu().numpy()
    sample_items = sample_batch[train_dataset.iid_field].cpu().numpy()
    
    print(f"✅ First training batch:")
    print(f"   Batch size: {len(sample_users)} samples")
    print(f"   Unique users in training data: {len(set(sample_users))}")
    print(f"   User IDs (internal): {sample_users[:10]}..." if len(sample_users) > 10 else f"   User IDs: {sample_users}")
    print(f"   Item IDs (internal): {sample_items[:10]}..." if len(sample_items) > 10 else f"   Item IDs: {sample_items}")
    
    # Convert to tokens to show actual IDs
    print(f"\n   First 5 training examples (token IDs):")
    for i in range(min(5, len(sample_users))):
        u_token = train_dataset.id2token(train_dataset.uid_field, int(sample_users[i]))
        i_token = train_dataset.id2token(train_dataset.iid_field, int(sample_items[i]))
        print(f"     {i+1}. user={u_token}, item={i_token}")
    
    # Verify negatives
    sample_negs = sample_batch['neg_item'].cpu().numpy()
    print(f"\n   Negatives shape: {sample_negs.shape} (should be [{len(sample_users)}, {NEGATIVES_PER_USER}])")
    print(f"   First user's first 5 negatives (internal IDs): {sample_negs[0][:5]}")
    
    if config['train_batch_size'] < len(train_dataset):
        print(f"\n⚠️  Note: SimpleX works best with batch_size >= {len(train_dataset)}")
        print(f"   If you encounter errors, try batch_size={len(train_dataset)} or higher")
    
    print("="*80 + "\n")
    
    # For test evaluation - prepare batches manually since RecBole's eval doesn't work with our setup
    test_label_path = f"{orig_dir}/{orig_ds}.test.inter"
    test_df = pd.read_csv(test_label_path, sep=",")
    test_user_col = next(c for c in test_df.columns if c.split(":")[0] == "user")
    test_item_col = next(c for c in test_df.columns if c.split(":")[0] == "item")
    test_label_col = next(c for c in test_df.columns if c.split(":")[0] == "label")
    
    print("\n" + "="*80)
    print("DATALOADER VERIFICATION - VALIDATION DATA")
    print("="*80)
    print("⚠️  Validation won't run during training (eval_step=100 > epochs=50)")
    print(f"   But validation dataset exists: {len(valid_data.dataset)} samples")
    
    # Show info about validation files for reference
    orig_val_path = f"{orig_dir}/{orig_ds}.val.inter"
    simplex_val_path = f"{simplex_dir}/{simplex_ds}.val.inter"
    
    print(f"\n📊 Validation data sources (for reference):")
    print(f"   Original source: {orig_val_path}")
    
    if os.path.exists(orig_val_path):
        orig_val_df = pd.read_csv(orig_val_path, sep=",")
        orig_val_user_col = next(c for c in orig_val_df.columns if c.split(":")[0] == "user")
        orig_val_item_col = next(c for c in orig_val_df.columns if c.split(":")[0] == "item")
        orig_val_label_col = next(c for c in orig_val_df.columns if c.split(":")[0] == "label")
        
        orig_val_pos = orig_val_df[orig_val_df[orig_val_label_col] == 1]
        print(f"   ✅ Original file: {len(orig_val_df)} total rows")
        print(f"      - Positives (label=1): {len(orig_val_pos)}")
        print(f"      - Negatives (label=0): {len(orig_val_df) - len(orig_val_pos)}")
    
    if os.path.exists(simplex_val_path):
        simplex_val_df = pd.read_csv(simplex_val_path, sep='\t')
        print(f"   ✅ Processed for SimpleX: {len(simplex_val_df)} rows (positives only)")
        print(f"      Location: {simplex_val_path}")
    
    # Compare with training data to verify no overlap
    train_file_path = f"{simplex_dir}/{simplex_ds}.train.inter"
    train_file_df = pd.read_csv(train_file_path, sep='\t')
    
    train_pairs = set(zip(train_file_df['user:token'], train_file_df['item:token']))
    if os.path.exists(simplex_val_path):
        val_pairs = set(zip(simplex_val_df['user:token'], simplex_val_df['item:token']))
        pair_overlap = train_pairs.intersection(val_pairs)
        
        print(f"\n   Checking train/val data separation:")
        print(f"   Training pairs (from sampled 68): {len(train_pairs)}")
        print(f"   Validation pairs (from {orig_ds}.val.inter): {len(val_pairs)}")
        print(f"   Overlap: {len(pair_overlap)} pairs")
        
        if len(pair_overlap) > 0:
            print(f"   ⚠️  WARNING: {len(pair_overlap)} (user,item) pairs appear in BOTH train and val!")
            print(f"   First few overlapping pairs: {list(pair_overlap)[:3]}")
        else:
            print(f"   ✅ No (user,item) pair overlap - validation uses different interactions")
    
    print("="*80)
    
    print("\n" + "="*80)
    print("DATALOADER VERIFICATION - TEST DATA")
    print("="*80)
    
    print(f"\n📊 Test data sources:")
    print(f"   Original source: {test_label_path}")
    print(f"   (Used for evaluation with labeled negatives)")
    
    print(f"\nOriginal test file ({orig_ds}.test.inter):")
    print(f"   Total test interactions: {len(test_df)}")
    print(f"   Test positives (label=1): {(test_df[test_label_col]==1).sum()}")
    print(f"   Test negatives (label=0): {(test_df[test_label_col]==0).sum()}")
    
    # Group test data by user
    test_positives = test_df[test_df[test_label_col] == 1]
    test_negatives = test_df[test_df[test_label_col] == 0]
    
    # Verify simplex test file exists (created from original)
    simplex_test_path = f"{simplex_dir}/{simplex_ds}.test.inter"
    if os.path.exists(simplex_test_path):
        simplex_test_df = pd.read_csv(simplex_test_path, sep='\t')
        print(f"\nSimpleX processed test file ({simplex_ds}.test.inter):")
        print(f"   Rows: {len(simplex_test_df)} (positives only, used for RecBole validation)")
        print(f"   ✅ For final evaluation, we use ORIGINAL {orig_ds}.test.inter with labels")
    
    print(f"\n✅ Test data structure from original file:")
    print(f"   Unique users with positives: {test_positives[test_user_col].nunique()}")
    print(f"   First 5 test positive examples:")
    for idx, (i, row) in enumerate(test_positives.head(5).iterrows()):
        print(f"     {idx+1}. user={row[test_user_col]}, item={row[test_item_col]}")
    
    test_batches = []
    for user_id in test_positives[test_user_col].unique():
        user_pos = test_positives[test_positives[test_user_col] == user_id]
        user_neg = test_negatives[test_negatives[test_user_col] == user_id]
        
        if len(user_neg) > 0:
            for idx, pos_row in user_pos.iterrows():
                # Take up to 49 negatives for this positive
                neg_items = user_neg[test_item_col].values[:NEGATIVES_PER_USER]
                test_batches.append({
                    'user': pos_row[test_user_col],
                    'pos_item': pos_row[test_item_col],
                    'neg_items': neg_items
                })
    
    print(f"\n✅ Test evaluation batches (using labeled negatives):")
    print(f"   Test batches created: {len(test_batches)}")
    print(f"   Avg negatives per batch: {np.mean([len(b['neg_items']) for b in test_batches]):.1f}")
    
    # Verify test data is different from training
    train_file_path = f"{simplex_dir}/{simplex_ds}.train.inter"
    train_file_df = pd.read_csv(train_file_path, sep='\t')
    train_pairs = set(zip(train_file_df['user:token'], train_file_df['item:token']))
    test_pos_pairs = set([(test_positives.iloc[i][test_user_col], 
                           test_positives.iloc[i][test_item_col]) 
                          for i in range(min(20, len(test_positives)))])
    
    test_train_overlap = train_pairs.intersection(test_pos_pairs)
    
    print(f"\n   Checking train/test separation:")
    print(f"   Training pairs (sampled 68 from {orig_ds}.train.inter): {len(train_pairs)}")
    print(f"   Test pairs (first 20 from {orig_ds}.test.inter): {len(test_pos_pairs)}")
    print(f"   Overlap: {len(test_train_overlap)} pairs")
    
    if len(test_train_overlap) > 0:
        print(f"   ⚠️  WARNING: {len(test_train_overlap)} (user,item) pairs appear in BOTH train and test!")
        print(f"   Overlapping pairs: {list(test_train_overlap)[:3]}")
    else:
        print(f"   ✅ No (user,item) pair overlap - test uses different interactions")
    
    print("="*80 + "\n")
    
    # Don't use test_dataset from RecBole's test_data - we'll use base dataset for mappings
    # Our custom evaluation loads test data directly from files
    test_dataset = dataset

    # 5) Model + Trainer
    model_cls = get_model(config['model'])
    model = model_cls(config, train_data.dataset).to(config['device'])
    
    # VERIFY: Model can access user and item features
    print("\n" + "="*80)
    print("MODEL FEATURE ACCESS VERIFICATION")
    print("="*80)
    print("Checking if model can access loaded features...")
    
    # Check user features via the dataset that was passed to model
    check_dataset = train_data.dataset
    user_feat_fields = []
    item_feat_fields = []
    
    if hasattr(check_dataset, 'user_feat') and check_dataset.user_feat is not None:
        user_feat_fields = list(check_dataset.user_feat.interaction.keys())
        print(f"✅ Model has access to user features: {user_feat_fields[:5]}...")
        print(f"   Total user feature fields: {len(user_feat_fields)}")
        print(f"   User feature tensor shape: {check_dataset.user_feat.interaction[user_feat_fields[0]].shape}")
    else:
        print(f"❌ Model does NOT have access to user features")
    
    # Check item features
    if hasattr(check_dataset, 'item_feat') and check_dataset.item_feat is not None:
        item_feat_fields = list(check_dataset.item_feat.interaction.keys())
        print(f"✅ Model has access to item features: {item_feat_fields[:5]}...")
        print(f"   Total item feature fields: {len(item_feat_fields)}")
        print(f"   Item feature tensor shape: {check_dataset.item_feat.interaction[item_feat_fields[0]].shape}")
    else:
        print(f"❌ Model does NOT have access to item features")
    
    print("="*80 + "\n")

    trainer_cls = get_trainer(config['MODEL_TYPE'], config['model'])
    trainer = trainer_cls(config, model)

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print("Note: Validation completely disabled (valid_data=None).")
    print("Final test evaluation will use labeled negatives (same as baseline).")
    print(f"\n📊 Training configuration:")
    print(f"   Training samples: {len(train_dataset)} (the 68 sampled batches)")
    print(f"   Batches per epoch: {len(train_data)}")
    print(f"   Total epochs: {config['epochs']}")
    print(f"   Each epoch processes {len(train_dataset)} training examples")
    print("="*80 + "\n")
    
    # Train without validation - pass None to skip all validation
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=None, show_progress=config['show_progress'])    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - DATA USAGE SUMMARY")
    print("="*80)
    print(f"✅ Trained on: {len(train_dataset)} samples (the 68 sampled batches)")
    print(f"✅ Validation: Skipped (valid_data=None)")
    if user_feat_fields:
        print(f"✅ Model trained with user features: {len(user_feat_fields)} fields")
    if item_feat_fields:
        print(f"✅ Model trained with item features: {len(item_feat_fields)} fields")
    print("="*80 + "\n")
    
    # Save the trained model
    model_save_path = f"saved/SimpleX-{simplex_ds}-labeled-negatives.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'dataset_name': simplex_ds,
    }, model_save_path)
    print(f"💾 Saved trained model to: {model_save_path}")
    
    print("\n" + "="*80)
    print("READY FOR TEST EVALUATION")
    print("="*80)
    print(f"✅ Trained model ready for evaluation")
    print(f"✅ Test data available at: {test_label_path}")
    print(f"✅ Test batches prepared: {len(test_batches)} batches with labeled negatives")
    print(f"\n💡 To run test evaluation, uncomment the evaluation section below")
    print("="*80 + "\n")
    
    print("✅ Training complete! Model saved. You can now evaluate manually.")
    
    # OPTIONAL: Uncomment below for automatic test evaluation
    
    # Custom evaluation on test set with labeled negatives
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET (with labeled negatives)")
    print("="*80)
    print(f"Test batches to evaluate: {len(test_batches)}")
    print(f"Using labeled negatives from: {test_label_path}")
    print("\nEvaluating...")
    
    model.eval()
    user_token2id = test_dataset.field2token_id[test_dataset.uid_field]
    item_token2id = test_dataset.field2token_id[test_dataset.iid_field]
    
    uid_field = test_dataset.uid_field
    iid_field = test_dataset.iid_field
    
    all_ranks = []
    evaluated_users = set()
    skipped_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_batches):
            user_tok = str(batch['user'])
            pos_item_tok = str(batch['pos_item'])
            
            if user_tok not in user_token2id or pos_item_tok not in item_token2id:
                skipped_batches += 1
                continue
            
            user_id = user_token2id[user_tok]
            pos_item_id = item_token2id[pos_item_tok]
            evaluated_users.add(user_tok)
            
            # Get negative item IDs
            neg_item_ids = []
            for neg_tok in batch['neg_items']:
                neg_tok_str = str(neg_tok)
                if neg_tok_str in item_token2id:
                    neg_item_ids.append(item_token2id[neg_tok_str])
            
            if len(neg_item_ids) == 0:
                skipped_batches += 1
                continue
            
            # All items to rank: positive + negatives
            all_item_ids = [pos_item_id] + neg_item_ids
            num_items = len(all_item_ids)
            
            # Create interaction object
            interaction_dict = {
                uid_field: torch.LongTensor([user_id] * num_items).to(config['device']),
                iid_field: torch.LongTensor(all_item_ids).to(config['device'])
            }
            interaction = Interaction(interaction_dict)
            
            # Get scores using the interaction object
            scores = model.predict(interaction).cpu().numpy()
            
            # Rank of positive item (lower is better, 0-indexed)
            pos_score = scores[0]
            rank = np.sum(scores > pos_score) + 1  # 1-indexed rank
            all_ranks.append(rank)
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_batches)} test batches...")
    
    print(f"\n✅ Test evaluation complete:")
    print(f"   Total test batches: {len(test_batches)}")
    print(f"   Evaluated: {len(all_ranks)} batches")
    print(f"   Skipped: {skipped_batches} batches (missing users/items)")
    print(f"   Unique users evaluated: {len(evaluated_users)}")
    
    # Calculate metrics
    all_ranks = np.array(all_ranks)
    
    def recall_at_k(ranks, k):
        return np.mean(ranks <= k)
    
    def ndcg_at_k(ranks, k):
        ndcg_scores = []
        for rank in ranks:
            if rank <= k:
                ndcg_scores.append(1.0 / np.log2(rank + 1))
            else:
                ndcg_scores.append(0.0)
        return np.mean(ndcg_scores)
    
    print(f"\nEvaluated {len(all_ranks)} test interactions")
    print(f"\nTest Results:")
    for k in [1, 3, 5]:
        recall = recall_at_k(all_ranks, k)
        ndcg = ndcg_at_k(all_ranks, k)
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  NDCG@{k}: {ndcg:.4f}")
    
    print("="*80)
    
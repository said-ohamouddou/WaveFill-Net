"""
@Author: Said Ohamouddou
@File: TreeSpeciesDatasetHELIALS.py
@Time: 2025/02/23
"""
"""
HeliALS Dataset Loader (16D version)
Loads pre-processed tree species point cloud data from HeliALS_fps_8192_16D.
Supports train/val/test partitions based on test_set_flag in CSV.
16D features: XYZ + multi-wavelength (1550nm, 905nm, 532nm) intensity, amplitude, reflectance, deviation + return_number
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from tqdm import tqdm
try:
    from .build import DATASETS
except ImportError:
    # Fallback when this file is imported outside its package.
    class _NullRegistry:
        @staticmethod
        def register_module():
            return lambda cls: cls
    DATASETS = _NullRegistry()
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Species code to name mapping (from HeliALS dataset)
SPECIES_NAMES = {
    1: 'Pine',    # Pinus sylvestris
    2: 'Spruce',  # Picea sp.
    3: 'Birch',   # Betula sp.
    4: 'Maple',   # Acer platanoides
    5: 'Aspen',   # Populus tremula
    6: 'Rowan',   # Sorbus sp.
    7: 'Oak',     # Quercus robur
    8: 'Linden',  # Tilia sp.
    9: 'Alder',   # Alnus sp.
}

# 16D feature names
FEATURE_NAMES_16D = [
    'x', 'y', 'z',
    'intensity_1550', 'intensity_905', 'intensity_532',
    'amplitude_1550', 'amplitude_905', 'amplitude_532',
    'reflectance_1550', 'reflectance_905', 'reflectance_532',
    'deviation_1550', 'deviation_905', 'deviation_532',
    'return_number'
]


def load_helials_npy_data(data_path, csv_path, final_csv_path=None):
    """Load HeliALS point cloud data from pre-processed .npy files.

    Args:
        data_path: Path to the HeliALS_fps_8192_16D directory containing .npy files
        csv_path: Path to the CSV file with train/test metadata
                  (training-and-test-segments-with-species.csv — has test_set_flag)
        final_csv_path: Optional path to final-segments-with-species.csv. If given,
                        segments NOT present in this CSV are dropped. This excludes
                        the 88 segments flagged by the paper as non-optimal profile
                        images, sparse point clouds, or building intersections
                        (paper Section 4.4).

    Returns:
        Tuple of (point_clouds, labels, test_flags, strat_groups, segment_ids, classes)
        strat_groups: Combined stratification groups (site + profile + crown + species)
        segment_ids: Segment IDs (e.g., "A_1000") for each sample
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("="*80)
    logger.info(f"Loading HeliALS 16D data from: {data_path}")
    logger.info(f"CSV file: {csv_path}")
    logger.info("="*80)
    print(f"\n{'='*80}")
    print(f"Loading HeliALS 16D dataset")
    print(f"{'='*80}")

    # Read CSV metadata
    df = pd.read_csv(csv_path)
    logger.info(f"CSV contains {len(df)} rows")
    print(f"\nCSV contains {len(df)} rows")

    # Filter to segments that are also present in final-segments-with-species.csv.
    # This excludes segments the paper authors removed for quality reasons.
    if final_csv_path is not None:
        if not os.path.exists(final_csv_path):
            raise FileNotFoundError(f"final_csv_path does not exist: {final_csv_path}")
        final_df = pd.read_csv(final_csv_path)[['test_site_name', 'segment_id']]
        before = len(df)
        df = df.merge(final_df, on=['test_site_name', 'segment_id'], how='inner')
        dropped = before - len(df)
        logger.info(f"Filtered against {final_csv_path}: kept {len(df)}/{before} (dropped {dropped})")
        print(f"Filtered against final-segments CSV: "
              f"kept {len(df)}/{before} rows (dropped {dropped} not in final CSV)")
    
    # Get class names from species codes
    classes = [SPECIES_NAMES[i] for i in range(1, 10)]  # codes 1-9
    logger.info(f"Classes: {classes}")
    print(f"Classes: {classes}")
    
    point_clouds = []
    labels = []
    test_flags = []
    strat_groups = []  # For stratified sampling by multiple factors
    segment_ids = []   # For validation split consistency
    skipped_files = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading .npy files"):
        # Construct filename: {test_site_name}_{segment_id}.npy
        site_name = row['test_site_name']
        segment_id = row['segment_id']
        npy_filename = f"{site_name}_{segment_id}.npy"
        npy_path = os.path.join(data_path, npy_filename)
        
        if not os.path.exists(npy_path):
            logger.warning(f"File not found: {npy_path}")
            skipped_files += 1
            continue
        
        try:
            # Load pre-processed point cloud (already FPS-sampled, 8192x16)
            points = np.load(npy_path)
            point_clouds.append(points)
            
            # Label is species_code - 1 (0-indexed)
            label = int(row['species_code']) - 1
            labels.append(label)
            
            # Store test flag
            test_flags.append(int(row['test_set_flag']))
            
            # Store segment ID for validation split consistency
            full_segment_id = f"{site_name}_{segment_id}"
            segment_ids.append(full_segment_id)
            
            # Create stratification group: site_profile_crown_species
            profile_cat = str(row.get('profile_category', 'unknown'))
            crown_class = str(row.get('crown_class', 'unknown'))
            strat_group = f"{site_name}_{profile_cat}_{crown_class}_{label}"
            strat_groups.append(strat_group)
            
        except Exception as e:
            logger.error(f"Error processing {npy_path}: {str(e)}")
            skipped_files += 1
            continue
    
    if not point_clouds:
        raise ValueError("No valid point cloud data was loaded")
    
    logger.info(f"\nLoaded {len(point_clouds)} point clouds, skipped {skipped_files} files")
    print(f"\n✓ Loaded {len(point_clouds)} point clouds, skipped {skipped_files} files")

    # Convert to numpy arrays
    point_clouds = np.array(point_clouds, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    test_flags = np.array(test_flags, dtype=np.int64)
    strat_groups = np.array(strat_groups)
    segment_ids = np.array(segment_ids)

    # Leakage audit: confirm test segments don't share `test_site_name` with
    # train segments. test_set_flag is a per-segment column, so site-level
    # overlap is possible if the curator assigned the flag randomly within
    # site. We surface the answer once so the paper can document it.
    site_per_sample = np.array(
        [seg.rsplit('_', 1)[0] for seg in segment_ids.tolist()]
    )
    train_sites = set(site_per_sample[test_flags == 0].tolist())
    test_sites = set(site_per_sample[test_flags == 1].tolist())
    overlap_sites = sorted(train_sites & test_sites)
    n_train_seg_in_overlap = int(np.isin(
        site_per_sample[test_flags == 0], list(overlap_sites)
    ).sum()) if overlap_sites else 0
    n_test_seg_in_overlap = int(np.isin(
        site_per_sample[test_flags == 1], list(overlap_sites)
    ).sum()) if overlap_sites else 0
    seg_dups = len(segment_ids) - len(set(segment_ids.tolist()))

    print(f"\n[holdout audit]")
    print(f"  Unique sites — train: {len(train_sites)} | test: {len(test_sites)} "
          f"| both: {len(overlap_sites)}")
    if overlap_sites:
        print(f"  ⚠ Site-level overlap: {len(overlap_sites)} site(s) appear in "
              f"BOTH train and test pools "
              f"({n_train_seg_in_overlap} train seg / {n_test_seg_in_overlap} test seg).")
        print(f"    -> Holdout is SEGMENT-LEVEL within shared sites.")
        print(f"    Examples: {overlap_sites[:5]}")
    else:
        print(f"  ✓ Site-level holdout: train and test sites are disjoint.")
    if seg_dups:
        print(f"  ⚠ {seg_dups} duplicate segment_id(s) detected in loaded data.")
    else:
        print(f"  ✓ Segment IDs are unique across the loaded data.")
    logger.info(f"Holdout audit — train sites: {len(train_sites)}, "
                f"test sites: {len(test_sites)}, shared: {len(overlap_sites)}, "
                f"segment-id duplicates: {seg_dups}")

    logger.info(f"Point clouds shape: {point_clouds.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Test flags: {np.sum(test_flags == 1)} test, {np.sum(test_flags == 0)} train")
    logger.info(f"Unique stratification groups: {len(np.unique(strat_groups))}")
    print(f"  Point clouds: {point_clouds.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Test flags: {np.sum(test_flags == 1)} test, {np.sum(test_flags == 0)} train")
    print(f"  Unique stratification groups: {len(np.unique(strat_groups))}")

    return point_clouds, labels, test_flags, strat_groups, segment_ids, classes


def create_splits_16d(point_clouds, labels, test_flags, classes, output_path,
                       val_ratio=0.0, random_state=42,
                       strat_groups=None, segment_ids=None):
    """Save train/val/test groups to a single H5 file using the CSV's
    test_set_flag column as the partition oracle.

    Convention (set by make_partition_csv.py):
      0 = train, 1 = test, 2 = val.

    Backwards compatibility: if no flag==2 rows are present (binary CSV),
    fall back to a stratified sklearn split of the train pool with
    test_size=val_ratio. Run `python data\\ preparation/make_partition_csv.py`
    first to switch to the three-way partition.

    `val_ratio` is only consulted in the fallback path.
    """
    split_h5_path = os.path.join(output_path, 'data_split_helials_16d.h5')

    has_explicit_val = bool((test_flags == 2).any())

    if has_explicit_val:
        train_mask = test_flags == 0
        val_mask   = test_flags == 2
        test_mask  = test_flags == 1
        train_pcs    = point_clouds[train_mask]
        train_labels = labels[train_mask]
        val_pcs      = point_clouds[val_mask]
        val_labels   = labels[val_mask]
        test_pcs     = point_clouds[test_mask]
        test_labels  = labels[test_mask]
        print(f"\n{'='*80}\nCSV-driven partition (test_set_flag 0/1/2)\n{'='*80}")
        print(f"  Train (flag=0): {len(train_pcs)}")
        print(f"  Val   (flag=2): {len(val_pcs)}")
        print(f"  Test  (flag=1): {len(test_pcs)}")
    else:
        train_pool_mask = test_flags == 0
        test_mask = test_flags == 1
        train_pool_pcs    = point_clouds[train_pool_mask]
        train_pool_labels = labels[train_pool_mask]
        test_pcs    = point_clouds[test_mask]
        test_labels = labels[test_mask]
        print(f"\n{'='*80}\nLegacy binary CSV; carving val from train pool"
              f" via sklearn (val_ratio={val_ratio})\n{'='*80}")
        train_pcs, val_pcs, train_labels, val_labels = train_test_split(
            train_pool_pcs, train_pool_labels,
            test_size=max(val_ratio, 1e-3),
            stratify=train_pool_labels,
            random_state=random_state,
        )
        print(f"  Train: {len(train_pcs)}")
        print(f"  Val:   {len(val_pcs)}")
        print(f"  Test:  {len(test_pcs)}")
    
    # Save to H5 file
    logger.info(f"Saving split data to: {split_h5_path}")
    print(f"\n💾 Saving split data to {split_h5_path}...")
    
    with h5py.File(split_h5_path, 'w') as f:
        # Store classes
        f.create_dataset('classes', data=np.array(classes, dtype='S'))
        
        # Store metadata
        f.attrs['val_ratio'] = val_ratio
        f.attrs['random_state'] = random_state
        f.attrs['input_dim'] = 16
        f.attrs['num_points'] = point_clouds.shape[1]
        
        # Train data
        train_group = f.create_group('train')
        train_group.create_dataset('point_clouds', data=train_pcs)
        train_group.create_dataset('labels', data=train_labels)
        
        # Validation data
        val_group = f.create_group('val')
        val_group.create_dataset('point_clouds', data=val_pcs)
        val_group.create_dataset('labels', data=val_labels)
        
        # Test data
        test_group = f.create_group('test')
        test_group.create_dataset('point_clouds', data=test_pcs)
        test_group.create_dataset('labels', data=test_labels)
    
    # Print class distribution for each split
    print(f"\n📊 Class distribution:")
    for split_name, split_labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
        unique, counts = np.unique(split_labels, return_counts=True)
        print(f"\n  {split_name}:")
        for u, c in zip(unique, counts):
            print(f"    {classes[u]}: {c} ({c/len(split_labels)*100:.1f}%)")
    
    file_size_mb = os.path.getsize(split_h5_path) / (1024 * 1024)
    logger.info(f"Split data saved successfully. File size: {file_size_mb:.2f} MB")
    print(f"\n✓ Split data saved successfully ({file_size_mb:.2f} MB)")
    print("="*80 + "\n")
    
    return split_h5_path

@DATASETS.register_module()
class TreeSpeciesDatasetHELIALS(Dataset):
    """HeliALS Tree Species Dataset (16D version).
    
    Loads pre-processed tree species point cloud data from HeliALS_fps_8192_16D.
    Supports train/val/test partitions based on CSV metadata.
    16D features: XYZ + multi-wavelength features.
    """
    
    def __init__(self, config):
        self.num_points = config.N_POINTS
        self.partition = config.subset
        data_path = config.DATA_PATH
        
        # Get random seed for reproducibility (default: 42)
        self.seed = getattr(config, 'seed', 42)

        # Validation ratio. Default 0.05 matches paper's 95-5 split for
        # FGI-PointTransformer-DL-3D on HeliALS (Table 3).
        self.val_ratio = getattr(config, 'val_ratio', 0.05)

        # Partition CSV (final-segments-with-species.csv with test_set_flag
        # column added by make_partition_csv.py). Single source of truth for
        # both the segment list and the train/val/test partition.
        parent_dir = os.path.dirname(data_path.rstrip('/'))
        csv_path = getattr(config, 'CSV_PATH', None)
        if csv_path is None:
            for cand in (
                os.path.join(parent_dir, "final-segments-with-species.csv"),
                os.path.join(parent_dir, "training-and-test-segments-with-species.csv"),
            ):
                if os.path.exists(cand):
                    csv_path = cand
                    break
            if csv_path is None:
                csv_path = os.path.join(parent_dir,
                                         "final-segments-with-species.csv")

        # The partition CSV already represents the quality-filtered list of
        # segments, so no additional inner-join filter is needed by default.
        # Set FINAL_CSV_PATH explicitly on the config to re-enable.
        final_csv_path = getattr(config, 'FINAL_CSV_PATH', None)
        self.final_csv_path = final_csv_path

        try:
            split_h5_path = os.path.join(data_path, 'data_split_helials_16d.h5')

            def _newest_npy_mtime(dir_path):
                """Most recent mtime of .npy files feeding the H5 cache."""
                mt = 0.0
                for entry in os.scandir(dir_path):
                    if entry.is_file() and entry.name.endswith('.npy'):
                        mt = max(mt, entry.stat().st_mtime)
                return mt

            if not os.path.exists(split_h5_path):
                logger.info(f"Split file not found. Creating data split...")
                print(f"\n⚠️  Split file not found at {split_h5_path}")
                print("Creating data split from .npy files...")

                # Load from .npy files (now returns strat_groups and segment_ids)
                point_clouds, labels, test_flags, strat_groups, segment_ids, classes = load_helials_npy_data(
                    data_path, csv_path, final_csv_path=final_csv_path
                )

                # Create splits with multi-factor stratification and segment_ids for validation consistency
                create_splits_16d(point_clouds, labels, test_flags, classes, data_path,
                                 val_ratio=self.val_ratio, random_state=self.seed,
                                 strat_groups=strat_groups, segment_ids=segment_ids)
            else:
                with h5py.File(split_h5_path, 'r') as f:
                    existing_seed = f.attrs.get('random_state', None)
                    existing_val_ratio = f.attrs.get('val_ratio', None)
                h5_mtime  = os.path.getmtime(split_h5_path)
                npy_mtime = _newest_npy_mtime(data_path)
                csv_mtime = (os.path.getmtime(csv_path)
                              if os.path.exists(csv_path) else 0.0)
                params_match = (existing_seed == self.seed
                                and existing_val_ratio == self.val_ratio)
                cache_stale  = (npy_mtime > h5_mtime or csv_mtime > h5_mtime)

                if not params_match or cache_stale:
                    reason = ("parameters differ" if not params_match
                              else ("CSV is newer than the H5 cache"
                                    if csv_mtime > h5_mtime
                                    else "underlying .npy files are newer than the H5 cache"))
                    logger.info(f"Existing split is invalid ({reason}); recreating...")
                    print(f"\n⚠️  Split cache invalid ({reason}); recreating...")
                    point_clouds, labels, test_flags, strat_groups, segment_ids, classes = load_helials_npy_data(
                        data_path, csv_path, final_csv_path=final_csv_path
                    )
                    create_splits_16d(point_clouds, labels, test_flags, classes, data_path,
                                     val_ratio=self.val_ratio, random_state=self.seed,
                                     strat_groups=strat_groups, segment_ids=segment_ids)
                else:
                    logger.info(f"Loading existing split from: {split_h5_path}")
                    print(f"✓ Using existing split file: {split_h5_path}")
            
            # Load the split data
            with h5py.File(split_h5_path, 'r') as f:
                self.classes = [c.decode() if isinstance(c, bytes) else c for c in f['classes'][:]]
                self.input_dim = f.attrs.get('input_dim', 16)
                
                if self.partition not in f:
                    available = [k for k in f.keys() if k != 'classes']
                    raise ValueError(f"Partition '{self.partition}' not found. Available: {available}")
                
                self.data = f[self.partition]['point_clouds'][:]
                self.label = f[self.partition]['labels'][:]
                print(f"✓ Loaded {len(self.data)} samples from '{self.partition}' partition")
                print(f"  Shape: {self.data.shape} ({self.input_dim}D features, {len(self.classes)} classes)")
            
            # Ensure labels are properly shaped
            if self.label.ndim > 1:
                self.label = self.label.flatten()
            
            logger.info(f"Dataset loaded: {len(self.data)} samples in '{self.partition}' partition")
            logger.info(f"Number of classes: {len(self.classes)}, Input dim: {self.input_dim}")
            
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise
    
    def __getitem__(self, item):
        try:
            pointcloud = self.data[item][:self.num_points].copy()
            label = self.label[item]
            
            # Extract label value robustly
            if isinstance(label, (np.ndarray, list, tuple)):
                label_value = int(label[0] if len(label) > 0 else label)
            else:
                label_value = int(label)
            
            # XYZ already unit-sphere normalized and attributes already scaled during preprocessing
            return 'HeliALS', 'sample', (pointcloud.astype(np.float32), label_value)
        except Exception as e:
            logger.error(f"Error getting item {item}: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.data)

def normalize_pc(points):
    """Normalize point cloud to unit sphere (3D only)."""
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    points /= furthest_distance
    return points


def normalize_pc_16d(points):
    """Normalize 16D point cloud - only normalize XYZ (first 3 dims) to unit sphere.
    
    Other features (dims 3:15) are already normalized during preprocessing.
    """
    # Extract XYZ and features
    xyz = points[:, :3].copy()
    features = points[:, 3:].copy()
    
    # Normalize XYZ to unit sphere
    centroid = np.mean(xyz, axis=0)
    xyz -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(xyz**2, axis=-1)))
    if furthest_distance > 0:
        xyz /= furthest_distance
    
    # Concatenate back
    return np.concatenate([xyz, features], axis=-1)

def analyze_class_distribution(dataset, title="Class Distribution"):
    """Analyze and display class distribution in a dataset."""
    class_counts = {}
    total_samples = len(dataset)
    
    for i in range(len(dataset)):
        label = int(dataset.label[i])
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n{title}:")
    print("-" * 50)
    print(f"{'Class':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{class_name:<20} {count:<10} {percentage:>6.2f}%")
    
    print("-" * 50)
    print(f"Total samples: {total_samples}\n")
    
    return class_counts

if __name__ == '__main__':
    try:
        # Paper-aligned config for FGI-PointTransformer-DL-3D on HeliALS:
        #   N_POINTS      = 8192                (paper Table 3)
        #   val_ratio     = 0.05                (paper: "95-5 train-validation split")
        #   DATA_PATH     = voxel-aggregated preprocessed folder
        #   CSV_PATH      = training-and-test-segments-with-species.csv (has test_set_flag)
        #   FINAL_CSV_PATH= final-segments-with-species.csv
        #                   (quality filter: drop the 88 excluded segments)
        data_path = 'data/HeliALS_voxelagg_8192_16D'
        final_csv = 'data preparation/final-segments-with-species.csv'

        def cfg(subset):
            return SimpleNamespace(
                N_POINTS=8192,
                subset=subset,
                DATA_PATH=data_path,
                val_ratio=0.05,
                seed=42,
                FINAL_CSV_PATH=final_csv,
            )

        train_cfg, val_cfg, test_cfg = cfg('train'), cfg('val'), cfg('test')
        
        train_dataset = TreeSpeciesDatasetHELIALS(train_cfg)
        val_dataset = TreeSpeciesDatasetHELIALS(val_cfg)
        test_dataset = TreeSpeciesDatasetHELIALS(test_cfg)
        
        print(f"\nClasses: {train_dataset.classes}")
        print(f"Input dimension: {train_dataset.input_dim}")
        
        # Analyze class distributions
        analyze_class_distribution(train_dataset, "Training Set Distribution")
        analyze_class_distribution(val_dataset, "Validation Set Distribution")
        analyze_class_distribution(test_dataset, "Test Set Distribution")
        
        # Verify data loading
        _, _, (sample_data, sample_label) = train_dataset[0]
        print(f"\nSample point cloud shape: {sample_data.shape}")
        print(f"Sample label: {sample_label} ({train_dataset.classes[sample_label]})")
        print(f"XYZ range: [{sample_data[:, :3].min():.3f}, {sample_data[:, :3].max():.3f}]")
        print(f"Features range: [{sample_data[:, 3:].min():.3f}, {sample_data[:, 3:].max():.3f}]")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

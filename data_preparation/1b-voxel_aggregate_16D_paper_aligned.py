"""HeliALS 16D preprocessing: voxel-aggregated multi-wavelength point clouds.

Pipeline:
  1. Read LAS, drop ground (class=2) and noise (class=7).
  2. Replace sentinel '1' -> NaN in Reflectance / Amplitude / Deviation.
  3. Voxel-aggregate at 5 cm: per-scanner nan-aware means per attribute.
  4. 1-NN spatial fill within 20 cm for per-scanner coverage gaps.
  5. Global-median fallback for any residual NaN.
  6. Downsample to target points (FPS by default, random optional).
  7. XYZ -> unit sphere; other attributes min-max scaled; return_number raw.

Output: (N_points, 16) per segment
  [0:3]    x, y, z (unit sphere, per-segment)
  [3:6]    intensity   1550 / 905 / 532 nm
  [6:9]    amplitude   1550 / 905 / 532 nm
  [9:12]   reflectance 1550 / 905 / 532 nm
  [12:15]  deviation   1550 / 905 / 532 nm
  [15]     return_number (raw)

Global min-max stats are computed over `test_set_flag == 0` (train) only
and applied to every segment; val (flag==2) and test (flag==1) are
scaled with the train bounds and clipped to [0, 1]. Run
`data preparation/make_partition_csv.py` once beforehand to set up the
3-valued flag. `--per-segment-scaling` disables global stats entirely.
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import laspy
import torch


# ----- FPS downsampling --------------------------------------------------

try:
    import pointnet2_ops._ext as _ext
    HAS_GPU_FPS = True
except ImportError:
    HAS_GPU_FPS = False


def farthest_point_sample_gpu(xyz, npoint):
    N = xyz.shape[0]
    if N <= npoint:
        if N < npoint:
            pad = np.random.choice(N, npoint - N, replace=True)
            indices = np.concatenate([np.arange(N), pad])
            np.random.shuffle(indices)
        else:
            indices = np.arange(N)
        return indices
    xyz_tensor = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(0).cuda()
    fps_indices = _ext.furthest_point_sampling(xyz_tensor, npoint)
    return fps_indices.squeeze(0).cpu().numpy().astype(np.int64)


def farthest_point_sample_cpu(xyz, npoint):
    N = xyz.shape[0]
    if N <= npoint:
        if N < npoint:
            pad = np.random.choice(N, npoint - N, replace=True)
            indices = np.concatenate([np.arange(N), pad])
            np.random.shuffle(indices)
        else:
            indices = np.arange(N)
        return indices
    indices = np.zeros(npoint, dtype=np.int64)
    indices[0] = np.random.randint(0, N)
    dists = np.full(N, np.inf, dtype=np.float64)
    for i in range(1, npoint):
        last_point = xyz[indices[i - 1]]
        new_dists = np.sum((xyz - last_point) ** 2, axis=1)
        dists = np.minimum(dists, new_dists)
        indices[i] = np.argmax(dists)
    return indices


def random_subsample(xyz, npoint):
    N = xyz.shape[0]
    if N <= npoint:
        if N < npoint:
            pad = np.random.choice(N, npoint - N, replace=True)
            indices = np.concatenate([np.arange(N), pad])
            np.random.shuffle(indices)
        else:
            indices = np.arange(N)
        return indices
    return np.random.choice(N, npoint, replace=False)


# ----- Sentinel handling -------------------------------------------------

SENTINEL_VALUE = 1.0
SENTINEL_ATTRS = ('Reflectance', 'Amplitude', 'Deviation')


def replace_sentinel_with_nan(attrs):
    n = 0
    for name in SENTINEL_ATTRS:
        if name in attrs:
            col = attrs[name]
            mask = (col == SENTINEL_VALUE)
            hit = int(mask.sum())
            if hit:
                col = col.copy()
                col[mask] = np.nan
                attrs[name] = col
                n += hit
    return attrs, n


# ----- Noise / ground removal --------------------------------------------

def remove_noise_and_ground(xyz, attrs, classification):
    """Paper: 'Removal of noise and ground points' (class 2 and class 7)."""
    valid = ~np.isin(classification, [2, 7])
    xyz = xyz[valid]
    attrs = {k: v[valid] for k, v in attrs.items()}
    return xyz, attrs


# ----- Voxel-level aggregation -------------------------------------------

SCANNER_IDS = (1, 2, 3)  # VUX-1HA (1550 nm), miniVUX-1DL (905 nm), VQ-840-G (532 nm)


def voxel_aggregate(xyz, attrs, voxel_size=0.05):
    """
    Aggregate points into voxels. One output point per voxel.

      - xyz_out      = mean of member xyz
      - per scanner s in {1,2,3}:
          intensity_s, amplitude_s, reflectance_s, deviation_s
          = nan-aware mean of scanner-s member values within the voxel
          (NaN if the voxel contains no scanner-s points, or all values
           were sentinel -> NaN)
      - return_number = mean over all members (raw, not per-scanner)

    Returns
    -------
    xyz_out         : (M, 3) float32
    intensity_3ch   : (M, 3) float32, possibly with NaN
    amplitude_3ch   : (M, 3) float32, possibly with NaN
    reflectance_3ch : (M, 3) float32, possibly with NaN
    deviation_3ch   : (M, 3) float32, possibly with NaN
    return_number_1 : (M,)    float32
    """
    N = len(xyz)
    if N == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return (empty,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    xyz_min = xyz.min(axis=0)
    voxel_idx = np.floor((xyz - xyz_min) / voxel_size).astype(np.int64)
    dims = voxel_idx.max(axis=0) + 1
    keys = (voxel_idx[:, 0] * (dims[1] * dims[2])
            + voxel_idx[:, 1] * dims[2]
            + voxel_idx[:, 2])

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    M = len(unique_keys)
    counts = np.bincount(inverse, minlength=M).astype(np.float64)

    xyz_out = np.empty((M, 3), dtype=np.float64)
    for c in range(3):
        xyz_out[:, c] = np.bincount(inverse, weights=xyz[:, c], minlength=M) / counts

    ret = attrs['return_number'].astype(np.float64)
    return_number_out = np.bincount(inverse, weights=ret, minlength=M) / counts

    intensity = attrs['intensity'].astype(np.float64)
    amplitude = attrs['Amplitude'].astype(np.float64)
    reflectance = attrs['Reflectance'].astype(np.float64)
    deviation = attrs['Deviation'].astype(np.float64)
    user_data = attrs['user_data'].astype(np.int32)

    intensity_3ch = np.full((M, 3), np.nan, dtype=np.float32)
    amplitude_3ch = np.full((M, 3), np.nan, dtype=np.float32)
    reflectance_3ch = np.full((M, 3), np.nan, dtype=np.float32)
    deviation_3ch = np.full((M, 3), np.nan, dtype=np.float32)

    attr_pairs = (
        (intensity, intensity_3ch),
        (amplitude, amplitude_3ch),
        (reflectance, reflectance_3ch),
        (deviation, deviation_3ch),
    )

    for ch, sid in enumerate(SCANNER_IDS):
        scanner_mask = (user_data == sid)
        if not scanner_mask.any():
            continue
        inv_s = inverse[scanner_mask]

        for attr_arr, out_arr in attr_pairs:
            vals = attr_arr[scanner_mask]
            valid = np.isfinite(vals)
            if not valid.any():
                continue
            inv_v = inv_s[valid]
            vals_v = vals[valid]
            num = np.bincount(inv_v, weights=vals_v, minlength=M)
            den = np.bincount(inv_v, minlength=M).astype(np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.where(den > 0, num / den, np.nan)
            out_arr[:, ch] = mean.astype(np.float32)

    # xyz_out stays float64; caller must shift to a local frame before any float32 cast.
    return (xyz_out,
            intensity_3ch,
            amplitude_3ch,
            reflectance_3ch,
            deviation_3ch,
            return_number_out.astype(np.float32))


# ----- Cross-channel 1-NN spatial fill -----------------------------------

def fill_missing_channels_by_nn(xyz, channel_arrays, search_radius=0.20):
    """
    Cross-scanner spatial gap-fill for voxel-aggregated attributes.

    For each voxel that has NaN in attribute column c, find the nearest voxel
    that has a finite value for that column and copy it verbatim (1-NN), provided
    the nearest neighbor is within `search_radius`. This is a point-density
    harmonisation step: scanner-s missed this 5 cm voxel, but a neighbouring
    voxel within 20 cm did record scanner-s, so we propagate its captor-derived
    value. The output is therefore still a captor measurement, just one
    displaced by up to `search_radius`.

    This is NOT the wavelength-scale imputation addressed by WaveFill-Net.
    Here we resolve per-voxel coverage gaps caused by different scanner point
    densities; WaveFill-Net handles the case where an entire wavelength is
    absent across the whole tree.

    Per-attribute independence: intensity_c, amplitude_c, reflectance_c,
    deviation_c are filled separately because sentinel -> NaN can remove
    some but not all of them even when scanner-c was present.

    Parameters
    ----------
    xyz : (M, 3) float
    channel_arrays : list of (M, 3) float32 arrays in canonical attribute
        order: [intensity, amplitude, reflectance, deviation].

    Returns
    -------
    filled : list of (M, 3) arrays with NaNs replaced where possible.
    counts : list of (3,)-int arrays, one per input array, giving the number
        of voxels that were 1-NN-filled for each wavelength channel.
        Useful for the coverage statistics reported in metadata.json.
    """
    filled = []
    fill_counts = []  # per-array, per-channel count of 1-NN fills applied
    for arr in channel_arrays:
        out = arr.copy()
        per_ch = np.zeros(arr.shape[1], dtype=np.int64)
        for ch in range(arr.shape[1]):
            col = out[:, ch]
            has = np.isfinite(col)
            miss = ~has
            if not miss.any() or not has.any():
                continue
            tree = cKDTree(xyz[has])
            src_vals = col[has]
            dists, idx = tree.query(xyz[miss], k=1)
            within = dists <= search_radius
            miss_global = np.where(miss)[0]
            out[miss_global[within], ch] = src_vals[idx[within]]
            per_ch[ch] = int(within.sum())
        filled.append(out)
        fill_counts.append(per_ch)
    return filled, fill_counts


def impute_residual_nan_with_median(features_13, attr_names_13):
    """Last-resort fallback: global-median replacement for any value that
    survived the voxel-aggregate and 1-NN steps as NaN/inf.

    Unlike the 1-NN step, this branch produces a value that is NOT a captor
    measurement: it is the per-segment median of finite entries in the same
    column. It fires only when no scanner-s neighbour was within
    `nn_search_radius` of a voxel, which is rare on dense IR channels and
    occasional on the sparser 532 nm channel.

    return_number is kept raw (no replacement). If an entire column is NaN
    (segment has no data for that scanner AND no spatial neighbour within
    radius), fall back to 0 -- otherwise NaNs would survive through scaling
    and poison training.

    Returns
    -------
    out : (N, 13) float32 with all NaN/inf replaced.
    median_counts : (13,) int64 -- per-column number of points that received
        the median fallback. Useful for coverage statistics.
    """
    out = features_13.copy()
    median_counts = np.zeros(features_13.shape[1], dtype=np.int64)
    for i, name in enumerate(attr_names_13):
        if name == 'return_number':
            continue
        col = out[:, i]
        bad = ~np.isfinite(col)
        if not bad.any():
            continue
        median_counts[i] = int(bad.sum())
        if (~bad).any():
            out[bad, i] = np.median(col[~bad])
        else:
            out[:, i] = 0.0
    return out, median_counts


# ----- Normalization -----------------------------------------------------

def normalize_to_unit_sphere(xyz):
    """Center on centroid then scale into unit sphere; computed in float64."""
    xyz64 = np.asarray(xyz, dtype=np.float64)
    centroid = xyz64.mean(axis=0)
    centered = xyz64 - centroid
    max_dist = float(np.max(np.linalg.norm(centered, axis=1)))
    if max_dist > 0:
        return (centered / max_dist).astype(np.float32), centroid, max_dist
    return centered.astype(np.float32), centroid, max_dist


def scale_attributes(attributes, attr_names, global_stats=None):
    """Min-max scale attributes; return_number passes through unchanged."""
    scaled = attributes.copy().astype(np.float32)
    params = {}
    for i, name in enumerate(attr_names):
        if name == 'return_number':
            params[name] = {'min': 0, 'max': 1, 'scaled': False}
            continue
        col = scaled[:, i]
        if global_stats is not None and name in global_stats:
            cmin = global_stats[name]['min']
            cmax = global_stats[name]['max']
            if cmax > cmin:
                scaled[:, i] = np.clip((col - cmin) / (cmax - cmin), 0.0, 1.0)
            else:
                scaled[:, i] = 0.0
            params[name] = {'min': cmin, 'max': cmax, 'scaled': True, 'scope': 'global'}
        else:
            finite = col[np.isfinite(col)]
            if len(finite) and finite.max() > finite.min():
                cmin, cmax = float(finite.min()), float(finite.max())
                scaled[:, i] = (col - cmin) / (cmax - cmin)
            else:
                cmin, cmax = 0.0, 1.0
                scaled[:, i] = 0.0
            params[name] = {'min': cmin, 'max': cmax, 'scaled': True, 'scope': 'per_segment'}
        bad = ~np.isfinite(scaled[:, i])
        if bad.any():
            scaled[bad, i] = 0.0
    return scaled, params


# ----- Attribute names / dimensions --------------------------------------

ATTR_NAMES_13 = [
    'intensity_1550', 'intensity_905', 'intensity_532',
    'amplitude_1550', 'amplitude_905', 'amplitude_532',
    'reflectance_1550', 'reflectance_905', 'reflectance_532',
    'deviation_1550', 'deviation_905', 'deviation_532',
    'return_number',
]

DIM_NAMES_16 = [
    'x', 'y', 'z',
    'intensity_1550', 'intensity_905', 'intensity_532',
    'amplitude_1550', 'amplitude_905', 'amplitude_532',
    'reflectance_1550', 'reflectance_905', 'reflectance_532',
    'deviation_1550', 'deviation_905', 'deviation_532',
    'return_number',
]


# ----- Per-file pipeline -------------------------------------------------

def _read_las(las_path):
    las = laspy.read(las_path)
    if len(las.points) < 1:
        return None
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    classification = np.array(las.classification, dtype=np.int32)
    attrs = {
        'user_data':     np.array(las.user_data, dtype=np.float32),
        'intensity':     np.array(las.intensity, dtype=np.float32),
        'Amplitude':     np.array(getattr(las, 'Amplitude'), dtype=np.float32),
        'Reflectance':   np.array(getattr(las, 'Reflectance'), dtype=np.float32),
        'Deviation':     np.array(getattr(las, 'Deviation'), dtype=np.float32),
        'return_number': np.array(las.return_number, dtype=np.float32),
    }
    return xyz, classification, attrs


def preprocess_single_file(las_path, voxel_size=0.05, nn_search_radius=0.20):
    """Per-file pipeline. Returns (result_dict, error_or_None)."""
    try:
        parsed = _read_las(las_path)
    except Exception as e:
        return None, f"Read error: {e}"
    if parsed is None:
        return None, "Empty point cloud"
    xyz, classification, attrs = parsed
    n_original = len(xyz)

    xyz, attrs = remove_noise_and_ground(xyz, attrs, classification)
    n_after_clean = len(xyz)
    if n_after_clean < 1:
        return None, "No points after cleaning"

    attrs, n_sentinels = replace_sentinel_with_nan(attrs)

    (xyz_v, intensity_3ch, amplitude_3ch, reflectance_3ch, deviation_3ch,
     return_number_1) = voxel_aggregate(xyz, attrs, voxel_size=voxel_size)
    n_after_voxel = len(xyz_v)
    if n_after_voxel < 1:
        return None, "No voxels"

    n_native_pre = {
        'intensity':   np.isfinite(intensity_3ch).sum(axis=0).astype(int).tolist(),
        'amplitude':   np.isfinite(amplitude_3ch).sum(axis=0).astype(int).tolist(),
        'reflectance': np.isfinite(reflectance_3ch).sum(axis=0).astype(int).tolist(),
        'deviation':   np.isfinite(deviation_3ch).sum(axis=0).astype(int).tolist(),
    }

    (intensity_3ch, amplitude_3ch, reflectance_3ch, deviation_3ch), nn_fill_counts = \
        fill_missing_channels_by_nn(
            xyz_v,
            [intensity_3ch, amplitude_3ch, reflectance_3ch, deviation_3ch],
            search_radius=nn_search_radius,
        )
    n_nn_filled = {
        'intensity':   nn_fill_counts[0].tolist(),
        'amplitude':   nn_fill_counts[1].tolist(),
        'reflectance': nn_fill_counts[2].tolist(),
        'deviation':   nn_fill_counts[3].tolist(),
    }

    features_13 = np.hstack([
        intensity_3ch,
        amplitude_3ch,
        reflectance_3ch,
        deviation_3ch,
        return_number_1.reshape(-1, 1),
    ]).astype(np.float32)

    features_13, median_counts = impute_residual_nan_with_median(
        features_13, ATTR_NAMES_13
    )
    n_median_filled = {
        ATTR_NAMES_13[i]: int(median_counts[i])
        for i in range(len(ATTR_NAMES_13))
        if ATTR_NAMES_13[i] != 'return_number'
    }

    # shift to a local frame so the float32 cast does not quantize UTM coordinates
    raw_centroid_utm = xyz_v.mean(axis=0)
    xyz_v_local = xyz_v - raw_centroid_utm

    return {
        'xyz': xyz_v_local.astype(np.float32),
        'features_13': features_13,
        'raw_centroid_utm': raw_centroid_utm,
        'n_original': n_original,
        'n_after_clean': n_after_clean,
        'n_after_voxel': n_after_voxel,
        'n_sentinels_replaced': int(n_sentinels),
        'coverage': {
            'n_voxels': int(n_after_voxel),
            'n_native_per_channel':       n_native_pre,
            'n_nn_filled_per_channel':    n_nn_filled,
            'n_median_filled_per_column': n_median_filled,
        },
    }, None


def finalize_single(xyz, features_13, ds_indices, global_stats=None):
    """Post-downsampling: select, unit-sphere XYZ, scale attributes -> (K, 16)."""
    xyz_sel = xyz[ds_indices]
    feat_sel = features_13[ds_indices]
    xyz_norm, centroid, scale = normalize_to_unit_sphere(xyz_sel)
    feat_scaled, scale_params = scale_attributes(
        feat_sel, ATTR_NAMES_13, global_stats=global_stats
    )
    data_16d = np.hstack([xyz_norm, feat_scaled]).astype(np.float32)
    return data_16d, centroid, scale, scale_params


# ----- Global stats (for min-max scaling) --------------------------------

def compute_global_stats_single(las_path, voxel_size, nn_search_radius):
    result, _ = preprocess_single_file(
        las_path, voxel_size=voxel_size, nn_search_radius=nn_search_radius
    )
    if result is None:
        return None
    features_13 = result['features_13']
    stats = {}
    for i, name in enumerate(ATTR_NAMES_13):
        if name == 'return_number':
            continue
        col = features_13[:, i]
        finite = col[np.isfinite(col)]
        if len(finite):
            stats[name] = {'min': float(finite.min()), 'max': float(finite.max())}
    return stats


def _aggregate_coverage_summary(metadata_list):
    """Roll up per-segment coverage stats into dataset-wide totals."""
    attr_names = ('intensity', 'amplitude', 'reflectance', 'deviation')
    wl_names   = ('1550nm', '905nm', '532nm')

    n_voxels_total = 0
    native    = {a: [0, 0, 0] for a in attr_names}
    nn_filled = {a: [0, 0, 0] for a in attr_names}
    median_filled = {f"{a}_{wl_names[ch].replace('nm','')}": 0
                     for a in attr_names for ch in range(3)}

    for entry in metadata_list:
        cov = entry.get('coverage')
        if not cov:
            continue
        n_voxels_total += int(cov.get('n_voxels', 0))
        for a in attr_names:
            for ch in range(3):
                native[a][ch]    += int(cov['n_native_per_channel'][a][ch])
                nn_filled[a][ch] += int(cov['n_nn_filled_per_channel'][a][ch])
        for col_name, n in cov.get('n_median_filled_per_column', {}).items():
            if col_name in median_filled:
                median_filled[col_name] += int(n)

    def _fractions(counts_dict):
        out = {}
        denom = n_voxels_total if n_voxels_total > 0 else 1
        for a in attr_names:
            out[a] = {
                wl_names[ch]: {
                    'count':    int(counts_dict[a][ch]),
                    'fraction': counts_dict[a][ch] / denom,
                }
                for ch in range(3)
            }
        return out

    median_summary = {
        col: {
            'count':    int(median_filled[col]),
            'fraction': median_filled[col] / (n_voxels_total or 1),
        }
        for col in median_filled
    }

    return {
        'n_voxels_total':       int(n_voxels_total),
        'native_captor':        _fractions(native),
        'nn_filled_within_20cm': _fractions(nn_filled),
        'global_median_fallback': median_summary,
        'notes': (
            "native_captor + nn_filled_within_20cm = fraction of voxels "
            "whose value is a captor measurement (native or spatially "
            "propagated by 1-NN within nn_search_radius). "
            "global_median_fallback is the only path that produces a "
            "non-measurement value; for clean datasets it should be near 0."
        ),
    }


def aggregate_global_stats(per_file_stats_list):
    out = {}
    for stats in per_file_stats_list:
        if stats is None:
            continue
        for name, vals in stats.items():
            if name not in out:
                out[name] = {'min': vals['min'], 'max': vals['max']}
            else:
                out[name]['min'] = min(out[name]['min'], vals['min'])
                out[name]['max'] = max(out[name]['max'], vals['max'])
    return out


# ----- Train partition (Pass 1 uses test_set_flag == 0) ------------------
# Under the 3-valued CSV (see data preparation/make_partition_csv.py),
# test_set_flag is 0 = train, 1 = test, 2 = val. Filtering to flag == 0
# naturally excludes both val and test from Pass 1.

def _load_train_stems(partition_csv: str,
                       final_csv: str | None,
                       ) -> set[str]:
    if not os.path.exists(partition_csv):
        raise FileNotFoundError(
            f"partition CSV not found: {partition_csv}. Pass --partition-csv "
            "or disable global scaling with --per-segment-scaling."
        )
    df = pd.read_csv(partition_csv)
    required = {"test_site_name", "segment_id", "test_set_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"partition CSV {partition_csv} is missing columns: "
            f"{sorted(missing)}. Expected at least: {sorted(required)}."
        )
    if final_csv is not None:
        if not os.path.exists(final_csv):
            raise FileNotFoundError(f"final CSV not found: {final_csv}")
        final_df = pd.read_csv(final_csv)[["test_site_name", "segment_id"]]
        df = df.merge(final_df, on=["test_site_name", "segment_id"],
                       how="inner")
    train_rows = df[df["test_set_flag"] == 0]
    return {f"{r.test_site_name}_{r.segment_id}" for r in train_rows.itertuples()}


# ----- Batch downsampling ------------------------------------------------

def batch_gpu_fps(xyz_list, npoint):
    return [farthest_point_sample_gpu(xyz, npoint) for xyz in xyz_list]


def batch_cpu_fps(xyz_list, npoint):
    return [farthest_point_sample_cpu(xyz, npoint) for xyz in xyz_list]


def batch_random(xyz_list, npoint):
    return [random_subsample(xyz, npoint) for xyz in xyz_list]


# ----- Main pipeline -----------------------------------------------------

def process_all(input_folder, output_folder, target_count=8192, file_ext='las',
                voxel_size=0.05, nn_search_radius=0.20,
                num_workers=4, gpu_batch_size=64,
                use_gpu=True, downsample_method='fps',
                use_global_scaling=True,
                partition_csv=None, final_csv=None):
    os.makedirs(output_folder, exist_ok=True)
    actual_use_gpu = (use_gpu and HAS_GPU_FPS and torch.cuda.is_available()
                      and downsample_method == 'fps')

    files = sorted(glob.glob(os.path.join(input_folder, f"*.{file_ext}")))
    if not files:
        print(f"No .{file_ext} files found in {input_folder}")
        return

    train_stems = None
    train_files = None
    global_stats_source = 'per_segment'
    if use_global_scaling:
        if partition_csv is None:
            raise SystemExit(
                "Global min-max scaling is leak-prone unless Pass 1 sees only "
                "train segments. Pass --partition-csv <file.csv> (the same "
                "CSV the dataset class uses), or run with --per-segment-scaling."
            )
        train_stems = _load_train_stems(partition_csv, final_csv)
        train_files = [f for f in files if Path(f).stem in train_stems]
        if not train_files:
            raise SystemExit(
                f"Partition CSV lists {len(train_stems)} train stems but none "
                f"are present as .{file_ext} files in {input_folder}. "
                "Refusing to compute global stats."
            )
        n_other_visible = len(files) - len(train_files)
        global_stats_source = (
            f"train_only ({len(train_files)} files, test_set_flag==0)"
        )
        print("=" * 80)
        print(f"  Partition CSV    : {partition_csv}")
        if final_csv:
            print(f"  Final-CSV filter : {final_csv}")
        print(f"  Train files      : {len(train_files)} (test_set_flag==0; "
              "used for Pass 1 stats)")
        print(f"  Other files      : {n_other_visible} "
              "(test + val; scaled with train stats)")

    print("=" * 80)
    print("HeliALS 16D Preprocessing — Voxel-Aggregated (paper + author-email)")
    print("=" * 80)
    print(f"Input:              {input_folder} ({len(files)} files)")
    print(f"Output:             {output_folder}")
    print(f"Target points:      {target_count}")
    print(f"Voxel size:         {voxel_size*100:.0f} cm  (paper: 5 cm)")
    print(f"Voxelization:       per-voxel per-scanner mean aggregation (email)")
    print(f"NN search radius:   {nn_search_radius} m  (used to fill missing channels)")
    print(f"Sentinel '1':       -> NaN before aggregation (paper Section 3.1)")
    print(f"Downsample:         {downsample_method}"
          f"{' (GPU)' if actual_use_gpu else ' (CPU)'}")
    print(f"Attribute scaling:  {'global min-max' if use_global_scaling else 'per-segment min-max'}")
    if use_global_scaling:
        print(f"Stats source:       {global_stats_source}  "
              "(val/test excluded from Pass 1 to prevent leakage)")
    print("=" * 80)

    t_start = time.time()

    global_stats = None
    if use_global_scaling:
        print("\n--- Pass 1: global attribute statistics  (test_set_flag==0) ---")
        t_p1 = time.time()
        per_file_stats = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(compute_global_stats_single, f, voxel_size, nn_search_radius): f
                for f in train_files
            }
            for fut in as_completed(futures):
                try:
                    per_file_stats.append(fut.result())
                except Exception:
                    per_file_stats.append(None)
        global_stats = aggregate_global_stats(per_file_stats)
        print(f"  Global stats in {time.time()-t_p1:.1f}s "
              f"over {len(train_files)} train files")
        for name, vals in sorted(global_stats.items()):
            print(f"    {name}: [{vals['min']:.4f}, {vals['max']:.4f}]")

    print("\n--- Pass 2: preprocessing + downsampling ---")
    successful = 0
    skipped = []
    metadata_list = []
    total_sentinels = 0

    batch_data, batch_names, batch_paths = [], [], []

    def flush_batch():
        nonlocal successful
        if not batch_data:
            return
        xyz_list = [d['xyz'] for d in batch_data]
        if downsample_method == 'random':
            indices_list = batch_random(xyz_list, target_count)
        elif actual_use_gpu:
            indices_list = batch_gpu_fps(xyz_list, target_count)
        else:
            indices_list = batch_cpu_fps(xyz_list, target_count)

        for data, name, _fpath, idx in zip(batch_data, batch_names, batch_paths, indices_list):
            data_16d, centroid, scale, scale_params = finalize_single(
                data['xyz'], data['features_13'], idx,
                global_stats=global_stats,
            )
            np.save(os.path.join(output_folder, f"{name}.npy"), data_16d)
            metadata_list.append({
                'file': f"{name}.{file_ext}",
                'original_points': data['n_original'],
                'after_cleaning': data['n_after_clean'],
                'after_voxel_aggregation': data['n_after_voxel'],
                'sentinels_replaced': data['n_sentinels_replaced'],
                'raw_centroid_utm': data['raw_centroid_utm'].tolist(),
                'local_centroid': centroid.tolist(),
                'scale': float(scale),
                'scale_params': scale_params,
                'coverage': data['coverage'],
            })
            successful += 1
        batch_data.clear(); batch_names.clear(); batch_paths.clear()

    chunk_size = gpu_batch_size * 2
    for chunk_start in range(0, len(files), chunk_size):
        chunk_files = files[chunk_start:chunk_start + chunk_size]
        chunk_results = {}
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(preprocess_single_file, f, voxel_size, nn_search_radius): f
                for f in chunk_files
            }
            for fut in as_completed(futures):
                fpath = futures[fut]
                stem = Path(fpath).stem
                try:
                    result, error = fut.result()
                except Exception as e:
                    result, error = None, str(e)
                if result is None:
                    print(f"  SKIP {stem}: {error}")
                    skipped.append(stem)
                else:
                    chunk_results[fpath] = result

        for fpath in chunk_files:
            if fpath not in chunk_results:
                continue
            stem = Path(fpath).stem
            data = chunk_results[fpath]
            total_sentinels += data['n_sentinels_replaced']
            batch_data.append(data)
            batch_names.append(stem)
            batch_paths.append(fpath)
            idx = chunk_start + chunk_files.index(fpath) + 1
            sent = (f" [{data['n_sentinels_replaced']} sentinels]"
                    if data['n_sentinels_replaced'] else "")
            print(f"[{idx}/{len(files)}] {stem}: "
                  f"{data['n_original']} -> {data['n_after_clean']} -> "
                  f"{data['n_after_voxel']} -> {target_count}{sent}")
            if len(batch_data) >= gpu_batch_size:
                flush_batch()

    flush_batch()

    cov_summary = _aggregate_coverage_summary(metadata_list)

    meta_path = os.path.join(output_folder, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'config': {
                'version': 'v3_voxel_aggregate_paper_aligned',
                'target_points': target_count,
                'dimensions': 16,
                'dim_names': DIM_NAMES_16,
                'preprocessing': {
                    'ground_removal': 'LAS classification == 2',
                    'noise_removal': 'LAS classification == 7',
                    'voxel_size_cm': voxel_size * 100,
                    'voxelization': 'per-voxel per-scanner nan-aware mean aggregation',
                    'sentinel_detection': "value==1 -> NaN for Reflectance/Amplitude/Deviation",
                    'nn_search_radius_m': nn_search_radius,
                    'nn_step': 'spatial 1-NN copy: fills per-channel missing attributes '
                               'from the nearest voxel within nn_search_radius that has '
                               'a captor measurement for that channel '
                               '(input-side co-registration, not the wavelength-scale '
                               'imputation addressed by WaveFill-Net)',
                    'residual_imputation': 'global median per attribute (return_number excluded); '
                                           'fires only when no spatial neighbour exists within radius',
                    'nn_channel_order': [
                        '1550nm (VUX-1HA, scanner=1)',
                        '905nm (miniVUX-1DL, scanner=2)',
                        '532nm (VQ-840-G, scanner=3)',
                    ],
                },
                'downsample_method': downsample_method,
                'downsample_device': 'GPU' if actual_use_gpu else 'CPU',
                'xyz_normalization': 'unit_sphere',
                'attribute_normalization': {
                    'method': 'min_max',
                    'scope': 'global' if use_global_scaling else 'per_segment',
                    'return_number': 'raw (not scaled)',
                    'global_stats_source': global_stats_source,
                    'partition_csv': partition_csv,
                    'final_csv':     final_csv,
                    'n_train_files_used_for_stats':
                        (len(train_files) if train_files is not None else 0),
                    'leakage_note':  (
                        "Global bounds computed over train (test_set_flag==0) "
                        "only. val (test_set_flag==2) and test "
                        "(test_set_flag==1) are scaled with those bounds, "
                        "clipped to [0, 1]."
                    ),
                },
                'global_stats': global_stats if use_global_scaling else None,
                'total_sentinels_replaced': total_sentinels,
                'coverage_summary': cov_summary,
            },
            'files': metadata_list,
        }, f, indent=2)

    if skipped:
        with open(os.path.join(output_folder, "skipped_files.txt"), 'w') as f:
            for s in skipped:
                f.write(f"{s}\n")

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("Done.")
    print(f"  Successful:     {successful}")
    print(f"  Skipped:        {len(skipped)}")
    print(f"  Sentinels '1':  {total_sentinels}")
    print(f"  Time:           {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Output:         {output_folder}")
    print(f"  Metadata:       {meta_path}")

    print()
    print("  Coverage summary  (intensity column, by wavelength):")
    print(f"    {'wavelength':>12}  {'native':>10}  {'1-NN fill':>10}  {'median fb':>10}")
    n_vox = cov_summary['n_voxels_total'] or 1
    for wl, wl_short in (('1550nm', '1550'), ('905nm', '905'), ('532nm', '532')):
        nat = cov_summary['native_captor']['intensity'][wl]['count']
        nnf = cov_summary['nn_filled_within_20cm']['intensity'][wl]['count']
        med = cov_summary['global_median_fallback'][f'intensity_{wl_short}']['count']
        print(f"    {wl:>12}  {nat/n_vox*100:>9.2f}%  {nnf/n_vox*100:>9.2f}%  {med/n_vox*100:>9.2f}%")
    print("    (native + 1-NN fill = captor-derived;  median fb = non-measurement)")
    print("=" * 80)


# ----- Entry point -------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Voxel-aggregated 16D preprocessing '
    )
    parser.add_argument('--input-folder', type=str,
                        default='full_data_HeliALS',
                        help='directory with raw .las files')
    parser.add_argument('--output-folder', type=str,
                        default='HeliALS_voxelagg_1024_16D',
                        help='directory where preprocessed .npy files are written')
    parser.add_argument('--target-points', type=int, default=1024,
                        help='paper: 8192')
    parser.add_argument('--file-ext', type=str, default='las')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='paper: 5 cm = 0.05 m')
    parser.add_argument('--nn-search-radius', type=float, default=0.20,
                        help='cross-channel NN fill radius; paper does not specify '
                             'for model step, 20 cm matches TerraScan triplet radius')
    parser.add_argument('--downsample-method', type=str, default='fps',
                        choices=['fps', 'random'])
    parser.add_argument('--per-segment-scaling', action='store_true',
                        help='disable global min-max scaling (use per-segment '
                             'instead). Per-segment scaling is leak-free by '
                             'construction; if you use it, --partition-csv is '
                             'not needed.')
    parser.add_argument('--partition-csv', type=str,
                        default='data/final-segments-with-species.csv',
                        help='CSV with columns test_site_name, segment_id, '
                             'test_set_flag. Global scaling stats are computed '
                             'ONLY over rows with test_set_flag == 0 (train) to '
                             'prevent test-set extrema from leaking into the '
                             'normalisation parameters. Default is the project '
                             'CSV used by the dataset class.')
    parser.add_argument('--final-csv', type=str, default='',
                        help='optional quality-filter CSV (intersected with '
                             'the partition CSV). Empty string = disabled, '
                             'which is the default now that the partition CSV '
                             '(data/final-segments-with-species.csv) is itself '
                             'the post-filter list.')
    parser.add_argument('--num-workers', type=int, default=min(8, mp.cpu_count()))
    parser.add_argument('--gpu-batch-size', type=int, default=64)
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    process_all(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        target_count=args.target_points,
        file_ext=args.file_ext,
        voxel_size=args.voxel_size,
        nn_search_radius=args.nn_search_radius,
        num_workers=args.num_workers,
        gpu_batch_size=args.gpu_batch_size,
        use_gpu=(not args.no_gpu),
        downsample_method=args.downsample_method,
        use_global_scaling=(not args.per_segment_scaling),
        partition_csv=(None if args.per_segment_scaling else args.partition_csv),
        final_csv=(args.final_csv if args.final_csv else None),
    )

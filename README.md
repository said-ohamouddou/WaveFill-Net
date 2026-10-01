# WaveFill-Net

Code for the paper **"WaveFill-Net: Missing-Wavelength Imputation with
Kolmogorov–Arnold Networks for Multi-Sensor Multispectral LiDAR Tree-Species
Classification"** (Ohamouddou, El Afia, El Afia, Chiheb, Ben Hamza).

When one wavelength of a multispectral airborne laser scanning (ALS) point
cloud is missing, WaveFill-Net fills its radiometric channels point by point
so that a classifier trained on complete inputs can be used unchanged. The
imputer is trained with a reconstruction loss plus a classification loss
propagated through the frozen classifier. Three variants are provided: MLP,
ReLU-KAN, and FastKAN.

## Repository layout

| Path | Purpose |
|---|---|
| `data_preparation/1b-voxel_aggregate_16D_paper_aligned.py` | Raw HeliALS `.las` segments → 1024-point, 16-feature `.npy` files (5 cm voxel aggregation, cross-scanner NN fill, FPS, train-only min–max scaling) |
| `data_preparation/make_partition_csv.py` | Creates a stratified train/val/test split (not needed if you use `splits/`) |
| `splits/final-segments-with-species.csv` | The fixed split used in the paper (`test_set_flag`: 0 = train, 1 = test, 2 = val; 1060 / 3868 / 1310 segments) |
| `TreeSpeciesDatasetHELIALS.py` | Dataset loader |
| `train_classifier_ensemble.py`, `train_classifiers.sh` | Stage 1: train the 5-seed classifiers (PointNet, DGCNN, DeepGCN, PCT, Point Transformer V2) on complete inputs |
| `helials_classifier.py`, `models/`, `config/` | Classifier backbones and their configs |
| `main.py` | WaveFill-Net imputers, baselines (zero, mean, linear regression, kNN, random forest), training, and paired seed-matched evaluation |
| `torch_relu_kan.py`, `torch_fast_kan.py`, `fast_kan_utils.py` | ReLU-KAN and FastKAN layers |
| `run_imputation_experiments.py` | Stage 2: all baselines and WaveFill-Net variants for every backbone and seed |
| `ablation_components_kan.py` | Stage 3: component ablation and cross-classifier transfer (FastKAN, PTv2 teacher) |
| `ablation_hparam_kan.py` | Stage 4: loss-weight, hidden-width and RBF-grid sweeps |
| `run_main_experiment.sh` | Runs stages 1–5 end to end |
| `build_master_table.py`, `build_recall_time_params.py`, `export_ablation_figures.py`, `export_confusion_matrices.py`, `class_distribution_from_csv.py` | Build the paper's tables and figures from `runs/` |
| `extentions/` | CUDA extensions `pointops` and `pointnet2_ops` (sources only) |

## Installation

Tested with Python 3.11, PyTorch 2.4.1 + CUDA 11.8 on an NVIDIA RTX A4500.

```bash
conda create -n torch311 python=3.11 -y
conda activate torch311
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install -r requirements.txt

# CUDA extensions (needed by PointTransformerV2, PCT and others)
pip install ./extentions/pointops
pip install ./extentions/pointnet2_ops
```

The shell scripts activate a conda env named `torch311` if one exists under
`$CONDA_ROOT` (default `~/miniconda3`); otherwise they use the active Python.

## Data

The HeliALS multispectral ALS tree-species dataset is public:
<https://doi.org/10.5281/zenodo.17077255> (Taher et al., ISPRS J. Photogramm.
Remote Sens., 2026). It is **not** included in this repository.

1. Download the dataset and put the raw per-tree `.las` segments in one folder,
   e.g. `full_data_HeliALS/`.
2. Copy the paper's split to where the code expects it:
   ```bash
   mkdir -p data
   cp splits/final-segments-with-species.csv data/
   ```
3. Preprocess (the scaling statistics are computed on training segments only):
   ```bash
   python data_preparation/1b-voxel_aggregate_16D_paper_aligned.py \
       --input-folder full_data_HeliALS \
       --output-folder data/HeliALS_voxelagg_1024_16D \
       --partition-csv data/final-segments-with-species.csv \
       --target-points 1024
   ```

Expected layout afterwards:

```
data/
├── final-segments-with-species.csv
└── HeliALS_voxelagg_1024_16D/   # one .npy per tree segment
```

## Reproducing the paper

Full pipeline (classifiers, main comparison, both ablations, master table),
with the settings used in the paper:

```bash
ABL_SEEDS_HP=0,1,2,3,4 bash run_main_experiment.sh
```

`ABL_SEEDS_HP` must be set: the hyperparameter sweep defaults to one seed,
but the paper reports five. The script is resumable, and anything already in
`runs/` is skipped. Other useful switches:

```bash
DRY_RUN=1 bash run_main_experiment.sh                        # print commands only
SKIP_CLASSIFIERS=1 bash run_main_experiment.sh               # classifiers already trained
BACKBONES="PointNet" SEEDS=0 EPOCHS_IMP=5 RUN_ABLATIONS=0 \
    bash run_main_experiment.sh                              # quick smoke test
```

Individual stages:

```bash
bash train_classifiers.sh                                    # stage 1
python run_imputation_experiments.py --epochs 100 \
    --seeds 0,1,2,3,4 --imputers mlp,relukan,fastkan         # stage 2
python ablation_components_kan.py --seeds 0,1,2,3,4          # stage 3
python ablation_hparam_kan.py --seeds 0,1,2,3,4              # stage 4
```

Tables and figures:

| Paper item | Command |
|---|---|
| Main comparison table | `python build_master_table.py` |
| Per-class recall, fill time, parameters | `python build_recall_time_params.py --backbone PointTransformerV2` |
| Component ablation table, sweep figures | `python export_ablation_figures.py` |
| Confusion matrices | `python export_confusion_matrices.py --backbone PointTransformerV2` |
| Class distribution | `python class_distribution_from_csv.py` |

Outputs are written to `runs/`:

```
runs/
├── all_16d_backbones_1024pts_5seed/<BACKBONE>/classifier/model_seed{0..4}.pt
├── method/<BACKBONE>/summary.json, seed<S>/eval_results.json
├── ablation_components_kan/results_aggregated.csv
└── ablation_hparam_kan/results_aggregated.csv
```

### Default settings

| | Classifier | WaveFill-Net |
|---|---|---|
| Optimiser | AdamW, lr 1e-3, wd 0.05 | Adam, lr 2e-4, wd 1e-4 |
| Epochs | 200 (early stopping, patience 30) | 100 |
| Batch size | 16 | 16 |
| Loss | CE, label smoothing 0.1 | 10 · L1 reconstruction + 0.5 · CE (frozen classifier) |
| Hidden width | — | MLP 256 / ReLU-KAN 88 / FastKAN 84 |
| Seeds | 0–4 | 0–4, imputer seed *s* paired with classifier seed *s* |

Timings depend on hardware and batching. The neural imputers run on the GPU,
and the scikit-learn baselines run on the CPU.

## Citation

```bibtex
@article{ohamouddou_wavefillnet,
  title  = {WaveFill-Net: Missing-Wavelength Imputation with Kolmogorov--Arnold
            Networks for Multi-Sensor Multispectral LiDAR Tree-Species Classification},
  author = {Ohamouddou, Said and El Afia, Hanaa and El Afia, Abdellatif and
            Chiheb, Raddouane and Ben Hamza, A.},
  note   = {Under review},
  year   = {2026}
}
```

## Acknowledgements

The backbone implementations in `models/` and the CUDA extensions in
`extentions/` are adapted from their original authors; each folder keeps its
original `LICENSE`. The ReLU-KAN and FastKAN layers follow Qiu et al. (2024) and
Li (2024).

"""
Backbone factory for the HeliALS 16D multi-wavelength tree-species
classifier.

All four training/eval scripts do
`from helials_classifier import HeliALSClassifier` and construct it as
`HeliALSClassifier(cfg)` where `cfg` carries at least
`num_classes`, `input_dim`, `smooth`, and optionally `backbone`
(defaults to 'PointTransformer').

This file imports the `models/` package, which self-registers every
available backbone into `models.build.MODELS`, then dispatches on
`cfg.backbone`. Backbones whose CUDA extension is missing emit a warning at
package import time and are simply absent from the registry.

Each backbone in the zoo reads a different subset of config fields
(`channels_input`, `emb_dims`, `dropout`, `k`, ...). The factory ships a
generic set of common-knob defaults plus per-backbone overrides; anything
the caller sets on the input cfg always wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import yaml  # PyYAML, used to read config/classification/<BB>/HELIALS/*.yaml
except ImportError:  # graceful: factory still works on hardcoded defaults
    yaml = None  # type: ignore

import models  # noqa: F401  - side-effect: registers backbones
from models.build import MODELS


DEFAULT_BACKBONE = 'PointTransformer'

# Where the per-backbone HeliALS YAML configs live, relative to this file.
_CONFIG_ROOT = Path(__file__).resolve().parent / 'config' / 'classification'

# Some backbone names don't map 1:1 to a directory or filename in the
# config tree. The mapping is `registered_name -> (subdir, filename)`,
# both relative to `_CONFIG_ROOT/<subdir>/HELIALS/<filename>`.
# Anything not listed here defaults to `<name>/HELIALS/helials.yaml`.
_BACKBONE_YAML_MAP: dict[str, tuple[str, str]] = {
    'PointNet2_MSG':  ('PointNet2', 'helials_msg.yaml'),
}


# Generic fallback defaults, used when no YAML config is found for a
# backbone and no per-backbone override is provided either.
_COMMON_DEFAULTS: dict[str, Any] = {
    'emb_dims': 1024,
    'dropout': 0.5,
    'dropout1': 0.5,
    'dropout2': 0.5,
    'k_neighbors': 20,
    'k': 20,
    'num_select': 256,
    'patch_dim': 64,
    'depth': 4,
}


# Fields that may appear inside a YAML `model:` block but are NOT
# architecture hyperparameters. We strip them at load time so the YAML
# only contributes architecture; everything dataset- or training-related
# is set from CLI flags / the caller's cfg.
#
#   NAME                  - registry dispatch key, set by `cfg.backbone`
#   num_classes           - dataset property, set by `cfg.num_classes`
#   channels_input        - input feature width, set by --input-dim
#   input_channels        - PointNet alias for channels_input
#   channels              - DeepGCN alias for channels_input
#   label_smoothing       - training-loss hyperparameter, set by --label-smoothing
_NON_MODEL_FIELDS: set[str] = {
    'NAME',
    'num_classes',
    'channels_input',
    'input_channels',
    'channels',
    'label_smoothing',
}


def _load_backbone_yaml(name: str) -> dict[str, Any] | None:
    """Return the architecture-only `model:` block from the matching HELIALS
    YAML config, with non-architecture fields stripped out. Returns None if
    PyYAML is missing or no file exists.

    Training-side keys (optimizer, scheduler, max_epoch, grad_norm_clip,
    dataset, npoints, batch sizes, ...) live at the YAML top level and are
    never loaded by this factory - training config is fully owned by the
    CLI of train_classifier_ensemble.py / train_wavefill_net.py.
    """
    if yaml is None:
        return None
    subdir, filename = _BACKBONE_YAML_MAP.get(name, (name, 'helials.yaml'))
    path = _CONFIG_ROOT / subdir / 'HELIALS' / filename
    if not path.is_file():
        return None
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    model = doc.get('model') or {}
    return {k: v for k, v in model.items() if k not in _NON_MODEL_FIELDS}


# Backbone-specific overrides on top of _COMMON_DEFAULTS. Picked from each
# backbone's reference config; conservative enough to construct cleanly,
# the caller can override any field by setting it on the input cfg.
_BACKBONE_DEFAULTS: dict[str, dict[str, Any]] = {
    'PointTransformer': {},
    'PointTransformerV2': {},
    'DGCNN': {'emb_dims': 1024, 'dropout': 0.5, 'k': 20},
    'PointNet': {'emb_dims': 1024, 'dropout': 0.3, 'feature_transform': True},
    'PointNet2_MSG': {'dropout': 0.5},
    'PCT': {'dropout': 0.5},
    'PointMLP': {'k_neighbors': 24, 'dropout': 0.5},
    'DeepGCN': {
        'channels': 64,
        'k': 20,
        'act': 'relu',
        'norm': 'batch',
        'bias': True,
        'knn': 'matrix',
        'epsilon': 0.2,
        'stochastic': True,
        'conv': 'edge',
        'c_growth': 0,
        'emb_dims': 1024,
        'dropout': 0.5,
        'n_blocks': 14,
        'block_type': 'res',
        'use_dilation': True,
    },
    'GDAN': {'k': 30, 'num_select': 256, 'dropout': 0.4},
    'KANDGCNN': {'emb_dims': 1024, 'k': 20},
}


class _BackboneConfig:
    """Config object understood by every BasePointCloudModel subclass.

    Supports both attribute access (`config.channels_input`) and dict-style
    `config.get('k', 30)`. Unknown attribute reads raise AttributeError so
    missing required fields surface immediately, rather than silently
    propagating `None` into the backbone's arithmetic.
    """

    def __init__(self, fields: dict[str, Any]):
        self.__dict__.update(fields)

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return f"_BackboneConfig({self.__dict__})"


def _build_config(cfg) -> _BackboneConfig:
    """Translate the slim HeliALS cfg (num_classes/input_dim/smooth/backbone)
    into a full config the chosen backbone can consume.

    Lookup order for each field, lowest -> highest priority:
      1. `_COMMON_DEFAULTS`           (generic catch-all)
      2. `_BACKBONE_DEFAULTS[name]`   (hardcoded fallback per backbone)
      3. YAML at config/classification/<BB>/HELIALS/*.yaml   (model: block)
      4. Required runtime fields (NAME, num_classes, channels_input, etc.)
      5. Anything explicitly set on the caller's cfg object.
    """
    backbone = getattr(cfg, 'backbone', DEFAULT_BACKBONE)
    smooth = float(getattr(cfg, 'smooth', 0.0))
    input_dim = int(cfg.input_dim)

    fields: dict[str, Any] = dict(_COMMON_DEFAULTS)
    fields.update(_BACKBONE_DEFAULTS.get(backbone, {}))

    yaml_block = _load_backbone_yaml(backbone)
    if yaml_block is not None:
        fields.update(yaml_block)
        fields['_yaml_source'] = str(
            _CONFIG_ROOT / _BACKBONE_YAML_MAP.get(backbone, (backbone, 'helials.yaml'))[0]
            / 'HELIALS'
            / _BACKBONE_YAML_MAP.get(backbone, (backbone, 'helials.yaml'))[1]
        )

    # Required by every backbone (and the only fields the upstream cfg
    # actually carries by default).
    fields.update({
        'NAME': backbone,
        'num_classes': int(cfg.num_classes),
        'channels_input': input_dim,
        'input_channels': input_dim,
        'label_smoothing': smooth,
    })

    # Any extra attribute set on the input cfg wins over everything else.
    for key, val in vars(cfg).items():
        if key in ('num_classes', 'input_dim', 'smooth', 'backbone'):
            continue
        fields[key] = val

    return _BackboneConfig(fields)


def available_backbones() -> list[str]:
    """Return the list of backbones that successfully self-registered (i.e.
    whose CUDA extension is installed in the current env)."""
    return sorted(MODELS.keys())


class HeliALSClassifier(nn.Module):
    """Backbone-agnostic classifier wrapper for HeliALS 16D inputs.

    Accepts a SimpleNamespace-style `cfg` with:
      - num_classes : int, number of species classes
      - input_dim   : int, point feature width (16 for HeliALS)
      - smooth      : float, label smoothing factor (stored for callers;
                      not used in forward)
      - backbone    : str (optional), registered class name in
                      models.build.MODELS. Defaults to 'PointTransformer'.

    Forward: (B, N, input_dim) -> (B, num_classes) logits.

    The class name is kept verbatim for backwards compatibility with the
    `from helials_classifier import HeliALSClassifier` imports already
    sprinkled across the four scripts.
    """

    def __init__(self, cfg):
        super().__init__()
        backbone_name = getattr(cfg, 'backbone', DEFAULT_BACKBONE)
        if backbone_name not in MODELS:
            raise ValueError(
                f"Backbone {backbone_name!r} is not available. "
                f"Available: {available_backbones()}. "
                f"If you expected one of "
                f"{{PointTransformer, PointTransformerV2, PointNet, "
                f"PointNet2_MSG, DGCNN, PCT, PointMLP, "
                f"DeepGCN, GDAN, KANDGCNN}}, the relevant CUDA extension "
                f"(pointops, pointnet2_ops) is probably not "
                f"installed in this environment."
            )
        backbone_cfg = _build_config(cfg)
        self.backbone_name = backbone_name
        self.num_classes = backbone_cfg.num_classes
        self.input_dim = backbone_cfg.channels_input
        self.label_smoothing = backbone_cfg.label_smoothing
        self.backbone = MODELS.get(backbone_name)(backbone_cfg)

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        return self.backbone(pc)

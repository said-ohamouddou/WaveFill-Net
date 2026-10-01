"""
GDANet: Learning Geometry-Disentangled Representation for Complementary
        Understanding of 3D Object Point Cloud

Paper: Xu et al., AAAI 2021 — https://arxiv.org/abs/2012.10921
Source: https://github.com/mutianxu/GDANet (no license)

Clean implementation.
"""

from .gdan import GDAN
from .utils import (
    extract_local_features,
    geometry_disentangle,
    CrossAttentionModule,
)

__all__ = [
    "GDAN",
    "extract_local_features",
    "geometry_disentangle",
    "CrossAttentionModule",
]

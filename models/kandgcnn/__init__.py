"""
Introducing the Short-Time Fourier Kolmogorov Arnold Network: A Dynamic Graph CNN Approach for Tree Species Classification in 3D Point Clouds

Paper: Pattern Recognition 2026
Authors: Said Ohamouddou, Abdellatif El Afia, Mohamed Ohamouddou, Rafik Lasri, Hanaa El Afia, Raddouane Chiheb
Source: Implementation adapted from: https://github.com/said-ohamouddou/STFT-KAN-liteDGCNN
License: MIT
"""
from .kan_dgcnn import KANDGCNN

__all__ = ['KANDGCNN']

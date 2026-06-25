# SPHARM-Mamba: Rotation-Invariant Multiscale Modeling for Brain Age Prediction

Official preview for the MICCAI 2026 paper *SPHARM-Mamba: Rotation-Invariant Multiscale Modeling for Brain Age Prediction* in the `SPHARM-Mamba` repository.

The proposed method performs cortical surface-based brain age prediction using spherical harmonic representations and Mamba-based sequence modeling. It captures multiscale spectral patterns on cortical surfaces while preserving rotation-invariant properties.

<p align="center">
  <img src="fig/overview.png" width="85%">
</p>

## Description

Cortical surface analysis provides a compact representation of brain morphology and can capture structural patterns that are not fully described by volumetric summaries. However, surface-based learning requires careful handling of spherical geometry and rotation-related variability.

SPHARM-Mamba represents cortical surface features in the spherical harmonic domain and uses harmonic convolution to extract geometry-aware spectral features. Instead of treating harmonic degrees independently, the model applies Mamba-based sequence modeling to capture cross-degree dependencies across multiscale spectral descriptors. This design enables efficient and rotation-invariant modeling of cortical surface signals for brain age prediction.

The framework is designed for spherical cortical surface data with multiple anatomical features such as cortical thickness, surface area, sulcal depth, and curvature. By combining spherical harmonic convolution with sequence modeling in the spectral domain, SPHARM-Mamba aims to improve surface-based prediction while maintaining stable geometric inductive bias.

## Status

Code and pretrained resources are coming soon.

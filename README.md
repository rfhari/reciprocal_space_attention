# 📣🌟Reciprocal Space Attention (RSA) for Learning Long-Range Interactions 

[![arXiv](https://img.shields.io/badge/arXiv-2510.13055-b31b1b.svg)](https://arxiv.org/abs/2510.13055v1) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit)


# Abstract
Machine learning interatomic potentials (MLIPs) have revolutionized the modeling of materials and molecules by directly fitting to \emph{ab initio} data. However, while these models excel at capturing local and semi-local interactions, they often prove insufficient when an explicit and efficient treatment of long-range interactions is required. To address this limitation, we introduce Reciprocal-Space Attention (RSA), a framework designed to capture long-range interactions in the Fourier domain. RSA can be integrated with any existing local or semi-local MLIP framework. 
The central contribution of this work is the mapping of a linear-scaling attention mechanism into Fourier space, enabling the explicit modeling of long-range interactions such as electrostatics and dispersion without relying on predefined charges or other empirical assumptions. We demonstrate the effectiveness of our method as a long-range correction to the MACE backbone across diverse benchmarks, including dimer binding curves, dispersion-dominated layered phosphorene exfoliation, and the molecular dipole density of bulk water. Our results suggest that RSA consistently captures long-range physics across a broad range of chemical and materials systems.

## Reciprocal Space Attention Architecture

<p align="center">
  <img src="https://github.com/rfhari/reciprocal_space_attention/raw/main/figs/Figure1.png">
</p>

## Highlights 
Graph Neural Networks (GNN) based Machine Learning Potentials struggle with long-range effects, crucial for atomic systems dominated by electrostatics or dispersion, due to oversquashing/oversmoothing or no information flow on disconnected graphs. 

- Inherently encodes interactions across periodic images via Bloch phase factors, going beyond the purely real-space attention approach.

- Captures long-range interactions without relying on local/semi-local observables like atom-centered partial charges or empirical corrections.

- Generalizes across various benchmarks where dispersion, electrostatic, and polar interactions dominate.

## License

The code is published and distributed under the [MIT License](https://opensource.org/license/mit).
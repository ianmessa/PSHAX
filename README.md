# PSHAX

![PSHAX Logo](misc/logo.png)

PSHAX is a probabilistic seismic hazard analysis (PSHA) library written in Jax for my undergraduate honors thesis in the CU Boulder geography department. There are three parts:
- pshax.numerics: A module full of numerical tools.
- pshax.seismic: A module containing scenario modeling tools and the five GMMs from NGA-West2 (+AAY14 epistemic uncertainty).
- pshax.hazcalc: A module containing the HazCalculator class.

This library was written with the express purpose of implementing the Karhunen-Loève + Polynomial Chaos Expansion (KL-PCE) framework for epistemic uncertainty propagation (EUP) over ground motion models (GMMs) described in Lacour & Abrahamson, 2021. It includes the following improvements to the original framework:
- Non-parametric kernels for the inferred Gaussian process over GMMs for faster evaluation of the KL expansion.
- Inclusion of epistemic uncertainty for arbitrary input parameters (e.g., $V_{s30}$).
- Randomized quasi-Monte Carlo sampling over domain of input parameters for improved sample-efficiency.
- A randomized block-Krylov solver for the eigenvalue problem associated with the KL expansion, which is helpful for integration over many parameters $\[\vec{\theta} \in \mathbb{R}^{d_{in}}\]$ and high-accuracy reconstruction of the epistemic uncertainty function $\[\sigma_{epi,haz}(\vec{\theta})\]$.
- A polynomial-augmented radial basis function interpolator for eigenfunctions. This improves system conditioning and eigenfunction interpolation, especially for high-dimensional (large $d_{in}$) EUP. This is mostly intended for site-specific analyses. 
- A PCE fitted to mean-hazard residuals in log-log space for numerical stability.
- Max-sum, max-product, and hyperbolic truncation schemes for the Hermite basis. Note that lower PCE degrees have been shown to yield more liberal EU estimates.
- Automatic Taylor series construction for the PCE inspired by Lacour & Abrahamson, 2025. This is an experimental feature which displays some pretty nasty instabilities.
- Written in JAX so it's amenable to GPU acceleration and fully differentiable. 

I plan to expand this library as a demonstration of the continuous treatment of PSHA logic trees as nested stochastic processes, sort of like Bayesian hierarchical models. The branches of our logic tree are samples of the defensible model space; how can we use these samples to variationally infer the structure of that whole space using data-driven methods? I'm curious about variational inference because it can allow us to push a bigger part of the tree through the PCE and use additive methods (same as we use for the mean hazard) instead of MC/MCMC for EUP.

## Documentation

## Usage

Feel free to fork this and do whatever. There's no good CLI right now; it's mostly designed for writing PSHA scripts where you design the CLI and working in notebooks.

## Important Reminders

The pre- and post-averaged approaches to epistemic uncertainty calculation implemented in this library will *not* yield the same $\sigma_{epi,haz}$ estimates; they will not even yeild the same $\mu_{haz}$ estimates! Assuming the GMM-space to be a Gaussian distribution or process in GMM-space will reduce $\sigma_{epi,haz}$ relative to assuming no structure at all. If we only use the GMMs on our logic tree to intuit the shape of the GMM-space, we get something closer to a multi-modal distribution; for this reason, it would be helpful to implement the arbitrary (Paulson et al., 2017) or generalized (Xiu and Karniadakis, 2003) PCE in the future.

At the time of release, the accuracy of every single solver in this library will be adjustable. The default tolerances balance accuracy and computational speed. Ablation studies for these parameters have not yet been produced. 

At the time of release, this library has not yet been benchmarked against widely used PSHA libraries like [HAZ](https://github.com/abrahamson/HAZ) [OpenSHA](https://opensha.org/) or [OpenQuake Engine](https://github.com/gem/oq-engine). 

## Dependencies

## Installation

Just run 
```
pip install git+https://github.com/ianmessa/PSHAX
```

## Citing PSHAX

Just cite this paper: 

Messa, I. (2026). Accelerated Propagation of Epistemic Uncertainty for PSHA Using a Differentiable KL-PCE Framework. Geography Department, University of Colorado Boulder. *this link will be updated once the thesis is added to the UCB Honors Thesis Repository*

## Acknowledgements

This paper was written under the advisory of Morteza Karimzadeh and defended in front of a board led by Bill R. Travis, with sitting board members Stephen Becker and Shideh Dashti. 

Extended thanks to my supervisors and colleagues at the USGS who have given me input on this project, including Morgan Moschetti, Nicolas Luco, Peter Powers, and Andrew Makdisi.

Special thanks to Maxime Lacour, who met with me to discuss this research and clarify some details about his 2021 paper with Norm Abrahamson. 

## References

Lacour, M., & Abrahamson, N. (2021). Efficient Propagation of Epistemic Uncertainty for Probabilistic Seismic Hazard Analyses (PSHAs) Including Partial Correlation of Magnitude–Distance Scaling. Bulletin of the Seismological Society of America, 111 (6), 3332–3340. https://doi.org/10.1785/0120200381

Lacour, M., & Abrahamson, N. (2025). Reducing Calculation Times for Seismic Hazard Using Non-Ergodic Ground-Motion Models for Areal Source Zones. Applied Sciences, 15(5), 2454. https://doi.org/10.3390/app15052454
## Acknowledgements

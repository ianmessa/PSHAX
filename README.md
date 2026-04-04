# PSHAX

<img src='misc/logo.png' width = '500'>

**THIS REPO IS ONLY ONLINE AS AN EXAMPLE FROM MY HONORS THESIS. THE METHODS DESCRIBED IN IT ARE BEING RESEARCHED AND EXTENDED TO A NEW REPO UNDER THE SAME NAME WHICH WILL REPLACE THIS ONE IN THE FUTURE.**

PSHAX is a probabilistic seismic hazard analysis (PSHA) library written in Jax for my undergraduate honors thesis in the CU Boulder geography department. There are three parts:
- pshax.numerics: A module full of numerical tools.
- pshax.seismic: A module containing scenario modeling tools and the five GMMs from NGA-West2 (+AAY14 epistemic uncertainty).
- pshax.hazcalc: A module containing the HazCalculator class.

This library was written with the express purpose of implementing the Karhunen-Loève + Polynomial Chaos Expansion (KL-PCE) framework for propagating epistemic uncertainty from *pre-hazard averaged* ground motion models described in Lacour & Abrahamson 2021. I'll link the paper here when it's added to the Boulder honors thesis repo. 

I plan to expand this library as a demonstration of treating PSHA logic trees as nested stochastic processes, sort of like Bayesian hierarchical models. The branches of our logic tree are samples of the defensible model space; how can we use these samples to variationally infer the structure of that whole space using data-driven methods? I'm curious about variational inference because it can allow us to push a bigger part of the tree through the PCE and avoid time-consuming MCMC.

## Documentation

I'm not documenting this repo because it isn't robust enough to be useful. It was written as a demonstration for my thesis, and while I'm proud of it, generalizing the methods enough to be useful will require total restructuring. This is something I'm working on right now.

## Usage

Feel free to fork this and do whatever. There's no good CLI right now; it's mostly designed for writing PSHA scripts where you design the CLI and working in notebooks.

## Important Reminders

The pre- and post-averaged approaches to epistemic uncertainty calculation implemented in this library will *not* yield the same $\sigma_{epi,haz}$ estimates; they will not even yeild the same $\mu_{haz}$ estimates! Assuming the GMM-space to be a Gaussian distribution or process in GMM-space will reduce $\sigma_{epi,haz}$ relative to assuming no structure at all. If we only use the GMMs on our logic tree to intuit the shape of the GMM-space, we get something closer to a multi-modal distribution; for this reason, it would be helpful to implement the arbitrary (Paulson et al., 2017) or generalized (Xiu and Karniadakis, 2003) PCE in the future.

At the time of release, this library has not yet been benchmarked against widely used PSHA libraries like [HAZ](https://github.com/abrahamson/HAZ) [OpenSHA](https://opensha.org/) or [OpenQuake Engine](https://github.com/gem/oq-engine). 

## Dependencies

Don't worry about it.

## Installation

I would recommend you just copy paste anything you're interested in or download the zip. Jax should be close to all you need.

## Citing PSHAX

Just cite this paper: 

Messa, I. (2026). Accelerated Propagation of Epistemic Uncertainty for PSHA Using a Differentiable KL-PCE Framework. Geography Department, University of Colorado Boulder. *this link will be updated once the thesis is added to the UCB Honors Thesis Repository*

## Acknowledgements

This paper was written under the advisory of Morteza Karimzadeh and defended in front of a board led by Bill R. Travis, with sitting board members Stephen Becker and Shideh Dashti. Huge thanks to everyone on the board, especially Morteza for giving himself a PSHA crash course in like a week.

Extended thanks to my supervisors and colleagues at the USGS who have given me input on this project, including Morgan Moschetti, Nicolas Luco, Peter Powers, and Andrew Makdisi.

Thanks also to Maxime Lacour, who met with me to discuss this research and clarify some details about his 2021 paper with Norm Abrahamson. 

## References

Lacour, M., & Abrahamson, N. (2021). Efficient Propagation of Epistemic Uncertainty for Probabilistic Seismic Hazard Analyses (PSHAs) Including Partial Correlation of Magnitude–Distance Scaling. Bulletin of the Seismological Society of America, 111 (6), 3332–3340. https://doi.org/10.1785/0120200381

Lacour, M., & Abrahamson, N. (2025). Reducing Calculation Times for Seismic Hazard Using Non-Ergodic Ground-Motion Models for Areal Source Zones. Applied Sciences, 15(5), 2454. https://doi.org/10.3390/app15052454

---
title: 'Furax: A JAX-based framework for linear operators in cosmological data analysis'
tags:
  - Python
  - JAX
  - cosmology
  - CMB
  - inverse problems
  - linear operators
authors:
  - name: Pierre Chanial
    orcid: 0000-0003-1753-524X
    corresponding: true
    affiliation: 1
  - name: Simon Biquard
    orcid: 0000-0001-5649-4182
    affiliation: 1
  - name: Wassim Kabalan
    orcid: 0000-0003-2651-0314
    affiliation: 1
  - name: Wuhyun Sohn
    affiliation: 1
  - name: Josquin Errard
    orcid: 0000-0002-1419-0031
    affiliation: 1
affiliations:
  - name: Université Paris Cité, CNRS, Astroparticule et Cosmologie, F-75013 Paris, France
    index: 1
    ror: 03tnjrr49
date: 21 March 2026
bibliography: paper.bib
---

# Summary

Furax (Framework for Unified and Robust data Analysis with JAX) is an open-source Python framework designed for constructing and manipulating linear operators in astrophysical inverse problems. Built on the JAX ecosystem [@jax2018], Furax provides a composable operator algebra with support for automatic differentiation, just-in-time compilation, and hardware acceleration on GPUs and TPUs.

The framework combines generic linear-algebra operators with domain-specific components used in cosmic microwave background (CMB) data analysis. These include pointing operators, polarization modulation models, Stokes parameter rotations, and spectral energy distribution (SED) operators used in component separation. Furax enables researchers to rapidly prototype and test analysis pipelines while maintaining performance compatible with large-scale cosmological datasets.

# Statement of Need

Modern CMB experiments such as LiteBIRD [@litebird2023], the Simons Observatory [@simons2019], the South Pole Observatory [@spo] and CMB-S4 [@cmbs4-2022] produce extremely large time-ordered datasets that must be processed to reconstruct sky maps and extract cosmological parameters. These analyses rely heavily on repeated applications of linear operators describing the instrument response, noise filtering, and astrophysical models.

Existing frameworks provide high-performance data processing pipelines but generally lack differentiability and tight integration with modern machine-learning tools. As gradient-based inference methods become increasingly important in cosmological data analysis, there is a growing need for software frameworks that combine efficient operator algebra with automatic differentiation and accelerator support.

Furax addresses this need by providing a differentiable operator framework built on JAX. The framework allows researchers to construct complex forward models from composable operators while benefiting from JAX’s automatic differentiation and hardware acceleration capabilities.

# State of the Field

Several software packages support CMB data analysis workflows. The TOAST framework [@toast2021] provides large-scale MPI-parallel pipelines used in production analyses for experiments such as Planck, LiteBIRD, and the Simons Observatory. However, its C++ core prevents automatic differentiation and limits integration with modern differentiable programming workflows.

Healpy [@zonca2019] provides Python access to HEALPix routines widely used in CMB analysis but runs only on CPUs and does not provide an operator algebra framework. Simulation tools such as PySM [@pysm3] focus on forward sky modelling, while component separation packages such as FGBuster [@fgbuster2022] implement parametric fitting methods with simplified noise models.

Recent JAX-based tools such as jax-healpy and s2fft provide GPU-accelerated spherical transforms but do not offer a general framework for composing linear operators.

Furax complements these tools by providing a unified operator algebra framework that integrates with the JAX ecosystem and supports differentiable forward models for cosmological data analysis.

# Software Design

Furax is built around composable linear operators derived from the `lineax.AbstractLinearOperator` interface. Operators can be combined using standard algebraic syntax and applied to arrays or PyTrees:

```python
H = instrument_operator @ pointing_operator
data = H(sky_map)
sky_estimate = H.T(data)
```

This abstraction allows complex models to be expressed as compositions of simpler operators, enabling clear and modular pipeline construction.

The framework provides a collection of generic operators for common mathematical transformations, including diagonal operators, reshaping operators, masking operators, and structured matrices such as banded Toeplitz operators used in correlated noise modelling.

In addition to these generic components, Furax includes domain-specific operators relevant to CMB experiments. These include polarization rotation operators, half-wave plate modulation models, linear polarizers, and spectral energy distribution operators for astrophysical components such as the CMB, Galactic dust, and synchrotron emission.

Furax also supports block operators that enable efficient representation of multi-detector and multi-frequency systems commonly encountered in CMB experiments.

By relying on JAX transformations, Furax operators are compatible with automatic differentiation, vectorization, and just-in-time compilation, enabling efficient execution on CPUs, GPUs, and TPUs.

# Research Impact

Furax was developed within the ERC-funded SciPol project to support the development of next-generation CMB data analysis pipelines. The framework enables gradient-based optimization and differentiable forward modelling, opening new possibilities for end-to-end inference pipelines and integration with machine-learning methods.

The modular operator-based design facilitates rapid prototyping of new analysis techniques while maintaining compatibility with existing large-scale processing pipelines through interfaces with tools such as TOAST.

Furax therefore provides an important piece of infrastructure for developing analysis methods for future cosmological surveys, including LiteBIRD, the Simons Observatory, and CMB-S4.

# AI Usage Disclosure

AI-assisted tools were used for code documentation and manuscript preparation. All technical content was verified by the authors.

# Acknowledgements

This work was carried out within the \textsc{SciPol} project (\href{https://scipol.in2p3.fr}{scipol.in2p3.fr}), supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No.~101044073, PI: Josquin Errard).

Computing resources were provided by GENCI at IDRIS (Jean Zay supercomputer) under allocations 2024-AD010414161R2 and 2025-A0190416919.

This work has also received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement no. 101007633 CMB-Inflate.

# References

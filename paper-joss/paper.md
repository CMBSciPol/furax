---
title: 'Furax: A Modular JAX Framework for Linear Operators in Cosmological Data Analysis'
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
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: et al. TBD (SciPol team)
    affiliation: 1
affiliations:
  - name: Université Paris Cité, CNRS, Astroparticule et Cosmologie, F-75013 Paris, France
    index: 1
    ror: 03tnjrr49
date: 20 January 2026
bibliography: paper.bib
---

# Summary

Furax is an open-source Python framework designed to construct and manipulate linear operators for solving inverse problems in astrophysics and cosmology. Built on JAX [@jax2018], Furax draws inspiration from the PyOperators library [@chanial2012pyoperators] and lineax [@kidger2024lineax] to provide generic operators for solving inverse problems and domain-specific operators for cosmic microwave background (CMB) data analysis. The framework provides composable building blocks including pointing operators, half-wave plate models, polarizers, Stokes parameter rotations, and spectral energy distribution (SED) operators. Furax leverages JAX's automatic differentiation, just-in-time compilation, and hardware acceleration to enable gradient-based optimization on GPUs and TPUs. The modular architecture allows researchers to rapidly prototype and test analysis pipelines while maintaining computational efficiency for production-scale datasets.

# Statement of Need

Modern CMB experiments such as LiteBIRD [@litebird2023], the Simons Observatory [@simons2019], and CMB-S4 [@cmbs4-2022] generate massive time-ordered data (TOD) streams that must be processed to extract cosmological information. The central problem in CMB data analysis is map-making: recovering the sky signal $\mathbf{m}$ from noisy observations $\mathbf{d}$ through the linear model

$$\mathbf{d} = \mathbf{P}\mathbf{m} + \mathbf{n}$$

where $\mathbf{P}$ is the pointing matrix encoding the instrument response and $\mathbf{n}$ is the noise. The generalized least-squares solution

$$\hat{\mathbf{m}} = (\mathbf{P}^T \mathbf{N}^{-1} \mathbf{P})^{-1} \mathbf{P}^T \mathbf{N}^{-1} \mathbf{d}$$

requires efficient application of the pointing operator and its transpose. Existing tools like TOAST [@toast2021] provide MPI-parallel production pipelines but lack differentiability. The healpy library [@zonca2019] offers a standard HEALPix interface but is CPU-only and does not support operator algebra. Sky simulation tools like PySM [@pysm3] focus on forward modeling, while component separation codes like FGBuster [@fgbuster2022] have limited noise modeling capabilities. Furax fills this gap by providing a differentiable operator framework that integrates with modern machine learning workflows while maintaining the performance required for CMB data analysis.

# State of the Field

Several tools exist for CMB data processing. TOAST provides a comprehensive MPI-parallel framework used in production pipelines for experiments like Planck and LiteBIRD, but its C++ core prevents automatic differentiation. The healpy library wraps the HEALPix C library for Python, offering essential spherical harmonic transforms and pixel operations, but runs only on CPU and does not support operator composition. PySM generates realistic sky simulations including multiple astrophysical components, but operates strictly in forward mode. Component separation tools like FGBuster implement parametric methods but rely on simplified noise models. Other JAX-based tools such as jax-healpy [@jax-healpy2024] and s2fft [@s2fft2024] provide GPU-accelerated spherical transforms but do not offer a complete operator algebra framework. Furax complements these tools by providing a unified, differentiable operator framework that can integrate with existing pipelines through interfaces to TOAST and other libraries.

# Software Design

Furax's architecture centers on composable linear operators that extend lineax's `AbstractLinearOperator`. Operators are combined using standard mathematical notation:

```python
H = instrument_operator @ pointing @ rotation
y = H(sky_map)      # Forward model
x = H.T(data)       # Transpose application
solution = H.I(y)   # Inverse via lineax solver
```

**Operator Algebra.** The base class `AbstractLinearOperator` provides a default implementation for standard linear algebra operations that enable intuitive composition and manipulation of operators:

| Operation | Syntax | Comment |
|-----------|--------|---------|
| Addition | `A + B` |  |
| Composition | `A @ B` |  |
| Multiplication by scalar | `k * A` | Returns the composition of a HomothetyOperator and A |
| Transpose | `A.T` | Through JAX autodiff, but can be overridden |
| Inverse | `A.I` or `A.I(solver=..., maxiter=..., preconditioner=...)` | By default, the CG solver is used, but it can be overridden |
| Block Assembly | `BlockColumnOperator([A, B])`, `BlockDiagonalOperator([A, B])`, `BlockRowOperator([A, B])` | Handle any PyTree of Operators: `Block*Operator({'a': A, 'b': B})` |
| Flattened dense matrix | `A.as_matrix()` |  |
| Algebraic reduction | `A.reduce()` |  |

Table: Supported operator operations in Furax. {#tbl:operations}

**Generic Operators.** Furax provides a comprehensive suite of generic operators for common mathematical operations:

| Operator | Description |
|----------|-------------|
| `IdentityOperator` |  |
| `HomothetyOperator` |  |
| `DiagonalOperator` |  |
| `BroadcastDiagonalOperator` | Non-square operator for broadcasting |
| `TensorOperator` | For dense matrix operations |
| `TreeOperator` | For generalized matrix operations |
| `SumOperator` | Sum along axes |
| `IndexOperator` | Can be used for projecting skies onto time-ordered series |
| `MaskOperator` | Bit-encoded 0- or 1-valued mask |
| `MoveAxisOperator` |  |
| `ReshapeOperator` |  |
| `RavelOperator` |  |
| `SymmetricBandToeplitzOperator` | Methods: direct, FFT, overlap and save |
| `Block*Operator` | Block assembly operators (column, diagonal, row) |

Table: Generic operators available in Furax. {#tbl:generic-operators}

**Domain-Specific Operators.** For CMB data analysis, Furax includes specialized operators tailored to instrument modeling and astrophysical components:

| Operator | Description |
|----------|-------------|
| `QURotationOperator` |  |
| `HWPOperator` | Ideal HWP |
| `LinearPolarizerOperator` | Ideal linear polarizer |
| `CMBOperator` | Parametrized CMB SED |
| `DustOperator` | Parametrized dust SED |
| `SynchrotronOperator` | Parametrized synchrotron SED |
| `PointingOperator` | On-the-fly projection matrix |
| `MapSpaceBeamOperator` | Sparse Beam operator |
| `TemplateOperator` | For template map-making |

Table: Domain-specific operators for CMB data analysis. {#tbl:applied-operators}

**Stokes Parameter Types.** Furax represents polarization through dedicated PyTree-compatible types: `StokesI`, `StokesQU`, `StokesIQU`, and `StokesIQUV`. These types support arithmetic operations, broadcasting, and seamless integration with JAX transformations.

**Block Operators.** The framework provides three block operator types for structuring complex models: `BlockRowOperator` for horizontal concatenation, `BlockDiagonalOperator` for independent parallel operations, and `BlockColumnOperator` for vertical stacking. These operators enable efficient representation of multi-detector and multi-frequency systems.

**Algebraic Reduction.** Furax implements automatic operator simplification through a rule-based system. For example, consecutive QU rotations combine their angles, and identity operators are eliminated from compositions. The reduction system handles complex patterns including the commutation rule for half-wave plates: $R(\theta) \circ \text{HWP} = \text{HWP} \circ R(-\theta)$.

**CMB Operators.** Domain-specific operators include `HWPOperator` for half-wave plate modeling, `LinearPolarizerOperator` for polarization extraction implementing $d = \frac{1}{2}(I + Q\cos 2\psi + U\sin 2\psi)$, `QURotationOperator` for polarization angle rotations, and `HealpixLandscape` for spherical pixelization. Spectral operators (`CMBOperator`, `DustOperator`, `SynchrotronOperator`) enable frequency-dependent component separation with support for spatially varying spectral indices.

**Toeplitz Operations.** The `SymmetricBandToeplitzOperator` provides efficient convolution operations with five algorithm choices: dense multiplication, direct convolution, FFT-based, overlap-save, and overlap-add methods. This operator is central to correlated noise modeling and gap-filling procedures based on constrained Gaussian realizations [@stompor2002].

# Research Impact Statement

Furax was developed within the ERC-funded SciPol project to enable gradient-based optimization in CMB data analysis pipelines. The framework's differentiability opens new possibilities for neural network integration and end-to-end optimization of map-making and component separation. The modular design supports rapid prototyping of analysis methods while maintaining compatibility with production pipelines through TOAST integration. Furax provides essential infrastructure for developing next-generation analysis techniques for LiteBIRD, the Simons Observatory, and CMB-S4.

# AI Usage Disclosure

AI-assisted tools were used for code documentation and manuscript preparation. All technical content was verified by the authors.

# Acknowledgements

This work was supported by the European Research Council under the European Union's Horizon 2020 research and innovation programme (Grant agreement No. 101044073 - SciPol). We acknowledge the developers of JAX, lineax, and jax-healpy for providing the foundational tools upon which Furax is built. Computing resources were provided by GENCI at IDRIS (Jean Zay supercomputer) under allocation 2024-AD011015549.

# References

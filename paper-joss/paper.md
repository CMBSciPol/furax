---
title: '`Furax`: A Modular JAX Framework for Linear Operators in Cosmological Data Analysis'
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
    orcid: 0000-0002-1493-2963
    affiliation: 1,2
  - name: Wassim Kabalan
    orcid: 0000-0003-2651-0314
    affiliation: 1
  - name: Wuhyun Sohn
    orcid: 0000-0002-6039-8247
    affiliation: 1
  - name: Artem Basyrov
    orcid: 0000-0002-4365-4405
    affiliation: 1
  - name: Benjamin Beringue
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Alexandre Boucaud
    orcid: 0000-0001-7387-2633
    affiliation: 1
  - name: Magdy Morshed
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Radek Stompor
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Ema Tsang King Sang
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Amalia Villarrubia-Aguilar
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Andréa Landais
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Josquin Errard
    orcid: 0000-0002-1419-0031
    corresponding: true
    affiliation: 1
affiliations:
  - name: Université Paris Cité, CNRS, Astroparticule et Cosmologie, F-75013 Paris, France
    index: 1
    ror: 03tnjrr49
  - name: Jodrell Bank Centre for Astrophysics, The University of Manchester, Oxford Road, Manchester M13 9PL, UK
    index: 2
    ror: 027m9bs27
date: 21 March 2026
bibliography: paper.bib
---

# Summary

  The _Framework for Unified and Robust data Analysis with JAX_ (`Furax`) is an open-source Python framework for modeling data acquisition systems and solving inverse problems in astrophysics and cosmology. Built on `JAX` [@jax2018] and drawing inspiration from `PyOperators` [@chanial2012pyoperators] and `Lineax` [@kidger2024lineax], `Furax` provides composable building blocks in the form of generic and domain-specific linear operators.
  Generic operators include diagonal, block, Toeplitz, and indexing operators. Domain-specific operators are provided for cosmic microwave background (CMB) data analysis, with the architecture designed to extend to other fields: pointing matrices, half-wave plate models, polarizers, Stokes parameter rotations, and spectral energy distribution (SED) operators.
  `Furax` leverages JAX's automatic differentiation, just-in-time compilation, and hardware acceleration to enable gradient-based optimization on GPUs and TPUs. Its modular architecture allows researchers to rapidly prototype analysis pipelines while maintaining computational efficiency for production-scale datasets.


# Statement of Need

Contemporary and future CMB experiments such as the Simons Observatory [@simons2019], the South Pole Observatory [@spo], QUBIC [@qubic2022] and LiteBIRD [@litebird2023] will generate massive time-ordered data (TOD) streams that must be processed to extract cosmological information. A central problem in CMB data analysis is to exploit data acquisition redundancy through map-making, i.e. recovering the sky signal $\mathbf{m}$ from noisy observations $\mathbf{d}$ through the linear model

$$\mathbf{d} = \mathbf{H}\mathbf{m} + \mathbf{n}$$

where $\mathbf{H}$ represents the data acquisition system — encoding the pointing matrix, instrument response, and other effects — and $\mathbf{n}$ is the noise.   Several techniques can be used to estimate the solution to this equation, from the generalized least-squares estimator:

$$\hat{\mathbf{m}} = (\mathbf{H}^\top \mathbf{N}^{-1} \mathbf{H})^{-1} \mathbf{H}^\top \mathbf{N}^{-1} \mathbf{d}$$

to more sophisticated methods such as template-based map-making [@poletti2017]. All require efficient application of the acquisition operator and its transpose, and would benefit from a framework supporting operator algebra.

Furax addresses these two challenges: (1) providing a differentiable operator framework that lends itself to integration into machine learning workflows while maintaining the performance required for production-scale data analysis, and (2) offering a modular architecture that facilitates experimentation with data acquisition models and noise systematics.


# State of the Field

Several tools exist for CMB data processing in a somewhat fragmented landscape
- Sky simulation tools like `PySM` [@pysm3] generates realistic sky simulations including multiple astrophysical components, but operates strictly in forward mode
- while component separation codes like `FGBuster` [@fgbuster2022] have limited noise modeling capabilities.
- `TOAST` [@toast2021] provides a comprehensive MPI-parallel modular framework used in production pipelines for experiments like Planck, the Simons Observatory and LiteBIRD, but its C++ core prevents automatic differentiation.
- The `healpy` library [@zonca2019] wraps the HEALPix C library for Python, offering essential spherical harmonic transforms and pixel operations, but runs only on CPU and does not support operator composition.
- `jax-healpy` [@jax-healpy2024] is JAX-compatible but does not support operator algebra.
- `Commander3` [@galloway2023beyondplanck]
- `MAPRAISER` [@mappraiser2022]
- `DUCC` [@ducc] collection of highly optimized CPU C++17 subroutines
- `FGBuster` [@rizzieri2025] implement parametric methods but rely on simplified noise models.
- `PyOperators` [@chanial2012pyoperators] precursor but CPU only
- `lineax` [@kidger2024lineax] precursor, JAX-compatible but no CMB analysis operators, reliance on libraries other than `JAX`.
- Other JAX-based tools such as `s2fft` [@s2fft2024] provide GPU-accelerated spherical transforms but do not offer a complete operator algebra framework.

`Furax` complements these tools by providing a unified, differentiable operator framework that can glue together various libraries and integrate with existing pipelines through interfaces to TOAST and other libraries.

# Software Design

`Furax`'s architecture centers on composable linear operators, which are implemented as Python dataclasses registered as `JAX`
Pytrees. Operators are combined using standard mathematical notation:

```python
H = detector_response @ filter @ hwp @ pointing @ rotation  @ mixing_matrix
N = HomothetyOperator(σ**2, in_structure=H.out_structure)  # Noise covariance
m = {'cmb': jnp.random(…), 'dust': …, 'atmosphere': …, …}  # Sky components
A = (H.T @ N.I @ H).I @ H.T @ N.I  # Sky components' maximum-likelihood estimator
d = H(m) + noise                   # noisy TOD
solution = A(d)                    # Inverse via solvers
```

**Operator Algebra.** The base class `AbstractLinearOperator` provides a default implementation for standard linear algebra operations that enable intuitive composition and manipulation of operators:

| Operation                | Syntax                                                             |
|--------------------------|--------------------------------------------------------------------|
| Addition                 | `A + B`                                                            |
| Composition              | `A @ B`                                                            |
| Multiplication by scalar | `k * A`                                                            |
| Transpose                | `A.T`                                                              |
| Inverse                  | `A.I` or `A.I(solver=…, preconditioner=…)`                         |
| Block Assembly           | `BlockColumnOperator`, `BlockDiagonalOperator`, `BlockRowOperator` |
| Flattened dense matrix   | `A.as_matrix()`                                                    |
| Algebraic reduction      | `A.reduce()`                                                       |

Table: Supported operator operations in `Furax`.

**Generic Operators.** `Furax` provides a comprehensive suite of generic operators for common mathematical operations:

| Operator                        | Description                                               |
|---------------------------------|-----------------------------------------------------------|
| `IdentityOperator`              | Returns the input unchanged                               |
| `HomothetyOperator`             | Multiplication by a scalar                                |
| `DiagonalOperator`              | Element-wise multiplication                               |
| `BroadcastDiagonalOperator`     | Non-square operator for broadcasting                      |
| `TensorOperator`                | For dense matrix operations                               |
| `TreeOperator`                  | For generalized matrix operations                         |
| `SumOperator`                   | Sum along axes                                            |
| `IndexOperator`                 | Can be used for projecting skies onto time-ordered series |
| `MaskOperator`                  | Bit-encoded 0- or 1-valued mask                           |
| `MoveAxisOperator`              | Manipulate axes of input pytrees                          |
| `ReshapeOperator`               | Reshape input pytrees                                     |
| `RavelOperator`                 | Flatten input pytrees                                     |
| `FFTOperator`                   | Fast Fourier transform                                    |
| `SymmetricBandToeplitzOperator` | Methods: direct convolution, FFT, overlap and save        |
| `Block...Operator`              | Block assembly operators (column, diagonal, row)          |

Table: Generic operators available in `Furax`.

**Domain-Specific Operators.** For CMB data analysis, `Furax` includes specialized operators tailored to instrument modeling and astrophysical components:

| Operator                  | Description                  |
|---------------------------|------------------------------|
| `QURotationOperator`      | Stokes QU rotation           |
| `HWPOperator`             | Ideal HWP                    |
| `LinearPolarizerOperator` | Ideal linear polarizer       |
| `CMBOperator`             | Parametrized CMB SED         |
| `DustOperator`            | Parametrized dust SED        |
| `SynchrotronOperator`     | Parametrized synchrotron SED |
| `PointingOperator`        | On-the-fly projection matrix |
| `MapSpaceBeamOperator`    | Sparse Beam operator         |
| `TemplateOperator`        | For template map-making      |

Table: Domain-specific operators for CMB data analysis.

**Stokes Parameter Types.** `Furax` represents polarization through dedicated PyTree-compatible types: `StokesI`, `StokesQU`, `StokesIQU`, and `StokesIQUV`. These types support arithmetic operations, broadcasting, and seamless integration with JAX transformations.

**Block Operators.** The framework provides three block operator types for structuring complex models: `BlockRowOperator` for horizontal concatenation, `BlockDiagonalOperator` for independent parallel operations, and `BlockColumnOperator` for vertical stacking. These operators enable efficient representation of multi-detector and multi-frequency systems.

**Algebraic Reduction.** `Furax` implements automatic operator simplification through a rule-based system. For example, consecutive QU rotations combine their angles, and identity operators are eliminated from compositions. The reduction system handles complex patterns including the commutation rule for half-wave plates: $R(\theta) \circ \text{HWP} = \text{HWP} \circ R(-\theta)$.

**CMB Operators.** Domain-specific operators include `HWPOperator` for half-wave plate modeling, `LinearPolarizerOperator` for polarization extraction implementing $d = \frac{1}{2}(I + Q\cos 2\psi + U\sin 2\psi)$, `QURotationOperator` for polarization angle rotations, and `HealpixLandscape` for spherical pixelization. Spectral operators (`CMBOperator`, `DustOperator`, `SynchrotronOperator`) enable frequency-dependent component separation with support for spatially varying spectral indices.

**Toeplitz Operations.** The `SymmetricBandToeplitzOperator` provides efficient convolution operations with five algorithm choices: dense multiplication, direct convolution, FFT-based and overlap-save methods. This operator is central to correlated noise modeling and gap-filling procedures based on constrained Gaussian realizations [@stompor2002].

# Research Impact Statement

`Furax` was developed within the SciPol project to enable gradient-based optimization in CMB data analysis pipelines. The framework's differentiability opens new possibilities for neural network integration and end-to-end optimization of map-making and component separation. The modular design supports rapid prototyping of analysis methods while maintaining compatibility with production pipelines through TOAST integration. `Furax` provides essential infrastructure for developing next-generation analysis techniques for e.g., the Simons Observatory, QUBIC and LiteBIRD.

# AI Usage Disclosure

AI-assisted tools were used for code documentation and manuscript preparation. All technical content was verified by the authors.

# Acknowledgements

This work was carried out within the \textsc{SciPol} project (\href{https://scipol.in2p3.fr}{scipol.in2p3.fr}), supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No.~101044073, PI: Josquin Errard).

Computing resources were provided by GENCI at IDRIS (Jean Zay supercomputer) under allocations 2024-AD010414161R2 and 2025-A0190416919.

This work has also received funding by the European Union’s Horizon 2020 research and innovation program under grant agreement no. 101007633 CMB-Inflate.

# References

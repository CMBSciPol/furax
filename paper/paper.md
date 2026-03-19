---
title: '`Furax`: A Modular JAX Framework for Linear Operators in Cosmological Data Analysis'
tags:
  - Python
  - JAX
  - astrophysics
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

  The _Framework for Unified and Robust data Analysis with JAX_ (`Furax`) is an open-source Python framework for modeling data acquisition systems and solving inverse problems in astrophysics and cosmology. Built on `JAX` [@jax2018], `Furax` provides composable building blocks in the form of general-purpose and domain-specific linear operators, along with pre-conditioners and solvers for their numerical inversion.
  Domain-specific tools are provided for astrophysical and cosmic microwave background (CMB) data analysis—including map-making, instrument modeling, and astrophysical component separation—with an modular architecture designed to extend to other fields.

Furax fully utilises JAX's just-in-time compilation and automatic differentiation to achieve competitive performance, further accelerated using GPUs or TPUs. With Furax, researchers can rapidly prototype and validate analysis pipelines with production-ready computational efficiency.

`Furax` is hosted on [GitHub](https://github.com/CMBSciPol/furax), installable via [PyPI](https://pypi.org/project/furax) and documented on [Read the Docs](https://furax.readthedocs.io).


# Statement of Need

Contemporary and future CMB experiments such as the Simons Observatory [@simons2019], the South Pole Observatory [@spo], QUBIC [@qubic2022] and LiteBIRD [@litebird2023] will generate massive time-ordered data (TOD) streams that must be processed to extract cosmological information. A central problem in CMB data analysis is to exploit data acquisition redundancy through map-making, i.e. recovering the sky signal $\mathbf{m}$ (and potentially separating its components) from noisy observations $\mathbf{d}$ through the linear model

$$\mathbf{d} = \mathbf{H}\mathbf{m} + \mathbf{n}$$

where $\mathbf{H}$ represents the data acquisition system—encoding the pointing matrix, instrument response, and other effects—and $\mathbf{n}$ is the noise. Several techniques can be used to estimate the solution to this equation, from the generalized least-squares estimator:

$$\hat{\mathbf{m}} = (\mathbf{H}^\top \mathbf{N}^{-1} \mathbf{H})^{-1} \mathbf{H}^\top \mathbf{N}^{-1} \mathbf{d}$$

to more sophisticated methods such as template-based map-making [@poletti2017]. All require efficient application of the acquisition operator and its transpose, and would benefit from a framework supporting operator algebra.

Historically, many data reduction pipelines developed by large collaborations have been tied to specific experiments and did not outlive them, often due to the lack of genericity, reliance on legacy technologies or evolving hardware paradigms. Furax aims to break this pattern by being experiment-agnostic and built on Python and JAX—a modern, sustainable foundation.

`Furax` addresses the above challenges by: (1) providing a differentiable operator algebra framework, (2) offering a modular architecture that facilitates experimentation with realistic instrument models and complex noise systematics, (3) supporting the exploration of novel map-making techniques, and (4) enabling integration with production pipelines through GPU-accelerated performance for terabyte-scale datasets.


# State of the Field

Few experiment-agnostic framework for astrophysics and CMB data analysis exist.
- `TOAST` [@toast2021] provides a comprehensive MPI-parallel modular framework used in production pipelines for experiments like Planck and the Simons Observatory, but its C++ core does not fully support differentiability or GPU acceleration, although this has been explored [@demeure2023].
- `PyOperators` [@chanial2012pyoperators]: provides an operator algebra but is only used by the QUBIC data analysis pipeline. This library is `Furax` CPU-only precursor.
- `lineax` [@kidger2024lineax]: offers a JAX-compatible operator algebra but lacks domain-specific operators and relies on a third-party library for its base operator class.

<!--
On the other hand, many low-level libraries:
- `DUCC` [@ducc] collection of highly optimized CPU C++17 subroutines
- The `healpy` library [@zonca2019] wraps the HEALPix C library for Python, offering essential spherical harmonic transforms and pixel operations, but runs only on CPU and does not support operator composition.
- `jax-healpy` [@jax-healpy2024] is JAX-compatible but does not support operator algebra.
- Other JAX-based tools such as `s2fft` [@s2fft2024] provide GPU-accelerated spherical transforms but do not offer a complete operator algebra framework.
-->

  Most experiment-agnostic data analysis libraries focus on specific tasks—map-making, component separation, or sky simulation—and are to our knowledge CPU-only. MAPPRAISER [@mappraiser2022] and Commander [@galloway2023beyondplanck] are map-makers; FGBuster [@fgbuster2022; @rizzieri2025] implements parametric component separation but relies on simplified noise models; PySM [@pysm3] generates realistic multi-component sky simulations but operates strictly in forward mode.

  Furax fills this gap by providing a unified, differentiable operator framework that integrates low-level JAX-compatible libraries (such as jax-healpy [@jax-healpy2024] and s2fft [@s2fft2024]) and connects with production pipelines through interfaces to TOAST and other tools.


# Software Design

`Furax`'s architecture centers on composable linear operators, which are implemented as Python dataclasses registered as `JAX` Pytrees. Operators are combined using standard mathematical notation:

```python
H = detector_response @ band_pass @ hwp @ pointing @ rotation  @ mixing_matrix
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
| `BlockRowOperator`              | For horizontal stacking                                   |
| `BlockDiagonalOperator`         | For independent parallel computations                     |
| `BlockColumnOperator`           | For vertical stacking                                     |

Table: Generic operators available in `Furax`.

The block operators provided by the framework enable efficient structuring of complex multi-observation or multi-component systems. `SymmetricBandToeplitzOperator` provides efficient convolution operations using the overlap-save method. This operator is central to correlated noise modeling and gap-filling procedures based on constrained Gaussian realizations [@stompor2002].


**Domain-Specific Operators.** For CMB data analysis, `Furax` includes specialized operators tailored to instrument modeling and astrophysical components:

| Operator                  | Description                  |
|---------------------------|------------------------------|
| `QURotationOperator`      | Stokes QU rotation           |
| `HWPOperator`             | Ideal half-wave plate        |
| `LinearPolarizerOperator` | Ideal linear polarizer       |
| `CMBOperator`             | Parametrized CMB SED         |
| `DustOperator`            | Parametrized dust SED        |
| `SynchrotronOperator`     | Parametrized synchrotron SED |
| `PointingOperator`        | On-the-fly projection matrix |
| `MapSpaceBeamOperator`    | Sparse Beam operator         |
| `TemplateOperator`        | For template map-making      |

Table: Domain-specific operators for astrophysics or CMB data analysis.

For instance, `HWPOperator` is used for half-wave plate modeling, `LinearPolarizerOperator` for polarization extraction and `QURotationOperator` for polarization angle rotations. The spectral operators (`CMBOperator`, `DustOperator`, `SynchrotronOperator`) enable frequency-dependent component separation with support for spatially varying spectral indices.

**Algebraic Reduction.** Furax implements operator simplification through a rule-based system, complementing XLA's low-level optimizations [@xla2017]. For example, consecutive QU rotations combine their angles, and compositions involving block operators such as $P^\top N^{-1} P$ are decomposed into $\sum_i P_i^\top N_i^{-1} P_i$, exploiting block structure to reduce computational cost. The system also handles algebraic identities such as the commutation rule for half-wave plates: $R(\theta) \circ \text{HWP} = \text{HWP} \circ R(-\theta)$.

**Stokes Parameter Types.** `Furax` represents polarization through dedicated JAX Pytrees: `StokesI`, `StokesQU`, `StokesIQU`, and `StokesIQUV`. These types support arithmetic operations, broadcasting, and seamless integration with JAX transformations.

**Landscape Types.** They specify how input signals are discretized on the sky (e.g., HEALPix spherical pixelization) and provide the mapping to world coordinates.

**Map-making classes.** Several map-making algorithms are featured in `Furax`(bin filtered, template-based) (IMPROVEMENT REQUIRED).


# Research Impact Statement

`Furax` was developed within the [\textsc{SciPol} project](https://scipol.in2p3.fr) to enable GPU-accelerated and gradient-based optimization in CMB data analysis pipelines. The framework's differentiability opens new possibilities for neural network integration and end-to-end optimization of map-making and component separation. The modular design supports rapid prototyping of analysis methods while maintaining compatibility with production pipelines through TOAST integration. `Furax` provides essential infrastructure for developing next-generation analysis techniques for e.g., the Simons Observatory, QUBIC and LiteBIRD.

# AI Usage Disclosure

AI-assisted tools were used for code documentation and manuscript preparation. All AI-generated content was verified by the authors.

# Acknowledgements

Furax draws inspiration from `PyOperators` [@chanial2012pyoperators] and `lineax` [@kidger2024lineax].

This work was supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No.~101044073, PI: Josquin Errard).

Computing resources were provided by GENCI at IDRIS (Jean Zay supercomputer) under allocations 2024-AD010414161R2 and 2025-A0190416919.

This work has also received funding by the European Union’s Horizon 2020 research and innovation program under grant agreement no. 101007633 CMB-Inflate.

# References

# Changelog

All notable changes to Furax will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial documentation website with Sphinx and ReadTheDocs
- Comprehensive user guide covering data structures and operators
- Examples for CMB component separation and mapmaking
- Complete API reference with autodoc
- Contributing guidelines and development setup

## [0.1.0] - 2024-XX-XX

### Added

**Core Linear Operators**

- `AbstractLinearOperator` base class with composition support
- `DiagonalOperator` and `BroadcastDiagonalOperator` for pixel-wise operations
- `BlockDiagonalOperator`, `BlockRowOperator`, `BlockColumnOperator` for structured matrices
- `SymmetricBandToeplitzOperator` for convolution-like operations
- `IndexOperator` for pixel selection and masking
- `ReshapeOperator`, `MoveAxisOperator`, `RavelOperator` for array manipulation
- `TreeOperator` for PyTree-structured operations
- `DenseOperator` and `DenseBlockDiagonalOperator` for general matrices

**Observation Framework**

- `Stokes` abstract base class for polarization parameters
- `StokesI`, `StokesQU`, `StokesIQU`, `StokesIQUV` implementations
- `Landscape` abstract base class for sky pixelization
- `HealpixLandscape` for HEALPix sky maps
- `StokesLandscape` and `FrequencyLandscape` for multi-dimensional maps

**Instrument Operators**

- `CMBOperator` for CMB blackbody spectrum
- `DustOperator` for thermal dust emission
- `SynchrotronOperator` for synchrotron radiation
- `MixingMatrixOperator` for component separation
- HWP and polarizer response operators

**Configuration System**

- `Config` context manager for solver settings
- Integration with Lineax solvers (CG, BiCGStab, etc.)
- Default solver configuration (CG with rtol=1e-6, max_steps=500)

**Development Infrastructure**

- Pre-commit hooks with ruff, mypy, pycln
- Comprehensive test suite with pytest
- JAX x64 precision enabled in tests
- Parametrized tests for different Stokes combinations
- HPC testing support with SLURM scripts

**External Integrations**

- TOAST framework integration for observation matrices
- Gap filling and preprocessing utilities
- PyTree utilities for JAX data structures

### Changed

- N/A (initial release)

### Deprecated

- N/A (initial release)

### Removed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Security

- N/A (initial release)

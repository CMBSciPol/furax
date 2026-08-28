# Changelog

All notable changes to Furax will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `StreamOperator.block_row`/`block_column`: fuse several streams into one, evaluating a joint system in a single pass over the data (#190)
- API reference page for `furax.mapmaking.streaming` (#190)
- `AbstractLinearOperator.profile()` and a new `furax.profiling` module for estimated cost analysis (#193)
- `furax.obs.spin2`: parallel transport of Q and U across a sampling stencil, exposing `transported_gather`/`transported_scatter` and the frame rotation they are built on (#204)
- `furax.obs.stencil`, a pixelisation-free module holding `Stencil`: the pixels one sample reads, their weights and their sky positions, in one type that a landscape produces and a sampler consumes (#204)
  - a stencil is resolved when it is built, by `Stencil.resolve` or `Stencil.nearest`: its indices are in bounds and its weights sum to one, so no sampler can normalise them differently from another
  - nearest-neighbour sampling is the one-neighbour case of the same type, not a second shape
  - `StencilOrder` names how many pixels a sample reads: `NEAREST` (one) or `BILINEAR` (four)
  - `Stencil.scalar` builds one for a grid that is not the sphere, such as the atmosphere screen: it carries no neighbour positions, and sampling polarisation through it raises `ValueError` instead of transporting from a plausible wrong frame
- `StokesLandscape.world2stencil(theta, phi, order)`, returning the `Stencil` a sample reads at the requested order, implemented for HEALPix and WCS/CAR (both orders) and for astropy-WCS, horizon and subset landscapes (#204)
  - a landscape answers for every order it supports in this one method, so it cannot define one order and leave another to a mismatched inherited definition; an unsupported order raises `NotImplementedError`
  - `world2interp` now derives from it and is no longer overridable; measured on HEALPix and CAR, the compiled cost of `PointingOperator.mv` is unchanged, because XLA drops the neighbour positions nothing reads
- `CARLandscape.pixel2world`, `AstropyWCSLandscape.pixel2world` and `HorizonLandscape.pixel2world`, the inverses of their `world2pixel` (#204)
- `furax.obs.stencil.resolve_stencil`, which sends out-of-map neighbours to a safe index and normalises the remaining weights (#204)
- API reference pages for `furax.obs.spin2` and `furax.obs.stencil` (#204)

### Changed

- **Breaking:** the four stream operator classes collapse into a single `StreamOperator` (#190):
  - `StreamDiagonalOperator.create(op)` → `StreamOperator.diagonal(op)`, and likewise `.column`, `.row`, `.addition`
  - `in_stacked`/`out_stacked` say which side carries the stream axis, per component, so one stream can mix shared and per-slice components
- **Breaking:** `AbstractLinearOperator.__call__`, `BJPreconditioner.create`, and the `LBSObservation`/`ToastObservation` pointing helpers now raise `TypeError` (previously `ValueError`/`RuntimeError`) for invalid argument/operator/landscape types (#194)
- Bumped ruff to 0.16.1 and updated rule selection accordingly (#194)
- **Breaking:** pointing now carries the Q and U of every pixel it reads into the frame of the direction it samples at, so `PointingOperator` and `XSamplingOperator` return different values for a polarised map, on both the nearest-neighbour and the bilinear path (#204):
  - each pixel stores Q and U in its own meridian basis, which is not the basis of a sample that sits off the pixel centre; combining the two without the rotation leaks E into B, and the leakage grows towards the poles as $\cot\theta$
  - bilinear summed four different bases; nearest returned the pixel centre's basis unrotated
  - intensity-only maps are bit-identical, and so is which pixel each sample lands in
  - the transport is implied by sampling a map containing Q or U, and is not configurable
- **Breaking:** a nearest-neighbour sample of a polarised map that falls outside the map now contributes nothing, instead of reading and writing the last pixel; on a subset landscape it no longer reads the sink slot (#204)
- **Breaking:** `StokesLandscape.world2interp` returns a resolved stencil: a neighbour outside the map comes back as index 0 with weight 0, where it used to come back as index -1 with its raw weight, and the surviving weights are rescaled to sum to one (#204)
- **Breaking:** removed `StokesLandscape.pixel2interp` and `WCSLandscape.pixel2interp`; a landscape defines its interpolation by overriding `world2stencil` (#204)
- `PointingOperator` samples a stencil through the single `_quat2stencil` hook, replacing `_quat2interp`; a subclass that moves the pointing in `_quat2index` alone raises `NotImplementedError` when it samples a polarised map (#204)

## [0.12.0] - 2026-07-24

This version drops support for Python 3.11 following the latest JAX release (0.11.0).

### Added

- Introduce `XSamplingOperator` to support precomputed bilinear pointing (#181)
- Implement `LocalStokesLandscape` (#186)

### Changed

- Rename scan-block operators to `Stream*` and make the module public (`streaming.py`) (#173)
- Avoid unnecessary array copies in observation readers (#175)
- Migrate multi-observation mapmaker to furax CG and support verbose mode (#176)
- Split long API documentation pages (#179)
- Improve `MapMakingConfig` documentation and add examples (#179)
- Generalise `XSamplingOperator` to cache world angles, enabling precomputed bilinear pointing for HEALPix (#183)
- Allow the ML mapmaker to run with any weighting mode (identity/diagonal/Toeplitz), and restrict bilinear pointing to the ML solver (the direct binned solvers now raise on bilinear) (#187)
- Bumped minimum bounds for Python (3.12+), JAX (0.11+), Numpy (2.1+), and SciPy (1.15+) (#188)

### Fixed

- Revert change to `lax.map` in forward on-the-fly pointing (#180)
- Fix precomputed nearest pointing in multi-observation mapmaking (#182)

## [0.11.3] - 2026-07-14

### Changed

- Migrated the documentation website from Sphinx to Zensical (Material for MkDocs), with the
  API reference now generated by mkdocstrings

## [0.11.2] - 2026-07-14

### Added

- `BandedCholeskyOperator` and block-banded Cholesky routines (#167)
- More reader fields (#166)

### Changed

- Backed `Stokes` maps with a single dense array (#165)
- Optimised `PointingOperator` matvec using `lax.map` (#172)
- Streamed fused scan-block bodies (#170)
- Improved logging in the multi-observation mapmaker (#171)

## [0.11.1] - 2026-07-01

### Added

- Nested PCG capabilities in the multi-observation mapmaker (#162)
- Elevation modulation in the atmosphere pointing operator (#161)
- `WeightOperator` bundling noise weights and mask (#158)
- Spline-based HWP-synchronous signal template (#146)
- Basis-template operators for mapmaking (#144)
- `IDEMPOTENT` operator tag and `P @ P = P` reduction rule (#155)
- Sharding-aware CG solver (#83)
- Direct loading of sotodlib observations from the preproc db (#152)
- `cuda` dependency group to install `jax[cuda]` (#139)

### Changed

- Split binary rules into composition/addition flavours (#137)
- Rebased observation timestamps to a zero origin before the float32 downcast (#145)
- Supported static leaves and scalar fusion in scan-block operators (#143)
- Improved reader fields handling (#148)
- Structure compatibility now checks only shape and dtype (#153)

### Fixed

- CG instability past convergence with stabilisation enabled (#159)
- dtype of `MaskOperator.to_boolean_mask()` (#160)
- Scan-block fusion to keep obs-axis sharding (#147)
- `MapMakingConfig.double_precision=False` (non-functional path) (#118)

## [0.11.0] - 2026-06-05

Drops support for Python 3.10 (now requires JAX 0.10+) and includes breaking changes to the
mapmaking configuration.

### Added

- Distributed multi-observation mapmaking (#117, #132)
- Identity noise option to disable noise weighting (#120)
- Low-rank approximation using the Lanczos algorithm (#80, #122, #123)
- `atomic` flag on `QURotationOperator` to prevent reduction (#119)
- SHT beam convolution support (#114)

### Changed

- **Breaking:** Dropped support for Python 3.10; now requires JAX 0.10+ (#104, #112, #115)
- **Breaking:** Refactored the mapmaking noise weighting configuration (#126)
- Observation handling now returns host (NumPy) arrays and pads host-side (#128, #130)
- Moved the SO multi-observation CLI to the standalone `so_mapmaking` package (#133)
- Improved error messages for incompatible operator composition and comparison (#113, #116)

### Fixed

- Metadata hashing in gap-filling noise realizations (#131)
- Various mapmaking and ATOP/MLMapMaker hotfixes (#108, #109, #111)

## [0.10.4] - 2026-04-10

### Added

- Improved FFT and overlap-and-save methods in `SymmetricBandToeplitzOperator` (#100)

### Changed

- Cleaned up PSD fitting code by relying on CADRE (#101, #103)

### Fixed

- Computation of pixel indices in the expanded pointing operator (#102)
- POMME operator with no tail (#105, #106)

## [0.10.3] - 2026-04-01

### Added

- Bilinear interpolation (#95)
- Atmosphere separation infrastructure (#96)
- ATOP support in the multi-observation mapmaker (#93)

### Changed

- Reorganised `MapMakingConfig` (#92)
- Reduced memory usage in `MultiObsMapMaker` (#97)
- Improved `MapMakingResults` (#99)

## [0.10.2] - 2026-03-25

### Added

- `tree.norm` (#82)
- `WCSLandscape` with CAR projection (#89)
- `asoperator` factory (#32)
- SO-specific mapmaking configuration (#88)

### Changed

- Refactored `MapMakingResults` (#84) and multi-observation logic into a helper class (#87)
- Improved JAX compatibility of the FGBuster instrument (#85)
- Improved documentation (#75)
- Bumped pre-commit hooks and GitHub actions (#78)

### Fixed

- Stacking when creating `DetectorArray` (#81)

## [0.10.1] - 2026-03-06

### Changed

- Avoid a direct dependency on `litebird_sim` (#76)

## [0.10] - 2026-03-06

Major mapmaking and observation release.

### Added

- Multi-observation mapmaker (#40) with gap-filling (#54) and noise-weighting improvements (#53)
- Data reader for sotodlib-format observations (#29, #30, #33, #39) and multi-observation
  mapmaking script (#65)
- `MaskOperator`, supporting PyTree inputs (#31, #72)
- `FourierOperator` (#35) and noise weighting built on it (#49)
- `furax.linalg` module with analytic 2×2 and 3×3 `eigvalsh` (#73)
- Analytic likelihood gradient (#58)
- Preliminary LiteBIRD (`lbs`) interface (#57)
- Support for demodulated data in mapmaking (#68)
- `quat2index` for any `StokesLandscape` (#41)
- Citation metadata (#38, #52)

### Changed

- Dataclass-based abstract linear operators (#62)
- Mapmaking renaming (#27)
- Simplified `PointingOperator` (#45) and factored out QU rotation logic (#74)
- Made `fft_size` static in the Toeplitz operator (#46)
- Clarified angle conventions (#70)
- Bumped `sotodlib` for Python 3.13 support (#63); faster CI (#51); revisited ruff rules (#55)

### Fixed

- Traced arrays used in scanning mask (#34)
- FGBuster instrument PyTree (#61)
- Error if `solver_options` is passed directly to parametrize an inverse (#71)

## [0.9] - 2025-07-25

### Added

- Parametrizable `.I` (inverse) on operators (#19)
- ReadTheDocs documentation (#24)

### Changed

- Updated component separation (#16)
- More linting rules (#12); updated pre-commit hooks (#21)

### Fixed

- Tests involving inexact arrays on the CPU platform (#14)
- Block row operator with a single leaf (#13)
- `IndexOperator` and `ReshapeOperator` edge cases (#15, #23)

## [0.8] - 2025-02-05

First public release with CI and a release workflow.

### Added

- Continuous integration (#2) and release workflow
- Generic `IndexOperator` (#3)
- Gap filling (#4)
- SED operators for component separation (#1)
- `TreeOperator` and basic PyTree operations: matvec, vecmat, matmat (#9)

### Changed

- Reorganised the codebase
- Renamed `StokesPyTree` → `Stokes`, `StokesIQUPyTree` → `StokesIQU`, etc.
- Updated installation instructions (#10)

### Fixed

- Toeplitz operator fix (#8)

## [0.7] - 2024-10-30

### Added

- `FrequencyLandscape` and tests

## [0.6] - 2024-08-02

### Changed

- `SymmetricBandToeplitzOperator`: broadcastable and multidimensional band values

## [0.5] - 2024-07-02

### Added

- `ToastObservationMatrixOperator`

## [0.4] - 2024-05-31

### Added

- `.I` (inverse) for all operators

## [0.3] - 2024-05-06

### Added

- Multidimensional landscapes

## [0.2] - 2024-05-13

### Fixed

- Deployment

## [0.1] - 2024-02-12

Initial tagged release.

### Added

- Project classifiers and editable-mode installation instructions

[unreleased]: https://github.com/CMBSciPol/furax/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/CMBSciPol/furax/compare/v0.11.3...v0.12.0
[0.11.3]: https://github.com/CMBSciPol/furax/compare/v0.11.2...v0.11.3
[0.11.2]: https://github.com/CMBSciPol/furax/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/CMBSciPol/furax/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/CMBSciPol/furax/compare/v0.10.4...v0.11.0
[0.10.4]: https://github.com/CMBSciPol/furax/compare/v0.10.3...v0.10.4
[0.10.3]: https://github.com/CMBSciPol/furax/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/CMBSciPol/furax/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/CMBSciPol/furax/compare/v0.10...v0.10.1
[0.10]: https://github.com/CMBSciPol/furax/compare/v0.9...v0.10
[0.9]: https://github.com/CMBSciPol/furax/compare/v0.8...v0.9
[0.8]: https://github.com/CMBSciPol/furax/compare/v0.7...v0.8
[0.7]: https://github.com/CMBSciPol/furax/compare/v0.6...v0.7
[0.6]: https://github.com/CMBSciPol/furax/compare/v0.5...v0.6
[0.5]: https://github.com/CMBSciPol/furax/compare/v0.4...v0.5
[0.4]: https://github.com/CMBSciPol/furax/compare/v0.3...v0.4
[0.3]: https://github.com/CMBSciPol/furax/compare/v0.2...v0.3
[0.2]: https://github.com/CMBSciPol/furax/compare/v0.1...v0.2
[0.1]: https://github.com/CMBSciPol/furax/releases/tag/v0.1

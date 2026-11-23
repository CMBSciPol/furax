# Atmosphere screens in Fourier space: prototype findings

This note summarises a prototype that models atmospheric emission as frozen turbulent screens
represented by band-limited Fourier series, instead of the pixelised screens of
`furax.obs.atmosphere.ScreenSampler`. It covers the model, its furax implementation, how well it
estimates the wind and reconstructs the atmosphere on simulations, and the numerical problems met
in solving for the screen amplitudes.

All results come from small simulations (a few thousand samples, a few hundred unknowns per
layer), chosen so that every experiment runs in seconds to minutes on a laptop. The operator is
`furax.obs.atmosphere.FourierScreenOperator`; the experiments live in
`experiments/fourier_atmosphere/`, see [Reproducing the results](#reproducing-the-results).

## Summary

- **The model works and fits furax well.** A layer is a Fourier series on a periodic cell of the
  screen plane; drift with the wind is an exact phase shift. The new `FourierScreenOperator` reads
  it along the detector lines of sight, expanding the pointing from furax quaternions, and is
  applied with a non-uniform FFT (`jax-finufft`). It agrees with the existing pixel
  `ScreenSampler` to within the pixel interpolation error, its transpose is exact, and gradients
  with respect to the wind come from autodiff.
- **Wind estimation is accurate and robust.** With a von Kármán prior and the Gaussian marginal
  likelihood, the drifts of three simulated layers are recovered to about 1%, also when the
  simulated turbulence spectrum differs from the assumed one (slope and outer scale), provided the
  per-layer amplitude is fitted.
- **The prior is worth keeping for the reconstruction.** The Wiener (prior-on) reconstruction is
  4–7 times more accurate than a prior-free pixel reconstruction at the fitted winds, and is
  insensitive to the spectral mismatch tested.
- **Counting the layers is fragile.** A greedy search with a BIC stopping rule finds the right
  number of layers when the prior matches the simulation, but not when it does not: misspecified
  layers attract extra screens, and BIC never stops. Merging/pruning screens and stopping on
  held-out detectors are the obvious remedies (untested).
- **The amplitude solve is badly conditioned, and only partly fixable.** The prior itself is the
  best simple preconditioner for the joint multi-layer system. Two independent causes limit
  convergence: Fourier modes in the regions of the cell that no sample visits, and combinations of
  layers that cancel on the samples. A pixel-space preconditioner applied through an FFT fixes the
  first for a single layer (about 2× fewer iterations); nothing cheap fixes the second.

## The model

### Screen coordinates and drift

A horizontal layer at height $h$ is crossed by the line of sight of direction
$\hat{v} = (v_x, v_y, v_z)$ (horizon frame, $z$ = zenith) at the point
$h\,\mathbf{u}$, with

$$
\mathbf{u} = \left(\frac{v_x}{v_z}, \frac{v_y}{v_z}\right) = \cot(\mathrm{el})\,(\cos \mathrm{az}, \sin \mathrm{az}).
$$

This is exact for a flat layer: no flat-sky approximation is needed across the field of view.
Working in the dimensionless coordinate $\mathbf{u}$ (screen position divided by height), a layer
moving with wind $\mathbf{v}$ drifts at the angular rate $\boldsymbol{\omega} = \mathbf{v}/h$, and
detector $i$ at time $t$ reads the frozen screen at

$$
\mathbf{y}_i(t) = \mathbf{u}_i(t) + \boldsymbol{\omega}\, t ,
$$

the same drift convention as `ScreenSampler`.[^drift] The data only see $\boldsymbol{\omega}$: a layer at
3000 m moving at 9 m/s is indistinguishable from one at 1500 m moving at 4.5 m/s, except through
the angular size of its turbulence. Layers are therefore separated by their drift rates, and
heights are not estimated.

[^drift]: A pattern carried by a wind $\mathbf{v}$ is $m(\mathbf{x}, t) = m_0(\mathbf{x} - \mathbf{v} t)$,
    so a fixed line of sight reads the frozen screen at $\mathbf{u} - \boldsymbol{\omega} t$, not
    $\mathbf{u} + \boldsymbol{\omega} t$. With the sign used here, and by `ScreenSampler`, the
    pattern moves with velocity $-\boldsymbol{\omega}$: $\boldsymbol{\omega}$ is the velocity of
    the lines of sight relative to the screen, minus the wind over the height. The simulations and
    the fits share the convention, so the results are unaffected, but a fitted
    $\boldsymbol{\omega}$ is the opposite of the physical wind over the height.

### Field representation

Each layer is a real Fourier series on a periodic cell, the parallelogram
$\mathbf{y} = \mathbf{y}_0 + B\,\boldsymbol{\xi}$, $\boldsymbol{\xi} \in [0, 1)^2$, whose edges
are the columns of $B$ (a square box of side $L$ is $B = L\,I$):

$$
m(\mathbf{y}) = \sum_{\mathbf{n}} \Big[ c_{\mathbf{n}} \cos(2\pi\,\mathbf{n}\cdot\boldsymbol{\xi}) + s_{\mathbf{n}} \sin(2\pi\,\mathbf{n}\cdot\boldsymbol{\xi}) \Big],
\qquad \mathbf{q}_{\mathbf{n}} = 2\pi B^{-\mathsf{T}} \mathbf{n},
\qquad |\mathbf{q}_{\mathbf{n}}| \le q_{\max}.
$$

One $\mathbf{n}$ is kept per $\pm\mathbf{n}$ pair, and $\mathbf{n} = 0$ is excluded (a constant
is degenerate with detector offsets). The unknowns of a layer are the $2M$ real coefficients
$(c_{\mathbf{n}}, s_{\mathbf{n}})$. The only discretisation choices are the cell and the band
limit $q_{\max}$, both physical: the cell must contain every visited $\mathbf{y}$ plus a margin,
and $q_{\max}$ is set by the smallest angular scale worth modelling (here
$2\pi/q_{\max} = 0.1$ in units of the height).

### Forward model, prior and solution

With $P_k$ the operator reading layer $k$ at the sample positions, and $N$ the noise covariance,

$$
d = \sum_k P_k\, a_k + n .
$$

The prior is Gaussian and diagonal in this basis, with a von Kármán spectrum:

$$
\langle a_{k,\mathbf{n}}\, a_{k,\mathbf{n}'} \rangle = \delta_{\mathbf{n}\mathbf{n}'}\, S_k(\mathbf{q}_{\mathbf{n}}),
\qquad S_k(q) \propto \sigma_k^2\,\big(q^2 + q_{0,k}^2\big)^{-11/6},
$$

where $\sigma_k^2$ is the field variance and $q_{0,k} = 2\pi h_k / L_0$ the outer scale in screen
units (a physical outer scale $L_0$ looks smaller on the sky for a higher layer). The amplitudes
are the Wiener solution

$$
\big(P^\mathsf{T} N^{-1} P + S^{-1}\big)\, \hat{a} = P^\mathsf{T} N^{-1} d ,
$$

with $P = [P_1, \dots, P_K]$ and $S = \mathrm{diag}(S_1, \dots, S_K)$, and the atmosphere template
at the samples is $P\hat{a}$.

### Wind estimation

The drifts and the spectral parameters $\theta = \{\boldsymbol{\omega}_k, \sigma_k, q_{0,k}\}$ are
fitted on the Gaussian marginal likelihood, with $C = P S P^\mathsf{T} + N$:

$$
-2 \ln \mathcal{L}(\theta) = d^\mathsf{T} C^{-1} d + \ln \det C + \text{const}.
$$

By the Woodbury identity this is evaluated in amplitude space,

$$
-2 \ln \mathcal{L} = d^\mathsf{T} N^{-1} d - b^\mathsf{T} G^{-1} b + \ln \det G,
\qquad G = I + S^{1/2} P^\mathsf{T} N^{-1} P S^{1/2},\quad b = S^{1/2} P^\mathsf{T} N^{-1} d,
$$

which the prototype does with a dense Cholesky factorisation (small problems only). Gradients
with respect to $\boldsymbol{\omega}$ come from autodiff through the operator, and the fit uses
BFGS. The $\ln \det$ term matters: it penalises the extra freedom a model gains when a fast drift
spreads the samples over more of the cell. A plain "$\chi^2$ explained" criterion is biased
towards large $|\boldsymbol{\omega}|$ for that reason.

## Implementation in furax

`furax.obs.atmosphere` holds the operator and its helpers.

- **`FourierScreenOperator`** (an `AbstractLinearOperator`): coefficients of shape $(2, M)$ to
  TOD of shape (detectors, samples).
  - It stores a furax `QuaternionSampler` and expands the pointing per detector on the fly, like
    `PointingOperator`. Samples with a zero quaternion (padding) read zero.
  - Backends: `'nufft'` (default) evaluates the series with a type-2 NUFFT
    (`jax_finufft.nufft2`); `'dense'` sums it directly, for checks and small problems.
  - Only `mv` is defined: furax derives the transpose, and JAX the derivatives with respect to the
    drift and the pointing.
  - `as_matrix()` returns the dense design matrix $[\cos \mid \sin]$ in closed form, for direct
    solves on small problems.
- **`cell_index(cell, q_max)`**: the integer wavevectors of a cell within the band limit.
- **`fit_cell(u, times, omega, margin)`**: the parallelogram a layer's samples sweep (one edge along
  the scan strip, the other along the drift), grown by `margin`.

The operator adds `jax-finufft` as a dependency of furax.

### Validation

| check | result |
|---|---|
| exact transpose (adjoint test) | relative error $< 10^{-15}$ |
| NUFFT vs dense sum: forward / transpose / $\partial/\partial\boldsymbol{\omega}$ | $4.9 \times 10^{-10}$ / $3.4 \times 10^{-10}$ / $4.5 \times 10^{-10}$ (requested accuracy $10^{-9}$) |
| same smooth field read by `ScreenSampler` (bilinear) at pixel size 40 m / 10 m (layer at 1 km) | rms difference 11% / 0.8% of the signal, decreasing as the pixel interpolation error |

Cost of one application (ms, CPU, float64):

| samples | modes | dense $P$ | NUFFT $P$ | dense $P^\mathsf{T}$ | NUFFT $P^\mathsf{T}$ |
|---|---|---|---|---|---|
| 4 000 | 158 | 6.8 | 2.3 | 14.6 | 2.4 |
| 32 000 | 980 | 300 | 5.8 | 688 | 3.8 |
| 128 000 | 52 592 | – | 10.8 | – | 12.8 |
| 1 280 000 | 1 063 916 | – | 294 | – | 331 |

The dense sum costs $O(N_\mathrm{samp} M)$ in time and memory; the NUFFT
$O(N_\mathrm{samp} + M \log M)$. The NUFFT is needed as soon as the problem is solved iteratively
at realistic sizes. Note that the number of modes grows with the area swept by the drift, i.e.
with the observation length: long observations are best cut into chunks, which frozen flow also
requires.

## Simulations

Unless stated otherwise:

- constant-elevation scan at 50°, azimuth $\pm 10°$ sinusoidal with a 20 s period, for 100 s;
- 8 detectors uniformly within $\pm 2°$ of the boresight, 240 samples each (1920 samples);
  "dense" runs use 16 detectors × 480 samples (7680 samples);
- three frozen layers, each a von Kármán screen with $L_0 = 300$ m:

| layer | height | wind [m/s] | drift $\boldsymbol{\omega}$ [$10^{-3}$/s] | rms |
|---|---|---|---|---|
| 0 | 500 m | (2, 1) | (4, 2) | 1.0 |
| 1 | 1500 m | (−4.5, 4.5) | (−3, 3) | 0.6 |
| 2 | 3000 m | (9, −9) | (3, −3) | 0.4 |

- white noise of rms 0.1, band limit $2\pi/q_{\max} = 0.1$ (50 m for the lowest layer, 300 m for
  the highest).

No beam is modelled: the samples read the field along the line of sight. For a small-aperture
telescope the beam footprint on a layer (roughly the larger of the aperture, about 0.5 m, and the
far-field width times the height, a few to a few tens of metres) is well below this band limit.
If finer scales are modelled, the beam enters as a diagonal factor $B_k(\mathbf{q})$ on each
layer's coefficients.

## Wind estimation

### One layer: Fourier vs pixels

On a single-layer simulation (1 km, wind 3 m/s), a scan of the profile likelihood in $v_x$
(0.4 m/s steps) puts the Fourier minimum at the true 3.0 m/s and the minimum of the existing pixel
`profile_neg_log_likelihood` (pixel size 40 m) at 3.4 m/s. This is suggestive only: the data were
generated by the Fourier model, the pixel likelihood has no prior, and one realisation was used.
A fair comparison has not been done.

### Several layers: greedy search

The number of layers and their drifts are found greedily:

1. **Scan** $\boldsymbol{\omega}$ on a grid ($13 \times 13$ over $\pm 6 \times 10^{-3}$/s):
   for each candidate, the gain in marginal likelihood of explaining the current residual with one
   more screen at that drift.
2. **Add** a screen at the best candidate, then **refit all screens jointly**
   (drift, amplitude, outer scale per layer) on the marginal likelihood of the full data. The
   joint refit corrects earlier mistakes: a screen placed at a wrong drift moves to a true one as
   others are added.
3. **Stop** when the improvement no longer pays for the screen's parameters (BIC:
   $\Delta(-2\ln\mathcal{L}) < 4 \ln N_\mathrm{samp}$).

With the prior matching the simulation, the search stops at 3 layers with drifts within
$0.1 \times 10^{-3}$/s of the truth and rms 1.02, 0.64, 0.36 (true 1.0, 0.6, 0.4). Two details
were necessary:

- the marginal likelihood, not the $\chi^2$ explained, as scan statistic (see above);
- a per-layer outer scale. With one outer scale shared by all layers, the search found a spurious
  fourth screen next to the lowest layer, whose large eddies the shared prior could not
  represent.

## Solving for the amplitudes

The Wiener system is solved with `furax.linalg.cg` (its implicit differentiation also gives exact
gradients through the solve). Its conditioning turned out to be the main numerical difficulty.

### Why it is badly conditioned

In prior-whitened variables the system is $I + H$, with
$H = S^{1/2} P^\mathsf{T} N^{-1} P S^{1/2}$. Its eigenvalues split into:

- modes the data barely see (small scales, and anything describing unvisited parts of the cell),
  with eigenvalue 1;
- the data-dominated modes, whose eigenvalues spread from 1 to $1 + \max_q S(q)\, w$, the
  signal-to-noise of the largest scales: 88 of 376 modes, up to $4.9 \times 10^4$, for one layer
  of the simulation.

The data term is not diagonal in the Fourier basis: on a cell,

$$
\big(P^\mathsf{T} W P\big)_{\mathbf{n}\mathbf{n}'} = \sum_{\text{samples}} w\, e^{-2\pi i (\mathbf{n} - \mathbf{n}')\cdot\boldsymbol{\xi}} = \hat{w}(\mathbf{n} - \mathbf{n}'),
$$

a convolution by the Fourier transform of the hit density. It is diagonal ($\propto I$) only if
the samples fill the cell evenly. A scan strip dragged by the wind covers about 20% of a square box
around it.

### Preconditioners tried

Iterations to reach the stated tolerance, one layer (layer 0) and the three layers jointly:

| preconditioner | tolerance | 1 layer | 3 layers |
|---|---|---|---|
| none | $10^{-8}$ | 1070 | 2024 |
| $\big(\bar{w} + S^{-1}\big)^{-1}$, assuming uniform coverage | $10^{-8}$ | 1083 | 2013 |
| $S$ (the prior) | $10^{-8}$ | 366 | 556 |
| $S$, square box | $10^{-5}$ | 219 | 270 |
| $S$, cell fitted to the coverage (margin 0.1) | $10^{-5}$ | 220 | 259 |
| pixel Jacobi through an FFT, fitted cell | $10^{-5}$ | **113** | 387 |
| $S$ + deflation of the 40 largest eigenvalues, fitted cell | $10^{-5}$ | – | **91** |

**The prior $S$ is the default:** it clusters every prior-dominated mode at 1 whatever the
coverage, costs nothing, and is the best simple choice for the joint system. The uniform-coverage
preconditioner is no better than none, because the coverage is far from uniform. A block-diagonal
preconditioner with an exact solve on the lowest modes was also worse than $S$ alone (it drops the
strong coupling partial coverage creates between low and high modes).

### Fitting the cell to the coverage

Using as cell the parallelogram the samples sweep, instead of a square box, cuts the number of
modes by 3 (210 to about 67 per layer) and raises the fraction of the cell that is hit from 17%
to about 50%, but does not reduce the number of iterations. The margin around the swept region,
required so that the periodic field does not tie opposite edges of the data together, is
unvisited, and its modes see no data. Without a margin, the single-layer system becomes well
conditioned with the uniform-coverage preconditioner:

| margin | modes (layer 0) | cell hit | $\kappa$, preconditioner $S$ | $\kappa$, $\big(\bar{w} + S^{-1}\big)^{-1}$ | iterations with $\big(\bar{w} + S^{-1}\big)^{-1}$, tolerance $10^{-5}$ |
|---|---|---|---|---|---|
| 0.1 | 66 | 56% | $2.9 \times 10^4$ | $2.7 \times 10^4$ | 218 |
| 0.03 | 39 | 100% | $8.2 \times 10^3$ | $9.2 \times 10^2$ | – |
| 0 | 27 | 100% | $3.5 \times 10^2$ | **18** | **20** |

but the margin is needed for accuracy (three layers jointly, three noise realisations):

| basis | modes | iterations, $S$ / uniform | template error | drift error per layer [$10^{-3}$/s] |
|---|---|---|---|---|
| square box (the simulation's basis) | 630 | 272 / 634 | 0.029 | – |
| fitted cell, margin 0 | 86 | 250 / 234 | 0.095 | 0.51, 0.68, 0.63 |
| fitted cell, margin 0.03 | 115 | 253 / 320 | 0.038 | 0.19, 0.17, 0.40 |
| fitted cell, margin 0.1 | 202 | 265 / 447 | 0.031 | 0.10, 0.10, 0.12 |

The template error is $\|P\hat{a} - s\| / \|s\|$ at the samples, $s$ the true atmosphere signal.
A margin of 0.1 (in units of the height) matches the accuracy of the full square box with 3 times
fewer modes; that, not faster convergence, is the benefit of fitting the cell.

### A pixel preconditioner applied through an FFT

On a regular grid of $(2N_1 + 1) \times (2N_2 + 1)$ points over the cell, the map $Q$ from the
coefficients to the grid values is one inverse FFT, orthonormal after scaling
($Q^\mathsf{T} Q = I$). An operator diagonal in pixels then acts on the coefficients as
$Q^\mathsf{T} D Q$, one FFT pair. The pixel Jacobi preconditioner

$$
M = Q^\mathsf{T}\, \mathrm{diag}\big(Q A Q^\mathsf{T}\big)^{-1} Q
$$

sees the coverage: unvisited pixels are weighted by the prior, visited ones by the data. For one
layer it halves the iterations (212 to 113; $\kappa$ from $2.9 \times 10^4$ to
$2.4 \times 10^3$). Here the pixel diagonal was computed exactly from dense matrices; at scale it
would come from the hit counts binned on the grid (not yet written).

For three layers it is worse than $S$ (387 against 264 iterations), and reducing each layer's data
weight (equally, $1/K$, or by its share of the field variance) does not help.

### Layers that cancel on the samples

The reason is visible in the eigenvectors of the preconditioned joint system. For each, the ratio
$\big\|\sum_k P_k v_k\big\|^2 / \sum_k \|P_k v_k\|^2$ measures whether the layers act
independently (1) or cancel on the samples (0):

| directions | eigenvalue of $MA$ | ratio |
|---|---|---|
| 5 smallest | from $3 \times 10^{-4}$ | 0.000 |
| 5 largest | up to 29 | 2.3–2.9 |
| median | 0.13 | 0.075 |

At the bottom of the spectrum, the layers cancel exactly on the samples: only the prior constrains
these combinations. At the top, all layers read the same pattern along the same lines of sight, and
the data constrain it about $K$ times more than each layer alone. A preconditioner acting layer by
layer cannot distinguish these cases, since the difference only exists in how the layers combine.

Deflating the extreme eigenvectors (with the existing `make_two_level_preconditioner`) helps
steadily but without a sharp threshold:

| first level | deflated | $\kappa$ | iterations |
|---|---|---|---|
| pixel Jacobi | none / 5 / 10 / 20 / 40 smallest | $9.2 \times 10^4$ … $2.4 \times 10^3$ | 388 / 258 / 223 / 173 / 118 |
| $S$ | none / 5 / 10 / 20 / 40 largest | $3.4 \times 10^4$ … $1.1 \times 10^3$ | 263 / 199 / 170 / 136 / 91 |

The badly scaled directions form a continuous tail: about 10% of the unknowns had to be deflated
for a threefold gain, and computing the eigenvectors by Lanczos costs about as much as a solve.
Deflation pays only when the same eigenvectors serve many solves (a wind fit, stochastic
estimates of $\ln\det$), and its cost likely grows with the problem. The cancellation is a genuine
degeneracy of the data, resolved only by the prior; no cheap fix was found.

## Is the prior needed for the reconstruction?

An alternative is to estimate the layers and winds with the Fourier model, then reconstruct the
screens by prior-free least squares on pixel screens (`ScreenSampler`, bilinear) at the fitted
winds. Pixels suit a prior-free solve: unvisited pixels are dropped and $1/\text{hits}$ is a
near-exact preconditioner.

On the default (sparse) simulation, the prior-free pixel template is about twice worse than the
Wiener one at best (template error 0.062–0.067 against 0.031), and is sensitive to the pixel size:
coarse pixels cannot represent the signal ($\chi^2/N \approx 3$–4), fine ones fit the noise
($\chi^2/N$ down to 0.33). With three layers and fine pixels the solve does not converge, because
of the same cancelling combinations.

### Robustness to a wrong spectrum

The decisive test simulates the layers with a spectrum different from the assumed one (slope $-3$
instead of $-11/3$, outer scale 100 m instead of 300 m), on the dense simulation, and fits with the
standard von Kármán prior. Drift errors are the mean over layers in $10^{-3}$/s, on drifts of
3–4.5:

| simulated spectrum | layers | method | drift error | template error | $\chi^2/N$ |
|---|---|---|---|---|---|
| matched | 1 | Wiener, amplitude fitted | 0.02 | 0.013 | 0.99 |
| matched | 1 | Wiener, amplitude and outer scale fitted | 0.02 | 0.013 | 0.99 |
| matched | 1 | pixels, no prior | – | 0.055 | 1.21 |
| matched | 3 | Wiener, amplitude fitted | 0.04 | 0.016 | 0.98 |
| matched | 3 | Wiener, amplitude and outer scale fitted | 0.04 | 0.016 | 0.98 |
| matched | 3 | pixels, no prior | – | 0.062 | 1.41 |
| mismatched | 1 | Wiener, amplitude fitted | 0.02 | 0.015 | 1.00 |
| mismatched | 1 | Wiener, amplitude and outer scale fitted | 0.02 | 0.015 | 1.00 |
| mismatched | 1 | pixels, no prior | – | 0.107 | 2.11 |
| mismatched | 3 | Wiener, amplitude fitted | 0.04 | 0.017 | 0.98 |
| mismatched | 3 | Wiener, amplitude and outer scale fitted | 0.04 | 0.017 | 0.98 |
| mismatched | 3 | pixels, no prior | – | 0.094 | 2.19 |

- The drifts and the Wiener template barely notice the mismatch once the amplitude is fitted.
- With the outer scale held at a wrong value, the fitted amplitude absorbs the difference; with it
  free, the lowest layer's outer scale is recovered (95–102 m for a true 100 m), while the
  highest layer's is unconstrained, its eddies being near the band limit. The spectral parameters
  behave as nuisances.
- The prior-free pixel reconstruction is 4–7 times worse, more so under the mismatch, whose extra
  small-scale power pixels of 0.04 cannot resolve.

**Conclusion: keep the prior, fit each layer's amplitude, and treat the outer scale as optional.**

### Counting layers under a wrong spectrum

The greedy search run on the mismatched simulation (sparse sampling) recovers the three true
drifts within $0.1 \times 10^{-3}$/s after five steps, but:

- the first steps go astray (drifts at the edge of the search range, absurd amplitudes) until
  enough screens are present for the joint refit to settle;
- it ends with two extra screens: a duplicate next to the lowest layer, sharing its signal, and a
  screen of zero amplitude;
- the BIC rule never stops the search (gain 593 at the fifth step, against a cost of 30): when the
  spectrum is misspecified, each extra screen keeps absorbing some misfit.

The winds and template are robust to the mismatch; the layer count is not.

## Open problems and next steps

- **Robust layer counting.** Merge screens whose drifts differ by less than the scan resolution and
  drop screens of negligible amplitude (enough here to get 3); stop on the prediction of held-out
  detectors rather than BIC; possibly fit a slope per layer.
- **Cross-layer cancellation in the joint solve.** No cheap preconditioner found. Options are
  deflation reused across many solves, or a coarse space built for the cancelling combinations.
- **Scaling up.** Replace the dense Cholesky of the likelihood by CG for the data term and
  stochastic Lanczos quadrature for $\ln\det$; build the pixel preconditioner from binned hit
  counts; chunk the NUFFT over detectors; fit observations in chunks short enough for frozen flow.
- **Realism.** Continuous vertical structure with wind shear (a ridge of drifts rather than
  discrete layers), evolving turbulence, $1/f$ noise, a common mode, and ground pickup, which
  looks like a screen with $\boldsymbol{\omega} = 0$ and should be modelled from the start.
- **A fair pixel vs Fourier comparison of the wind estimate,** with data independent of both models
  and several noise realisations.

## Caveats

- Small simulations: 1920 or 7680 samples, a few hundred unknowns per layer, one scan geometry.
- One to three noise realisations per configuration.
- Wind fits started at the true drifts, except in the greedy searches.
- Frozen, Gaussian turbulence; white noise; no beam.
- The pixel diagonal of the FFT preconditioner and the deflation eigenvectors were computed
  exactly from dense matrices.

## Reproducing the results

Run from the repository root of this branch; `--extra mapmaking` provides matplotlib. Each script
runs in seconds to a few minutes.

| script | content |
|---|---|
| `screen_operator.py` | helpers shared by the scripts (square-box modes, von Kármán spectrum, screen coordinates) |
| `fourier_screen.py` | one layer: transpose, agreement with `ScreenSampler`, wind scan vs pixels |
| `nufft_check.py` | NUFFT vs dense: accuracy and timings |
| `multilayer.py` | greedy layer search; `SIM_SPECTRUM=mismatched` for the wrong-spectrum run |
| `conditioning.py` | square box vs fitted cell, preconditioners $S$ and uniform-coverage; `TOY_SCALE=2` for dense sampling |
| `margin.py` | margin of the fitted cell: iterations, template and drift errors |
| `pixel_preconditioner.py` | pixel Jacobi through an FFT, with shared data weights |
| `deflation.py` | two-level deflation on top of $S$ and pixel Jacobi |
| `pixel_reconstruction.py` | prior-free pixel reconstruction at the Fourier-fitted winds |
| `mismatch.py` | wrong-spectrum test of the winds and the Wiener template |

```sh
uv run --extra mapmaking python experiments/fourier_atmosphere/mismatch.py
SIM_SPECTRUM=mismatched uv run --extra mapmaking python experiments/fourier_atmosphere/multilayer.py
```

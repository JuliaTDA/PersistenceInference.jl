# Functional confidence bands

```@meta
CurrentModule = PersistenceInference
```

## From bars to functions

An interval `(b, d)` defines a tent of height
``max(0, \min(t-b, d-t))``. The first landscape λ₁ is the highest tent at each
`t`; λ₂ is the second highest, and so on. The Betti curve counts active bars,
using `birth ≤ t < death`. Landscapes encode persistence as smooth tent heights;
Betti curves answer the more direct question, “how many features are alive?”

For a sample of independent diagrams, the target is the population mean of one
of these functions. Averaging landscapes does not reconstruct an average cloud
or generally produce the landscape of a particular diagram. The sample size is
the number of independent diagrams, usually one per subject, experiment, or
simulation. Bars within a diagram and bootstrap replicas of one cloud are not
independent observations for this workflow.

## A mean landscape and a mean Betti curve

```@example functional
using PersistenceInference, TDAPersistenceDiagrams, Random
rng = Xoshiro(20261003)
diagrams = [PersistenceDiagram([(0.0, 1 + rand(rng))]; dim=1) for _ in 1:80]
grid = range(0, 2; length=101)
landscape = confidence_band(diagrams; grid=grid, n_boot=199, rng=rng)
betti = confidence_band(diagrams; grid=grid, representation=:betti,
                        n_boot=199, rng=rng)
@assert landscape.n_samples == betti.n_samples == 80
(landscape=landscape, betti=betti)
```

The default `representation=:landscape, k=1` selects λ₁. Set `k=2` for λ₂.
All diagrams must have the same homology-dimension metadata. Landscapes ignore
essential intervals; Betti curves count them for every time after birth.
The conversion evaluates each function **exactly at each supplied time** rather
than averaging over time bins.

The bands simultaneously cover the **population mean at the supplied grid**
asymptotically under i.i.d. sampling and finite second moments, with appropriate
nondegeneracy for stochastic coordinates. They are not prediction intervals,
bands for one subject, or guarantees between sampled times. Betti curves can
jump; landscape continuum theory does not automatically give continuum Betti
coverage. Choose the grid and representation before inspecting results.
The functional landscape theory is developed in
[Chazal et al. (2015)](https://doi.org/10.20382/jocg.v6i2a8).

## How the calibration works

Let `n` be the number of curves and `f̄` their pointwise mean. The default
Gaussian multiplier method generates

```math
G_b(t_j) = n^{-1/2}\sum_{i=1}^n \xi_{bi}(f_i(t_j)-\bar f(t_j)),
\qquad \xi_{bi}\sim N(0,1).
```

It records `maximum(abs.(G_b))` for each replicate. If their `(1-alpha)`
quantile is `c`, the unstudentized band has half-width `c/sqrt(n)` at stochastic
coordinates. Because one supremum controls the entire grid, this differs from
constructing a separate unadjusted confidence interval at each time.

`method=:bootstrap` resamples **whole rows/curves with replacement**, computes
`sqrt(n) * (bootstrap_mean - sample_mean)`, and calibrates its supremum instead.
[`multiplier_bootstrap`](@ref) is a convenience entry point for the multiplier
method.

`studentize=true` divides process coordinates by sample standard deviations and
multiplies them back when forming the band. This permits varying widths across
times. Coordinates constant in the observed sample receive width zero in both
methods. A coordinate observed as constant need not be constant in the population:
rare events can be missed in a finite sample. Inspect this issue when curves are
sparse or the sample is small. Band endpoints are not clipped to zero even for
nonnegative descriptors.

## Matrix input for other functional summaries

Rows are independent observations; columns are fixed grid coordinates. Supply a
finite matrix and a finite, strictly increasing grid with one value per column.
At least two observations are required.

```@example matrix_band
using PersistenceInference, Random
rng = Xoshiro(91)
grid = collect(range(0, 1; length=11))
samples = [sin(π*t) + 0.15randn(rng) for _ in 1:50, t in grid]
band = multiplier_bootstrap(samples; grid=grid, n_boot=199,
                            studentize=true, rng=rng)
resampled = confidence_band(samples; grid=grid, method=:bootstrap,
                            n_boot=199, rng=rng)
@assert band.studentized && resampled.method == :bootstrap
(band=band, resampled=resampled)
```

The matrix overload accepts descriptors beyond diagrams, provided their sampling
assumptions are appropriate. It does not infer subject identities or correct
correlation between rows. For longitudinal or clustered data, these unrestricted
row resamplers do not implement block or cluster inference.

## Read a `ConfidenceBand`

| Field | Interpretation |
| --- | --- |
| `grid` | Evaluation times, stored as a `Vector{Float64}` |
| `estimate` | Pointwise sample mean |
| `lower`, `upper` | Simultaneous band endpoints on that grid |
| `maxima` | Replicate supremum statistics on the √n scale |
| `critical_value` | `(1-alpha)` quantile of `maxima`, before dividing by √n |
| `alpha`, `n_samples` | Requested level and number of independent rows/diagrams |
| `method`, `studentized` | Calibration method and scaling choice |

To display uncertainty with a plotting library, use `band.grid` as the x-axis,
`band.estimate` as the line, and `band.lower`/`band.upper` as the ribbon edges.
Do not label `critical_value` as the band half-width; the sample-size factor and
studentization scales must first be applied, as they already are in the endpoints.

See [Practical choices and validation](practical.md) for the existing numerical
and coverage studies, including observed finite-sample undercoverage.

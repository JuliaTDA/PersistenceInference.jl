# Point-cloud bootstrap and two-sample inference

```@meta
CurrentModule = PersistenceInference
```

## Resampling one cloud

Suppose points are an independent sample from a distribution supported near a
circle. Repeated samples need not contain the same angles, so even a prominent
loop can have changing birth and death scales. [`bootstrap_diagram`](@ref)
approximates that variation by drawing point indices **with replacement** and
recomputing persistent homology on each resampled cloud.

```@example point_bootstrap
using PersistenceInference, TDAPersistenceDiagrams, Random
rng = Xoshiro(42)
angles = 2π .* rand(rng, 32)
X = [(cos(t) + 0.03randn(rng), sin(t) + 0.03randn(rng)) for t in angles]
result = bootstrap_diagram(X; n_boot=49, alpha=0.05, dim_max=1, rng=rng)
@assert length(result.diagrams) == 2
@assert all(length(ds) == 49 for ds in result.distances)
(h1_radius=result.radius[2], retained_h1=length(significant(result, 1)))
```

The input is a vector of points accepted by `TDARipserer.ripserer`, such as
tuples or `SVector`s. Homology dimensions are zero based; Julia array positions
are one based. Thus `result.diagrams[2]`, `result.radius[2]`, and
`result.distances[2]` all concern H₁.

| Result field | Meaning |
| --- | --- |
| `diagrams[d+1]` | Diagram computed on all original points in dimension `d` |
| `distances[d+1]` | One reference-to-resample distance per bootstrap replicate |
| `radius[d+1]` | Empirical `(1-alpha)` quantile of those distances |
| `alpha`, `n_boot` | Requested level and replicate count |

### Why twice the radius?

For the bottleneck distance, a finite bar `(b, d)` has L∞ distance `(d-b)/2`
to the diagonal. If a *validly calibrated* confidence ball has radius `c`, a
bar longer than `2c` cannot be matched to the diagonal inside that ball. This
is the geometric reason for the threshold used by [`significant`](@ref).
The [confidence-set framework of Fasy et al.](https://doi.org/10.1214/14-AOS1252)
motivates this interpretation; coverage still depends on the estimator,
population target, sampling assumptions, and resampling calibration.

The current function takes an unscaled empirical distance quantile. It does
not estimate a support/density model or verify the conditions of a coverage
theorem. Stability connects perturbations to diagram distances; it does not
establish the bootstrap's statistical coverage. Treat the output as resampling
uncertainty unless your application justifies that calibration. Radii are
computed separately in each homology dimension and do not adjust for testing
many dimensions, clouds, or parameter settings.

Essential intervals have infinite death and are **always retained** by
`significant(result, dim)`. That is a convention of this helper, not statistical
evidence for essential features. A filtration threshold can leave classes
unresolved; inspect that threshold before interpreting an infinite bar.

### Changing the resampling or metric

`subsample_size=m` draws `m` points **with replacement**, including when `m<n`.
It is not sampling without replacement, and the implementation does not apply
an `m/n` normalization. Smaller resamples may save computation, but their raw
radius has a different calibration and does not automatically inherit an
ordinary bootstrap interpretation.

`distance=Bottleneck()` is the default. To compute a Wasserstein-distance
quantile, import and pass `TDAPersistenceDiagrams.Wasserstein(q)` explicitly.
The same `2c` selection helper is used; any confidence interpretation requires
a valid confidence radius for the chosen metric and its diagonal geometry.
For the conventional Wasserstein metric choose `q ≥ 1`.

Options such as `threshold` and `modulus` are forwarded to `ripserer` for both
reference and resampled clouds. Keep them fixed when comparing results, and
record them with the RNG seed and replicate count.

## Relabelling two independent point samples

[`permutation_test`](@ref) pools all points from `X` and `Y`, randomly relabels
them into groups of the **original sizes**, and recomputes two diagrams each
time. Its statistic is the diagram Wasserstein loss ``W_q^q`` in one homology
dimension. A larger statistic means more separated diagrams for this descriptor.

```@example two_clouds
using PersistenceInference, Random
rng = Xoshiro(7)
circle = [(cos(t), sin(t)) for t in 2π .* rand(rng, 20)]
radii = sqrt.(rand(rng, 20))
angles = 2π .* rand(rng, 20)
disk = [(r*cos(t), r*sin(t)) for (r, t) in zip(radii, angles)]
test = permutation_test(circle, disk; dim=1, q=2, n_perm=39, rng=rng)
@assert length(test.null_distribution) == 39
@assert test.pvalue == (1 + count(>=(test.observed), test.null_distribution)) / 40
(statistic=test.observed, pvalue=test.pvalue)
```

These tiny clouds illustrate the calculation, not a guaranteed rejection.
`observed` contains the original statistic, `null_distribution` contains the
permuted statistics, and `dim` records the tested dimension. `dim_max` defaults
to `dim` and must be at least as large as `dim`.

### What null is actually tested?

The point-label null requires that the pooled points are exchangeable between
groups, for example independent observations from the **same distribution**.
Equality of underlying topology alone does not guarantee this: two equally
connected supports with different sampling densities need not be exchangeable.
A small p-value is evidence against the exchangeable distributional null
through a topological summary; it does not uniquely identify a changed number
of holes. A large p-value is not proof that the populations have equal topology.

This implementation compares one cloud per group through its diagram distance.
It is motivated by [Robinson and Turner's topological testing
framework](https://doi.org/10.1007/s41468-017-0008-7), and should not be described
as every multi-diagram joint-loss procedure from that paper. Repeated-measures,
paired subjects, clusters, and time series do not fit unrestricted point-label
permutations without additional design-specific justification.

For `B=n_perm`, the Monte Carlo p-value is
``(1 + \#\{T_b \ge T_{obs}\})/(B+1)``. It is never zero and has minimum
`1/(B+1)`. Under a valid exchangeable null this correction gives a conservative
random-permutation test; the function samples permutations rather than
enumerating an exact full permutation distribution. It does two PH computations
per replicate. Use `progress=true` for sparse progress messages.

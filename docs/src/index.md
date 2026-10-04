# PersistenceInference.jl

```@meta
CurrentModule = PersistenceInference
```

A persistence diagram records *when* a topological feature appears and disappears
as a scale changes. For a bar `(b, d)`, its persistence is `d - b`: connected
components are in H₀, loops in H₁, and cavities in H₂. A bar that lives a long
time is often interesting, but a statistical question needs more than its length.
What is random? What population or null model are you studying?

`PersistenceInference.jl` supplies five complementary workflows. Start by
identifying the unit that was sampled independently.

| Data and target | Workflow | Read next |
| --- | --- | --- |
| One point cloud; resampling variability of its diagram | [`bootstrap_diagram`](@ref), [`significant`](@ref) | [Point-cloud inference](point_clouds.md) |
| Two independent point samples; an exchangeable-label null | [`permutation_test`](@ref) | [Point-cloud inference](point_clouds.md) |
| Many independent diagrams; a population mean function | [`confidence_band`](@ref), [`multiplier_bootstrap`](@ref) | [Functional bands](functional_inference.md) |
| One diagram/cloud; maximum finite persistence under an explicit null | [`max_persistence_significance`](@ref) | [Hypothesis tests](hypothesis_tests.md) |
| Paired diagrams and covariates; independence | [`independence_test`](@ref) | [Hypothesis tests](hypothesis_tests.md) |

For example, 40 subjects with one scan each give 40 independent diagrams if the
subjects are independently sampled. The 200 bars in a single subject's diagram
are not 200 independent subjects. Bootstrap replicas of that scan are also not
new subjects. This distinction determines which API is appropriate.

## Install the package and its fork dependencies

Use Julia ≥1.9 and a dedicated environment:

```julia
using Pkg
Pkg.activate("tda-inference")
Pkg.develop([
    PackageSpec(url="https://github.com/JuliaTDA/PersistenceDiagrams.jl"),
    PackageSpec(url="https://github.com/JuliaTDA/Ripserer.jl"),
    PackageSpec(url="https://github.com/JuliaTDA/PersistenceInference.jl"),
])
```

The first two repositories provide the modules `TDAPersistenceDiagrams` and
`TDARipserer`; their names and UUIDs differ from the upstream registered packages.
For local checkouts in a shared workspace, use `PackageSpec(path="...")` instead.
Loading `PersistenceInference` loads its dependencies, but names such as
`PersistenceDiagram`, `Bottleneck`, and `Wasserstein` belong to
`TDAPersistenceDiagrams`, so import them explicitly in your own code.

## A complete first example

Here the data are independent diagrams, each with one randomly sized loop.
This controlled toy experiment makes it easy to see the target: a mean
landscape function across diagrams, rather than a diagram of an averaged cloud.

```@example first_band
using PersistenceInference, TDAPersistenceDiagrams, Random
rng = Xoshiro(42)
diagrams = [PersistenceDiagram([(0.0, 1.0 + rand(rng))]; dim=1) for _ in 1:40]
band = confidence_band(diagrams; grid=range(0, 2; length=41),
                       n_boot=199, rng=rng)
@assert band.n_samples == 40
@assert all(band.lower .<= band.estimate .<= band.upper)
(samples=band.n_samples, grid_points=length(band.grid),
 mean_at_one=band.estimate[21], interval_at_one=(band.lower[21], band.upper[21]))
```

The endpoints are a simultaneous band for the **population mean at the fixed
supplied times**, under the assumptions described in [Functional confidence
bands](functional_inference.md). They are not intervals for an individual
subject or guarantees at times between grid points.

## A sensible reading order

Read the tutorial matching your design, then [Practical choices and
validation](practical.md) before interpreting a p-value or confidence level.
The [API reference](api.md) lists every exported function and result type.
Tutorial replicate counts keep local builds quick; choose analysis counts based
on Monte Carlo resolution and computation cost.

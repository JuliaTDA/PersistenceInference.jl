# PersistenceInference.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaTDA.github.io/PersistenceInference.jl/dev/)
[![CI](https://github.com/JuliaTDA/PersistenceInference.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaTDA/PersistenceInference.jl/actions/workflows/CI.yml)

Statistical tools for persistence diagrams in the [JuliaTDA](https://github.com/JuliaTDA)
ecosystem. A long bar describes a feature that survives across scales. To ask
whether that feature exceeds sampling noise, or whether topology is associated
with a covariate, you also need a sampling design and a statistical target.

| Your question | Function | Sampling unit |
| --- | --- | --- |
| How much does one cloud's diagram change under resampling? | `bootstrap_diagram`, `significant` | Point within an independent point sample |
| Do two point distributions differ through a diagram statistic? | `permutation_test` | Point, with exchangeable group labels under the null |
| What is the population mean landscape or Betti curve? | `confidence_band`, `multiplier_bootstrap` | Independent diagram/curve |
| Is the longest finite bar unusual under a specified null? | `max_persistence_significance` | Observed cloud and exchangeable null clouds/statistics |
| Are topology and a covariate independent? | `independence_test` | Independent paired observation |

## Install

Use Julia ≥1.9. This package uses the JuliaTDA forks whose **Julia package names**
are `TDARipserer` and `TDAPersistenceDiagrams`, although their repositories are
named `Ripserer.jl` and `PersistenceDiagrams.jl`. The upstream registered packages
have different identities and do not replace these dependencies.

```julia
using Pkg
Pkg.activate("tda-inference"; shared=false)
Pkg.develop([
    PackageSpec(url="https://github.com/JuliaTDA/PersistenceDiagrams.jl"),
    PackageSpec(url="https://github.com/JuliaTDA/Ripserer.jl"),
    PackageSpec(url="https://github.com/JuliaTDA/PersistenceInference.jl"),
])
```

For existing sibling checkouts, replace those URLs with the paths
`TDAPersistenceDiagrams.jl`, `TDARipserer.jl`, and `PersistenceInference.jl`.
The package also uses Julia's `LinearAlgebra`, `Statistics`, and `Random`.

## First experiment: resample a noisy circle

```julia
using PersistenceInference, Random

rng = Xoshiro(42)
angles = 2π .* rand(rng, 32)
X = [(cos(t) + 0.03randn(rng), sin(t) + 0.03randn(rng)) for t in angles]
result = bootstrap_diagram(X; n_boot=49, dim_max=1, rng=rng)
result.radius[2]            # H₁ radius; Julia indexing starts at 1
significant(result, 1)      # bars with persistence > twice that radius
```

This is a small executable illustration, not a claim that a particular number
of loops must pass the threshold. Increase the replicate count for analysis.
The bootstrap computes an empirical diagram-distance quantile. Stability explains
the `2c` threshold for a correctly calibrated bottleneck confidence set; it does
not, by itself, prove that this resampling scheme covers every population target.
Per-dimension radii also do not give simultaneous coverage across dimensions.

## Learn the workflows

- [Getting started and choosing the sampling unit](docs/src/index.md).
- [Point-cloud bootstrap and permutation tests](docs/src/point_clouds.md): what
  is resampled, what the null means, and how to read result fields.
- [Functional confidence bands](docs/src/functional_inference.md): independent
  diagrams, landscape/Betti definitions, matrix input, and finite-grid coverage.
- [Null maxima and independence](docs/src/hypothesis_tests.md): explicit null
  samplers, conservative Monte Carlo p-values, and paired HSIC inference.
- [Practical choices and numerical validation](docs/src/practical.md).
- [Complete API](docs/src/api.md).

Every tutorial contains seeded examples that execute during the documentation
build. Repeated points are valid bootstrap observations; bars in one diagram
are not independent diagram samples. Unrestricted permutations need exchangeability:
paired, clustered, or time-dependent designs require a different randomization
scheme than the functions currently provide.

## Build and verify locally

From the ecosystem workspace containing the three sibling checkouts:

```sh
julia --project=PersistenceInference.jl/docs -e 'using Pkg; Pkg.develop([PackageSpec(path="TDAPersistenceDiagrams.jl"), PackageSpec(path="TDARipserer.jl"), PackageSpec(path="PersistenceInference.jl")]); Pkg.instantiate()'
julia --project=PersistenceInference.jl/docs PersistenceInference.jl/docs/make.jl
```

Open `docs/build/index.html` inside this repository. The build does not deploy
unless `JULIATDA_DOCS_DEPLOY=true`. To test the package, use `Pkg.test()` in an
environment with the local dependencies developed. The existing
[validation guide](validation/README.md) reproduces calibration simulations
and an independent NumPy/SciPy formula check. Those simulations reveal finite
sample undercoverage in some examples; they do not certify arbitrary datasets.

## Background

The confidence-set viewpoint comes from [Fasy et al. (2014)](https://doi.org/10.1214/14-AOS1252).
Landscape mean inference is developed in [Chazal et al. (2015)](https://doi.org/10.20382/jocg.v6i2a8).
See the tutorials for the precise connection between these ideas and this implementation.

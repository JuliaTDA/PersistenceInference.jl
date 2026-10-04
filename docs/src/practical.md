# Practical choices and validation

```@meta
CurrentModule = PersistenceInference
```

## Choose the randomization before computing

Record the independent sampling unit, filtration, homology dimension, scale
units, any threshold or modulus, functional grid, kernel settings, null-model
parameters, RNG type/seed, and replicate count. Data-selected preprocessing and
parameter searches can alter calibration; a seeded RNG makes a calculation
reproducible but does not correct that selection.

Use the same scale units and filtration convention throughout. Scaling coordinates
also scales Euclidean Rips births, deaths, and diagram-distance radii. A PH
threshold truncates the filtration and can create essential/unresolved classes.
The bootstrap helper always retains infinite intervals; the maximum test excludes
them, landscapes ignore them, and Betti curves count them. These are distinct
conventions with different targets.

## Replicate counts and cost

| Procedure | What one replicate does | Main cost |
| --- | --- | --- |
| Point bootstrap | Resamples points and recomputes PH | `n_boot` PH runs plus the reference |
| Point-label permutation | Splits pooled points and recomputes two diagrams | `2*n_perm` PH runs plus two references |
| Functional band | Resamples curves or draws normal multipliers | Matrix operations on observations × grid |
| Null maximum from clouds | Generates one cloud and recomputes PH | `n_null` PH runs plus the reference |
| HSIC | Reorders a centered Gram matrix | Gram construction/validation and `n_perm` statistics |

Start with a small pilot cloud/count to assess runtime, then use a count suited
to the precision you need. The tutorials intentionally use modest counts for
builds. For Monte Carlo hypothesis tests, `B=99` permits p-values in steps of
0.01; `B=999` permits steps of 0.001. These steps do not imply the p-value itself
is known without Monte Carlo error. Empirical band quantiles also need enough
replicates in the tail. Larger replicate counts improve Monte Carlo calibration;
they do not fix model misspecification, dependence, or small-sample approximation.
The nonzero p-value convention follows [Phipson and Smyth (2010)](https://doi.org/10.2202/1544-6115.1585).

Comparing many dimensions, landscapes, bandwidths, or subjects requires a
multiplicity strategy appropriate to the analysis. The functional band is
simultaneous only over its one supplied grid/representation; the maximum test
has the specified global-null interpretation described in [Hypothesis
tests](hypothesis_tests.md). There is no automatic correction across calls.

## Common input mistakes

- **Confusing bars with observations:** provide independent diagrams to bands
  and independent paired observations to HSIC.
- **Passing a distance matrix as a cloud:** the point resamplers index the input
  as a vector of points; use supported point representations rather than relying
  on a matrix being resampled correctly.
- **Treating matrix HSIC input as raw data:** supply a square Gram matrix for
  each variable, or vector inputs with kernels.
- **Using `radius[1]` for loops:** it is H₀; H₁ uses position 2.
- **Using a coarse grid as a continuum guarantee:** bands only concern the
  supplied finite grid.
- **Receiving a zero-width band:** sample coordinates may be constant, including
  when a rare feature simply was not observed.
- **Receiving `Inf` as a null threshold:** the simulation count may not resolve
  the requested `alpha`; increase it and inspect ties/p-values.
- **Interpreting failure to reject as equality:** tests can have limited power.

## Reproduce the existing checks

The repository's [validation guide](https://github.com/JuliaTDA/PersistenceInference.jl/blob/main/validation/README.md)
contains exact commands and pinned NumPy/SciPy dependencies. Its independent
reference checks multiplier/bootstrap statistics, band endpoints, studentized
and deterministic coordinates, HSIC permutations, and maximum-test formulas at
absolute/relative tolerance `1e-10`. Julia exports the random draws and
permutations, so Julia and NumPy need not use the same RNG algorithm. The
recorded maximum discrepancy is `2.22e-15` (Julia 1.12.5, NumPy 2.2.6,
SciPy 1.15.3).

`validation/calibration.jl` checks simultaneous coverage against exact Bernoulli
mixture landscape/Betti means and a correlated Gaussian grid process. It also
checks type I error under independent Gaussian pairs and exchangeable Uniform
null maxima. The CSV records seeds, sample sizes, Monte Carlo settings, Wilson
intervals, and acceptance tolerances.

The recorded 1,000-replication run (seed `20261003`, 120 independent curves,
399 multipliers) observed 93.7% landscape, 92.8% Betti, and 95.3% Gaussian-grid
coverage at nominal 95%. Null rejection rates were 4.4% for HSIC and 6.0% for
maximum persistence at nominal 5%. The discrete examples show finite-sample
undercoverage at this size: an asymptotic band is not an exact finite-sample
guarantee. These studies validate the implementation in controlled scenarios;
they do not establish coverage for arbitrary dependent data, an estimated or
misspecified null, or the point-cloud bootstrap. Preserve that scope when
reporting validation.

## Build these tutorials locally

From the ecosystem workspace with the sibling fork checkouts:

```sh
julia --project=PersistenceInference.jl/docs -e 'using Pkg; Pkg.develop([PackageSpec(path="TDAPersistenceDiagrams.jl"), PackageSpec(path="TDARipserer.jl"), PackageSpec(path="PersistenceInference.jl")]); Pkg.instantiate()'
julia --project=PersistenceInference.jl/docs PersistenceInference.jl/docs/make.jl
```

Documenter executes the `@example` blocks and checks the documented API.
Open `PersistenceInference.jl/docs/build/index.html`. Deployment is opt-in via
`JULIATDA_DOCS_DEPLOY=true`; an ordinary local build only creates local output.

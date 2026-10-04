# Reproducing statistical and numerical validation

From the ecosystem workspace, with Julia ≥1.9:

```sh
julia --project=PersistenceInference.jl/validation -e 'using Pkg; Pkg.develop([PackageSpec(path="TDAPersistenceDiagrams.jl"),PackageSpec(path="TDARipserer.jl"),PackageSpec(path="PersistenceInference.jl")]); Pkg.instantiate()'
julia --project=PersistenceInference.jl/validation PersistenceInference.jl/validation/calibration.jl 1000
julia --project=PersistenceInference.jl/validation PersistenceInference.jl/validation/export_reference.jl /tmp/tda-inference-reference
python3 -m venv /tmp/tda-numpy-reference
/tmp/tda-numpy-reference/bin/pip install -r PersistenceInference.jl/validation/requirements.txt
/tmp/tda-numpy-reference/bin/python PersistenceInference.jl/validation/numpy_reference.py /tmp/tda-inference-reference
```

The independent NumPy/SciPy check uses centered Gaussian multiplier processes,
ordinary curve resampling, Type-7/linear quantiles, `tr(K H L H)/n²` for biased
HSIC, and conservative Monte Carlo +1 ranks. It checks unstudentized/studentized
bands, deterministic coordinates, every replicate statistic, permutation
p-values and the maximum-persistence threshold. Raw random draws/permutations
are shared through CSV; matching Julia and NumPy RNG algorithms is unnecessary.
Absolute/relative tolerance is **1e-10**; Julia 1.12.5 produced a maximum error
of 2.22e-15 with NumPy 2.2.6 and SciPy 1.15.3.

`calibration.jl` checks known population means of a two-diagram Bernoulli mixture
(landscape and Betti), a correlated seven-dimensional Gaussian mean, independent
Gaussian HSIC samples, and an exchangeable Uniform maximum null. Seed 20261003,
1000 repetitions, 120 curves, 399 multipliers and 199 permutations are recorded
in `calibration.csv`, together with Wilson intervals and predeclared tolerances.
Acceptance allows three Monte Carlo standard errors; asymptotic bands also
allow 2 percentage points of finite-sample approximation error. This is an
accuracy regression check, not a proof of coverage for arbitrary distributions.

The recorded coverage is 93.7% landscape, 92.8% Betti and 95.3% Gaussian at a
nominal 95%; type-I error is 4.4% HSIC and 6.0% maximum-persistence at nominal 5%.
Betti's interval reveals finite-sample undercoverage: do not interpret an
asymptotic band as an exact guarantee. See `docs/src/functional_inference.md`
for sampling-unit, null-model and fixed-grid restrictions.

Published formulas: [Chazal et al. (2015)](https://doi.org/10.20382/jocg.v6i2a8),
[Gretton et al. (2008)](https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html),
[Phipson & Smyth (2010)](https://doi.org/10.2202/1544-6115.1585).

# Null maxima and independence tests

```@meta
CurrentModule = PersistenceInference
```

## The longest finite bar under a specified null

A bootstrap asks how observations vary when resampled. A null-model test asks
whether an observation is unusual under a scientifically specified alternative
explanation for the data. [`max_persistence_significance`](@ref) compares the
largest **finite** persistence to null maxima. You supply either null diagrams,
precomputed maxima, or a point-cloud generator.

```@example null_maxima
using PersistenceInference, TDAPersistenceDiagrams, Random
rng = Xoshiro(31)
observed = PersistenceDiagram([(0.0, 1.5), (0.0, 0.1)]; dim=1)
null_maxima = 0.5 .* rand(rng, 199)
result = max_persistence_significance(observed, null_maxima; alpha=0.05)
@assert result.pvalue == 1/200
@assert length(significant(result)) == 1
(result=result, interval_pvalues=result.interval_pvalues)
```

This toy null assumes maxima are Uniform on `[0, 0.5]`. It demonstrates the
calculation, not a null distribution for realistic persistence diagrams. The
large bar cannot occur under that toy null; the smaller bar is compared to the
same null maximum distribution.

### Generate null point clouds

Suppose the scientific null is independent uniform observations in a fixed unit
disk. The generator receives the RNG and sample size, and must return a cloud
of the same size. The function uses identical PH options for all clouds.

```@example null_clouds
using PersistenceInference, Random
function uniform_disk(rng, n)
    radii = sqrt.(rand(rng, n))
    angles = 2π .* rand(rng, n)
    [(r*cos(t), r*sin(t)) for (r, t) in zip(radii, angles)]
end
rng = Xoshiro(73)
angles = 2π .* rand(rng, 20)
X = [(cos(t), sin(t)) for t in angles]
result = max_persistence_significance(X; null_sampler=uniform_disk,
    n_null=39, dim=1, alpha=0.05, rng=rng)
(result.observed, result.pvalue, result.threshold)
```

The observed and simulated statistics must be exchangeable under the null.
This holds for independent simulation from the correct known null model, or a
justified conditional randomization. An estimated disk radius, density, noise
level, or other nuisance parameter fitted to these observations changes that
argument and requires separate calibration. Resampling the observed cloud
alone does not automatically generate a no-topological-signal null.

With precomputed diagrams, match the filtration, dimension, units, sample size,
and sampling design. The diagram overload checks dimension metadata; it cannot
verify the other conditions. With precomputed numeric maxima, you are responsible
for all of them.

### Global and barwise results

`observed` is the largest finite persistence (zero if no finite intervals exist).
`null_maxima` stores all simulated maxima. The global `pvalue` is
`(1 + count(null_maximum ≥ observed))/(B+1)`. `interval_pvalues` applies that
same formula to **each observed finite bar**, comparing it to the null maxima,
not to a pool of individual null bars. `significant(result)` selects bars whose
values are at most `alpha`.

This comparison controls the chance of flagging any finite bar **under the
specified global null**, assuming the exchangeability conditions. It does not
establish strong familywise error control under every possible configuration
of real and spurious features. Essential intervals have p-value 1 and are
excluded, unlike the bootstrap selection convention.

The conservative `threshold` uses sorted null maximum number
`ceil((1-alpha)*(B+1))`; if this is larger than `B`, the threshold is `Inf`.
A finite bar must strictly exceed the threshold; ties are conservative.
Use `interval_pvalues` or `significant(result)` for the documented selection rule. With `B` simulations the finest possible p-value is `1/(B+1)`.

## Independence of a diagram and a covariate

Now the independent unit is a **pair**, such as `(subject diagram, subject age)`.
[`independence_test`](@ref) uses the Hilbert–Schmidt independence criterion
(HSIC): kernel similarities among diagrams and among covariates should be
unrelated under independence. Permuting the pairings preserves each marginal
sample and breaks their association.

```@example hsic_diagrams
using PersistenceInference, TDAPersistenceDiagrams, Random
rng = Xoshiro(44)
lifetimes = 0.5 .+ rand(rng, 30)
diagrams = [PersistenceDiagram([(0.0, life)]; dim=1) for life in lifetimes]
covariate = lifetimes .+ 0.05randn(rng, 30)
gaussian(x, y) = exp(-(x-y)^2 / (2*0.2^2))
test = independence_test(diagrams, covariate; kernel_y=gaussian,
                         n_perm=199, rng=rng)
@assert test.n_samples == 30
@assert length(test.null_distribution) == 199
(test.observed, test.pvalue)
```

The defaults use `TDAPersistenceDiagrams.SlicedWassersteinKernel()` for **both**
inputs. Numeric covariates therefore need an appropriate supplied `kernel_y`.
Both kernels must generate finite, symmetric positive semidefinite Gram matrices.
The implementation checks these conditions numerically.

### Gram matrices are an alternate input

For numeric vectors `x` and `y`, create kernels directly, or pass precomputed
Gram matrices `K` and `L` of matching `n × n` shape. Matrices are interpreted as
**Gram matrices**, not raw observations with features in columns.

```@example hsic_grams
using PersistenceInference, Random
rng = Xoshiro(81)
x = randn(rng, 24)
y = x .^ 2 .+ 0.2randn(rng, 24)
k(a, b) = exp(-(a-b)^2 / 2)
K = [k(a,b) for a in x, b in x]
L = [k(a,b) for a in y, b in y]
test = independence_test(K, L; n_perm=99, rng=rng)
@assert test.n_samples == size(K, 1)
test
```

At least three paired observations are required. After centering both Gram
matrices, `observed` is the biased HSIC V-statistic
``\sum_{ij}(K_c)_{ij}(L_c)_{ij}/n^2``. Each permutation jointly reorders rows and
columns of `L_c`; `null_distribution` stores the resulting statistics. The
p-value uses the same `+1` Monte Carlo correction.

Valid size requires exchangeable i.i.d. pairs under independence. Sensitivity to
*every* dependence alternative additionally requires characteristic kernels;
this is not promised for every supplied diagram kernel. The default kernel
also ignores essential intervals, so choose a representation that reflects your
scientific question. Do not permute individual points in a cloud as though they
were diagram subjects. Time series, paired longitudinal records, and clustered
subjects need a design-aware procedure that this API does not currently provide.
The HSIC testing framework comes from [Gretton et al.](https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html).

# Permutation (two-sample) test for the topology of two point clouds.
#
# Reference:
#   Robinson & Turner (2017), "Hypothesis testing for topological data
#   analysis", Journal of Applied and Computational Topology, 1(2), 241-261.
#   https://doi.org/10.1007/s41468-017-0008-7

"""
    PermutationTestResult

Result of [`permutation_test`](@ref): a two-sample permutation test comparing
the persistent homology of two point clouds at a fixed homology dimension.

# Fields

- `pvalue::Float64`: the permutation p-value,
  `(1 + #{perm stat ≥ observed}) / (1 + n_perm)`.
- `observed::Float64`: the observed test statistic `W_q^q` between the diagrams
  of the two samples at dimension `dim`.
- `null_distribution::Vector{Float64}`: the `n_perm` statistics computed under
  random relabellings of the pooled points.
- `n_perm::Int`: the number of permutations used.
- `dim::Int`: the homology dimension tested.

A small p-value is evidence against the exchangeable point-distribution null,
detected through persistence diagrams in dimension `dim`. Equality of underlying
topology alone does not ensure exchangeability or test validity.
"""
struct PermutationTestResult
    pvalue::Float64
    observed::Float64
    null_distribution::Vector{Float64}
    n_perm::Int
    dim::Int
end

# W_q^q between the dim-th diagrams of two ripserer outputs.
# Wasserstein(q) sets p = q, so `Wasserstein(q)(a, b)` is W_q = (Σ d^q)^(1/q);
# raising to the q-th power gives the W_q^q joint-loss statistic of
# Robinson & Turner (2017).
function _wqq(diags_a, diags_b, dim::Int, q::Real)
    idx = dim + 1
    a = diags_a[idx]
    b = diags_b[idx]
    wq = Wasserstein(q)(a, b)
    return Float64(wq)^q
end

"""
    permutation_test(X, Y; n_perm=1000, dim=1, q=2, dim_max=dim,
                     rng=Random.default_rng(), progress=false, workers=1,
                     ripserer_kwargs...)

Permutation test for whether two point clouds `X` and `Y` have different
persistent homology in dimension `dim`, following Robinson & Turner (2017).

`X` and `Y` are vectors of points of the same kind accepted by
[`TDARipserer.ripserer`](https://github.com/JuliaTDA/Ripserer.jl) (tuples,
`SVector`s, or a `MetricSpaces.EuclideanSpace`, which is a vector of points).
They must contain points of compatible type so they can be pooled.

# Method

The test statistic is the `q`-Wasserstein loss `W_q^q` between the
dimension-`dim` persistence diagrams of `X` and `Y` (the "joint loss" of
Robinson & Turner with a single diagram per group). Under the null hypothesis
that `X` and `Y` are drawn from the same distribution, the group labels are
exchangeable, so:

1. Compute the observed statistic from `diagram(X)` and `diagram(Y)`.
2. For each of `n_perm` permutations: pool all points of `X` and `Y`, randomly
   split them into two groups of the *original* sizes, recompute the two
   diagrams and the statistic.
3. The p-value is `(1 + #{perm stat ≥ observed}) / (1 + n_perm)`.

The `+1` in numerator and denominator (Phipson & Smyth 2010) keeps the p-value
in `(0, 1]`, with minimum `1/(n_perm+1)`. Under the exchangeable null it gives
a conservative Monte Carlo permutation test; this function samples permutations
rather than enumerating the full distribution. Paired, clustered or dependent
samples need design-specific randomization not implemented here.

!!! warning "Cost"
    This is `O(n_perm × cost(ripserer))`. Each permutation recomputes
    persistent homology of each relabelled group. TDARipserer is fast on small
    clouds, but keep `n_perm` and the cloud sizes modest, or be patient. The
    default examples in this package are intentionally tiny.

# Arguments

- `X`, `Y`: the two point clouds.
- `n_perm=1000`: number of random permutations.
- `dim=1`: homology dimension to test.
- `q=2`: order of the Wasserstein distance (the statistic is `W_q^q`).
- `dim_max=dim`: maximum homology dimension to compute (must be `≥ dim`).
- `rng=Random.default_rng()`: random number generator, for reproducibility.
- `progress=false`: when `true`, print sparse progress (roughly every 10%).
- `workers=1`: maximum concurrent replicates, capped by the available Julia
  threads. Permutations are generated on the calling task in seeded order, so
  their statistics and RNG consumption match serial execution. Use a
  thread-safe metric when opting into parallelism.
- `ripserer_kwargs...`: forwarded to `ripserer`.

# Returns

A [`PermutationTestResult`](@ref).

# Example

```julia
using PersistenceInference, Random

circle(n, σ, rng) = [(cos(t) + σ*randn(rng), sin(t) + σ*randn(rng))
                     for t in range(0, 2π; length = n+1)[1:n]]
blob(n, rng)      = [(2*rand(rng) - 1, 2*rand(rng) - 1) for _ in 1:n]

rng = Random.Xoshiro(1)
X = circle(30, 0.05, rng)
Y = blob(30, rng)

result = permutation_test(X, Y; n_perm = 99, dim = 1, rng = rng, workers = 4)
result.pvalue   # small: circle and blob differ topologically
```

# See also

- [`PermutationTestResult`](@ref), [`bootstrap_diagram`](@ref).
"""
function permutation_test(
    X,
    Y;
    n_perm::Integer=1000,
    dim::Integer=1,
    q::Real=2,
    dim_max::Integer=dim,
    rng::Random.AbstractRNG=Random.default_rng(),
    progress::Bool=false,
    workers::Integer=1,
    ripserer_kwargs...,
)
    n_perm ≥ 1 || throw(ArgumentError("`n_perm` must be at least 1"))
    workers ≥ 1 || throw(ArgumentError("`workers` must be at least 1"))
    dim ≥ 0 || throw(ArgumentError("`dim` must be non-negative"))
    dim_max ≥ dim || throw(ArgumentError("`dim_max` must be at least `dim`"))
    q > 0 || throw(ArgumentError("`q` must be positive"))
    nx = length(X)
    ny = length(Y)
    (nx ≥ 1 && ny ≥ 1) || throw(ArgumentError("`X` and `Y` must be non-empty"))

    dim = Int(dim)

    diags_x = ripserer(X; dim_max=dim_max, ripserer_kwargs...)
    diags_y = ripserer(Y; dim_max=dim_max, ripserer_kwargs...)
    observed = _wqq(diags_x, diags_y, dim, q)

    pooled = vcat(collect(X), collect(Y))
    n_total = nx + ny

    null_distribution = Vector{Float64}(undef, n_perm)
    worker_count = Int(min(workers, Threads.nthreads(), n_perm))
    step = max(1, n_perm ÷ 10)

    function statistic(perm)
        group_x = pooled[view(perm, 1:nx)]
        group_y = pooled[view(perm, (nx + 1):n_total)]
        dx = ripserer(group_x; dim_max=dim_max, ripserer_kwargs...)
        dy = ripserer(group_y; dim_max=dim_max, ripserer_kwargs...)
        return _wqq(dx, dy, dim, q)
    end

    for lo in 1:worker_count:n_perm
        hi = min(lo + worker_count - 1, n_perm)
        if worker_count == 1
            null_distribution[lo] = statistic(randperm(rng, n_total))
        else
            permutations = [randperm(rng, n_total) for _ in lo:hi]
            @sync for (p, perm) in zip(lo:hi, permutations)
                Threads.@spawn null_distribution[p] = statistic(perm)
            end
        end
        if progress && (div(hi, step) > div(lo - 1, step) || hi == n_perm)
            pct = round(Int, 100 * hi / n_perm)
            println("permutation_test: ", hi, "/", n_perm, " (", pct, "%)")
        end
    end

    count_ge = count(>=(observed), null_distribution)
    pvalue = (1 + count_ge) / (1 + n_perm)

    return PermutationTestResult(pvalue, observed, null_distribution, Int(n_perm), dim)
end

function Base.show(io::IO, ::MIME"text/plain", result::PermutationTestResult)
    print(io, "PermutationTestResult (H", result.dim, ", ", result.n_perm, " permutations)")
    print(io, "\n  observed W_q^q = ", round(result.observed; sigdigits=5))
    print(io, "\n  p-value        = ", round(result.pvalue; sigdigits=4))
    print(io, "\n  null dist.     = [", round(minimum(result.null_distribution); sigdigits=4),
        ", ", round(maximum(result.null_distribution); sigdigits=4), "] (min, max)")
end

function Base.show(io::IO, result::PermutationTestResult)
    print(io, "PermutationTestResult(pvalue=", round(result.pvalue; sigdigits=4),
        ", observed=", round(result.observed; sigdigits=5),
        ", dim=", result.dim, ", n_perm=", result.n_perm, ")")
end

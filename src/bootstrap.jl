# Bootstrap confidence sets for persistence diagrams.
#
# Reference:
#   Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan & Singh (2014),
#   "Confidence sets for persistence diagrams", The Annals of Statistics,
#   42(6), 2301-2339. https://doi.org/10.1214/14-AOS1252

"""
    BootstrapResult

Result of [`bootstrap_diagram`](@ref): a bootstrap confidence set for the
persistence diagram of a point cloud.

# Fields

- `diagrams::Vector{<:PersistenceDiagram}`: the reference diagrams computed
  from the full sample, one per homology dimension `0:dim_max`.
- `radius::Vector{Float64}`: the per-dimension confidence-set radius `c`,
  i.e. the `(1 - alpha)` quantile of the bootstrap distances. A feature with
  persistence greater than `2 * radius[d + 1]` is significant at level `alpha`.
- `distances::Vector{Vector{Float64}}`: the raw bootstrap distances, one
  vector of length `n_boot` per homology dimension.
- `alpha::Float64`: the significance level used.
- `n_boot::Int`: the number of bootstrap resamples used.

# See also

- [`significant`](@ref) to extract the features distinguishable from noise.
"""
struct BootstrapResult{D<:PersistenceDiagram}
    diagrams::Vector{D}
    radius::Vector{Float64}
    distances::Vector{Vector{Float64}}
    alpha::Float64
    n_boot::Int
end

"""
    bootstrap_diagram(X; n_boot=100, alpha=0.05, dim_max=1,
                      subsample_size=nothing, distance=Bottleneck(),
                      rng=Random.default_rng(), ripserer_kwargs...)

Compute a bootstrap confidence set for the persistence diagram of the point
cloud `X`, following Fasy et al. (2014).

`X` is a vector of points; each point may be a `Tuple`, an `SVector`, or any
type accepted by [`Ripserer.ripserer`](https://github.com/JuliaTDA/Ripserer.jl).
A `MetricSpaces.EuclideanSpace` works directly because it is a vector of
`SVector` points.

# Method

1. Compute the reference diagrams `ripserer(X; dim_max, ripserer_kwargs...)`.
2. For `b in 1:n_boot`, draw a resample of `X` *with replacement* of size
   `subsample_size === nothing ? length(X) : subsample_size`, compute its
   diagrams, and record the matching `distance` between the reference and
   bootstrap diagram *separately in each homology dimension*.
3. The confidence-set radius `c` in dimension `d` is the empirical
   `(1 - alpha)` quantile of those `n_boot` distances.

By the stability theorem, the true diagram lies within bottleneck distance `c`
of the reference diagram with asymptotic probability `1 - alpha`. In the
diagram, this is the "confidence band" of half-width `c` around the diagonal:
a feature is *distinguishable from noise* only if its persistence exceeds
`2c`. See [`significant`](@ref).

# Arguments

- `X`: the point cloud (a vector of points).
- `n_boot=100`: number of bootstrap resamples.
- `alpha=0.05`: significance level; the confidence set has level `1 - alpha`.
- `dim_max=1`: maximum homology dimension to compute.
- `subsample_size=nothing`: resample size. `nothing` means the full sample
  size `length(X)` (the standard nonparametric bootstrap). A smaller value
  gives the `m`-out-of-`n` subsampling bootstrap, which can be both faster and
  more stable.
- `distance=Bottleneck()`: a `PersistenceDiagrams` matching distance, callable
  as `distance(d1, d2)`. `Bottleneck()` matches Fasy et al.; `Wasserstein(q)`
  is also accepted.
- `rng=Random.default_rng()`: random number generator, for reproducibility.
- `ripserer_kwargs...`: forwarded to `ripserer` (e.g. `modulus`, `threshold`).

# Returns

A [`BootstrapResult`](@ref).

# Example

```julia
using PersistenceInference, Random

# noisy circle
rng = Random.Xoshiro(42)
n = 60
θ = range(0, 2π; length = n + 1)[1:n]
X = [(cos(t) + 0.05 * randn(rng), sin(t) + 0.05 * randn(rng)) for t in θ]

result = bootstrap_diagram(X; n_boot = 100, alpha = 0.05, dim_max = 1, rng = rng)
significant(result, 1)   # the loop of the circle, distinguishable from noise
```

# See also

- [`significant`](@ref), [`BootstrapResult`](@ref), [`permutation_test`](@ref).
"""
function bootstrap_diagram(
    X;
    n_boot::Integer=100,
    alpha::Real=0.05,
    dim_max::Integer=1,
    subsample_size::Union{Nothing,Integer}=nothing,
    distance=Bottleneck(),
    rng::Random.AbstractRNG=Random.default_rng(),
    ripserer_kwargs...,
)
    n_boot ≥ 1 || throw(ArgumentError("`n_boot` must be at least 1"))
    0 < alpha < 1 || throw(ArgumentError("`alpha` must be in (0, 1)"))
    dim_max ≥ 0 || throw(ArgumentError("`dim_max` must be non-negative"))
    n = length(X)
    n ≥ 1 || throw(ArgumentError("`X` must be non-empty"))
    m = subsample_size === nothing ? n : Int(subsample_size)
    m ≥ 1 || throw(ArgumentError("`subsample_size` must be at least 1"))

    reference = ripserer(X; dim_max=dim_max, ripserer_kwargs...)
    n_dims = length(reference)

    # distances[d] holds the n_boot bootstrap distances for homology dim (d - 1)
    distances = [Vector{Float64}(undef, n_boot) for _ in 1:n_dims]

    for b in 1:n_boot
        idx = rand(rng, 1:n, m)
        boot_diagrams = ripserer(X[idx]; dim_max=dim_max, ripserer_kwargs...)
        for d in 1:n_dims
            # ripserer always returns dim_max + 1 diagrams, so the dimensions
            # of `reference` and `boot_diagrams` line up by index.
            distances[d][b] = Float64(distance(reference[d], boot_diagrams[d]))
        end
    end

    radius = [quantile(dists, 1 - alpha) for dists in distances]

    return BootstrapResult(reference, radius, distances, Float64(alpha), Int(n_boot))
end

"""
    significant(result::BootstrapResult, dim::Integer)

Return the intervals of the reference diagram in homology dimension `dim`
whose persistence exceeds `2 * result.radius[dim + 1]` — that is, the features
distinguishable from noise at level `result.alpha`.

In the confidence-set picture of Fasy et al. (2014), the band of half-width
`c = radius[dim + 1]` around the diagonal is the region where points cannot be
told apart from noise; an interval `(b, d)` is significant exactly when its
distance to the diagonal, `(d - b) / 2`, exceeds `c`, i.e. when
`death - birth > 2c`.

# Arguments

- `result`: a [`BootstrapResult`](@ref).
- `dim`: the homology dimension (`0`-based, as in `ripserer`).

# Returns

A vector of `PersistenceInterval`s (a subset of `result.diagrams[dim + 1]`).
Infinite intervals are always returned as significant.

# Example

```julia
result = bootstrap_diagram(X; dim_max = 1)
significant(result, 1)   # the significant loops
```
"""
function significant(result::BootstrapResult, dim::Integer)
    dim ≥ 0 || throw(ArgumentError("`dim` must be non-negative"))
    idx = dim + 1
    1 ≤ idx ≤ length(result.diagrams) ||
        throw(ArgumentError("dimension $dim was not computed (dim_max = $(length(result.diagrams) - 1))"))
    threshold = 2 * result.radius[idx]
    diagram = result.diagrams[idx]
    return filter(int -> !isfinite(int) || persistence(int) > threshold, diagram)
end

function Base.show(io::IO, ::MIME"text/plain", result::BootstrapResult)
    print(io, "BootstrapResult")
    print(io, " (", result.n_boot, " resamples, α = ", result.alpha, ")")
    for (d, (diag, c)) in enumerate(zip(result.diagrams, result.radius))
        dim = d - 1
        n_sig = count(int -> !isfinite(int) || persistence(int) > 2c, diag)
        print(io, "\n  H", dim, ": radius c = ", round(c; sigdigits=4),
            ", ", length(diag), " feature", length(diag) == 1 ? "" : "s",
            ", ", n_sig, " significant (persistence > ", round(2c; sigdigits=4), ")")
    end
end

function Base.show(io::IO, result::BootstrapResult)
    print(io, "BootstrapResult(n_boot=", result.n_boot, ", alpha=", result.alpha,
        ", radius=", round.(result.radius; sigdigits=4), ")")
end

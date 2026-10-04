"""
    ConfidenceBand

A simultaneous confidence band for a population mean on the finite `grid`.
`estimate`, `lower`, and `upper` have one entry per grid point. `maxima` stores
the bootstrap supremum statistics on the sqrt(n) scale; `critical_value` is
their (1 - alpha) quantile. `studentized` records whether pointwise sample
standard deviations were used. `n_samples` counts independent observations,
not points in a single cloud. Coverage is asymptotic and applies to the supplied
grid, not an unobserved continuum or individual future curves.
"""
struct ConfidenceBand
    grid::Vector{Float64}
    estimate::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    critical_value::Float64
    maxima::Vector{Float64}
    alpha::Float64
    n_samples::Int
    method::Symbol
    studentized::Bool
end

function _band_inputs(samples, grid, n_boot, alpha, studentize)
    n, p = size(samples)
    n >= 2 || throw(ArgumentError("at least two independent curves are required"))
    p >= 1 || throw(ArgumentError("curves must have at least one grid point"))
    n_boot >= 1 || throw(ArgumentError("n_boot must be positive"))
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1)"))
    all(isfinite, samples) || throw(ArgumentError("sample curves must be finite"))
    ts = Float64.(collect(grid))
    length(ts) == p || throw(DimensionMismatch("grid and sample columns must agree"))
    all(isfinite, ts) && all(>(0), diff(ts)) ||
        throw(ArgumentError("grid must be finite and strictly increasing"))
    values = Matrix{Float64}(samples)
    estimate = vec(mean(values; dims=1))
    centered = values .- transpose(estimate)
    deviations = vec(std(values; dims=1))
    scales = studentize ? deviations : map(s -> iszero(s) ? 0.0 : 1.0, deviations)
    # Constant coordinates are deterministic in the empirical distribution.
    # Their multiplier process and band width are exactly zero.
    normalizers = map(s -> iszero(s) ? 1.0 : s, scales)
    return values, ts, estimate, centered, scales, normalizers
end

function _confidence_band(samples; grid=1:size(samples, 2), n_boot=1000,
        alpha=0.05, studentize=false, rng=Random.default_rng(), method=:multiplier)
    values, ts, estimate, centered, scales, normalizers =
        _band_inputs(samples, grid, n_boot, alpha, studentize)
    n = size(values, 1)
    maxima = Vector{Float64}(undef, n_boot)
    for b in 1:n_boot
        process = if method == :multiplier
            vec(transpose(centered) * randn(rng, n)) / sqrt(n)
        else
            sqrt(n) .* (vec(mean(values[rand(rng, 1:n, n), :]; dims=1)) .- estimate)
        end
        maxima[b] = maximum(abs.(process ./ normalizers))
    end
    critical = quantile(maxima, 1 - alpha)
    widths = critical .* scales ./ sqrt(n)
    return ConfidenceBand(ts, estimate, estimate .- widths, estimate .+ widths,
        critical, maxima, Float64(alpha), n, method, studentize)
end

"""
    multiplier_bootstrap(samples::AbstractMatrix; grid=1:size(samples, 2),
        n_boot=1000, alpha=0.05, studentize=false, rng=Random.default_rng())
    multiplier_bootstrap(diagrams; grid, representation=:landscape, k=1, kwargs...)

Gaussian multiplier bootstrap for the mean of independent functional samples.
Rows are independent observations; columns are values on a fixed grid. Each
replicate is `n^(-1/2) * sum(ξᵢ * (fᵢ - mean(f)))`, with independent standard
normal ξᵢ, and the supremum of its absolute values calibrates the band.
`studentize=true` divides the process by pointwise sample standard deviations;
zero-variance coordinates have zero width. Returns a [`ConfidenceBand`](@ref).

For diagrams, use the same fixed `grid` for all independent diagrams, with
`representation=:landscape` (k-th landscape) or `:betti` (pointwise Betti curve).
See [`confidence_band`](@ref) for the statistical target and limitations.

Reference: Chazal et al. (2015), *Stochastic convergence of persistence landscapes
and silhouettes*, https://doi.org/10.20382/jocg.v6i2a8.
"""
function multiplier_bootstrap(samples::AbstractMatrix; kwargs...)
    return _confidence_band(samples; method=:multiplier, kwargs...)
end

"""
    confidence_band(samples::AbstractMatrix; method=:multiplier, kwargs...)
    confidence_band(diagrams::AbstractVector{<:PersistenceDiagram}; grid,
        representation=:landscape, k=1, method=:multiplier, kwargs...)

Simultaneous finite-grid confidence band for the **population mean** of i.i.d.
curves or diagrams. `method=:multiplier` uses Gaussian multipliers;
`method=:bootstrap` resamples entire curves with replacement. Both calibrate
the supremum deviation, rather than separately constructing pointwise intervals.
Other keywords are those of [`multiplier_bootstrap`](@ref).

`grid` is fixed in advance. Landscape values are sampled exactly at the supplied
times (not bucket-integrated); `k` must be positive. Landscapes ignore essential
intervals. Betti curves count `birth <= t < death`, including essential intervals.
All diagrams must represent the same homology dimension and independent sampling
units. Do not treat bars or bootstrap replicas of one cloud as independent data.
For a fixed finite grid, ordinary multivariate CLT/bootstrap conditions apply
(finite second moments and nondegenerate stochastic coordinates); the landscape
functional theory does not automatically extend to continuous Betti bands.

# Example
```julia
using PersistenceInference, TDAPersistenceDiagrams, Random
rng = Xoshiro(42)
diagrams = [PersistenceDiagram([(0.0, 1 + rand(rng))]; dim=1) for _ in 1:80]
band = confidence_band(diagrams; grid=range(0, 2; length=101), rng=rng)
betti = confidence_band(diagrams; grid=band.grid, representation=:betti, rng=rng)
```
"""
function confidence_band(samples::AbstractMatrix; method::Symbol=:multiplier, kwargs...)
    method in (:multiplier, :bootstrap) ||
        throw(ArgumentError("method must be :multiplier or :bootstrap"))
    return _confidence_band(samples; method=method, kwargs...)
end

function _diagram_curves(diagrams, grid, representation, k)
    representation in (:landscape, :betti) ||
        throw(ArgumentError("representation must be :landscape or :betti"))
    k >= 1 || throw(ArgumentError("landscape order k must be positive"))
    ts = collect(grid)
    curves = zeros(length(diagrams), length(ts))
    dims = [get(d.meta, :dim, missing) for d in diagrams]
    isempty(dims) || all(d -> isequal(d, first(dims)), dims) ||
        throw(ArgumentError("diagrams must have the same homology dimension"))
    for (i, diagram) in enumerate(diagrams), (j, t) in enumerate(ts)
        if representation == :betti
            curves[i, j] = count(int -> birth(int) <= t < death(int), diagram)
        else
            heights = [max(0.0, min(t - birth(int), death(int) - t))
                for int in diagram if isfinite(int)]
            sort!(heights; rev=true)
            curves[i, j] = length(heights) < k ? 0.0 : heights[k]
        end
    end
    return curves
end

function confidence_band(diagrams::AbstractVector{<:PersistenceDiagram}; grid,
        representation::Symbol=:landscape, k::Integer=1, kwargs...)
    curves = _diagram_curves(diagrams, grid, representation, k)
    return confidence_band(curves; grid=grid, kwargs...)
end

function multiplier_bootstrap(diagrams::AbstractVector{<:PersistenceDiagram}; kwargs...)
    return confidence_band(diagrams; method=:multiplier, kwargs...)
end

function Base.show(io::IO, band::ConfidenceBand)
    print(io, "ConfidenceBand(n=", band.n_samples, ", points=", length(band.grid),
        ", alpha=", band.alpha, ", method=", band.method, ")")
end

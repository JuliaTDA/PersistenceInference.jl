"""
    MaxPersistenceResult

Monte Carlo test of the maximum **finite** persistence against a supplied null
model. Contains `observed`, `pvalue`, the conservative `threshold`,
`null_maxima`, `alpha`, and the observed `diagram`. `interval_pvalues` compares
each finite interval against the null maximum, providing simultaneous control
under the specified global null. Essential intervals have p-value 1 and are
excluded from this test. [`significant`](@ref) extracts rejected intervals.
"""
struct MaxPersistenceResult
    diagram::PersistenceDiagram
    observed::Float64
    pvalue::Float64
    threshold::Float64
    null_maxima::Vector{Float64}
    interval_pvalues::Vector{Float64}
    alpha::Float64
end

_max_finite_persistence(d) = maximum((persistence(i) for i in d if isfinite(i)); init=0.0)

"""
    max_persistence_significance(diagram, null_diagrams; alpha=0.05)
    max_persistence_significance(diagram, null_maxima::AbstractVector{<:Real}; alpha=0.05)
    max_persistence_significance(X; null_sampler, n_null=999, dim=1,
        alpha=0.05, rng=Random.default_rng(), ripserer_kwargs...)

Test whether the longest finite bar exceeds those generated under an explicit
global null. The Monte Carlo p-value is `(1 + count(null_max >= observed))/(B+1)`.
The threshold uses order `ceil((1-alpha)*(B+1))`, becoming `Inf` when the number
of simulations cannot resolve `alpha`. Ties are conservative. Each interval is
also compared to the null **maximum**, rather than to pooled null bars.

For point clouds, `null_sampler(rng, length(X))` must generate an independent
cloud under a scientifically justified null with the same design/sample size.
The observed and null clouds use identical PH options. Resampling the observed
cloud alone is not automatically a null model. Precomputed null diagrams must
use the same filtration, homology dimension, sampling design, and units.
Essential bars are excluded; an empty finite diagram has maximum zero.

Valid size requires exchangeability of the observed statistic and the simulated
null statistics under the global null (known null parameters or a valid conditional
randomization). Fitting a null to these observations requires separate calibration.

# Example
```julia
using PersistenceInference, TDAPersistenceDiagrams
observed = PersistenceDiagram([(0.0, 3.0), (0.0, 0.1)]; dim=1)
result = max_persistence_significance(observed, fill(0.5, 99))
result.pvalue == 0.01
significant(result) # the long interval
```

Reference for Monte Carlo p-values: Phipson & Smyth (2010),
https://doi.org/10.2202/1544-6115.1585.
"""
function max_persistence_significance(diagram::PersistenceDiagram,
        null_maxima::AbstractVector{<:Real}; alpha::Real=0.05)
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1)"))
    isempty(null_maxima) && throw(ArgumentError("at least one null replicate is required"))
    all(x -> isfinite(x) && x >= 0, null_maxima) ||
        throw(ArgumentError("null maxima must be finite and nonnegative"))
    maxima = Float64.(null_maxima)
    b = length(maxima)
    observed = _max_finite_persistence(diagram)
    pvalue = (1 + count(>=(observed), maxima)) / (b + 1)
    rank = ceil(Int, (1 - alpha) * (b + 1))
    threshold = rank > b ? Inf : sort(maxima)[rank]
    pvalues = [isfinite(int) ? (1 + count(>=(persistence(int)), maxima)) / (b + 1) : 1.0
        for int in diagram]
    return MaxPersistenceResult(diagram, observed, pvalue, threshold, maxima,
        pvalues, Float64(alpha))
end

function max_persistence_significance(diagram::PersistenceDiagram,
        null_diagrams::AbstractVector{<:PersistenceDiagram}; kwargs...)
    all(d -> isequal(get(d.meta, :dim, missing), get(diagram.meta, :dim, missing)),
        null_diagrams) || throw(ArgumentError("null diagram dimensions must agree"))
    return max_persistence_significance(diagram, _max_finite_persistence.(null_diagrams);
        kwargs...)
end

function max_persistence_significance(X; null_sampler, n_null::Integer=999,
        dim::Integer=1, alpha::Real=0.05, rng::Random.AbstractRNG=Random.default_rng(),
        ripserer_kwargs...)
    n_null >= 1 || throw(ArgumentError("n_null must be positive"))
    dim >= 0 || throw(ArgumentError("dim must be nonnegative"))
    isempty(X) && throw(ArgumentError("X must be nonempty"))
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1)"))
    diagram = ripserer(X; dim_max=dim, ripserer_kwargs...)[dim + 1]
    nulls = Vector{Float64}(undef, n_null)
    for b in 1:n_null
        cloud = null_sampler(rng, length(X))
        length(cloud) == length(X) ||
            throw(ArgumentError("null_sampler must preserve the cloud sample size"))
        nulls[b] = _max_finite_persistence(ripserer(cloud;
            dim_max=dim, ripserer_kwargs...)[dim + 1])
    end
    return max_persistence_significance(diagram, nulls; alpha=alpha)
end

significant(result::MaxPersistenceResult) = result.diagram[findall(<=(result.alpha), result.interval_pvalues)]

function Base.show(io::IO, result::MaxPersistenceResult)
    print(io, "MaxPersistenceResult(observed=", result.observed,
        ", pvalue=", result.pvalue, ", threshold=", result.threshold, ")")
end

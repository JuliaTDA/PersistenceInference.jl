"""
    IndependenceTestResult

HSIC permutation test result: `observed` is the biased HSIC V-statistic,
`pvalue` uses the Monte Carlo +1 correction, `null_distribution` stores
permuted statistics, and `n_perm` and `n_samples` record the design.
"""
struct IndependenceTestResult
    pvalue::Float64
    observed::Float64
    null_distribution::Vector{Float64}
    n_perm::Int
    n_samples::Int
end

function _validated_gram(matrix)
    n, m = size(matrix)
    n == m || throw(DimensionMismatch("Gram matrices must be square"))
    n >= 3 || throw(ArgumentError("at least three paired observations are required"))
    all(isfinite, matrix) || throw(ArgumentError("Gram matrices must be finite"))
    isapprox(matrix, transpose(matrix); atol=1e-10, rtol=1e-10) ||
        throw(ArgumentError("Gram matrices must be symmetric"))
    k = Matrix{Float64}((matrix + transpose(matrix)) / 2)
    eigmin(Symmetric(k)) >= -1e-8 * max(1.0, maximum(abs, k)) ||
        throw(ArgumentError("kernels must produce positive semidefinite Gram matrices"))
    return k
end

_center_gram(k) = k .- mean(k; dims=1) .- mean(k; dims=2) .+ mean(k)

"""
    independence_test(X, Y; kernel_x=SlicedWassersteinKernel(),
        kernel_y=SlicedWassersteinKernel(), n_perm=999, rng=Random.default_rng())
    independence_test(K::AbstractMatrix, L::AbstractMatrix; n_perm=999, rng=...)

Test independence of paired i.i.d. observations using the Hilbert-Schmidt
independence criterion (HSIC). Diagram samples use Sliced Wasserstein kernels
by default. Numeric observations may use a supplied kernel such as
`kernel_y=(x,y)->exp(-(x-y)^2/2)`. Matrix inputs are **Gram matrices**, not data
matrices. Kernels must be symmetric and positive semidefinite.

For centered Gram matrices, the statistic is `sum(Kc .* Lc)/n^2`. Permutations
jointly permute the rows and columns of L, breaking the pairing while preserving
each marginal. P-values use `(1 + count(null >= observed))/(n_perm + 1)`.
The permutation test controls size under independence/exchangeability; sensitivity
to all dependence alternatives additionally requires characteristic kernels.
No universal consistency claim is made for the default diagram kernel.
Do not permute individual points within one cloud as if they were independent
diagram observations. Time series require an appropriate block/randomization design.

Reference: Gretton et al. (2008), *A Kernel Statistical Test of Independence*,
https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html.
"""
function independence_test(K::AbstractMatrix, L::AbstractMatrix;
        n_perm::Integer=999, rng::Random.AbstractRNG=Random.default_rng())
    n_perm >= 1 || throw(ArgumentError("n_perm must be positive"))
    size(K) == size(L) || throw(DimensionMismatch("paired Gram matrices must agree"))
    kc, lc = _center_gram(_validated_gram(K)), _center_gram(_validated_gram(L))
    n = size(kc, 1)
    observed = sum(kc .* lc) / n^2
    nulls = Vector{Float64}(undef, n_perm)
    for b in 1:n_perm
        perm = randperm(rng, n)
        nulls[b] = sum(kc .* lc[perm, perm]) / n^2
    end
    pvalue = (1 + count(>=(observed), nulls)) / (n_perm + 1)
    return IndependenceTestResult(pvalue, observed, nulls, Int(n_perm), n)
end

function independence_test(X::AbstractVector, Y::AbstractVector;
        kernel_x=SlicedWassersteinKernel(), kernel_y=SlicedWassersteinKernel(), kwargs...)
    length(X) == length(Y) || throw(DimensionMismatch("paired samples must agree"))
    K = [kernel_x(x, y) for x in X, y in X]
    L = [kernel_y(x, y) for x in Y, y in Y]
    return independence_test(K, L; kwargs...)
end

function Base.show(io::IO, result::IndependenceTestResult)
    print(io, "IndependenceTestResult(n=", result.n_samples, ", HSIC=",
        result.observed, ", pvalue=", result.pvalue, ")")
end

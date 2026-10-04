"""
    PersistenceInference

Statistical tools for persistence diagrams: point-cloud bootstrap radii,
two-sample point-label permutation tests, finite-grid mean functional bands,
maximum finite persistence under explicit null models, and paired HSIC tests.

This package answers the question that turns persistent homology from a
descriptive tool into an inferential one: *which topological features are
real, and which are noise?*

# Exports

- [`bootstrap_diagram`](@ref) — bootstrap confidence sets for a persistence
  diagram, with [`significant`](@ref) to extract the features distinguishable
  from noise.
- [`permutation_test`](@ref) — permutation test comparing the topology of two
  point clouds.
- [`confidence_band`](@ref), [`multiplier_bootstrap`](@ref) — simultaneous
  finite-grid bands for population mean curves from independent diagrams/rows.
- [`max_persistence_significance`](@ref) — finite-bar maxima under a supplied
  exchangeable global null.
- [`independence_test`](@ref) — independence of paired observations through HSIC.
- Result types [`BootstrapResult`](@ref), [`PermutationTestResult`](@ref),
  [`ConfidenceBand`](@ref), [`MaxPersistenceResult`](@ref),
  [`IndependenceTestResult`](@ref).

See the README for background and references.
"""
module PersistenceInference

using Random
using Statistics: quantile, mean, std
using LinearAlgebra: Symmetric, eigmin
using TDARipserer: ripserer
using TDAPersistenceDiagrams: PersistenceDiagram, PersistenceInterval, Bottleneck, Wasserstein,
    birth, death, persistence, SlicedWassersteinKernel

export bootstrap_diagram, BootstrapResult, significant
export permutation_test, PermutationTestResult
export confidence_band, multiplier_bootstrap, ConfidenceBand
export max_persistence_significance, MaxPersistenceResult
export independence_test, IndependenceTestResult

include("bootstrap.jl")
include("permutation.jl")
include("bands.jl")
include("max_persistence.jl")
include("independence.jl")

end # module PersistenceInference

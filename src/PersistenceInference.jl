"""
    PersistenceInference

Statistical inference on persistence diagrams: confidence sets via the
bootstrap (Fasy et al. 2014) and two-sample hypothesis testing via
permutation (Robinson & Turner 2017).

This package answers the question that turns persistent homology from a
descriptive tool into an inferential one: *which topological features are
real, and which are noise?*

# Exports

- [`bootstrap_diagram`](@ref) — bootstrap confidence sets for a persistence
  diagram, with [`significant`](@ref) to extract the features distinguishable
  from noise.
- [`permutation_test`](@ref) — permutation test comparing the topology of two
  point clouds.
- Result types [`BootstrapResult`](@ref) and [`PermutationTestResult`](@ref).

See the README for background and references.
"""
module PersistenceInference

using Random
using Statistics: quantile
using Ripserer: ripserer
using PersistenceDiagrams: PersistenceDiagram, PersistenceInterval, Bottleneck, Wasserstein,
    birth, death, persistence

export bootstrap_diagram, BootstrapResult, significant
export permutation_test, PermutationTestResult

include("bootstrap.jl")
include("permutation.jl")

end # module PersistenceInference

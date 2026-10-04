# Run in an environment containing the local PersistenceInference package:
# julia --project=... validation/calibration.jl [replications] [output.csv]
using PersistenceInference, TDAPersistenceDiagrams, Random, Statistics, Test

function wilson_interval(successes, trials; z=1.959963984540054)
    p = successes / trials
    center = (p + z^2 / (2trials)) / (1 + z^2 / trials)
    half = z * sqrt(p*(1-p)/trials + z^2/(4trials^2)) / (1 + z^2/trials)
    return center - half, center + half
end

function calibration(; replications=1000, seed=20261003, n=120, n_boot=399, n_perm=199)
    rng = Xoshiro(seed)
    a = PersistenceDiagram([(0.0, 2.0)]; dim=1)
    b = PersistenceDiagram([(0.0, 1.0)]; dim=1)
    grid = collect(range(0, 2; length=21))
    landscape_truth = [0.5*(max(0,min(t,2-t)) + max(0,min(t,1-t))) for t in grid]
    betti_truth = [0.5*((0 <= t < 2) + (0 <= t < 1)) for t in grid]
    covered = Dict(:landscape=>0, :betti=>0, :gaussian=>0)
    rejected = Dict(:independence=>0, :max_persistence=>0)
    gaussian_kernel = (x,y) -> exp(-(x-y)^2 / 2)
    for _ in 1:replications
        samples = [rand(rng, Bool) ? a : b for _ in 1:n]
        for (representation, truth) in ((:landscape,landscape_truth), (:betti,betti_truth))
            band = confidence_band(samples; grid=grid, representation=representation,
                n_boot=n_boot, rng=rng)
            covered[representation] += all(band.lower .<= truth .<= band.upper)
        end
        # Known seven-dimensional N(0, Σ) process: correlated, non-rank-one grid.
        gaussian = randn(rng, n, 7) .+ randn(rng, n)
        band = multiplier_bootstrap(gaussian; n_boot=n_boot, rng=rng)
        covered[:gaussian] += all(band.lower .<= 0 .<= band.upper)
        x, y = randn(rng, 30), randn(rng, 30)
        test = independence_test(x,y; kernel_x=gaussian_kernel, kernel_y=gaussian_kernel,
            n_perm=n_perm, rng=rng)
        rejected[:independence] += test.pvalue <= 0.05
        # Exchangeable iid global-null maxima, with an exactly known Uniform(0,1) law.
        observed = PersistenceDiagram([(0.0,rand(rng))]; dim=1)
        test_max = max_persistence_significance(observed, rand(rng,199))
        rejected[:max_persistence] += test_max.pvalue <= 0.05
    end
    rows = NamedTuple[]
    for (kind, counts, target) in ((:coverage, covered, 0.95), (:type_I_error,rejected,0.05))
        for name in sort!(collect(keys(counts)))
            hits = counts[name]
            lo, hi = wilson_interval(hits,replications)
            rate = hits / replications
            # Bands allow a 2% finite-sample approximation; the exchangeable-null
            # tests use only three Monte Carlo standard errors plus resolution.
            tolerance = (kind == :coverage ? 0.02 : 1/replications) +
                3sqrt(target*(1-target)/replications)
            @test abs(rate-target) <= tolerance
            push!(rows,(kind=kind, method=name, rate=rate, target=target,
                lower=lo, upper=hi, seed=seed, replications=replications,
                n=n, n_boot=n_boot, n_perm=n_perm, tolerance=tolerance))
        end
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    replications = isempty(ARGS) ? 1000 : parse(Int,ARGS[1])
    rows = calibration(; replications=replications)
    output = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__,"calibration.csv")
    open(output,"w") do io
        println(io,join(string.(keys(first(rows))),","))
        for row in rows
            println(io,join(string.(values(row)),","))
        end
    end
    foreach(println,rows)
end

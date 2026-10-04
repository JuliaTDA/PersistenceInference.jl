using PersistenceInference, TDAPersistenceDiagrams, Random, DelimitedFiles

function export_reference(output=joinpath(@__DIR__,"reference"))
    mkpath(output)
    seed,n,B = 20261003,31,199
    rng = Xoshiro(seed)
    curves = randn(rng,n,7) .+ randn(rng,n)
    curves[:,end] .= 2 # deterministic coordinate
    writedlm(joinpath(output,"curves.csv"),curves,',')
    # The API's sequential RNG stream, shared with NumPy via raw random draws.
    draws_rng = Xoshiro(seed)
    multipliers = hcat([randn(draws_rng,n) for _ in 1:B]...)
    writedlm(joinpath(output,"multipliers.csv"),multipliers,',')
    for studentize in (false,true)
        band = multiplier_bootstrap(curves;rng=Xoshiro(seed),n_boot=B,studentize=studentize)
        prefix = studentize ? "studentized" : "multiplier"
        writedlm(joinpath(output,prefix*"_maxima.csv"),band.maxima,',')
        writedlm(joinpath(output,prefix*"_band.csv"),hcat(band.estimate,band.lower,band.upper),',')
    end
    draws_rng = Xoshiro(seed)
    indices = hcat([rand(draws_rng,1:n,n) for _ in 1:B]...)
    writedlm(joinpath(output,"bootstrap_indices.csv"),indices,',')
    band = confidence_band(curves;method=:bootstrap,rng=Xoshiro(seed),n_boot=B)
    writedlm(joinpath(output,"bootstrap_maxima.csv"),band.maxima,',')
    writedlm(joinpath(output,"bootstrap_band.csv"),hcat(band.estimate,band.lower,band.upper),',')

    x,y = randn(rng,17),randn(rng,17)
    K,L = [exp(-(a-b)^2/2) for a in x,b in x],[exp(-(a-b)^2/2) for a in y,b in y]
    writedlm(joinpath(output,"K.csv"),K,','); writedlm(joinpath(output,"L.csv"),L,',')
    draws_rng = Xoshiro(seed)
    permutations = hcat([randperm(draws_rng,17) for _ in 1:B]...)
    writedlm(joinpath(output,"permutations.csv"),permutations,',')
    result = independence_test(K,L;rng=Xoshiro(seed),n_perm=B)
    writedlm(joinpath(output,"hsic.csv"),[result.observed,result.pvalue],',')
    writedlm(joinpath(output,"hsic_nulls.csv"),result.null_distribution,',')
    nulls = rand(rng,B)
    diagram = PersistenceDiagram([(0.0,0.7)];dim=1)
    test = max_persistence_significance(diagram,nulls)
    writedlm(joinpath(output,"null_maxima.csv"),nulls,',')
    writedlm(joinpath(output,"max_test.csv"),[test.observed,test.pvalue,test.threshold],',')
    println("Exported numeric reference inputs/results to ",output," (seed ",seed,")")
end

export_reference(isempty(ARGS) ? joinpath(@__DIR__,"reference") : ARGS[1])

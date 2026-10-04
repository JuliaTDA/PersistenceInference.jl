using PersistenceInference, TDAPersistenceDiagrams, Random

rng = Xoshiro(20261003)
# Each diagram represents one independent experimental unit.
diagrams = [PersistenceDiagram([(0.0, 1.0 + rand(rng))]; dim=1) for _ in 1:80]
grid = range(0, 2; length=101)
landscape_band = confidence_band(diagrams; grid=grid, rng=rng)
betti_band = confidence_band(diagrams; grid=grid, representation=:betti, rng=rng)
resampled_band = confidence_band(diagrams; grid=grid, method=:bootstrap, rng=rng)
println(landscape_band)
println(betti_band)

# Independence between topology and an observed covariate.
covariate = [death(only(d)) + 0.05randn(rng) for d in diagrams]
dependency = independence_test(diagrams, covariate;
    kernel_y=(x,y)->exp(-(x-y)^2/0.1), rng=rng)
println(dependency)

# Known toy global null: all finite persistences are Uniform(0, 0.5).
observed = PersistenceDiagram([(0.0,1.5),(0.0,0.1)]; dim=1)
maximum_test = max_persistence_significance(observed, 0.5rand(rng,999))
println(maximum_test)
println("Significant intervals: ", significant(maximum_test))

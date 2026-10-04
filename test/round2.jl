using Statistics: mean
using LinearAlgebra: I, diag
using TDAPersistenceDiagrams: PersistenceInterval

@testset "finite-grid simultaneous confidence bands" begin
    values = [0.0 2.0 5.0; 2.0 4.0 5.0]
    a = multiplier_bootstrap(values; n_boot=20_000, rng=Xoshiro(101))
    @test a.estimate == [1.0, 3.0, 5.0]
    @test a.lower[1] - a.estimate[1] ≈ a.lower[2] - a.estimate[2]
    @test a.lower[3] == a.upper[3] == 5.0
    # In the two-observation case the standardized supremum is |N(0,1)|.
    # This verifies the centering, sqrt(n) normalization and quantile jointly.
    @test abs(a.critical_value - 1.959963984540054) < 0.045
    @test a.upper[1] - a.estimate[1] ≈ a.critical_value / sqrt(2)
    b = multiplier_bootstrap(values; n_boot=20_000, rng=Xoshiro(101))
    @test a.maxima == b.maxima
    s = multiplier_bootstrap(values; studentize=true, rng=Xoshiro(101), n_boot=20_000)
    @test s.upper ≈ a.upper
    @test s.studentized
    c = confidence_band(values; method=:bootstrap, n_boot=100, rng=Xoshiro(1))
    @test c.method == :bootstrap
    @test all(x -> x in (0.0, sqrt(2)), c.maxima)
    @test sprint(show, c) isa String

    diagrams = [PersistenceDiagram([(0.0, 2.0), (0.5, 1.5), (0.0, Inf)]; dim=1)
        for _ in 1:3]
    landscape = confidence_band(diagrams; grid=[0.0, 0.5, 1.0, 2.0], n_boot=5)
    @test landscape.estimate == [0.0, 0.5, 1.0, 0.0]
    @test landscape.lower == landscape.upper == landscape.estimate
    second = confidence_band(diagrams; grid=[0.0, 0.5, 1.0, 2.0], k=2, n_boot=5)
    @test second.estimate == [0.0, 0.0, 0.5, 0.0]
    betti = confidence_band(diagrams; grid=[0.0, 0.5, 1.0, 1.5, 2.0],
        representation=:betti, n_boot=5)
    @test betti.estimate == [2.0, 3.0, 3.0, 2.0, 1.0]
    empty_diagrams = [PersistenceDiagram(PersistenceInterval[]; dim=1) for _ in 1:2]
    @test all(iszero, confidence_band(empty_diagrams; grid=[0.0, 1.0]).upper)
    @test_throws ArgumentError confidence_band(ones(1, 2))
    @test_throws ArgumentError confidence_band(ones(2, 0))
    @test_throws ArgumentError confidence_band(values; alpha=0)
    @test_throws ArgumentError confidence_band(values; n_boot=0)
    @test_throws ArgumentError confidence_band(values; grid=[2, 1, 0])
    @test_throws ArgumentError confidence_band(values; grid=[0, 0, 1])
    @test_throws ArgumentError confidence_band(values; method=:pointwise)
    @test_throws DimensionMismatch confidence_band(values; grid=[1, 2])
    @test_throws ArgumentError confidence_band([NaN 1; 2 3])
    @test_throws ArgumentError confidence_band(diagrams; grid=[1.0], k=0)
    @test_throws ArgumentError confidence_band(diagrams; grid=[1.0], representation=:other)
    @test_throws ArgumentError confidence_band([diagrams[1],
        PersistenceDiagram([(0.0, 2.0)]; dim=0)]; grid=[1.0])
end

@testset "maximum finite persistence under a global null" begin
    observed = PersistenceDiagram([(0.0, 3.0), (0.0, 0.1), (0.0, Inf)]; dim=1)
    r = max_persistence_significance(observed, fill(0.5, 99))
    @test r.pvalue == 0.01
    @test r.observed == 3.0
    @test r.threshold == 0.5
    @test r.interval_pvalues == [0.01, 1.0, 1.0]
    @test length(significant(r)) == 1
    @test persistence(only(significant(r))) == 3.0
    @test sprint(show, r) isa String
    @test max_persistence_significance(observed, [0.5]; alpha=0.05).threshold == Inf
    tied = max_persistence_significance(observed, fill(3.0, 99))
    @test tied.pvalue == 1.0
    @test isempty(significant(tied))
    nulls = [PersistenceDiagram([(0.0, 0.5)]; dim=1) for _ in 1:99]
    @test max_persistence_significance(observed, nulls).null_maxima == r.null_maxima
    @test max_persistence_significance(PersistenceDiagram(PersistenceInterval[]),
        [0.1, 0.2]).pvalue == 1.0
    for alpha in (0.01, 0.05, 0.1, 0.5), b in (9, 19, 99), value in 0:0.5:(b + 1)
        result = max_persistence_significance(PersistenceDiagram([(0.0, value)]),
            collect(1.0:b); alpha=alpha)
        @test (result.pvalue <= alpha) == (value > result.threshold)
    end
    @test_throws ArgumentError max_persistence_significance(observed, Float64[])
    @test_throws ArgumentError max_persistence_significance(observed, [NaN])
    @test_throws ArgumentError max_persistence_significance(observed, [-1.0])
    @test_throws ArgumentError max_persistence_significance(observed, [1.0]; alpha=1)
    cloud = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
    sampler = (rng, n) -> [(rand(rng), rand(rng)) for _ in 1:n]
    a = max_persistence_significance(cloud; null_sampler=sampler, n_null=5,
        dim=0, rng=Xoshiro(15))
    b = max_persistence_significance(cloud; null_sampler=sampler, n_null=5,
        dim=0, rng=Xoshiro(15))
    @test a.null_maxima == b.null_maxima
    @test 0 < a.pvalue <= 1
    @test_throws ArgumentError max_persistence_significance(cloud;
        null_sampler=(rng,n)->cloud[1:2], n_null=1)
end

@testset "HSIC paired-sample independence" begin
    x = collect(range(-2.0, 2.0; length=30))
    kernel = (a,b) -> exp(-(a-b)^2 / 2)
    K = [kernel(a,b) for a in x, b in x]
    result = independence_test(x, x; kernel_x=kernel, kernel_y=kernel,
        n_perm=199, rng=Xoshiro(17))
    @test result.pvalue <= 0.01
    @test result.n_samples == 30
    H = Matrix{Float64}(I, 30, 30) .- 1 / 30
    # Independent matrix identity: biased HSIC = tr(K H L H) / n².
    @test result.observed ≈ sum(diag(K * H * K * H)) / 30^2
    repeated = independence_test(K, K; n_perm=199, rng=Xoshiro(17))
    @test repeated.null_distribution == result.null_distribution
    @test sprint(show, repeated) isa String
    constant = independence_test(K, ones(30,30); n_perm=9)
    @test constant.pvalue == 1.0
    @test constant.observed == 0.0
    diagrams = [PersistenceDiagram([(0.0, 0.5 + i / 30)]; dim=1) for i in 1:30]
    @test independence_test(diagrams, diagrams; n_perm=19, rng=Xoshiro(2)).pvalue <= 0.1
    @test_throws ArgumentError independence_test(K, K; n_perm=0)
    @test_throws DimensionMismatch independence_test(K, K[1:3,1:3])
    @test_throws ArgumentError independence_test(ones(2,2), ones(2,2))
    @test_throws ArgumentError independence_test([1.0 2 0; 0 1 0; 0 0 1], ones(3,3))
    @test_throws ArgumentError independence_test([1.0 2 0; 2 1 0; 0 0 1], ones(3,3))
    @test_throws ArgumentError independence_test(fill(NaN,3,3), ones(3,3))
end

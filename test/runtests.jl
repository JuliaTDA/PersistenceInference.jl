using PersistenceInference
using TDAPersistenceDiagrams: PersistenceDiagram, Bottleneck, Wasserstein, persistence
using Random
using Test

# --- helpers ----------------------------------------------------------------

"Noisy circle of `n` points (vector of 2-tuples), reproducible via `rng`."
function noisy_circle(n, σ, rng)
    θ = range(0, 2π; length=n + 1)[1:n]
    return [(cos(t) + σ * randn(rng), sin(t) + σ * randn(rng)) for t in θ]
end

"Uniform blob of `n` points in [-1, 1]^2."
uniform_blob(n, rng) = [(2rand(rng) - 1, 2rand(rng) - 1) for _ in 1:n]

# ----------------------------------------------------------------------------

@testset "PersistenceInference.jl" begin
    @testset "bootstrap_diagram on a noisy circle" begin
        rng = Random.Xoshiro(20240611)
        X = noisy_circle(50, 0.05, rng)

        result = bootstrap_diagram(X; n_boot=50, alpha=0.05, dim_max=1, rng=rng)

        @test result isa BootstrapResult
        @test result.n_boot == 50
        @test result.alpha == 0.05
        @test length(result.diagrams) == 2          # H0 and H1
        @test length(result.radius) == 2
        @test length(result.distances) == 2
        @test all(d -> length(d) == 50, result.distances)

        # The H1 radius must be a positive, finite confidence half-width.
        @test result.radius[2] > 0
        @test isfinite(result.radius[2])

        # The circle has exactly one significant loop at α = 0.05.
        sig_h1 = significant(result, 1)
        @test length(sig_h1) == 1
        @test persistence(sig_h1[1]) > 2 * result.radius[2]

        # `significant` returns a subset of the reference diagram in that dim.
        @test all(int -> int in result.diagrams[2], sig_h1)

        # show methods don't error
        @test sprint(show, result) isa String
        @test sprint((io, x) -> show(io, MIME("text/plain"), x), result) isa String
    end

    @testset "bootstrap reproducibility and subsample_size" begin
        X = noisy_circle(50, 0.05, Random.Xoshiro(1))

        r1 = bootstrap_diagram(X; n_boot=30, dim_max=1, rng=Random.Xoshiro(7))
        r2 = bootstrap_diagram(X; n_boot=30, dim_max=1, rng=Random.Xoshiro(7))
        @test r1.radius == r2.radius
        @test r1.distances == r2.distances

        # subsample_size kwarg works (m-out-of-n bootstrap).
        rsub = bootstrap_diagram(X; n_boot=30, dim_max=1, subsample_size=30,
            rng=Random.Xoshiro(7))
        @test rsub isa BootstrapResult
        @test rsub.radius[2] > 0
        @test all(d -> length(d) == 30, rsub.distances)
    end

    @testset "bootstrap with Wasserstein distance" begin
        X = noisy_circle(40, 0.05, Random.Xoshiro(3))
        result = bootstrap_diagram(X; n_boot=30, dim_max=1, distance=Wasserstein(2),
            rng=Random.Xoshiro(3))
        @test result isa BootstrapResult
        @test all(isfinite, result.radius)
    end

    @testset "significant() on a hand-built case" begin
        # Reference H1 diagram: one tiny (noise) feature, one big (real) loop.
        diag0 = PersistenceDiagram([(0.0, Inf)]; dim=0)
        diag1 = PersistenceDiagram([(0.1, 0.2), (0.5, 2.5)]; dim=1)

        # radius 0.3 => threshold 2c = 0.6. Persistences are 0.1 and 2.0.
        result = BootstrapResult([diag0, diag1], [0.0, 0.3],
            [Float64[], Float64[]], 0.05, 0)

        sig = significant(result, 1)
        @test length(sig) == 1
        @test persistence(sig[1]) == 2.0

        # H0: the infinite component is always significant.
        sig0 = significant(result, 0)
        @test length(sig0) == 1
        @test !isfinite(sig0[1])

        # Requesting an uncomputed dimension errors.
        @test_throws ArgumentError significant(result, 5)
    end

    @testset "bootstrap argument validation" begin
        X = noisy_circle(20, 0.05, Random.Xoshiro(5))
        @test_throws ArgumentError bootstrap_diagram(X; n_boot=0)
        @test_throws ArgumentError bootstrap_diagram(X; alpha=0.0)
        @test_throws ArgumentError bootstrap_diagram(X; alpha=1.0)
        @test_throws ArgumentError bootstrap_diagram(X; dim_max=-1)
        @test_throws ArgumentError bootstrap_diagram(X; subsample_size=0)
        @test_throws ArgumentError bootstrap_diagram(eltype(X)[])
    end

    @testset "permutation_test: same distribution => large p-value" begin
        # Two noisy circles from the same generating process, different seeds.
        X = noisy_circle(30, 0.05, Random.Xoshiro(11))
        Y = noisy_circle(30, 0.05, Random.Xoshiro(22))

        result = permutation_test(X, Y; n_perm=99, dim=1, rng=Random.Xoshiro(99))

        @test result isa PermutationTestResult
        @test result.dim == 1
        @test result.n_perm == 99
        @test length(result.null_distribution) == 99
        @test 0 < result.pvalue ≤ 1
        # Same distribution: should not reject. Use a loose bound to avoid flakiness.
        @test result.pvalue > 0.01

        @test sprint(show, result) isa String
        @test sprint((io, x) -> show(io, MIME("text/plain"), x), result) isa String
    end

    @testset "permutation_test: circle vs blob => small p-value" begin
        X = noisy_circle(30, 0.05, Random.Xoshiro(101))
        Y = uniform_blob(30, Random.Xoshiro(202))

        result = permutation_test(X, Y; n_perm=99, dim=1, rng=Random.Xoshiro(303))

        @test 0 < result.pvalue ≤ 1
        @test result.pvalue ≤ 0.05
        @test result.observed > 0
    end

    @testset "permutation_test reproducibility and bounds" begin
        X = noisy_circle(25, 0.05, Random.Xoshiro(1))
        Y = uniform_blob(25, Random.Xoshiro(2))

        r1 = permutation_test(X, Y; n_perm=50, dim=1, rng=Random.Xoshiro(42))
        r2 = permutation_test(X, Y; n_perm=50, dim=1, rng=Random.Xoshiro(42))
        @test r1.pvalue == r2.pvalue
        @test r1.observed == r2.observed
        @test r1.null_distribution == r2.null_distribution

        # p-value is always in (0, 1].
        @test 0 < r1.pvalue ≤ 1
    end

    @testset "permutation_test argument validation" begin
        X = noisy_circle(15, 0.05, Random.Xoshiro(1))
        Y = noisy_circle(15, 0.05, Random.Xoshiro(2))
        @test_throws ArgumentError permutation_test(X, Y; n_perm=0)
        @test_throws ArgumentError permutation_test(X, Y; dim=-1)
        @test_throws ArgumentError permutation_test(X, Y; q=0)
        @test_throws ArgumentError permutation_test(X, Y; dim=2, dim_max=1)
        @test_throws ArgumentError permutation_test(eltype(X)[], Y)
    end
end

module TestDP

using Test
include("../src/dp.jl")
@testset "same result from two versions" begin
    n = 10; zoom = 2; f = 10; σ = 1;
    X, M = gen_zoomed_data(n*zoom^2, f, σ = σ, zoom=zoom)
    res = bottom_up_match_optim(X, method="prior_max", σ = σ,
                                case_study=false, knearest=false, δ = 1);
    Ms = bottom_up_match_optim2(X, method="prior_max", σ = σ, δ = 1);
    Mshat = bottom_up_match_optim2(X, fill(σ*1.0, f-1), method="prior_max", δ = 1);
    @test res == Ms
    @test Ms == Mshat
end

@testset "orderIdx to exchange Idx" begin
    @test ex_idx(4,4,4) == 7
    @test ex_idx(3,4,4) == 6
    @test inv_exIdx(7,4) == (1,1)
    @test inv_exIdx(6,4) == (3,4)
end

end

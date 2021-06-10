module TestCost

using Test
include("../src/dp.jl")

@testset "custom width and height" begin
    @test dist_boundary([1024, 1024], width = 1024, height = 1024) == 0
    @test_throws ErrorException dist_boundary([1024, 1024])

    @test_throws ErrorException pr_link_disappear([1020,1020])
    @test pr_link_disappear([1000,1000], width = 1024, height = 1024) >
          pr_link_disappear([1020,1020], width = 1024, height = 1024)

    @test_throws ErrorException pr_link_disappear_max2_bvn2([1020, 1020])
    @test pr_link_disappear_max2_bvn2([1000, 1000], width = 1024, height = 1024) >
          pr_link_disappear_max2_bvn2([1020, 1020], width = 1024, height = 1024)

    @test_throws ErrorException pr_link_x3_remain_x2_appeared_priormax([1020, 1020], [1010, 1010], σ = 1)
    @test pr_link_x3_remain_x2_appeared_priormax([1020, 1020], [910, 910], σ = 1, width = 1024, height = 1024) >
          pr_link_x3_remain_x2_appeared_priormax([1020, 1020], [1010, 1010], σ = 1, width = 1024, height = 1024)

    @test_throws ErrorException pr_link_disappear_x2_hybrid([1020, 1020])
    @test pr_link_disappear_x2_hybrid([1020, 1020], width = 1024, height = 1024) > 0
end

@testset "pr link works" begin
    @test pr_link_disappear([1, 1]) ≈ pr_link_appear([1, 1])
    @test pr_link_disappear_IS([1, 1]; ns=100, f2i=false) >= 0
    @test pr_link_disappear_IS([1, 1]; ns=100, f2i=true) >= 0
    @test pr_link_disappear_IS([1, 1], [1, 1]; ns=1000, f2i=true) ≈
          pr_link_disappear_IS([1, 1]; ns=1000, f2i=true) atol = 0.5
    @test pr_link_disappear_IS([1, 1], [1, 1]; ns=1000, f2i=false) ≈
          pr_link_disappear_IS([1, 1]; ns=1000, f2i=false) atol = 0.5

    @test pr_link_disappear_mc([1, 1]; ns=100, f2i=true) >= 0
    @test pr_link_disappear_mc([1, 1]; ns=100, f2i=false) >= 0
    @test pr_link_disappear_mc([1, 1]; ns=1000, f2i=true) ≈
          pr_link_disappear_mc([1, 1]; ns=1000, f2i=true) atol = 0.5
    @test pr_link_disappear_mc([1, 1]; ns=1000, f2i=false) ≈
          pr_link_disappear_mc([1, 1]; ns=1000, f2i=false) atol = 0.5

    @test pr_link_disappear_bvn([3, 4], [3, 4]) == pr_link_disappear_bvn([3, 4])
    @test pr_link_disappear_bvn2([3, 4], [3, 4]) == pr_link_disappear_bvn2([3, 4])
    @test pr_link_disappear_bvn([3, 4]) ≈ pr_link_disappear_bvn2([3, 4])
end

# using Serialization
# # https://discourse.julialang.org/t/how-to-get-the-path-of-a-julia-script/1728/2
# lstX, lstM=deserialize("$(@__DIR__)/../src/oracle_setting_2019-09-24T22:50:10/lstXM_7_50_10_1.sil")
# mincost(lstX[1][3], lstX[1][4])
# length(varmincost_D2A(lstX[1][3], lstX[1][4], nmax = 0))
# length(varmincost_D2A(lstX[1][3], lstX[1][4], nmax = 1))
# length(varmincost_D2A(lstX[1][3], lstX[1][4], nmax = 2))

@testset "uniform motion" begin
    @test !isnan(calc_hv_optim([2, 1], [1, 2], [[10, 10], [100, 100]], [[101,101],[9,11]], [[102,102],[8,12]],σ=0))
    @test calc_hv_optim([2, 1], [1, 2], [[10, 10], [100, 100]], [[101,101],[9,11]], [[102,102],[8,12]],σ=0) <
            calc_hv_optim([2, 1], [2, 1], [[10, 10], [100, 100]], [[101,101],[9,11]], [[102,102],[8,12]],σ=0)
    @test calc_hv_optim([1, 2], [1, 2], [[10, 10], [100, 100]], [[101,101],[9,11]], [[102,102],[8,12]], 2, 1, 1, 1,σ=0) ==
            calc_hv_optim([2, 1], [1, 2], [[10, 10], [100, 100]], [[101,101],[9,11]], [[102,102],[8,12]],σ=0)
end

@testset "first two frames work" begin
    @test calc_h([1], [[10, 10]], [[9, 11]], σ = 1, method = "prior_max") ≈ 2.8378770664093453
    @test calc_h([1], [[10, 10]], [[9, 11]], 1, 1, σ = 1, method = "prior_max") ≈ 2.8378770664093453
    @test calc_h(Int64[], Array{Int64, 1}[], [[10, 10]], σ = 1, method = "prior_max") ≈ -logpdf(MvNormal([10,10], 1), [10, 1]) ≈ 42.33787706640935
    @test calc_h([2, 1], [[10, 10], [20, 20]], [[9, 11], [22, 22]], 2, 1, σ = 1, method = "prior_max") ==
            calc_h([2, 1], [[10, 10], [20, 20]], [[9, 11], [22, 22]], 2, 1, σ = 1, method = "prior_max")
end

@testset "myfunctions work" begin
    @test neglogpdf_bimvn([1, 2], [0, 0], 1) == -logpdf(MvNormal([0, 0], 1), [1, 2])
end

@testset "diff cost works" begin
    @test calc_hv_optim([2, 1], [2, 1], [[10, 10], [100, 100]], [[101,101],[9,11]], [[102,102],[8,12]],σ=1, method = "prior_max") ==
        calc_hv_optim([2, 1], [1, 2], [[10, 10], [100, 100]], [[101,101],[9,11]], [[102,102],[8,12]],σ=1, method = "prior_max") +
        diff_hv([[100, 100], [10, 10]], [[101,101],[9,11]], [[102,102],[8,12]],σ=1, method = "prior_max")
    @test calc_hv_optim([2, 1], [1, 3, 2], [[10, 11], [100, 100]], [[101,101],[9,11],[670, 510]], [[102,102],[8,12], [669,509]],σ=1, method = "prior_max") ≈
        calc_hv_optim([2, 1], [1, 2, 3], [[10, 11], [100, 100]], [[101,101],[9,11],[670, 510]], [[102,102],[8,12], [669,509]],σ=1, method = "prior_max") +
        diff_hv([10, 11], [[9,11],[670, 510]], [[8,12], [669,509]], σ=1)
    @test calc_hv_optim([1], [1, 3, 2], [[100, 100]], [[101,101],[9,11],[670, 510]], [[102,102],[8,12], [669,509]],σ=1, method = "prior_max") ≈
        calc_hv_optim([1], [1, 2, 3], [[100, 100]], [[101,101],[9,11],[670, 510]], [[102,102],[8,12], [669,509]],σ=1, method = "prior_max") +
        diff_hv([[9,11],[670, 510]], [[8,12], [669,509]], σ=1)
    @test calc_h([1, 2], [[10, 10], [100, 100]], [[101,101],[9,11]],σ=1, method = "prior_max") ==
        calc_h([2, 1], [[10, 10], [100, 100]], [[101,101],[9,11]],σ=1, method = "prior_max") +
        diff_h([[10, 10], [100, 100]], [[9,11], [101,101]])
end

end

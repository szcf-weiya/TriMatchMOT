module TestMincost

using Test
include("../src/dp.jl")

@testset "custom width and height" begin
    @test mincost2([[1020, 1020]], [[10, 10]], -1, width = 1024, height = 1024) == [-1]
    @test_throws ErrorException mincost2([[1020, 1020]], [[1000, 1000]], -1)
end

@testset "min cost flow for position" begin
    @test mincost2([[10, 10], [100,100]], [[101,101], [9,11]], -1) == [2, 1]
    @test mincost2([[10, 10], [100,100]], [[101,101], [9,11]], 2) == [-1, -1]
    @test_throws ErrorException mincost2([[10, 10], [100,100]], [[101,101], [9,11]], 3)
    @test match_by_mincost([[[10,10], [100,100]],
                            [[101,101], [9,11]],
                            [[102,102], [8,11]]], method2=true) == [[2, 1], [1, 2]]
end


@testset "estimate sigma from paths matched by position" begin
    @test estimate_sigma([[[10,10], [100,100]],
                          [[101,101], [9,11]],
                          [[102,102], [8,11]]],
                         [[1, 2, 2], [2, 1, 1]]) == ([0.5], [2])
    # TODO: what if
    @test estimate_sigma([[[10,10], [100,100]],
                          [[101,101], [9,11]],
                          [[102,102], [8,12]]],
                         [[1, 2, 2], [2, 1, 1]]) == ([0], [2])
    @test estimate_sigma([[[10,10], [100,100]],
                           [[101,101], [9,11]],
                           [[102,102], [8,11]]],
                          [[1, 2, 2], [2, 1, 1]], include_first_two = true) == ([1, 0.5], [2, 2])
    # TODO: what if
    @test estimate_sigma([[[10,10], [100,100]],
                           [[101,101], [9,11]],
                           [[102,102], [8,12]]],
                          [[1, 2, 2], [2, 1, 1]], include_first_two = true) == ([1, 0], [2, 2])

end

end

module TestData

using Test
include("../src/data.jl")
using .Data

@testset "id2match works" begin
    id1 = [-1, -1, 1, 2, -1]
    id2 = [-1, 1, -1, 2, -1]
    @test Data.id2match(id1, id2) == [-1, 2]
    id1 = [-1, -1, 2, 1, -1]
    @test Data.id2match(id1, id2) == [2, -1]
end

@testset "reflection works" begin
    x1 = [10, 10]
    x2 = [-5, 10]
    @test Data.reflect(x1, x2) == [7, 10]
    x2 = [-5, 4]
    @test Data.reflect(x1, x2) == [7, 4]
    x2 = [-5, -3]
    @test Data.reflect(x1, x2) == [7, 5]
    x2 = [-5, -6]
    @test Data.reflect(x1, x2) == [7, 8]
    @test_throws ErrorException Data.reflect(x2, x1)
    @test Data.reflect([10, 10], [-680, 10], allow_continue_reflection = true) == [678, 10]
    @test_throws ErrorException Data.reflect([10, 10], [-680, 10], allow_continue_reflection = false)
end

end

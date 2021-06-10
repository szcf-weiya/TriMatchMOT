module TestMeasure

using Test
include("../src/measure.jl")

@testset "accuracy works" begin
    @test acc2fβ(0.5, 0.5) == 0.5
    @test acc2fβ([0.5 0.5]) == [0.5]
end

@testset "compare matching vectors" begin
    @test cpr_match([[1, 2]], [[1, 3]], acc = false) == [1]
    @test cpr_match([[1, 2]], [[1, 3]], acc = true) == [0.5]

    # count the appears
    @test cpr_match([[1, 2]], [[1, 3]], [2, 2], acc = false) == [1]
    @test cpr_match([[1, 2]], [[1, 3]], [2, 2], acc = true) == [0.5]
    @test cpr_match([[1, 2]], [[1, 3]], [2, 3], acc = false) == [2]
    @test cpr_match([[1, 2]], [[1, 3]], [2, 3], acc = true) == [2/3]
end

@testset "recover path" begin
    @test recover_path([[2, 1], [1, 2]], [2,2,2]) == [[1, 2, 2], [2, 1, 1]]
end

@testset "accuracy" begin
    @test calc_path_accuracy([[2, 1], [1, 2]], [[2, 1], [1, 2]], [2, 2, 2]) == [1.0, 1.0]
    @test calc_path_accuracy([[2, 1], [1, 2]], [[2, 1], [2, 1]], [2, 2, 2]) == [0.0, 0.0]
end

@testset "cummulative accuracy" begin
    @test calc_path_accuracy_point([[2, 1], [1, 2]], [[2, 1], [1, 2]], [2, 2, 2]) == [1.0 1.0; 1.0 1.0; 1.0 1.0]
    @test calc_path_accuracy_point([[2, 1], [1, 2]], [[2, 1], [2, 1]], [2, 2, 2]) == [1.0 1.0; 1.0 1.0; 0.0 0.0]
end

end

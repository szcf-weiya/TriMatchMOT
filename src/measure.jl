"""
    acc2fβ(precision, recall)

Convert precision and recall accuracy to [Fβ score](https://en.wikipedia.org/wiki/F1_score).
"""
function acc2fβ(precision::Float64, recall::Float64; β = 1)
    return 2 / (1 / precision + β^2 / recall)
end

"""
    acc2fβ(acc)

Convert the accuracy array to F1 score, the order of the columns of `acc` should be precision and recall. If `β = 1`, no difference.
"""
function acc2fβ(acc::Array{Float64, 2}; β = 1)
    return acc2fβ.(acc[:, 1], acc[:, 2])
end

"""
    recover_path(M, ns)

Convert the matching vectors `M` to the paths for each cell given the number of cells in each frame, `ns`.
"""
function recover_path(M::Array{Array{Int64, 1}, 1}, nums_to_match::Array{Int64, 1})
    f = length(M)
    # path = Array{Array{Int64,1},1}(undef, 0)
    path = [[i] for i = 1:length(M[1])]
    for i = 1:f
        for j = 1:length(path)
            if path[j][end] == -1
                append!(path[j], -1)
            else
                append!(path[j], M[i][ path[j][end] ])
            end
        end
        # appeared
        id_appeared = setdiff(1:nums_to_match[i+1], M[i])
        for k in id_appeared
            push!(path, append!(-1*ones(Int, i), k))
        end
    end
    return path
end

# path accuracy
"""
    calc_path_accuracy(M1, M2, n)

Measure the matching accuracy by comparing paths based on the matching `M1` with the ground truth `M2`, and `n` is the number of cells in each frame.
"""
function calc_path_accuracy(M1::Array{Array{Int64,1},1}, M2::Array{Array{Int64,1},1}, nums_to_match::Array{Int64,1})
    path1 = recover_path(M1, nums_to_match)
    path2 = recover_path(M2, nums_to_match)
    # complete accuracy
    return length(intersect(path1, path2)) ./ [length(path1), length(path2)]
end

function calc_path_accuracy(path1::Array{Int64, 2}, M2::Array{Array{Int64, 1}, 1}, nums_to_match::Array{Int64, 1})
    if 0 in path1
        # index from 0
        path1[path1 .!= -1] .+= 1
    end
    path1 = mapslices(x->[x], path1, dims=2)[:]
    path2 = recover_path(M2, nums_to_match)
    return length(intersect(path1, path2)) ./ [length(path1), length(path2)]
end

"""
    calc_path_accuracy_point(M1, M2, n)

Measure the accumulative accuracy (precision and recall) by comparing paths based on the matching `M1` with the ground truth `M2`, and `n` is the number of cells in each frame.
"""
function calc_path_accuracy_point(M1::Array{Array{Int64,1},1}, M2::Array{Array{Int64,1},1}, nums_to_match::Array{Int64,1})
    path1 = recover_path(M1, nums_to_match)
    path2 = recover_path(M2, nums_to_match)
    # cumsum accuracy
    # f = length(path1[1]) - 1
    return calc_path_accuracy_point(path1, path2)
end

function calc_path_accuracy_point(path1::Array{Array{Int64,1},1}, path2::Array{Array{Int64,1},1})
    f = length(path1[1])
    acc = zeros(f, 2)
    # for i = 2:length(path1[1]) # the matching starts from the second frame
    for i = 1:length(path1[1])
        subpath1 = [x[1:i] for x in path1]
        idx1 = [x[1:i] != fill(-1, i) for x in path1]
        subpath2 = [x[1:i] for x in path2]
        idx2 = [x[1:i] != fill(-1, i) for x in path2]
        # exclude [-1, -1, -1]
        subpath1 = subpath1[idx1]
        subpath2 = subpath2[idx2]
        # 0/0 = NaN
        acc[i, :] = length(intersect(subpath1, subpath2)) ./ [length(subpath1), length(subpath2)]
    end
    acc[isnan.(acc)] .= 0.0
    return acc
end

function calc_path_accuracy_point(path1::Array{Int64, 2}, M2::Array{Array{Int64,1},1}, nums_to_match::Array{Int64,1})
    if 0 in path1
        # index from 0
        path1[path1 .!= -1] .+= 1
    end
    # each row is a path
    path1 = mapslices(x->[x], path1, dims=2)[:]
    path2 = recover_path(M2, nums_to_match)
    return calc_path_accuracy_point(path1, path2)
end

# path got from KTH-SE
function calc_path_accuracy_point(path1::Array{Float64, 2}, M2::Array{Array{Int64,1},1}, nums_to_match::Array{Int64,1})
    # transpose (each col is a path -> each row is a path) & convert to int
    path1 = Array{Int, 2}(path1')
    # path1 = Array{Int, 2}(path1[:, sortperm(path1[1,:])]') # actually no need to reorder
    # convert 0 (matlab type for disappear/appear) to -1
    path1[path1 .== 0] .= -1
    return calc_path_accuracy_point(path1, M2, nums_to_match)
end

function calc_path_accuracy(path1::Array{Float64, 2}, M2::Array{Array{Int64,1},1}, nums_to_match::Array{Int64,1})
    # transpose (each col is a path -> each row is a path) & convert to int
    path1 = Array{Int, 2}(path1')
    # path1 = Array{Int, 2}(path1[:, sortperm(path1[1,:])]') # actually no need to reorder
    # convert 0 (matlab type for disappear/appear) to -1
    path1[path1 .== 0] .= -1
    return calc_path_accuracy(path1, M2, nums_to_match)
end

"""
    cpr_match(M1, M2; acc = true)
    cpr_match(M1, M2, nums_to_match; acc = true)

Compare two matching vectors. If `nums_to_match` is given, the appearing objects are also
considerd. If `acc = true`, the accuracy would be calculated, otherwise, the counts of
differences.
"""
function cpr_match(M1::Array{Array{Int64,1},1}, M2::Array{Array{Int64,1},1}; acc = true)
    n1 = length(M1)
    err = zeros(Int, n1)
    # suppose n1 = n2
    for i = 1:n1
        err[i] = length(M1[i]) - sum(M1[i] .== M2[i])
    end
    if acc
        return err ./ length.(M1)
    else
        return err
    end
end

# consider the mismatches of newcomers
function cpr_match(M1::Array{Array{Int64,1},1}, M2::Array{Array{Int64,1},1}, nums_to_match::Array{Int64,1}; acc = true)
    n = length(M1)
    err = zeros(Int, n)
    for i = 1:n
        pair1 = [[j, M1[i][j]] for j = 1:length(M1[i])]
        pair2 = [[j, M2[i][j]] for j = 1:length(M1[i])]
        # number of cell at frame i+1
        n_cell = nums_to_match[i+1]
        # appeared at frame i+1 for M1
        id_appeared1 = setdiff(1:n_cell, M1[i])
        for k in id_appeared1
            push!(pair1, [-1, k])
        end
        # appeared at frame i+1 for M2
        id_appeared2 = setdiff(1:n_cell, M2[i])
        for k in id_appeared2
            push!(pair2, [-1, k])
        end
        err[i] = length(setdiff(pair1, pair2)) #/ length(pair1)
    end
    if acc
        return err ./ nums_to_match[2:end]
    else
        return err
    end
end

function final_acc(folder, n = 50, σ = 1; nrep = 100)
    acc = zeros(nrep)
    for i = 1:nrep
        acc[i] = readdlm(folder * "/$(i)_acc_$(n)_$(σ).txt", Float64)[end]
    end
    return sum(acc) / nrep
end

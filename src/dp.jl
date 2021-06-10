using Distributions
using Combinatorics
using TimerOutputs
const to = TimerOutput()
try
    includet("data.jl")
    includet("cost.jl")
catch
    # work if `julia -L`
    include("data.jl")
    include("cost.jl")
end
using .Data

include("measure.jl")


# truth coverage
"""
    truth_coverage(X, M)

If `method2 = true`, it is sigma invariant, and it decomposes the space by the number of disappeared cells (#80), where the mincost is restricted to distance method.
"""
function truth_coverage(X::Array{Array{Array{Int64, 1}, 1}, 1}, M::Array{Array{Int64,1},1}; kw...)
    # number of frames
    f = length(X)
    ans = zeros(Bool, f-1)
    for i = 1:f-1
        D = varmincost(X[i], X[i+1]; kw...)
        if M[i] in D
            ans[i] = 1
        end
    end
    return ans
end

# allow disappear and appear
function bottom_up_match_optim(X::Array{Array{Array{Int64, 1}, 1}, 1}; case_study=false, kw...)
    # number of frames
    f = length(X)
    # store minimum
    m = Array{Array{Float64,1},1}(undef, f-1)
    # skip m[1]
    D = Array{Array{Array{Int64,1},1},1}(undef, f-1)
    D[1] = varmincost2(X[1], X[2]; kw...)
    # store the minimum value
    #println("length of D[1]: $(length(D[1]))")
    m[1] = zeros(length(D[1]))
    for i = 1:length(m[1])
        m[1][i] = calc_h(D[1][i], X[1], X[2]; kw...)
    end
    # store the index of argmin
    index = Array{Array{Int64, 1}, 1}(undef, f-1)
    for t=2:f-1
        D[t] = varmincost2(X[t], X[t+1]; kw...)
        #println("length of D[$t]: $(length(D[t]))")
        m[t] = zeros(length(D[t]))
        index[t] = zeros(Int64, length(D[t]))
        for k=1:length(D[t]) # for each x in Dt
            q = Inf
            for i=1:length(D[t-1]) # traverse M_{t-1,t}
                curr_q = m[t-1][i] + calc_hv_optim(D[t-1][i], D[t][k], X[t-1], X[t], X[t+1]; kw...)
                # if isnan(curr_q)
                #     println("NaN")
                # elseif isinf(curr_q)
                #     println("Inf")
                # end
                if q > curr_q
                    q = curr_q
                    index[t][k] = i
                end
                # if (k == 22) && (t == 30)
                #     println(curr_q)
                # end
            end
            m[t][k] = q
        end
    end
    # return m, index
    res = Array{Array{Int64, 1}, 1}(undef, f-1)
    val, idx = findmin(m[f-1])
    res[f-1] = D[f-1][idx]
    for i = length(index):-1:2
        # at time i, the idx-th element = previous argmin such that the current idx-th element minimized
        idx = index[i][idx]
        res[i-1] = D[i-1][idx]
    end
    if case_study
        return res, D
    else
        return res
    end
end

function ex_idx(i::Int, j::Int, n::Int)
    idx = Int( (2n - i) / 2 * (i - 1) + (j - i) )
    if i == j == n
        idx += 1
    end
    return idx
end
function inv_exIdx(k::Int, n::Int)
    if k > binomial(n, 2) # for M_star
        return 1, 1
    end
    i = n - ceil(Int, (sqrt(4n^2-4n+9-8k) - 1) / 2)
    # the orderIdx of the first element on the i-th line
    k0 = Int((2n - i) * (i - 1) / 2 + 1)
    j = k - k0 + i + 1
    return i, j
end
function cal_offset(M::Array{Int, 1})
    where_d = M .== -1
    return cumsum(where_d)[BitArray(1 .- where_d)]
end

# allow disappear and appear
function bottom_up_match_optim2(X::Array{Array{Array{Int64, 1}, 1}, 1}; history = false, δ::Int = 1, maxdistance = Inf, width = 680, height = 512, kw...)
    @timeit to "declaration" begin
        # number of frames
        f = length(X)
        # store minimum
        m = Array{Array{Float64,1},2}(undef, f-1, 2δ+1)
        # store the index of argmin (orderIdx, num_disappear)
        idx_argmin = Array{Array{Array{Int, 1}, 1}, 2}(undef, f-1, 2δ+1)
        # skip m[1]
        # store m_star
        M_star = Array{Array{Int, 1}, 2}(undef, f-1, 2δ+1)
        offsets = Array{Array{Int, 1}, 2}(undef, f-1, 2δ+1)
        prev_npos_star = 0 # no use but to avoid UndefVarError
        # store cost0
        lst_cost0 = Array{Array{Float64, 1}, 1}(undef, 2δ+1)
        lst_X3_remain_X2_appeared = Array{Array{Set{Int64}, 1}, 1}(undef, 2δ+1)
        lst_X3_remain_X2_remain = Array{Array{Set{Int64}, 1}, 1}(undef, 2δ+1)
        lst_X2_remain = Array{Array{Dict{Int, Int}, 1}, 1}(undef, 2δ+1)
    end
    @timeit to "main" begin
        for t = 1:f-1
            # store the mincost vector
            @timeit to "mincost" M_star[t, δ+1] = mincost2(X[t], X[t+1], -1; maxdistance = maxdistance, width = width, height = height)
            d_star = sum(M_star[t, δ+1] .== -1)
            n1 = length(X[t]); n2 = length(X[t+1])
            npos_star = n1 - d_star
            for d = max(0, d_star - δ, n1 - n2):min(d_star + δ, n1)
                @timeit to "mincost-d" begin
                    id = δ + 1 + d - d_star
                    if d != d_star
                        M_star[t, id] = mincost2(X[t], X[t+1], d; maxdistance = maxdistance, width = width, height = height)
                    end
                    # offset of the index between raw idx and the sub idx
                    offsets[t, id] = cal_offset(M_star[t, id])
                    # number of non-disappeared, i.e., positive elements
                    npos = npos_star - (d - d_star)
                    # the size of the reduced space
                    nstate = binomial(npos, 2) + 1 #plus the one mincost
                    m[t, id] = ones(nstate)*Inf
                    idx_argmin[t, id] = Array{Array{Int, 1}, 1}(undef, nstate)
                end
                @timeit to "DP search" begin
                    if t == 1
                        @timeit to "Mstar cost" m[t, id][nstate] = calc_h(M_star[t, id], X[t], X[t+1]; width = width, height = height, kw...)
                        @timeit to "search" begin
                            for i1 = 1:npos-1
                                for i2 = i1+1:npos
                                    i = ex_idx(i1, i2, npos)
                                    i1r = i1 + offsets[t, id][i1]
                                    i2r = i2 + offsets[t, id][i2]
                                    @timeit to "add diff" m[t, id][i] = m[t, id][nstate] + diff_h(X[t][[i1r, i2r]], X[t+1][ M_star[t, id][[i1r, i2r]] ]; kw...)
                                    #m[t, id][i] = calc_h(M_star[t, id], X[t], X[t+1], i1 + offsets[t, id][i1], i2 + offsets[t, id][i2]; kw...)
                                end
                            end
                        end
                    else
                        @timeit to "Mstar cost" begin
                            for k = 1:2δ+1
                                if !isassigned(M_star, (f-1)*(k-1) + (t-1) )
                                    continue
                                end
                                prev_npos = prev_npos_star - (k - (δ + 1))
                                prev_nstate = binomial(prev_npos, 2) + 1
                                lst_cost0[k] = zeros(prev_nstate)
                                lst_X3_remain_X2_appeared[k] = Array{Set{Int64}, 1}(undef, prev_nstate)
                                lst_X3_remain_X2_remain[k] = Array{Set{Int64}, 1}(undef, prev_nstate)
                                lst_X2_remain[k] = Array{Dict{Int, Int}, 1}(undef, prev_nstate)
                                for j1 = 1:prev_npos
                                    for j2 = j1:prev_npos
                                        if j1 < prev_npos && j2 == j1
                                            continue
                                        end
                                        j = ex_idx(j1, j2, prev_npos)
                                        # cost for M23_star and given M12
                                        lst_cost0[k][j], lst_X3_remain_X2_appeared[k][j], lst_X3_remain_X2_remain[k][j], lst_X2_remain[k][j] = calc_hv_optim(M_star[t-1, k], M_star[t, id], X[t-1], X[t], X[t+1], j1 + offsets[t-1, k][j1], j2 + offsets[t-1, k][j2]; width = width, height = height, kw...)
                                    end
                                end
                            end
                        end
                        @timeit to "search" begin
                            for i1 = 1:npos # npos-1 is enough, but to include the M_star
                                for i2 = i1:npos # i1+1 is enough, but to include the M_star
                                    if i1 < npos && i2 == i1
                                        continue
                                    end
                                    i = ex_idx(i1, i2, npos)
                                    # if i1 == npos # for M_star
                                    #     i += 1 # since ex_idx(npos, npos, npos) = ex_idx(npos-1, npos, npos)
                                    # end
                                    i1r = i1 + offsets[t, id][i1]
                                    i2r = i2 + offsets[t, id][i2]
                                    q = Inf
                                    # find in the space unioned by all d
                                    for k = 1:2δ+1
                                        if !isassigned(M_star, (f-1)*(k-1) + (t-1) )
                                            continue
                                        end
                                        prev_npos = prev_npos_star - (k - (δ + 1))
                                        for j1 = 1:prev_npos
                                            for j2 = j1:prev_npos
                                                if j1 < prev_npos && j2 == j1
                                                    continue
                                                end
                                                j = ex_idx(j1, j2, prev_npos)
                                                # cost for M23_star and given M12
            #                                    cost0, X3_remain_X2_appeared, X3_remain_X2_remain, X2_remain = calc_hv_optim(M_star[t-1, k], M_star[t, id], X[t-1], X[t], X[t+1], j1 + offsets[t-1, k][j1], j2 + offsets[t-1, k][j2]; kw...)
                                                # if j1 == prev_npos
                                                #     j += 1 # for M_star
                                                # end
            #                                    curr_q = m[t-1, k][j] + calc_hv_optim(M_star[t-1, k], M_star[t, id], X[t-1], X[t], X[t+1], j1 + offsets[t-1, k][j1], j2 + offsets[t-1, k][j2], i1 + offsets[t, id][i1], i2 + offsets[t, id][i2]; kw...)
                                                curr_q = Inf
                                                if i1 == npos # M23_star
                                                    @timeit to "add diff" curr_q = m[t-1, k][j] + lst_cost0[k][j]
                                                else
                                                    # raw idx by adding back the offset
                                                    if i1r in lst_X3_remain_X2_remain[k][j]
                                                        prev_i1 = lst_X2_remain[k][j][i1r]
                                                        if i2r in lst_X3_remain_X2_remain[k][j]
                                                            prev_i2 = lst_X2_remain[k][j][i2r]
                                                            @timeit to "add diff"  curr_q = m[t-1, k][j] + lst_cost0[k][j] + diff_hv(X[t-1][[prev_i1, prev_i2]], X[t][[i1r, i2r]], X[t+1][M_star[t, id][[i1r, i2r]]]; width = width, height = height, kw...)
                                                        elseif i2r in lst_X3_remain_X2_appeared[k][j]
                                                            @timeit to "add diff"  curr_q = m[t-1, k][j] + lst_cost0[k][j] + diff_hv(X[t-1][prev_i1], X[t][[i1r, i2r]], X[t+1][M_star[t, id][[i1r, i2r]]]; width = width, height = height, kw...)
                                                        else
                                                            @warn "Something was odd."
                                                        end
                                                    elseif i1r in lst_X3_remain_X2_appeared[k][j]
                                                        if i2r in lst_X3_remain_X2_remain[k][j]
                                                            prev_i2 = lst_X2_remain[k][j][i2r]
                                                            @timeit to "add diff"  curr_q = m[t-1, k][j] + lst_cost0[k][j] + diff_hv(X[t-1][prev_i2], X[t][[i2r, i1r]], X[t+1][M_star[t, id][[i2r, i1r]]]; width = width, height = height, kw...)
                                                        elseif i2r in lst_X3_remain_X2_appeared[k][j]
                                                            @timeit to "add diff"  curr_q = m[t-1, k][j] + lst_cost0[k][j] + diff_hv(X[t][[i1r, i2r]], X[t+1][M_star[t, id][[i1r, i2r]]]; width = width, height = height, kw...)
                                                        else
                                                            @warn "Something was odd."
                                                        end
                                                    else
                                                        @warn "Something was odd."
                                                    end
                                                end
                                                @timeit to "compare" begin
                                                    if q > curr_q
                                                        q = curr_q
                                                        idx_argmin[t, id][i] = [j, k]
                                                    end
                                                end
                                            end
                                        end
                                    end
                                    m[t, id][i] = q
                                end
                            end
                        end
                    end
                end
            end
            # store the previous d_star
            # npos for m_star
            prev_npos_star = npos_star
        end
    end
    @timeit to "backtracking" begin
        optval = Inf
        optidx = [1, 1]
        for k = 1:2δ+1
            if !isassigned(m, (f-1)*k)
                continue
            else
                val, idx = findmin(m[f-1, k])
                if val < optval
                    optval = val
                    optidx .= [idx, k] # idx is CartesianIndex
                end
            end
        end
        Ms = Array{Array{Int64, 1}, 1}(undef, f-1)
        for t = f-1:-1:1
            Ms[t] = copy(M_star[t, optidx[2]]) # maybe do not need copy, since the following part does not use it again
            if t != 1
                orderIdx = inv_exIdx(optidx[1], sum(Ms[t] .!= -1))
                idx = [offsets[t, optidx[2]][orderIdx[i]] + orderIdx[i] for i = 1:2]
                Ms[t][idx] .= Ms[t][reverse(idx)]
            else
                break
            end
            optidx .= idx_argmin[t, optidx[2]][optidx[1]]
        end
        if history
            Mh = Array{Array{Array{Int64, 1}, 1}}(undef, f-1)
            for T = 1:f-2
                optval = Inf
                optidx = [1, 1]
                for k = 1:2δ+1
                    if !isassigned(m, (f-1)*(k-1) + T )
                        continue
                    else
                        val, idx = findmin(m[T, k])
                        if val < optval
                            optval = val
                            optidx .= [idx, k] # idx is CartesianIndex
                        end
                    end
                end
                Mh[T] = Array{Array{Int64, 1}, 1}(undef, T)
                for t = T:-1:1
                    Mh[T][t] = copy(M_star[t, optidx[2]]) # maybe do not need copy, since the following part does not use it again
                    if t != 1
                        orderIdx = inv_exIdx(optidx[1], sum(Ms[t] .!= -1))
                        idx = [offsets[t, optidx[2]][orderIdx[i]] + orderIdx[i] for i = 1:2]
                        Mh[T][t][idx] .= Mh[T][t][reverse(idx)]
                    else
                        break
                    end
                    optidx .= idx_argmin[t, optidx[2]][optidx[1]]
                end
            end
            Mh[f-1] = Ms
        end
    end
    if history
        return Mh
    else
        return Ms
    end
end

function bottom_up_match_optim2(X::Array{Array{Array{Int64, 1}, 1}, 1}, σs::Array{Float64, 1}; history = false, δ::Int = 1, maxdistance = Inf, width = 680, height = 512, kw...)
    @timeit to "declaration" begin
        # number of frames
        f = length(X)
        # if length(σs) == f-2
        #     include_first_two = false
        # else
        #     # f-1?
        #     include_first_two = true
        # end
        # store minimum
        m = Array{Array{Float64,1},2}(undef, f-1, 2δ+1)
        # store the index of argmin (orderIdx, num_disappear)
        idx_argmin = Array{Array{Array{Int, 1}, 1}, 2}(undef, f-1, 2δ+1)
        # skip m[1]
        # store m_star
        M_star = Array{Array{Int, 1}, 2}(undef, f-1, 2δ+1)
        offsets = Array{Array{Int, 1}, 2}(undef, f-1, 2δ+1)
        prev_npos_star = 0 # no use but to avoid UndefVarError
        # store cost0
        lst_cost0 = Array{Array{Float64, 1}, 1}(undef, 2δ+1)
        lst_X3_remain_X2_appeared = Array{Array{Set{Int64}, 1}, 1}(undef, 2δ+1)
        lst_X3_remain_X2_remain = Array{Array{Set{Int64}, 1}, 1}(undef, 2δ+1)
        lst_X2_remain = Array{Array{Dict{Int, Int}, 1}, 1}(undef, 2δ+1)
    end
    @timeit to "main" begin
        for t = 1:f-1
            # store the mincost vector
            @timeit to "mincost" M_star[t, δ+1] = mincost2(X[t], X[t+1], -1; maxdistance = maxdistance, width = width, height = height)
            d_star = sum(M_star[t, δ+1] .== -1)
            n1 = length(X[t]); n2 = length(X[t+1])
            npos_star = n1 - d_star
            for d = max(0, d_star - δ, n1 - n2):min(d_star + δ, n1)
                @timeit to "mincost-d" begin
                    id = δ + 1 + d - d_star
                    if d != d_star
                        M_star[t, id] = mincost2(X[t], X[t+1], d; maxdistance = maxdistance, width = width, height = height)
                    end
                    # offset of the index between raw idx and the sub idx
                    offsets[t, id] = cal_offset(M_star[t, id])
                    # number of non-disappeared, i.e., positive elements
                    npos = npos_star - (d - d_star)
                    # the size of the reduced space
                    nstate = binomial(npos, 2) + 1 #plus the one mincost
                    m[t, id] = ones(nstate)*Inf
                    idx_argmin[t, id] = Array{Array{Int, 1}, 1}(undef, nstate)
                end
                @timeit to "DP search" begin
                    if t == 1
                        @timeit to "Mstar cost" m[t, id][nstate] = calc_h(M_star[t, id], X[t], X[t+1]; σ = σs[t], kw...)
                        @timeit to "search" begin
                            for i1 = 1:npos-1
                                for i2 = i1+1:npos
                                    i = ex_idx(i1, i2, npos)
                                    i1r = i1 + offsets[t, id][i1]
                                    i2r = i2 + offsets[t, id][i2]
                                    @timeit to "add diff" m[t, id][i] = m[t, id][nstate] + diff_h(X[t][[i1r, i2r]], X[t+1][ M_star[t, id][[i1r, i2r]] ]; σ = σs[t], kw...)
                                    #m[t, id][i] = calc_h(M_star[t, id], X[t], X[t+1], i1 + offsets[t, id][i1], i2 + offsets[t, id][i2]; kw...)
                                end
                            end
                        end
                    else
                        @timeit to "Mstar cost" begin
                            for k = 1:2δ+1
                                if !isassigned(M_star, (f-1)*(k-1) + (t-1) )
                                    continue
                                end
                                prev_npos = prev_npos_star - (k - (δ + 1))
                                prev_nstate = binomial(prev_npos, 2) + 1
                                lst_cost0[k] = zeros(prev_nstate)
                                lst_X3_remain_X2_appeared[k] = Array{Set{Int64}, 1}(undef, prev_nstate)
                                lst_X3_remain_X2_remain[k] = Array{Set{Int64}, 1}(undef, prev_nstate)
                                lst_X2_remain[k] = Array{Dict{Int, Int}, 1}(undef, prev_nstate)
                                for j1 = 1:prev_npos
                                    for j2 = j1:prev_npos
                                        if j1 < prev_npos && j2 == j1
                                            continue
                                        end
                                        j = ex_idx(j1, j2, prev_npos)
                                        # cost for M23_star and given M12
                                        lst_cost0[k][j], lst_X3_remain_X2_appeared[k][j], lst_X3_remain_X2_remain[k][j], lst_X2_remain[k][j] = calc_hv_optim(M_star[t-1, k], M_star[t, id], X[t-1], X[t], X[t+1], j1 + offsets[t-1, k][j1], j2 + offsets[t-1, k][j2]; σ = σs[t], kw...)
                                    end
                                end
                            end
                        end
                        @timeit to "search" begin
                            for i1 = 1:npos # npos-1 is enough, but to include the M_star
                                for i2 = i1:npos # i1+1 is enough, but to include the M_star
                                    if i1 < npos && i2 == i1
                                        continue
                                    end
                                    i = ex_idx(i1, i2, npos)
                                    # if i1 == npos # for M_star
                                    #     i += 1 # since ex_idx(npos, npos, npos) = ex_idx(npos-1, npos, npos)
                                    # end
                                    i1r = i1 + offsets[t, id][i1]
                                    i2r = i2 + offsets[t, id][i2]
                                    q = Inf
                                    # find in the space unioned by all d
                                    for k = 1:2δ+1
                                        if !isassigned(M_star, (f-1)*(k-1) + (t-1) )
                                            continue
                                        end
                                        prev_npos = prev_npos_star - (k - (δ + 1))
                                        for j1 = 1:prev_npos
                                            for j2 = j1:prev_npos
                                                if j1 < prev_npos && j2 == j1
                                                    continue
                                                end
                                                j = ex_idx(j1, j2, prev_npos)
                                                # cost for M23_star and given M12
            #                                    cost0, X3_remain_X2_appeared, X3_remain_X2_remain, X2_remain = calc_hv_optim(M_star[t-1, k], M_star[t, id], X[t-1], X[t], X[t+1], j1 + offsets[t-1, k][j1], j2 + offsets[t-1, k][j2]; kw...)
                                                # if j1 == prev_npos
                                                #     j += 1 # for M_star
                                                # end
            #                                    curr_q = m[t-1, k][j] + calc_hv_optim(M_star[t-1, k], M_star[t, id], X[t-1], X[t], X[t+1], j1 + offsets[t-1, k][j1], j2 + offsets[t-1, k][j2], i1 + offsets[t, id][i1], i2 + offsets[t, id][i2]; kw...)
                                                curr_q = Inf
                                                if i1 == npos # M23_star
                                                    @timeit to "add diff" curr_q = m[t-1, k][j] + lst_cost0[k][j]
                                                else
                                                    # raw idx by adding back the offset
                                                    if i1r in lst_X3_remain_X2_remain[k][j]
                                                        prev_i1 = lst_X2_remain[k][j][i1r]
                                                        if i2r in lst_X3_remain_X2_remain[k][j]
                                                            prev_i2 = lst_X2_remain[k][j][i2r]
                                                            @timeit to "add diff"  curr_q = m[t-1, k][j] + lst_cost0[k][j] + diff_hv(X[t-1][[prev_i1, prev_i2]], X[t][[i1r, i2r]], X[t+1][M_star[t, id][[i1r, i2r]]]; σ = σs[t], kw...)
                                                        elseif i2r in lst_X3_remain_X2_appeared[k][j]
                                                            @timeit to "add diff"  curr_q = m[t-1, k][j] + lst_cost0[k][j] + diff_hv(X[t-1][prev_i1], X[t][[i1r, i2r]], X[t+1][M_star[t, id][[i1r, i2r]]]; σ = σs[t], kw...)
                                                        else
                                                            @warn "Something was odd."
                                                        end
                                                    elseif i1r in lst_X3_remain_X2_appeared[k][j]
                                                        if i2r in lst_X3_remain_X2_remain[k][j]
                                                            prev_i2 = lst_X2_remain[k][j][i2r]
                                                            @timeit to "add diff"  curr_q = m[t-1, k][j] + lst_cost0[k][j] + diff_hv(X[t-1][prev_i2], X[t][[i2r, i1r]], X[t+1][M_star[t, id][[i2r, i1r]]]; σ = σs[t], kw...)
                                                        elseif i2r in lst_X3_remain_X2_appeared[k][j]
                                                            @timeit to "add diff"  curr_q = m[t-1, k][j] + lst_cost0[k][j] + diff_hv(X[t][[i1r, i2r]], X[t+1][M_star[t, id][[i1r, i2r]]]; σ = σs[t], kw...)
                                                        else
                                                            @warn "Something was odd."
                                                        end
                                                    else
                                                        @warn "Something was odd."
                                                    end
                                                end
                                                @timeit to "compare" begin
                                                    if q > curr_q
                                                        q = curr_q
                                                        idx_argmin[t, id][i] = [j, k]
                                                    end
                                                end
                                            end
                                        end
                                    end
                                    m[t, id][i] = q
                                end
                            end
                        end
                    end
                end
            end
            # store the previous d_star
            # npos for m_star
            prev_npos_star = npos_star
        end
    end
    @timeit to "backtracking" begin
        optval = Inf
        optidx = [1, 1]
        for k = 1:2δ+1
            if !isassigned(m, (f-1)*k)
                continue
            else
                val, idx = findmin(m[f-1, k])
                if val < optval
                    optval = val
                    optidx .= [idx, k] # idx is CartesianIndex
                end
            end
        end
        Ms = Array{Array{Int64, 1}, 1}(undef, f-1)
        for t = f-1:-1:1
            Ms[t] = copy(M_star[t, optidx[2]]) # maybe do not need copy, since the following part does not use it again
            if t != 1
                orderIdx = inv_exIdx(optidx[1], sum(Ms[t] .!= -1))
                idx = [offsets[t, optidx[2]][orderIdx[i]] + orderIdx[i] for i = 1:2]
                Ms[t][idx] .= Ms[t][reverse(idx)]
            else
                break
            end
            optidx .= idx_argmin[t, optidx[2]][optidx[1]]
        end
        if history
            Mh = Array{Array{Array{Int64, 1}, 1}}(undef, f-1)
            for T = 1:f-2
                optval = Inf
                optidx = [1, 1]
                for k = 1:2δ+1
                    if !isassigned(m, (f-1)*(k-1) + T )
                        continue
                    else
                        val, idx = findmin(m[T, k])
                        if val < optval
                            optval = val
                            optidx .= [idx, k] # idx is CartesianIndex
                        end
                    end
                end
                Mh[T] = Array{Array{Int64, 1}, 1}(undef, T)
                for t = T:-1:1
                    Mh[T][t] = copy(M_star[t, optidx[2]]) # maybe do not need copy, since the following part does not use it again
                    if t != 1
                        orderIdx = inv_exIdx(optidx[1], sum(Ms[t] .!= -1))
                        idx = [offsets[t, optidx[2]][orderIdx[i]] + orderIdx[i] for i = 1:2]
                        Mh[T][t][idx] .= Mh[T][t][reverse(idx)]
                    else
                        break
                    end
                    optidx .= idx_argmin[t, optidx[2]][optidx[1]]
                end
            end
            Mh[f-1] = Ms
        end
    end
    if history
        return Mh
    else
        return Ms
    end
end



# estimate sigmas from min-cost flow matching based on positions
"""
    estimate_sigma(X, paths)

Estimate the sigma based on the paths, which can be converted from the matching vector.
"""
function estimate_sigma(X::Array{Array{Array{Int, 1}, 1}, 1}, paths::Array{Array{Int, 1}, 1}; type = "paths", include_first_two = false)
    if type == "matching"
        paths = recover_path(paths, length.(X))
    end
    # number of frames
    f = length(X)
    # start from the 3rd frame
    σs = zeros(f-2)
    ls = zeros(f-2)
    for t = 2:f-1
        vs = []
        for i = 1:length(paths)
            if any(paths[i][t-1:t+1] .== -1) # any -1
                continue
            else
                a, b, c = paths[i][t-1:t+1]
                append!(vs, (X[t+1][c] - X[t][b]) - (X[t][b] - X[t-1][a]))
            end
        end
        if length(vs) <= 1
            continue
        else
            σs[t-1] = std(vs)
            ls[t-1] = length(vs) / 2
        end
    end
    if include_first_two
        # the first two frame should also be considered
        ds = []
        for i = 1:length(paths)
            if any(paths[i][1:2] .== -1)
                continue
            else
                a, b = paths[i][1:2]
                append!(ds, X[2][b] - X[1][a])
            end
        end
        if length(ds) <= 1
            insert!(σs, 1, 0.0)
            insert!(ls, 1, length(ds) / 2)
        else
            insert!(σs, 1, std(ds))
            insert!(ls, 1, length(ds) / 2)
        end
    end
    return σs, ls
end




# min-cost flow
using LightGraphsFlows
using Clp: ClpSolver
import LightGraphs
const lg = LightGraphs
# mincost allow disappeared and appeared
"""
    mincost2(X1, X2, d)

Given the number of disappear, `d`, run the min-cost flow to match `X1` and `X2`.

Propose in https://github.com/szcf-weiya/Cell-Video/commit/3aabacf8820f97bf9275da2eb410425813aa5716, and discuss in https://github.com/szcf-weiya/Cell-Video/issues/80

If `method = distance` and `d = -1`, it reduces to classical min-cost match by position.
"""
function mincost2(X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1}, 1}, n_disappear::Int;
                                σ = 5.0, scale = 10, method="distance",
                                maxdistance = Inf, maxdist2boundary = 341^2, kw...)
    n1 = size(X1, 1)
    n2 = size(X2, 1)
    if !( (max(0, n1 - n2) <= n_disappear <= n1) || (n_disappear == -1))
        error("n_disappear ranges from [max(0, n1-n2), n1]")
    end
    source = n1 + n2 + 1
    appear = n1 + n2 + 2
    disappear = n1 + n2 + 3
    sink = n1 + n2 + 4
    num_nodes = sink
    # create a flow graph
    g = lg.DiGraph(num_nodes)
    w = zeros(num_nodes, num_nodes)
    capacity = ones(Int, num_nodes, num_nodes)
    # demand = zeros(num_nodes, num_nodes)
    node_demand = zeros(Int, num_nodes)
    edge_demand = zeros(Int, num_nodes, num_nodes)
    for i = 1:n1
        for j = 1:n2
            lg.add_edge!(g, i, n1 + j)
            if method == "distance"
                w[i, n1+j] = sum((X1[i] - X2[j]).^2)
                if w[i, n1+j] > maxdistance
                    w[i, n1+j] = maxdist2boundary
                end
            else
                w[i, n1+j] = Int(round(-scale*logpdf(MvNormal(X1[i],σ), X2[j])))
            end
        end
    end
    # for source
    for i = 1:n1
        lg.add_edge!(g, source, i)
        # demand[source, i] = 1
    end
    # for sink
    for j = 1:n2
        lg.add_edge!(g, n1+j, sink)
        # demand[n1+j, sink] = 1
    end
    # for appear
    # demand[source, appear] = n2
    node_demand[source] = n1+n2
    edge_demand[appear, disappear] = min(n2, n1 - n_disappear)
    if n_disappear == n1
        for i = 1:n1
            edge_demand[i, disappear] = 1
        end
    end
    capacity[source, appear] = n2
    lg.add_edge!(g, source, appear)
    for j = 1:n2
        lg.add_edge!(g, appear, n1+j)
        if method == "IS"
            w[appear, n1+j] = Int(round(scale*pr_link_appear_IS(X2[j]; σ = σ, kw...)))
        elseif method == "bvn"
            w[appear, n1+j] = Int(round(scale*pr_link_appear_bvn(X2[j]; σ = σ, kw...)))
        elseif method == "bvn2"
            w[appear, n1+j] = Int(round(scale*pr_link_appear_bvn2(X2[j]; σ = σ, kw...)))
        elseif method == "mc"
            w[appear, n1+j] = Int(round(scale*pr_link_appear_mc(X2[j]; σ = σ, kw...)))
        elseif method == "self"
            w[appear, n1+j] = Int(round(scale*pr_link_appear(X2[j]; σ = σ, kw...)))
        elseif split(method, '_')[1] == "prior"
            w[appear, n1+j] = Int(round(scale*pr_link_appear_prior(X2[j]; σ = σ, kw...)))
        else
            w[appear, n1+j] = dist_boundary(X2[j]; kw...)^2
        end
    end
    # for disappear
    # demand[disappear, sink] = n1
    node_demand[sink] = -(n1+n2)
    capacity[disappear, sink] = n1
    lg.add_edge!(g, disappear, sink)
    for i = 1:n1
        lg.add_edge!(g, i, disappear)
        if method == "IS"
            w[i, disappear] = Int(round(scale*pr_link_disappear_IS(X1[i]; σ = σ, kw...)))
        elseif method == "bvn"
            w[i, disappear] = Int(round(scale*pr_link_disappear_bvn(X1[i]; σ = σ, kw...)))
        elseif method == "bvn2"
            w[i, disappear] = Int(round(scale*pr_link_disappear_bvn2(X1[i]; σ = σ, kw...)))
        elseif method == "mc"
            w[i, disappear] = Int(round(scale*pr_link_disappear_mc(X1[i]; σ = σ, kw...)))
        elseif method == "self"
            w[i, disappear] = Int(round(scale*pr_link_disappear_bvn2(X1[i]; σ = σ, kw...)))
        elseif split(method, '_')[1] == "prior"
            w[i, disappear] = Int(round(scale*pr_link_disappear_bvn2(X1[i]; σ = σ, kw...)))
        else
            w[i, disappear] = dist_boundary(X1[i]; kw...)^2
        end
    end
    # appear to disappear
    capacity[appear, disappear] = min(n1, n2)
    lg.add_edge!(g, appear, disappear)

    if n_disappear == -1
        flow = mincost_flow(g, node_demand, capacity, w, ClpSolver())
    else
        flow = mincost_flow(g, node_demand, capacity, w, ClpSolver(); edge_demand = edge_demand, edge_demand_exact = true)
    end
    # convert to matching vector
    m = zeros(Int64, n1)
    for i = 1:n1
        m[i] = findfirst(!iszero, flow[i, :]) - n1
        # if disappeared
        if m[i] == disappear - n1
            m[i] = -1
        end
    end
    return m
end

mincost(X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1}, 1}; kw...) = mincost2(X1, X2, -1; kw...)

"""
    match_by_mincost(X)

If `method2 = true`, match the video `X` with min-cost flow for position. The definition of cost depends on the keyword `method`, [here](https://github.com/szcf-weiya/Cell-Video/issues/83#issue-505693763) are some thoughts whether to choose a particular cost definition for tripartitle model or based on distance.
"""
function match_by_mincost(X::Array{Array{Array{Int64, 1}, 1}, 1}; IS=false, method2 = false, kw...)
    f = length(X)
    m = Array{Array{Int64, 1}, 1}(undef, f-1)
    for i = 1:f-1
        if method2
            m[i] = mincost2(X[i], X[i+1], -1; kw...)
        else
            m[i] = mincost(X[i], X[i+1]; kw...)
        end
    end
    return m
end


module Data

using Distributions

# initialize data
"""
    gen_data(N)

Generate `N` cells uniformly in a rectangle region, with width `zoom`*`width` and height `zoom`*`height`.
"""
function gen_data(N; width = 680, height = 512, zoom = 1)
    wdist = DiscreteUniform(1, width*zoom)
    hdist = DiscreteUniform(1, height*zoom)
    w = rand(wdist, N)
    h = rand(hdist, N)
    return hcat(w, h)
end

# generate data near the boundary
# x x x x x
# x o o o x
# x o o o x
# x o o o x
# x x x x x
# assume each boundary with probability proportional to the length
function gen_data_boundary(N::Int; width = 680, height = 512, strip = 10)
    if rand() < width / (width + height)
        # width edge
        wdist = DiscreteUniform(1, width)
        if rand() < 1/2
            # top
            hdist = DiscreteUniform(1, strip)
        else
            # bottom
            hdist = DiscreteUniform(height-strip+1, height)
        end
    else
        # height edge
        hdist = DiscreteUniform(1, height)
        if rand() < 1/2
            # left
            wdist = DiscreteUniform(1, strip)
        else
            # right
            wdist = DiscreteUniform(width-strip+1, width)
        end
    end
    w = rand(wdist, N)
    h = rand(hdist, N)
    #return hcat(w, h) # Array{Int, 2}
    return [[w[i], h[i]] for i = 1:N] # Array{Array{Int, 1}, 1}
end

# data from random walk and without disappeared and appeared
function gen_data_rw0(N, f; sigma = 5)
    # init
    x = gen_data(N)
    X = Array{Array{Int,1},2}(undef, N, f)
    for i = 1:N
        X[i, 1] = x[i,:]
    end
    for k in 2:f
        x = X[:, k-1]
        for i = 1:N
            while true
                y = Int.(round.(rand(MvNormal(x[i], sigma))))
                if is_inside(y)
                    X[i, k] = y
                    break
                end
            end
        end
    end
    return X
end

# data from random walk and with disappeared and appeared
function gen_data_rw(N, f; sigma = 50.0)
    # init
    x = gen_data(N)
    X = Array{Array{Array{Int, 1}, 1}, 1}(undef, f)
    M = Array{Array{Int, 1}, 1}(undef, f-1)
    X[1] = Array{Array{Int, 1}, 1}(undef, N)
    for i = 1:N
        X[1][i] = x[i, :]
    end
    for k in 2:f
        x = X[k-1]
        M[k-1] = zeros(Int, length(x))
        id = 0
        X[k] = Array{Array{Int, 1}, 1}(undef, 0)
        for i = 1:length(x)
            y = Int.(round.(rand(MvNormal(x[i], sigma))))
            if is_inside(y)
                push!(X[k], y)
                id = id + 1
                M[k-1][i] = id
            else
                M[k-1][i] = -1
            end
        end
        # newcomer, must near the boundary
        if length(x) < N
            num_newcomer = rand(0:2(N-length(x)))
        else
            num_newcomer = rand(0:1)
        end
        if num_newcomer != 0
            newcomers = gen_data_boundary(num_newcomer)
            append!(X[k], newcomers) # push! only for a single element
        end
    end
    return X, M
end

export gen_data_rw

function gen_data0(N, f; σ = 5.0, κ = 32.0)
    # init data
    x = gen_data(N)
    X = Array{Array{Int,1},2}(undef, N, f)
    for i = 1:N
        X[i, 1] = x[i, :]
    end

    # frame 2
    x = X[:, 1]
    for i = 1:N
        while true
            y = Int.(round.(rand(MvNormal(x[i], σ))))
            if is_inside(y)
                X[i, 2] = y
                break
            end
        end
    end

    for k = 3:f
        for i = 1:N
            Δx = X[i, k-1] - X[i, k-2]
            # current s and θ
            s = sqrt(sum( Δx.^2 ))
            if s == 0
                s = 1e-5
            end
            θ = acos(Δx[1] / s)
            # update s and θ
            while true
                s_p = rand(Normal(s, σ))
                θ_p = rand(VonMises(θ, κ))
                # update locations
                y = Int.(round.([X[i, k-1][1] + s_p*cos(θ_p), X[i, k-1][2] + s_p*sin(θ_p)]))
                if is_inside(y)
                    X[i, k] = y
                    break
                end
            end
        end
    end
    return X
end

function id2match(id1::Array{Int, 1}, id2::Array{Int, 1})
    idx = id1 .!= -1
    return id2[idx][sortperm(id1[idx])]
end

# x1 inside, x2 outside
function reflect(x1::Array{Int, 1}, x2::Array{Int, 1}; width=680, height=512, allow_continue_reflection = false)
    if !(is_inside(x1, width = width, height = height) & (!is_inside(x2, width = width, height = height)))
        error("x1 should be inside and x2 should be outside")
    end
    # determine the reflection axis
    if (x1[1] >= 1) && (x2[1] < 1) # left axis (maybe two reflection)
        if x2[2] < 1
            res = [2-x2[1], 2-x2[2]]
        elseif x2[2] > height
            res = [2-x2[1], 2*height-x2[2]]
        else
            res = [2-x2[1], x2[2]]
        end
    elseif (x1[1] <= width) && (x2[1] > width) # right axis (maybe two reflection)
        if x2[2] < 1
            res = [2*width-x2[1], 2-x2[2]]
        elseif x2[2] > height
            res = [2*width-x2[1], 2*height-x2[2]]
        else
            res = [2*width-x2[1], x2[2]]
        end
    elseif (x1[2] >= 1) && (x2[2] < 1) # top axis & only need to one reflection
        res = [x2[1], 2-x2[2]]
    elseif (x1[2] <= height) && (x2[2] > height)
        res = [x2[1], 2*height-x2[2]]
    else
        # no need
        res = x2
    end
    if is_inside(res, width = width, height = height)
        return res
    else
        if allow_continue_reflection
            return reflect(x1, res, allow_continue_reflection = true, width = width, height = height)
        else
            error("after reflection, x2 is still outside, check if the sigma is too large")
        end
    end
end

"""
    gen_zoomed_data(N, f; zoom)

Simulate `N` cells in a big region, and let them move for `f` frames. Then zoom into the visible region with scaling factor `zoom` to get the observed positions `X` and ground truth matching `M`. See illustrations in https://user-images.githubusercontent.com/13688320/73755197-a2ce7400-47a0-11ea-8906-afb25a7d9399.png
"""
function gen_zoomed_data(N::Int64, f::Int64; σ = 5.0, ρ = 0.0, κ = 1.0, zoom = 10, width=680, height=512, kw...)
    Σ = [1 ρ*κ; ρ*κ κ^2] * σ^2
    x = zeros(Int, N, 2, f)
    local_id = fill(-1, N, f)
    x[:, :, 1] .= gen_data(N, zoom=zoom)
    offset = [0, 0]
    try
        offset = Int.([width, height]*(zoom-1) / 2)
    catch InexactError
        println("zoom should be integer") # since width and height are even
    end
    id = 0
    for i = 1:N
        if is_inside(x[i, :, 1] .- offset)
            id += 1
            local_id[i, 1] = id
        end
    end
    # movement
    is_reflection = zeros(Bool, N)
    for j = 2:f
        id = 0
        for i = 1:N
            if j == 2
                x[i, :, j] = round.(Int, rand(MvNormal(x[i, :, j-1], σ)))
                if !is_inside(x[i, :, j], width=width*zoom, height=height*zoom)
                    # ensure the reflection point is inside
                    x[i, :, j] = reflect(x[i, :, j-1], x[i, :, j], width=width*zoom, height=height*zoom)
                    is_reflection[i] = true
                end
            else
                if is_reflection[i]
                    # treat the reflection object as a newcomer
                    x[i, :, j] = round.(Int, rand(MvNormal(x[i, :, j-1], σ)))
                    if !is_inside(x[i, :, j], width=width*zoom, height=height*zoom)
                        x[i, :, j] = reflect(x[i, :, j-1], x[i, :, j], width=width*zoom, height=height*zoom)
                        is_reflection[i] = true
                    else
                        # re-assign the status
                        is_reflection[i] = false
                    end
                else
                    Δx = x[i,:,j-1] - x[i,:,j-2]
                    x[i, :, j] = x[i, :, j-1] + round.(Int, rand(MvNormal(Δx, Σ)))
                    if !is_inside(x[i, :, j], width=width*zoom, height=height*zoom)
                        x[i, :, j] = reflect(x[i, :, j-1], x[i, :, j], width=width*zoom, height=height*zoom)
                        is_reflection[i] = true
                    end
                end
            end
            if is_inside(x[i, :, j] .- offset)
                id += 1
                local_id[i, j] = id
            end
            if !is_inside(x[i, :, j], width=width*zoom, height=height*zoom)
                println("error")
                # println(x[i, :, j])
            end
        end
    end
    # zoom in the central region
    X = Array{Array{Array{Int, 1}, 1}, 1}(undef, f)
    M = Array{Array{Int, 1}, 1}(undef, f-1)
    for j = 1:f
        Xj = x[local_id[:, j] .!= -1, :, j]
        # convert Array{Int, 2} to Array{Array{Int,1},1}
        X[j] = [Xj[i,:] - offset for i=1:size(Xj, 1)]
    end
    # matching
    for j = 1:f-1
        M[j] = id2match(local_id[:, j], local_id[:, j+1])
    end
    return X, M
end

# with sampling rate
function gen_zoomed_data(N::Int64, f::Int64, r::Array{Int, 1}; σ = 5.0, ρ = 0.0, κ = 1.0, zoom = 10, width=680, height=512, kw...)
    Σ = [1 ρ*κ; ρ*κ κ^2] * σ^2
    x = zeros(Int, N, 2, f)
    local_id = fill(-1, N, f)
    x[:, :, 1] .= gen_data(N, zoom=zoom)
    offset = [0, 0]
    try
        offset = Int.([width, height]*(zoom-1) / 2)
    catch InexactError
        println("zoom should be integer") # since width and height are even
    end
    id = 0
    # frame 1
    for i = 1:N
        if is_inside(x[i, :, 1] .- offset)
            id += 1
            local_id[i, 1] = id
        end
    end
    # movement
    is_reflection = zeros(Bool, N)
    for j = 2:f
        id = 0
        for i = 1:N
            if j == 2
                x[i, :, j] = round.(Int, rand(MvNormal(x[i, :, j-1], σ)))
                if !is_inside(x[i, :, j], width=width*zoom, height=height*zoom)
                    x[i, :, j] = reflect(x[i, :, j-1], x[i, :, j], width=width*zoom, height=height*zoom)
                    is_reflection[i] = true
                end
            else
                if is_reflection[i]
                    # treat the reflection object as a newcomer
                    x[i, :, j] = round.(Int, rand(MvNormal(x[i, :, j-1], σ)))
                    if !is_inside(x[i, :, j], width=width*zoom, height=height*zoom)
                        x[i, :, j] = reflect(x[i, :, j-1], x[i, :, j], width=width*zoom, height=height*zoom)
                        is_reflection[i] = true
                    else
                        # re-assign the status
                        is_reflection[i] = false
                    end
                else
                    Δx = x[i,:,j-1] - x[i,:,j-2]
                    x[i, :, j] = x[i, :, j-1] + round.(Int, rand(MvNormal(Δx, Σ)))
                    if !is_inside(x[i, :, j], width=width*zoom, height=height*zoom)
                        x[i, :, j] = reflect(x[i, :, j-1], x[i, :, j], width=width*zoom, height=height*zoom)
                        is_reflection[i] = true
                    end
                end
            end
            if is_inside(x[i, :, j] .- offset)
                id += 1
                local_id[i, j] = id
            end
            if !is_inside(x[i, :, j], width=width*zoom, height=height*zoom)
                println("error")
                # println(x[i, :, j])
            end
        end
    end
    nr = length(r)
    lstX = Array{Array{Array{Array{Int64, 1}, 1}, 1}}(undef, nr)
    lstM = Array{Array{Array{Int64, 1}, 1}}(undef, nr)
    for k = 1:nr
        # frame idx
        fids = collect(1:r[k]:f)
        nfids = length(fids)
        # zoom in the central region
        X = Array{Array{Array{Int, 1}, 1}, 1}(undef, nfids)
        M = Array{Array{Int, 1}, 1}(undef, nfids-1)
        for j in 1:length(fids)
            fid = fids[j]
            Xj = x[local_id[:, fid] .!= -1, :, fid]
            # convert Array{Int, 2} to Array{Array{Int,1},1}
            X[j] = [Xj[i,:] - offset for i=1:size(Xj, 1)]
        end
        # matching
        for j = 1:nfids-1
            M[j] = id2match(local_id[:, fids[j]], local_id[:, fids[j+1] ])
        end
        lstX[k] = X
        lstM[k] = M
    end
    return lstX, lstM
end

export gen_zoomed_data

# allow disappear and appeared
function gen_data(N::Int64, f::Int64; σ = 5.0, ρ = 0.0, σ_pos = 0.0, allow_newcomer = true, κ = 1.0)
    Σ = [1 ρ*κ; ρ*κ κ^2] * σ^2
    # init
    x = gen_data(N)
    X = Array{Array{Array{Int64, 1}, 1}, 1}(undef, f)
    M = Array{Array{Int64, 1}, 1}(undef, f-1)
    X[1] = [x[i,:] for i=1:N]

    # frame 2 (same with gen_data_rw)
    for k in 2
        x = X[k-1]
        M[k-1] = zeros(Int, length(x))
        id = 0
        X[k] = Array{Array{Int, 1}, 1}(undef, 0)
        for i = 1:length(x)
            y = Int.(round.(rand(MvNormal(x[i], σ))))
            if is_inside(y)
                push!(X[k], y)
                id = id + 1
                M[k-1][i] = id
            else
                M[k-1][i] = -1
            end
        end
        if allow_newcomer
            num_newcomer = 0
            # newcomer, must near the boundary
            if length(x) < N
                num_newcomer = rand(0:2(N-length(x)))
            # else
            #     num_newcomer = rand(0:1)
            end
            if num_newcomer != 0
                newcomers = gen_data_boundary(num_newcomer)
                append!(X[k], newcomers) # push! only for a single element
            end
        end
    end

    # remain frames
    for k = 3:f
        # begin at frame k-2: velocity
        X[k] = Array{Array{Int, 1}, 1}(undef, 0)
        M[k-1] = zeros(Int64, length(X[k-1]))
        id = 0
        for i = 1:length(X[k-2])
            # exist at k-1
            if M[k-2][i] != -1
                Δx = X[k-1][M[k-2][i]] - X[k-2][i]
                # y = X[k-1][M[k-2][i]] + Δx*ρ + randn(2)*σ_pos
                y = X[k-1][M[k-2][i]] + rand(MvNormal(Δx, Σ))
                y = Int.(round.(y))
                if is_inside(y)
                    push!(X[k], y)
                    id += 1
                    M[k-1][M[k-2][i]] = id
                else
                    M[k-1][M[k-2][i]] = -1
                end
            else
                continue
                # do nothing
            end
        end
        # begin at frame k-1: position
        for i = 1:length(X[k-1])
            if M[k-1][i] == 0
                # appear at k-1
                y = Int.(round.(rand(MvNormal(X[k-1][i], σ))))
                if is_inside(y)
                    push!(X[k], y)
                    id += 1
                    M[k-1][i] = id
                else
                    M[k-1][i] = -1
                end
            end
        end
        if allow_newcomer
            # newcomer at random
            num_newcomer = 0
            if length(X[k]) < N
                num_newcomer = rand(0:2(N-length(X[k])))
            # else
            #     num_newcomer = rand(0:1)
            end
            if num_newcomer != 0
                newcomers = gen_data_boundary(num_newcomer)
                append!(X[k], newcomers)
            end
        end
    end
    return X, M
end

## generate data at different sampling rate
function gen_data_sampling(N::Int64, f::Int64; r::Int64 = 10, zoom = 1, kw...)
    if zoom != 1
        ax, aM = gen_zoomed_data(N, f*r, zoom = zoom, kw...)
    else
        aX, aM = gen_data(N, f*r; kw...)
    end
    X = Array{Array{Array{Int64, 1}, 1}, 1}(undef, f)
    M = Array{Array{Int64, 1}, 1}(undef, f-1)
    X[1] = aX[1]
    for i = 2:f
        X[i] = aX[(i-1)*r+1]
        M[i-1] = zeros(Int, length(X[i-1]))
        for k = 1:length(X[i-1])
            ki = k
            for j = 1:r
                ki = aM[(i-2)*r+j][ki]
                if ki == -1
                    break
                end
            end
            M[i-1][k] = ki
        end
    end
    return X, M
end

function sampling(aX::Array{Array{Array{Int64, 1}, 1}, 1}, aM::Array{Array{Int64, 1}, 1}, r::Int64)
    f = floor(Int, length(aX) / r)
    X = Array{Array{Array{Int64, 1}, 1}, 1}(undef, f)
    M = Array{Array{Int64, 1}, 1}(undef, f-1)
    X[1] = aX[1]
    for i = 2:f
        X[i] = aX[(i-1)*r+1]
        M[i-1] = zeros(Int, length(X[i-1]))
        for k = 1:length(X[i-1])
            ki = k
            for j = 1:r
                ki = aM[(i-2)*r+j][ki]
                if ki == -1
                    break
                end
            end
            M[i-1][k] = ki
        end
    end
    return X, M
end

## generate data at different sampling rate
function gen_data_sampling(N::Int64, f::Int64; r::Array{Int64, 1} = [10, 5], zoom=1, kw...)
    if zoom == 1
        aX, aM = gen_data(N, f*maximum(r); kw...)
    else
        aX, aM = gen_zoomed_data(N, f*maximum(r); zoom = zoom, kw...)
    end
    nr = length(r)
    # lstX = Array{Array{Array{Array{Int64, 1}, 1}, 1}}(undef, nr+1)
    # lstM = Array{Array{Array{Int64, 1}, 1}}(undef, nr+1)
    lstX = Array{Array{Array{Array{Int64, 1}, 1}, 1}}(undef, nr)
    lstM = Array{Array{Array{Int64, 1}, 1}}(undef, nr)
    for i = 1:nr
        lstX[i], lstM[i] = sampling(aX, aM, r[i])
    end
    if !(1 in r)
        push!(lstX, aX)
        push!(lstM, aM)
    end
    # lstX[nr+1] = aX
    # lstM[nr+1] = aM
    return lstX, lstM
end

export gen_data_sampling

# special cases
#
# o x x x x x x o
# x o x x x x o x
# x x o x x o x x
# x x x o o x x x
# x x x o o x x x
# x x o x x o x x
# x o x x x x o x
# o x x x x x x o
function gen_data_cross(f::Int; width = 680, height = 512, sigma = 5, origin = true)
    # init
    if origin
        x = [1 1; width 1]
    else
        x = gen_data(2)
    end
    x1_begin = x[1, :]
    x2_begin = x[2, :]
    x1_end = [width, height] - x1_begin .+ 1
    x2_end = [width, height] - x2_begin .+ 1
    # paths
    x1_path = collect(LinRange(x1_begin, x1_end, f))
    x2_path = collect(LinRange(x2_begin, x2_end, f))
    # add some noise
    for i = 2:f-1
        x1_path[i] = x1_path[i] + randn(2) * sigma
        x2_path[i] = x2_path[i] + randn(2) * sigma
    end
    # convert to int
    # NOTE: permutedims is non-recusive, while transpose is recursive
    return Array.(permutedims(hcat([Int.(round.(x1_path[i])) for i = 1:f],
                                [Int.(round.(x2_path[i])) for i = 1:f])))
    # return Array{Array{Int, 1}, 2}
end

export gen_data_cross

function is_inside(x; width = 680, height = 512)
    return (x[1] <= width) & (x[1] >= 1) & (x[2] <= height) & (x[2] >=1)
end

export is_inside

export gen_data

function render_img(data::Array{Int64,2}; width = 680, height = 512)
    img = zeros(Int, width, height)
    n = size(data, 1)
    for i in 1:n
        img[data[i, 1], data[i, 2]] = 1
        for j in 1:5
            img[data[i, 1]+j, data[i, 2]+j] = 1
            img[data[i, 1]+j, data[i, 2]-j] = 1
            img[data[i, 1]-j, data[i, 2]+j] = 1
            img[data[i, 1]-j, data[i, 2]-j] = 1
        end
    end
    return img
end

function render_img(data::Array{Array{Int64,1},1}; width = 680, height = 512)
    img = zeros(Int, width, height)
    n = size(data, 1)
    for i in 1:n
        img[data[i][1], data[i][2]] = 1
        for j in 1:5
            try
                img[data[i][1]+j, data[i][2]+j] = 1
                img[data[i][1]+j, data[i][2]-j] = 1
                img[data[i][1]-j, data[i][2]+j] = 1
                img[data[i][1]-j, data[i][2]-j] = 1
            catch

            end
        end
    end
    return img
end

using AxisArrays

function gen_video(X::Array{Int, 2})
    # number of frames
    f = size(X, 2)
    # concatenate images
    img = render_img(X[:,1])'
    for i = 2:f
        tmp = render_img(X[:, i])'
        img = cat(img, tmp, dims=3)
    end
    video = AxisArray(img, (:x, :y, :time), (1, 1, 1))
    return video
end

function gen_video(X::Array{Array{Array{Int, 1}, 1}, 1})
    # number of frames
    f = length(X)
    img = render_img(X[1])'
    for i = 2:f
        tmp = render_img(X[i])'
        img = cat(img, tmp, dims = 3)
    end
    video = AxisArray(img, (:x, :y, :time), (1,1,1))
    return video
end

export gen_video

end  # module

#=
using ImageView
using .Data
X = gen_data(5, 6)
video = gen_video(X)
imshow(video)
=#

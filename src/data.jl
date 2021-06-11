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



function is_inside(x; width = 680, height = 512)
    return (x[1] <= width) & (x[1] >= 1) & (x[2] <= height) & (x[2] >=1)
end

export is_inside



export gen_zoomed_data, gen_data

end  # module

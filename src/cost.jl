include("compute.jl")

function calc_h(M_star::Array{Int64, 1}, X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1},1}, i::Int, j::Int; kw...)
    M12 = copy(M_star)
    M12[[i, j]] .= M12[[j, i]]
    return calc_h(M12, X1, X2; kw...)
end

function diff_h(X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1},1}; σ = 1.0, kw...)
    return neglogpdf_bimvn(X2[2], X1[1], σ) +
            neglogpdf_bimvn(X2[1], X1[2], σ) -
            neglogpdf_bimvn(X2[1], X1[1], σ) -
            neglogpdf_bimvn(X2[2], X1[2], σ)
end


# x2 appear
pr_link_xo(method::String) = get(Dict(
    "IS" => pr_link_appear_IS,
    "bvn" => pr_link_appear_bvn,
    "bvn2" => pr_link_appear_bvn2,
    "mc" => pr_link_appear_mc,
    "self" => pr_link_appear,
    "prior" => pr_link_appear_prior,
    "prior_max" => pr_link_appear,
    "prior_hybrid" => pr_link_appear_prior
), method, pr_link_appear)

# x2 disappear
pr_link_ox(method::String) = get(Dict(
    "IS" => pr_link_disappear_IS,
    "bvn" => pr_link_disappear_bvn,
    "bvn2" => pr_link_disappear_bvn2,
    "mc" => pr_link_disappear_mc,
    "self" => pr_link_disappear_bvn2,
    "prior" => pr_link_disappear_bvn2,
    "prior_max" => pr_link_disappear_bvn2,
    "prior_hybrid" => pr_link_disappear_bvn2
), method, pr_link_disappear)

# allow disappear and appear
function calc_h(M12::Array{Int64, 1}, X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1},1}; width = 680, height = 512, σ = 5.0, method="bvn", robust = 1.0, kw...)
    n1 = size(X1, 1)
    n2 = size(X2, 1)
    X2_appeared = setdiff(1:n2, M12) # index of X2
    X2_disappeared = findall(M12 .== -1) # index of X1
    X2_remain = intersect(1:n2, M12) # index of X2
    X2_remain_X1_idx = findall(in(1:n2), M12) # index of X1
    # calculate the probability
    pr_link = 0.0
    for i in X2_appeared
        pr_link += pr_link_xo(method)(X2[i]; σ = robust*σ, width = width, height = height, kw...)
    end
    for i in X2_disappeared
        pr_link += pr_link_ox(method)(X1[i]; σ = robust*σ, width = width, height = height, kw...)
    end
    for i in X2_remain_X1_idx
        pr_link += neglogpdf_bimvn(X2[ M12[i] ], X1[i], σ)
    end
    return pr_link
end

# probability of disappeared
# two cases:
#   newcomer: only consider the position
#   not newcomer: consider the velocity and the position

# newcomer: only depend on the distance to boundary
function pr_link_disappear(x::Array{Int64, 1}; width = 680, height = 512, σ = 5.0, kw...)
    # minmum distance to the boundary
    # min_w = min(x[1], width - x[1])
    # min_h = min(x[2], height - x[2])
    dist = dist_boundary(x; width = width, height = height) # kw... would overwrite width and height
    # choose the corner [corner may not be suitable]!!
    # return -logpdf(MvNormal(x, σ), x + [min_w, min_h])
    #return -logpdf(MvNormal(x, σ), x + [dist, 0])
    return neglogpdf_bimvn([0, 0], [dist, 0], σ)
end

function dist_boundary(x::Array{Int64, 1}; width = 680, height = 512, arg = false, kw...)
    dist, idx = findmin([x[1] - 1, width - x[1], x[2] - 1, height - x[2]])
    if dist < 0
        error("x is outside the boundary")
    end
    if arg
        return dist, idx
    else
        return dist
    end
end

function pr_link_disappear_max2_bvn2(x::Array{Int64, 1}; width = 680, height = 512, σ = 5.0, kw...)
    # minmum distance to the boundary
    dist, idx = dist_boundary(x, width = width, height = height, arg = true)
    # choose the corner [corner may not be suitable]!!
    # return -logpdf(MvNormal(x, σ), x + [min_w, min_h])
    if idx == 1
        return pr_link_disappear_bvn2([1, x[2]], x; σ = σ, width = width, height = height, kw...)
    elseif idx == 2
        return pr_link_disappear_bvn2([width, x[2]], x; σ = σ, width = width, height = height, kw...)
    elseif idx == 3
        return pr_link_disappear_bvn2([x[1], 1], x; σ = σ, width = width, height = height, kw...)
    else
        return pr_link_disappear_bvn2([x[1], height], x; σ = σ, width = width, height = height, kw...)
    end
end


# calculate as expectation, and via importance sampling
function pr_link_disappear_IS(x::Array{Int64, 1}; width = 680, height = 512, σ = 5.0, ns = 1000, f2i = true, kw...)
    dist = MvNormal(x, σ)
    num = 0
    for i = 1:ns
        if f2i
            y = round.(Int, rand(dist))
        else
            y = rand(dist)
        end
        if is_inside(y, width = width, height = height)
            num += 1
        end
    end
    # return -log(1 - num / ns)
    return -log(1 - num / (ns + 1e-10) )
end

# calculate as expectation, and via importance sampling
function pr_link_disappear_mc(x::Array{Int64, 1}; width = 680, height = 512,
                                                  ϱ=3.0, σ = 5.0, ns = 1000, f2i = true,
                                                  α = 0, # 1 for disappear from middle
                                                  kw...)
    dist = MvNormal(x, ϱ*σ)
    w = zeros(ns)
    h = zeros(ns)
    for i = 1:ns
        if f2i
            y = round.(Int, rand(dist))
        else
            y = rand(dist)
        end
        if !is_inside(y, width = width, height = height)
            h[i] = 1
        else
            h[i] = α
        end
        w[i] = pdf(MvNormal(x, σ), y)
    end
    # return -log(1 - num / ns)
    return -log(sum(w .* h) / ns + nextfloat(0.0) )
end

# calculate the region probability for disappearing via `bvn`
function pr_link_disappear_bvn(x::Array{Int64, 1}; width = 680, height = 512, σ = 5.0, ns = 1000, f2i = true, kw...)
    nodes = [width height;
             width 1;
             1 height;
             1 1]
    # in the general coordinate system
    upperright = bvnuppercdf(nodes[1,:], x, σ)
    lowerright = bvnuppercdf(nodes[2,:], x, σ)
    upperleft = bvnuppercdf(nodes[3,:], x, σ)
    lowerleft = bvnuppercdf(nodes[4,:], x, σ)
    res = -log( 1 - (lowerleft - lowerright - upperleft + upperright) )
    if isinf(res)
        return 2^15
    else
        return res
    end
end

# calculate the region probability for disappearing via `bvn2`
function pr_link_disappear_bvn2(μ::Array{Int64, 1}; width = 680, height = 512, σ = 5.0, kw...)
    nodes = [width height;
             width 1;
             1 height;
             1 1]
    # in the general coordinate system
    upper = bvnuppercdf(-Inf, (height - μ[2]) / σ, 0.0)
    right = bvnuppercdf((width - μ[1])/σ, -Inf, 0.0)
    lower = bvnuppercdf(-Inf, -(1 - μ[2])/σ, 0.0)
    left = bvnuppercdf(-(1-μ[1])/σ, -Inf, 0.0)
    upperright = bvnuppercdf((width - μ[1])/σ, (height - μ[2]) / σ, 0.0)
    lowerright = bvnuppercdf((width - μ[1])/σ, -(1 - μ[2])/σ, 0.0)
    upperleft = bvnuppercdf(-(1-μ[1])/σ, (height - μ[2]) / σ, 0.0)
    lowerleft = bvnuppercdf(-(1-μ[1])/σ, -(1 - μ[2])/σ, 0.0)
    probs = upper + right + lower + left - upperright - upperleft - lowerleft - lowerright
    res = -log( probs )
    if isinf(res)
        return -log(nextfloat(0.0))
    else
        return res
    end
end

# probability of appeared
# only one case
const pr_link_appear = pr_link_disappear
const pr_link_appear_IS = pr_link_disappear_IS
const pr_link_appear_bvn = pr_link_disappear_bvn
const pr_link_appear_bvn2 = pr_link_disappear_bvn2
const pr_link_appear_mc = pr_link_disappear_mc

function pr_link_appear_prior(x::Array{Int64, 1}; width = 680, height = 512, σ = 5.0, θ = 2.0, ns = 1000, kw...)
    # sample the distance to the boundary
    dist = Exponential(θ)
    πs = zeros(ns)
    xy = zeros(Int64, 2, ns)
    ps = zeros(8)
    ps[[1,5]] .= width
    ps[[3,7]] .= height
    for i = 1:ns
        ρ = rand(dist)
        # proportional to the region area (or say length)
        ps[[2,4,6,8]] .= π/4*ρ
        # locate the random number
        idx = findfirst(rand() .< cumsum(ps ./ (2width + 2height + π*ρ)))
        if (idx == 1) || (idx == 5)
            xpos = sample(1:width)
            if idx == 1
                ypos = round(Int, height + ρ)
            else
                ypos = round(Int, 1 - ρ)
            end
        elseif (idx == 3) || (idx == 7)
            ypos = sample(1:height)
            if idx == 3
                xpos = round(Int, width - ρ)
            else
                xpos = round(Int, 1 - ρ)
            end
        else
            ang = rand()*2π
            if ang < π/2
                xpos = round(Int, width + ρ*cos(ang))
                ypos = round(Int, height + ρ*sin(ang))
            elseif ang < π
                xpos = round(Int, 1+ρ*cos(ang))
                ypos = round(Int, height + ρ*sin(ang))
            elseif ang < π*3/2
                xpos = round(Int, 1+ρ*cos(ang))
                ypos = round(Int, 1+ρ*sin(ang))
            else
                xpos = round(Int, width + ρ*cos(ang))
                ypos = round(Int, 1+ρ*sin(ang))
            end
        end
        xy[:, i] .= [xpos, ypos]
        # πs[i] = pdf(MvNormal([xpos, ypos], σ), x)
        # πs[i] = 1 / (2π*σ^2) * exp(-sum((x - [xpos, ypos]).^2) / (2σ^2) )
    end
    Distributions.pdf!(πs, MvNormal(x, σ), xy)
    res = -log(mean(πs))
    if isinf(res)
        return -log(nextfloat(0.0))
    else
        return res
    end
end

# not newcomer: find the outside point along the previous direction
function pr_link_disappear(x1::Array{Int64, 1}, x2::Array{Int64, 1}; width = 680, height = 512, σ = 5.0)
    # find four interaction points
    # xleft xright ytop ybottom
    points = ones(4)*Inf
    if x1[1] == x2[1]
        if x2[2] > x1[2] # cannot change to the opposite direction
            points[4] = height - x2[2]
        else
            points[3] = x2[2]
        end
        points[1:2] .= [x2[1], width - x2[1]]
    elseif x1[2] == x2[2]
        if x2[1] > x1[1]
            points[2] = width - x2[1]
        else
            points[1] = x2[1]
        end
        points[3:4] .= [x2[2], height - x2[2]]
    else
        slope = (x2[2] - x1[2]) / (x2[1] - x1[1])
        c1 = sqrt(1+slope^2)
        c2 = sqrt(1+(1/slope)^2)
        # points[1] = c1*x2[1]
        # points[2] = c1*(width - x2[1])
        # points[3] = c2*x2[2]
        # points[4] = c2*(height - x2[2])
        if slope > 0
            if x2[2] > x1[2]
                # xright ybottom
                points[2] = c1*(width - x2[1])
                points[4] = c2*(height - x2[2])
            else
                points[1] = c1*x2[1]
                points[3] = c2*x2[2]
            end
        else
            if x2[2] > x1[2]
                # xleft ybottom
                points[1] = c1*x2[1]
                points[4] = c2*(height - x2[2])
            else
                points[2] = c1*(width - x2[1])
                points[3] = c2*x2[2]
            end
        end
    end
    val, idx = findmin(points)
    # one more same step
    dist0 = sqrt(sum((x2 - x1).^2))
    if val < dist0
        return -logpdf(MvNormal(x2, σ), x2 + [dist0, 0])
    else
        # seem not proper:
        return -logpdf(MvNormal(x2, σ), x2 + [val, 0])
    end
    # Some thoughs TODO: replace σ with dist0
end

# via importance sampling
function pr_link_disappear_IS(x1::Array{Int64, 1}, x2::Array{Int64, 1}; kw...)
    return pr_link_disappear_IS(2x2-x1; kw...)
end

function pr_link_disappear_mc(x1::Array{Int64, 1}, x2::Array{Int64, 1}; kw...)
    return pr_link_disappear_mc(2x2-x1; kw...)
end

# calculate the region probability for disappearing via `bvn`
function pr_link_disappear_bvn(x1::Array{Int64, 1}, x2::Array{Int64, 1}; kw...)
    μ = 2x2 - x1 # x2 + N(x2-x1, Σ)
    return pr_link_disappear_bvn(μ; kw...)
end

# calculate the region probability for disappearing via `bvn2`
function pr_link_disappear_bvn2(x1::Array{Int64, 1}, x2::Array{Int64, 1}; kw...)
    μ = 2x2 - x1 # x2 + N(x2-x1, Σ)
    return pr_link_disappear_bvn2(μ; kw...)
end


# recheck the disappeared/appeared matches
function pr_link_disappear_recheck(x1::Array{Int64, 1}, x2::Array{Int64, 1}; width = 680, height = 512, σ = 5.0)
    # find four interaction points
    # xleft xright ytop ybottom
    points = ones(4)*Inf
    if x1[1] == x2[1]
        if x2[2] > x1[2] # cannot change to the opposite direction
            points[4] = height - x2[2]
        else
            points[3] = x2[2]
        end
        points[1:2] .= [x2[1], width - x2[1]]
    elseif x1[2] == x2[2]
        if x2[1] > x1[1]
            points[2] = width - x2[1]
        else
            points[1] = x2[1]
        end
        points[3:4] .= [x2[2], height - x2[2]]
    else
        slope = (x2[2] - x1[2]) / (x2[1] - x1[1])
        c1 = sqrt(1+slope^2)
        c2 = sqrt(1+(1/slope)^2)
        # points[1] = c1*x2[1]
        # points[2] = c1*(width - x2[1])
        # points[3] = c2*x2[2]
        # points[4] = c2*(height - x2[2])
        if slope > 0
            if x2[2] > x1[2]
                # xright ybottom
                points[2] = c1*(width - x2[1])
                points[4] = c2*(height - x2[2])
            else
                points[1] = c1*x2[1]
                points[3] = c2*x2[2]
            end
        else
            if x2[2] > x1[2]
                # xleft ybottom
                points[1] = c1*x2[1]
                points[4] = c2*(height - x2[2])
            else
                points[2] = c1*(width - x2[1])
                points[3] = c2*x2[2]
            end
        end
    end
    val, idx = findmin(points)
    # one more same step
    dist0 = sqrt(sum((x2 - x1).^2))
    if val < dist0
        return -logpdf(MvNormal(x2, σ), x2 + [dist0, 0])
    else
        # seem not proper:
        return -logpdf(MvNormal(x2, σ), x2 + [val, 0])
    end
end

# consider disappeared & disappeared
function calc_hv(M12::Array{Int64, 1}, M23::Array{Int64, 1}, X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1}, 1}, X3::Array{Array{Int64, 1}, 1}; σ = 10.0)
    n1 = length(X1)
    n2 = length(X2)
    n3 = length(X3)
    # error checking: invalid matching vector
    if (length(M12) == 0) || (length(M23) == 0)
        print("Matching vector cannot be empty.")
        return
    end
    if length(M12) != n1
        print("Incompatible length of matching vector M12")
        return
    end
    if length(M23) != n2
        print("Incompatible length of matching vector M23")
        return
    end

    if maximum(M12) > n2
        print("Invalid element matching vector M12")
        return
    end
    if maximum(M23) > n3
        print("Invalid element matching vector M23")
        return
    end

    # divide X3 into four class: remain, newcomer, disappeared, appeared

    # appeared at X2
    X2_appeared = setdiff(1:n2, M12) # index of X2
    # disappeared at X2
    # (actually NO need) disappeared at X2: -1 for disappeared
    X2_disappeared = findall(M12 .== -1) # index of X1
    # remain at X2
    X2_remain = intersect(1:n2, M12) #index of X2
    X2_remain_X1_idx = findall(in(1:n2), M12) #index of X1
    # appeared at X3
    X3_appeared = setdiff(1:n3, M23) # index of X3
    # disappeared at X3
    X3_disappeared = findall(M23 .== -1) # index of X2
    # two subcases for disappeared
    X3_disappeared_X2_appeared = intersect(X3_disappeared, X2_appeared)
    X3_disappeared_X2_remain = intersect(X3_disappeared, X2_remain)
    # newcomer at X3: appeared at X2 but not disappeared at X3
    X3_remain_X2_appeared = setdiff(X2_appeared, X3_disappeared) # index of X2
    # the remain: not disappeared in the remain at X2
    X3_remain_X2_remain = X2_remain[ findall(M23[X2_remain] .!= -1) ] # index of X2
    # convert to index of X1
    X3_remain_X2_remain_X1_idx = findall(in(X3_remain_X2_remain), M12) #index of X1

    pr_link = 0.0
    for i in X3_appeared
        pr_link += pr_link_appear(X3[i])
    end
    for i in X3_disappeared_X2_appeared
        pr_link += pr_link_disappear(X2[i])
    end
    for i in X3_disappeared_X2_remain
        # find the index of X1
        x1_idx = findall(M12 .== i)
        pr_link += pr_link_disappear(X1[x1_idx][1], X2[i])
    end
    for i in X3_remain_X2_appeared
        pr_link += -logpdf(MvNormal(X2[i], σ), X3[ M23[i] ])
    end
    for i in X3_remain_X2_remain_X1_idx
        v12 = X2[M12[i]] - X1[i]
        v23 = X3[M23[M12[i]]] - X2[M12[i]]
        pr_link += -logpdf(MvNormal(v12, σ), v23)
    end
    return pr_link
end

function calc_hv_optim(M12_star::Array{Int64, 1}, M23_star::Array{Int64, 1}, X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1}, 1}, X3::Array{Array{Int64, 1}, 1}, i::Int, j::Int, s::Int, t::Int; kw...)
    M12 = copy(M12_star)
    M23 = copy(M23_star)
    M12[[i, j]] .= M12[[j, i]]
    M23[[s, t]] .= M23[[t, s]]
    return calc_hv_optim(M12, M23, X1, X2, X3; kw...)
end

function calc_hv_optim(M12_star::Array{Int64, 1}, M23_star::Array{Int64, 1}, X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1}, 1}, X3::Array{Array{Int64, 1}, 1}, i::Int, j::Int; kw...)
    M12 = copy(M12_star)
    M12[[i, j]] .= M12[[j, i]]
    return calc_hv_optim(M12, M23_star, X1, X2, X3; forstar = true, kw...)
end


# swap two elements in S_{&&&}
function diff_hv(X1::Array{Array{Int64, 1}, 1}, X2::Array{Array{Int64, 1}, 1}, X3::Array{Array{Int64, 1}, 1}; σ = 1.0, kw...)
    return neglogpdf_bimvn(X3[2] .- X2[1], X2[1] .- X1[1], σ) +
            neglogpdf_bimvn(X3[1] .- X2[2], X2[2] .- X1[2], σ) -
            neglogpdf_bimvn(X3[1] .- X2[1], X2[1] .- X1[1], σ) -
            neglogpdf_bimvn(X3[2] .- X2[2], X2[2] .- X1[2], σ)
end

# swap two elements in S_{\&&}
function diff_hv(X2::Array{Array{Int64, 1}, 1}, X3::Array{Array{Int64, 1}, 1}; σ = 1.0, width = 680, height = 512, kw...)
    return pr_link_x3_remain_x2_appeared_priormax(X2[1], X3[2], σ = σ, width = width, height = height) +
            pr_link_x3_remain_x2_appeared_priormax(X2[2], X3[1], σ = σ, width = width, height = height) -
            pr_link_x3_remain_x2_appeared_priormax(X2[1], X3[1], σ = σ, width = width, height = height) -
            pr_link_x3_remain_x2_appeared_priormax(X2[2], X3[2], σ = σ, width = width, height = height)
end

# swap two elements, one in S_{&&&}, another one in S_{\&&}
function diff_hv(X1::Array{Int64, 1}, X2::Array{Array{Int64, 1}, 1}, X3::Array{Array{Int64, 1}, 1}; σ = 1.0, width = 680, height = 512, kw...)
    return neglogpdf_bimvn(X3[2] .- X2[1], X2[1] .- X1, σ) +
            pr_link_x3_remain_x2_appeared_priormax(X2[2], X3[1], σ = σ, width = width, height = height) -
            neglogpdf_bimvn(X3[1] .- X2[1], X2[1] .- X1, σ) -
            pr_link_x3_remain_x2_appeared_priormax(X2[2], X3[2], σ = σ, width = width, height = height)
end

# u: unknown
# x: not exist
# o: exist

# x3 appear
pr_link_uxo(method::String) = get(Dict(
    "IS" => pr_link_appear_IS,
    "bvn" => pr_link_appear_bvn,
    "bvn2" => pr_link_appear_bvn2,
    "mc" => pr_link_appear_mc,
    "self" => pr_link_appear,
    "prior" => pr_link_appear_prior,
    "prior_max" => pr_link_appear,
    "prior_hybrid" => pr_link_appear_prior
), method, pr_link_appear)

# x2 appear and x3 disappear
pr_link_xox(method::String) = get(Dict(
    "IS" => pr_link_disappear_IS,
    "bvn" => pr_link_disappear_bvn,
    "bvn2" => pr_link_disappear_bvn2,
    "mc" => pr_link_disappear_mc,
    "self" => pr_link_disappear_bvn2,
    "prior" => pr_link_disappear_max2_bvn2,
    "prior_max" => pr_link_disappear_max2_bvn2,
    "prior_hybrid" => pr_link_disappear_max2_bvn2
), method, pr_link_disappear)

# x2 remain and x3 disappear
pr_link_oox(method::String) = get(Dict(
    "IS" => pr_link_disappear_IS,
    "bvn" => pr_link_disappear_bvn,
    "bvn2" => pr_link_disappear_bvn2,
    "mc" => pr_link_disappear_mc,
    "self" => pr_link_disappear_bvn2,
    "prior" => pr_link_disappear_bvn2,
    "prior_max" => pr_link_disappear_bvn2,
    "prior_hybrid" => pr_link_disappear_bvn2
), method, pr_link_disappear)

# x2 appeared, x3 remain
pr_link_xoo(method::String) = get(Dict(
    "prior_max" => pr_link_x3_remain_x2_appeared_priormax,
    "prior_hybrid" => pr_link_x3_remain_x2_appeared_prior
), method, pr_link_x3_remain_x2_appeared)

# optimizing....
# consider disappeared & disappeared
"""
    calc_hv_optim(M12, M23, X1, X2, X3)

Calculate ``P(X3 | X2, X1, M12, M23)``
"""
function calc_hv_optim(M12::Array{Int64, 1}, M23::Array{Int64, 1}, X1::Array{Array{Int64, 1}, 1},
                                        X2::Array{Array{Int64, 1}, 1}, X3::Array{Array{Int64, 1}, 1};
                                        σ = 10.0, method="bvn", forstar = false, robust = 1.0,
                                        width = 680, height = 512,
                                        kw...)
    # for the special uniform motion; see https://github.com/szcf-weiya/Cell-Video/issues/104
   if σ == 0
       σ = 5e-1
   end
    n1 = length(X1)
    n2 = length(X2)
    n3 = length(X3)
    # error checking: invalid matching vector
    if (length(M12) == 0) || (length(M23) == 0)
        print("Matching vector cannot be empty.")
        return
    end
    if length(M12) != n1
        print("Incompatible length of matching vector M12")
        return
    end
    if length(M23) != n2
        print("Incompatible length of matching vector M23")
        return
    end

    if maximum(M12) > n2
        print("Invalid element matching vector M12")
        return
    end
    if maximum(M23) > n3
        print("Invalid element matching vector M23")
        return
    end

    # divide X3 into four class: remain, newcomer, disappeared, appeared

    X2_appeared = Set{Int64}(1:n2)
    X2_disappeared = Set{Int64}()
    X2_remain = Dict{Int, Int}()
    for i = 1:length(M12)
        # disappeared at X2
        # (actually NO need) disappeared at X2: -1 for disappeared
        if M12[i] == -1
            push!(X2_disappeared, i) # index of X1
        else
            # remain at X2
            # push!(X2_remain, M12[i]) #index of X2
            # push!(X2_remain_X1_idx, i) #index of X1
            push!(X2_remain, M12[i] => i) # idx_X2 => idx_X1
            # appeared at X2
            delete!(X2_appeared, M12[i]) # index of X2
        end
    end
    X2_remain_X1_idx = values(X2_remain)

    X3_appeared = Set{Int64}(1:n3)
    X3_disappeared_X2_remain = Set{Int64}()
    X3_disappeared_X2_appeared = Set{Int64}()
    X3_remain_X2_appeared = Set{Int64}()
    X3_remain_X2_remain = Set{Int64}()
    X3_remain_X2_remain_X1_idx = Set{Int64}()
    for i = 1:length(M23)
        if M23[i] == -1
            # disappeared at X3
            if i in keys(X2_remain)
                push!(X3_disappeared_X2_remain, i) # index of X2
            else
                push!(X3_disappeared_X2_appeared, i) # index of X1
            end
        else
            if i in X2_appeared
                # newcomer at X3: appeared at X2 but not disappeared at X3
                push!(X3_remain_X2_appeared, i) #index of X2
            else#if i in X2_remain
                push!(X3_remain_X2_remain, i) # index of X2
                push!(X3_remain_X2_remain_X1_idx, X2_remain[i]) # index of X1
            end
            delete!(X3_appeared, M23[i]) # index of X3
        end
    end

    pr_link = 0.0
    for i in X3_appeared
        pr_link += pr_link_uxo(method)(X3[i]; σ=robust*σ, width = width, height = height, kw...)
    end
    for i in X3_disappeared_X2_appeared
        pr_link += pr_link_xox(method)(X2[i]; σ=robust*σ, width = width, height = height, kw...)
    end
    for i in X3_disappeared_X2_remain
        # find the index of X1
        x1_idx = findall(M12 .== i)
        pr_link += pr_link_oox(method)(X1[x1_idx][1], X2[i]; σ=robust*σ, width = width, height = height, kw...)
    end
    for i in X3_remain_X2_appeared
        pr_link += pr_link_xoo(method)(X2[i], X3[ M23[i] ];σ = robust*σ, width = width, height = height, kw...)
    end
    for i in X3_remain_X2_remain_X1_idx
        pr_link += neglogpdf_bimvn(X3[M23[M12[i]]] .- X2[M12[i]], X2[M12[i]] .- X1[i], σ)
    end
    # Given M12 (not necessarily M_star), but M23 should be M_star
    if forstar
        return pr_link,
                X3_remain_X2_appeared,
                X3_remain_X2_remain, # the idex for S_{&&&} in X2
                X2_remain # convert the above index in X2 to the index in X1
    else
        return pr_link
    end
end

function neglogpdf_bimvn(x::Array{Int, 1}, μ::Array{Int, 1}, σ::Union{Float64, Int})
    return sum((x .- μ).^2) / (2σ^2) + log(2π) + 2log(σ)
end

pr_link_x3_remain_x2_appeared(x2::Array{Int, 1}, x3::Array{Int, 1}; σ = 1.0, kw...) = neglogpdf_bimvn(x3, x2, σ)

function pr_link_x3_remain_x2_appeared_priormax(x2::Array{Int, 1}, x3::Array{Int, 1}; width=680, height=512, σ = σ, kw...)
    dist, idx = dist_boundary(x2, width = width, height = height, arg = true)
    v23 = x3 - x2
    if idx == 1
        v12 = x2 - [1, x2[2]]
    elseif idx == 2
        v12 = x2 - [width, x2[2]]
    elseif idx == 3
        v12 = x2 - [x2[1], 1]
    else
        v12 = x2 - [x2[1], height]
    end
    return neglogpdf_bimvn(v23, v12, σ)
end

function pr_link_disappear_x2_hybrid(x::Array{Int64, 1}; width = 680, height = 512, σ = 5.0, kw...)
    # minmum distance to the boundary
    dist, idx = dist_boundary(x, width = width, height = height, arg = true)
    # choose the corner [corner may not be suitable]!!
    # return -logpdf(MvNormal(x, σ), x + [min_w, min_h])
    if idx == 1
        return pr_link_x3_remain_x2_appeared_prior(x, [1, x[2]]; σ = σ, width = width, height = height, kw...)
    elseif idx == 2
        return pr_link_x3_remain_x2_appeared_prior(x, [width, x[2]]; σ = σ, width = width, height = height, kw...)
    elseif idx == 3
        return pr_link_x3_remain_x2_appeared_prior(x, [x[1], 1]; σ = σ, width = width, height = height, kw...)
    else
        return pr_link_x3_remain_x2_appeared_prior(x, [x[1], height]; σ = σ, width = width, height = height, kw...)
    end
end

function pr_link_x3_remain_x2_appeared_prior(x2::Array{Int, 1}, x3::Array{Int, 1}; width=680, height=512, σ = σ, ns=100, θ=2, kw...)
    dist = Exponential(θ)
    πs = zeros(ns)
    xy = zeros(Int64, 2, ns)
    ps = zeros(8)
    ps[[1,5]] .= width
    ps[[3,7]] .= height
    for i = 1:ns
        ρ = rand(dist)
        # proportional to the region area (or say length)
        ps[[2,4,6,8]] .= π/4*ρ
        # locate the random number
        idx = findfirst(rand() .< cumsum(ps ./ (2width + 2height + π*ρ)))
        if (idx == 1) || (idx == 5)
            xpos = sample(1:width)
            if idx == 1
                ypos = round(Int, height + ρ)
            else
                ypos = round(Int, 1 - ρ)
            end
        elseif (idx == 3) || (idx == 7)
            ypos = sample(1:height)
            if idx == 3
                xpos = round(Int, width - ρ)
            else
                xpos = round(Int, 1 - ρ)
            end
        else
            ang = rand()*2π
            if ang < π/2
                xpos = round(Int, width + ρ*cos(ang))
                ypos = round(Int, height + ρ*sin(ang))
            elseif ang < π
                xpos = round(Int, 1+ρ*cos(ang))
                ypos = round(Int, height + ρ*sin(ang))
            elseif ang < π*3/2
                xpos = round(Int, 1+ρ*cos(ang))
                ypos = round(Int, 1+ρ*sin(ang))
            else
                xpos = round(Int, width + ρ*cos(ang))
                ypos = round(Int, 1+ρ*sin(ang))
            end
        end
        xy[:, i] .= [xpos, ypos]
        # πs[i] = pdf(MvNormal([xpos, ypos], σ), x)
        # πs[i] = 1 / (2π*σ^2) * exp(-sum((x - [xpos, ypos]).^2) / (2σ^2) )
    end
    Distributions.pdf!(πs, MvNormal(2x2-x3, σ), xy)
    res = -log(mean(πs))
    if isinf(res)
        return -log(nextfloat(0.0))
    else
        return res
    end
end



function calc_h(M12, M23, X1::Array{Array{Int64,1},1}, X2::Array{Array{Int64,1},1}, X3::Array{Array{Int64,1},1}; σ = 10.0, κ = 1.0)
    n1 = size(X1, 1)
    n2 = size(X2, 1)
    n3 = size(X3, 1)
    # suppose n1 = n2 = n3 = n
    s2 = zeros(n1)
    θ2 = zeros(n1)
    for i = 1:n1
        s2[i], θ2[i] = xy2sθ(X1[i], X2[M12[i]])
    end
    s3 = zeros(n2)
    θ3 = zeros(n2)
    for i = 1:n2
        s3[i], θ3[i] = xy2sθ(X2[i], X3[M23[i]])
    end

    # calculate the probability
    pr_link = 0
    for i = 1:n2
        pr_link += -logpdf(VonMises(θ2[i], κ), θ3[ M23[i] ])-logpdf(LogNormal(log(s2[i]), σ), s3[ M23[i] ])
    end
    return pr_link
end

function calc_h(M12, M23, X1::Array{Int64, 2}, X2::Array{Int64, 2}, X3::Array{Int64, 2}; σ = 5.0, κ = 16.0)
    n1 = size(X1, 1)
    n2 = size(X2, 1)
    n3 = size(X3, 1)

    # suppose n1 = n2 = n3 = n
    s2 = zeros(n1)
    θ2 = zeros(n1)
    for i = 1:n1
        s2[i], θ2[i] = xy2sθ(X1[i,:], X2[M12[i], :])
    end
    s3 = zeros(n2)
    θ3 = zeros(n2)
    for i = 1:n2
        s3[i], θ3[i] = xy2sθ(X2[i,:], X3[M23[i], :])
    end

    # calculate the probability
    pr_link = 0
    for i = 1:n2
        pr_link += -logpdf(VonMises(θ2[i], κ), θ3[ M23[i] ])-logpdf(Normal(s2[i], σ), s3[ M23[i] ])
    end
    return pr_link
end


function xy2sθ(x1, x2)
    s = sqrt( sum( (x2 - x1).^2 ) )
    θ = acos( (x2[1] - x1[1]) / s )
    return s, θ
end

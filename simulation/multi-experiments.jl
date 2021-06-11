using Distributed
using Dates
@everywhere using Serialization
@everywhere using DelimitedFiles
@everywhere using JLD2

const jobs = RemoteChannel(()->Channel{Tuple}(32))
const res = RemoteChannel(()->Channel{Tuple}(32))

# count the number of jobs
num = 0
function make_jobs(ns::Array{Int, 1}, σs::Array{Int, 1})
    for n in ns
        for σ in σs
            put!(jobs, (n, σ))
        end
    end
    global num = length(ns) * length(σs)
end

@everywhere function do_work(jobs, res, folder, nrep; adaptive = false)
    while true
        n, σ = take!(jobs)
        # fix zoom
        zoom = 5
        f = 50
        acc = zeros(f, 1 + 2*4) #the first column for distance
        isfallin = zeros(Int, f-1, 1+4)
        exec_times = zeros(1+4)
        exec_time = @elapsed begin
            # no sampling rate
            X, M = gen_zoomed_data(n*zoom^2, f, σ = σ, zoom=zoom)

            # naive method: based on distance
            exec_times[1] = @elapsed begin
            m = match_by_mincost(X, method2=true, method="distance")
            end
            acc[:, 1] = acc2fβ(calc_path_accuracy_point(m, M, length.(X)))
            isfallin[:, 1] = (M .== m)
            # use sigma estimated from the naive method
            σs, lens = estimate_sigma(X, m, type = "matching", include_first_two = true)
            σs0, lens0 = estimate_sigma(X, M, type = "matching", include_first_two = true)
            # matched based on prior max method
            for δ = 0:3
                exec_times[δ+2] = @elapsed begin
                m = bottom_up_match_optim2(X, method="prior_max",
                                           σ = σ, ns=100,
                                           δ = δ, robust = 1,
                                           history = false);
                end
                mhat = bottom_up_match_optim2(X, σs, method="prior_max",
                                              ns=100, δ = δ,
                                              robust = 1,
                                              history = false);
                isfallin[:, 1+δ+1] = truth_coverage(X, M,
                                                    method="prior_max",
                                                    σ = σ, ns = 100,
                                                    method2=true,
                                                    knearest=false,
                                                    δ = δ);
                acc[:, 1+2δ+1] = acc2fβ(calc_path_accuracy_point(m, M, length.(X)))
                acc[:, 1+2δ+2] = acc2fβ(calc_path_accuracy_point(mhat, M, length.(X)))
            end
        end
        @save "$(folder)/$(nrep)_XM_$(n)_$(σ).jld2" X M
        writedlm("$(folder)/$(nrep)_sigma_$(n)_$(σ).txt", hcat(σs, lens, σs0, lens0))
        writedlm("$(folder)/$(nrep)_acc_$(n)_$(σ).txt", acc)
        writedlm("$(folder)/$(nrep)_isfallin_$(n)_$(σ).txt", isfallin)
        writedlm("$(folder)/$(nrep)_exectime_$(n)_$(σ).txt", exec_times)
        put!(res, ("$(n)_$(σ)", exec_time, myid()))
    end
end

# main function
nrep, parent_folder = ARGS
@async make_jobs(collect(15:5:50),
                 collect(1:4))

for p in workers()
    remote_do(do_work, p, jobs, res, parent_folder, nrep; adaptive = true)
end

while num > 0
    job_id, exec_time, where = take!(res)
    println("$job_id finished in $(round(exec_time; digits=2)) secs on worker $where")
    global num -= 1
end

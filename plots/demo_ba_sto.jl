using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators, MLDatasets, RegularizedProblems
using NLPModels, NLPModelsModifiers #ReverseADNLSModels
using RegularizedOptimization
using DataFrames
using SolverBenchmark
using PlotlyJS

# Random.seed!(1234)

function demo_ba_sto(name_list::Vector{String}; sample_rate = .05, n_runs::Int = 1, MaxEpochs::Int = 20, MaxTime = 3600.0, version::Int = 4, suffix::String = "dubrovnik-h1", compare::Bool = false, smooth::Bool = false)
    temp_PLM = []
    temp_LM = []
    temp_LMTR = []
    for name in name_list
        nls = BundleAdjustmentModel(name)
        sampled_nls = BAmodel_sto(name; sample_rate = sample_rate)

        #nlp_train = LSR1Model(nlp_train)
        λ = 1e-1
        # h = RootNormLhalf(λ)
        h = NormL1(λ)
        χ = NormLinf(1.0)

        if compare
            if smooth
                @info " using LM to solve"
                reset!(nls)
                LM_out = levenberg_marquardt(nls; max_iter = 20, in_itmax = 300)
                nlm = neval_residual(nls)
                nglm = neval_jtprod_residual(nls) + neval_jprod_residual(nls)

                sol = LM_out.solution
                x = [sol[3*i+1] for i in 0:(nls.npnts-1)]
                y = [sol[3*i+2] for i in 0:(nls.npnts-1)]
                z = [sol[3*i] for i in 1:nls.npnts]
                plt3d = PlotlyJS.plot(PlotlyJS.scatter(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=attr(
                        size=1,
                        opacity=0.8
                    ),
                    type="scatter3d"
                ), Layout(margin=attr(l=0, r=0, b=0, t=0)))
                display(plt3d)
                

                @info " using LMTR to solve with" χ
                reset!(nls)
                LMTR_out = levenberg_marquardt(nls, TR = true; max_iter = 20, in_itmax = 300)
                nlmtr = neval_residual(nls)
                nglmtr = neval_jtprod_residual(nls) + neval_jprod_residual(nls)

                sol = LMTR_out.solution
                x = [sol[3*i+1] for i in 0:(nls.npnts-1)]
                y = [sol[3*i+2] for i in 0:(nls.npnts-1)]
                z = [sol[3*i] for i in 1:nls.npnts]
                plt3d = PlotlyJS.plot(PlotlyJS.scatter(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=attr(
                        size=1,
                        opacity=0.8
                    ),
                    type="scatter3d"
                ), Layout(margin=attr(l=0, r=0, b=0, t=0)))
                display(plt3d)

                if name == name_list[1]
                    temp_LM = [0.5 * LM_out.rNorm^2, nlm, nglm, LM_out.elapsed_time]
                    temp_LMTR = [0.5 * LMTR_out.rNorm^2, nlmtr, nglmtr, LMTR_out.elapsed_time]
                else
                    temp_LM = hcat(temp_LM, [0.5 * LM_out.rNorm^2, nlm, nglm, LM_out.elapsed_time])
                    temp_LMTR = hcat(temp_LMTR, [0.5 * LMTR_out.rNorm^2, nlmtr, nglmtr, LMTR_out.elapsed_time])
                end
            end
        end

        # demo_solver_ba(nls, sampled_nls, h, χ, "l0-linf-$name"; n_runs = n_runs) #

        options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-4, ϵr = 1e-4, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
        suboptions = RegularizedOptimization.ROSolverOptions(maxIter = 300)

        sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = 1e-4, ϵr = 1e-4, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)

        @info " using Prob_LM to solve with" h

        PLM_outs = []
        plm_obj = []
        for k in 1:n_runs
            reset!(sampled_nls)
            sampled_nls.epoch_counter = Int[1]
            Prob_LM_out_k = Prob_LM(sampled_nls, h, sampled_options, x0=sampled_nls.meta.x0, subsolver_options = suboptions, version = version)
            push!(PLM_outs, Prob_LM_out_k)
            push!(plm_obj, Prob_LM_out_k.objective)
        end
        if n_runs%2 == 1
            med_ind = (n_runs ÷ 2) + 1
        else
            med_ind = (n_runs ÷ 2)
        end
        sorted_obj_vec = sort(plm_obj)
        ref_value = sorted_obj_vec[med_ind]
        origin_ind = 0
        for i in eachindex(PLM_outs)
            if plm_obj[i] == ref_value
                origin_ind = i
            end
        end

        # Prob_LM_out is the run associated to the median accuracy on the training set
        Prob_LM_out = PLM_outs[origin_ind]

        sol = Prob_LM_out.solution
        x = [sol[3*i+1] for i in 0:(sampled_nls.npnts-1)]
        y = [sol[3*i+2] for i in 0:(sampled_nls.npnts-1)]
        z = [sol[3*i] for i in 1:sampled_nls.npnts]

        plt3d = PlotlyJS.plot(PlotlyJS.scatter(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=attr(
                        size=1,
                        opacity=0.8
                    ),
                    type="scatter3d"
                ), Layout(margin=attr(l=0, r=0, b=0, t=0)))
        display(plt3d)

        nplm = neval_residual(sampled_nls) / 100
        ngplm = (neval_jtprod_residual(sampled_nls) + neval_jprod_residual(sampled_nls)) / 100

        # Results Table #
        if name == name_list[1]
            temp_PLM = [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, nplm, ngplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time]
        else
            temp_PLM = hcat(temp_PLM, [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, nplm, ngplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time])
        end

    end

    temp = temp_PLM'
    df = DataFrame(temp, [:f, :h, :fh, :n, :g, :p, :s])
    #df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
    df[!, :Alg] = name_list
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :f => "%10.2f",
        :h => "%10.2f",
        :fh => "%10.2f",
        :n => "%i",
        :g => "%i",
        :p => "%i",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :f => "\$ f \$",
        :h => "\$ h \$",
        :fh => "\$ f+h \$",
        :n => "\\# \$f\$",
        :g => "\\# \$ \\nabla f \$",
        :p => "\\# \$ \\prox{}\$",
        :s => "\$t \$ (s)")
    open("BA-plm-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
    
    temp = temp_LM'
    df = DataFrame(temp, [:fh, :n, :g, :s])
    #df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
    df[!, :Alg] = name_list
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :fh => "%10.2f",
        :n => "%i",
        :g => "%i",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :fh => "\$ f+h \$",
        :n => "\\# \$f\$",
        :g => "\\# \$ \\nabla f \$",
        :s => "\$t \$ (s)")
    open("BA-lm-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end

    temp = temp_LMTR'
    df = DataFrame(temp, [:fh, :n, :g, :s])
    #df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
    df[!, :Alg] = name_list
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :fh => "%10.2f",
        :n => "%i",
        :g => "%i",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :fh => "\$ f+h \$",
        :n => "\\# \$f\$",
        :g => "\\# \$ \\nabla f \$",
        :s => "\$t \$ (s)")
    open("BA-lmtr-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
end
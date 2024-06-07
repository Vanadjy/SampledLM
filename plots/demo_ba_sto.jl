using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators, MLDatasets, RegularizedProblems
using NLPModels, NLPModelsModifiers #ReverseADNLSModels
using RegularizedOptimization
using DataFrames
using SolverBenchmark
using PlotlyJS

# Random.seed!(1234)

function demo_solver_ba(nls, sampled_nls, h, χ, suffix="l0-linf"; n_runs = 1)
    MaxEpochs = 20
    MaxTime = 3600.0
    version = 4
    options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-4, ϵr = 1e-4, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
    suboptions = RegularizedOptimization.ROSolverOptions(maxIter = 300)

    sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = 1e-4, ϵr = 1e-4, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)

    @info " using Prob_LM to solve with" h

    Prob_LM_out = Prob_LM(sampled_nls, h, sampled_options, x0=sampled_nls.meta.x0, subsolver_options = suboptions, version = version)
    sol = Prob_LM_out.solution
    x = [sol[3*i+1] for i in 0:(sampled_nls.npnts-1)]
    y = [sol[3*i+2] for i in 0:(sampled_nls.npnts-1)]
    z = [sol[3*i] for i in 1:sampled_nls.npnts]
    #plt3d = plot(x, y, z, seriestype=:scatter, markersize = .5, title = "BA-$suffix", markerstrokewidth=0, ylim=(40,80), xlim=(-75, 0), zlim=(-100, -50))
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

    # Results Table #

    #=temp = hcat([R2_out.solver_specific[:Fhist][end], R2_out.solver_specific[:Hhist][end],R2_out.objective, acc(r2train), acc(r2test), nr2, ngr2, sum(R2_out.solver_specific[:SubsolverCounter]), R2_out.elapsed_time],
        #[LM_out.solver_specific[:Fhist][end], LM_out.solver_specific[:Hhist][end], LM_out.objective, acc(lmtrain), acc(lmtest), nlm, nglm, sum(LM_out.solver_specific[:SubsolverCounter]), LM_out.elapsed_time],
        [LMTR_out.solver_specific[:Fhist][end], LMTR_out.solver_specific[:Hhist][end], LMTR_out.objective, acc(lmtrtrain), acc(lmtrtest), nlmtr, nglmtr, sum(LMTR_out.solver_specific[:SubsolverCounter]), LMTR_out.elapsed_time],
        #[Sto_LM_out.solver_specific[:ExactFhist][end], Sto_LM_out.solver_specific[:Hhist][end], Sto_LM_out.solver_specific[:ExactFhist][end] + Sto_LM_out.solver_specific[:Hhist][end], acc(slmtrain), acc(slmtest), nslm, ngslm, sum(Sto_LM_out.solver_specific[:SubsolverCounter]), Sto_LM_out.elapsed_time],
        [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, acc(plmtrain), acc(plmtest), nplm, ngplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time])'=#

    #=df = DataFrame(temp, [:f, :h, :fh, :x,:xt, :n, :g, :p, :s])
    T = []
    for i = 1:nrow(df)
      push!(T, Tuple(df[i, [:x, :xt]]))
    end
    select!(df, Not(:xt))
    df[!, :x] = T
    df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :f => "%10.2f",
        :h => "%10.2f",
        :fh => "%10.2f",
        :x => "%10.2f, %10.2f",
        :n => "%i",
        :g => "%i",
        :p => "%i",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :f => "\$ f \$",
        :h => "\$ h \$",
        :fh => "\$ f+h \$",
        :x => "(Train, Test)",
        :n => "\\# \$f\$",
        :g => "\\# \$ \\nabla f \$",
        :p => "\\# \$ \\prox{}\$",
        :s => "\$t \$ (s)")
    open("svm-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end=#
end

function demo_ba_sto(name::String; sample_rate = .05, n_runs = 1)
    nls = BundleAdjustmentModel(name)
    sampled_nls = BAmodel_sto(name; sample_rate = sample_rate)

    #nlp_train = LSR1Model(nlp_train)
    λ = 1e-1
    # h = RootNormLhalf(λ)
    h = NormL1(λ)
    χ = NormLinf(1.0)

    demo_solver_ba(nls, sampled_nls, h, χ, "l0-linf-$name"; n_runs = n_runs)
end
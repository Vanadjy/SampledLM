function greymaps_tables_mnist(version, sample_rate0; digits = (1, 7), selected_h = "lhalf", smooth = false)
    local mnist, mnist_nls, mnist_nls_sol = RegularizedProblems.svm_train_model()
    local mnist_nlp_test, mnist_nls_test, mnist_sol_test = RegularizedProblems.svm_test_model()

    k_R2, R2_out, R2_stats = load_mnist_r2()
    LM_out, LMTR_out = load_mnist_lm_lmtr()

    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    @info "using R2 to solve with $selected_h"
    reset!(mnist)
    mnist = LSR1Model(mnist)
    nr2 = NLPModels.neval_obj(mnist)
    ngr2 = NLPModels.neval_grad(mnist)
    r2train = residual(mnist_nls, R2_stats.solution) #||e - tanh(b * <A, x>)||^2, b ∈ {-1,1}^n
    r2test = residual(mnist_nls_test, R2_stats.solution)
    @show acc(r2train), acc(r2test)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps")
    r2dec = plot_svm(R2_stats, R2_stats.solution, "r2-$version-lhalf-$digits")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    @info " using LM to solve with $selected_h"
    lmtrain = residual(mnist_nls, LM_out.solution)
    lmtest = residual(mnist_nls_test, LM_out.solution)
    @show acc(lmtrain), acc(lmtest)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps")
    lmdec = plot_svm(LM_out, LM_out.solution, "lm-$version-lhalf-$digits")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    #nlm = NLPModels.neval_residual(nls_tr)
    nlm = LM_out.iter
    nglm = NLPModels.neval_jtprod_residual(mnist_nls) + NLPModels.neval_jprod_residual(mnist_nls)

    @info " using LMTR to solve with $selected_h" NormLinf(1.0)
    reset!(mnist_nls)
    lmtrtrain = residual(mnist_nls, LMTR_out.solution)
    lmtrtest = residual(mnist_nls_test, LMTR_out.solution)
    @show acc(lmtrtrain), acc(lmtrtest)
    #nlmtr = NLPModels.neval_residual(nls_tr)
    nlmtr = LMTR_out.iter
    nglmtr = NLPModels.neval_jtprod_residual(mnist_nls) + NLPModels.neval_jprod_residual(mnist_nls)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps")
    lmtrdec = plot_svm(LMTR_out, LMTR_out.solution, "lmtr-$version-lhalf-$digits")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    @info "using PLM to solve with $selected_h"
    local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate0; digits = digits)
    local mnist_nlp_test_sto, mnist_nls_test_sto, mnist_sto_sol_test = MNIST_test_model_sto(sample_rate0; digits = digits)

    med_obj_prob_mnist, med_metr_prob_mnist, med_mse_prob_mnist, std_obj_prob_mnist, std_metr_prob_mnist, std_mse_prob_mnist, PLM_outs, plm_trains = load_mnist_plm(version, selected_h)

    if n_exec%2 == 1
        med_ind = (n_exec ÷ 2) + 1
    else
        med_ind = (n_exec ÷ 2)
    end
    acc_vec = acc.(plm_trains)
    sorted_acc_vec = sort(acc_vec)
    ref_value = sorted_acc_vec[med_ind]
    origin_ind = 0
    for i in eachindex(PLM_outs)
        if acc_vec[i] == ref_value
            origin_ind = i
        end
    end

    Prob_LM_out = PLM_outs[origin_ind]
    plmtrain = residual(mnist_nls, Prob_LM_out.solution)
    plmtest = residual(mnist_nls_test_sto, Prob_LM_out.solution)
    @show acc(plmtrain), acc(plmtest)
    #nplm = neval_residual(sampled_nls_tr)
    nplm = length(mnist_nls.epoch_counter)
    ngplm = (neval_jtprod_residual(mnist_nls) + neval_jprod_residual(mnist_nls))

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps")
    plmdec = plot_svm(Prob_LM_out, Prob_LM_out.solution, "prob-lm-$version-lhalf-$digits")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    if smooth
        med_obj_prob_mnist_smooth, med_metr_prob_mnist_smooth, med_mse_prob_mnist_smooth, std_obj_prob_mnist_smooth, std_metr_prob_mnist_smooth, std_mse_prob_mnist_smooth, SPLM_outs, splm_trains = load_mnist_splm(version, selected_h)

        @info "using SPLM to solve"
        if n_exec%2 == 1
            med_ind = (n_exec ÷ 2) + 1
        else
            med_ind = (n_exec ÷ 2)
        end
        acc_vec = acc.(splm_trains)
        sorted_acc_vec = sort(acc_vec)
        ref_value = sorted_acc_vec[med_ind]
        origin_ind = 0
        for i in eachindex(SPLM_outs)
            if acc_vec[i] == ref_value
                origin_ind = i
            end
        end

        SProb_LM_out = SPLM_outs[origin_ind]
        splmtrain = residual(mnist_nls, SProb_LM_out.solution)
        splmtest = residual(mnist_nls_test_sto, SProb_LM_out.solution)
        @show acc(splmtrain), acc(splmtest)
        #nplm = neval_residual(sampled_nls_tr)
        nsplm = length(mnist_nls.epoch_counter)
        ngsplm = (neval_jtprod_residual(mnist_nls) + neval_jprod_residual(mnist_nls))

        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps")
        plmdec = plot_svm(SProb_LM_out, SProb_LM_out.solution, "smooth-prob-lm-$version-lhalf-$digits")
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
    end

    temp = hcat([R2_stats.solver_specific[:Fhist][end], R2_stats.solver_specific[:Hhist][end],R2_stats.objective, acc(r2train), acc(r2test), nr2, ngr2, sum(R2_stats.solver_specific[:SubsolverCounter]), R2_stats.elapsed_time],
        [LM_out.solver_specific[:Fhist][end], LM_out.solver_specific[:Hhist][end], LM_out.objective, acc(lmtrain), acc(lmtest), nlm, nglm, sum(LM_out.solver_specific[:SubsolverCounter]), LM_out.elapsed_time],
        [LMTR_out.solver_specific[:Fhist][end], LMTR_out.solver_specific[:Hhist][end], LMTR_out.objective, acc(lmtrtrain), acc(lmtrtest), nlmtr, nglmtr, sum(LMTR_out.solver_specific[:SubsolverCounter]), LMTR_out.elapsed_time],
        #[Sto_LM_out.solver_specific[:ExactFhist][end], Sto_LM_out.solver_specific[:Hhist][end], Sto_LM_out.solver_specific[:ExactFhist][end] + Sto_LM_out.solver_specific[:Hhist][end], acc(slmtrain), acc(slmtest), nslm, ngslm, sum(Sto_LM_out.solver_specific[:SubsolverCounter]), Sto_LM_out.elapsed_time],
        [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, acc(plmtrain), acc(plmtest), nplm, ngplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time])

    if smooth
        temp = hcat(temp,
        [SProb_LM_out.objective, 0.0, SProb_LM_out.objective, acc(splmtrain), acc(splmtest), nsplm, ngsplm, sum(SProb_LM_out.solver_specific[:SubsolverCounter]), SProb_LM_out.elapsed_time]
        )
    end

    temp = temp'

    df = DataFrame(temp, [:f, :h, :fh, :x, :xt, :n, :g, :p, :s])
    T = []
    for i = 1:nrow(df)
    push!(T, Tuple(df[i, [:x, :xt]]))
    end
    select!(df, Not(:xt))
    df[!, :x] = T
    df[!, :Alg] = !smooth ? ["R2", "LM", "LMTR", "PLM-$(prob_versions_names[version])"] : ["R2", "LM", "LMTR", "PLM-$(prob_versions_names[version])", "SPLM-$(prob_versions_names[version])"]
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :f => "%10.2f",
        :h => "%10.2f",
        :fh => "%10.2f",
        :x => "%10.2f, %10.2f",
        :n => "%10.2f",
        :g => "%10.2f",
        :p => "%10.2f",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :f => "\$ f \$",
        :h => "\$ h \$",
        :fh => "\$ f+h \$",
        :x => "(Train, Test)",
        :n => "\\# epoch",
        :g => "\\# \$ \\nabla f \$",
        :p => "\\# inner",
        :s => "\$t \$ (s)")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\tables")
    open("svm-lhalf-$version-$digits.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
end
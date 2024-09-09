function greymaps_tables_mnist(versions, sample_rates, sample_rate0; digits = (1, 7), selected_h = "lhalf", smooth = false, MaxEpochs::Int = 1000)
    local mnist, mnist_nls, mnist_nls_sol = RegularizedProblems.svm_train_model()
    local mnist_nlp_test, mnist_nls_test, mnist_sol_test = RegularizedProblems.svm_test_model()

    R2_stats, r2_metric_hist, r2_obj_hist, r2_numjac_hist = load_mnist_r2(selected_h; MaxEpochs = MaxEpochs)
    #LM_out, LMTR_out = load_mnist_lm_lmtr(selected_h)

    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    @info " R2 with $selected_h"
    reset!(mnist)
    mnist = LSR1Model(mnist)
    nr2 = NLPModels.neval_obj(mnist)
    ngr2 = NLPModels.neval_grad(mnist)
    r2train = residual(mnist_nls, R2_stats.solution) #||e - tanh(b * <A, x>)||^2, b ∈ {-1,1}^n
    r2test = residual(mnist_nls_test, R2_stats.solution)
    @show acc(r2train), acc(r2test)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps\tikz_and_dats")
    r2dec = plot_svm(R2_stats, R2_stats.solution, "r2-lhalf-$digits")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    #=@info " LM with $selected_h"
    lmtrain = residual(mnist_nls, LM_out.solution)
    lmtest = residual(mnist_nls_test, LM_out.solution)
    @show acc(lmtrain), acc(lmtest)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps\tikz_and_dats")
    lmdec = plot_svm(LM_out, LM_out.solution, "lm-lhalf-$digits")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    #nlm = NLPModels.neval_residual(nls_tr)
    nlm = LM_out.iter
    nglm = NLPModels.neval_jtprod_residual(mnist_nls) + NLPModels.neval_jprod_residual(mnist_nls)

    @info " LMTR with $selected_h" NormLinf(1.0)
    reset!(mnist_nls)
    lmtrtrain = residual(mnist_nls, LMTR_out.solution)
    lmtrtest = residual(mnist_nls_test, LMTR_out.solution)
    @show acc(lmtrtrain), acc(lmtrtest)
    #nlmtr = NLPModels.neval_residual(nls_tr)
    nlmtr = LMTR_out.iter
    nglmtr = NLPModels.neval_jtprod_residual(mnist_nls) + NLPModels.neval_jprod_residual(mnist_nls)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps\tikz_and_dats")
    lmtrdec = plot_svm(LMTR_out, LMTR_out.solution, "lmtr-lhalf-$digits")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")=#

    @info "using PLM to solve with $selected_h"
    local mnist_sto, mnist_nls_sto, mnist_sol = MNIST_train_model_sto(sample_rate0; digits = digits)
    local mnist_nlp_test_sto, mnist_nls_test_sto, mnist_sto_sol_test = MNIST_test_model_sto(sample_rate0; digits = digits)

    temp = [R2_stats.solver_specific[:smooth_obj], R2_stats.solver_specific[:nonsmooth_obj], R2_stats.objective, acc(r2train), acc(r2test), R2_stats.iter, r2_numjac_hist[end], R2_stats.iter, R2_stats.elapsed_time]

    for version in versions
        if selected_h != "smooth"
            med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob, PLM_outs, plm_trains, nplm, ngplm, epoch_counters_plm = load_mnist_plm(version, selected_h)
        else
            med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob, PLM_outs, plm_trains, nplm, ngplm = load_mnist_splm(version, selected_h)
        end

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
        plmtest = residual(mnist_nls_test, Prob_LM_out.solution)
        @show acc(plmtrain), acc(plmtest)
        #nplm = neval_residual(sampled_nls_tr)
        #nplm = length(mnist_nls.epoch_counter)
        #ngplm = (neval_jtprod_residual(mnist_nls) + neval_jprod_residual(mnist_nls))

        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps\tikz_and_dats")
        plmdec = plot_svm(Prob_LM_out, Prob_LM_out.solution, "prob-lm-$version-lhalf-$digits")
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
        
        if selected_h != "smooth"
            temp = hcat(temp,
                #[LM_out.solver_specific[:Fhist][end], LM_out.solver_specific[:Hhist][end], LM_out.objective, acc(lmtrain), acc(lmtest), nlm, nglm, sum(LM_out.solver_specific[:SubsolverCounter]), LM_out.elapsed_time],
                #[LMTR_out.solver_specific[:Fhist][end], LMTR_out.solver_specific[:Hhist][end], LMTR_out.objective, acc(lmtrtrain), acc(lmtrtest), nlmtr, nglmtr, sum(LMTR_out.solver_specific[:SubsolverCounter]), LMTR_out.elapsed_time],
                #[Sto_LM_out.solver_specific[:ExactFhist][end], Sto_LM_out.solver_specific[:Hhist][end], Sto_LM_out.solver_specific[:ExactFhist][end] + Sto_LM_out.solver_specific[:Hhist][end], acc(slmtrain), acc(slmtest), nslm, ngslm, sum(Sto_LM_out.solver_specific[:SubsolverCounter]), Sto_LM_out.elapsed_time],
                [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, acc(plmtrain), acc(plmtest), nplm, ngplm - 2 * Prob_LM_out.iter, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time])
        else
            temp = hcat(temp,
            #[LM_out.solver_specific[:Fhist][end], LM_out.solver_specific[:Hhist][end], LM_out.objective, acc(lmtrain), acc(lmtest), nlm, nglm, sum(LM_out.solver_specific[:SubsolverCounter]), LM_out.elapsed_time],
            #[LMTR_out.solver_specific[:Fhist][end], LMTR_out.solver_specific[:Hhist][end], LMTR_out.objective, acc(lmtrtrain), acc(lmtrtest), nlmtr, nglmtr, sum(LMTR_out.solver_specific[:SubsolverCounter]), LMTR_out.elapsed_time],
            #[Sto_LM_out.solver_specific[:ExactFhist][end], Sto_LM_out.solver_specific[:Hhist][end], Sto_LM_out.solver_specific[:ExactFhist][end] + Sto_LM_out.solver_specific[:Hhist][end], acc(slmtrain), acc(slmtest), nslm, ngslm, sum(Sto_LM_out.solver_specific[:SubsolverCounter]), Sto_LM_out.elapsed_time],
            [Prob_LM_out.solver_specific[:Fhist][end], 0.0, Prob_LM_out.objective, acc(plmtrain), acc(plmtest), nplm, ngplm - Prob_LM_out.iter, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time])
        end
    end

    for sample_rate in sample_rates
        med_obj_sto, med_metr_sto, med_mse_sto, std_obj_sto, std_metr_sto, std_mse_sto, SLM_outs, slm_trains, nslm, ngslm = load_mnist_sto(sample_rate, selected_h)
        @info "using SLM to solve"
        if n_exec%2 == 1 && sample_rate < 1.0
            med_ind = (n_exec ÷ 2) + 1
        elseif sample_rate == 1.0
            med_ind = 1
        else
            med_ind = (n_exec ÷ 2)
        end
        acc_vec = acc.(slm_trains)
        sorted_acc_vec = sort(acc_vec)
        ref_value = sorted_acc_vec[med_ind]
        origin_ind = 0
        for i in eachindex(SLM_outs)
            if acc_vec[i] == ref_value
                origin_ind = i
            end
        end

        SLM_out = SLM_outs[origin_ind]
        slmtrain = residual(mnist_nls, SLM_out.solution)
        slmtest = residual(mnist_nls_test, SLM_out.solution)
        @show acc(slmtrain), acc(slmtest)
        #nplm = neval_residual(sampled_nls_tr)
        #nsplm = length(mnist_nls.epoch_counter)
        #ngsplm = (neval_jtprod_residual(mnist_nls) + neval_jprod_residual(mnist_nls))

        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\greymaps\tikz_and_dats")
        plmdec = plot_svm(SLM_out, SLM_out.solution, "slm-$(sample_rate*100)-lhalf-$digits")
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

        if selected_h != "smooth"
            temp = hcat(temp,
            [SLM_out.solver_specific[:ExactFhist][end], SLM_out.solver_specific[:Hhist][end], SLM_out.solver_specific[:ExactFhist][end] + SLM_out.solver_specific[:Hhist][end], acc(slmtrain), acc(slmtest), nslm, ngslm - 2*SLM_out.iter, sum(SLM_out.solver_specific[:SubsolverCounter]), SLM_out.elapsed_time]
            )
        else
            temp = hcat(temp,
            [SLM_out.solver_specific[:ExactFhist][end], 0.0, SLM_out.solver_specific[:ExactFhist][end], acc(slmtrain), acc(slmtest), nslm, ngslm, sum(SLM_out.solver_specific[:SubsolverCounter]), SLM_out.elapsed_time]
            )
        end
    end

    temp = temp'

    df = DataFrame(temp, [:f, :h, :fh, :x, :xt, :n, :g, :p, :s])
    T = []
    for i = 1:nrow(df)
    push!(T, Tuple(df[i, [:x, :xt]]))
    end
    select!(df, Not(:xt))
    df[!, :x] = T
    df[!, :Alg] = vcat(["R2"], ["PLM-$(prob_versions_names[version])" for version in versions], ["PLM-$(sample_rate*100)" for sample_rate in sample_rates])
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
    open("svm-lhalf-$digits-$selected_h.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
end
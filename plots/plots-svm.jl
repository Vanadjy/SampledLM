function plot_Sampled_LM_SVM_epoch(sample_rates::AbstractVector, versions::AbstractVector, selected_probs::AbstractVector, selected_hs::AbstractVector, selected_digits::AbstractVector; abscissa = "CPU time", n_exec = 10, smooth::Bool = false, sample_rate0::Float64 = .05, param::String = "MSE", compare::Bool = false, guide::Bool = false, MaxEpochs::Int = 1000, MaxTime = 3600.0, precision = 1e-4)
    compound = 1
    include("plot-configuration.jl")

    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    for selected_prob in selected_probs
        for selected_h in selected_hs
            for digits in selected_digits

                data_obj = GenericTrace{Dict{Symbol, Any}}[]
                data_metr = GenericTrace{Dict{Symbol, Any}}[]
                data_mse = GenericTrace{Dict{Symbol, Any}}[]

                #plots of other algorithms
                if compare && !smooth
                    bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound)
                    mnist_full, mnist_nls_full, mnist_nls_sol = RegularizedProblems.svm_train_model()
                    mnist_nlp_test, mnist_nls_test, mnist_sol_test = RegularizedProblems.svm_test_model()
                    A_ijcnn1, b_ijcnn1 = ijcnn1_generate_data_tr()
                    ijcnn1_full, ijcnn1_nls_full = RegularizedProblems.svm_model(A_ijcnn1', b_ijcnn1)
                    sampled_options_full = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = precision, ϵr = precision, verbose = 1, σmin = 1e-5, maxIter = 2*MaxEpochs, maxTime = MaxTime;)
                    subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))

                    if selected_prob == "ijcnn1"
                        prob = ijcnn1_full
                        prob_nls = ijcnn1_nls_full
                    elseif selected_prob == "mnist"
                        prob = mnist_full
                        prob_nls = mnist_nls_full
                    end

                    x0 = ones(prob.meta.nvar)
                    m = prob_nls.nls_meta.nequ
                    l_bound = prob.meta.lvar
                    u_bound = prob.meta.uvar

                    λ = .1
                    if selected_h == "l0"
                        h = NormL0(λ)
                    elseif selected_h == "l1"
                        h = NormL1(λ)
                    elseif selected_h == "lhalf"
                        h = RootNormLhalf(λ)
                    elseif selected_h == "smooth"
                        h = NormL1(0.0)
                    end

                    @info "using R2 to solve with" h
                    reset!(prob)
                    #xk_R2, k_R2, R2_out = RegularizedOptimization.R2(prob.f, prob.∇f!, h, sampled_options_full, x0)
                    prob_lsr1 = LSR1Model(prob)
                    R2_stats = RegularizedOptimization.R2(prob_lsr1, h, sampled_options_full, x0 = x0)

                    ## For R2 metric callback ##
                    reg_prob = RegularizedNLPModel(prob, h)
                    reg_stats = GenericExecutionStats(reg_prob.model)
                    reg_solver = RegularizedOptimization.R2Solver(x0, sampled_options_full, l_bound, u_bound; ψ = shifted(h, x0))
                    cb = (nlp, solver, stats) -> begin
                                                    solver.Fobj_hist[stats.iter+1] = stats.solver_specific[:smooth_obj] + stats.solver_specific[:nonsmooth_obj]
                                                    solver.Hobj_hist[stats.iter+1] = stats.solver_specific[:xi]
                                                    solver.Complex_hist[stats.iter+1] += 1
                                                 end
                    RegularizedOptimization.solve!(reg_solver, reg_prob, reg_stats; callback = cb)
                    r2_metric_hist = filter!(!iszero, reg_solver.Hobj_hist)

                    nr2 = NLPModels.neval_obj(prob_lsr1)
                    ngr2 = NLPModels.neval_grad(prob_lsr1)
                    r2train = residual(prob_nls, R2_stats.solution) #||e - tanh(b * <A, x>)||^2, b ∈ {-1,1}^n
                    if selected_prob == "mnist"
                        r2test = residual(mnist_nls_test, R2_stats.solution)
                        @show acc(r2train), acc(r2test)
                    end
                    r2dec = plot_svm(R2_stats, R2_stats.solution, "r2-lhalf-$digits")

                    @info " using LM to solve with" h
                    LM_out = LM(prob_nls, h, sampled_options_full; x0 = prob_nls.meta.x0, subsolver_options = subsolver_options)
                    lmtrain = residual(prob_nls, LM_out.solution)
                    if selected_prob == "mnist"
                        lmtest = residual(mnist_nls_test, LM_out.solution)
                        @show acc(lmtrain), acc(lmtest)
                    end
                    lmdec = plot_svm(LM_out, LM_out.solution, "lm-lhalf-$digits")
                    #nlm = NLPModels.neval_residual(nls_tr)
                    nlm = LM_out.iter
                    nglm = NLPModels.neval_jtprod_residual(prob_nls) + NLPModels.neval_jprod_residual(prob_nls)

                    if (h == NormL0(λ)) || (h == RootNormLhalf(λ))
                        @info " using LMTR to solve with" h NormLinf(1.0)
                        reset!(prob_nls)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormLinf(1.0), sampled_options_full; x0 = prob_nls.meta.x0, subsolver_options = subsolver_options)
                        lmtrtrain = residual(prob_nls, LMTR_out.solution)
                        if selected_prob == "mnist"
                            lmtrtest = residual(mnist_nls_test, LMTR_out.solution)
                            @show acc(lmtrtrain), acc(lmtrtest)
                        end
                        #nlmtr = NLPModels.neval_residual(nls_tr)
                        nlmtr = LMTR_out.iter
                        nglmtr = NLPModels.neval_jtprod_residual(prob_nls) + NLPModels.neval_jprod_residual(prob_nls)
                        lmtrdec = plot_svm(LMTR_out, LMTR_out.solution, "lmtr-lhalf-$digits")
                    elseif h == NormL1(λ)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormL2(1.0), sampled_options_full; x0 = prob_nls.meta.x0, subsolver_options = subsolver_options)
                    elseif h == NormL1(0.0)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormL2(1.0), sampled_options_full; x0 = prob_nls.meta.x0, subsolver_options = subsolver_options)
                    end

                    #save_object("k_R2-$selected_prob-$selected_h.jld2", k_R2)
                    #save_object("R2_out-$selected_prob-$selected_h.jld2", R2_out)
                    save_object("R2_stats-$selected_prob-$selected_h.jld2", R2_stats)
                    save_object("r2_metric_hist-$selected_prob-$selected_h.jld2", r2_metric_hist)
                    save_object("LM_out-$selected_prob-$selected_h.jld2", LM_out)
                    save_object("LMTR_out-$selected_prob-$selected_h.jld2", LMTR_out)

                    # --------------- OBJECTIVE DATA -------------------- #

                    data_obj_r2 = PlotlyJS.scatter(; x = 1:length(R2_stats.solver_specific[:Fhist]) , y = R2_stats.solver_specific[:Fhist] + R2_stats.solver_specific[:Hhist], mode="lines", name = "R2", line=attr(
                        color="rgb(220,20,60)", dash = "dashdot", width = 1
                        )
                    )

                    data_obj_lm = PlotlyJS.scatter(; x = 1:length(LM_out.solver_specific[:Fhist]) , y = LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], mode="lines", name = "LM", line=attr(
                        color="rgb(255,165,0)", dash = "dot", width = 1
                        )
                    )
                    
                    data_obj_lmtr = PlotlyJS.scatter(; x = 1:length(LMTR_out.solver_specific[:Fhist]) , y = LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], mode="lines", name = "LMTR", line=attr(
                            color="black", dash = "dash", width = 1
                            )
                    )

                    push!(data_obj, data_obj_r2, data_obj_lm, data_obj_lmtr)
                    
                    # --------------- MSE DATA -------------------- #

                    data_mse_r2 = PlotlyJS.scatter(; x = 1:k_R2 , y = 0.5*(R2_out[:Fhist] + R2_out[:Hhist])/m, mode="lines", name = "R2", line=attr(
                        color="rgb(220,20,60)", dash = "dashdot", width = 1
                        )
                    )

                    data_mse_lm = PlotlyJS.scatter(; x = 1:length(LM_out.solver_specific[:Fhist]) , y = 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, mode="lines", name = "LM", line=attr(
                        color="rgb(255,165,0)", dash = "dot", width = 1
                        )
                    )
                    
                    data_mse_lmtr = PlotlyJS.scatter(; x = 1:length(LMTR_out.solver_specific[:Fhist]) , y = 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, mode="lines", name = "LMTR", line=attr(
                            color="black", dash = "dash", width = 1
                            )
                    )

                    push!(data_mse, data_mse_r2, data_mse_lm, data_mse_lmtr)
                elseif compare && smooth
                    ## -------------------------------------------------------------------------------------------- ##
                    ## ---------------------------------- SMOOTH VERSIONS ------------------------------------------ ##
                    ## -------------------------------------------------------------------------------------------- ##
                    bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound)
                    mnist_full, mnist_nls_full, mnist_nls_sol = RegularizedProblems.svm_train_model()
                    mnist_nlp_test, mnist_nls_test, mnist_sol_test = RegularizedProblems.svm_test_model()
                    A_ijcnn1, b_ijcnn1 = ijcnn1_load_data()
                    ijcnn1_full, ijcnn1_nls_full = RegularizedProblems.svm_model(A_ijcnn1', b_ijcnn1)
                    sampled_options_full = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = precision, ϵr = precision, verbose = 10, σmin = 1e-5, maxIter = MaxEpochs, maxTime = MaxTime;)
                    subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))

                    if selected_prob == "ijcnn1"
                        prob = ijcnn1_full
                        prob_nls = ijcnn1_nls_full
                    elseif selected_prob == "mnist"
                        prob = mnist_full
                        prob_nls = mnist_nls_full
                    end

                    x0 = ones(prob.meta.nvar)
                    m = prob_nls.nls_meta.nequ
                    l_bound = prob.meta.lvar
                    u_bound = prob.meta.uvar

                    λ = .1
                    if selected_h == "l0"
                        h = NormL0(λ)
                    elseif selected_h == "l1"
                        h = NormL1(λ)
                    elseif selected_h == "lhalf"
                        h = RootNormLhalf(λ)
                    elseif selected_h == "smooth"
                        h = NormL1(0.0)
                    end

                    @info "using R2 to solve with" h
                    reset!(prob)
                    xk_R2, k_R2, R2_out = JSOSolvers.R2(prob.f, prob.∇f!, sampled_options_full, x0)
                    prob = LSR1Model(prob)
                    R2_stats = JSOSolvers.R2(prob, sampled_options_full, x0 = x0)
                    nr2 = NLPModels.neval_obj(prob)
                    ngr2 = NLPModels.neval_grad(prob)
                    r2train = residual(prob_nls, R2_stats.solution) #||e - tanh(b * <A, x>)||^2, b ∈ {-1,1}^n
                    if selected_prob == "mnist"
                        r2test = residual(mnist_nls_test, R2_stats.solution)
                        @show acc(r2train), acc(r2test)
                    end
                    r2dec = plot_svm(R2_stats, R2_stats.solution, "r2-lhalf-$digits")

                    @info " using LM to solve with" h
                    LM_out = LM(prob_nls, h, sampled_options_full; x0 = prob_nls.meta.x0, subsolver_options = subsolver_options)
                    lmtrain = residual(prob_nls, LM_out.solution)
                    if selected_prob == "mnist"
                        lmtest = residual(mnist_nls_test, LM_out.solution)
                        @show acc(lmtrain), acc(lmtest)
                    end
                    lmdec = plot_svm(LM_out, LM_out.solution, "lm-lhalf-$digits")
                    #nlm = NLPModels.neval_residual(nls_tr)
                    nlm = LM_out.iter
                    nglm = NLPModels.neval_jtprod_residual(prob_nls) + NLPModels.neval_jprod_residual(prob_nls)

                    if (h == NormL0(λ)) || (h == RootNormLhalf(λ))
                        @info " using LMTR to solve with" h NormLinf(1.0)
                        reset!(prob_nls)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormLinf(1.0), sampled_options_full; x0 = prob_nls.meta.x0, subsolver_options = subsolver_options)
                        lmtrtrain = residual(prob_nls, LMTR_out.solution)
                        if selected_prob == "mnist"
                            lmtrtest = residual(mnist_nls_test, LMTR_out.solution)
                            @show acc(lmtrtrain), acc(lmtrtest)
                        end
                        #nlmtr = NLPModels.neval_residual(nls_tr)
                        nlmtr = LMTR_out.iter
                        nglmtr = NLPModels.neval_jtprod_residual(prob_nls) + NLPModels.neval_jprod_residual(prob_nls)
                        lmtrdec = plot_svm(LMTR_out, LMTR_out.solution, "lmtr-lhalf-$digits")
                    elseif h == NormL1(λ)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormL2(1.0), sampled_options_full; x0 = prob_nls.meta.x0, subsolver_options = subsolver_options)
                    elseif h == NormL1(0.0)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormL2(1.0), sampled_options_full; x0 = prob_nls.meta.x0, subsolver_options = subsolver_options)
                    end

                    save_object("k_R2-$selected_prob-$selected_h.jld2", k_R2)
                    save_object("R2_out-$selected_prob-$selected_h.jld2", R2_out)
                    save_object("R2_stats-$selected_prob-$selected_h.jld2", R2_stats)
                    save_object("LM_out-$selected_prob-$selected_h.jld2", LM_out)
                    save_object("LMTR_out-$selected_prob-$selected_h.jld2", LMTR_out)

                    # --------------- OBJECTIVE DATA -------------------- #

                    data_obj_r2 = PlotlyJS.scatter(; x = 1:k_R2 , y = R2_out[:Fhist] + R2_out[:Hhist], mode="lines", name = "R2", line=attr(
                        color="rgb(220,20,60)", dash = "dashdot", width = 1
                        )
                    )

                    data_obj_lm = PlotlyJS.scatter(; x = 1:length(LM_out.solver_specific[:Fhist]) , y = LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], mode="lines", name = "LM", line=attr(
                        color="rgb(255,165,0)", dash = "dot", width = 1
                        )
                    )
                    
                    data_obj_lmtr = PlotlyJS.scatter(; x = 1:length(LMTR_out.solver_specific[:Fhist]) , y = LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], mode="lines", name = "LMTR", line=attr(
                            color="black", dash = "dash", width = 1
                            )
                    )

                    push!(data_obj, data_obj_r2, data_obj_lm, data_obj_lmtr)
                    
                    # --------------- MSE DATA -------------------- #

                    data_mse_r2 = PlotlyJS.scatter(; x = 1:k_R2 , y = 0.5*(R2_out[:Fhist] + R2_out[:Hhist])/m, mode="lines", name = "R2", line=attr(
                        color="rgb(220,20,60)", dash = "dashdot", width = 1
                        )
                    )

                    data_mse_lm = PlotlyJS.scatter(; x = 1:length(LM_out.solver_specific[:Fhist]) , y = 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, mode="lines", name = "LM", line=attr(
                        color="rgb(255,165,0)", dash = "dot", width = 1
                        )
                    )
                    
                    data_mse_lmtr = PlotlyJS.scatter(; x = 1:length(LMTR_out.solver_specific[:Fhist]) , y = 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, mode="lines", name = "LMTR", line=attr(
                            color="black", dash = "dash", width = 1
                            )
                    )

                    push!(data_mse, data_mse_r2, data_mse_lm, data_mse_lmtr)
                end


                ## ------------------------------------------------------------------------------------------- ##
                ## -------------------------------- SAMPLED VERSION ------------------------------------------ ##
                ## ------------------------------------------------------------------------------------------- ##


                if smooth
                    ## -------------------------------------------------------------------------------------------- ##
                    ## ---------------------------------- SMOOTH VERSIONS ------------------------------------------ ##
                    ## -------------------------------------------------------------------------------------------- ##

                    for sample_rate in sample_rates
                        nz = 10 * compound
                        #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                        sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 1.0, β = 1e16, σmax = 1e16, σmin = 1e-5, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                        local subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))
                        local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate)
                        #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
                        local ijcnn1, ijcnn1_nls, ijcnn1_sol = ijcnn1_model_sto(sample_rate)
                        #a9a, a9a_nls = a9a_model_sto(sample_rate)
                        local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate; digits = digits)
                        local mnist_test, mnist_nls_test, mnist_sol_test = MNIST_test_model_sto(sample_rate; digits = digits)
                        #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)
                        local λ = .1

                        if selected_prob == "ijcnn1"
                            nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
                        elseif selected_prob == "mnist"
                            nls_prob_collection = [(mnist_nls, "mnist-train-ls")]
                        end

                        Obj_Hists_epochs_sto = zeros(1 + MaxEpochs, n_exec)
                        Metr_Hists_epochs_sto = zero(Obj_Hists_epochs_sto)
                        MSE_Hists_epochs_sto = zero(Obj_Hists_epochs_sto)

                        for (prob, prob_name) in nls_prob_collection
                            if selected_h == "l0"
                                h = NormL0(λ)
                                h_name = "l0-norm"
                            elseif selected_h == "l1"
                                h = NormL1(λ)
                                h_name = "l1-norm"
                            elseif selected_h == "lhalf"
                                h = RootNormLhalf(λ)
                                h_name = "lhalf-norm"
                            elseif selected_h == "smooth"
                                h = NormL1(0.0)
                            end

                            SLM_outs = []
                            slm_trains = []

                            for k in 1:n_exec
                                # executes n_exec times Sto_LM with the same inputs
                                x0 = digits[1] * ones(prob.meta.nvar)
                                #p = randperm(prob.meta.nvar)[1:nz]
                                #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
                                reset!(prob)
                                #try
                                PLM_out = SSLM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                                push!(SLM_outs, PLM_out)
                                push!(slm_trains, residual(prob, PLM_out.solution))

                                sample_size = length(prob.sample)

                                selected_iterations = (sample_rate == 1.0) ? prob.epoch_counter[2:end] : prob.epoch_counter[1:end-1]

                                # get objective value for each run #
                                @views Obj_Hists_epochs_sto[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:ExactFhist][selected_iterations]

                                # get MSE for each run #
                                @views MSE_Hists_epochs_sto[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:Fhist][selected_iterations]
                                @views MSE_Hists_epochs_sto[:, k][1:length(selected_iterations)] ./= (2 * sample_size)

                                # get metric for each run #
                                @views Metr_Hists_epochs_sto[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:ExactMetricHist][selected_iterations]

                                if k < n_exec # reset epoch counter for each run
                                    prob.epoch_counter = Int[1]
                                end
                            end

                            save_object("SSLM_outs-$(sample_rate*100)%-$selected_prob-$selected_h.jld2", SLM_outs)
                            save_object("sslm_trains-$(sample_rate*100)%-$selected_prob-$selected_h.jld2", slm_trains)

                            if n_exec%2 == 1
                                med_ind = (n_exec ÷ 2) + 1
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

                            Prob_LM_out = SLM_outs[origin_ind]
                            slmtrain = residual(prob, Prob_LM_out.solution)
                            if prob_name == "mnist-train-ls"
                                slmtest = residual(mnist_nls_test, Prob_LM_out.solution)
                                @show acc(slmtrain), acc(slmtest)
                            end
                            #nplm = neval_residual(sampled_nls_tr)
                            nslm = length(prob.epoch_counter)-1
                            save_object("nsslm-mnist-PLM-$(100*sample_rate)%.jld2", nslm)
                            ngslm = (neval_jtprod_residual(prob) + neval_jprod_residual(prob))
                            save_object("ngsslm-mnist-PLM-$(100*sample_rate)%.jld2", ngslm)
                            if prob_name == "mnist-train-ls"
                                slmdec = plot_svm(Prob_LM_out, Prob_LM_out.solution, "sto-lm-$(100*sample_rate)%-lhalf-$digits")
                            end

                            med_obj_sto = zeros(axes(Obj_Hists_epochs_sto, 1))
                            std_obj_sto = zeros(axes(Obj_Hists_epochs_sto, 1))

                            med_metr_sto = zeros(axes(Metr_Hists_epochs_sto, 1))
                            std_metr_sto = zeros(axes(Metr_Hists_epochs_sto, 1))

                            med_mse_sto = zeros(axes(MSE_Hists_epochs_sto, 1))
                            std_mse_sto = zeros(axes(MSE_Hists_epochs_sto, 1))

                            # compute median of objective value #
                            for l in eachindex(med_obj_sto)
                                cleared_obj_sto = filter(!isnan, filter(!iszero, Obj_Hists_epochs_sto[l, :]))
                                if isempty(cleared_obj_sto)
                                    med_obj_sto[l] = NaN
                                    std_obj_sto[l] = NaN
                                elseif length(cleared_obj_sto) == 1
                                    med_obj_sto[l] = median(cleared_obj_sto)
                                    std_obj_sto[l] = 0.0
                                else
                                    med_obj_sto[l] = median(cleared_obj_sto)
                                    std_obj_sto[l] = std(cleared_obj_sto)
                                end
                            end
                            filter!(!isnan, med_obj_sto)
                            std_obj_sto *= Confidence[conf]
                            filter!(!isnan, std_obj_sto)
                            save_object("med_obj_sto_smooth-$(sample_rate*100)-$prob_name-$selected_h.jld2", med_obj_sto)
                            save_object("std_obj_sto_smooth-$(sample_rate*100)-$prob_name-$selected_h.jld2", std_obj_sto)

                            # compute median of metric #
                            for l in eachindex(med_metr_sto)
                                cleared_metr_sto = filter(!isnan, filter(!iszero, Metr_Hists_epochs_sto[l, :]))
                                if isempty(cleared_metr_sto)
                                    med_metr_sto[l] = NaN
                                    std_metr_sto[l] = NaN
                                elseif length(cleared_metr_sto) == 1
                                    med_metr_sto[l] = median(cleared_metr_sto)
                                    std_metr_sto[l] = 0.0
                                else
                                    med_metr_sto[l] = median(cleared_metr_sto)
                                    std_metr_sto[l] = std(cleared_metr_sto)
                                end
                            end
                            filter!(!isnan, med_metr_sto)
                            std_metr_sto *= Confidence[conf]
                            filter!(!isnan, std_metr_sto)
                            save_object("med_metr_sto_smooth-$(sample_rate*100)-$selected_prob-$selected_h.jld2", med_metr_sto)
                            save_object("std_metr_sto_smooth-$(sample_rate*100)-$selected_prob-$selected_h.jld2", std_metr_sto)
                            
                            # compute median of MSE #
                            for l in eachindex(med_mse_sto)
                                cleared_mse_sto = filter(!isnan, filter(!iszero, MSE_Hists_epochs_sto[l, :]))
                                if isempty(cleared_mse_sto)
                                    med_mse_sto[l] = NaN
                                    std_mse_sto[l] = NaN
                                elseif length(cleared_mse_sto) == 1
                                    med_mse_sto[l] = median(cleared_mse_sto)
                                    std_mse_sto[l] = 0.0
                                else
                                    med_mse_sto[l] = (median(cleared_mse_sto))
                                    std_mse_sto[l] = std(cleared_mse_sto)
                                end
                            end
                            filter!(!isnan, med_mse_sto)
                            std_mse_sto *= Confidence[conf]
                            filter!(!isnan, std_mse_sto)
                            save_object("med_mse_sto_smooth-$(sample_rate*100)-$selected_prob-$selected_h.jld2", med_mse_sto)
                            save_object("std_mse_sto_smooth-$(sample_rate*100)-$selected_prob-$selected_h.jld2", std_mse_sto)

                            # --------------- OBJECTIVE DATA -------------------- #

                            data_obj_slm = PlotlyJS.scatter(; x = 1:length(med_obj_sto), y = med_obj_sto, mode="lines", name = "SLM - $(sample_rate*100)%", line=attr(
                                color = color_scheme[sample_rate], width = 1
                                )
                            )

                            data_std_obj_slm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_sto), length(med_obj_sto):-1:1), y = vcat(med_obj_sto + std_obj_sto, reverse!(med_obj_sto - std_obj_sto)), mode="lines", name = "SLM - $(sample_rate*100)%", fill="tozerox",
                                fillcolor = color_scheme_std[sample_rate],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_obj, data_obj_slm, data_std_obj_slm)

                            # --------------- METRIC DATA -------------------- #

                            data_metr_slm = PlotlyJS.scatter(; x = 1:length(med_metr_sto), y = med_metr_sto, mode="lines", name = "SLM - $(sample_rate*100)%", line=attr(
                                color = color_scheme[sample_rate], width = 1
                                )
                            )
                            reverse = reverse!(med_metr_sto - std_metr_sto)
                            for l in eachindex(reverse)
                                if reverse[l] < 0
                                    reverse[l] = med_metr_sto[l]
                                end
                            end


                            data_std_metr_slm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_sto), length(med_metr_sto):-1:1), y = vcat(med_metr_sto + std_metr_sto, reverse), mode="lines", name = "SLM - $(sample_rate*100)%", fill="tozerox",
                                fillcolor = color_scheme_std[sample_rate],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_metr, data_metr_slm)#, data_std_metr_slm)
                            
                            # --------------- MSE DATA -------------------- #

                            data_mse_slm = PlotlyJS.scatter(; x = 1:length(med_mse_sto), y = med_mse_sto, mode="lines", name = "SLM - $(sample_rate*100)%", line=attr(
                                color = color_scheme[sample_rate], width = 1
                                )
                            )

                            data_std_mse_slm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_sto), length(med_mse_sto):-1:1), y = vcat(med_mse_sto + std_mse_sto, reverse!(med_mse_sto - std_mse_sto)), mode="lines", name = "SLM - $(sample_rate*100)%", fill="tozerox",
                                fillcolor = color_scheme_std[sample_rate],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_mse, data_mse_slm, data_std_mse_slm)
                            
                            # reset epoch counter for next problem
                            prob.epoch_counter = Int[1]
                        end
                    end

                    for version in versions
                        nz = 10 * compound
                        #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                        sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 1.0, β = 1e16, σmax = 1e16, σmin = 1e-5, μmin = 1e-5, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                        local subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))
                        local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate0)
                        #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
                        local ijcnn1, ijcnn1_nls, ijcnn1_sol = ijcnn1_model_sto(sample_rate0)
                        #a9a, a9a_nls = a9a_model_sto(sample_rate)
                        local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate0; digits = digits)
                        local mnist_nlp_test_sto, mnist_nls_test_sto, mnist_sto_sol_test = MNIST_test_model_sto(sample_rate0; digits = digits)
                        #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)
                        local λ = .1
                        if selected_prob == "ijcnn1"
                            nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
                        elseif selected_prob == "mnist"
                            nls_prob_collection = [(mnist_nls, "mnist-train-ls")]
                        end
                        
                        Obj_Hists_epochs_prob = zeros(1 + MaxEpochs, n_exec)
                        Metr_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)
                        MSE_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)
                        for (prob, prob_name) in nls_prob_collection
                            if selected_h == "l0"
                                h = NormL0(λ)
                                h_name = "l0-norm"
                            elseif selected_h == "l1"
                                h = NormL1(λ)
                                h_name = "l1-norm"
                            elseif selected_h == "lhalf"
                                h = RootNormLhalf(λ)
                                h_name = "lhalf-norm"
                            elseif selected_h == "smooth"
                                h = NormL1(0.0)
                            end

                            @info "using PLM to solve with" h
                            # routine to select the output with the median accuracy on the training set
                            SPLM_outs = []
                            splm_trains = []

                            for k in 1:n_exec
                                # executes n_exec times Sto_LM with the same inputs
                                x0 = ones(prob.meta.nvar)
                                #p = randperm(prob.meta.nvar)[1:nz]
                                #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
                                reset!(prob)
                                prob.epoch_counter = Int[1]
                                PLM_out = SPLM(prob, sampled_options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version, Jac_lop = true)
                                push!(SPLM_outs, PLM_out)
                                push!(splm_trains, residual(prob, PLM_out.solution))
                                #PLM_out = SPLM(prob, sampled_options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version)

                                selected_iterations = (sample_rate0 == 1.0) ? prob.epoch_counter[2:end] : prob.epoch_counter[1:end-1]

                                # get objective value for each run #
                                @views Obj_Hists_epochs_prob[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:ExactFhist][selected_iterations]

                                # get MSE for each run #
                                @views MSE_Hists_epochs_prob[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:Fhist][selected_iterations]
                                @views MSE_Hists_epochs_prob[:, k][1:length(selected_iterations)] ./= ceil.(2 * prob.nls_meta.nequ * PLM_out.solver_specific[:SampleRateHist][selected_iterations])

                                # get metric for each run #
                                @views Metr_Hists_epochs_prob[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:ExactMetricHist][selected_iterations]
                            end

                            save_object("SPLM_outs-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", SPLM_outs)
                            save_object("splm_trains-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", splm_trains)

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
                            splmtrain = residual(prob, SProb_LM_out.solution)
                            if prob_name == "mnist-train-ls"
                                splmtest = residual(mnist_nls_test_sto, SProb_LM_out.solution)
                                @show acc(splmtrain), acc(splmtest)
                            end
                            #nplm = neval_residual(sampled_nls_tr)
                            nsplm = length(prob.epoch_counter) - 1
                            save_object("nsplm-mnist-PLM-$(prob_versions_names[version]).jld2", nsplm)
                            ngsplm = (neval_jtprod_residual(prob) + neval_jprod_residual(prob))
                            save_object("ngsplm-mnist-PLM-$(prob_versions_names[version]).jld2", ngsplm)
                            if prob_name == "mnist-train-ls"
                                plmdec = plot_svm(SProb_LM_out, SProb_LM_out.solution, "smooth-prob-lm-$(prob_versions_names[version])-lhalf-$digits")
                            end

                            med_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))
                            std_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))

                            med_metr_prob = zeros(axes(Metr_Hists_epochs_prob, 1))
                            std_metr_prob = zeros(axes(Metr_Hists_epochs_prob, 1))

                            med_mse_prob = zeros(axes(MSE_Hists_epochs_prob, 1))
                            std_mse_prob = zeros(axes(MSE_Hists_epochs_prob, 1))

                            # compute median of objective value #
                            for l in eachindex(med_obj_prob)
                                cleared_obj_prob = filter(!isnan, filter(!iszero, Obj_Hists_epochs_prob[l, :]))
                                if isempty(cleared_obj_prob)
                                    med_obj_prob[l] = NaN
                                    std_obj_prob[l] = NaN
                                elseif length(cleared_obj_prob) == 1
                                    med_obj_prob[l] = median(cleared_obj_prob)
                                    std_obj_prob[l] = 0.0
                                else
                                    med_obj_prob[l] = median(cleared_obj_prob)
                                    std_obj_prob[l] = std(cleared_obj_prob)
                                end
                            end
                            filter!(!isnan, med_obj_prob)
                            std_obj_prob *= Confidence[conf]
                            filter!(!isnan, std_obj_prob)

                            save_object("med_obj_prob_smooth-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", med_obj_prob)
                            save_object("std_obj_prob_smooth-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", std_obj_prob)

                            # compute median of metric #
                            for l in eachindex(med_metr_prob)
                                cleared_metr_prob = filter(!isnan, filter(!iszero, Metr_Hists_epochs_prob[l, :]))
                                if isempty(cleared_metr_prob)
                                    med_metr_prob[l] = NaN
                                    std_metr_prob[l] = NaN
                                elseif length(cleared_metr_prob) == 1
                                    med_metr_prob[l] = median(cleared_metr_prob)
                                    std_metr_prob[l] = 0.0
                                else
                                    med_metr_prob[l] = median(cleared_metr_prob)
                                    std_metr_prob[l] = std(cleared_metr_prob)
                                end
                            end
                            filter!(!isnan, med_metr_prob)
                            std_metr_prob *= Confidence[conf]
                            filter!(!isnan, std_metr_prob)

                            save_object("med_metr_prob_smooth-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", med_metr_prob)
                            save_object("std_metr_prob_smooth-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", std_metr_prob)
                            
                            # compute median of MSE #
                            for l in eachindex(med_mse_prob)
                                cleared_mse_prob = filter(!isnan, filter(!iszero, MSE_Hists_epochs_prob[l, :]))
                                if isempty(cleared_mse_prob)
                                    med_mse_prob[l] = NaN
                                    std_mse_prob[l] = NaN
                                elseif length(cleared_mse_prob) == 1
                                    med_mse_prob[l] = median(cleared_mse_prob)
                                    std_mse_prob[l] = 0.0
                                else
                                    med_mse_prob[l] = (median(cleared_mse_prob))
                                    std_mse_prob[l] = std(cleared_mse_prob)
                                end
                            end
                            filter!(!isnan, med_mse_prob)
                            std_mse_prob *= Confidence[conf]
                            filter!(!isnan, std_mse_prob)

                            save_object("med_mse_prob_smooth-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", med_mse_prob)
                            save_object("std_mse_prob_smooth-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", std_mse_prob)

                            # --------------- OBJECTIVE DATA -------------------- #

                            data_obj_splm = PlotlyJS.scatter(; x = 1:length(med_obj_prob), y = med_obj_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
                                color = smooth_versions_colors[version], width = 1
                                )
                            )

                            data_std_obj_splm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_prob), length(med_obj_prob):-1:1), y = vcat(med_obj_prob + std_obj_prob, reverse!(med_obj_prob - std_obj_prob)), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
                                fillcolor = smooth_versions_colors_std[version],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_obj, data_obj_splm)#, data_std_obj_splm)

                            # --------------- METRIC DATA -------------------- #

                            data_metr_splm = PlotlyJS.scatter(; x = 1:length(med_metr_prob), y = med_metr_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
                                color = smooth_versions_colors[version], width = 1
                                )
                            )

                            reverse = reverse!(med_metr_prob - std_metr_prob)
                            for l in eachindex(reverse)
                                if reverse[l] < 0
                                    reverse[l] = med_metr_prob[l]
                                end
                            end

                            data_std_metr_splm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_prob), length(med_metr_prob):-1:1), y = vcat(med_metr_prob + std_metr_prob, reverse), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
                                fillcolor = smooth_versions_colors_std[version],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_metr, data_metr_splm)#, data_std_metr_splm)
                            
                            # --------------- MSE DATA -------------------- #

                            data_mse_splm = PlotlyJS.scatter(; x = 1:length(med_mse_prob), y = med_mse_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
                                color = smooth_versions_colors[version], width = 1
                                )
                            )

                            data_std_mse_splm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_prob), length(med_mse_prob):-1:1), y = vcat(med_mse_prob + std_mse_prob, reverse!(med_mse_prob - std_mse_prob)), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
                                fillcolor = smooth_versions_colors_std[version],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_mse, data_mse_splm)#, data_std_mse_splm)

                            prob.epoch_counter = Int[1]

                            if compare
                                temp = hcat([R2_stats.solver_specific[:Fhist][end], R2_stats.solver_specific[:Hhist][end],R2_stats.objective, acc(r2train), acc(r2test), nr2, ngr2, sum(R2_stats.solver_specific[:SubsolverCounter]), R2_stats.elapsed_time],
                                    [LM_out.solver_specific[:Fhist][end], LM_out.solver_specific[:Hhist][end], LM_out.objective, acc(lmtrain), acc(lmtest), nlm, nglm, sum(LM_out.solver_specific[:SubsolverCounter]), LM_out.elapsed_time],
                                    [LMTR_out.solver_specific[:Fhist][end], LMTR_out.solver_specific[:Hhist][end], LMTR_out.objective, acc(lmtrtrain), acc(lmtrtest), nlmtr, nglmtr, sum(LMTR_out.solver_specific[:SubsolverCounter]), LMTR_out.elapsed_time],
                                    #[Sto_LM_out.solver_specific[:ExactFhist][end], Sto_LM_out.solver_specific[:Hhist][end], Sto_LM_out.solver_specific[:ExactFhist][end] + Sto_LM_out.solver_specific[:Hhist][end], acc(slmtrain), acc(slmtest), nslm, ngslm, sum(Sto_LM_out.solver_specific[:SubsolverCounter]), Sto_LM_out.elapsed_time],
                                    [SProb_LM_out.objective, 0.0, SProb_LM_out.objective, acc(splmtrain), acc(splmtest), nsplm, ngsplm, sum(SProb_LM_out.solver_specific[:SubsolverCounter]), SProb_LM_out.elapsed_time])

                                temp = temp'

                                df = DataFrame(temp, [:f, :h, :fh, :x, :xt, :n, :g, :p, :s])
                                T = []
                                for i = 1:nrow(df)
                                push!(T, Tuple(df[i, [:x, :xt]]))
                                end
                                select!(df, Not(:xt))
                                df[!, :x] = T
                                df[!, :Alg] = ["R2", "LM", "LMTR", "SPLM-$(prob_versions_names[version])"]
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
                                open("svm-lhalf-$digits.tex", "w") do io
                                    SolverBenchmark.pretty_latex_stats(io, df,
                                        col_formatters=fmt_override,
                                        hdr_override=hdr_override)
                                end
                            end
                        end
                    end
                else
                    for sample_rate in sample_rates
                        nz = 10 * compound
                        #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                        sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 1.0, β = 1e16, σmax = 1e16, σmin = 1e-5, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                        local subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))
                        local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate)
                        #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
                        local ijcnn1, ijcnn1_nls, ijcnn1_sol = ijcnn1_model_sto(sample_rate)
                        #a9a, a9a_nls = a9a_model_sto(sample_rate)
                        local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate; digits = digits)
                        local mnist_test, mnist_nls_test, mnist_sol_test = MNIST_test_model_sto(sample_rate; digits = digits)
                        #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)
                        local λ = .1

                        if selected_prob == "ijcnn1"
                            nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
                        elseif selected_prob == "mnist"
                            nls_prob_collection = [(mnist_nls, "mnist-train-ls")]
                        end

                        Obj_Hists_epochs_sto = zeros(1 + MaxEpochs, n_exec)
                        Metr_Hists_epochs_sto = zero(Obj_Hists_epochs_sto)
                        MSE_Hists_epochs_sto = zero(Obj_Hists_epochs_sto)

                        for (prob, prob_name) in nls_prob_collection
                            if selected_h == "l0"
                                h = NormL0(λ)
                                h_name = "l0-norm"
                            elseif selected_h == "l1"
                                h = NormL1(λ)
                                h_name = "l1-norm"
                            elseif selected_h == "lhalf"
                                h = RootNormLhalf(λ)
                                h_name = "lhalf-norm"
                            elseif selected_h == "smooth"
                                h = NormL1(0.0)
                            end

                            SLM_outs = []
                            slm_trains = []
                            nslms = []
                            ngslms = []

                            for k in 1:n_exec
                                # executes n_exec times Sto_LM with the same inputs
                                x0 = digits[1] * ones(prob.meta.nvar)
                                #p = randperm(prob.meta.nvar)[1:nz]
                                #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
                                reset!(prob)
                                #try
                                PLM_out = Sto_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                                push!(SLM_outs, PLM_out)
                                push!(slm_trains, residual(prob, PLM_out.solution))
                                push!(nslms, length(prob.epoch_counter)-1)
                                push!(ngslms, (neval_jtprod_residual(prob) + neval_jprod_residual(prob)))

                                sample_size = length(prob.sample)

                                selected_iterations = (sample_rate == 1.0) ? prob.epoch_counter[2:end] : prob.epoch_counter[1:end-1]

                                # get objective value for each run #
                                @views Obj_Hists_epochs_sto[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:ExactFhist][selected_iterations]
                                @views Obj_Hists_epochs_sto[:, k][1:length(selected_iterations)] += PLM_out.solver_specific[:Hhist][selected_iterations]

                                # get MSE for each run #
                                @views MSE_Hists_epochs_sto[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:Fhist][selected_iterations]
                                @views MSE_Hists_epochs_sto[:, k][1:length(selected_iterations)] += PLM_out.solver_specific[:Hhist][selected_iterations]
                                @views MSE_Hists_epochs_sto[:, k][1:length(selected_iterations)] ./= (2 * sample_size)

                                # get metric for each run #
                                @views Metr_Hists_epochs_sto[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:ExactMetricHist][selected_iterations]

                                if k < n_exec # reset epoch counter for each run
                                    prob.epoch_counter = Int[1]
                                end
                            end

                            save_object("SLM_outs-$(sample_rate*100)%-$selected_prob-$selected_h.jld2", SLM_outs)
                            save_object("slm_trains-$(sample_rate*100)%-$selected_prob-$selected_h.jld2", slm_trains)

                            if n_exec%2 == 1
                                med_ind = (n_exec ÷ 2) + 1
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

                            Prob_LM_out = SLM_outs[origin_ind]
                            slmtrain = residual(prob, Prob_LM_out.solution)
                            if prob_name == "mnist-train-ls"
                                slmtest = residual(mnist_nls_test, Prob_LM_out.solution)
                                @show acc(slmtrain), acc(slmtest)
                            end
                            #nplm = neval_residual(sampled_nls_tr)
                            nslm = nslms[origin_ind]
                            save_object("nslm-mnist-PLM-$(100*sample_rate)%.jld2", nslm)
                            ngslm = ngslms[origin_ind]
                            save_object("ngslm-mnist-PLM-$(100*sample_rate)%.jld2", ngslm)
                            if prob_name == "mnist-train-ls"
                                slmdec = plot_svm(Prob_LM_out, Prob_LM_out.solution, "sto-lm-$(100*sample_rate)%-lhalf-$digits")
                            end

                            med_obj_sto = zeros(axes(Obj_Hists_epochs_sto, 1))
                            std_obj_sto = zeros(axes(Obj_Hists_epochs_sto, 1))

                            med_metr_sto = zeros(axes(Metr_Hists_epochs_sto, 1))
                            std_metr_sto = zeros(axes(Metr_Hists_epochs_sto, 1))

                            med_mse_sto = zeros(axes(MSE_Hists_epochs_sto, 1))
                            std_mse_sto = zeros(axes(MSE_Hists_epochs_sto, 1))

                            # compute median of objective value #
                            for l in eachindex(med_obj_sto)
                                cleared_obj_sto = filter(!isnan, filter(!iszero, Obj_Hists_epochs_sto[l, :]))
                                if isempty(cleared_obj_sto)
                                    med_obj_sto[l] = NaN
                                    std_obj_sto[l] = NaN
                                elseif length(cleared_obj_sto) == 1
                                    med_obj_sto[l] = median(cleared_obj_sto)
                                    std_obj_sto[l] = 0.0
                                else
                                    med_obj_sto[l] = median(cleared_obj_sto)
                                    std_obj_sto[l] = std(cleared_obj_sto)
                                end
                            end
                            filter!(!isnan, med_obj_sto)
                            std_obj_sto *= Confidence[conf]
                            filter!(!isnan, std_obj_sto)
                            save_object("med_obj_sto-$(sample_rate*100)-$prob_name-$selected_h.jld2", med_obj_sto)
                            save_object("std_obj_sto-$(sample_rate*100)-$prob_name-$selected_h.jld2", std_obj_sto)

                            # compute median of metric #
                            for l in eachindex(med_metr_sto)
                                cleared_metr_sto = filter(!isnan, filter(!iszero, Metr_Hists_epochs_sto[l, :]))
                                if isempty(cleared_metr_sto)
                                    med_metr_sto[l] = NaN
                                    std_metr_sto[l] = NaN
                                elseif length(cleared_metr_sto) == 1
                                    med_metr_sto[l] = median(cleared_metr_sto)
                                    std_metr_sto[l] = 0.0
                                else
                                    med_metr_sto[l] = median(cleared_metr_sto)
                                    std_metr_sto[l] = std(cleared_metr_sto)
                                end
                            end
                            filter!(!isnan, med_metr_sto)
                            std_metr_sto *= Confidence[conf]
                            filter!(!isnan, std_metr_sto)
                            save_object("med_metr_sto-$(sample_rate*100)-$selected_prob-$selected_h.jld2", med_metr_sto)
                            save_object("std_metr_sto-$(sample_rate*100)-$selected_prob-$selected_h.jld2", std_metr_sto)
                            
                            # compute median of MSE #
                            for l in eachindex(med_mse_sto)
                                cleared_mse_sto = filter(!isnan, filter(!iszero, MSE_Hists_epochs_sto[l, :]))
                                if isempty(cleared_mse_sto)
                                    med_mse_sto[l] = NaN
                                    std_mse_sto[l] = NaN
                                elseif length(cleared_mse_sto) == 1
                                    med_mse_sto[l] = median(cleared_mse_sto)
                                    std_mse_sto[l] = 0.0
                                else
                                    med_mse_sto[l] = (median(cleared_mse_sto))
                                    std_mse_sto[l] = std(cleared_mse_sto)
                                end
                            end
                            filter!(!isnan, med_mse_sto)
                            std_mse_sto *= Confidence[conf]
                            filter!(!isnan, std_mse_sto)
                            save_object("med_mse_sto-$(sample_rate*100)-$selected_prob-$selected_h.jld2", med_mse_sto)
                            save_object("std_mse_sto-$(sample_rate*100)-$selected_prob-$selected_h.jld2", std_mse_sto)

                            # --------------- OBJECTIVE DATA -------------------- #

                            data_obj_slm = PlotlyJS.scatter(; x = 1:length(med_obj_sto), y = med_obj_sto, mode="lines", name = "SLM - $(sample_rate*100)%", line=attr(
                                color = color_scheme[sample_rate], width = 1
                                )
                            )

                            data_std_obj_slm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_sto), length(med_obj_sto):-1:1), y = vcat(med_obj_sto + std_obj_sto, reverse!(med_obj_sto - std_obj_sto)), mode="lines", name = "SLM - $(sample_rate*100)%", fill="tozerox",
                                fillcolor = color_scheme_std[sample_rate],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_obj, data_obj_slm, data_std_obj_slm)

                            # --------------- METRIC DATA -------------------- #

                            data_metr_slm = PlotlyJS.scatter(; x = 1:length(med_metr_sto), y = med_metr_sto, mode="lines", name = "SLM - $(sample_rate*100)%", line=attr(
                                color = color_scheme[sample_rate], width = 1
                                )
                            )
                            reverse = reverse!(med_metr_sto - std_metr_sto)
                            for l in eachindex(reverse)
                                if reverse[l] < 0
                                    reverse[l] = med_metr_sto[l]
                                end
                            end


                            data_std_metr_slm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_sto), length(med_metr_sto):-1:1), y = vcat(med_metr_sto + std_metr_sto, reverse), mode="lines", name = "SLM - $(sample_rate*100)%", fill="tozerox",
                                fillcolor = color_scheme_std[sample_rate],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_metr, data_metr_slm)#, data_std_metr_slm)
                            
                            # --------------- MSE DATA -------------------- #

                            data_mse_slm = PlotlyJS.scatter(; x = 1:length(med_mse_sto), y = med_mse_sto, mode="lines", name = "SLM - $(sample_rate*100)%", line=attr(
                                color = color_scheme[sample_rate], width = 1
                                )
                            )

                            data_std_mse_slm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_sto), length(med_mse_sto):-1:1), y = vcat(med_mse_sto + std_mse_sto, reverse!(med_mse_sto - std_mse_sto)), mode="lines", name = "SLM - $(sample_rate*100)%", fill="tozerox",
                                fillcolor = color_scheme_std[sample_rate],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_mse, data_mse_slm, data_std_mse_slm)

                            # reset epoch counter for next problem
                            prob.epoch_counter = Int[1]
                        end
                    end

                    ## ---------------------- DYNAMIC SAMPLE RATE STRATEGIES ------------------------------ ##

                    for version in versions
                        nz = 10 * compound
                        #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                        sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 1.0, β = 1e16, σmax = 1e16, σmin = 1e-5, μmin = 1e-5, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                        local subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))
                        local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate0)
                        #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
                        local ijcnn1, ijcnn1_nls, ijcnn1_sol = ijcnn1_model_sto(sample_rate0)
                        #a9a, a9a_nls = a9a_model_sto(sample_rate)
                        local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate0; digits = digits)
                        local mnist_nlp_test_sto, mnist_nls_test_sto, mnist_sto_sol_test = MNIST_test_model_sto(sample_rate0; digits = digits)
                        #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)
                        local λ = .1
                        if selected_prob == "ijcnn1"
                            nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
                        elseif selected_prob == "mnist"
                            nls_prob_collection = [(mnist_nls, "mnist-train-ls")]
                        end
                        
                        Obj_Hists_epochs_prob = zeros(1 + MaxEpochs, n_exec)
                        Metr_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)
                        MSE_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)

                        for (prob, prob_name) in nls_prob_collection
                            if selected_h == "l0"
                                h = NormL0(λ)
                                h_name = "l0-norm"
                            elseif selected_h == "l1"
                                h = NormL1(λ)
                                h_name = "l1-norm"
                            elseif selected_h == "lhalf"
                                h = RootNormLhalf(λ)
                                h_name = "lhalf-norm"
                            elseif selected_h == "smooth"
                                h = NormL1(0.0)
                            end

                            @info "using PLM to solve with" h
                            # routine to select the output with the median accuracy on the training set
                            PLM_outs = []
                            plm_trains = []
                            nplms = []
                            ngplms = []
                            epoch_counters_plm = []

                            for k in 1:n_exec
                                x0 = ones(prob.meta.nvar)
                                #p = randperm(prob.meta.nvar)[1:nz]
                                #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
                                reset!(prob)
                                prob.epoch_counter = Int[1]
                                PLM_out = Prob_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version)
                                push!(PLM_outs, PLM_out)
                                push!(plm_trains, residual(prob, PLM_out.solution))
                                push!(nplms, length(prob.epoch_counter))
                                push!(ngplms, (neval_jtprod_residual(prob) + neval_jprod_residual(prob)))
                                push!(epoch_counters_plm, prob.epoch_counter)
                                #PLM_out = SPLM(prob, sampled_options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version)

                                selected_iterations = (sample_rate0 == 1.0) ? prob.epoch_counter[2:end] : prob.epoch_counter[1:end-1]

                                # get objective value for each run #
                                @views Obj_Hists_epochs_prob[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:ExactFhist][selected_iterations]
                                @views Obj_Hists_epochs_prob[:, k][1:length(selected_iterations)] += PLM_out.solver_specific[:Hhist][selected_iterations]

                                # get MSE for each run #
                                @views MSE_Hists_epochs_prob[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:Fhist][selected_iterations]
                                @views MSE_Hists_epochs_prob[:, k][1:length(selected_iterations)] += PLM_out.solver_specific[:Hhist][selected_iterations]
                                @views MSE_Hists_epochs_prob[:, k][1:length(selected_iterations)] ./= ceil.(2 * prob.nls_meta.nequ * PLM_out.solver_specific[:SampleRateHist][selected_iterations])

                                # get metric for each run #
                                @views Metr_Hists_epochs_prob[:, k][1:length(selected_iterations)] = PLM_out.solver_specific[:ExactMetricHist][selected_iterations]
                            end
                            save_object("PLM_outs-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", PLM_outs)
                            save_object("plm_trains-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", plm_trains)
                            save_object("epoch_counters_plm-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", epoch_counters_plm)

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
                            plmtrain = residual(prob, Prob_LM_out.solution)
                            if prob_name == "mnist-train-ls"
                                plmtest = residual(mnist_nls_test_sto, Prob_LM_out.solution)
                                @show acc(plmtrain), acc(plmtest)
                            end
                            #nplm = neval_residual(sampled_nls_tr)
                            nplm = nplms[origin_ind]
                            save_object("nplm-mnist-PLM-$(prob_versions_names[version]).jld2", nplm)
                            ngplm = ngplms[origin_ind]
                            save_object("ngplm-mnist-PLM-$(prob_versions_names[version]).jld2", ngplm)
                            if prob_name == "mnist-train-ls"
                                plmdec = plot_svm(Prob_LM_out, Prob_LM_out.solution, "prob-lm-$(prob_versions_names[version])-lhalf-$digits")
                            end

                            med_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))
                            std_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))

                            med_metr_prob = zeros(axes(Metr_Hists_epochs_prob, 1))
                            std_metr_prob = zeros(axes(Metr_Hists_epochs_prob, 1))

                            med_mse_prob = zeros(axes(MSE_Hists_epochs_prob, 1))
                            std_mse_prob = zeros(axes(MSE_Hists_epochs_prob, 1))

                            # compute median of objective value #
                            for l in eachindex(med_obj_prob)
                                cleared_obj_prob = filter(!isnan, filter(!iszero, Obj_Hists_epochs_prob[l, :]))
                                if isempty(cleared_obj_prob)
                                    med_obj_prob[l] = NaN
                                    std_obj_prob[l] = NaN
                                elseif length(cleared_obj_prob) == 1
                                    med_obj_prob[l] = median(cleared_obj_prob)
                                    std_obj_prob[l] = 0.0
                                else
                                    med_obj_prob[l] = median(cleared_obj_prob)
                                    std_obj_prob[l] = std(cleared_obj_prob)
                                end
                            end
                            filter!(!isnan, med_obj_prob)
                            std_obj_prob *= Confidence[conf]
                            filter!(!isnan, std_obj_prob)

                            save_object("med_obj_prob-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", med_obj_prob)
                            save_object("std_obj_prob-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", std_obj_prob)

                            # compute median of metric #
                            for l in eachindex(med_metr_prob)
                                cleared_metr_prob = filter(!isnan, filter(!iszero, Metr_Hists_epochs_prob[l, :]))
                                if isempty(cleared_metr_prob)
                                    med_metr_prob[l] = NaN
                                    std_metr_prob[l] = NaN
                                elseif length(cleared_metr_prob) == 1
                                    med_metr_prob[l] = median(cleared_metr_prob)
                                    std_metr_prob[l] = 0.0
                                else
                                    med_metr_prob[l] = median(cleared_metr_prob)
                                    std_metr_prob[l] = std(cleared_metr_prob)
                                end
                            end
                            filter!(!isnan, med_metr_prob)
                            std_metr_prob *= Confidence[conf]
                            filter!(!isnan, std_metr_prob)

                            save_object("med_metr_prob-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", med_metr_prob)
                            save_object("std_metr_prob-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", std_metr_prob)
                            
                            # compute median of MSE #
                            for l in eachindex(med_mse_prob)
                                cleared_mse_prob = filter(!isnan, filter(!iszero, MSE_Hists_epochs_prob[l, :]))
                                if isempty(cleared_mse_prob)
                                    med_mse_prob[l] = NaN
                                    std_mse_prob[l] = NaN
                                elseif length(cleared_mse_prob) == 1
                                    med_mse_prob[l] = median(cleared_mse_prob)
                                    std_mse_prob[l] = 0.0
                                else
                                    med_mse_prob[l] = (median(cleared_mse_prob))
                                    std_mse_prob[l] = std(cleared_mse_prob)
                                end
                            end
                            filter!(!isnan, med_mse_prob)
                            std_mse_prob *= Confidence[conf]
                            filter!(!isnan, std_mse_prob)

                            save_object("med_mse_prob-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", med_mse_prob)
                            save_object("std_mse_prob-$(prob_versions_names[version])-$selected_prob-$selected_h.jld2", std_mse_prob)

                            # --------------- OBJECTIVE DATA -------------------- #

                            data_obj_plm = PlotlyJS.scatter(; x = 1:length(med_obj_prob), y = med_obj_prob, mode="lines", name = "PLM - $(prob_versions_names[version])", line=attr(
                                color = prob_versions_colors[version], width = 1
                                )
                            )

                            data_std_obj_plm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_prob), length(med_obj_prob):-1:1), y = vcat(med_obj_prob + std_obj_prob, reverse!(med_obj_prob - std_obj_prob)), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
                                fillcolor = prob_versions_colors_std[version],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_obj, data_obj_plm)#, data_std_obj_plm)

                            # --------------- METRIC DATA -------------------- #

                            data_metr_plm = PlotlyJS.scatter(; x = 1:length(med_metr_prob), y = med_metr_prob, mode="lines", name = "PLM - $(prob_versions_names[version])", line=attr(
                                color = prob_versions_colors[version], width = 1
                                )
                            )

                            reverse = reverse!(med_metr_prob - std_metr_prob)
                            for l in eachindex(reverse)
                                if reverse[l] < 0
                                    reverse[l] = med_metr_prob[l]
                                end
                            end

                            data_std_metr_plm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_prob), length(med_metr_prob):-1:1), y = vcat(med_metr_prob + std_metr_prob, reverse), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
                                fillcolor = prob_versions_colors_std[version],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_metr, data_metr_plm)#, data_std_metr_plm)
                            
                            # --------------- MSE DATA -------------------- #

                            data_mse_plm = PlotlyJS.scatter(; x = 1:length(med_mse_prob), y = med_mse_prob, mode="lines", name = "PLM - $(prob_versions_names[version])", line=attr(
                                color = prob_versions_colors[version], width = 1
                                )
                            )

                            data_std_mse_plm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_prob), length(med_mse_prob):-1:1), y = vcat(med_mse_prob + std_mse_prob, reverse!(med_mse_prob - std_mse_prob)), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
                                fillcolor = prob_versions_colors_std[version],
                                line_color = "transparent",
                                showlegend = false
                            )

                            push!(data_mse, data_mse_plm)#, data_std_mse_plm)

                        end


                        if compare
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
                            df[!, :Alg] = ["R2", "LM", "LMTR", "PLM-$(prob_versions_names[version])"]
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
                            open("svm-lhalf-$digits.tex", "w") do io
                                SolverBenchmark.pretty_latex_stats(io, df,
                                    col_formatters=fmt_override,
                                    hdr_override=hdr_override)
                            end
                        end
                    end
                end

                if selected_prob == "ijcnn1"
                    prob_name = "ijcnn1-ls"
                elseif selected_prob == "mnist"
                    prob_name = "mnist-train-ls"
                end

                layout_obj = Layout(title="$prob_name - $n_exec runs - h = $selected_h-norm",
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis =attr(
                                showexponent = "all",
                                exponentformat = "e"
                            ),
                        yaxis_type="log",
                        yaxis_title="Exact f+h",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.4)"
                        ),
                        font=attr(size=13))
                
                layout_metr = Layout(title="$prob_name - $n_exec runs - h = $selected_h-norm",
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis =attr(
                            showexponent = "all",
                            exponentformat = "e"
                        ),
                        yaxis_type="log",
                        yaxis_title="√ξcp/ν",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.4)"
                        ),
                        font=attr(size=13))

                layout_mse = Layout(title="$prob_name - $n_exec runs - h = $selected_h-norm",
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis =attr(
                            showexponent = "all",
                            exponentformat = "e"
                        ),
                        yaxis_type="log",
                        yaxis_title="MSE",
                        template="simple_white",
                        legend = attr(
                            xanchor="right",
                            bgcolor="rgba(255,255,255,.4)"
                        ),
                        font=attr(size=13))
                
                plt_obj = PlotlyJS.plot(data_obj, layout_obj)
                plt_metr = PlotlyJS.plot(data_metr, layout_metr)
                plt_mse = PlotlyJS.plot(data_mse, layout_mse)

                display(plt_obj)
                display(plt_metr)
                display(plt_mse)

                #=if selected_prob == "ijcnn1"
                    PlotlyJS.savefig(plt_obj, "$selected_prob-exactobj-$(n_exec)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
                    PlotlyJS.savefig(plt_metr, "$selected_prob-metric-$(n_exec)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
                    PlotlyJS.savefig(plt_mse, "$selected_prob-MSE-$(n_exec)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
                elseif selected_prob == "mnist"
                    PlotlyJS.savefig(plt_obj, "$selected_prob-exactobj-$(n_exec)runs-$digits-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth-version$(versions[end]).pdf"; format = "pdf")
                    PlotlyJS.savefig(plt_metr, "$selected_prob-metric-$(n_exec)runs-$digits-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth-version$(versions[end]).pdf"; format = "pdf")
                    PlotlyJS.savefig(plt_mse, "$selected_prob-MSE-$(n_exec)runs-$digits-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth-version$(versions[end]).pdf"; format = "pdf")
                end=#
            end
        end
    end
end
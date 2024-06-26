function plot_Sampled_LM_SVM_epoch(sample_rates::AbstractVector, versions::AbstractVector, selected_probs::AbstractVector, selected_hs::AbstractVector, selected_digits::AbstractVector; abscissa = "CPU time", n_exec = 10, smooth::Bool = false, sample_rate0::Float64 = .05, param::String = "MSE", compare::Bool = false, guide::Bool = false, MaxEpochs::Int = 1000, MaxTime = 3600.0, precision = 1e-4)
    compound = 1
    color_scheme = Dict([(1.0, "rgb(255,105,180)"), (.2, "rgb(176,196,222)"), (.1, "rgb(205,133,63)"), (.05, "rgb(154,205,50)"), (.01, 8)])
    color_scheme_std = Dict([(1.0, "rgba(255,105,180, .2)"), (.2, "rgba(176,196,222, 0.2)"), (.1, "rgba(205,133,63, 0.2)"), (.05, "rgba(154,205,50, 0.2)"), (.01, 8)])

    prob_versions_names = Dict([(1, "mobmean"), (2, "nondec"), (3, "each-it"), (4, "hybrid")])
    prob_versions_colors = Dict([(1, "rgb(30,144,255)"), (2, "rgb(255,140,0)"), (3, "rgb(50,205,50)"), (4, "rgb(123,104,238)")])
    prob_versions_colors_std = Dict([(1, "rgba(30,144,255, 0.2)"), (2, "rgba(255,140,0, 0.2)"), (3, "rgba(50,205,50, 0.2)"), (4, "rgba(123,104,238, 0.2)")])

    Confidence = Dict([("95%", 1.96), ("99%", 2.58)])
    conf = "95%"

    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    for selected_prob in selected_probs
        for selected_h in selected_hs
            for digits in selected_digits

                data_obj = GenericTrace{Dict{Symbol, Any}}[]
                data_metr = GenericTrace{Dict{Symbol, Any}}[]
                data_mse = GenericTrace{Dict{Symbol, Any}}[]

                #plots of other algorithms
                if compare
                    bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound)
                    mnist_full, mnist_nls_full, mnist_nls_sol = RegularizedProblems.svm_train_model(digits)
                    mnist_nlp_test, mnist_nls_test, mnist_sol_test = RegularizedProblems.svm_test_model(digits)
                    A_ijcnn1, b_ijcnn1 = ijcnn1_load_data()
                    ijcnn1_full, ijcnn1_nls_full = RegularizedProblems.svm_model(A_ijcnn1', b_ijcnn1)
                    sampled_options_full = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs+1, maxTime = MaxTime;)
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
                    end

                    @info "using R2 to solve with" h
                    reset!(prob)
                    xk_R2, k_R2, R2_out = RegularizedOptimization.R2(prob.f, prob.∇f!, h, sampled_options_full, x0)
                    prob = LSR1Model(prob)
                    R2_stats = RegularizedOptimization.R2(prob, h, sampled_options_full, x0 = x0)
                    nr2 = NLPModels.neval_obj(prob)
                    ngr2 = NLPModels.neval_grad(prob)
                    r2train = residual(prob_nls, R2_stats.solution) #||e - tanh(b * <A, x>)||^2, b ∈ {-1,1}^n
                    if selected_prob == "mnist"
                        r2test = residual(mnist_nls_test, R2_stats.solution)
                        @show acc(r2train), acc(r2test)
                    end
                    r2dec = plot_svm(R2_stats, R2_stats.solution, "r2-$version-lhalf-$digits")

                    @info " using LM to solve with" h
                    LM_out = LM(prob_nls, h, sampled_options_full; x0 = x0, subsolver_options = subsolver_options)
                    lmtrain = residual(prob_nls, LM_out.solution)
                    if selected_prob == "mnist"
                        lmtest = residual(mnist_nls_test, LM_out.solution)
                        @show acc(lmtrain), acc(lmtest)
                    end
                    lmdec = plot_svm(LM_out, LM_out.solution, "lm-$version-lhalf-$digits")
                    #nlm = NLPModels.neval_residual(nls_tr)
                    nlm = LM_out.iter
                    nglm = NLPModels.neval_jtprod_residual(prob_nls) + NLPModels.neval_jprod_residual(prob_nls)

                    if (h == NormL0(λ)) || (h == RootNormLhalf(λ))
                        @info " using LMTR to solve with" h NormLinf(1.0)
                        reset!(prob_nls)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormLinf(1.0), sampled_options_full; x0 = x0, subsolver_options = subsolver_options)
                        lmtrtrain = residual(prob_nls, LMTR_out.solution)
                        if selected_prob == "mnist"
                            lmtrtest = residual(mnist_nls_test, LMTR_out.solution)
                            @show acc(lmtrtrain), acc(lmtrtest)
                        end
                        #nlmtr = NLPModels.neval_residual(nls_tr)
                        nlmtr = LMTR_out.iter
                        nglmtr = NLPModels.neval_jtprod_residual(prob_nls) + NLPModels.neval_jprod_residual(prob_nls)
                        lmtrdec = plot_svm(LMTR_out, LMTR_out.solution, "lmtr-$version-lhalf-$digits")
                    elseif h == NormL1(λ)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormL2(1.0), sampled_options_full; x0 = x0, subsolver_options = subsolver_options)
                    elseif h == NormL1(0.0)
                        LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormL2(1.0), sampled_options_full; x0 = x0, subsolver_options = subsolver_options)
                    end

                    #=if param == "MSE"
                        plot!(1:k_R2, 0.5*(R2_out[:Fhist] + R2_out[:Hhist])/m, label = "R2", lc = :red, ls = :dashdot, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                        plot!(1:length(LM_out.solver_specific[:Fhist]), 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, label = "LM", lc = :orange, ls = :dot, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                        plot!(1:length(LMTR_out.solver_specific[:Fhist]), 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, label = "LMTR", lc = :black, ls=:dash, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                    #elseif param == "accuracy"
                    elseif param == "objective"
                        plot!(1:k_R2, R2_out[:Fhist] + R2_out[:Hhist], label = "R2", lc = :red, ls = :dashdot, yaxis = yscale)
                        plot!(1:length(LM_out.solver_specific[:Fhist]), LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], yaxis = yscale, label = "LM", lc = :orange, ls = :dot, legend=:outertopright)
                        plot!(1:length(LMTR_out.solver_specific[:Fhist]), LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], yaxis = yscale, label = "LMTR", lc = :black, ls=:dash, legend=:outertopright)
                    end=#

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

                ## -------------------------------- GREYMAPS AND TABLES -------------------------------------- ##



                ## ------------------------------------------------------------------------------------------- ##
                ## -------------------------------- SAMPLED VERSION ------------------------------------------ ##
                ## ------------------------------------------------------------------------------------------- ##

                for sample_rate in sample_rates
                    nz = 10 * compound
                    #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                    sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                    local subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))
                    local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate)
                    #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
                    local ijcnn1, ijcnn1_nls, ijcnn1_sol = ijcnn1_model_sto(sample_rate)
                    #a9a, a9a_nls = a9a_model_sto(sample_rate)
                    local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate; digits = digits)
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
                        end
                        for k in 1:n_exec
                            # executes n_exec times Sto_LM with the same inputs
                            x0 = digits[1] * ones(prob.meta.nvar)
                            #p = randperm(prob.meta.nvar)[1:nz]
                            #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
                            reset!(prob)
                            #try
                            PLM_out = Sto_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)

                            #=if param == "objective"
                                if abscissa == "epoch"
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactFhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]
                                elseif abscissa == "CPU time"
                                    push!(Obj_Hists_time_prob, PLM_out.solver_specific[:ExactFhist] + PLM_out.solver_specific[:Hhist])
                                end
                            elseif param == "MSE"
                                sample_size = length(prob.sample)
                                if abscissa == "epoch"
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:Fhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] ./= (2 * sample_size)
                                elseif abscissa == "CPU time"
                                    Obj_Hists_time_prob_vec = PLM_out.solver_specific[:Fhist] + PLM_out.solver_specific[:Hhist]
                                    Obj_Hists_time_prob_vec ./= (2 * sample_size)
                                    push!(Obj_Hists_time_prob, Obj_Hists_time_prob_vec)
                                end
                            elseif param == "accuracy"
                                if abscissa == "epoch"
                                    Obj_Hists_epochs[:, k] = acc.(residual.(prob, PLM_out.solver_specific[:Xhist][prob.epoch_counter]))
                                elseif abscissa == "CPU time"
                                    Obj_Hists_time_vec_prob = []
                                    for i in 1:length(PLM_out.solver_specific[:Xhist])
                                        push!(Obj_Hists_time_vec_prob, acc(residual(prob, PLM_out.solver_specific[:Xhist][i])))
                                    end
                                    push!(Obj_Hists_time_prob, Obj_Hists_time_vec_prob)
                                end
                            elseif param == "metric"
                                if abscissa == "epoch"
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactMetricHist][prob.epoch_counter]
                                end
                            end=#

                            sample_size = length(prob.sample)

                            # get objective value for each run #
                            @views Obj_Hists_epochs_sto[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactFhist][prob.epoch_counter]
                            @views Obj_Hists_epochs_sto[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]

                            # get MSE for each run #
                            @views MSE_Hists_epochs_sto[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:Fhist][prob.epoch_counter]
                            @views MSE_Hists_epochs_sto[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]
                            @views MSE_Hists_epochs_sto[:, k][1:length(prob.epoch_counter)] ./= (2 * sample_size)

                            # get metric for each run #
                            @views Metr_Hists_epochs_sto[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactMetricHist][prob.epoch_counter]

                            if k < n_exec # reset epoch counter for each run
                                prob.epoch_counter = Int[1]
                            end
                        end
                        med_obj_sto = zeros(axes(Obj_Hists_epochs_sto, 1))
                        std_obj_sto = zeros(axes(Obj_Hists_epochs_sto, 1))

                        med_metr_sto = zeros(axes(Metr_Hists_epochs_sto, 1))
                        std_metr_sto = zeros(axes(Metr_Hists_epochs_sto, 1))

                        med_mse_sto = zeros(axes(MSE_Hists_epochs_sto, 1))
                        std_mse_sto = zeros(axes(MSE_Hists_epochs_sto, 1))

                        # compute mean of objective value #
                        for l in eachindex(med_obj_sto)
                            med_obj_sto[l] = mean(filter(!iszero, Obj_Hists_epochs_sto[l, :]))
                            std_obj_sto[l] = std(filter(!iszero, Obj_Hists_epochs_sto[l, :]))
                        end
                        std_obj_sto *= Confidence[conf]
                        replace!(std_obj_sto, NaN=>0.0)

                        # compute mean of metric #
                        for l in eachindex(med_metr_sto)
                            med_metr_sto[l] = mean(filter(!iszero, Metr_Hists_epochs_sto[l, :]))
                            std_metr_sto[l] = std(filter(!iszero, Metr_Hists_epochs_sto[l, :]))
                        end
                        std_metr_sto *= Confidence[conf]
                        replace!(std_metr_sto, NaN=>0.0)                      
                        
                        # compute mean of MSE #
                        for l in eachindex(med_mse_sto)
                            med_mse_sto[l] = mean(filter(!iszero, MSE_Hists_epochs_sto[l, :]))
                            std_mse_sto[l] = std(filter(!iszero, MSE_Hists_epochs_sto[l, :]))
                        end
                        std_mse_sto *= Confidence[conf]
                        replace!(std_mse_sto, NaN=>0.0)

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

                        data_std_metr_slm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_sto), length(med_metr_sto):-1:1), y = vcat(med_metr_sto + std_metr_sto, reverse!(med_metr_sto - std_metr_sto)), mode="lines", name = "SLM - $(sample_rate*100)%", fill="tozerox",
                            fillcolor = color_scheme_std[sample_rate],
                            line_color = "transparent",
                            showlegend = false
                        )

                        push!(data_metr, data_metr_slm, data_std_metr_slm)
                        
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

                        println(med_mse_sto)

                        push!(data_mse, data_mse_slm, data_std_mse_slm)

                        #=    if (param == "MSE") || (param == "accuracy") || (param == "objective") || (param == "metric")
                                if prob_name == "mnist-train-ls"
                                    plot!(1:length(med_obj_prob), med_obj_prob, color = color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "SLM - $(sample_rate*100)%", title = "$prob_name - $n_exec runs - $digits - h = $h_name", ribbon=(std_obj_prob, std_obj_prob), xaxis = xscale, legend=:outertopright)
                                else
                                    plot!(1:length(med_obj_prob), med_obj_prob, color = color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "SLM - $(sample_rate*100)%", title = "$prob_name - $n_exec runs - h = $h_name", ribbon=(std_obj_prob, std_obj_prob), xaxis = xscale, legend=:outertopright)
                                end
                            #elseif param == "metric"
                                #plot!(axes(Metr_Hists_epochs, 1), med_metric, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs - $digits", ribbon=(std_metric, std_metric), xaxis = xscale, legend=:outertopright)
                            end=#
                        
                        # reset epoch counter for next problem
                        prob.epoch_counter = Int[1]
                    end
                end

                ## ---------------------- DYNAMIC SAMPLE RATE STRATEGIES ------------------------------ ##

                for version in versions
                    nz = 10 * compound
                    #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                    sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
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
                        end

                        @info " using Prob_LM to solve with" h
                        # routine to select the output with the median accuracy on the training set
                        PLM_outs = []
                        plm_trains = []

                        for k in 1:n_exec
                            # executes n_exec times Sto_LM with the same inputs
                            x0 = digits[1] * ones(prob.meta.nvar)
                            #p = randperm(prob.meta.nvar)[1:nz]
                            #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
                            reset!(prob)
                            prob.epoch_counter = Int[1]
                            PLM_out = Prob_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version)
                            push!(PLM_outs, PLM_out)
                            push!(plm_trains, residual(prob, PLM_out.solution))
                            #PLM_out = SPLM(prob, sampled_options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version)

                            #=if param == "objective"
                                if abscissa == "epoch"
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactFhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]
                                elseif abscissa == "CPU time"
                                    push!(Obj_Hists_time_prob, PLM_out.solver_specific[:ExactFhist] + PLM_out.solver_specific[:Hhist])
                                end
                            elseif param == "MSE"
                                if abscissa == "epoch"
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:Fhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] ./= ceil.(2 * prob.nls_meta.nequ * PLM_out.solver_specific[:SampleRateHist][prob.epoch_counter])
                                elseif abscissa == "CPU time"
                                    Obj_Hists_time_prob_vec = PLM_out.solver_specific[:Fhist] + PLM_out.solver_specific[:Hhist]
                                    Obj_Hists_time_prob_vec ./= ceil.(2 * prob.nls_meta.nequ * PLM_out.solver_specific[:SampleRateHist])
                                    push!(Obj_Hists_time_prob, Obj_Hists_time_prob_vec)
                                end
                            elseif param == "accuracy"
                                if abscissa == "epoch"
                                    Obj_Hists_epochs[:, k] = acc.(residual.(prob, PLM_out.solver_specific[:Xhist][prob.epoch_counter]))
                                elseif abscissa == "CPU time"
                                    Obj_Hists_time_vec_prob = []
                                    for i in 1:length(PLM_out.solver_specific[:Xhist])
                                        push!(Obj_Hists_time_vec_prob, acc(residual(prob, PLM_out.solver_specific[:Xhist][i])))
                                    end
                                    push!(Obj_Hists_time_prob, Obj_Hists_time_vec_prob)
                                end
                            elseif param == "metric"
                                @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactMetricHist][prob.epoch_counter]
                            end=#

                            # get objective value for each run #
                            @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactFhist][prob.epoch_counter]
                            @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]

                            # get MSE for each run #
                            @views MSE_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:Fhist][prob.epoch_counter]
                            @views MSE_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]
                            @views MSE_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] ./= ceil.(2 * prob.nls_meta.nequ * PLM_out.solver_specific[:SampleRateHist][prob.epoch_counter])

                            # get metric for each run #
                            @views Metr_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactMetricHist][prob.epoch_counter]
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
                        plmtrain = residual(prob, Prob_LM_out.solution)
                        if prob_name == "mnist-train-ls"
                            plmtest = residual(mnist_nls_test_sto, Prob_LM_out.solution)
                            @show acc(plmtrain), acc(plmtest)
                        end
                        #nplm = neval_residual(sampled_nls_tr)
                        nplm = length(prob.epoch_counter)
                        ngplm = (neval_jtprod_residual(prob) + neval_jprod_residual(prob))
                        plmdec = plot_svm(Prob_LM_out, Prob_LM_out.solution, "prob-lm-$version-lhalf-$digits")

                        med_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))
                        std_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))

                        med_metr_prob = zeros(axes(Metr_Hists_epochs_prob, 1))
                        std_metr_prob = zeros(axes(Metr_Hists_epochs_prob, 1))

                        med_mse_prob = zeros(axes(MSE_Hists_epochs_prob, 1))
                        std_mse_prob = zeros(axes(MSE_Hists_epochs_prob, 1))

                        # compute mean of objective value #
                        for l in eachindex(med_obj_prob)
                            med_obj_prob[l] = mean(filter(!iszero, Obj_Hists_epochs_prob[l, :]))
                            std_obj_prob[l] = std(filter(!iszero, Obj_Hists_epochs_prob[l, :]))
                        end
                        std_obj_prob *= Confidence[conf]
                        replace!(std_obj_prob, NaN=>0.0)

                        # compute mean of metric #
                        for l in eachindex(med_metr_prob)
                            med_metr_prob[l] = mean(filter(!iszero, Metr_Hists_epochs_prob[l, :]))
                            std_metr_prob[l] = std(filter(!iszero, Metr_Hists_epochs_prob[l, :]))
                        end
                        std_metr_prob *= Confidence[conf]
                        replace!(std_metr_prob, NaN=>0.0)
                        
                        # compute mean of MSE #
                        for l in eachindex(med_mse_prob)
                            med_mse_prob[l] = mean(filter(!iszero, MSE_Hists_epochs_prob[l, :]))
                            std_mse_prob[l] = std(filter(!iszero, MSE_Hists_epochs_prob[l, :]))
                        end
                        std_mse_prob *= Confidence[conf]
                        replace!(std_mse_prob, NaN=>0.0)

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

                        push!(data_obj, data_obj_plm, data_std_obj_plm)

                        # --------------- METRIC DATA -------------------- #

                        data_metr_plm = PlotlyJS.scatter(; x = 1:length(med_metr_prob), y = med_metr_prob, mode="lines", name = "PLM - $(prob_versions_names[version])", line=attr(
                            color = prob_versions_colors[version], width = 1
                            )
                        )

                        data_std_metr_plm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_prob), length(med_metr_prob):-1:1), y = vcat(med_metr_prob + std_metr_prob, reverse!(med_metr_prob - std_metr_prob)), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
                            fillcolor = prob_versions_colors_std[version],
                            line_color = "transparent",
                            showlegend = false
                        )

                        push!(data_metr, data_metr_plm, data_std_metr_plm)
                        
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

                        println(med_mse_prob)

                        push!(data_mse, data_mse_plm, data_std_mse_plm)
                        
                        #=med_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))
                        std_obj_prob = similar(med_obj_prob)
                        for l in 1:length(med_obj_prob)
                            med_obj_prob[l] = mean(filter(!iszero, Obj_Hists_epochs_prob[l, :]))
                            std_obj_prob[l] = std(filter(!iszero, Obj_Hists_epochs_prob[l, :]))
                        end
                        std_obj_prob *= Confidence[conf]
                        replace!(std_obj_prob, NaN=>0.0)
                        #std_metric *= Confidence[conf] / sqrt(sample_size)=#

                        #=if (param == "MSE") || (param == "accuracy") || (param == "objective") || (param == "metric")
                            if prob_name == "mnist-train-ls"
                                plot!(1:length(med_obj_prob), med_obj_prob, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "$prob_name - $n_exec runs - $digits - h = $h_name", ribbon=(std_obj_prob, std_obj_prob), xaxis = xscale, legend=:outertopright)
                            else
                                plot!(1:length(med_obj_prob), med_obj_prob, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "$prob_name - $n_exec runs - h = $h_name", ribbon=(std_obj_prob, std_obj_prob), xaxis = xscale, legend=:outertopright)
                            end
                        #elseif param == "metric"
                            #plot!(axes(Metr_Hists_epochs, 1), med_metric, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs - $digits", ribbon=(std_metric, std_metric), xaxis = xscale, legend=:outertopright)
                        end=#
                        prob.epoch_counter = Int[1]

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
                            df[!, :Alg] = !smooth ? ["R2", "LM", "LMTR", "PLM"] : ["R2", "LM", "LMTR", "PLM", "smooth PLM"]
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
                            open("svm-lhalf-$version-$digits.tex", "w") do io
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
                        yaxis_type="log",
                        yaxis_title="Exact f+h",
                        template="simple_white")
                
                layout_metr = Layout(title="$prob_name - $n_exec runs - h = $selected_h-norm",
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis_type="log",
                        yaxis_title="√ξcp/ν",
                        template="simple_white")

                layout_mse = Layout(title="$prob_name - $n_exec runs - h = $selected_h-norm",
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis_type="log",
                        yaxis_title="MSE",
                        template="simple_white")
                
                plt_obj = PlotlyJS.plot(data_obj, layout_obj)
                plt_metr = PlotlyJS.plot(data_metr, layout_metr)
                plt_mse = PlotlyJS.plot(data_mse, layout_mse)

                display(plt_obj)
                display(plt_metr)
                display(plt_mse)

                if selected_prob == "ijcnn1"
                    PlotlyJS.savefig(plt_obj, "$selected_prob-exactobj-$(n_exec)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.png"; format = "png")
                    PlotlyJS.savefig(plt_metr, "$selected_prob-metric-$(n_exec)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.png"; format = "png")
                    PlotlyJS.savefig(plt_mse, "$selected_prob-MSE-$(n_exec)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.png"; format = "png")
                elseif selected_prob == "mnist"
                    PlotlyJS.savefig(plt_obj, "$selected_prob-exactobj-$(n_exec)runs-$digits-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth-version$(versions[end]).png"; format = "png")
                    PlotlyJS.savefig(plt_metr, "$selected_prob-metric-$(n_exec)runs-$digits-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth-version$(versions[end]).png"; format = "png")
                    PlotlyJS.savefig(plt_mse, "$selected_prob-MSE-$(n_exec)runs-$digits-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth-version$(versions[end]).png"; format = "png")
                end
            end
        end
    end
end
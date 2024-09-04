function plot_mnist(sample_rates::AbstractVector, versions::AbstractVector, selected_hs::AbstractVector; abscissa = "CPU time", n_runs = 10, smooth::Bool = false, sample_rate0::Float64 = .05, compare::Bool = false, guide::Bool = false, MaxEpochs::Int = 1000)
    include("plot-configuration.jl")
    data_obj = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_metr = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_mse = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_neval_f = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_neval_jac = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]

    scatter_log = log_scale(MaxEpochs)

    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    local mnist, mnist_nls, mnist_nls_sol = RegularizedProblems.svm_train_model()

    ## ------------------------------------ R2, LM, LMTR ----------------------------------------- ##

    R2_stats, r2_metric_hist, r2_obj_hist, r2_numjac_hist = load_mnist_r2(selected_hs[1])
    #LM_out, LMTR_out = load_mnist_lm_lmtr(selected_hs[1])
    m = mnist_nls.nls_meta.nequ

    for sample_rate in sample_rates
        med_obj_sto, med_metr_sto, med_mse_sto, std_obj_sto, std_metr_sto, std_mse_sto, SLM_outs, slm_trains, nslm, ngslm = load_mnist_sto(sample_rate, "lhalf")
        legend_mse_slm = PGFPlots.Plots.Linear(1:2, [1e16 for i in 1:2], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])", legendentry = "PLM (Cst\\_batch=$(Int(sample_rate*100))\\%)")
        push!(data_mse, legend_mse_slm)
    end

    for version in versions
        med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob = load_mnist_plm(version, "lhalf")
        legend_mse_plm = PGFPlots.Plots.Linear(1:2, [1e16 for i in 1:2], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])", legendentry = "PLM ($(prob_versions_names[version]))")
        push!(data_mse, legend_mse_plm)
    end
    
    # --------------- MSE DATA -------------------- #
    data_mse_r2 = PGFPlots.Plots.Linear(1:length(r2_obj_hist), 0.5*(r2_obj_hist)/m, mark="none", style="cyan, solid", legendentry = "R2")
    #data_mse_lm = PGFPlots.Plots.Linear(1:length(LM_out.solver_specific[:Fhist]), 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, mark="none", style="black, dotted", legendentry = "LM")
    #data_mse_lmtr = PGFPlots.Plots.Linear(1:length(LMTR_out.solver_specific[:Fhist]), 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, mark="none", style="black, solid", legendentry = "LMTR")

    push!(data_mse, data_mse_r2)#, data_mse_lm, data_mse_lmtr)

    # --------------- OBJECTIVE DATA -------------------- #
    data_obj_r2 = PGFPlots.Plots.Linear(1:length(r2_obj_hist), r2_obj_hist, mark="none", style="cyan, solid")
    #data_obj_lm = PGFPlots.Plots.Linear(1:length(LM_out.solver_specific[:Fhist]), LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], mark="none", style="black, dotted")
    #data_obj_lmtr = PGFPlots.Plots.Linear(1:length(LMTR_out.solver_specific[:Fhist]), LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], mark="none", style="black, solid")

    push!(data_obj, data_obj_r2)#, data_obj_lm, data_obj_lmtr)

    # --------------- METRIC DATA -------------------- #
    data_metr_r2 = PGFPlots.Plots.Linear(1:length(r2_metric_hist), r2_metric_hist, mark="none", style="cyan, solid")

    push!(data_metr, data_metr_r2)

    # --------------- NEVAL_F DATA -------------------- #
    data_neval_f_r2 = PGFPlots.Plots.Linear(1:length(r2_metric_hist), 1:length(r2_metric_hist), mark="none", style="cyan, solid")
    push!(data_neval_f, data_neval_f_r2)

    # --------------- NEVAL_JAC_DATA -------------------- #
    data_neval_jac_r2 = PGFPlots.Plots.Linear(1:length(r2_numjac_hist), r2_numjac_hist, mark="none", style="cyan, solid")
    push!(data_neval_jac, data_neval_jac_r2)


    ## ------------------------------------------------------------------------------------------- ##
    ## -------------------------------- CONSTANT SAMPLE RATE ------------------------------------- ##
    ## ------------------------------------------------------------------------------------------- ##


    for selected_h in selected_hs
        for sample_rate in sample_rates
            med_obj_sto, med_metr_sto, med_mse_sto, std_obj_sto, std_metr_sto, std_mse_sto, SLM_outs, slm_trains, nslm, ngslm = load_mnist_sto(sample_rate, selected_h)

            # --------------- OBJECTIVE DATA -------------------- #
            markers_obj = vcat(filter(!>=(length(med_obj_sto)), scatter_log), length(med_obj_sto))
            data_obj_slm = PGFPlots.Plots.Linear(1:length(med_obj_sto), med_obj_sto, mark="none", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")
            markers_obj_slm = PGFPlots.Plots.Scatter(markers_obj, med_obj_sto[markers_obj], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate])", onlyMarks = true, markSize = 1.5)

            push!(data_obj, data_obj_slm, markers_obj_slm)#, data_std_obj_slm)

            # --------------- METRIC DATA -------------------- #
            markers_metr = vcat(filter(!>=(length(med_metr_sto)), scatter_log), length(med_metr_sto))
            data_metr_slm = PGFPlots.Plots.Linear(1:length(med_metr_sto), med_metr_sto, mark="none", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")
            markers_metr_slm = PGFPlots.Plots.Scatter(markers_metr, med_metr_sto[markers_metr], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate])", onlyMarks = true, markSize = 1.5)

            push!(data_metr, data_metr_slm, markers_metr_slm)#, data_std_metr_slm)
            
            # --------------- MSE DATA -------------------- #
            markers_mse = vcat(filter(!>=(length(med_mse_sto)), scatter_log), length(med_mse_sto))
            data_mse_slm = PGFPlots.Plots.Linear(1:length(med_mse_sto), med_mse_sto, mark="none", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")
            markers_mse_slm = PGFPlots.Plots.Scatter(markers_mse, med_mse_sto[markers_mse], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate])", onlyMarks = true, markSize = 1.5)

            push!(data_mse, data_mse_slm, markers_mse_slm)#, data_std_mse_slm)
            
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
            ecs = epoch_counter_slm(length(SLM_out.solver_specific[:NLSGradHist]), sample_rate)

            # --------------- NEVAL_F DATA -------------------- #
            markers_neval_f = vcat(filter(!>=(length(SLM_out.solver_specific[:ResidHist][ecs])), scatter_log), length(SLM_out.solver_specific[:ResidHist][ecs]))
            data_neval_f_slm = PGFPlots.Plots.Linear(1:length(SLM_out.solver_specific[:ResidHist][ecs]), SLM_out.solver_specific[:ResidHist][ecs], mark="none", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")
            markers_neval_f_slm = PGFPlots.Plots.Scatter(markers_neval_f, SLM_out.solver_specific[:ResidHist][ecs][markers_neval_f], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate])", onlyMarks = true, markSize = 1.5)
            push!(data_neval_f, data_neval_f_slm, markers_neval_f_slm)

            # --------------- NEVAL_JAC_DATA -------------------- #
            markers_neval_jac = vcat(filter(!>=(length(SLM_out.solver_specific[:NLSGradHist][ecs])), scatter_log), length(SLM_out.solver_specific[:NLSGradHist][ecs]))
            data_neval_jac_slm = PGFPlots.Plots.Linear(1:length(SLM_out.solver_specific[:NLSGradHist][ecs]), SLM_out.solver_specific[:NLSGradHist][ecs], mark="none", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")
            markers_neval_jac_slm = PGFPlots.Plots.Scatter(markers_neval_jac, SLM_out.solver_specific[:NLSGradHist][ecs][markers_neval_jac], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate])", onlyMarks = true, markSize = 1.5)
            push!(data_neval_jac, data_neval_jac_slm, markers_neval_jac_slm)
        end

        ## -------------------------------- DYNAMIC SAMPLE RATE ------------------------------------- ##

        for version in versions
            if selected_h != "smooth"
                med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob, PLM_outs, plm_trains, nplm, ngplm, epoch_counters_plm = load_mnist_plm(version, selected_h)
            else
                med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob, PLM_outs, plm_trains, nplm, ngplm = load_mnist_splm(version, selected_h)
            end
                # --------------- OBJECTIVE DATA -------------------- #
            markers_obj = vcat(filter(!>=(length(med_obj_prob)), scatter_log), length(med_obj_prob))
            data_obj_plm = PGFPlots.Plots.Linear(1:length(med_obj_prob), med_obj_prob, mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
            markers_obj_plm = PGFPlots.Plots.Scatter(markers_obj, med_obj_prob[markers_obj], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)

            push!(data_obj, data_obj_plm, markers_obj_plm)#, data_std_obj_plm)

            # --------------- METRIC DATA -------------------- #
            markers_metr = vcat(filter(!>=(length(med_metr_prob)), scatter_log), length(med_metr_prob))
            data_metr_plm = PGFPlots.Plots.Linear(1:length(med_metr_prob), med_metr_prob, mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
            markers_metr_plm = PGFPlots.Plots.Scatter(markers_metr, med_metr_prob[markers_metr], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)

            push!(data_metr, data_metr_plm, markers_metr_plm)#, data_std_metr_plm)
            
            # --------------- MSE DATA -------------------- #
            markers_metr = vcat(filter(!>=(length(med_mse_prob)), scatter_log), length(med_mse_prob))
            data_mse_plm = PGFPlots.Plots.Linear(1:length(med_mse_prob), med_mse_prob, mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
            markers_mse_plm = PGFPlots.Plots.Scatter(markers_metr, med_mse_prob[markers_metr], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)

            push!(data_mse, data_mse_plm, markers_mse_plm)#, data_std_mse_plm)

            #=if n_exec%2 == 1
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

            PLM_out = PLM_outs[origin_ind]
            ecp = epoch_counters_plm[origin_ind]
            adjusted_neval_f = similar(PLM_out.solver_specific[:ResidHist])
            adjusted_neval_f .= PLM_out.solver_specific[:ResidHist] .- collect(1:length(PLM_out.solver_specific[:ResidHist]))
            adjusted_neval_jac = similar(PLM_out.solver_specific[:NLSGradHist])
            adjusted_neval_jac .= PLM_out.solver_specific[:NLSGradHist] .- (2 .* collect(1:length(PLM_out.solver_specific[:NLSGradHist])))

            # --------------- NEVAL_F DATA -------------------- #
            markers_neval_f = vcat(filter(!>=(length(adjusted_neval_f[ecp])), scatter_log), length(adjusted_neval_f[ecp]))
            data_neval_f_slm = PGFPlots.Plots.Linear(1:length(adjusted_neval_f[ecp]), adjusted_neval_f[ecp], mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
            markers_neval_f_plm = PGFPlots.Plots.Scatter(markers_neval_f, adjusted_neval_f[ecp][markers_neval_f], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)
            push!(data_neval_f, data_neval_f_slm, markers_neval_f_plm)

            # --------------- NEVAL_JAC_DATA -------------------- #
            markers_neval_jac = vcat(filter(!>=(length(adjusted_neval_jac[ecp])), scatter_log), length(adjusted_neval_jac[ecp]))
            data_neval_jac_slm = PGFPlots.Plots.Linear(1:length(adjusted_neval_jac[ecp]), adjusted_neval_jac[ecp], mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
            markers_neval_jac_plm = PGFPlots.Plots.Scatter(markers_neval_jac, adjusted_neval_jac[ecp][markers_neval_jac], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)
            push!(data_neval_jac, data_neval_jac_slm, markers_neval_jac_plm)=#
        end

    plt_obj = PGFPlots.Axis(
        data_obj,
        xlabel="\$ j^{th}\$   epoch",
        ylabel="\$ (f+h)(x_j) \$",
        ymode="log",
        xmode="log",
    )
    plt_metr = PGFPlots.Axis(
        data_metr,
        xlabel="\$ j^{th}\$   epoch",
        ylabel="Stationarity measure",
        ymode="log",
        xmode="log",
    )
    plt_mse = PGFPlots.Axis(
        data_mse,
        xlabel="\$ j^{th}\$   epoch",
        ylabel="MSE",
        ymode="log",
        xmode="log",
        ymax = 0.6,
        legendPos = "south west"
    )

    plt_neval_f = PGFPlots.Axis(
        data_neval_f,
        xlabel="\$ j^{th}\$   epoch",
        ylabel="\$ j^{th} f \$ Call",
        ymode="log",
        xmode="log",
    )

    plt_neval_jac = PGFPlots.Axis(
        data_neval_jac,
        xlabel="\$ j^{th}\$   epoch",
        ylabel="\$ j^{th} \nabla f \$ Call",
        ymode="log",
        xmode="log",
    )

    #display(plt_obj)
    #display(plt_metr)
    #display(plt_mse)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\plots")
    PGFPlots.save("mnist-exactobj-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.tikz", plt_obj)
    PGFPlots.save("mnist-metric-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.tikz", plt_metr)
    PGFPlots.save("mnist-MSE-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.tikz", plt_mse)
    PGFPlots.save("mnist-neval_f-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.tikz", plt_neval_f)
    PGFPlots.save("mnist-neval_jac-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.tikz", plt_neval_jac)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
    end
end
function plot_mnist(sample_rates::AbstractVector, versions::AbstractVector, selected_hs::AbstractVector; abscissa = "CPU time", n_runs = 10, smooth::Bool = false, sample_rate0::Float64 = .05, compare::Bool = false, guide::Bool = false, MaxEpochs::Int = 1000)
    include("plot-configuration.jl")
    data_obj = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_metr = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_mse = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]

    scatter_log = log_scale(MaxEpochs)

    local mnist, mnist_nls, mnist_nls_sol = RegularizedProblems.svm_train_model()

    ## ------------------------------------ R2, LM, LMTR ----------------------------------------- ##

    k_R2, R2_out, R2_stats, r2_metric_hist = load_mnist_r2()
    LM_out, LMTR_out = load_mnist_lm_lmtr()
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
    data_mse_r2 = PGFPlots.Plots.Linear(1:k_R2, 0.5*(R2_out[:Fhist] + R2_out[:Hhist])/m, mark="none", style="cyan, dashed", legendentry = "R2")
    data_mse_lm = PGFPlots.Plots.Linear(1:length(LM_out.solver_specific[:Fhist]), 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, mark="none", style="black, dotted", legendentry = "LM")
    data_mse_lmtr = PGFPlots.Plots.Linear(1:length(LMTR_out.solver_specific[:Fhist]), 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, mark="none", style="black, dashed", legendentry = "LMTR")

    push!(data_mse, data_mse_r2)#, data_mse_lm, data_mse_lmtr)

    # --------------- OBJECTIVE DATA -------------------- #
    data_obj_r2 = PGFPlots.Plots.Linear(1:k_R2, R2_out[:Fhist] + R2_out[:Hhist], mark="none", style="cyan, dashed")
    data_obj_lm = PGFPlots.Plots.Linear(1:length(LM_out.solver_specific[:Fhist]), LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], mark="none", style="black, dotted")
    data_obj_lmtr = PGFPlots.Plots.Linear(1:length(LMTR_out.solver_specific[:Fhist]), LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], mark="none", style="black, dashed")

    push!(data_obj, data_obj_r2)#, data_obj_lm, data_obj_lmtr)

    # --------------- METRIC DATA -------------------- #
    display(r2_metric_hist)
    println(k_R2)
    data_metr_r2 = PGFPlots.Plots.Linear(1:k_R2, r2_metric_hist, mark="none", style="cyan, dashed")

    push!(data_metr, data_metr_r2)

    ## -------------------------------- CONSTANT SAMPLE RATE ------------------------------------- ##

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
        end

        ## -------------------------------- DYNAMIC SAMPLE RATE ------------------------------------- ##

        for version in versions
            med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob = load_mnist_plm(version, selected_h)
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

            #=if smooth
                med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob = load_mnist_splm(version, selected_h)

                # --------------- OBJECTIVE DATA -------------------- #

                data_obj_splm = PlotlyJS.scatter(; x = 1:length(med_obj_prob), y = med_obj_prob, mode="lines+markers", name = "$(prob_versions_names[version])-S", 
                line=attr(
                    color = smooth_versions_colors[version], 
                    width = 1,
                    dash = line_style_plm[version]
                    ),
                marker = attr(
                        color = smooth_versions_colors[version],
                        symbol = "circle",
                        size = 8
                    ),
                    showlegend = false
                )

                #=data_std_obj_splm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_prob), length(med_obj_prob):-1:1), y = vcat(med_obj_prob + std_obj_prob, reverse!(med_obj_prob - std_obj_prob)), mode="lines+markers", name = "$(prob_versions_names[version])-S", fill="tozerox",
                    fillcolor = smooth_versions_colors_std[version],
                    line_color = "transparent",
                    showlegend = false
                )=#

                push!(data_obj, data_obj_splm)#, data_std_obj_splm)

                # --------------- METRIC DATA -------------------- #

                data_metr_splm = PlotlyJS.scatter(; x = 1:length(med_metr_prob), y = med_metr_prob, mode="lines+markers", name = "$(prob_versions_names[version])-S", 
                line=attr(
                    color = smooth_versions_colors[version], 
                    width = 1,
                    dash = line_style_plm[version]
                    ),
                marker = attr(
                        color = smooth_versions_colors[version],
                        symbol = "circle",
                        size = 8
                    ),
                    showlegend = false
                )

                reverse = reverse!(med_metr_prob - std_metr_prob)
                for l in eachindex(reverse)
                    if reverse[l] < 0
                        reverse[l] = med_metr_prob[l]
                    end
                end

                #=data_std_metr_splm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_prob), length(med_metr_prob):-1:1), y = vcat(med_metr_prob + std_metr_prob, reverse), mode="lines+markers", name = "$(prob_versions_names[version])-S", fill="tozerox",
                    fillcolor = smooth_versions_colors_std[version],
                    line_color = "transparent",
                    showlegend = false
                )=#

                push!(data_metr, data_metr_splm)#, data_std_metr_splm)
                
                # --------------- MSE DATA -------------------- #

                data_mse_splm = PlotlyJS.scatter(; x = 1:length(med_mse_prob), y = med_mse_prob, mode="lines+markers", name = "$(prob_versions_names[version])-S", 
                line=attr(
                    color = smooth_versions_colors[version], 
                    width = 1,
                    dash = line_style_plm[version]
                    ),
                marker = attr(
                        color = smooth_versions_colors[version],
                        symbol = "circle",
                        size = 8
                    ),
                    showlegend = true
                )

                #=data_std_mse_splm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_prob), length(med_mse_prob):-1:1), y = vcat(med_mse_prob + std_mse_prob, reverse!(med_mse_prob - std_mse_prob)), mode="lines+markers", name = "$(prob_versions_names[version])-S", fill="tozerox",
                    fillcolor = smooth_versions_colors_std[version],
                    line_color = "transparent",
                    showlegend = false
                )=#

                push!(data_mse, data_mse_splm)#, data_std_mse_splm)
            end=#
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
        ymax = 0.6
    )

    #display(plt_obj)
    #display(plt_metr)
    #display(plt_mse)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\plots")
    PGFPlots.save("mnist-exactobj-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.tikz", plt_obj)
    PGFPlots.save("mnist-metric-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.tikz", plt_metr)
    PGFPlots.save("mnist-MSE-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.tikz", plt_mse)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
    end
end
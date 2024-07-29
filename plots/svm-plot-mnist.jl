function plot_mnist(sample_rates::AbstractVector, versions::AbstractVector, selected_hs::AbstractVector; abscissa = "CPU time", n_runs = 10, smooth::Bool = false, sample_rate0::Float64 = .05, compare::Bool = false, guide::Bool = false)
    include("plot-configuration.jl")
    data_obj = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_metr = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_mse = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]

    local mnist, mnist_nls, mnist_nls_sol = RegularizedProblems.svm_train_model()

    ## ------------------------------------ R2, LM, LMTR ----------------------------------------- ##

    k_R2, R2_out, R2_stats = load_mnist_r2()
    LM_out, LMTR_out = load_mnist_lm_lmtr()
    m = mnist_nls.nls_meta.nequ

    # --------------- OBJECTIVE DATA -------------------- #
    data_obj_r2 = PGFPlots.Plots.Linear(1:k_R2, R2_out[:Fhist] + R2_out[:Hhist], mark="none", style="cyan, dashed")
    data_obj_lm = PGFPlots.Plots.Linear(1:length(LM_out.solver_specific[:Fhist]), LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], mark="none", style="black, dotted")
    data_obj_lmtr = PGFPlots.Plots.Linear(1:length(LMTR_out.solver_specific[:Fhist]), LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], mark="none", style="black, dashed")

    push!(data_obj, data_obj_r2)#, data_obj_lm, data_obj_lmtr)
    
    # --------------- MSE DATA -------------------- #
    data_mse_r2 = PGFPlots.Plots.Linear(1:k_R2, 0.5*(R2_out[:Fhist] + R2_out[:Hhist])/m, mark="none", style="cyan, dashed", legendentry = "R2")
    data_mse_lm = PGFPlots.Plots.Linear(1:length(LM_out.solver_specific[:Fhist]), 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, mark="none", style="black, dotted", legendentry = "LM")
    data_mse_lmtr = PGFPlots.Plots.Linear(1:length(LMTR_out.solver_specific[:Fhist]), 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, mark="none", style="black, dashed", legendentry = "LMTR")

    push!(data_mse, data_mse_r2)#, data_mse_lm, data_mse_lmtr)

    ## -------------------------------- CONSTANT SAMPLE RATE ------------------------------------- ##

    for selected_h in selected_hs
        for sample_rate in sample_rates
            med_obj_sto, med_metr_sto, med_mse_sto, std_obj_sto, std_metr_sto, std_mse_sto, SLM_outs, slm_trains, nslm, ngslm = load_mnist_sto(sample_rate, selected_h)

            # --------------- OBJECTIVE DATA -------------------- #
            data_obj_slm = PGFPlots.Plots.Linear(1:length(med_obj_sto), med_obj_sto, mark=(sample_rate == 1.0 ? "none" : "square"), style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")

            push!(data_obj, data_obj_slm)#, data_std_obj_slm)

            # --------------- METRIC DATA -------------------- #
            data_metr_slm = PGFPlots.Plots.Linear(1:length(med_metr_sto), med_metr_sto, mark=(sample_rate == 1.0 ? "none" : "square"), style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")

            push!(data_metr, data_metr_slm)#, data_std_metr_slm)
            
            # --------------- MSE DATA -------------------- #
            data_mse_slm = PGFPlots.Plots.Linear(1:length(med_mse_sto), med_mse_sto, mark=(sample_rate == 1.0 ? "none" : "square"), style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])", legendentry = "PLM (Cst-$(sample_rate*100)"*L"\%)")

            push!(data_mse, data_mse_slm)#, data_std_mse_slm)
        end

        ## -------------------------------- DYNAMIC SAMPLE RATE ------------------------------------- ##

        for version in versions
            med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob = load_mnist_plm(version, selected_h)
            # --------------- OBJECTIVE DATA -------------------- #
            data_obj_plm = PGFPlots.Plots.Linear(1:length(med_obj_prob), med_obj_prob, mark="triangle", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")

            push!(data_obj, data_obj_plm)#, data_std_obj_plm)

            # --------------- METRIC DATA -------------------- #
            data_metr_plm = PGFPlots.Plots.Linear(1:length(med_metr_prob), med_metr_prob, mark="triangle", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")

            push!(data_metr, data_metr_plm)#, data_std_metr_plm)
            
            # --------------- MSE DATA -------------------- #
            data_mse_plm = PGFPlots.Plots.Linear(1:length(med_mse_prob), med_mse_prob, mark="triangle", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])", legendentry = "PLM ($(prob_versions_names[version]))")

            push!(data_mse, data_mse_plm)#, data_std_mse_plm)

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
        xlabel="\$ j^{th}\$   Epoch",
        ylabel="\$ (f+h)(x_j) \$",
        ymode="log",
        xmode="log",
    )
    plt_metr = PGFPlots.Axis(
        data_metr,
        xlabel="\$ j^{th}\$   Epoch",
        ylabel=L"\left(\xi_{j,cp}^*(x_j,\nu_j^{-1})\nu_j^{-1} \right)^{\frac{1}{2}}",
        ymode="log",
        xmode="log",
    )
    plt_mse = PGFPlots.Axis(
        data_mse,
        xlabel="\$ j^{th}\$   Epoch",
        ylabel="\$ MSE \$",
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
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
    end
end
function plot_mnist(sample_rates::AbstractVector, versions::AbstractVector, selected_hs::AbstractVector; abscissa = "CPU time", n_runs = 10, smooth::Bool = false, sample_rate0::Float64 = .05, compare::Bool = false, guide::Bool = false)
    include("plot-configuration.jl")
    data_obj = GenericTrace{Dict{Symbol, Any}}[]
    data_metr = GenericTrace{Dict{Symbol, Any}}[]
    data_mse = GenericTrace{Dict{Symbol, Any}}[]

    local mnist, mnist_nls, mnist_nls_sol = RegularizedProblems.svm_train_model()

    ## ------------------------------------ R2, LM, LMTR ----------------------------------------- ##

    k_R2, R2_out, R2_stats = load_mnist_r2()
    LM_out, LMTR_out = load_mnist_lm_lmtr()
    m = mnist_nls.nls_meta.nequ

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
        ),
        showlegend = false
    )

    data_mse_lm = PlotlyJS.scatter(; x = 1:length(LM_out.solver_specific[:Fhist]) , y = 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, mode="lines", name = "LM", line=attr(
        color="rgb(255,165,0)", dash = "dot", width = 1
        ),
        showlegend = false
    )
    
    data_mse_lmtr = PlotlyJS.scatter(; x = 1:length(LMTR_out.solver_specific[:Fhist]) , y = 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, mode="lines", name = "LMTR", line=attr(
            color="black", dash = "dash", width = 1
            ),
        showlegend = false
    )

    push!(data_mse, data_mse_r2, data_mse_lm, data_mse_lmtr)

    ## -------------------------------- CONSTANT SAMPLE RATE ------------------------------------- ##

    for selected_h in selected_hs
        for sample_rate in sample_rates
            med_obj_sto, med_metr_sto, med_mse_sto, std_obj_sto, std_metr_sto, std_mse_sto = load_mnist_sto(sample_rate, selected_h)

            # --------------- OBJECTIVE DATA -------------------- #

            data_obj_slm = PlotlyJS.scatter(; x = 1:length(med_obj_sto), y = med_obj_sto, mode="lines+markers", name = "$(sample_rate*100)%-N", 
            line=attr(
                color = color_scheme[sample_rate], 
                width = 1,
                dash = line_style_sto[sample_rate]
                ),
            marker = attr(
                    color = color_scheme[sample_rate],
                    symbol = "square",
                    size = 4
                )
            )

            data_std_obj_slm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_sto), length(med_obj_sto):-1:1), y = vcat(med_obj_sto + std_obj_sto, reverse!(med_obj_sto - std_obj_sto)), mode="lines+markers", name = "$(sample_rate*100)%-N", fill="tozerox",
                fillcolor = color_scheme_std[sample_rate],
                line_color = "transparent",
                showlegend = false
            )

            push!(data_obj, data_obj_slm, data_std_obj_slm)

            # --------------- METRIC DATA -------------------- #

            data_metr_slm = PlotlyJS.scatter(; x = 1:length(med_metr_sto), y = med_metr_sto, mode="lines+markers", name = "$(sample_rate*100)%-N", 
            line=attr(
                color = color_scheme[sample_rate],
                width = 1,
                dash = line_style_sto[sample_rate]
                ),
            marker = attr(
                    color = color_scheme[sample_rate],
                    symbol = "square",
                    size = 4
                )
            )
            reverse = reverse!(med_metr_sto - std_metr_sto)
            for l in eachindex(reverse)
                if reverse[l] < 0
                    reverse[l] = med_metr_sto[l]
                end
            end

            data_std_metr_slm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_sto), length(med_metr_sto):-1:1), y = vcat(med_metr_sto + std_metr_sto, reverse), mode="lines+markers", name = "$(sample_rate*100)%-N", fill="tozerox",
                fillcolor = color_scheme_std[sample_rate],
                line_color = "transparent",
                showlegend = false
            )

            push!(data_metr, data_metr_slm)#, data_std_metr_slm)
            
            # --------------- MSE DATA -------------------- #

            data_mse_slm = PlotlyJS.scatter(; x = 1:length(med_mse_sto), y = med_mse_sto, mode="lines+markers", name = "$(sample_rate*100)%-N", 
            line=attr(
                color = color_scheme[sample_rate], 
                width = 1,
                dash = line_style_sto[sample_rate]
                ),
            marker = attr(
                    color = color_scheme[sample_rate],
                    symbol = "square",
                    size = 4
                ),
                showlegend = false
            )

            data_std_mse_slm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_sto), length(med_mse_sto):-1:1), y = vcat(med_mse_sto + std_mse_sto, reverse!(med_mse_sto - std_mse_sto)), mode="lines+markers", name = "$(sample_rate*100)%-N", fill="tozerox",
                fillcolor = color_scheme_std[sample_rate],
                line_color = "transparent",
                showlegend = false
            )

            push!(data_mse, data_mse_slm, data_std_mse_slm)
        end

        ## -------------------------------- DYNAMIC SAMPLE RATE ------------------------------------- ##

        for version in versions
            med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob = load_mnist_plm(version, selected_h)
            # --------------- OBJECTIVE DATA -------------------- #

            data_obj_plm = PlotlyJS.scatter(; x = 1:length(med_obj_prob), y = med_obj_prob, mode="lines+markers", name = "$(prob_versions_names[version])-N", 
            line=attr(
                color = prob_versions_colors[version], 
                width = 1,
                dash = line_style_plm[version]
                ),
            marker = attr(
                    color = prob_versions_colors[version],
                    symbol = "triangle-up",
                    size = 4
                )
            )

            data_std_obj_plm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_prob), length(med_obj_prob):-1:1), y = vcat(med_obj_prob + std_obj_prob, reverse!(med_obj_prob - std_obj_prob)), mode="lines+markers", name = "$(prob_versions_names[version])-N", fill="tozerox",
                fillcolor = prob_versions_colors_std[version],
                line_color = "transparent",
                showlegend = false
            )

            push!(data_obj, data_obj_plm)#, data_std_obj_plm)

            # --------------- METRIC DATA -------------------- #

            data_metr_plm = PlotlyJS.scatter(; x = 1:length(med_metr_prob), y = med_metr_prob, mode="lines+markers", name = "$(prob_versions_names[version])-N", 
            line=attr(
                color = prob_versions_colors[version],
                width = 1,
                dash = line_style_plm[version]
                ),
            marker = attr(
                    color = prob_versions_colors[version],
                    symbol = "triangle-up",
                    size = 4
                )
            )

            reverse = reverse!(med_metr_prob - std_metr_prob)
            for l in eachindex(reverse)
                if reverse[l] < 0
                    reverse[l] = med_metr_prob[l]
                end
            end

            data_std_metr_plm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_prob), length(med_metr_prob):-1:1), y = vcat(med_metr_prob + std_metr_prob, reverse), mode="lines+markers", name = "$(prob_versions_names[version])-N", fill="tozerox",
                fillcolor = prob_versions_colors_std[version],
                line_color = "transparent",
                showlegend = false
            )

            push!(data_metr, data_metr_plm)#, data_std_metr_plm)
            
            # --------------- MSE DATA -------------------- #

            data_mse_plm = PlotlyJS.scatter(; x = 1:length(med_mse_prob), y = med_mse_prob, mode="lines+markers", name = "$(prob_versions_names[version])-N", 
            line=attr(
                color = prob_versions_colors[version], 
                width = 1,
                dash = line_style_plm[version]
                ),
            marker = attr(
                    color = prob_versions_colors[version],
                    symbol = "triangle-up",
                    size = 4
                ),
                showlegend = false
            )

            data_std_mse_plm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_prob), length(med_mse_prob):-1:1), y = vcat(med_mse_prob + std_mse_prob, reverse!(med_mse_prob - std_mse_prob)), mode="lines+markers", name = "$(prob_versions_names[version])-N", fill="tozerox",
                fillcolor = prob_versions_colors_std[version],
                line_color = "transparent",
                showlegend = false
            )

            push!(data_mse, data_mse_plm)#, data_std_mse_plm)

            if smooth
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
                        size = 4
                    )
                )

                data_std_obj_splm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_prob), length(med_obj_prob):-1:1), y = vcat(med_obj_prob + std_obj_prob, reverse!(med_obj_prob - std_obj_prob)), mode="lines+markers", name = "$(prob_versions_names[version])-S", fill="tozerox",
                    fillcolor = smooth_versions_colors_std[version],
                    line_color = "transparent",
                    showlegend = false
                )

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
                        size = 4
                    )
                )

                reverse = reverse!(med_metr_prob - std_metr_prob)
                for l in eachindex(reverse)
                    if reverse[l] < 0
                        reverse[l] = med_metr_prob[l]
                    end
                end

                data_std_metr_splm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_prob), length(med_metr_prob):-1:1), y = vcat(med_metr_prob + std_metr_prob, reverse), mode="lines+markers", name = "$(prob_versions_names[version])-S", fill="tozerox",
                    fillcolor = smooth_versions_colors_std[version],
                    line_color = "transparent",
                    showlegend = false
                )

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
                        size = 4
                    ),
                    showlegend = false
                )

                data_std_mse_splm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_prob), length(med_mse_prob):-1:1), y = vcat(med_mse_prob + std_mse_prob, reverse!(med_mse_prob - std_mse_prob)), mode="lines+markers", name = "$(prob_versions_names[version])-S", fill="tozerox",
                    fillcolor = smooth_versions_colors_std[version],
                    line_color = "transparent",
                    showlegend = false
                )

                push!(data_mse, data_mse_splm)#, data_std_mse_splm)
            end
        end

    layout_obj, layout_metr, layout_mse = layout("mnist-ls", n_runs, selected_h)
    plt_obj = PlotlyJS.plot(data_obj, layout_obj)
    plt_metr = PlotlyJS.plot(data_metr, layout_metr)
    plt_mse = PlotlyJS.plot(data_mse, layout_mse)

    display(plt_obj)
    display(plt_metr)
    display(plt_mse)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\MNIST_Graphs\1vs7\plots")
    PlotlyJS.savefig(plt_obj, "mnist-exactobj-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
    PlotlyJS.savefig(plt_metr, "mnist-metric-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
    PlotlyJS.savefig(plt_mse, "mnist-MSE-$digits-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
    end
end
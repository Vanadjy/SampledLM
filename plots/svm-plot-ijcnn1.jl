function plot_ijcnn1(sample_rates::AbstractVector, versions::AbstractVector, selected_hs::AbstractVector; abscissa = "CPU time", n_runs = 10, smooth::Bool = false, sample_rate0::Float64 = .05, compare::Bool = false, guide::Bool = false)
    include("plot-configuration.jl")

    ## -------------------------------- CONSTANT SAMPLE RATE ------------------------------------- ##

    for selected_h in selected_hs
        data_obj = GenericTrace{Dict{Symbol, Any}}[]
        data_metr = GenericTrace{Dict{Symbol, Any}}[]
        data_mse = GenericTrace{Dict{Symbol, Any}}[]
        for sample_rate in sample_rates
            med_obj_sto, med_metr_sto, med_mse_sto, std_obj_sto, std_metr_sto, std_mse_sto = load_ijcnn1_sto(sample_rate, selected_h)

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
                    size = 8
                ),
            showlegend = false
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
                    size = 8
                ),
                showlegend = false
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
                    size = 8
                ),
                showlegend = true
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
            med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob = load_ijcnn1_plm(version, selected_h)
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
                    size = 8
                ),
                showlegend = false
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
                    size = 8
                ),
                showlegend = true
            )

            data_std_mse_plm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_prob), length(med_mse_prob):-1:1), y = vcat(med_mse_prob + std_mse_prob, reverse!(med_mse_prob - std_mse_prob)), mode="lines+markers", name = "$(prob_versions_names[version])-N", fill="tozerox",
                fillcolor = prob_versions_colors_std[version],
                line_color = "transparent",
                showlegend = false
            )

            push!(data_mse, data_mse_plm)#, data_std_mse_plm)
        end

        layout_obj, layout_metr, layout_mse = layout("ijcnn1-ls", n_runs, selected_h)
        plt_obj = PlotlyJS.plot(data_obj, layout_obj)
        plt_metr = PlotlyJS.plot(data_metr, layout_metr)
        plt_mse = PlotlyJS.plot(data_mse, layout_mse)

        display(plt_obj)
        display(plt_metr)
        display(plt_mse)

        if isempty(sample_rates)
            cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\ijcnn1_graphs\PLM_only")
            PlotlyJS.savefig(plt_obj, "ijcnn1-exactobj-plm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
            PlotlyJS.savefig(plt_metr, "ijcnn1-metric-plm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
            PlotlyJS.savefig(plt_mse, "ijcnn1-MSE-plm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
        elseif isempty(versions)
            cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\ijcnn1_graphs\SLM_only")
            PlotlyJS.savefig(plt_obj, "ijcnn1-exactobj-slm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
            PlotlyJS.savefig(plt_metr, "ijcnn1-metric-slm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
            PlotlyJS.savefig(plt_mse, "ijcnn1-MSE-slm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
        else
            cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\ijcnn1_graphs\SLM_and_PLM")
            PlotlyJS.savefig(plt_obj, "ijcnn1-exactobj-slm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
            PlotlyJS.savefig(plt_metr, "ijcnn1-metric-slm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
            PlotlyJS.savefig(plt_mse, "ijcnn1-MSE-slm_only-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h-compare=$compare-smooth=$smooth.pdf"; format = "pdf")
        end
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
    end
end
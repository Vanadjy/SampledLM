function plot_ijcnn1(sample_rates::AbstractVector, versions::AbstractVector, selected_hs::AbstractVector; abscissa = "CPU time", n_runs = 10, smooth::Bool = false, sample_rate0::Float64 = .05, compare::Bool = false, guide::Bool = false, MaxEpochs::Int = 100)
    include("plot-configuration.jl")

    ## -------------------------------- CONSTANT SAMPLE RATE ------------------------------------- ##

    for selected_h in selected_hs
        data_obj = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
        data_metr = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
        data_mse = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]

        scatter_log = log_scale(MaxEpochs)

        for sample_rate in sample_rates
            med_obj_sto, med_metr_sto, med_mse_sto = load_ijcnn1_sto(sample_rate, selected_h)
            legend_mse_slm = PGFPlots.Plots.Linear(1:2, [1e16 for i in 1:2], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])", legendentry = "Cst\\_batch=$(Int(sample_rate*100))\\%")
            push!(data_mse, legend_mse_slm)
        end

        for version in versions
            med_obj_prob, med_metr_prob, med_mse_prob = load_ijcnn1_plm(version, selected_h)
            legend_mse_plm = PGFPlots.Plots.Linear(1:2, [1e16 for i in 1:2], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])", legendentry = "$(prob_versions_names[version])")
            push!(data_mse, legend_mse_plm)
        end

        for sample_rate in sample_rates
            med_obj_sto, med_metr_sto, med_mse_sto = load_ijcnn1_sto(sample_rate, selected_h)

            # --------------- OBJECTIVE DATA -------------------- #
            data_obj_slm = PGFPlots.Plots.Linear(1:length(med_obj_sto), (med_obj_sto), mark = "none", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")
            markers_obj_slm = PGFPlots.Plots.Scatter(scatter_log, (med_obj_sto[scatter_log]), mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate])", onlyMarks = true, markSize = 1.5)

            #=data_obj_slm = PlotlyJS.scatter(; x = 1:length(med_obj_sto), y = med_obj_sto, mode="lines+markers", name = "$(sample_rate*100)%-N", 
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
            )=#

            push!(data_obj, data_obj_slm, markers_obj_slm)#, data_std_obj_slm)

            # --------------- METRIC DATA -------------------- #

            data_metr_slm = PGFPlots.Plots.Linear(1:length(med_metr_sto), med_metr_sto, mark = "none", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")
            markers_metr_slm = PGFPlots.Plots.Scatter(scatter_log, med_metr_sto[scatter_log], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate])", onlyMarks = true, markSize = 1.5)


            #=data_metr_slm = PlotlyJS.scatter(; x = 1:length(med_metr_sto), y = med_metr_sto, mode="lines+markers", name = "$(sample_rate*100)%-N", 
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
            )=#

            push!(data_metr, data_metr_slm, markers_metr_slm)#, data_std_metr_slm)
            
            # --------------- MSE DATA -------------------- #

            data_mse_slm = PGFPlots.Plots.Linear(1:length(med_mse_sto), med_mse_sto, mark = "none", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])")
            markers_mse_slm = PGFPlots.Plots.Scatter(scatter_log, med_mse_sto[scatter_log], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate])", onlyMarks = true, markSize = 1.5)

            #=data_mse_slm = PlotlyJS.scatter(; x = 1:length(med_mse_sto), y = med_mse_sto, mode="lines+markers", name = "$(sample_rate*100)%-N", 
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
            )=#

            push!(data_mse, data_mse_slm, markers_mse_slm)#, data_std_mse_slm)
        end

        ## -------------------------------- DYNAMIC SAMPLE RATE ------------------------------------- ##

        for version in versions #plot ND versions of PLM
            med_obj_prob, med_metr_prob, med_mse_prob = load_ijcnn1_plm(version, selected_h)
            # --------------- OBJECTIVE DATA -------------------- #

            data_obj_plm = PGFPlots.Plots.Linear(1:length(med_obj_prob), (med_obj_prob), mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
            markers_obj_plm = PGFPlots.Plots.Scatter(scatter_log, (med_obj_prob[scatter_log]), mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)

            #=data_obj_plm = PlotlyJS.scatter(; x = 1:length(med_obj_prob), y = med_obj_prob, mode="lines+markers", name = "$(prob_versions_names[version])-N", 
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
            )=#

            push!(data_obj, data_obj_plm, markers_obj_plm)#, data_std_obj_plm)

            # --------------- METRIC DATA -------------------- #

            data_metr_plm = PGFPlots.Plots.Linear(1:length(med_metr_prob), med_metr_prob, mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
            markers_metr_plm = PGFPlots.Plots.Scatter(scatter_log, med_metr_prob[scatter_log], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)


            #=data_metr_plm = PlotlyJS.scatter(; x = 1:length(med_metr_prob), y = med_metr_prob, mode="lines+markers", name = "$(prob_versions_names[version])-N", 
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
            )=#

            push!(data_metr, data_metr_plm, markers_metr_plm)#, data_std_metr_plm)
            
            # --------------- MSE DATA -------------------- #
            
            data_mse_plm = PGFPlots.Plots.Linear(1:length(med_mse_prob), med_mse_prob, mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
            markers_mse_plm = PGFPlots.Plots.Scatter(scatter_log, med_mse_prob[scatter_log], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)

            #=data_mse_plm = PlotlyJS.scatter(; x = 1:length(med_mse_prob), y = med_mse_prob, mode="lines+markers", name = "$(prob_versions_names[version])-N", 
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
            )=#

            push!(data_mse, data_mse_plm, markers_mse_plm)#, data_std_mse_plm)
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
            ymax = 1.0
        )

        #display(plt_obj)
        #display(plt_metr)
        #display(plt_mse)

        if length(sample_rates) == 1
            if 2 in versions
                cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\ijcnn1_graphs\PLM_nd")
                PGFPlots.save("ijcnn1-exactobj-plm_nd-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_obj)
                PGFPlots.save("ijcnn1-metric-plm_nd-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_metr)
                PGFPlots.save("ijcnn1-MSE-plm_nd-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_mse)
            else
                cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\ijcnn1_graphs\PLM_ad")
                PGFPlots.save("ijcnn1-exactobj-plm_ad-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_obj)
                PGFPlots.save("ijcnn1-metric-plm_ad-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_metr)
                PGFPlots.save("ijcnn1-MSE-plm_ad-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_mse)
            end
        elseif isempty(versions)
            cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\ijcnn1_graphs\PLM_cst")
            PGFPlots.save("ijcnn1-exactobj-plm_cst-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_obj)
            PGFPlots.save("ijcnn1-metric-plm_cst-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_metr)
            PGFPlots.save("ijcnn1-MSE-plm_cst-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_mse)
        else
            cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\ijcnn1_graphs\SLM_and_PLM")
            PGFPlots.save("ijcnn1-exactobj-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_obj)
            PGFPlots.save("ijcnn1-metric-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_metr)
            PGFPlots.save("ijcnn1-MSE-$(n_runs)runs-$(MaxEpochs)epochs-$selected_h.tikz", plt_mse)
        end
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
    end
end
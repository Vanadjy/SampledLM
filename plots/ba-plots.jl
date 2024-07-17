function plot_ba(sample_rate, version::Int; n_runs = 10, smooth::Bool = false)
    include("plot-configuration.jl")

    data_obj = GenericTrace{Dict{Symbol, Any}}[]
    data_metr = GenericTrace{Dict{Symbol, Any}}[]
    data_mse = GenericTrace{Dict{Symbol, Any}}[]

    ## ---------------------------------------------------------------------------------------------------##
    ## ----------------------------------- CONSTANT SAMPLE RATE ------------------------------------------##
    ## ---------------------------------------------------------------------------------------------------##

    SLM_outs, slm_obj, med_obj_sto, std_obj_sto, med_metr_sto, std_metr_sto, med_mse_sto, std_mse_sto, nslm, ngslm = load_ba_slm(name, sample_rate; n_runs = n_runs)

    # --------------- OBJECTIVE DATA -------------------- #
    data_obj_slm = PlotlyJS.scatter(; x = 1:length(med_obj_sto), y = med_obj_sto, mode="lines", name = "SLM - $(prob_versions_names[version])", line=attr(
        color = prob_versions_colors[version], width = 1
        )
    )

    data_std_obj_slm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_sto), length(med_obj_sto):-1:1), y = vcat(med_obj_sto + std_obj_sto, reverse!(med_obj_sto - std_obj_sto)), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
    fillcolor = prob_versions_colors_std[version],
    line_color = "transparent",
    showlegend = false
    )

    push!(data_obj, data_obj_slm, data_std_obj_slm)

    # --------------- METRIC DATA -------------------- #

    data_metr_slm = PlotlyJS.scatter(; x = 1:length(med_metr_sto), y = med_metr_sto, mode="lines", name = "PLM - $(prob_versions_names[version])", line=attr(
    color = prob_versions_colors[version], width = 1
    )
    )

    data_std_metr_slm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_sto), length(med_metr_sto):-1:1), y = vcat(med_metr_sto + std_metr_sto, reverse!(med_metr_sto - std_metr_sto)), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
    fillcolor = prob_versions_colors_std[version],
    line_color = "transparent",
    showlegend = false
    )

    push!(data_metr, data_metr_slm, data_std_metr_slm)

    # --------------- MSE DATA -------------------- #

    data_mse_slm = PlotlyJS.scatter(; x = 1:length(med_mse_sto), y = med_mse_sto, mode="lines", name = "SLM - $(prob_versions_names[version])", line=attr(
    color = prob_versions_colors[version], width = 1
    )
    )

    data_std_mse_slm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_sto), length(med_mse_sto):-1:1), y = vcat(med_mse_sto + std_mse_sto, reverse!(med_mse_sto - std_mse_sto)), mode="lines", name = "PLM - $(prob_versions_names[version])", fill="tozerox",
    fillcolor = prob_versions_colors_std[version],
    line_color = "transparent",
    showlegend = false
    )

    push!(data_mse, data_mse_slm, data_std_mse_slm)

    ## ---------------------------------------------------------------------------------------------------##
    ## ----------------------------------- DYNAMIC SAMPLE RATE -------------------------------------------##
    ## ---------------------------------------------------------------------------------------------------##

    PLM_outs, plm_obj, med_obj_prob, std_obj_prob, med_metr_prob, std_metr_prob, med_mse_prob, std_mse_prob, nplm, ngplm = load_ba_plm(name, version; n_runs = n_runs)

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

    push!(data_mse, data_mse_plm, data_std_mse_plm)

    ## ---------------------------------------------------------------------------------------------------##
    ## -------------------------------------- SMOOTH VERSIONS --------------------------------------------##
    ## ---------------------------------------------------------------------------------------------------##

    if smooth
        # --------------- OBJECTIVE DATA -------------------- #
        data_obj_splm = PlotlyJS.scatter(; x = 1:length(med_obj_prob), y = med_obj_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
                            color = smooth_versions_colors[version], width = 1
                            )
                        )

        data_std_obj_splm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_prob), length(med_obj_prob):-1:1), y = vcat(med_obj_prob + std_obj_prob, reverse!(med_obj_prob - std_obj_prob)), mode="lines", name = "SPLM - $(prob_versions_names[version])", fill="tozerox",
            fillcolor = smooth_versions_colors_std[version],
            line_color = "transparent",
            showlegend = false
        )

        push!(data_obj, data_obj_splm, data_std_obj_splm)

        # --------------- METRIC DATA -------------------- #

        data_metr_splm = PlotlyJS.scatter(; x = 1:length(med_metr_prob), y = med_metr_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
            color = smooth_versions_colors[version], width = 1
            )
        )

        data_std_metr_splm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_prob), length(med_metr_prob):-1:1), y = vcat(med_metr_prob + std_metr_prob, reverse!(med_metr_prob - std_metr_prob)), mode="lines", name = "SPLM - $(prob_versions_names[version])", fill="tozerox",
            fillcolor = smooth_versions_colors_std[version],
            line_color = "transparent",
            showlegend = false
        )

        push!(data_metr, data_metr_splm, data_std_metr_splm)
        
        # --------------- MSE DATA -------------------- #

        data_mse_splm = PlotlyJS.scatter(; x = 1:length(med_mse_prob), y = med_mse_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
            color = smooth_versions_colors[version], width = 1
            )
        )

        data_std_mse_splm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_prob), length(med_mse_prob):-1:1), y = vcat(med_mse_prob + std_mse_prob, reverse!(med_mse_prob - std_mse_prob)), mode="lines", name = "SPLM - $(prob_versions_names[version])", fill="tozerox",
            fillcolor = smooth_versions_colors_std[version],
            line_color = "transparent",
            showlegend = false
        )

        push!(data_mse, data_mse_splm, data_std_mse_splm)
    end

    layout_obj, layout_metr, layout_mse = layout(name_list[1], n_runs, suffix)
    plt_obj = PlotlyJS.plot(data_obj, layout_obj)
    plt_metr = PlotlyJS.plot(data_metr, layout_metr)
    plt_mse = PlotlyJS.plot(data_mse, layout_mse)

    display(plt_obj)
    display(plt_metr)
    display(plt_mse)

    PlotlyJS.savefig(plt_obj, "ba-SLM-$name-exactobj-$(n_runs)runs-$(MaxEpochs)epochs-$h_name.pdf"; format = "pdf")
    PlotlyJS.savefig(plt_metr, "ba-SLM-$name-metric-$(n_runs)runs-$(MaxEpochs)epochs-$h_name.pdf"; format = "pdf")
    PlotlyJS.savefig(plt_mse, "ba-SLM-$name-MSE-$(n_runs)runs-$(MaxEpochs)epochs-$h_name.pdf"; format = "pdf")
end
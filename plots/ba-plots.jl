function plot_ba(name, sample_rates, versions; n_runs = 10, smooth::Bool = false, MaxEpochs::Int = 100)
    include("plot-configuration.jl")
    data_obj = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_metr = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]
    data_mse = Union{PGFPlots.Plots.Linear, PGFPlots.Plots.Scatter}[]

    scatter_log = log_scale(MaxEpochs)

    ## --------------------------------------------------------------------------------##
    ## ----------------------------------- LEGENDS ----------------------------------- ##
    ## --------------------------------------------------------------------------------##

    for sample_rate in sample_rates
        med_obj_sto, med_metr_sto, med_mse_sto, std_obj_sto, std_metr_sto, std_mse_sto, SLM_outs, slm_trains, nslm, ngslm = load_ba_slm(name, sample_rate)
        legend_obj_slm = PGFPlots.Plots.Linear(1:2, [1e16 for i in 1:2], mark = "$(symbols_cst_pgf[sample_rate])", style="$(color_scheme_pgf[sample_rate]), $(line_style_sto_pgf[sample_rate])", legendentry = "PLM (Cst\\_batch=$(Int(sample_rate*100))\\%)")
        push!(data_obj, legend_obj_slm)
    end

    for version in versions
        med_obj_prob, med_metr_prob, med_mse_prob, std_obj_prob, std_metr_prob, std_mse_prob = load_ba_splm(name, version; n_runs = n_runs)
        legend_obj_plm = PGFPlots.Plots.Linear(1:2, [1e16 for i in 1:2], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])", legendentry = "PLM ($(prob_versions_names[version]))")
        push!(data_obj, legend_obj_plm)
    end

    ## ---------------------------------------------------------------------------------------------------##
    ## ----------------------------------- CONSTANT SAMPLE RATE ------------------------------------------##
    ## ---------------------------------------------------------------------------------------------------##

    for sample_rate in sample_rates
        SLM_outs, slm_obj, med_obj_sto, std_obj_sto, med_metr_sto, std_metr_sto, med_mse_sto, std_mse_sto, nslm, ngslm = load_ba_slm(name, sample_rate)

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

    ## ---------------------------------------------------------------------------------------------------##
    ## ----------------------------------- DYNAMIC SAMPLE RATE -------------------------------------------##
    ## ---------------------------------------------------------------------------------------------------##

    for version in versions
        SPLM_outs, splm_obj, med_obj_prob_smooth, med_metr_prob_smooth, med_mse_prob_smooth, nsplm, ngsplm = load_ba_splm(name, version; n_runs = n_runs)
        # --------------- OBJECTIVE DATA -------------------- #
        markers_obj = vcat(filter(!>=(length(med_obj_prob_smooth)), scatter_log), length(med_obj_prob_smooth))
        data_obj_plm = PGFPlots.Plots.Linear(1:length(med_obj_prob_smooth), med_obj_prob_smooth, mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
        markers_obj_plm = PGFPlots.Plots.Scatter(markers_obj, med_obj_prob_smooth[markers_obj], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)

        push!(data_obj, data_obj_plm, markers_obj_plm)#, data_std_obj_plm)

        # --------------- METRIC DATA -------------------- #
        markers_metr = vcat(filter(!>=(length(med_metr_prob_smooth)), scatter_log), length(med_metr_prob_smooth))
        data_metr_plm = PGFPlots.Plots.Linear(1:length(med_metr_prob_smooth), med_metr_prob_smooth, mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
        markers_metr_plm = PGFPlots.Plots.Scatter(markers_metr, med_metr_prob_smooth[markers_metr], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)

        push!(data_metr, data_metr_plm, markers_metr_plm)#, data_std_metr_plm)
        
        # --------------- MSE DATA -------------------- #
        markers_metr = vcat(filter(!>=(length(med_mse_prob_smooth)), scatter_log), length(med_mse_prob_smooth))
        data_mse_plm = PGFPlots.Plots.Linear(1:length(med_mse_prob_smooth), med_mse_prob_smooth, mark="none", style="$(prob_versions_colors_pgf[version]), $(line_style_plm_pgf[version])")
        markers_mse_plm = PGFPlots.Plots.Scatter(markers_metr, med_mse_prob_smooth[markers_metr], mark = "$(symbols_nd_pgf[version])", style="$(prob_versions_colors_pgf[version])", onlyMarks = true, markSize = 1.5)

        push!(data_mse, data_mse_plm, markers_mse_plm)#, data_std_mse_plm)
    end

    plt_obj = PGFPlots.Axis(
        data_obj,
        xlabel="\$ j^{th}\$   epoch",
        ylabel="\$ f(x_j) \$",
        ymode="log",
        xmode="log",
        ymax = 8*1e6
    )
    plt_metr = PGFPlots.Axis(
        data_metr,
        xlabel="\$ j^{th}\$   epoch",
        ylabel="Stationarity measure",
        ymode="log",
        xmode="log",
        ymax = 4*1e8,
        #legendPos = "south west"
    )
    plt_mse = PGFPlots.Axis(
        data_mse,
        xlabel="\$ j^{th}\$   epoch",
        ylabel="MSE",
        ymode="log",
        xmode="log",
        ymax = 250
    )

    #display(plt_obj)
    #display(plt_metr)
    #display(plt_mse)

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\BundleAdjustment_Graphs\dubrovnik\plots")
    PGFPlots.save("ba-SLM-$name-exactobj-$(n_runs)runs-$(MaxEpochs)epochs-smooth.tikz", plt_obj)
    PGFPlots.save("ba-SLM-$name-metric-$(n_runs)runs-$(MaxEpochs)epochs-smooth.tikz", plt_metr)
    PGFPlots.save("ba-SLM-$name-MSE-$(n_runs)runs-$(MaxEpochs)epochs-smooth.tikz", plt_mse)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
end
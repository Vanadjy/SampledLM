using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators, MLDatasets, RegularizedProblems
using NLPModels, NLPModelsModifiers #ReverseADNLSModels
using RegularizedOptimization
using DataFrames
using SolverBenchmark
using PlotlyJS, PlutoPlotly

# Random.seed!(1234)

function demo_ba_sto(name_list::Vector{String}; sample_rate = .05, n_runs::Int = 1, MaxEpochs::Int = 20, MaxTime = 3600.0, version::Int = 4, suffix::String = "dubrovnik-h1", compare::Bool = false, smooth::Bool = false, Jac_lop::Bool = true)
    temp_PLM = []
    temp_PLM_smooth = []
    temp_LM = []
    temp_LMTR = []

    camera_settings = Dict(
        "problem-16-22106-pre" => attr(center = attr(x = 0.2072211130691765, y = -0.10068338752805728, z = -0.048807925112545746), eye = attr(x = 0.16748022386771697, y = -0.3957357535725894, z = 0.5547492387721914), up = attr(x = 0, y = 0, z = 1)),
        "problem-88-64298-pre" => attr(center = attr(x = -0.0021615530883736145, y = -0.030543602186994832, z = -0.028300153803163062), eye = attr(x = 0.6199398252619821, y = -0.4431229879708768, z = 0.3694699626625795), up = attr(x = -0.13087330856114893, y = 0.5787247595812629, z = 0.8049533090520641)),
        "problem-52-64053-pre" => attr(center = attr(x = 0.2060347573851926, y = -0.22421275022169654, z = -0.05597905955228791), eye = attr(x = 0.2065816892336426, y = -0.3978440066064094, z = 0.6414786827075296), up = attr(x = 0, y = 0, z = 1)),
        "problem-89-110973-pre" => attr(center = attr(x = -0.1674117968407976, y = -0.1429803633607516, z = 0.01606765828188431), eye = attr(x = 0.1427370965379074, y = -0.19278139431870447, z = 0.7245395074933954), up = attr(x = 0.02575289497167061, y = 0.9979331596959415, z = 0.05887441872199366))
    )

    color_scheme = Dict([(1.0, "rgb(255,105,180)"), (.2, "rgb(176,196,222)"), (.1, "rgb(205,133,63)"), (.05, "rgb(154,205,50)"), (.01, 8)])
    color_scheme_std = Dict([(1.0, "rgba(255,105,180, .2)"), (.2, "rgba(176,196,222, 0.2)"), (.1, "rgba(205,133,63, 0.2)"), (.05, "rgba(154,205,50, 0.2)"), (.01, 8)])

    prob_versions_names = Dict([(1, "mobmean"), (2, "nondec"), (3, "each-it"), (4, "hybrid")])
    prob_versions_colors = Dict([(1, "rgb(30,144,255)"), (2, "rgb(255,140,0)"), (3, "rgb(50,205,50)"), (4, "rgb(123,104,238)")])
    prob_versions_colors_std = Dict([(1, "rgba(30,144,255, 0.2)"), (2, "rgba(255,140,0, 0.2)"), (3, "rgba(50,205,50, 0.2)"), (4, "rgba(123,104,238, 0.2)")])

    Confidence = Dict([("95%", 1.96), ("99%", 2.58)])
    conf = "95%"

    for name in name_list
        nls = BundleAdjustmentModel(name)
        sampled_nls = BAmodel_sto(name; sample_rate = sample_rate)

        data_obj = GenericTrace{Dict{Symbol, Any}}[]
        data_metr = GenericTrace{Dict{Symbol, Any}}[]
        data_mse = GenericTrace{Dict{Symbol, Any}}[]

        #nlp_train = LSR1Model(nlp_train)
        λ = 1e-1
        # h = RootNormLhalf(λ)
        h = NormL1(λ)
        h_name = "l1"
        χ = NormLinf(1.0)

        if compare
            if smooth
                @info " using LM to solve"
                reset!(nls)
                LM_out = levenberg_marquardt(nls; max_iter = 20, in_itmax = 300)
                nlm = neval_residual(nls)
                nglm = neval_jtprod_residual(nls) + neval_jprod_residual(nls)

                sol = LM_out.solution
                x = [sol[3*i+1] for i in 0:(nls.npnts-1)]
                y = [sol[3*i+2] for i in 0:(nls.npnts-1)]
                z = [sol[3*i] for i in 1:nls.npnts]
                plt3d = PlotlyJS.plot(PlotlyJS.scatter(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=attr(
                        size=1,
                        opacity=0.8
                    ),
                    type="scatter3d"
                ), Layout(margin=attr(l=0, r=0, b=0, t=0)))
                relayout!(plt3d, template=:simple_white)
                display(plt3d)
                

                @info " using LMTR to solve with" χ
                reset!(nls)
                LMTR_out = levenberg_marquardt(nls, TR = true; max_iter = 20, in_itmax = 300)
                nlmtr = neval_residual(nls)
                nglmtr = neval_jtprod_residual(nls) + neval_jprod_residual(nls)

                sol = LMTR_out.solution
                x = [sol[3*i+1] for i in 0:(nls.npnts-1)]
                y = [sol[3*i+2] for i in 0:(nls.npnts-1)]
                z = [sol[3*i] for i in 1:nls.npnts]
                plt3d = PlotlyJS.plot(PlotlyJS.scatter(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=attr(
                        size=1,
                        opacity=0.8
                    ),
                    type="scatter3d"
                ), Layout(margin=attr(l=0, r=0, b=0, t=0)))
                relayout!(plt3d, template=:simple_white)
                display(plt3d)

                if name == name_list[1]
                    temp_LM = [0.5 * LM_out.rNorm^2, nlm, nglm, LM_out.elapsed_time]
                    temp_LMTR = [0.5 * LMTR_out.rNorm^2, nlmtr, nglmtr, LMTR_out.elapsed_time]
                else
                    temp_LM = hcat(temp_LM, [0.5 * LM_out.rNorm^2, nlm, nglm, LM_out.elapsed_time])
                    temp_LMTR = hcat(temp_LMTR, [0.5 * LMTR_out.rNorm^2, nlmtr, nglmtr, LMTR_out.elapsed_time])
                end
            end
        end

        options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-4, ϵr = 1e-4, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
        suboptions = RegularizedOptimization.ROSolverOptions(maxIter = 300)

        sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = 1e-4, ϵr = 1e-4, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
        if smooth
            @info "using SPLM to solve"

            PLM_outs = []
            plm_obj = []
    
            Obj_Hists_epochs_prob = zeros(1 + MaxEpochs, n_runs)
            Metr_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)
            MSE_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)
    
            for k in 1:n_runs
                reset!(sampled_nls)
                sampled_nls.epoch_counter = Int[1]
                Prob_LM_out_k = SPLM(sampled_nls, sampled_options, x0=sampled_nls.meta.x0, subsolver_options = suboptions, sample_rate0 = sample_rate, version = version, Jac_lop = Jac_lop)
                push!(PLM_outs, Prob_LM_out_k)
                push!(plm_obj, Prob_LM_out_k.objective)
    
                # get objective value for each run #
                @views Obj_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:ExactFhist][sampled_nls.epoch_counter]    
                # get MSE for each run #
                @views MSE_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:Fhist][sampled_nls.epoch_counter]
                @views MSE_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] ./= ceil.(2 * sampled_nls.nls_meta.nequ * Prob_LM_out_k.solver_specific[:SampleRateHist][sampled_nls.epoch_counter])
    
                # get metric for each run #
                @views Metr_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:ExactMetricHist][sampled_nls.epoch_counter]
            end
    
            if n_runs%2 == 1
                med_ind = (n_runs ÷ 2) + 1
            else
                med_ind = (n_runs ÷ 2)
            end
            sorted_obj_vec = sort(plm_obj)
            ref_value = sorted_obj_vec[med_ind]
            origin_ind = 0
            for i in eachindex(PLM_outs)
                if plm_obj[i] == ref_value
                    origin_ind = i
                end
            end
    
            # Prob_LM_out is the run associated to the median accuracy on the training set
            Prob_LM_out = PLM_outs[origin_ind]
    
            sol = Prob_LM_out.solution
    
            x = [sol[3*i+1] for i in 0:(sampled_nls.npnts-1)]
            y = [sol[3*i+2] for i in 0:(sampled_nls.npnts-1)]
            z = [sol[3*i] for i in 1:sampled_nls.npnts]       
            plt3d = PlotlyJS.scatter(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=attr(
                    size=1,
                    opacity=0.8
                ),
                type="scatter3d",
                options=Dict(:showLink => true)
            )
            
            layout = Layout(scene = attr(
                xaxis = attr(
                     backgroundcolor="rgb(255, 255, 255)",
                     title_text = "",
                     gridcolor="white",
                     showbackground=false,
                     zerolinecolor="white",
                     tickfont=attr(size=0, color="white")),
                yaxis = attr(
                    backgroundcolor="rgb(255, 255, 255)",
                    title_text = "",
                    gridcolor="white",
                    showbackground=false,
                    zerolinecolor="white",
                    tickfont=attr(size=0, color="white")),
                zaxis = attr(
                    backgroundcolor="rgb(255, 255, 255)",
                    title_text = "",
                    gridcolor="white",
                    showbackground=false,
                    zerolinecolor="white",
                    tickfont=attr(size=0, color="white")),
                    margin=attr(
                        r=10, l=10,
                        b=10, t=10),
                    aspectmode = "manual",
                    showlegend = false
                    ),
                    scene_camera = camera_settings[name]
              )
            
            #options = PlotConfig(plotlyServerURL="https://chart-studio.plotly.com", showlink = true)
            fig_ba = PlotlyJS.Plot(plt3d, layout)#; config = options)
            display(fig_ba)
            PlotlyJS.savefig(fig_ba, "ba-$name-3D-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth=$smooth.png"; format = "png")
    
            #println("Press enter")
            #n = readline()
    
            #nplm = neval_residual(sampled_nls)
            nsplm = length(sampled_nls.epoch_counter)
            ngsplm = (neval_jtprod_residual(sampled_nls) + neval_jprod_residual(sampled_nls))
    
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
            data_obj_splm = PlotlyJS.scatter(; x = 1:length(med_obj_prob), y = med_obj_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
                                color = prob_versions_colors[version], width = 1
                                )
                            )
    
            data_std_obj_splm = PlotlyJS.scatter(; x = vcat(1:length(med_obj_prob), length(med_obj_prob):-1:1), y = vcat(med_obj_prob + std_obj_prob, reverse!(med_obj_prob - std_obj_prob)), mode="lines", name = "SPLM - $(prob_versions_names[version])", fill="tozerox",
                fillcolor = prob_versions_colors_std[version],
                line_color = "transparent",
                showlegend = false
            )
    
            push!(data_obj, data_obj_splm, data_std_obj_splm)
    
            # --------------- METRIC DATA -------------------- #
    
            data_metr_splm = PlotlyJS.scatter(; x = 1:length(med_metr_prob), y = med_metr_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
                color = prob_versions_colors[version], width = 1
                )
            )
    
            data_std_metr_splm = PlotlyJS.scatter(; x = vcat(1:length(med_metr_prob), length(med_metr_prob):-1:1), y = vcat(med_metr_prob + std_metr_prob, reverse!(med_metr_prob - std_metr_prob)), mode="lines", name = "SPLM - $(prob_versions_names[version])", fill="tozerox",
                fillcolor = prob_versions_colors_std[version],
                line_color = "transparent",
                showlegend = false
            )
    
            push!(data_metr, data_metr_splm, data_std_metr_splm)
            
            # --------------- MSE DATA -------------------- #
    
            data_mse_splm = PlotlyJS.scatter(; x = 1:length(med_mse_prob), y = med_mse_prob, mode="lines", name = "SPLM - $(prob_versions_names[version])", line=attr(
                color = prob_versions_colors[version], width = 1
                )
            )
    
            data_std_mse_splm = PlotlyJS.scatter(; x = vcat(1:length(med_mse_prob), length(med_mse_prob):-1:1), y = vcat(med_mse_prob + std_mse_prob, reverse!(med_mse_prob - std_mse_prob)), mode="lines", name = "SPLM - $(prob_versions_names[version])", fill="tozerox",
                fillcolor = prob_versions_colors_std[version],
                line_color = "transparent",
                showlegend = false
            )
    
            push!(data_mse, data_mse_splm, data_std_mse_splm)
    
            # Results Table #
            if name == name_list[1]
                temp_PLM_smooth = [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, nsplm, ngsplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time]
            else
                temp_PLM_smooth = hcat(temp_PLM_smooth, [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, nsplm, ngsplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time])
            end
    
            layout_obj = Layout(title="BA - $name - $n_runs runs - h = $h_name-norm",
                            xaxis_title="epoch",
                            xaxis_type="log",
                            yaxis_type="log",
                            yaxis_title="Exact f+h",
                            template="simple_white")
                    
            layout_metr = Layout(title="BA - $name - $n_runs runs - h = $h_name-norm",
                    xaxis_title="epoch",
                    xaxis_type="log",
                    yaxis_type="log",
                    yaxis_title="√ξcp/ν",
                    template="simple_white")
    
            layout_mse = Layout(title="BA - $name - $n_runs runs - h = $h_name-norm",
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
    
            PlotlyJS.savefig(plt_obj, "ba-$name-exactobj-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth=$smooth.png"; format = "png")
            PlotlyJS.savefig(plt_metr, "ba-$name-metric-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth=$smooth.png"; format = "png")
            PlotlyJS.savefig(plt_mse, "ba-$name-MSE-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth=$smooth.png"; format = "png")
        end

        @info "using Prob_LM to solve with" h

        PLM_outs = []
        plm_obj = []

        Obj_Hists_epochs_prob = zeros(1 + MaxEpochs, n_runs)
        Metr_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)
        MSE_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)

        for k in 1:n_runs
            reset!(sampled_nls)
            sampled_nls.epoch_counter = Int[1]
            Prob_LM_out_k = Prob_LM(sampled_nls, h, sampled_options, x0=sampled_nls.meta.x0, subsolver_options = suboptions, version = version)
            push!(PLM_outs, Prob_LM_out_k)
            push!(plm_obj, Prob_LM_out_k.objective)

            # get objective value for each run #
            @views Obj_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:ExactFhist][sampled_nls.epoch_counter]
            @views Obj_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] += Prob_LM_out_k.solver_specific[:Hhist][sampled_nls.epoch_counter]

            # get MSE for each run #
            @views MSE_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:Fhist][sampled_nls.epoch_counter]
            @views MSE_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] += Prob_LM_out_k.solver_specific[:Hhist][sampled_nls.epoch_counter]
            @views MSE_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] ./= ceil.(2 * sampled_nls.nls_meta.nequ * Prob_LM_out_k.solver_specific[:SampleRateHist][sampled_nls.epoch_counter])

            # get metric for each run #
            @views Metr_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:ExactMetricHist][sampled_nls.epoch_counter]
        end

        if n_runs%2 == 1
            med_ind = (n_runs ÷ 2) + 1
        else
            med_ind = (n_runs ÷ 2)
        end
        sorted_obj_vec = sort(plm_obj)
        ref_value = sorted_obj_vec[med_ind]
        origin_ind = 0
        for i in eachindex(PLM_outs)
            if plm_obj[i] == ref_value
                origin_ind = i
            end
        end

        # Prob_LM_out is the run associated to the median accuracy on the training set
        Prob_LM_out = PLM_outs[origin_ind]

        sol = Prob_LM_out.solution

        x = [sol[3*i+1] for i in 0:(sampled_nls.npnts-1)]
        y = [sol[3*i+2] for i in 0:(sampled_nls.npnts-1)]
        z = [sol[3*i] for i in 1:sampled_nls.npnts]       
        plt3d = PlotlyJS.scatter(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=attr(
                size=1,
                opacity=0.8
            ),
            type="scatter3d",
            options=Dict(:showLink => true)
        )
        
        layout = Layout(scene = attr(
            xaxis = attr(
                 backgroundcolor="rgb(255, 255, 255)",
                 title_text = "",
                 gridcolor="white",
                 showbackground=false,
                 zerolinecolor="white",
                 tickfont=attr(size=0, color="white")),
            yaxis = attr(
                backgroundcolor="rgb(255, 255, 255)",
                title_text = "",
                gridcolor="white",
                showbackground=false,
                zerolinecolor="white",
                tickfont=attr(size=0, color="white")),
            zaxis = attr(
                backgroundcolor="rgb(255, 255, 255)",
                title_text = "",
                gridcolor="white",
                showbackground=false,
                zerolinecolor="white",
                tickfont=attr(size=0, color="white")),
                margin=attr(
                    r=10, l=10,
                    b=10, t=10),
                aspectmode = "manual",
                showlegend = false
                ),
                scene_camera = camera_settings[name]
          )
        
        #options = PlotConfig(plotlyServerURL="https://chart-studio.plotly.com", showlink = true)
        fig_ba = PlotlyJS.Plot(plt3d, layout)#; config = options)
        display(fig_ba)
        PlotlyJS.savefig(fig_ba, "ba-$name-3D-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth=$smooth.png"; format = "png")

        #println("Press enter")
        #n = readline()

        #nplm = neval_residual(sampled_nls)
        nplm = length(sampled_nls.epoch_counter)
        ngplm = (neval_jtprod_residual(sampled_nls) + neval_jprod_residual(sampled_nls))

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

        push!(data_mse, data_mse_plm, data_std_mse_plm)

        # Results Table #
        if name == name_list[1]
            temp_PLM = [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, nplm, ngplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time]
        else
            temp_PLM = hcat(temp_PLM, [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, nplm, ngplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time])
        end

        layout_obj = Layout(title="BA - $name - $n_runs runs - h = $h_name-norm",
                        xaxis_title="epoch",
                        xaxis_type="log",
                        yaxis_type="log",
                        yaxis_title="Exact f+h",
                        template="simple_white")
                
        layout_metr = Layout(title="BA - $name - $n_runs runs - h = $h_name-norm",
                xaxis_title="epoch",
                xaxis_type="log",
                yaxis_type="log",
                yaxis_title="√ξcp/ν",
                template="simple_white")

        layout_mse = Layout(title="BA - $name - $n_runs runs - h = $h_name-norm",
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

        PlotlyJS.savefig(plt_obj, "ba-$name-exactobj-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth=$smooth.png"; format = "png")
        PlotlyJS.savefig(plt_metr, "ba-$name-metric-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth=$smooth.png"; format = "png")
        PlotlyJS.savefig(plt_mse, "ba-$name-MSE-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth=$smooth.png"; format = "png")
    end

    if smooth
        temp = temp_PLM_smooth'
        df = DataFrame(temp, [:f, :h, :fh, :n, :g, :p, :s])
        #df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
        df[!, :Alg] = name_list
        select!(df, :Alg, Not(:Alg), :)
        fmt_override = Dict(:Alg => "%s",
            :f => "%10.2f",
            :h => "%10.2f",
            :fh => "%10.2f",
            :n => "%10.2f",
            :g => "%10.2f",
            :p => "%10.2f",
            :s => "%02.2f")
        hdr_override = Dict(:Alg => "Alg",
            :f => "\$ f \$",
            :h => "\$ h \$",
            :fh => "\$ f+h \$",
            :n => "\\# epochs",
            :g => "\\# \$ \\nabla f \$",
            :p => "\\# inner",
            :s => "\$t \$ (s)")
        open("BA-smooth-plm-$suffix.tex", "w") do io
            SolverBenchmark.pretty_latex_stats(io, df,
                col_formatters=fmt_override,
                hdr_override=hdr_override)
        end
    end

    temp = temp_PLM'
    df = DataFrame(temp, [:f, :h, :fh, :n, :g, :p, :s])
    #df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
    df[!, :Alg] = name_list
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :f => "%10.2f",
        :h => "%10.2f",
        :fh => "%10.2f",
        :n => "%10.2f",
        :g => "%10.2f",
        :p => "%10.2f",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :f => "\$ f \$",
        :h => "\$ h \$",
        :fh => "\$ f+h \$",
        :n => "\\# epochs",
        :g => "\\# \$ \\nabla f \$",
        :p => "\\# inner",
        :s => "\$t \$ (s)")
    open("BA-plm-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
    
    if compare
        temp = temp_LM'
        df = DataFrame(temp, [:fh, :n, :g, :s])
        #df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
        df[!, :Alg] = name_list
        select!(df, :Alg, Not(:Alg), :)
        fmt_override = Dict(:Alg => "%s",
            :fh => "%10.2f",
            :n => "%10.2f",
            :g => "%10.2f",
            :s => "%02.2f")
        hdr_override = Dict(:Alg => "Alg",
            :fh => "\$ f+h \$",
            :n => "\\# \$f\$",
            :g => "\\# \$ \\nabla f \$",
            :s => "\$t \$ (s)")
        open("BA-lm-$suffix.tex", "w") do io
            SolverBenchmark.pretty_latex_stats(io, df,
                col_formatters=fmt_override,
                hdr_override=hdr_override)
        end

        temp = temp_LMTR'
        df = DataFrame(temp, [:fh, :n, :g, :s])
        #df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
        df[!, :Alg] = name_list
        select!(df, :Alg, Not(:Alg), :)
        fmt_override = Dict(:Alg => "%s",
            :fh => "%10.2f",
            :n => "%10.2f",
            :g => "%10.2f",
            :s => "%02.2f")
        hdr_override = Dict(:Alg => "Alg",
            :fh => "\$ f+h \$",
            :n => "\\# \$f\$",
            :g => "\\# \$ \\nabla f \$",
            :s => "\$t \$ (s)")
        open("BA-lmtr-$suffix.tex", "w") do io
            SolverBenchmark.pretty_latex_stats(io, df,
                col_formatters=fmt_override,
                hdr_override=hdr_override)
        end
    end
end
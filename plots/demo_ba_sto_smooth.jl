function demo_ba_sto_smooth(name_list::Vector{String}; sample_rate0 = .05, n_runs::Int = 1, MaxEpochs::Int = 20, MaxTime = 3600.0, version::Int = 1, suffix::String = "dubrovnik-h1", Jac_lop::Bool = true)
    temp_PLM = []
    temp_PLM_smooth = []
    temp_LM = []
    temp_LMTR = []

    camera_settings = Dict(
        "problem-16-22106-pre" => attr(center = attr(x = -0.00020325635111987706, y = -0.11306200606736602, z = 0.033161420134634856), eye = attr(x = 0.3739093733537299, y = -0.4056038526945577, z = 0.5100004866202379), up = attr(x = 0.11632703501712824, y = 0.8846666814500118, z = 0.45147855281989546)),
        #"problem-88-64298-pre" => attr(center = attr(x = -0.0021615530883736145, y = -0.030543602186994832, z = -0.028300153803163062), eye = attr(x = 0.6199398252619821, y = -0.4431229879708768, z = 0.3694699626625795), up = attr(x = -0.13087330856114893, y = 0.5787247595812629, z = 0.8049533090520641)),
        #"problem-52-64053-pre" => attr(center = attr(x = 0.2060347573851926, y = -0.22421275022169654, z = -0.05597905955228791), eye = attr(x = 0.2065816892336426, y = -0.3978440066064094, z = 0.6414786827075296), up = attr(x = 0, y = 0, z = 1)),
        #"problem-89-110973-pre" => attr(center = attr(x = -0.1674117968407976, y = -0.1429803633607516, z = 0.01606765828188431), eye = attr(x = 0.1427370965379074, y = -0.19278139431870447, z = 0.7245395074933954), up = attr(x = 0.02575289497167061, y = 0.9979331596959415, z = 0.05887441872199366)),
        "problem-21-11315-pre" => attr(center = attr(x = 0, y = 0, z = 1), eye = attr(x = 1.25, y = 1.25, z = 1.2), up = attr(x = 0, y = 0, z = 0)),
        #"problem-49-7776-pre" => attr(center = attr(x = 0.12011665286185144, y = 0.2437548728183421, z = 0.6340730201867651), eye = attr(x = 0.14156235059481262, y = 0.49561706850854814, z = 0.48335380789220556), up = attr(x = 0.9853593274726773, y = 0.01757909714618111, z = 0.169581753458674))
    )

    include("plot-configuration.jl")

    for name in name_list
        nls = BundleAdjustmentModel(name)

        sampled_nls_ba = BAmodel_sto(name; sample_rate = 1.0)
        meta_nls_ba = nls_meta(sampled_nls_ba)

        function F!(Fx, x)
            residual!(sampled_nls_ba, x, Fx)
        end

        #=rows = Vector{Int}(undef, nls.nls_meta.nnzj)
        cols = Vector{Int}(undef, nls.nls_meta.nnzj)
        vals = ones(Bool, nls.nls_meta.nnzj)
        jac_structure_residual!(nls, rows, cols)
        J = sparse(rows, cols, vals, meta_nls_ba.nequ, meta_nls_ba.nvar)
        
        jac_back = ADNLPModels.SparseADJacobian(meta_nls_ba.nvar, F!, meta_nls_ba.nequ, nothing, J)=#
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ba\jac_backs")
        #save_object("jac_back-ba-$(name).jld2", jac_back)
        jac_back = load_object("jac_back-ba-$(name).jld2")
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

        adnls = ADNLSModel!(F!, sampled_nls_ba.meta.x0,  meta_nls_ba.nequ, sampled_nls_ba.meta.lvar, sampled_nls_ba.meta.uvar, jacobian_residual_backend = jac_back,
            jprod_residual_backend = ADNLPModels.ForwardDiffADJprod,
            jtprod_residual_backend = ADNLPModels.ReverseDiffADJtprod,
            jacobian_backend = ADNLPModels.EmptyADbackend,
            hessian_backend = ADNLPModels.EmptyADbackend,
            hessian_residual_backend = ADNLPModels.EmptyADbackend,
            matrix_free = true)

        sampled_nls_ba.sample_rate = sample_rate0
        sampled_nls = SADNLSModel_BA(adnls, sampled_nls_ba)
        guess_0 = sampled_nls_ba.nls_meta.x0

        sol0 = guess_0
        x0 = [sol0[3*i+1] for i in 0:(sampled_nls_ba.npnts-1)]
        y0 = [sol0[3*i+2] for i in 0:(sampled_nls_ba.npnts-1)]
        z0 = [sol0[3*i] for i in 1:sampled_nls_ba.npnts]
        plt3d0 = PlotlyJS.scatter(
            x=x0,
            y=y0,
            z=z0,
            mode="markers",
            marker=attr(
                size=1,
                opacity=0.8,
                color = "firebrick"
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

        #options = PlotConfig(plotlyServerURL="https://chart-studio.plotly.com", showLink = true)
        fig_ba0 = PlotlyJS.Plot(plt3d0, layout)#; config = options)
        #display(fig_ba0)
        #println("Press enter")
        #n = readline()

        data_obj = GenericTrace{Dict{Symbol, Any}}[]
        data_metr = GenericTrace{Dict{Symbol, Any}}[]
        data_mse = GenericTrace{Dict{Symbol, Any}}[]

        #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, γ = 10, ϵa = 1e-4, ϵr = 1e-4, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
        suboptions = RegularizedOptimization.ROSolverOptions(maxIter = 2000)

        sampled_options = ROSolverOptions(η3 = .4, ν = 1e0, νcp = 1e0, β = 1e16, σmax = 1e6, ϵa = 1e-8, ϵr = 1e-8, σmin = 1e-6, μmin = 1e-10, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)    

        @info "using SPLM at starting rate: $(sample_rate0*100)%"

        PLM_outs = []
        plm_obj = []
        nplms = []
        ngplms = []

        Obj_Hists_epochs_prob = zeros(1 + MaxEpochs, n_runs)
        Metr_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)
        MSE_Hists_epochs_prob = zero(Obj_Hists_epochs_prob)

        for k in 1:n_runs
            reset!(sampled_nls)
            sampled_nls.epoch_counter = Int[1]
            Prob_LM_out_k = SPLM(sampled_nls, sampled_options, version; x0=guess_0, subsolver_options = suboptions, sample_rate0 = sample_rate0, Jac_lop = Jac_lop)

            push!(PLM_outs, Prob_LM_out_k)
            push!(plm_obj, Prob_LM_out_k.objective)
            push!(nplms, length(sampled_nls.epoch_counter))
            push!(ngplms, (neval_jtprod_residual(sampled_nls_ba) + neval_jprod_residual(sampled_nls_ba)))

            # get objective value for each run #
            @views Obj_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:ExactFhist][sampled_nls.epoch_counter]    
            # get MSE for each run #
            @views MSE_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:Fhist][sampled_nls.epoch_counter]
            @views MSE_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] ./= ceil.(2 * sampled_nls.nls_meta.nequ * Prob_LM_out_k.solver_specific[:SampleRateHist][sampled_nls.epoch_counter])

            # get metric for each run #
            @views Metr_Hists_epochs_prob[:, k][1:length(sampled_nls.epoch_counter)] = Prob_LM_out_k.solver_specific[:ExactMetricHist][sampled_nls.epoch_counter]
        end

        if sample_rate0 == 1.0
            save_object("SLM_outs-SLM-ba-$name-$(n_runs)runs-$(sample_rate0*100).jld2", PLM_outs)
            save_object("slm_obj-SLM-ba-$name-$(n_runs)runs-$(sample_rate0*100).jld2", plm_obj)
        else
            save_object("SPLM_outs-SPLM-ba-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2", PLM_outs)
            save_object("splm_obj-SPLM-ba-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2", plm_obj)
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
    
        
        fig_ba = PlotlyJS.Plot(plt3d, layout)
        #display(fig_ba)
        #PlotlyJS.savefig(fig_ba, "ba-$name-3D-$(n_runs)runs-$(MaxEpochs)epochs-smooth.pdf"; format = "pdf")
        #PlotlyJS.savefig(fig_ba0, "ba-$name-3D-x0-$(n_runs)runs-$(MaxEpochs)epochs-smooth.pdf"; format = "pdf")

        #nplm = neval_residual(sampled_nls)
        nsplm = length(sampled_nls.epoch_counter)
        ngsplm = Prob_LM_out.solver_specific[:NLSGradHist][end]

        if sample_rate0 == 1.0
            save_object("nslm-SLM-ba-$name-$(sample_rate0*100).jld2", nsplm)
            save_object("ngslm-SLM-ba-$name-$(sample_rate0*100).jld2", ngsplm)
        else
            save_object("nsplm-SPLM-ba-$name-$version.jld2", nsplm)
            save_object("ngsplm-SPLM-ba-$name-$version.jld2", ngsplm)
        end

        med_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))
        std_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))

        med_metr_prob = zeros(axes(Metr_Hists_epochs_prob, 1))
        std_metr_prob = zeros(axes(Metr_Hists_epochs_prob, 1))

        med_mse_prob = zeros(axes(MSE_Hists_epochs_prob, 1))
        std_mse_prob = zeros(axes(MSE_Hists_epochs_prob, 1))

        # compute median of objective value #
        for l in eachindex(med_obj_prob)
            cleared_obj_prob = filter(!isnan, filter(!iszero, Obj_Hists_epochs_prob[l, :]))
            if isempty(cleared_obj_prob)
                med_obj_prob[l] = NaN
                std_obj_prob[l] = NaN
            elseif length(cleared_obj_prob) == 1
                med_obj_prob[l] = median(cleared_obj_prob)
                std_obj_prob[l] = 0.0
            else
                med_obj_prob[l] = median(cleared_obj_prob)
                std_obj_prob[l] = std(cleared_obj_prob)
            end
        end
        filter!(!isnan, med_obj_prob)
        std_obj_prob *= Confidence[conf]
        filter!(!isnan, std_obj_prob)

        if sample_rate0 == 1.0
            save_object("med_obj_sto-$(n_runs)runs-ba-$name-$(sample_rate0*100).jld2", med_obj_prob)
            save_object("std_obj_sto-$(n_runs)runs-ba-$name-$(sample_rate0*100).jld2", std_obj_prob)
        else
            save_object("med_obj_prob_smooth-$(n_runs)runs-ba-$name-$(prob_versions_names[version]).jld2", med_obj_prob)
            save_object("std_obj_prob_smooth-$(n_runs)runs-ba-$name-$(prob_versions_names[version]).jld2", std_obj_prob)
        end

        # compute median of metric #
        for l in eachindex(med_metr_prob)
            cleared_metr_prob = filter(!isnan, filter(!iszero, Metr_Hists_epochs_prob[l, :]))
            if isempty(cleared_metr_prob)
                med_metr_prob[l] = NaN
                std_metr_prob[l] = NaN
            elseif length(cleared_metr_prob) == 1
                med_metr_prob[l] = median(cleared_metr_prob)
                std_metr_prob[l] = 0.0
            else
                med_metr_prob[l] = median(cleared_metr_prob)
                std_metr_prob[l] = std(cleared_metr_prob)
            end
        end
        filter!(!isnan, med_metr_prob)
        std_metr_prob *= Confidence[conf]
        filter!(!isnan, std_metr_prob)

        if sample_rate0 == 1.0
            save_object("med_metr_sto-$(n_runs)runs-ba-$name-$(sample_rate0*100).jld2", med_metr_prob)
            save_object("std_metr_sto-$(n_runs)runs-ba-$name-$(sample_rate0*100).jld2", std_metr_prob)
        else
            save_object("med_metr_prob_smooth-$(n_runs)runs-ba-$name-$(prob_versions_names[version]).jld2", med_metr_prob)
            save_object("std_metr_prob_smooth-$(n_runs)runs-ba-$name-$(prob_versions_names[version]).jld2", std_metr_prob)
        end
        
        # compute median of MSE #
        for l in eachindex(med_mse_prob)
            cleared_mse_prob = filter(!isnan, filter(!iszero, MSE_Hists_epochs_prob[l, :]))
            if isempty(cleared_mse_prob)
                med_mse_prob[l] = NaN
                std_mse_prob[l] = NaN
            elseif length(cleared_mse_prob) == 1
                med_mse_prob[l] = median(cleared_mse_prob)
                std_mse_prob[l] = 0.0
            else
                med_mse_prob[l] = (median(cleared_mse_prob))
                std_mse_prob[l] = std(cleared_mse_prob)
            end
        end
        filter!(!isnan, med_mse_prob)
        std_mse_prob *= Confidence[conf]
        filter!(!isnan, std_mse_prob)

        if sample_rate0 == 1.0
            save_object("med_mse_sto-$(n_runs)runs-ba-$name-$(sample_rate0*100).jld2", med_mse_prob)
            save_object("std_mse_sto-$(n_runs)runs-ba-$name-$(sample_rate0*100).jld2", std_mse_prob)
        else
            save_object("med_mse_prob_smooth-$(n_runs)runs-ba-$name-$(prob_versions_names[version]).jld2", med_mse_prob)
            save_object("std_mse_prob_smooth-$(n_runs)runs-ba-$name-$(prob_versions_names[version]).jld2", std_mse_prob)
        end

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

        # Results Table #
        if name == name_list[1]
            temp_PLM_smooth = [Prob_LM_out.solver_specific[:Fhist][end], 0.0, Prob_LM_out.objective, nsplm, ngsplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time]
        else
            temp_PLM_smooth = hcat(temp_PLM_smooth, [Prob_LM_out.solver_specific[:Fhist][end], 0.0, Prob_LM_out.objective, nsplm, ngsplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time])
        end

        #=layout_obj = Layout(title="$name - $n_runs runs - $suffix",
        xaxis_title="epoch",
        xaxis_type="log",
        yaxis =attr(
                showexponent = "all",
                exponentformat = "e"
            ),
        yaxis_type="log",
        yaxis_title="Exact f+h",
        template="simple_white",
        legend = attr(
            xanchor="right",
            bgcolor="rgba(255,255,255,.4)"
        ),
        font=attr(size=13))

        layout_metr = Layout(title="$name - $n_runs runs - $suffix",
                xaxis_title="epoch",
                xaxis_type="log",
                yaxis =attr(
                    showexponent = "all",
                    exponentformat = "e"
                ),
                yaxis_type="log",
                yaxis_title="√ξcp/ν",
                template="simple_white",
                legend = attr(
                    xanchor="right",
                    bgcolor="rgba(255,255,255,.4)"
                ),
                font=attr(size=13))

        layout_mse = Layout(title="$name - $n_runs runs - $suffix",
                xaxis_title="epoch",
                xaxis_type="log",
                yaxis =attr(
                    showexponent = "all",
                    exponentformat = "e"
                ),
                yaxis_type="log",
                yaxis_title="MSE",
                template="simple_white",
                legend = attr(
                    xanchor="right",
                    bgcolor="rgba(255,255,255,.4)"
                ),
                font=attr(size=13))
        
        plt_obj = PlotlyJS.plot(data_obj, layout_obj)
        plt_metr = PlotlyJS.plot(data_metr, layout_metr)
        plt_mse = PlotlyJS.plot(data_mse, layout_mse)

        display(plt_obj)
        display(plt_metr)
        display(plt_mse)

        PlotlyJS.savefig(plt_obj, "ba-$name-exactobj-$(n_runs)runs-$(MaxEpochs)epochs-smooth.pdf"; format = "pdf")
        PlotlyJS.savefig(plt_metr, "ba-$name-metric-$(n_runs)runs-$(MaxEpochs)epochs-smooth.pdf"; format = "pdf")
        PlotlyJS.savefig(plt_mse, "ba-$name-MSE-$(n_runs)runs-$(MaxEpochs)epochs-smooth.pdf"; format = "pdf")=#

        temp = temp_PLM_smooth'
        df = DataFrame(temp, [:f, :h, :fh, :n, :g, :p, :s])
        #df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
        df[!, :Alg] = name_list
        select!(df, :Alg, Not(:Alg), :)
        fmt_override = Dict(:Alg => "%s",
            :f => "%10.2e",
            :h => "%10.2e",
            :fh => "%10.2e",
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
end
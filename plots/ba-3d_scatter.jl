function ba_3d_scatter(name_list::Vector{String}, sample_rates::Vector{Float64}, versions::Vector{Int}; n_runs::Int = 10)
    include("plot-configuration.jl")

    # Camera settings for BA 3D-scatter
    camera_settings = Dict(
        "problem-16-22106-pre" => attr(center = attr(x = -0.00020325635111987706, y = -0.11306200606736602, z = 0.033161420134634856), eye = attr(x = 0.3739093733537299, y = -0.4056038526945577, z = 0.5100004866202379), up = attr(x = 0.11632703501712824, y = 0.8846666814500118, z = 0.45147855281989546)),
        #"problem-49-7776-pre" => attr(center = attr(x = 0.12011665286185144, y = 0.2437548728183421, z = 0.6340730201867651), eye = attr(x = 0.14156235059481262, y = 0.49561706850854814, z = 0.48335380789220556), up = attr(x = 0.9853593274726773, y = 0.01757909714618111, z = 0.169581753458674))
    )

    layout_3d = layout3d(name, camera_settings)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\BundleAdjustment_Graphs\dubrovnik\3d-scatter")

    for name in name_list
        sampled_nls = BAmodel_sto(name; sample_rate = sample_rate)

        sol0 = sampled_nls.nls_meta.x0
        x0 = [sol0[3*i+1] for i in 0:(sampled_nls.npnts-1)]
        y0 = [sol0[3*i+2] for i in 0:(sampled_nls.npnts-1)]
        z0 = [sol0[3*i] for i in 1:sampled_nls.npnts]       
        plt3d0 = PlotlyJS.scatter(
            x=x0,
            y=y0,
            z=z0,
            mode="markers",
            marker=attr(
                size = .7,
                opacity=0.8,
                color = "firebrick"
            ),
            type="scatter3d",
            options=Dict(:showLink => true)
        )
        fig_ba0 = PlotlyJS.Plot(plt3d0, layout_3d)
        PlotlyJS.savefig(fig_ba0, "ba-$name-3D-x0-$(n_runs)runs-$(MaxEpochs)epochs.pdf"; format = "pdf")

        for sample_rate in sample_rates
            SLM_outs, slm_obj, med_obj_sto, std_obj_sto, med_metr_sto, std_metr_sto, med_mse_sto, std_mse_sto, nslm, ngslm = load_ba_slm(name, sample_rate; n_runs = n_runs)

            # SLM_out is the run associated to the median final objective value
            if n_runs%2 == 1
                med_ind = (n_runs ÷ 2) + 1
            else
                med_ind = (n_runs ÷ 2)
            end
            sorted_obj_vec = sort(slm_obj)
            ref_value = sorted_obj_vec[med_ind]
            origin_ind = 0
            for i in eachindex(SLM_outs)
                if slm_obj[i] == ref_value
                    origin_ind = i
                end
            end
            SLM_out = SLM_outs[origin_ind]
            sol_slm = SLM_out.solution

            x_slm = [sol_slm[3*i+1] for i in 0:(sampled_nls.npnts-1)]
            y_slm = [sol_slm[3*i+2] for i in 0:(sampled_nls.npnts-1)]
            z_slm = [sol_slm[3*i] for i in 1:sampled_nls.npnts]       
            plt3d_slm = PlotlyJS.scatter(
                x=x_slm,
                y=y_slm,
                z=z_slm,
                mode="markers",
                marker=attr(
                    size = .7,
                    opacity=0.8,
                    color = "green"
                ),
                type="scatter3d",
                options=Dict(:showLink => true)
            )

            fig_ba_slm = PlotlyJS.plot(plt3d_slm, layout_3d)
            PlotlyJS.savefig(fig_ba_slm, "ba-$name-3D-SLM-$(n_runs)runs-$(MaxEpochs)epochs.pdf"; format = "pdf")
        end

        for version in versions
            SPLM_outs, splm_obj, med_obj_prob_smooth, med_metr_prob_smooth, med_mse_prob_smooth, nsplm, ngsplm = load_ba_splm(name, version)

            # Prob_LM_out is the run associated to the median final objective value
            if n_runs%2 == 1
                med_ind = (n_runs ÷ 2) + 1
            else
                med_ind = (n_runs ÷ 2)
            end
            sorted_obj_vec = sort(splm_obj)
            ref_value = sorted_obj_vec[med_ind]
            origin_ind = 0
            for i in eachindex(SPLM_outs)
                if splm_obj[i] == ref_value
                    origin_ind = i
                end
            end
            Prob_LM_out = SPLM_outs[origin_ind]
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
                    size = .7,
                    opacity=0.8
                ),
                type="scatter3d",
                options=Dict(:showLink => true)
            )

            #options = PlotConfig(plotlyServerURL="https://chart-studio.plotly.com", showlink = true)
            fig_ba = PlotlyJS.Plot(plt3d, layout_3d)#; config = options)
            PlotlyJS.savefig(fig_ba, "ba-$name-3D-PLM-$(n_runs)runs-$(MaxEpochs)epochs.pdf"; format = "pdf")
        end

        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
    end
end
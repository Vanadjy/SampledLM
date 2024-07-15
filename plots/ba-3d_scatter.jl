function ba_3d_scatter(name_list::Vector{String}; sample_rate = .05, n_runs::Int = 1)
    include("plot-configuration.jl")

    # Camera settings for BA 3D-scatter
    camera_settings = Dict(
        "problem-16-22106-pre" => attr(center = attr(x = 0.2072211130691765, y = -0.10068338752805728, z = -0.048807925112545746), eye = attr(x = 0.16748022386771697, y = -0.3957357535725894, z = 0.5547492387721914), up = attr(x = 0, y = 0, z = 1)),
        "problem-88-64298-pre" => attr(center = attr(x = -0.0021615530883736145, y = -0.030543602186994832, z = -0.028300153803163062), eye = attr(x = 0.6199398252619821, y = -0.4431229879708768, z = 0.3694699626625795), up = attr(x = -0.13087330856114893, y = 0.5787247595812629, z = 0.8049533090520641)),
        "problem-52-64053-pre" => attr(center = attr(x = 0.2060347573851926, y = -0.22421275022169654, z = -0.05597905955228791), eye = attr(x = 0.2065816892336426, y = -0.3978440066064094, z = 0.6414786827075296), up = attr(x = 0, y = 0, z = 1)),
        "problem-89-110973-pre" => attr(center = attr(x = -0.1674117968407976, y = -0.1429803633607516, z = 0.01606765828188431), eye = attr(x = 0.1427370965379074, y = -0.19278139431870447, z = 0.7245395074933954), up = attr(x = 0.02575289497167061, y = 0.9979331596959415, z = 0.05887441872199366)),
        "problem-21-11315-pre" => attr(center = attr(x = 0, y = 0, z = 1), eye = attr(x = 1.25, y = 1.25, z = 1.2), up = attr(x = 0, y = 0, z = 0)),
        "problem-49-7776-pre" => attr(center = attr(x = 0.12011665286185144, y = 0.2437548728183421, z = 0.6340730201867651), eye = attr(x = 0.14156235059481262, y = 0.49561706850854814, z = 0.48335380789220556), up = attr(x = 0.9853593274726773, y = 0.01757909714618111, z = 0.169581753458674))
    )

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
                size=1,
                opacity=0.8,
                color = "firebrick"
            ),
            type="scatter3d",
            options=Dict(:showLink => true)
        )

        PLM_outs, plm_obj = load_ba_plm(name, version; n_runs = n_runs)

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

        layout3d = layout3d(name)

        #options = PlotConfig(plotlyServerURL="https://chart-studio.plotly.com", showlink = true)
        fig_ba = PlotlyJS.Plot(plt3d, layout3d)#; config = options)
        fig_ba0 = PlotlyJS.Plot(plt3d0, layout3d)
        display(fig_ba)
        display(fig_ba0)
        PlotlyJS.savefig(fig_ba, "ba-$name-3D-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth.pdf"; format = "pdf")
        PlotlyJS.savefig(fig_ba0, "ba-$name-3D-x0-$(n_runs)runs-$(MaxEpochs)epochs-$h_name-compare=$compare-smooth.pdf"; format = "pdf")
    end
end
function ba_tables(name, sample_rates, versions; suffix::String = "smooth", n_runs = 10)
    model = BundleAdjustmentModel(name)

    meta_nls = nls_meta(model)
    Fx = similar(model.meta.x0, meta_nls.nequ)
    residual!(model, model.meta.x0, Fx)
    rows = Vector{Int}(undef, meta_nls.nnzj)
    cols = Vector{Int}(undef, meta_nls.nnzj)
    jac_structure_residual!(model, rows, cols)
    vals = similar(model.meta.x0, meta_nls.nnzj)
    jac_coord_residual!(model, model.meta.x0, vals)
    Jv = similar(model.meta.x0, meta_nls.nequ)
    Jtv = similar(model.meta.x0, meta_nls.nvar)
    Jx = jac_op_residual!(model, rows, cols, vals, Jv, Jtv)

    fx0 = 0.5*norm(Fx)^2
    gx0 = norm(Jx'Fx)
    temp_ba = [0, 0, 0, 0, 0, 0, 0]

    for sample_rate in sample_rates
        SLM_outs, slm_obj, med_obj_sto, std_obj_sto, med_metr_sto, std_metr_sto, med_mse_sto, std_mse_sto, nslm, ngslm = load_ba_slm(name, sample_rate)

        # SLM_out is the run associated to the median final objective value
        if n_runs%2 == 1 && sample_rate < 1.0
            med_ind = (n_runs ÷ 2) + 1
        elseif sample_rate == 1.0
            med_ind = 1
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

        temp_ba = hcat(temp_ba,
        [fx0, SLM_out.objective, gx0, SLM_out.solver_specific[:ExactMetricHist][end], nslm, ngslm, SLM_out.elapsed_time]
        )

    end
    
    for version in versions
        SPLM_outs, splm_obj, med_obj_prob_smooth, med_metr_prob_smooth, med_mse_prob_smooth, nsplm, ngsplm = load_ba_splm(name, version; n_runs = n_runs)#((name == "problem-16-22106-pre" && version == 2) ? 5 : n_runs))

        # Prob_LM_out is the run associated to the median final objective value
        if n_runs%2 == 1
            med_ind = (n_runs ÷ 2) + 1
        else
            med_ind = (n_runs ÷ 2)
        end
        if version == 2
            med_ind = 1
        end
        sorted_obj_vec = sort(splm_obj)
        ref_value = sorted_obj_vec[med_ind]
        origin_ind = 0
        for i in eachindex(SPLM_outs)
            if splm_obj[i] == ref_value
                origin_ind = i
            end
        end
        SPLM_out = SPLM_outs[origin_ind]
        SampleRateHist = SPLM_out.solver_specific[:SampleRateHist]
        non_deterministic_counter = length(filter!(x -> x < 1.0, SampleRateHist))

        temp_ba = hcat(temp_ba, 
            [fx0, SPLM_out.objective, gx0, SPLM_out.solver_specific[:ExactMetricHist][end], nsplm, ngsplm - non_deterministic_counter, SPLM_out.elapsed_time]
        )
    end

    temp = temp_ba'
    df = DataFrame(temp, [:f0, :fh, :xi0, :xi, :n, :g, :s])
    df[!, :Alg] = vcat(["f0"], ["PLM-$(sample_rate*100)" for sample_rate in sample_rates], ["PLM-$(prob_versions_names[version])" for version in versions])
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :f0 => "%10.2e",
        :fh => "%10.2e",
        :xi0 => "%10.2e",
        :xi => "%10.2e",
        :n => "%10.2f",
        :g => "%10.2f",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :f0 => "\$ f(x_0) \$",
        :fh => "\$ f(x) \$",
        :xi0 => "\$ \\| \\nabla f(x_0) \\| \$",
        :xi => "\$ \\| \\nabla f(x) \\| \$",
        :n => "\\# epochs",
        :g => "\\# \$ \\nabla f \$",
        :s => "\$t \$ (s)")
    
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Graphes\BundleAdjustment_Graphs\dubrovnik\tables")
    open("BA-$name-plm-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")
end
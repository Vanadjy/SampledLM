function ba_tables(name, sample_rates, versions; suffix::String = "smooth", n_runs = 10)
    nls = BundleAdjustmentModel(name)
    temp_ba = [obj(nls, nls.meta.x0), norm(grad(nls, nls.meta.x0)), "undefined", 0, 0, 0, 0]

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

        temp_ba = hcat(temp_ba,
        [SLM_out.objective, SLM_out.solver_specific[:ExactMetricHist][end], SLM_out.status, nslm, ngslm, sum(SLM_out.solver_specific[:SubsolverCounter]), SLM_out.elapsed_time]
        )

    end
    
    for version in versions
        SPLM_outs, splm_obj, med_obj_prob_smooth, med_metr_prob_smooth, med_mse_prob_smooth, nsplm, ngsplm = load_ba_splm(name, version; n_runs = n_runs)

        # Prob_LM_out is the run associated to the median final objective value
        if n_runs%2 == 1
            med_ind = (n_runs ÷ 2) + 1
        else
            med_ind = (n_runs ÷ 2)
        end
        sorted_obj_vec = sort(splm_obj)
        ref_value = sorted_obj_vec[med_ind]
        origin_ind = 0
        for i in eachindex(PLM_outs)
            if splm_obj[i] == ref_value
                origin_ind = i
            end
        end
        SPLM_out = SPLM_outs[origin_ind]

        temp_ba = hcat(temp_ba, 
            [SPLM_out.objective, SPLM_out.solver_specific[:ExactMetricHist][end], SPLM_out.status, nsplm, ngsplm, sum(SPLM_out.solver_specific[:SubsolverCounter]), SPLM_out.elapsed_time]
        )
    end

    temp = temp_ba'
    df = DataFrame(temp, [:fh, :xi, :st, :n, :g, :p, :s])
    df[!, :Alg] = vcat(["f0"], ["PLM-$(prob_versions_names[version])" for version in versions], ["PLM-$(sample_rate*100)" for sample_rate in sample_rates])
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :fh => "%10.2e",
        :xi => "%10.2e",
        :st => "%10s",
        :n => "%10.2f",
        :g => "%10.2f",
        :p => "%10.2f",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :fh => "\$ f \$",
        :xi => "\$ \\| \\nabla f \\| \$",
        :st => "status"
        :n => "\\# epochs",
        :g => "\\# \$ \\nabla f \$",
        :p => "\\# inner",
        :s => "\$t \$ (s)")
    open("BA-$name-plm-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
end
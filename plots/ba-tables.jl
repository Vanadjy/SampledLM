function ba_tables(name, sample_rate, version; suffix::String = "l1", n_runs = 10, smooth::Bool = false)

    SLM_outs, slm_obj, med_obj_sto, std_obj_sto, med_metr_sto, std_metr_sto, med_mse_sto, std_mse_sto, nslm, ngslm = load_ba_slm(name, sample_rate; n_runs = n_runs)
    PLM_outs, plm_obj, med_obj_prob, std_obj_prob, med_metr_prob, std_metr_prob, med_mse_prob, std_mse_prob, nplm, ngplm = load_ba_plm(name, version; n_runs = n_runs)

    # Prob_LM_out is the run associated to the median final objective value
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
    Prob_LM_out = PLM_outs[origin_ind]

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

    temp_ba = hcat(
        [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, nplm, ngplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]), Prob_LM_out.elapsed_time],
        [SLM_out.solver_specific[:Fhist][end], SLM_out.solver_specific[:Hhist][end], SLM_out.objective, nslm, ngslm, sum(SLM_out.solver_specific[:SubsolverCounter]), SLM_out.elapsed_time]
    )

    if smooth
        temp_ba = hcat(
            temp_ba, [SPLM_out.solver_specific[:Fhist][end], SPLM_out.solver_specific[:Hhist][end], SPLM_out.objective, nsplm, ngsplm, sum(SPLM_out.solver_specific[:SubsolverCounter]), SPLM_out.elapsed_time]
        )
    end

    temp = temp_ba'
    df = DataFrame(temp, [:f, :h, :fh, :n, :g, :p, :s])
    df[!, :Alg] = !smooth ? ["PLM-$(prob_versions_names[version])", "SLM-$(sample_rate*100)"] : ["PLM-$(prob_versions_names[version])", "SLM-$(sample_rate*100)", "SPLM-$(prob_versions_names[version])"]
    #df[!, :Alg] = name_list
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
    open("BA-$name-plm-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
end
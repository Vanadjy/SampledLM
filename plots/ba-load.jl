function load_ba_plm(name, version; n_runs = 10)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ba")

    PLM_outs = load_object("PLM_outs-PLM-ba-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2")
    plm_obj = load_object("plm_obj-PLM-ba-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2")

    med_obj_prob = load_object("med_obj_prob-$(n_runs)runs-ba-$name-$(prob_versions_names[version])-l1.jld2")
    std_obj_prob = load_object("std_obj_prob-$(n_runs)runs-ba-$name-$(prob_versions_names[version])-l1.jld2")
    med_metr_prob = load_object("med_metr_prob-$(n_runs)runs-ba-$name-$(prob_versions_names[version])-l1.jld2")
    std_metr_prob = load_object("std_metr_prob-$(n_runs)runs-ba-$name-$(prob_versions_names[version])-l1.jld2")
    med_mse_prob = load_object("med_mse_prob-$(n_runs)runs-ba-$name-$(prob_versions_names[version])-l1.jld2")
    std_mse_prob = load_object("std_mse_prob-$(n_runs)runs-ba-$name-$(prob_versions_names[version])-l1.jld2")

    nplm = load_object("nplm-PLM-ba-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2")
    ngplm = load_object("ngplm-PLM-ba-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    return PLM_outs, plm_obj, med_obj_prob, std_obj_prob, med_metr_prob, std_metr_prob, med_mse_prob, std_mse_prob, nplm, ngplm
end

function load_ba_slm(name, sample_rate; n_runs = 10)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ba")

    SLM_outs = load_object("SLM_outs-SLM-ba-$name-$(n_runs)runs-$(sample_rate*100).jld2")
    slm_obj = load_object("slm_obj-SLM-ba-$name-$(n_runs)runs-$(sample_rate*100).jld2")

    med_obj_sto = load_object("med_obj_sto-$(n_runs)runs-ba-$name-$(sample_rate*100)-l1.jld2")
    std_obj_sto = load_object("std_obj_sto-$(n_runs)runs-ba-$name-$(sample_rate*100)-l1.jld2")
    med_metr_sto = load_object("med_metr_sto-$(n_runs)runs-ba-$name-$(sample_rate*100)-l1.jld2")
    std_metr_sto = load_object("std_metr_sto-$(n_runs)runs-ba-$name-$(sample_rate*100)-l1.jld2")
    med_mse_sto = load_object("med_mse_sto-$(n_runs)runs-ba-$name-$(sample_rate*100)-l1.jld2")
    std_mse_sto = load_object("std_mse_sto-$(n_runs)runs-ba-$name-$(sample_rate*100)-l1.jld2")

    nslm = load_object("nslm-SLM-ba-$name-$(sample_rate*100).jld2")
    ngslm = load_object("ngslm-SLM-ba-$name-$(sample_rate*100).jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    return SLM_outs, slm_obj, med_obj_sto, std_obj_sto, med_metr_sto, std_metr_sto, med_mse_sto, std_mse_sto, nslm, ngslm
end
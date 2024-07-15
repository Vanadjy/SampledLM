function load_ba_plm(name, version; n_runs = 10)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ba")

    PLM_outs = load_object("PLM_outs-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2")
    plm_obj = load_object("plm_obj-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    return PLM_outs, plm_obj
end

function load_ba_slm(name, version; n_runs = 10)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ba")

    SLM_outs = load_object("SLM_outs-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2")
    slm_obj = load_object("slm_obj-$name-$(n_runs)runs-$(prob_versions_names[version]).jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    return SLM_outs, slm_obj
end
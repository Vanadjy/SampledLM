function load_mnist_r2()
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")
    k_R2 = load_object(raw"k_R2-mnist-lhalf.jld2")
    R2_out = load_object("R2_out-mnist-lhalf.jld2")
    R2_stats = load_object("R2_stats-mnist-lhalf.jld2")
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return k_R2, R2_out, R2_stats
end

function load_mnist_lm_lmtr()
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")
    LM_out = load_object("LM_out-mnist-lhalf.jld2")
    LMTR_out = load_object("LMTR_out-mnist-lhalf.jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return LM_out, LMTR_out
end

function load_mnist_plm(version, selected_h)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")

    med_obj_prob_mnist = load_object("med_obj_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    med_metr_prob_mnist = load_object("med_metr_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    med_mse_prob_mnist = load_object("med_mse_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    std_obj_prob_mnist = load_object("std_obj_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    std_metr_prob_mnist = load_object("std_metr_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    std_mse_prob_mnist = load_object("std_mse_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    PLM_outs = load_object("PLM_outs-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    plm_trains = load_object("plm_trains-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return med_obj_prob_mnist, med_metr_prob_mnist, med_mse_prob_mnist, std_obj_prob_mnist, std_metr_prob_mnist, std_mse_prob_mnist, PLM_outs, plm_trains
end

function load_mnist_splm(version, selected_h)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")

    med_obj_prob_mnist_smooth = load_object("med_obj_prob_smooth-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    med_metr_prob_mnist_smooth = load_object("med_metr_prob_smooth-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    med_mse_prob_mnist_smooth = load_object("med_mse_prob_smooth-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    std_obj_prob_mnist_smooth = load_object("std_obj_prob_smooth-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    std_metr_prob_mnist_smooth = load_object("std_metr_prob_smooth-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    std_mse_prob_mnist_smooth = load_object("std_mse_prob_smooth-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    SPLM_outs = load_object("SPLM_outs-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    splm_trains = load_object("splm_trains-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return med_obj_prob_mnist_smooth, med_metr_prob_mnist_smooth, med_mse_prob_mnist_smooth, std_obj_prob_mnist_smooth, std_metr_prob_mnist_smooth, std_mse_prob_mnist_smooth, SPLM_outs, splm_trains
end

function load_mnist_sto(sample_rate, selected_h)
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")

    med_obj_sto_mnist = load_object("med_obj_sto-$(sample_rate*100)-mnist-train-ls-$selected_h.jld2")
    med_metr_sto_mnist = load_object("med_metr_sto-$(sample_rate*100)-mnist-$selected_h.jld2")
    med_mse_sto_mnist = load_object("med_mse_sto-$(sample_rate*100)-mnist-$selected_h.jld2")

    std_obj_sto_mnist = load_object("std_obj_sto-$(sample_rate*100)-mnist-train-ls-$selected_h.jld2")
    std_metr_sto_mnist = load_object("std_metr_sto-$(sample_rate*100)-mnist-$selected_h.jld2")
    std_mse_sto_mnist = load_object("std_mse_sto-$(sample_rate*100)-mnist-$selected_h.jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return med_obj_sto_mnist, med_metr_sto_mnist, med_mse_sto_mnist, std_obj_sto_mnist, std_metr_sto_mnist, std_mse_sto_mnist
end
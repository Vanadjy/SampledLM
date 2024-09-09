function load_mnist_r2(selected_h; MaxEpochs::Int = 1000)
    if selected_h == "smooth"
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist\smooth_jld2")
    else
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")
    end
    if MaxEpochs == 20
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist\small_budget")
        R2_stats = load_object("R2_stats-mnist-$selected_h-Epoch=20.jld2")
        r2_metric_hist = load_object("r2_metric_hist-mnist-$selected_h-Epoch=20.jld2")
        r2_obj_hist = load_object("r2_obj_hist-mnist-$selected_h-Epoch=20.jld2")
        r2_numjac_hist = load_object("r2_numjac_hist-mnist-$selected_h-Epoch=20.jld2")
    else
        #k_R2 = load_object(raw"k_R2-mnist-lhalf.jld2")
        #R2_out = load_object("R2_out-mnist-lhalf.jld2")
        R2_stats = load_object("R2_stats-mnist-$selected_h.jld2")
        r2_metric_hist = load_object("r2_metric_hist-mnist-$selected_h.jld2")
        r2_obj_hist = load_object("r2_obj_hist-mnist-$selected_h.jld2")
        r2_numjac_hist = load_object("r2_numjac_hist-mnist-$selected_h.jld2")
    end
    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return R2_stats, r2_metric_hist, r2_obj_hist, r2_numjac_hist
end

function load_mnist_lm_lmtr(selected_h)
    if selected_h == "smooth"
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist\smooth_jld2")
    else
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")
    end
    LM_out = load_object("LM_out-mnist-$selected_h.jld2")
    LMTR_out = load_object("LMTR_out-mnist-$selected_h.jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return LM_out, LMTR_out
end

function load_mnist_plm(version, selected_h)
    if selected_h == "smooth"
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist\smooth_jld2")
    else
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")
    end

    med_obj_prob_mnist = load_object("med_obj_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    med_metr_prob_mnist = load_object("med_metr_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    med_mse_prob_mnist = load_object("med_mse_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    std_obj_prob_mnist = load_object("std_obj_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    std_metr_prob_mnist = load_object("std_metr_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    std_mse_prob_mnist = load_object("std_mse_prob-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    PLM_outs = load_object("PLM_outs-$(prob_versions_names[version])-mnist-$selected_h.jld2")
    plm_trains = load_object("plm_trains-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    nplm = load_object("nplm-mnist-PLM-$(prob_versions_names[version]).jld2")
    ngplm = load_object("ngplm-mnist-PLM-$(prob_versions_names[version]).jld2")

    epoch_counters_plm = load_object("epoch_counters_plm-$(prob_versions_names[version])-mnist-$selected_h.jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return med_obj_prob_mnist, med_metr_prob_mnist, med_mse_prob_mnist, std_obj_prob_mnist, std_metr_prob_mnist, std_mse_prob_mnist, PLM_outs, plm_trains, nplm, ngplm, epoch_counters_plm
end

function load_mnist_splm(version, selected_h)
    if selected_h == "smooth"
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist\smooth_jld2")
    else
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")
    end

    med_obj_prob_mnist_smooth = load_object("med_obj_prob_smooth-$(prob_versions_names[version])-mnist.jld2")
    med_metr_prob_mnist_smooth = load_object("med_metr_prob_smooth-$(prob_versions_names[version])-mnist.jld2")
    med_mse_prob_mnist_smooth = load_object("med_mse_prob_smooth-$(prob_versions_names[version])-mnist.jld2")

    std_obj_prob_mnist_smooth = load_object("std_obj_prob_smooth-$(prob_versions_names[version])-mnist.jld2")
    std_metr_prob_mnist_smooth = load_object("std_metr_prob_smooth-$(prob_versions_names[version])-mnist.jld2")
    std_mse_prob_mnist_smooth = load_object("std_mse_prob_smooth-$(prob_versions_names[version])-mnist.jld2")

    SPLM_outs = load_object("SPLM_outs-$(prob_versions_names[version])-mnist.jld2")
    splm_trains = load_object("splm_trains-$(prob_versions_names[version])-mnist.jld2")

    nsplm = load_object("nsplm-mnist-PLM-$(prob_versions_names[version]).jld2")
    ngsplm = load_object("ngsplm-mnist-PLM-$(prob_versions_names[version]).jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return med_obj_prob_mnist_smooth, med_metr_prob_mnist_smooth, med_mse_prob_mnist_smooth, std_obj_prob_mnist_smooth, std_metr_prob_mnist_smooth, std_mse_prob_mnist_smooth, SPLM_outs, splm_trains, nsplm, ngsplm
end

function load_mnist_sto(sample_rate, selected_h)
    if selected_h == "smooth"
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist\smooth_jld2")
        med_obj_sto_mnist = load_object("med_obj_sto_smooth-$(sample_rate*100)-mnist-train-ls.jld2")
        med_metr_sto_mnist = load_object("med_metr_sto_smooth-$(sample_rate*100)-mnist.jld2")
        med_mse_sto_mnist = load_object("med_mse_sto_smooth-$(sample_rate*100)-mnist.jld2")

        std_obj_sto_mnist = load_object("std_obj_sto_smooth-$(sample_rate*100)-mnist-train-ls.jld2")
        std_metr_sto_mnist = load_object("std_metr_sto_smooth-$(sample_rate*100)-mnist.jld2")
        std_mse_sto_mnist = load_object("std_mse_sto_smooth-$(sample_rate*100)-mnist.jld2")

        SLM_outs = load_object("SSLM_outs-$(sample_rate*100)%-mnist.jld2")
        slm_trains = load_object("sslm_trains-$(sample_rate*100)%-mnist.jld2")

        nslm = load_object("nsslm-mnist-PLM-$(sample_rate*100)%.jld2")
        ngslm = load_object("ngsslm-mnist-PLM-$(sample_rate*100)%.jld2")
    else
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\mnist")
        med_obj_sto_mnist = load_object("med_obj_sto-$(sample_rate*100)-mnist-train-ls-$selected_h.jld2")
        med_metr_sto_mnist = load_object("med_metr_sto-$(sample_rate*100)-mnist-$selected_h.jld2")
        med_mse_sto_mnist = load_object("med_mse_sto-$(sample_rate*100)-mnist-$selected_h.jld2")

        std_obj_sto_mnist = load_object("std_obj_sto-$(sample_rate*100)-mnist-train-ls-$selected_h.jld2")
        std_metr_sto_mnist = load_object("std_metr_sto-$(sample_rate*100)-mnist-$selected_h.jld2")
        std_mse_sto_mnist = load_object("std_mse_sto-$(sample_rate*100)-mnist-$selected_h.jld2")

        SLM_outs = load_object("SLM_outs-$(sample_rate*100)%-mnist-$selected_h.jld2")
        slm_trains = load_object("slm_trains-$(sample_rate*100)%-mnist-$selected_h.jld2")

        nslm = load_object("nslm-mnist-PLM-$(sample_rate*100)%.jld2")
        ngslm = load_object("ngslm-mnist-PLM-$(sample_rate*100)%.jld2")
    end


    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\SampledLM")

    return med_obj_sto_mnist, med_metr_sto_mnist, med_mse_sto_mnist, std_obj_sto_mnist, std_metr_sto_mnist, std_mse_sto_mnist, SLM_outs, slm_trains, nslm, ngslm
end
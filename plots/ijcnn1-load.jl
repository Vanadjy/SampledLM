function load_ijcnn1_sto(sample_rate, selected_h; MaxEpoch::Int = 100)
    if selected_h == "smooth"
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ijcnn1\smooth_jld2")
        med_obj_sto_ijcnn1 = load_object("med_obj_sto_smooth-$(sample_rate*100)-ijcnn1-ls-Epoch=$MaxEpoch.jld2")
        med_metr_sto_ijcnn1 = load_object("med_metr_sto_smooth-$(sample_rate*100)-ijcnn1-Epoch=$MaxEpoch.jld2")
        med_mse_sto_ijcnn1 = load_object("med_mse_sto_smooth-$(sample_rate*100)-ijcnn1-Epoch=$MaxEpoch.jld2")
    else
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ijcnn1\nonsmooth_jld2")
        med_obj_sto_ijcnn1 = load_object("med_obj_sto-$(sample_rate*100)-ijcnn1-ls-$selected_h-Epoch=$MaxEpoch.jld2")
        med_metr_sto_ijcnn1 = load_object("med_metr_sto-$(sample_rate*100)-ijcnn1-$selected_h-Epoch=$MaxEpoch.jld2")
        med_mse_sto_ijcnn1 = load_object("med_mse_sto-$(sample_rate*100)-ijcnn1-$selected_h-Epoch=$MaxEpoch.jld2")
    end

    #std_obj_sto_ijcnn1 = load_object("std_obj_sto-$(sample_rate*100)-ijcnn1-ls-$selected_h.jld2")
    #std_metr_sto_ijcnn1 = load_object("std_metr_sto-$(sample_rate*100)-ijcnn1-$selected_h.jld2")
    #std_mse_sto_ijcnn1 = load_object("std_mse_sto-$(sample_rate*100)-ijcnn1-$selected_h.jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    return med_obj_sto_ijcnn1, med_metr_sto_ijcnn1, med_mse_sto_ijcnn1
end

function load_ijcnn1_plm(version, selected_h; MaxEpoch::Int = 100)
    if selected_h == "smooth"
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ijcnn1\smooth_jld2")
        med_obj_prob_ijcnn1 = load_object("med_obj_prob_smooth-$(prob_versions_names[version])-ijcnn1-Epoch=$MaxEpoch.jld2")
        med_metr_prob_ijcnn1 = load_object("med_metr_prob_smooth-$(prob_versions_names[version])-ijcnn1-Epoch=$MaxEpoch.jld2")
        med_mse_prob_ijcnn1 = load_object("med_mse_prob_smooth-$(prob_versions_names[version])-ijcnn1-Epoch=$MaxEpoch.jld2")
    else
        cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\JLD2saves\ijcnn1\nonsmooth_jld2")
        med_obj_prob_ijcnn1 = load_object("med_obj_prob-$(prob_versions_names[version])-ijcnn1-$selected_h-Epoch=$MaxEpoch.jld2")
        med_metr_prob_ijcnn1 = load_object("med_metr_prob-$(prob_versions_names[version])-ijcnn1-$selected_h-Epoch=$MaxEpoch.jld2")
        med_mse_prob_ijcnn1 = load_object("med_mse_prob-$(prob_versions_names[version])-ijcnn1-$selected_h-Epoch=$MaxEpoch.jld2")
    end

    #std_obj_prob_ijcnn1 = load_object("std_obj_prob-$(prob_versions_names[version])-ijcnn1-$selected_h.jld2")
    #std_metr_prob_ijcnn1 = load_object("std_metr_prob-$(prob_versions_names[version])-ijcnn1-$selected_h.jld2")
    #std_mse_prob_ijcnn1 = load_object("std_mse_prob-$(prob_versions_names[version])-ijcnn1-$selected_h.jld2")

    cd(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages")

    return med_obj_prob_ijcnn1, med_metr_prob_ijcnn1, med_mse_prob_ijcnn1
end
Random.seed!(seed)

n_exec = 10
sample_rate0 = .05
sample_rates = Float64[1.0]
versions = Int[2, 7, 9]

filter_name = "dubrovnik"
name_list = ba_data(filter_name)

MaxEpochs = 100
MaxTime = 2e4

smooth = true
Jac_lop = false

local_plots = false

#plot_Sto_LM_BA(sample_rates, versions, name_list, selected_hs; abscissa = abscissa, n_exec = n_exec, smooth = true, sample_rate0 = sample_rate0, compare = true, MaxEpochs = MaxEpochs, MaxTime = MaxTime)
Random.seed!(seed)

if !local_plots
    if !smooth
        demo_ba_sto(name_list; sample_rate = sample_rate, sample_rate0 = sample_rate0, n_runs = n_exec, MaxEpochs = MaxEpochs, MaxTime = MaxTime, version = version, suffix = "$filter_name-l1", compare = false, smooth = smooth, Jac_lop = Jac_lop)
    else
        for version in versions
            demo_ba_sto_smooth(name_list; sample_rate0 = sample_rate0, n_runs = n_exec, MaxEpochs = MaxEpochs, MaxTime = MaxTime, version = version, suffix = "$filter_name-smooth", Jac_lop = Jac_lop)
        end
    end
else
    ba_3d_scatter(name_list, sample_rates, versions; n_runs = n_exec)
    for name in name_list
        plot_ba(name, sample_rates, versions; n_runs = n_exec, MaxEpochs = MaxEpochs)
        ba_tables(name, sample_rates, versions; n_runs = n_exec)
    end
end
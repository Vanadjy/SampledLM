n_exec = 10
selected_probs = ["mnist"]
MaxEpochs = 0
MaxTime = 0.0

if selected_probs == ["ijcnn1"]
    sample_rate0 = .05
    sample_rates = [1.0]#, .1, .05, .01]
    selected_digits = [(1, 7)] # let only one pair of random digits
    versions = [2, 9]
    #version = versions[end]
    ϵ = 1e-16
    selected_hs = ["l1"]#, "lhalf" "smooth"]
    MaxEpochs = 100
    MaxTime = 3600.0
    smooth = false
    compare = false
elseif selected_probs == ["mnist"]
    sample_rate0 = .05
    sample_rates = [1.0, .05]
    selected_digits = [(1, 7)]
    versions = [2, 9]
    #version = versions[end]
    selected_hs = ["lhalf"]
    ϵ = 1e-4
    MaxEpochs = 1000
    MaxTime = 3600.0
    smooth = false
    compare = true
end

local_plots = false
abscissas = ["epoch", "CPU time"]
abscissa = abscissas[1]

if !local_plots
    if !smooth
        svm_plot_epoch(sample_rates, versions, selected_probs, selected_hs, selected_digits; abscissa = abscissa, n_exec = n_exec, sample_rate0 = sample_rate0, compare = compare, MaxEpochs = MaxEpochs, MaxTime = MaxTime, precision = ϵ)
    else
        smooth_svm_plot_epoch(sample_rates, versions, selected_probs, selected_digits; abscissa = abscissa, n_exec = n_exec, sample_rate0 = sample_rate0, compare = compare, MaxEpochs = MaxEpochs, MaxTime = MaxTime, precision = ϵ)
    end
    # -- Plots for MNIST grey map -- #

    Random.seed!(seed)
    #=for digits in selected_digits
        demo_svm_sto(;sample_rate = sample_rate0, n_runs = n_exec, digits = digits, MaxEpochs = MaxEpochs, MaxTime = MaxTime, version = version, smooth = true)
    end=#
else
    # local plots #

    ## DISCLAIMER: Properly separate the versions to plot before calling plot functions

    if selected_probs == ["ijcnn1"]
        plot_ijcnn1(sample_rates, versions, selected_hs; n_runs = n_exec, MaxEpochs = MaxEpochs)
    elseif selected_probs == ["mnist"]
        plot_mnist(sample_rates, versions, selected_hs; n_runs = n_exec, smooth = smooth)
        greymaps_tables_mnist(versions, sample_rates, sample_rate0; smooth = smooth)
    end
end
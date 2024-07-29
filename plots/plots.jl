include("plot-configuration.jl")
#include("layout.jl")
include("layout-3d-ba.jl")

include("plot-utils-svm-sto.jl")
include("demo_svm_sto.jl")
include("plots-svm.jl")
include("PLM-plots-svm.jl")

include("PLM-plots-ba.jl")
include("demo_ba_sto.jl")

## local work for Plot ##

#ijcnn1
include("ijcnn1-load.jl")
include("svm-plot-ijcnn1.jl")

#mnist
include("mnist-load.jl")
include("svm-plot-mnist.jl")
include("svm-mnist-greymaps-tables.jl")

#Bundle Adjustment
include("ba-load.jl")
include("ba-3d_scatter.jl")
include("ba-plots.jl")
include("ba-tables.jl")

# Plots for Objective historic, MSE and accuracy #

Random.seed!(seed)

# ---------------- Hyperbolic SVM Models ---------------- #

n_exec = 10
selected_probs = ["mnist"]
MaxEpochs = 0
MaxTime = 0.0

if selected_probs == ["ijcnn1"]
    sample_rate0 = .05
    sample_rates = [1.0, .1, .05, .01]
    selected_digits = [(1, 7)] # let only one pair of random digits
    versions = []#2, 5]
    #version = versions[end]
    ϵ = 1e-16
    selected_hs = ["l1", "lhalf"]#, "smooth"]
    MaxEpochs = 100
    MaxTime = 3600.0
    smooth = false
    compare = false
elseif selected_probs == ["mnist"]
    sample_rate0 = .05
    sample_rates = [1.0, .05]
    selected_digits = [(1, 7)]
    versions = []
    #version = versions[end]
    selected_hs = ["lhalf"]
    ϵ = 1e-4
    MaxEpochs = 1000
    MaxTime = 3600.0
    smooth = false
    compare = true
end

abscissas = ["epoch", "CPU time"]
abscissa = abscissas[1]

#plot_Sto_LM_SVM(sample_rates, versions, selected_probs, selected_hs, selected_digits; abscissa = abscissa, n_exec = n_exec, smooth = true, sample_rate0 = sample_rate0, param = param, compare = false, MaxEpochs = MaxEpochs, MaxTime = MaxTime)
plot_Sampled_LM_SVM_epoch(sample_rates, versions, selected_probs, selected_hs, selected_digits; abscissa = abscissa, n_exec = n_exec, smooth = smooth, sample_rate0 = sample_rate0, compare = compare, MaxEpochs = MaxEpochs, MaxTime = MaxTime, precision = ϵ)

# -- Plots for MNIST grey map -- #

Random.seed!(seed)
#=for digits in selected_digits
    demo_svm_sto(;sample_rate = sample_rate0, n_runs = n_exec, digits = digits, MaxEpochs = MaxEpochs, MaxTime = MaxTime, version = version, smooth = true)
end=#

#=if selected_probs == ["ijcnn1"]
    plot_ijcnn1(sample_rates, versions, selected_hs; n_runs = n_exec, MaxEpochs = MaxEpochs)
elseif selected_probs == ["mnist"]
    plot_mnist(sample_rates, versions, selected_hs; n_runs = n_exec, smooth = smooth)
    for version in versions
        greymaps_tables_mnist(version, sample_rates, sample_rate0; smooth = smooth)
    end
end=#

# ---------------- Bundle Adjustment Models ---------------- #

Random.seed!(seed)

n_exec = 2
sample_rates = []
versions = [1, 2, 3, 4, 5, 6]
version = versions[2]

df = problems_df()
filter_name = "dubrovnik"

filter_df = df[ df.group .== filter_name, :]
sample_rate = 1.0
#name1 = filter_df[1, :name]
name_list = ["problem-49-7776-pre", "problem-16-22106-pre", "problem-52-64053-pre", "problem-21-11315-pre", "problem-88-64298-pre", "problem-89-110973-pre"]
name_list = [filter_df[i, :name] for i in [1]]

selected_hs = ["l1"]
sample_rate0 = .05
plot_parameter = ["objective", "metric", "MSE", "accuracy"]
param = plot_parameter[1]

MaxEpochs = 0
MaxTime = 0.0
if abscissa == "epoch"
    MaxEpochs = 20
    MaxTime = 2e4
elseif abscissa == "CPU time"
    MaxEpochs = 1000
    MaxTime = 10.0
end

smooth = false
Jac_lop = false

#plot_Sto_LM_BA(sample_rates, versions, name_list, selected_hs; abscissa = abscissa, n_exec = n_exec, smooth = true, sample_rate0 = sample_rate0, compare = true, MaxEpochs = MaxEpochs, MaxTime = MaxTime)
Random.seed!(seed)
#demo_ba_sto(name_list; sample_rate = 1.0, sample_rate0 = sample_rate0, n_runs = n_exec, MaxEpochs = MaxEpochs, MaxTime = MaxTime, version = version, suffix = "$filter_name-l1", compare = false, smooth = smooth, Jac_lop = Jac_lop)

#=ba_3d_scatter(name_list; sample_rate = sample_rate, n_runs = n_exec)
for name in name_list
    plot_ba(name, sample_rate, version; n_runs = n_exec, smooth = smooth)
    ba_tables(name, sample_rate, version; smooth = smooth)
end=#
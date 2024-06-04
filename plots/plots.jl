include("plot-utils-svm-sto.jl")
include("demo_svm_sto.jl")
include("PLM-plots-svm.jl")
include("PLM-plots-ba.jl")

# Plots for Objective historic, MSE and accuracy #

Random.seed!(seed)
n_exec = 3

versions = [4]

# ---------------- Hyperbolic SVM Models ---------------- #

selected_probs = ["mnist"]
selected_digits = [(3, 8)]

if selected_probs == ["ijcnn1"]
    sample_rate0 = .05
    sample_rates = [.05]
elseif selected_probs == ["mnist"]
    sample_rate0 = .1
    sample_rates = []
end

selected_hs = ["l1/2"]
abscissas = ["epoch", "CPU time"]
abscissa = abscissas[1]

#plot_Sto_LM_SVM(sample_rates, versions, selected_probs, selected_hs, selected_digits; abscissa = abscissa, n_exec = n_exec, smooth = false, sample_rate0 = sample_rate0)

# -- Plots for MNIST map -- #

Random.seed!(seed)
#demo_svm_sto(;sample_rate = sample_rate0, n_runs = n_exec)

# ---------------- Bundle Adjustment Models ---------------- #

df = problems_df()
filter_df = df[ df.group .== "dubrovnik", :]
name1 = filter_df[1, :name]
name_list = [name1]
selected_hs = ["l1"]
sample_rate0 = .1

plot_Sto_LM_BA(sample_rates, versions, name_list, selected_hs; abscissa = abscissa, n_exec = n_exec, smooth = false, sample_rate0 = sample_rate0)
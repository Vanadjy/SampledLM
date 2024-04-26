# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using ProximalOperators, TSVD

# dependencies from us
using LinearOperators, NLPModels, NLPModelsModifiers, ShiftedProximalOperators, SolverCore

using LinearAlgebra: length
using Random, Test, OnlineStats, Distributions, Noise
using RegularizedProblems, RegularizedOptimization
using NLSProblems, NLPModels
using FastClosures
using Plots, LaTeXStrings
using MLDatasets

seed = 10

include("input_struct_sto.jl")
#include("input_struct_prob.jl")
include("Problems/Sto_LM_Problems.jl")
#include("OldCodes/LM_alg.jl")
include("Sto_LM_alg.jl")
include("Sto_LM_algv3.jl")
include("Sto_LM_guided_alg.jl")
include("Sto_LM_cp.jl")
include("Prob_LM_alg.jl")
include("utils.jl")
#=@testset "Stochastic tests" begin
    include("test/sto_tests.jl")
end=#
include("plots/plots.jl")

# Plots for Objective historic, MSE and accuracy #

Random.seed!(seed)
n_exec = 10
#sample_rates = [1.0, .2, .1, .05, .01]
#sample_rates = [1.0, .2, .1, .05]
sample_rates = [.05]
#selected_probs = ["ijcnn1", "mnist"]
selected_probs = ["mnist"]
#selected_hs = ["l0", "l1", "l1/2"]
selected_hs = ["l1"]
abscissas = ["epoch", "CPU time"]
abscissa = abscissas[1]
plot_Sto_LM(sample_rates, selected_probs, selected_hs; abscissa = abscissa, n_exec = n_exec)

# Plots for MNIST map #

Random.seed!(seed)
#demo_svm_sto(;sample_rate = .1)

#include("prob_tests.jl")
#=@testset "Probabilistic tests" begin
    include("prob_tests.jl")
end=#
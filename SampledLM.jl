# base dependencies
using LinearAlgebra, Logging, Printf, Plots, DataFrames

# external dependencies
using ProximalOperators, FastClosures

# dependencies from us
using LinearOperators, NLPModels, NLPModelsModifiers, ShiftedProximalOperators, SolverCore, QRMumps, Krylov, ADNLPModels, JSOSolvers
using RegularizedProblems, RegularizedOptimization, BundleAdjustmentModels

# stochastic packages dependencies
using Random, Test, Statistics, OnlineStats, Distributions, Noise
using MLDatasets:MNIST
using JSON, JLD2

seed = 1234

#tools and problems
include("utils.jl")
include("input_struct_sto.jl")
include("Problems/Sto_LM_Problems.jl")

#PLM-Cst variants
include("SLM_alg.jl")
include("SLM_sp_alg.jl")
include("SLM_smooth_alg.jl")

#PLM-ND and PLM-AD variants
include("PLM_alg.jl")
include("PLM_sp_alg.jl")
include("PLM_smooth_alg.jl")
include("PLM_sp_smooth_alg.jl")

#tests
#=@testset "Probabilistic LM tests" begin
    include("test/runtests.jl")
end=#
#include(raw"C:\Users\valen\Downloads\demo_svm.jl")

#plots
include("plots/plots.jl")
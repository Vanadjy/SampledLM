# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using ProximalOperators#, TSVD

# dependencies from us
using LinearOperators, NLPModels, NLPModelsModifiers, ShiftedProximalOperators, SolverCore, QRMumps, Krylov, ADNLPModels, JSOSolvers

using Random, Test, Statistics, OnlineStats, Distributions, Noise
using RegularizedProblems, RegularizedOptimization
using FastClosures
using Plots
using MLDatasets:MNIST
using LIBSVMdata
using DataFrames, BundleAdjustmentModels
using JSON
using JLD2

seed = 1234

#tools and problems
include("utils.jl")
include("input_struct_sto.jl")
include("Problems/Sto_LM_Problems.jl")

#PLM-Cst variants
include("Sto_LM_alg.jl")
include("sp_SLM_alg.jl")
include("Sto_LM_algv3.jl")
include("Sto_LM_guided_alg.jl")
include("Sto_LM_cp.jl")
include("smooth_SLM.jl")

#PLM-ND and PLM-AD variants
include("Prob_LM_alg.jl")
include("Prob_LM_sparse_alg.jl")
include("smooth_PLM.jl")
include("sp_smooth_PLM.jl")
include("Prob_LM_sparse_alg.jl")

#tests
#=@testset "Probabilistic LM tests" begin
    include("test/runtests.jl")
end=#
#include(raw"C:\Users\valen\Downloads\demo_svm.jl")

#plots
include("plots/plots.jl")
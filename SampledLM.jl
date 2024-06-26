# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using ProximalOperators, TSVD

# dependencies from us
using LinearOperators, NLPModels, NLPModelsModifiers, ShiftedProximalOperators, SolverCore, QRMumps, Krylov #JSOSolvers, 

using LinearAlgebra: length
using Random, Test, Statistics, OnlineStats, Distributions, Noise
using RegularizedProblems, RegularizedOptimization
using NLSProblems, NLPModels
using FastClosures
using Plots, LaTeXStrings
using MLDatasets
using DataFrames, BundleAdjustmentModels
using JSON

seed = 1234

include("utils.jl")
include("input_struct_sto.jl")
#include("input_struct_prob.jl")
include("Problems/Sto_LM_Problems.jl")
#include(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\Packages\LevenbergMarquardt.jl\src\LevenbergMarquardt.jl")
#include("OldCodes/LM_alg.jl")
include("Sto_LM_alg.jl")
include("Sto_LM_algv3.jl")
include("Sto_LM_guided_alg.jl")
include("Sto_LM_cp.jl")
include("Prob_LM_alg.jl")
include("smooth_PLM.jl")
include("Prob_LM_sparse_alg.jl")
#=@testset "Probabilistic LM tests" begin
    include("test/runtests.jl")
end=#
include("plots/plots.jl")
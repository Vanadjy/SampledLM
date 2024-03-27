# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using ProximalOperators, TSVD

# dependencies from us
using LinearOperators, NLPModels, NLPModelsModifiers, ShiftedProximalOperators, SolverCore

using LinearAlgebra: length
using Random, Test, OnlineStats, Distributions, Noise
using RegularizedProblems, RegularizedOptimization
using NLSProblems
using FastClosures
using Plots, LaTeXStrings
using MLDatasets

Random.seed!(10)

include("input_struct_sto.jl")
#include("input_struct_prob.jl")
include("Problems/Sto_LM_Problems.jl")
include("OldCodes/Sto_LM_alg.jl")
include("Sto_LM_algv3.jl")
include("Sto_LM_algv4.jl")
#include("Prob_LM.jl")
include("utils.jl")
@testset "Stochastic tests" begin
    include("sto_tests.jl")
end
#include("prob_tests.jl")
#=@testset "Probabilistic tests" begin
    include("prob_tests.jl")
end=#
# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using ProximalOperators, TSVD

# dependencies from us
using LinearOperators, NLPModels, NLPModelsModifiers, ShiftedProximalOperators, SolverCore

using LinearAlgebra: length
using Random, Test
using RegularizedProblems, RegularizedOptimization
using NLSProblems
using FastClosures

include("input_struct_sto.jl")
include("input_struct_prob.jl")
include("bpdn_model_sampled_sto.jl")
include("bpdn_model_sampled_prob.jl")
include("OldCodes/Sto_LM_alg.jl")
include("Sto_LM_algv3.jl")
include("Sto_LM_algv4.jl")
include("tests.jl")
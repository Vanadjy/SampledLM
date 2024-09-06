using LinearAlgebra, Test
using ADNLPModels, NLPModels, MLDatasets, QuadraticModels
using RegularizedProblems

Random.seed!(seed)

function test_well_defined(model, nls_model, sol)
    @test typeof(model) <: FirstOrderModel
    @test typeof(sol) == typeof(model.meta.x0)
    @test typeof(nls_model) <: FirstOrderNLSModel
    @test model.meta.nvar == nls_model.meta.nvar
    @test all(model.meta.x0 .== nls_model.meta.x0)
end

function test_objectives(model, nls_model, x = model.meta.x0)
    f = obj(model, x)
    F = residual(nls_model, x)
    @test f ≈ dot(F, F) / 2

    g = grad(model, x)
    JtF = jtprod_residual(nls_model, x, F)
    @test all(g .≈ JtF)

    JF = jprod_residual(nls_model, x, x)
    @test JF' * F ≈ JtF' * x
end

include("bpdn_tests.jl")
include("ijcnn1_tests.jl")
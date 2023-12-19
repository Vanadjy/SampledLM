compound = 1
sample_rate = 1.0
nz = 10 * compound
#options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-3, verbose = 10, spectral = true)
bpdn, bpdn_nls, sol = bpdn_model_sampled(compound)
bpdn2, bpdn_nls2, sol2 = bpdn_model_sampled2(compound)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10


#=for (mod, mod_name) ∈ ((x -> x, "exact"), (LSR1Model, "lsr1"), (LBFGSModel, "lbfgs"))
  for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (IndBallL0(10 * compound), "B0"))
    for solver_sym ∈ (:R2, :TR)
      solver_sym == :TR && mod_name == "exact" && continue
      solver_sym == :TR && h_name == "B0" && continue  # FIXME
      solver_name = string(solver_sym)
      solver = eval(solver_sym)
      @testset "bpdn-$(mod_name)-$(solver_name)-$(h_name)" begin
        x0 = zeros(bpdn.meta.nvar)
        p = randperm(bpdn.meta.nvar)[1:nz]
        x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
        args = solver_sym == :R2 ? () : (NormLinf(1.0),)
        out = solver(mod(bpdn), h, args..., options, x0 = x0)
        @test typeof(out.solution) == typeof(bpdn.meta.x0)
        @test length(out.solution) == bpdn.meta.nvar
        @test typeof(out.solver_specific[:Fhist]) == typeof(out.solution)
        @test typeof(out.solver_specific[:Hhist]) == typeof(out.solution)
        @test typeof(out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
        @test typeof(out.dual_feas) == eltype(out.solution)
        @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:Hhist])
        @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:SubsolverCounter])
        @test obj(bpdn, out.solution) == out.solver_specific[:Fhist][end]
        @test h(out.solution) == out.solver_specific[:Hhist][end]
        @test out.status == :first_order
      end
    end
  end
end

# TR with h = L1 and χ = L2 is a special case
for (mod, mod_name) ∈ ((LSR1Model, "lsr1"), (LBFGSModel, "lbfgs"))
  for (h, h_name) ∈ ((NormL1(λ), "l1"),)
    @testset "bpdn-$(mod_name)-TR-$(h_name)" begin
      x0 = zeros(bpdn.meta.nvar)
      p = randperm(bpdn.meta.nvar)[1:nz]
      x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
      TR_out = TR(mod(bpdn), h, NormL2(1.0), options, x0 = x0)
      @test typeof(TR_out.solution) == typeof(bpdn.meta.x0)
      @test length(TR_out.solution) == bpdn.meta.nvar
      @test typeof(TR_out.solver_specific[:Fhist]) == typeof(TR_out.solution)
      @test typeof(TR_out.solver_specific[:Hhist]) == typeof(TR_out.solution)
      @test typeof(TR_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
      @test typeof(TR_out.dual_feas) == eltype(TR_out.solution)
      @test length(TR_out.solver_specific[:Fhist]) == length(TR_out.solver_specific[:Hhist])
      @test length(TR_out.solver_specific[:Fhist]) ==
            length(TR_out.solver_specific[:SubsolverCounter])
      @test obj(bpdn, TR_out.solution) == TR_out.solver_specific[:Fhist][end]
      @test h(TR_out.solution) == TR_out.solver_specific[:Hhist][end]
      @test TR_out.status == :first_order
    end
  end
end

for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (IndBallL0(10 * compound), "B0"))
  for solver_sym ∈ (:LM, :LMTR)
    solver_name = string(solver_sym)
    solver = eval(solver_sym)
    solver_sym == :LMTR && h_name == "B0" && continue  # FIXME
    @testset "bpdn-ls-$(solver_name)-$(h_name)" begin
      x0 = zeros(bpdn_nls.meta.nvar)
      p = randperm(bpdn_nls.meta.nvar)[1:nz]
      x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
      args = solver_sym == :LM ? () : (NormLinf(1.0),)
      out = solver(bpdn_nls, h, args..., options, x0 = x0)
      @test typeof(out.solution) == typeof(bpdn_nls.meta.x0)
      @test length(out.solution) == bpdn_nls.meta.nvar
      @test typeof(out.solver_specific[:Fhist]) == typeof(out.solution)
      @test typeof(out.solver_specific[:Hhist]) == typeof(out.solution)
      @test typeof(out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
      @test typeof(out.dual_feas) == eltype(out.solution)
      @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:Hhist])
      @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:SubsolverCounter])
      @test length(out.solver_specific[:Fhist]) == length(out.solver_specific[:NLSGradHist])
      @test out.solver_specific[:NLSGradHist][end] ==
            bpdn_nls.counters.neval_jprod_residual + bpdn_nls.counters.neval_jtprod_residual - 1
      @test obj(bpdn_nls, out.solution) == out.solver_specific[:Fhist][end]
      @test h(out.solution) == out.solver_specific[:Hhist][end]
      @test out.status == :first_order
    end
  end
end

# LM with h = L1
for (h, h_name) ∈ ((NormL1(λ), "l1"), (NormL0(λ), "l0"),)
  @testset "bpdn-ls-LM-$(h_name)" begin
    x0 = zeros(bpdn_nls.meta.nvar)
    p = randperm(bpdn_nls.meta.nvar)[1:nz]
    x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
    LM_out = LM(bpdn_nls, h, options, x0 = x0)
    @test typeof(LM_out.solution) == typeof(bpdn_nls.meta.x0)
    @test length(LM_out.solution) == bpdn_nls.meta.nvar
    @test typeof(LM_out.solver_specific[:Fhist]) == typeof(LM_out.solution)
    @test typeof(LM_out.solver_specific[:Hhist]) == typeof(LM_out.solution)
    @test typeof(LM_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
    @test typeof(LM_out.dual_feas) == eltype(LM_out.solution)
    @test length(LM_out.solver_specific[:Fhist]) == length(LM_out.solver_specific[:Hhist])
    @test length(LM_out.solver_specific[:Fhist]) ==
          length(LM_out.solver_specific[:SubsolverCounter])
    #@test length(LMTR_out.solver_specific[:Fhist]) == length(LMTR_out.solver_specific[:NLSGradHist])
    #@test LMTR_out.solver_specific[:NLSGradHist][end] ==
          bpdn_nls.counters.neval_jprod_residual + bpdn_nls.counters.neval_jtprod_residual - 1
    @test obj(bpdn_nls, LM_out.solution) == LM_out.solver_specific[:Fhist][end]
    @test h(LM_out.solution) == LM_out.solver_specific[:Hhist][end]
    @test LM_out.status == :first_order
  end
end

# LMTR with h = L1 and χ = L2 is a special case # FIXME : doesn't work with h = L0
for (h, h_name) ∈ ((NormL1(λ), "l1"),)
  @testset "bpdn-ls-LMTR-$(h_name)" begin
    x0 = zeros(bpdn_nls.meta.nvar)
    p = randperm(bpdn_nls.meta.nvar)[1:nz]
    x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
    LMTR_out = LMTR(bpdn_nls, h, NormL2(1.0), options, x0 = x0)
    @test typeof(LMTR_out.solution) == typeof(bpdn_nls.meta.x0)
    @test length(LMTR_out.solution) == bpdn_nls.meta.nvar
    @test typeof(LMTR_out.solver_specific[:Fhist]) == typeof(LMTR_out.solution)
    @test typeof(LMTR_out.solver_specific[:Hhist]) == typeof(LMTR_out.solution)
    @test typeof(LMTR_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
    @test typeof(LMTR_out.dual_feas) == eltype(LMTR_out.solution)
    @test length(LMTR_out.solver_specific[:Fhist]) == length(LMTR_out.solver_specific[:Hhist])
    @test length(LMTR_out.solver_specific[:Fhist]) ==
          length(LMTR_out.solver_specific[:SubsolverCounter])
    #@test length(LMTR_out.solver_specific[:Fhist]) == length(LMTR_out.solver_specific[:NLSGradHist])
    #@test LMTR_out.solver_specific[:NLSGradHist][end] ==
          bpdn_nls.counters.neval_jprod_residual + bpdn_nls.counters.neval_jtprod_residual - 1
    @test obj(bpdn_nls, LMTR_out.solution) == LMTR_out.solver_specific[:Fhist][end]
    @test h(LMTR_out.solution) == LMTR_out.solver_specific[:Hhist][end]
    @test LMTR_out.status == :first_order
  end
end=#

# ------------ SAMPLED VERSIONS ------------------- #

# Sto_LM with h = L1 and χ = L2 is a special case
for (h, h_name) ∈ ((NormL1(λ), "l1"), (NormL0(λ), "l0"))
  reset!(bpdn_nls)
  @testset "bpdn-ls-Sto_LM-$(h_name)" begin
    x0 = zeros(bpdn_nls.meta.nvar)
    p = randperm(bpdn_nls.meta.nvar)[1:nz]
    x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
    SLM_out = Sto_LM(bpdn_nls, h, sampled_options; x0 = x0)
    @test typeof(SLM_out.solution) == typeof(bpdn_nls.meta.x0)
    @test length(SLM_out.solution) == bpdn_nls.meta.nvar
    @test typeof(SLM_out.solver_specific[:Fhist]) == typeof(SLM_out.solution)
    @test typeof(SLM_out.solver_specific[:Hhist]) == typeof(SLM_out.solution)
    @test typeof(SLM_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
    @test typeof(SLM_out.dual_feas) == eltype(SLM_out.solution)
    @test length(SLM_out.solver_specific[:Fhist]) == length(SLM_out.solver_specific[:Hhist])
    @test length(SLM_out.solver_specific[:Fhist]) ==
          length(SLM_out.solver_specific[:SubsolverCounter])
    @test length(SLM_out.solver_specific[:Fhist]) == length(SLM_out.solver_specific[:NLSGradHist])
    @test SLM_out.solver_specific[:NLSGradHist][end] ==
          bpdn_nls.counters.neval_jprod_residual + bpdn_nls.counters.neval_jtprod_residual - 1
    @test obj(bpdn_nls, SLM_out.solution) == SLM_out.solver_specific[:Fhist][end]
    @test h(SLM_out.solution) == SLM_out.solver_specific[:Hhist][end]
    @test SLM_out.status == :first_order
  end
end

#=for (h, h_name) ∈ (((NormL1(λ), "l1"), (NormL0(λ), "l0")))
  reset!(bpdn_nls2)
  @testset "bpdn-ls-Sto_LM_v2-$(h_name)" begin
    x0 = zeros(bpdn_nls2.meta.nvar)
    p = randperm(bpdn_nls2.meta.nvar)[1:nz]
    x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
    SLM2_out = Sto_LM_v2(bpdn_nls2, h, sampled_options; x0 = x0)
    @test typeof(SLM2_out.solution) == typeof(bpdn_nls2.meta.x0)
    @test length(SLM2_out.solution) == bpdn_nls2.meta.nvar
    @test typeof(SLM2_out.solver_specific[:Fhist]) == typeof(SLM2_out.solution)
    @test typeof(SLM2_out.solver_specific[:Hhist]) == typeof(SLM2_out.solution)
    @test typeof(SLM2_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
    @test typeof(SLM2_out.dual_feas) == eltype(SLM2_out.solution)
    @test length(SLM2_out.solver_specific[:Fhist]) == length(SLM2_out.solver_specific[:Hhist])
    @test length(SLM2_out.solver_specific[:Fhist]) ==
          length(SLM2_out.solver_specific[:SubsolverCounter])
    @test length(SLM2_out.solver_specific[:Fhist]) == length(SLM2_out.solver_specific[:NLSGradHist])
    @test SLM2_out.solver_specific[:NLSGradHist][end] ==
    bpdn_nls2.counters.neval_jprod_residual + bpdn_nls2.counters.neval_jtprod_residual - 1
    @test obj(bpdn_nls2, SLM2_out.solution) == SLM2_out.solver_specific[:Fhist][end]
    @test h(SLM2_out.solution) == SLM2_out.solver_specific[:Hhist][end]
    @test SLM2_out.status == :first_order
  end
end=#

for (h, h_name) ∈ ((NormL1(λ), "l1"), (NormL0(λ), "l0"),)
  reset!(bpdn_nls)
  @testset "bpdn-ls-Sto_LM_v3-$(h_name)" begin
    x0 = zeros(bpdn_nls.meta.nvar)
    p = randperm(bpdn_nls.meta.nvar)[1:nz]
    x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
    SLM3_out = Sto_LM_v3(bpdn_nls, h, sampled_options; x0 = x0, sample_rate = sample_rate)
    @test typeof(SLM3_out.solution) == typeof(bpdn_nls.meta.x0)
    @test length(SLM3_out.solution) == bpdn_nls.meta.nvar
    @test typeof(SLM3_out.solver_specific[:Fhist]) == typeof(SLM3_out.solution)
    @test typeof(SLM3_out.solver_specific[:Hhist]) == typeof(SLM3_out.solution)
    @test typeof(SLM3_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
    @test typeof(SLM3_out.dual_feas) == eltype(SLM3_out.solution)
    @test length(SLM3_out.solver_specific[:Fhist]) == length(SLM3_out.solver_specific[:Hhist])
    @test length(SLM3_out.solver_specific[:Fhist]) ==
          length(SLM3_out.solver_specific[:SubsolverCounter])
    @test length(SLM3_out.solver_specific[:Fhist]) == length(SLM3_out.solver_specific[:NLSGradHist])
    @test SLM3_out.solver_specific[:NLSGradHist][end] ==
    bpdn_nls.counters.neval_jprod_residual + bpdn_nls.counters.neval_jtprod_residual - 1
    @test obj(bpdn_nls, SLM3_out.solution) == SLM3_out.solver_specific[:Fhist][end]
    @test h(SLM3_out.solution) == SLM3_out.solver_specific[:Hhist][end]
    @test SLM3_out.status == :first_order
  end
end

#=for (h, h_name) ∈ ((NormL1(λ), "l1"), (NormL0(λ), "l0"),)
  reset!(bpdn_nls2)
  @testset "bpdn-ls-Sto_LM_v4-$(h_name)" begin
    x0 = zeros(bpdn_nls2.meta.nvar)
    p = randperm(bpdn_nls2.meta.nvar)[1:nz]
    x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
    SLM4_out = Sto_LM_v4(bpdn_nls2, h, sampled_options; x0 = x0, sample_rate = sample_rate)
    @test typeof(SLM4_out.solution) == typeof(bpdn_nls2.meta.x0)
    @test length(SLM4_out.solution) == bpdn_nls2.meta.nvar
    @test typeof(SLM4_out.solver_specific[:Fhist]) == typeof(SLM4_out.solution)
    @test typeof(SLM4_out.solver_specific[:Hhist]) == typeof(SLM4_out.solution)
    @test typeof(SLM4_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
    @test typeof(SLM4_out.dual_feas) == eltype(SLM4_out.solution)
    @test length(SLM4_out.solver_specific[:Fhist]) == length(SLM4_out.solver_specific[:Hhist])
    @test length(SLM4_out.solver_specific[:Fhist]) ==
          length(SLM4_out.solver_specific[:SubsolverCounter])
    @test length(SLM4_out.solver_specific[:Fhist]) == length(SLM4_out.solver_specific[:NLSGradHist])
    @test SLM4_out.solver_specific[:NLSGradHist][end] ==
    bpdn_nls2.counters.neval_jprod_residual + bpdn_nls2.counters.neval_jtprod_residual - 1
    @test obj(bpdn_nls2, SLM4_out.solution) == SLM4_out.solver_specific[:Fhist][end]
    @test h(SLM4_out.solution) == SLM4_out.solver_specific[:Hhist][end]
    @test SLM4_out.status == :first_order
  end
end=#
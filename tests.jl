compound = 1
sample_rates = [1.0, .9, .7, .5]
gr()
p = plot()

for sample_rate in sample_rates
  nz = 10 * compound
  #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
  sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, ϵa = 1e-3, ϵr = 1e-3, verbose = 10;)
  bpdn, bpdn_nls, sol = bpdn_model_sto(compound; sample_rate = sample_rate)
  #bpdn2, bpdn_nls2, sol2 = bpdn_model_prob(compound)
  λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

  #=for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (IndBallL0(10 * compound), "B0"))
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
  end=#

  # ------------ SAMPLED VERSIONS ------------------- #

  # Sto_LM with h = L1 and χ = L2 is a special case
  if sample_rate == 1.0
    for (h, h_name) ∈ ((NormL1(λ), "l1"), (NormL0(λ), "l0"))
      reset!(bpdn_nls)
      @testset "bpdn-ls-Sto_LM-$(h_name) - τ = $(sample_rate*100)%" begin
        x0 = zeros(bpdn_nls.meta.nvar)
        p = randperm(bpdn_nls.meta.nvar)[1:nz]
        x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
        SLM_out, Metric_hist = Sto_LM(bpdn_nls, h, sampled_options; x0 = x0)
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

        SLM_out.solver_specific[:Fhist] .+ SLM_out.solver_specific[:Hhist]
        plot!(1:SLM_out.iter, Metric_hist, label="LM for h = $h_name")
      end
    end
  end

  #=for (h, h_name) ∈ ((NormL1(λ), "l1"), (NormL0(λ), "l0"),)
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
  end=#

  for (h, h_name) ∈ ((NormL1(λ), "l1"), (NormL0(λ), "l0"),)
    reset!(bpdn_nls)
    @testset "bpdn-ls-Sto_LM_v4-$(h_name) - τ = $(sample_rate*100)%" begin
      x0 = zeros(bpdn_nls.meta.nvar)
      p = randperm(bpdn_nls.meta.nvar)[1:nz]
      x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
      SLM4_out, Metric_hist = Sto_LM_v4(bpdn_nls, h, sampled_options; x0 = x0)
      @test typeof(SLM4_out.solution) == typeof(bpdn_nls.meta.x0)
      @test length(SLM4_out.solution) == bpdn_nls.meta.nvar
      @test typeof(SLM4_out.solver_specific[:Fhist]) == typeof(SLM4_out.solution)
      @test typeof(SLM4_out.solver_specific[:Hhist]) == typeof(SLM4_out.solution)
      @test typeof(SLM4_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
      @test typeof(SLM4_out.dual_feas) == eltype(SLM4_out.solution)
      @test length(SLM4_out.solver_specific[:Fhist]) == length(SLM4_out.solver_specific[:Hhist])
      @test length(SLM4_out.solver_specific[:Fhist]) ==
            length(SLM4_out.solver_specific[:SubsolverCounter])
      @test length(SLM4_out.solver_specific[:Fhist]) == length(SLM4_out.solver_specific[:NLSGradHist])
      @test SLM4_out.solver_specific[:NLSGradHist][end] ==
      bpdn_nls.counters.neval_jprod_residual + bpdn_nls.counters.neval_jtprod_residual - 1
      #@test obj(bpdn_nls, SLM4_out.solution) == SLM4_out.solver_specific[:Fhist][end]
      @test h(SLM4_out.solution) == SLM4_out.solver_specific[:Hhist][end]
      @test SLM4_out.status == :first_order

      plot!(1:SLM4_out.iter, Metric_hist, label = "Sto_LM for h = $h_name and τ = $(sample_rate*100)%")
    end
  end
end
display(p)
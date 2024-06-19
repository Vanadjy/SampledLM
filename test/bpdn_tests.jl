compound = 1
nz = 10 * compound
sample_rate0 = .05
ϵ = 1e-5
options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = ϵ, ϵr = ϵ, verbose = 10, maxIter = 1000, maxTime = 3600.0;)
subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = 30)
bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate0)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

for smooth in [false, true]
  for version in [1, 2, 4]
    for (h, h_name) ∈ ((NormL0(λ), "l0"), (NormL1(λ), "l1"), (RootNormLhalf(λ), "l1/2"))
      @testset "bpdn-smooth:$smooth-v$version-$(h_name)" begin
        x0 = zeros(bpdn_nls.meta.nvar)
        p = randperm(bpdn.meta.nvar)[1:nz]
        x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
        reset!(bpdn_nls)
        bpdn_nls.epoch_counter = Int[1]
        PLM_out = Prob_LM(bpdn_nls, h, options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version, smooth = smooth)
        @test typeof(PLM_out.solution) == typeof(bpdn_nls.meta.x0)
        @test length(PLM_out.solution) == bpdn_nls.meta.nvar
        @test typeof(PLM_out.solver_specific[:Fhist]) == typeof(PLM_out.solution)
        @test typeof(PLM_out.solver_specific[:Hhist]) == typeof(PLM_out.solution)
        @test typeof(PLM_out.solver_specific[:SubsolverCounter]) == Array{Float64, 1}
        @test typeof(PLM_out.dual_feas) == eltype(PLM_out.solution)
        @test length(PLM_out.solver_specific[:Fhist]) == length(PLM_out.solver_specific[:Hhist])
        @test length(PLM_out.solver_specific[:Fhist]) == length(PLM_out.solver_specific[:SubsolverCounter])
        #@test obj(bpdn_nls, PLM_out.solution) ≈ PLM_out.solver_specific[:Fhist][end] ± ϵ
        #@test h(PLM_out.solution) ≈ PLM_out.solver_specific[:Hhist][end] ± ϵ
        @test PLM_out.status == :first_order
        @test bpdn_nls.sample_rate == length(bpdn_nls.sample) / bpdn_nls.nls_meta.nequ
      end
    end
  end
end
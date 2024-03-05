compound = 1
sample_rates = [1.0, .7, .5, .2, .05, .01]

#sample_rates = [.5]

plot_parameter = ["objective", "metric"]
param = plot_parameter[2]
Confidence = Dict([("95%", 1.96), ("99%", 2.58)])
conf = "95%"
gr()
graph = plot()

for sample_rate in sample_rates
  nz = 10 * compound
  #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
  sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, ϵa = 1e-3, ϵr = 1e-3, verbose = 10;)
  bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate)
  glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
  ijcnn1, ijcnn1_nls = ijcnn1_model_sto(sample_rate)
  lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)

  #bpdn2, bpdn_nls2, sol2 = bpdn_model_prob(compound)
  λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

  #nls_prob_collection = [(bpdn_nls, "bpdn-ls"), (glasso_nls, "group-lasso-ls"), (ijcnn1_nls, "ijcnn1-ls"), (lrcomp_nls, "lrcomp-ls")]
  nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]

  for (prob, prob_name) in nls_prob_collection
      for (h, h_name) ∈ ((NormL1(λ), "l1"),)
        reset!(prob)
        @testset "$prob_name-Sto_LM_v4-$(h_name) - τ = $(sample_rate*100)%" begin
          x0 = zeros(prob.meta.nvar)
          #p = randperm(prob.meta.nvar)[1:nz]
          #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
          SLM4_out, Metric_hist, exact_F_hist, exact_Metric_hist = Sto_LM_v4(prob, h, sampled_options; x0 = x0)
          @test typeof(SLM4_out.solution) == typeof(prob.meta.x0)
          @test length(SLM4_out.solution) == prob.meta.nvar
          @test typeof(SLM4_out.solver_specific[:Fhist]) == typeof(SLM4_out.solution)
          @test typeof(SLM4_out.solver_specific[:Hhist]) == typeof(SLM4_out.solution)
          @test typeof(SLM4_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
          @test typeof(SLM4_out.dual_feas) == eltype(SLM4_out.solution)
          @test length(SLM4_out.solver_specific[:Fhist]) == length(SLM4_out.solver_specific[:Hhist])
          @test length(SLM4_out.solver_specific[:Fhist]) ==
                length(SLM4_out.solver_specific[:SubsolverCounter])
          @test length(SLM4_out.solver_specific[:Fhist]) == length(SLM4_out.solver_specific[:NLSGradHist])
          @test SLM4_out.solver_specific[:NLSGradHist][end] ==
          prob.counters.neval_jprod_residual + prob.counters.neval_jtprod_residual - 1
          #@test obj(prob, SLM4_out.solution) == SLM4_out.solver_specific[:Fhist][end]
          @test h(SLM4_out.solution) == SLM4_out.solver_specific[:Hhist][end]
          @test SLM4_out.status == :max_iter

          SLM4_out.solver_specific[:Fhist] .+ SLM4_out.solver_specific[:Hhist]
          X = uniform_sample(SLM4_out.iter, sample_rate)

          #plot metric
          #for param in plot_parameter

            if param == "metric"
              sample_size = length(prob.sample)
              plot!(1:length(prob.epoch_counter), Metric_hist[prob.epoch_counter], xaxis=:log10, label = "Sto_LM - $(sample_rate*100)%", title = "Evolution of sampled √ξcp/νcp for $prob_name", ribbon=(fill(Confidence[conf] / sqrt(sample_size), length(prob.epoch_counter)), fill(Confidence[conf] / sqrt(sample_size), length(prob.epoch_counter))))

              #plot!(1:length(prob.epoch_counter), exact_Metric_hist[prob.epoch_counter], xaxis=:log10, label = "Sto_LM - $(sample_rate*100)%", title = "Evolution of exact √ξcp/νcp for $prob_name")
              #=plot!(X, Metric_hist[X], label = "τ = $(sample_rate*100)%")
              plot!(X, exact_Metric_hist[X], label = "τ = $(sample_rate*100)%")=#
              #scatter!(prob.opt_counter, Metric_hist[prob.opt_counter], mc=:green, ms=2, ma=0.5, label = "Hits below tolerance : $(length(prob.opt_counter))")
              xlabel!("epoch")
              ylabel!("√ξcp/νcp")
            end

            #plot f+h
            if param == "objective"
              sample_size = length(prob.sample)
              plot!(1:length(prob.epoch_counter), SLM4_out.solver_specific[:Fhist][prob.epoch_counter] + SLM4_out.solver_specific[:Hhist][prob.epoch_counter], xaxis=:log10, yaxis=:log10, label = "Sto_LM - $(sample_rate*100)%", title = "Evolution of sampled f + h for $prob_name", ribbon=(fill(Confidence[conf] / sqrt(sample_size), length(prob.epoch_counter)), fill(Confidence[conf] / sqrt(sample_size), length(prob.epoch_counter))))

              #plot!(1:length(prob.epoch_counter), exact_F_hist[prob.epoch_counter] + SLM4_out.solver_specific[:Hhist][prob.epoch_counter], xaxis=:log10, yaxis=:log10, label = "Sto_LM - $(sample_rate*100)%", title = "Evolution of exact f + h for $prob_name")
              #=plot!(X, SLM4_out.solver_specific[:Fhist][X], label = "τ = $(sample_rate*100)%")
              plot!(X, exact_F_hist[X], label = "τ = $(sample_rate*100)%")=#
              #scatter!(prob.opt_counter, SLM4_out.solver_specific[:Fhist][prob.opt_counter] + SLM4_out.solver_specific[:Hhist][prob.opt_counter], mc=:green, ms=2, ma=0.5, label = "Hit below tolerance : : $(length(prob.opt_counter))")
              xlabel!("epoch")
              ylabel!("f + h")
            end
          #end
        end
      #push!(mosaique, graph)
      end
  end
end

#mosaique = plot(p1, p2, layout = (2, 1), legend = true)
display(graph)

#p = plot(mosaique[1], mosaique[2], mosaique[3], mosaique[4], mosaique[5], mosaique[6], layout = (2,3), legend = true)
#p = plot(mosaique[1])
#display(p)
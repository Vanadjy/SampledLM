compound = 1
#sample_rates = [.2, .1, .05, .01]
sample_rates = [1.0, .2, .1, .05]

#sample_rates = [.2]

plot_parameter = ["objective", "metric"]
param = plot_parameter[1]

abscissas = ["epoch", "CPU time"]
abscissa = abscissas[1]
sampled_res = true
Confidence = Dict([("95%", 1.96), ("99%", 2.58)])
conf = "95%"
n_exec = 5

MaxEpochs = 0
MaxTime = 0.0
if abscissa == "epoch"
  MaxEpochs = 20
  MaxTime = 3600.0
elseif abscissa == "CPU time"
  MaxEpochs = 1000
  MaxTime = 10.0
end

gr()
graph = plot()

for sample_rate in sample_rates
  nz = 10 * compound
  #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
  sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, ϵa = 1e-3, ϵr = 1e-3, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
  bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate)
  #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
  #ijcnn1, ijcnn1_nls = ijcnn1_model_sto(sample_rate)
  #a9a, a9a_nls = a9a_model_sto(sample_rate)
  mnist, mnist_nls = MNIST_train_model_sto(sample_rate)
  #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)

  #bpdn2, bpdn_nls2, sol2 = bpdn_model_prob(compound)
  λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

  #nls_prob_collection = [(bpdn_nls, "bpdn-ls"), (glasso_nls, "group-lasso-ls"), (ijcnn1_nls, "ijcnn1-ls"), (lrcomp_nls, "lrcomp-ls")]
  #nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
  #nls_prob_collection = [(a9a_nls, "a9a-ls")]
  nls_prob_collection = [(mnist_nls, "mnist-train-ls")]

  Obj_Hists_epochs = zeros(1 + MaxEpochs, n_exec)
  Metr_Hists_epochs = similar(Obj_Hists_epochs)
  Time_Hists = []
  Obj_Hists_time = []


  for (prob, prob_name) in nls_prob_collection
    for (h, h_name) ∈ ((NormL1(λ), "l1"),)
      for k in 1:n_exec
        # executes n_exec times Sto_LM with the same inputs
        reset!(prob)
        @testset "$prob_name-Sto_LM_v4-$(h_name) - τ = $(sample_rate*100)%" begin
          x0 = zeros(prob.meta.nvar)
          #p = randperm(prob.meta.nvar)[1:nz]
          #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
          SLM4_out, Metric_hist, exact_F_hist, exact_Metric_hist, TimeHist = Sto_LM_v4(prob, h, sampled_options; x0 = x0)

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

          #= #plot metric
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
            end=#
          push!(Time_Hists, TimeHist)
          if abscissa == "epoch"
            Obj_Hists_epochs[:, k] = exact_F_hist[prob.epoch_counter]
            Obj_Hists_epochs[:, k] += SLM4_out.solver_specific[:Hhist][prob.epoch_counter]
            Metr_Hists_epochs[:, k] = exact_Metric_hist[prob.epoch_counter]
          elseif abscissa == "CPU time"
            push!(Obj_Hists_time, exact_F_hist + SLM4_out.solver_specific[:Hhist])
          end
        end
        if k < n_exec
          prob.epoch_counter = Int[1]
        end
      end

      if abscissa == "epoch"
        sample_size = length(prob.sample)
        med_obj = zeros(axes(Obj_Hists_epochs, 1))
        std_obj = similar(med_obj)
        med_metric = zeros(axes(Metr_Hists_epochs, 1))
        std_metric = similar(med_metric)
        for l in 1:length(med_obj)
          #filter zero values if some executions fail
          med_obj[l] = median(filter(!iszero, Obj_Hists_epochs[l, :]))
          std_obj[l] = std(filter(!iszero, Obj_Hists_epochs[l, :]))
          med_metric[l] = median(filter(!iszero, Metr_Hists_epochs[l, :]))
          std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))
        end
        std_obj *= Confidence[conf] / sqrt(sample_size)
        std_metric *= Confidence[conf] / sqrt(sample_size)

        if param == "objective"
          plot!(axes(Obj_Hists_epochs, 1), med_obj, label = "Sto_LM - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec executions", xaxis=:log10, yaxis=:log10, ribbon=(std_obj, std_obj))
        elseif param == "metric"
          plot!(axes(Metr_Hists_epochs, 1), med_metric, label = "Sto_LM - $(sample_rate*100)%", title = "Sampled √ξcp/νcp for $prob_name on $n_exec executions", xaxis=:log10, ribbon=(std_metric, std_metric))
        end
        
      elseif abscissa == "CPU time"
        local t = maximum(length.(Time_Hists))
        local m = maximum(length.(Obj_Hists_time))
        Obj_Mat_time = zeros(m, n_exec)
        Time_mat = zeros(t, n_exec)
        for i in 1:n_exec
          Obj_Mat_time[:, i] .= vcat(Obj_Hists_time[i], zeros(m - length(Obj_Hists_time[i])))
          Time_mat[:, i] .= vcat(Time_Hists[i], zeros(m - length(Time_Hists[i])))
        end

        sample_size = length(prob.sample)
        med_obj = zeros(axes(Obj_Mat_time, 1))
        std_obj = similar(med_obj)
        #med_metric = zeros(axes(Metr_Hists_epochs, 1))
        #std_metric = similar(med_metric)
        med_time = zeros(axes(Time_mat, 1))

        for l in 1:length(med_obj)
          #filter zero values if some executions fail
          data = filter(!iszero, Obj_Mat_time[l, :])
          med_obj[l] = median(data)
          if length(data) > 1
            std_obj[l] = std(data)
          end
          #med_metric[l] = median(filter(!iszero, Metr_Hists_epochs[l, :]))
          #std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))
          med_time[l] = median(filter(!iszero, Time_mat[l, :]))
        end
        #std_obj *= Confidence[conf] / sqrt(sample_size)
        #std_metric *= Confidence[conf] / sqrt(sample_size)

        if param == "objective"
          plot!(sort(med_time), med_obj, label = "Sto_LM - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec executions", xaxis=:log10, yaxis=:log10, ribbon=(std_obj, std_obj))
        elseif param == "metric"
          plot!(axes(Metr_Hists, 1), med_metric, label = "Sto_LM - $(sample_rate*100)%", title = "Sampled √ξcp/νcp for $prob_name on $n_exec executions", xaxis=:log10, ribbon=(std_metric, std_metric))
        end
      end
      #push!(mosaique, graph)
    end
  end
end

#=bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound)
ijcnn1_full, ijcnn1_nls_full = ijcnn1_model_sto(1.0)
sampled_options_full = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10
h = NormL1(λ)
x0 = zeros(ijcnn1_full.meta.nvar)
l_bound = ijcnn1_full.meta.lvar
u_bound = ijcnn1_full.meta.uvar

xk_R2, k_R2, R2_out = R2(ijcnn1_full.f, ijcnn1_full.∇f!, h, sampled_options_full, x0)
R2_out[:Fhist] += R2_out[:Hhist]
plot!(1:k_R2, R2_out[:Fhist], label = "R2", xaxis=:log10, yaxis=:log10)=#

xlabel!(abscissa)
if param == "objective"
  ylabel!("f + h")
else
  ylabel!("√ξcp/νcp")
end
#mosaique = plot(p1, p2, layout = (2, 1), legend = true)
display(graph)
#export SPLM

"""
    SPLM(nls, options, version; kwargs...)

A Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖²

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous. This method uses different variable 
ample rate schemes each corresponding to a different number.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖²

where F(x) and J(x) are the residual and its Jacobian at x, respectively
and σ > 0 is a regularization parameter.

Both the objective and the model are estimations as F ad J are sampled.

### Arguments

* `nls::AbstractNLSModel`: a smooth nonlinear least-squares problem
* `options::ROSolverOptions`: a structure containing algorithmic parameters
* `version::Int`: integer specifying the sampling strategy

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nls.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver
* `subsolver`: the procedure used to compute a step (`PG` or `R2`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver.
* `selected::AbstractVector{<:Integer}`: list of selected indexes for the sampling
* `sample_rate0::Float64`: first sample rate used for the method

### Return values
Generic solver statistics including among others

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function SPLM(
  nls::SampledNLSModel,
  options::ROSolverOptions,
  version::Int;
  x0::AbstractVector = nls.meta.x0,
  subsolver_options = RegularizedOptimization.ROSolverOptions(ϵa = options.ϵa),
  sample_rate0::Float64 = .05
)

  # initializes epoch counting and progression
  epoch_count = 0
  epoch_progress = 0

  # initializes values for adaptive sample rate strategy
  Num_mean = 0
  mobile_mean = 0
  unchange_mm_count = 0
  sample_rates_collec = [.2, .5, .9, 1.0]
  epoch_limits = [1, 2, 5, 10]
  @assert length(sample_rates_collec) == length(epoch_limits)
  nls.sample_rate = sample_rate0
  ζk = Int(ceil(nls.sample_rate * nls.nls_meta.nequ))
  nls.sample = sort(randperm(nls.nls_meta.nequ)[1:Int(ceil(nls.sample_rate * nls.nls_meta.nequ))])
  sample_mem = copy(nls.sample)

  sample_counter = 1
  change_sample_rate = false

  # initialize time stats
  start_time = time()
  elapsed_time = 0.0

  # initialize passed options
  ϵ = options.ϵa
  ϵ_subsolver = subsolver_options.ϵa
  ϵr = options.ϵr
  verbose = options.verbose
  maxIter = options.maxIter
  maxEpoch = maxIter
  maxIter = Int(ceil(maxIter * (nls.nls_meta.nequ / length(nls.sample)))) #computing the sample rate
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  η3 = options.η3
  β = options.β
  θ = options.θ
  λ = options.λ
  neg_tol = options.neg_tol
  νcp = options.νcp
  σmin = options.σmin
  σmax = options.σmax
  μmin = options.μmin
  metric = options.metric
  ξ0 = options.ξ0
  balance = 10^(ceil(log10(nls.nls_meta.nequ / nls.meta.nvar)))
  threshold_relax = max((nls.nls_meta.nequ / (10^(floor(log10(nls.nls_meta.nequ / nls.meta.nvar))) * nls.meta.nvar)), 1.0) # < 1 if more equations than variables

  m = nls.nls_meta.nequ
  ζk = Int(ceil(balance))

  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  # initialize parameters
  σk = max(1 / options.ν, σmin)
  μk = max(1 / options.ν , μmin)
  xk = copy(x0)
  xkn = similar(xk)
  
  local ξ
  local ξ_mem

  count_fail = 0
  count_big_succ = 0
  count_succ = 0
  δ_sample = .05
  buffer = .05
  dist_succ = zero(eltype(xk))
  k = 0

  Fobj_hist = zeros(maxIter * 100)
  exact_Fobj_hist = zeros(maxIter * 100)
  Metric_hist = zeros(maxIter * 100)
  exact_Metric_hist = zeros(maxIter * 100)
  Complex_hist = zeros(maxIter * 100)
  Grad_hist = zeros(maxIter * 100)
  Resid_hist = zeros(maxIter * 100)
  Sample_hist = zeros(maxIter * 100)

  #Historic of time
  TimeHist = []

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %6s %7s %7s %8s %7s %7s %7s %7s %1s %6s" "outer" "inner" "f(x)" "‖∇f(x)‖" "ρ" "σ" "μ" "‖x‖" "‖s‖" "reg" "rate"
    #! format: on
  end

  #creating required objects
  Fk = residual(nls, xk)
  Fkn = similar(Fk)
  exact_Fk = zeros(1:m)

  fk = dot(Fk, Fk) / 2 #objective estimated without noise
  Jk = jac_op_residual(nls, xk)
  
  #sampled Jacobian
  ∇fk = similar(xk)
  exact_∇fk = similar(∇fk)
  JdFk = similar(Fk) # temporary storage
  jtprod_residual!(nls, xk, Fk, ∇fk)
  μmax = opnorm(Jk)

  s = zero(xk)

  optimal = false
  tired = epoch_count ≥ maxEpoch || elapsed_time > maxTime
  #tired = elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Grad_hist[k] = nls.counters.neval_jtprod_residual + nls.counters.neval_jprod_residual
    Resid_hist[k] = nls.counters.neval_residual
    Sample_hist[k] = nls.sample_rate
    if k == 1
      push!(TimeHist, 0.0)
    else
      push!(TimeHist, elapsed_time)
    end

    metric = norm(∇fk)
    Metric_hist[k] = metric

    if k == 1
      ϵ_increment = (ξ0 ≤ 10 * eps(eltype(xk))) ? ϵr * metric : ϵr * ξ0
      ϵ += ϵ_increment # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
      μk = 1 / metric
    end
    
    if version == 0 #including specific stopping criterion for constant sample rate strategies
      if (metric < ϵ) 
        push!(nls.opt_counter, k) #indicates the iteration where the tolerance has been reached by the metric
        if nls.sample_rate == 1.0
          optimal = true
        else
          if (length(nls.opt_counter) ≥ 5) && (nls.opt_counter[end-2:end] == range(k-2, k)) #if the last 3 iterations are successful
            optimal = true
          end
        end
      end
    else
      if (metric < ϵ) && nls.sample_rate == 1.0
        optimal = true
      end
    end

    subsolver_options.ϵa = min(1.0e-1, ϵ + ϵr*metric^(1.3))

    #update of σk
    σk = min(max(μk * metric, σmin), σmax)

    # TODO: reuse residual computation
    # model for subsequent prox-gradient iterations

    mk_smooth(d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2 #+ σk * dot(d, d) / 2
    end

    # LSMR strategy for LinearOperators #
    s, stats = lsmr(Jk, -Fk; λ = sqrt(σk), atol = subsolver_options.ϵa)#, rtol = ϵr)
    Complex_hist[k] = stats.niter

    xkn .= xk .+ s
    Fkn = residual(nls, xkn)
    fkn = dot(Fkn, Fkn) / 2
    mks = mk_smooth(s)
    Δobj = fk - fkn
    ξ = fk - mks
    (ξ < 0 && -ξ > neg_tol) &&
      error("PLM: lsmr step should produce a decrease but ξ = $(ξ)")
    ξ = (ξ < 0 && -ξ ≤ neg_tol) ? -ξ : ξ

    ρk = Δobj / ξ

    #μ_stat = ((η1 ≤ ρk < Inf) && ((metric ≥ η3 / μk))) ? "↘" : "↗"
    μ_stat = ρk < η1 ? "↘" : ((nls.sample_rate==1.0 && (metric > η2))||(nls.sample_rate<1.0 && (metric ≥ η3 / μk)) ? "↗" : "=")
    #μ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %6d %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s %6.2e" k stats.niter fk norm(∇fk) ρk σk μk norm(xk) norm(s) μ_stat nls.sample_rate
      #! format: off
    end
    
    #-- to compute exact quantities --#
    if nls.sample_rate < 1.0
      nls.sample = 1:m
      residual!(nls, xk, exact_Fk)
      exact_fk = dot(exact_Fk, exact_Fk) / 2
      jtprod_residual!(nls, xk, exact_Fk, exact_∇fk)
      exact_metric = norm(exact_∇fk)

      exact_Fobj_hist[k] = exact_fk
      exact_Metric_hist[k] = exact_metric
    elseif nls.sample_rate == 1.0
      exact_Fobj_hist[k] = fk
      exact_Metric_hist[k] = metric
    end
    # -- -- #

    #updating the indexes of the sampling
    epoch_progress += nls.sample_rate
    if epoch_progress >= 1 #we passed on all the data
      epoch_count += 1
      push!(nls.epoch_counter, k)
      epoch_progress -= 1
    end

    # Version 1: List of predetermined - switch with mobile average #
    if version == 1
      # Change sample rate
      #nls.sample_rate = basic_change_sample_rate(epoch_count)
      if nls.sample_rate < sample_rates_collec[end]
        Num_mean = Int(ceil(1 / nls.sample_rate))
        if k >= Num_mean
          @views mobile_mean = mean(Fobj_hist[(k - Num_mean + 1):k])
          if abs(mobile_mean - fk) ≤ 1e-1 #if the mean on the Num_mean last iterations is near the current objective value
            nls.sample_rate = sample_rates_collec[sample_counter]
            sample_counter += 1
            change_sample_rate = true
          end
        end
      end
    end

    # Version 2: List of predetermined - switch with arbitrary epochs #
    if version == 2
      if nls.sample_rate < sample_rates_collec[end]
        if epoch_count > epoch_limits[sample_counter]
          nls.sample_rate = sample_rates_collec[sample_counter]
          sample_counter += 1
          change_sample_rate = true
        end
      end
    end

    # Version 3: Adapt sample_size after each iteration #
    if version == 3
      # ζk = Int(ceil(k / (1e8 * min(1, 1 / μk^4))))
      p = .75
      q = .75
      ζk = Int(ceil(100 * (log(1 / (1-p)) * max(μk^4, μk^2) + log(1 / (1-q)) * μk^4)))
      nls.sample_rate = min(1.0, (ζk / nls.nls_meta.nequ) * (nls.meta.nvar + 1))
      change_sample_rate = true
    end

    # Version 4: Double sample_size after a fixed number of epochs or a mobile mean stagnation #
    if version == 4
      # Change sample rate
      #nls.sample_rate = basic_change_sample_rate(epoch_count)
      if nls.sample_rate < 1.0
        Num_mean = Int(ceil(1 / nls.sample_rate))
        if k >= Num_mean
          @views mobile_mean = mean(Fobj_hist[(k - Num_mean + 1):k])
          if abs(mobile_mean - fk) ≤ 1e-1 #if the mean on the Num_mean last iterations is near the current objective value
            nls.sample_rate = min(1.0, 2 * nls.sample_rate)
            change_sample_rate = true
            unchange_mm_count = 0
          else # don't have stagnation
            unchange_mm_count += nls.sample_rate
            if unchange_mm_count ≥ 3 # force to change sample rate after 3 epochs of unchanged sample rate using mobile mean criterion
              nls.sample_rate = min(1.0, 2 * nls.sample_rate)
              change_sample_rate = true
              unchange_mm_count = 0
            end
          end
        end
      end
    end

    # Version 5: change sample rate when gain factor 10 accuracy #
    if version == 5
      if k == 1
        ξ_mem = Metric_hist[1]
      end
      if nls.sample_rate < sample_rates_collec[end]
        if metric/ξ_mem ≤ 1e-1 #if the current metric is a factor 10 lower than the previously stored ξ_mem
          nls.sample_rate = sample_rates_collec[sample_counter]
          sample_counter += 1
          ξ_mem *= 1e-1
          change_sample_rate = true
        end
      end
    end

    # Version 6: Double sample_size after a fixed number of epochs or a metric decrease #
    if version == 6
      if k == 1
        ξ_mem = Metric_hist[1]
      end
      # Change sample rate
      #nls.sample_rate = basic_change_sample_rate(epoch_count)
      if nls.sample_rate < 1.0
        if metric/ξ_mem ≤ 1e-1 #if the current metric is a factor 10 lower than the previously stored ξ_mem
          nls.sample_rate = sample_rates_collec[sample_counter]
          sample_counter += 1
          ξ_mem *= 1e-1
          change_sample_rate = true
          unchange_mm_count = 0
        else # don't get more accurate ξ
          unchange_mm_count += nls.sample_rate
          if unchange_mm_count ≥ 3 # force to change sample rate after 3 epochs of unchanged sample rate using mobile mean criterion
            nls.sample_rate = sample_rates_collec[sample_counter]
            sample_counter += 1
            change_sample_rate = true
            unchange_mm_count = 0
          end
        end
      end
    end

    if version == 7
      if (count_fail == 2) && nls.sample_rate != sample_rate0 # if μk increased 3 times in a row -> decrease the batch size AND useless to try to make nls.sample rate decrease if its already equal to sample_rate0
        sample_counter = max(0, sample_counter - 1) # sample_counter-1 < length(sample_rates_collec)
        nls.sample_rate = (sample_counter == 0) ? sample_rate0 : sample_rates_collec[sample_counter]
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
      elseif (count_big_succ == 2) && nls.sample_rate != sample_rates_collec[end] # if μk decreased 3 times in a row -> increase the batch size AND useless to try to make nls.sample rate increase if its already equal to the highest available sample rate
        sample_counter = min(length(sample_rates_collec), sample_counter + 1) # sample_counter + 1 > 0
        nls.sample_rate = sample_rates_collec[sample_counter]
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
      end
    end

    if version == 8
      if (count_fail == 3) && nls.sample_rate != sample_rate0 # if μk increased 3 times in a row -> decrease the batch size AND useless to try to make nls.sample rate decrease if its already equal to sample_rate0
        nls.sample_rate -= δ_sample
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
      elseif (count_big_succ == 3) && nls.sample_rate != sample_rates_collec[end] # if μk decreased 3 times in a row -> increase the batch size AND useless to try to make nls.sample rate increase if its already equal to the highest available sample rate
        nls.sample_rate += δ_sample
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
      end
    end

    if (version == 9)
      if nls.sample_rate < 1.0
        if (count_fail == 2) && nls.sample_rate != sample_rates_collec[end] # if μk increased twice in a row -> decrease the batch size AND useless to try to make nls.sample rate decrease if its already equal to sample_rate0
          #=ζk *= λ^4
          @info "possible sample rate = $((ζk / nls.nls_meta.nequ) * (nls.meta.nvar + 1))"
          nls.sample_rate = min(1.0, max((ζk / nls.nls_meta.nequ) * (nls.meta.nvar + 1), buffer))=#
          nls.sample_rate = min(1.0, max(nls.sample_rate * λ, buffer))
          change_sample_rate = true
          count_fail = 0
          count_big_succ = 0
          count_succ = 0
          dist_succ = zero(eltype(xk))
        elseif (count_big_succ == 2) && nls.sample_rate != sample_rate0 # if μk decreased twice in a row -> increase the batch size AND useless to try to make nls.sample rate increase if its already equal to the highest available sample rate
          #ζk *= λ^(-4)
          #@info "possible sample rate = $((ζk / nls.nls_meta.nequ) * (nls.meta.nvar + 1))"
          nls.sample_rate = min(1.0, max(nls.sample_rate / λ, buffer))
          change_sample_rate = true
          count_fail = 0
          count_big_succ = 0
          count_succ = 0
          dist_succ = zero(eltype(xk))
        end
        if (nls.sample_rate < sample_rates_collec[end]) && ((dist_succ > (norm(ones(nls.meta.nvar)) / (threshold_relax * nls.sample_rate))) || (count_succ > 10)) # if μ did not change for too long, increase the buffer value
          @info "sample rate buffered at $(sample_rates_collec[sample_counter] * 100)%"
          buffer = sample_rates_collec[sample_counter]
          nls.sample_rate = min(1.0, max(nls.sample_rate, buffer))
          sample_counter += 1
          change_sample_rate = true
          count_succ = 0
          dist_succ = zero(eltype(xk))
        end
      end
    end

    if version == 0 #never change sample rate -> constant sample rate strategy

    end

    if change_sample_rate
      # mandatory updates whenever the sample_rate chages #
      nls.sample = sort(randperm(nls.nls_meta.nequ)[1:Int(ceil(nls.sample_rate * nls.nls_meta.nequ))])
      sample_mem = copy(nls.sample)

      Fk = residual(nls, xk)
      Fkn = similar(Fk)
      JdFk = similar(Fk)
      fk = dot(Fk, Fk) / 2

      Jk = jac_op_residual(nls, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)
      μmax = opnorm(Jk)
    end

    if (η1 ≤ ρk < Inf) #&& (metric ≥ η3 / μk) #successful step
      xk .= xkn
      #changes sample with new sample rate
      nls.sample = sort(randperm(nls.nls_meta.nequ)[1:Int(ceil(nls.sample_rate * nls.nls_meta.nequ))])
      sample_mem .= nls.sample
      if nls.sample_rate == 1.0
        nls.sample == 1:nls.nls_meta.nequ || error("Sample Error : Sample should be full for 100% sampling")
      end

      if (nls.sample_rate < 1.0) && metric ≥ η3 / μk #very successful step
        μk = max(μk / λ, μmin)
        count_big_succ += 1
        count_fail = 0
        count_succ = 0
        dist_succ = zero(eltype(xk))
      elseif (nls.sample_rate == 1.0) && (η2 ≤ ρk < Inf)
        μk = max(μk / λ, μmin)
        count_big_succ += 1
        count_fail = 0
        count_succ = 0
        dist_succ = zero(eltype(xk))
      else
        dist_succ += norm(s)
        count_succ += 1
      end

      if nls.sample_rate == 1.0
        Fk .= Fkn
        fk = fkn
        # update gradient & Hessian
        Jk = jac_op_residual(nls, xk)
        jtprod_residual!(nls, xk, Fk, ∇fk)
  
        μmax = opnorm(Jk)
      end

      if (nls.sample_rate < 1.0) || (nls.sample_rate == 1.0 && change_sample_rate)
        Fk = residual(nls, xk)
        Fkn = similar(Fk)
        JdFk = similar(Fk)
        fk = dot(Fk, Fk) / 2
  
        # update jacobian info
        Jk = jac_op_residual(nls, xk)
        jtprod_residual!(nls, xk, Fk, ∇fk)
        μmax = opnorm(Jk)
      end

      Complex_hist[k] += 1

    else # (ρk < η1 || ρk == Inf) #|| (metric < η3 / μk) #unsuccessful step
      if !change_sample_rate #if sample rate changed, sample changes whatever
        nls.sample = sample_mem
      end
      μk = max(λ * μk, μmin)
      count_big_succ = 0
      count_fail += 1
      count_succ = 0
      dist_succ = zero(eltype(xk))
    end

    if change_sample_rate
      change_sample_rate = false
    end

    tired = epoch_count ≥ maxEpoch || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e" k "" fk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8s %8.1e %7.4e %8s %7.1e %7.1e %7.1e %7.1e" k "" fk norm(∇fk) "" σk μk norm(xk) norm(s)
      #! format: on
      @info "SLM: terminating with ‖∇f(x)‖= $(norm(∇fk))"
    end
  end
  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end

  stats = GenericExecutionStats(nls)
  set_status!(stats, status)
  set_solution!(stats, xk)
  set_objective!(stats, fk)
  set_residuals!(stats, zero(eltype(xk)), norm(∇fk))
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:k])
  set_solver_specific!(stats, :ExactFhist, exact_Fobj_hist[1:k])
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k])
  set_solver_specific!(stats, :NLSGradHist, Grad_hist[1:k])
  set_solver_specific!(stats, :ResidHist, Resid_hist[1:k])
  set_solver_specific!(stats, :MetricHist, Metric_hist[1:k])
  set_solver_specific!(stats, :ExactMetricHist, exact_Metric_hist[1:k])
  set_solver_specific!(stats, :TimeHist, TimeHist)
  set_solver_specific!(stats, :SampleRateHist, Sample_hist[1:k])
  return stats
end
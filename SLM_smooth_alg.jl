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
  options::ROSolverOptions;
  x0::AbstractVector = nls.meta.x0,
  subsolver_options = RegularizedOptimization.ROSolverOptions(ϵa = options.ϵa),
)

@info "using smooth variant of PLM with constant sample rate"
  # initialize time stats
  start_time = time()
  elapsed_time = 0.0
  epoch_count = 0

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
  θ = options.θ
  λ = options.λ
  σmin = options.σmin
  σmax = options.σmax
  μmin = options.μmin
  metric = options.metric
  ξ0 = options.ξ0
  m = nls.nls_meta.nequ

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
  k = 0

  Fobj_hist = zeros(maxIter)
  exact_Fobj_hist = zeros(maxIter)
  Metric_hist = zeros(maxIter)
  exact_Metric_hist = zeros(maxIter)
  Complex_hist = zeros(maxIter)
  Grad_hist = zeros(maxIter)
  Resid_hist = zeros(maxIter)
  Sample_hist = zeros(maxIter)

  #Historic of time
  TimeHist = []

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %7s %7s %8s %7s %7s %7s %7s %1s %6s" "outer" "f(x)" "‖∇f(x)‖" "ρ" "σ" "μ" "‖x‖" "‖s‖" "reg" "rate"
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

  νcpInv = (1 + θ) * (μmax^2 + μmin)

  s = zero(xk)

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime
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

    if k == 1
      ϵ_increment = (ξ0 ≤ 10 * eps(eltype(xk))) ? ϵr * metric : ϵr * ξ0
      ϵ += ϵ_increment # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
      μk = 1 / metric
    end
    
    if (metric < ϵ) && nls.sample_rate == 1.0 #checks if the optimal condition is satisfied and if all of the data have been visited
      # the current xk is approximately first-order stationary
      optimal = true
    end

    #subsolver_options.ϵa = k == 1 ? 1.0e-1 : max(ϵ_subsolver, min(1.0e-2, metric^2 / 10))

    #update of σk
    σk = min(max(μk * metric, σmin), σmax)

    # TODO: reuse residual computation
    # model for subsequent prox-gradient iterations

    mk_smooth(d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2
    end

    # LSMR strategy for LinearOperators #
    s, stats = lsmr(Jk, -Fk; λ = sqrt(0.5*σk), atol = subsolver_options.ϵa, rtol = ϵr)
    Complex_hist[k] = stats.niter

    xkn .= xk .+ s

    Fkn = residual(nls, xkn)
    fkn = dot(Fkn, Fkn) / 2
    mks = mk_smooth(s)
    Δobj = fk - fkn
    ξ = fk - mks
    ρk = Δobj / ξ

    #μ_stat = ((η1 ≤ ρk < Inf) && ((metric ≥ η3 / μk))) ? "↘" : "↗"
    μ_stat = ρk < η1 ? "↘" : ((nls.sample_rate==1.0 && (metric > η2))||(nls.sample_rate<1.0 && (metric ≥ η3 / μk)) ? "↗" : "=")
    #μ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s %6.2e" k fk norm(∇fk) ρk σk μk norm(xk) norm(s) μ_stat nls.sample_rate
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
    if nls.sample_rate*k - epoch_count >= 1 #we passed on all the data
      epoch_count += 1
      push!(nls.epoch_counter, k)
    end

    #changes sample with new sample rate
    nls.sample = sort(randperm(nls.nls_meta.nequ)[1:Int(ceil(nls.sample_rate * nls.nls_meta.nequ))])
    #sparse_sample = sp_sample(rows, nls.sample)
    if nls.sample_rate == 1.0
      nls.sample == 1:nls.nls_meta.nequ || error("Sample Error : Sample should be full for 100% sampling")
    end

    if (η1 ≤ ρk < Inf) #&& (metric ≥ η3 / μk) #successful step
      xk .= xkn

      if (nls.sample_rate < 1.0) && metric ≥ η3 / μk #very successful step
        μk = max(μk / λ, μmin)
      elseif (nls.sample_rate == 1.0) && (η2 ≤ ρk < Inf)
        μk = max(μk / λ, μmin)
      end

      Fk .= Fkn
      fk = dot(Fk, Fk) / 2

      Jk = jac_op_residual(nls, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)

      μmax = opnorm(Jk)
      νcpInv = (1 + θ) * (μmax^2 + μmin)

      Complex_hist[k] += 1

    else # (ρk < η1 || ρk == Inf) #|| (metric < η3 / μk) #unsuccessful step
      μk = max(λ * μk, μmin)
    end

    tired = epoch_count ≥ maxEpoch || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e" k "" fk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8.1e %7.4e %8s %7.1e %7.1e %7.1e %7.1e" k fk norm(∇fk) "" σk μk norm(xk) norm(s)
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
  return stats
end
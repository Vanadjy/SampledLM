#export Sto_LM

"""
    Sto_LM(nls, h, options; kwargs...)

A Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖² + ψ(s; x)

where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; x) = h(x + s),
and σ > 0 is a regularization parameter.

In this version of the algorithm, the smooth part of both the objective and the model are estimations as 
the quantities are sampled ones from the original data of the Problem.

### Arguments

* `nls::AbstractNLSModel`: a smooth nonlinear least-squares problem
* `options::ROSolverOptions`: a structure containing algorithmic parameters

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nls.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver
* `subsolver`: the procedure used to compute a step (`PG` or `R2`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver.
* `selected::AbstractVector{<:Integer}`: list of selected indexes for the sampling 

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function SPLM(
  nls::SampledNLSModel,
  options::ROSolverOptions;
  x0::AbstractVector = nls.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = RegularizedOptimization.R2,
  subsolver_options = RegularizedOptimization.ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:(nls.meta.nvar),
  Jac_lop::Bool = true
)

  start_time = time()
  elapsed_time = 0.0
  # initialize passed options
  ϵ = options.ϵa
  ϵ_subsolver = subsolver_options.ϵa
  ϵr = options.ϵr
  verbose = options.verbose
  maxIter = options.maxIter
  maxIter = Int(ceil(maxIter * (nls.nls_meta.nequ / length(nls.sample)))) #computing the sample rate
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  η3 = options.η3
  θ = options.θ
  λ = options.λ
  β = options.β
  νcp = options.νcp
  σmin = options.σmin
  σmax = options.σmax
  μmin = options.μmin
  metric = options.metric

  m = nls.nls_meta.nequ

  # Initializes epoch_counter
  epoch_count = 0

  # store initial values of the subsolver_options fields that will be modified
  ν_subsolver = subsolver_options.ν
  ϵa_subsolver = subsolver_options.ϵa

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

  local ξcp
  local exact_ξcp
  local ξ
  k = 0
  X_hist = []
  Fobj_hist = zeros(maxIter)
  exact_Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Metric_hist = zeros(maxIter)
  exact_Metric_hist = zeros(maxIter)
  Complex_hist = zeros(maxIter)
  Grad_hist = zeros(maxIter)
  Resid_hist = zeros(maxIter)

  #Historic of time
  TimeHist = []

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "f(x)" "‖∇f(x)‖" "ρ" "σ" "μ" "‖x‖" "‖s‖" "reg"
    #! format: on
  end

  #creating required objects
  Fk = residual(nls, xk)
  Fkn = similar(Fk)
  exact_Fk = zeros(1:m)

  Jk = jac_op_residual(nls, xk)

  fk = dot(Fk, Fk) / 2 #objective estimated without noise

  #sampled Jacobian
  ∇fk = similar(xk)
  jtprod_residual!(nls, xk, Fk, ∇fk)
  JdFk = similar(Fk)   # temporary storage
  Jt_Fk = similar(∇fk)
  exact_Jt_Fk = similar(∇fk)

  μmax = opnorm(Jk)
  νcpInv = (1 + θ) * (μmax^2 + μmin)
  νInv = (1 + θ) * (μmax^2 + σk)  # ‖J'J + σₖ I‖ = ‖J‖² + σₖ

  s = zero(xk)
  scp = similar(s)

  optimal = false
  tired = k ≥ maxIter || elapsed_time > maxTime
  #tired = elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    mₛ = length(nls.sample) #current length of the sample
    elapsed_time = time() - start_time
    push!(X_hist, xk)
    Fobj_hist[k] = fk
    Grad_hist[k] = nls.counters.neval_jtprod_residual + nls.counters.neval_jprod_residual
    Resid_hist[k] = nls.counters.neval_residual
    if k == 1
      push!(TimeHist, 0.0)
    else
      push!(TimeHist, elapsed_time)
    end

    metric = norm(∇fk)
    Metric_hist[k] = metric

    if k == 1
      ϵ_increment = ϵr * metric
      ϵ += ϵ_increment  # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
      μk = 1 / metric
    end

    if (metric < ϵ) #checks if the optimal condition is satisfied and if all of the data have been visited
      # the current xk is approximately first-order stationary
      push!(nls.opt_counter, k) #indicates the iteration where the tolerance has been reached by the metric
      if nls.sample_rate == 1.0
        optimal = true
      else
        if (length(nls.opt_counter) ≥ 3) && (nls.opt_counter[end-2:end] == range(k-2, k)) #if the last 5 iterations are successful
          optimal = true
        end
      end
    end

    subsolver_options.ϵa = k == 1 ? 1.0e-1 : max(ϵ_subsolver, min(1.0e-2, metric^2 / 10))
    #update of σk
    σk = min(max(μk * metric, σmin), σmax)

    # TODO: reuse residual computation
    # model for subsequent prox-gradient iterations
    mk_smooth(d) = begin
        jprod_residual!(nls, xk, d, JdFk)
        JdFk .+= Fk
        return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2
    end
  
    if Jac_lop
        # LSMR strategy for LinearOperators #
        s, stats = lsmr(Jk, -Fk; λ = σk)#, atol = subsolver_options.ϵa, rtol = ϵr)
        Complex_hist[k] = stats.niter
    else
        spmat = qrm_spmat_init(length(nls.sample), meta_nls.nvar, rows[sparse_sample], cols[sparse_sample], vals[sparse_sample])
        spfct = qrm_analyse(spmat)
        qrm_factorize!(spmat, spfct)
        z = qrm_apply(spfct, -Fk, transp = 't') #TODO include complex compatibility
        s = qrm_solve(spfct, z, transp = 'n')
    end

    xkn .= xk .+ s

    Fkn = residual(nls, xkn)
    fkn = dot(Fkn, Fkn) / 2
    mks = mk_smooth(s)
    Δobj = fk - fkn
    ξ = fk - mks
    ρk = Δobj / ξ

    #μ_stat = ((η1 ≤ ρk < Inf) && ((metric ≥ η3 / μk))) ? "↘" : "↗"
    μ_stat = ρk < η1 ? "↘" : ((metric ≥ η3 / μk) ? "↗" : "↘")
    #μ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8.1e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k fk norm(∇fk) ρk σk μk norm(xk) norm(s) μ_stat
      #! format: off
    end
    
    
    #-- to compute exact quantities --#
    if nls.sample_rate < 1.0
      nls.sample = 1:m
      residual!(nls, xk, exact_Fk)
      exact_fk = dot(exact_Fk, exact_Fk) / 2
      jtprod_residual!(nls, xk, exact_Fk, ∇fk)
      exact_metric = norm(∇fk)

      exact_Fobj_hist[k] = exact_fk
      exact_Metric_hist[k] = exact_metric
    elseif nls.sample_rate == 1.0
      exact_Fobj_hist[k] = fk
      exact_Metric_hist[k] = metric
    end
    # -- -- #

    #updating the indexes of the sampling
    nls.sample = sort(randperm(nls.nls_meta.nequ)[1:mₛ])
    if nls.sample_rate*k - epoch_count >= 1 #we passed on all the data
      epoch_count += 1
      push!(nls.epoch_counter, k)
    end

    if (η1 ≤ ρk < Inf) #&& (metric ≥ η3 / μk) #successful step
      xk .= xkn

      if (nls.sample_rate < 1.0) && metric ≥ η3 / μk #very successful step
        μk = max(μk / λ, μmin)
      elseif (nls.sample_rate == 1.0) && (η2 ≤ ρk < Inf)
        μk = max(μk / λ, μmin)
      end

      # update functions #FIXME : obligés de refaire appel à residual! après changement du sampling --> on fait des évaluations du résidus en plus qui pourraient peut-être être évitées...
      Fk .= Fkn
      fk = fkn

      # update gradient & Hessian
      Jk = jac_op_residual(nls, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)

      μmax = opnorm(Jk)
      #η3 = μmax^2
      νcpInv = (1 + θ) * (μmax^2 + μmin) 

      Complex_hist[k] += 1
    else # (ρk < η1 || ρk == Inf) #|| (metric < η3 / μk) #unsuccessful step
      μk = max(λ * μk, μmin)
    end

    tired = k ≥ maxIter || elapsed_time > maxTime
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
  set_residuals!(stats, zero(eltype(xk)), sqrt(dot(∇fk, ∇fk)))
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :Xhist, X_hist[1:k])
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
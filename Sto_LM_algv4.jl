#export SLM

"""
    Sto_LM_v4(nls, h, options; kwargs...)

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
* `h`: a regularizer such as those defined in ProximalOperators
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
function Sto_LM_v4(
  nls::SampledNLSModel,
  h::H,
  options::ROSolverOptions;
  x0::AbstractVector = nls.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = RegularizedOptimization.ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:(nls.meta.nvar)
) where {H}

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
  νcp = options.νcp
  σmin = options.σmin
  μmin = options.μmin
  metric = options.metric

  m = nls.nls_meta.nequ

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
  hk = h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "SLM: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "SLM: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")
  ψ = shifted(h, xk)

  xkn = similar(xk)

  local ξcp
  local exact_ξcp
  local ξ
  k = 0
  Fobj_hist = zeros(maxIter)
  exact_Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Metric_hist = zeros(maxIter)
  exact_Metric_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  Grad_hist = zeros(Int, maxIter)
  Resid_hist = zeros(Int, maxIter)

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %7s %7s %1s %6s" "outer" "inner" "f(x)" "h(x)" "√ξcp/νcp" "√ξ/ν" "ρ" "σ" "μ" "ν" "‖x‖" "‖s‖" "‖Jₖ‖²" "reg" "count"
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
  νcpInv = (1 + θ) * μmax^2
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
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    Grad_hist[k] = nls.counters.neval_jtprod_residual + nls.counters.neval_jprod_residual
    Resid_hist[k] = nls.counters.neval_residual

    # model for the Cauchy-Point decrease
    φcp(d) = begin
      jtprod_residual!(nls, xk, Fk, Jt_Fk)
      dot(Fk, Fk) / 2 + dot(Jt_Fk, d)
    end

    #submodel to find scp
    mkcp(d) = φcp(d) + ψ(d) #+ νcpInv * dot(d,d) / 2
    
    #computes the Cauchy step
    νcp = 1 / νcpInv
    ∇fk .*= -νcp
    # take first proximal gradient step s1 and see if current xk is nearly stationary
    # s1 minimizes φ1(s) + ‖s‖² / 2 / ν + ψ(s) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0)).
    prox!(scp, ψ, ∇fk, νcp)
    ξcp = fk + hk - φcp(scp) - ψ(scp) + max(1, abs(fk + hk)) * 10 * eps()  # TODO: isn't mk(s) returned by subsolver?

    #ξcp > 0 || error("LM: first prox-gradient step should produce a decrease but ξcp = $(ξcp)")
    
    if ξcp ≤ 0
      ξcp = - ξcp
    end

    metric = sqrt(ξcp*νcpInv)
    Metric_hist[k] = metric

    if ξcp ≥ 0 && k == 1
      ϵ_increment = ϵr * metric
      ϵ += ϵ_increment  # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
    end

    if (metric < ϵ) #checks if the optimal condition is satisfied and if all of the data have been visited
      # the current xk is approximately first-order stationary
      push!(nls.opt_counter, k) #indicates the iteration where the tolerance has been reached by the metric
      #=if length(nls.opt_counter) == 200
        optimal = true
      end=#
    end

    subsolver_options.ϵa = k == 1 ? 1.0e-1 : max(ϵ_subsolver, min(1.0e-2, ξcp / 10))
    #update of σk
    σk = μk * metric

    # TODO: reuse residual computation
    # model for subsequent prox-gradient iterations
    φ(d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2
    end

    ∇φ!(g, d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      jtprod_residual!(nls, xk, JdFk, g)
      g .+= σk * d
      return g
    end

    mk(d) = begin
      jprod_residual!(nls, xk, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2 + ψ(d)
    end
  
    νInv = (1 + θ) * (μmax^2 + σk) # μmax^2 + σk = ||Jmk||² + σk 
    ν = 1 / νInv
    subsolver_options.ν = ν

    @debug "setting inner stopping tolerance to" subsolver_options.optTol
    s, iter, _ = with_logger(subsolver_logger) do
      subsolver(φ, ∇φ!, ψ, subsolver_options, s)
    end
    # restore initial subsolver_options here so that it is not modified if there is an error
    subsolver_options.ν = ν_subsolver
    subsolver_options.ϵa = ϵa_subsolver

    Complex_hist[k] = iter
    # additionnal condition on step s
    if dot(s,s) > 2 / μk
      println("cauchy step used")
      s .= scp # Cauchy step allows a minimum decrease
    end

    xkn .= xk .+ s

    residual!(nls, xkn, Fkn)
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")
    mks = mk(s)
    ξ = fk + hk - mks + max(1, abs(hk)) * 10 * eps()

    #=if (ξ ≤ 0 || isnan(ξ))
      error("LM: failed to compute a step: ξ = $ξ")
    end=#

    if ξ ≤ 0
      ξ = - ξ
    end

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    #Δobj ≥ 0 || error("Δobj should be positive while Δobj = $Δobj, we should have a decreasing direction but fk + hk - (fkn + hkn) = $(fk + hk - (fkn + hkn))")
    ρk = Δobj / ξ

    μ_stat = ((η1 ≤ ρk < Inf) && ((metric ≥ η3 / μk))) ? "↘" : "↗"
    #μ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.4e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e %7.1e %1s %6d" k iter fk hk sqrt(ξcp*νcpInv) sqrt(ξ*νInv) ρk σk μk ν norm(xk) norm(s) νInv μ_stat length(nls.opt_counter)
      #! format: off
    end

    #=if (η2 ≤ ρk < Inf) #&& (metric ≥ η3 / μk) #If very successful, decrease the penalisation parameter
      μk = max(μk / λ, μmin)
    end=#
    #-- to compute exact quantities --#
    nls.sample = 1:m
    residual!(nls, xk, exact_Fk)
    exact_fk = dot(exact_Fk, exact_Fk) / 2

    exact_φcp(d) = begin
      jtprod_residual!(nls, xk, exact_Fk, exact_Jt_Fk)
      dot(exact_Fk, exact_Fk) / 2 + dot(exact_Jt_Fk, d)
    end

    exact_ξcp = exact_fk + hk - exact_φcp(scp) - ψ(scp) + max(1, abs(fk + hk)) * 10 * eps()
    exact_metric = sqrt(abs(exact_ξcp * νcpInv))

    exact_Fobj_hist[k] = exact_fk
    exact_Metric_hist[k] = exact_metric
    # -- -- #

    #updating the indexes of the sampling
    #nls.sample = sort(randperm(nls.nls_meta.nequ)[1:mₛ])
    nls.sample = randperm(nls.nls_meta.nequ)[1:mₛ]
    update_sample!(nls, k)
    #display(nls.data_mem)

    if (η1 ≤ ρk < Inf) && (metric ≥ η3 / μk) #successful step
      xk .= xkn
      μk = max(μk / λ, μmin)

      # update functions #FIXME : obligés de refaire appel à residual! après changement du sampling --> on fait des évaluations du résidus en plus qui pourraient peut-être être évitées...
      residual!(nls, xk, Fk)
      fk = dot(Fk, Fk) / 2
      hk = hkn

      # update gradient & Hessian
      shift!(ψ, xk)
      Jk = jac_op_residual(nls, xk)
      jtprod_residual!(nls, xk, Fk, ∇fk)

      μmax = opnorm(Jk)
      #η3 = μmax^2
      νcpInv = (1 + θ) * (μmax^2) 

      Complex_hist[k] += 1
    #end

    else # (ρk < η1 || ρk == Inf) #|| (metric < η3 / μk) #unsuccessful step
      μk = λ * μk
    end

    #update of the sampling rate towards the accuracy reached
    #=if metric < 1e-2
      nls.sample_rate = .50
    elseif metric < 1e-3
      nls.sample_rate = .80
    elseif metric < 1e-4
      nls.sample_rate = .90
    elseif metric < 1e-5
      nls.sample_rate = 1.0
    end=#

    tired = k ≥ maxIter || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.4e %7.1e %8s %7.1e %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt(ξcp*νcpInv) sqrt(ξ*νInv) "" σk μk norm(xk) norm(s) νInv
      #! format: on
      @info "SLM: terminating with √ξcp/νcp = $metric"
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
  set_objective!(stats, fk + hk)
  set_residuals!(stats, zero(eltype(xk)), ξcp ≥ 0 ? sqrt(ξcp * νcpInv) : ξcp)
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:k])
  set_solver_specific!(stats, :Hhist, Hobj_hist[1:k])
  set_solver_specific!(stats, :NonSmooth, h)
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k])
  set_solver_specific!(stats, :NLSGradHist, Grad_hist[1:k])
  set_solver_specific!(stats, :ResidHist, Resid_hist[1:k])
  return stats, Metric_hist[1:k], exact_Fobj_hist[1:k], exact_Metric_hist[1:k]
end
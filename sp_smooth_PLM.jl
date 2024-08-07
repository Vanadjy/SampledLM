function SPLM(
    nls::SampledBAModel,
    options::ROSolverOptions;
    x0::AbstractVector = nls.meta.x0,
    subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
    subsolver = RegularizedOptimization.R2,
    subsolver_options = RegularizedOptimization.ROSolverOptions(ϵa = options.ϵa),
    selected::AbstractVector{<:Integer} = 1:(nls.meta.nvar),
    sample_rate0::Float64 = .05,
    version::Int = 1,
    Jac_lop::Bool = false
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
    #nls.sample = sort(randperm(nls.nls_meta.nequ)[1:ζk])
  
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
    νcp = options.νcp
    σmin = options.σmin
    σmax = options.σmax
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
  
    xkn = similar(xk)
  
    local ξcp
    local exact_ξcp
    local ξ
    local ξ_mem
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
      @info @sprintf "%6s %7s %7s %8s %7s %7s %7s %7s %1s %6s" "outer" "f(x)" "  ‖∇f(x)‖" "ρ" "σ" "μ" "‖x‖" "‖s‖" "reg" "rate"
      #! format: on
    end
  
    #creating required objects
    Fk = residual(nls, xk)
    Fkn = similar(Fk)
    exact_Fk = zeros(1:m)
  
    fk = dot(Fk, Fk) / 2 #objective estimated without noise
    
    #sampled Jacobian
    ∇fk = similar(xk)
    JdFk = similar(Fk) # temporary storage
    Jt_Fk = similar(∇fk)
    exact_Jt_Fk = similar(∇fk)
  
    meta_nls = nls_meta(nls)
    rows = Vector{Int}(undef, meta_nls.nnzj)
    cols = Vector{Int}(undef, meta_nls.nnzj)
    vals = similar(xk, meta_nls.nnzj)
    jac_structure_residual!(nls, rows, cols)
    jac_coord_residual!(nls, nls.meta.x0, vals)
    jtprod_residual!(nls, rows, cols, vals, Fk, ∇fk)
    qrm_init()
  
    if Jac_lop
      Jk = jac_op_residual!(nls, rows, cols, vals, JdFk, Jt_Fk)
      μmax = opnorm(Jk)
    else
      sparse_sample = sp_sample(rows, nls.sample)
      μmax = norm(vals)
    end
  
    νcpInv = (1 + θ) * (μmax^2 + μmin)
    νInv = (1 + θ) * (μmax^2 + σk)  # ‖J'J + σₖ I‖ = ‖J‖² + σₖ
  
    s = zero(xk)
    scp = similar(s)
  
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
      
      if (metric < ϵ) #checks if the optimal condition is satisfied and if all of the data have been visited
        # the current xk is approximately first-order stationary
        push!(nls.opt_counter, k) #indicates the iteration where the tolerance has been reached by the metric
        if nls.sample_rate == 1.0
          optimal = true
        else
          if (length(nls.opt_counter) ≥ 5) && (nls.opt_counter[end-2:end] == range(k-2, k)) #if the last 5 iterations are successful
            optimal = true
          end
        end
      end
  
      subsolver_options.ϵa = (length(nls.epoch_counter) ≤ 1 ? 1.0e-1 : max(ϵ_subsolver, min(1.0e-2, metric / 10)))
  
      #update of σk
      σk = min(max(μk * metric, σmin), σmax)
  
      # TODO: reuse residual computation
      # model for subsequent prox-gradient iterations
  
      mk_smooth(d) = begin
        jprod_residual!(nls, rows, cols, vals, d, JdFk)
        JdFk .+= Fk
        return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2
      end
  
      if Jac_lop
        # LSMR strategy for LinearOperators #
        s, stats = lsmr(Jk, -Fk; λ = sqrt(σk), itmax = 300)#, atol = subsolver_options.ϵa, rtol = ϵr)
        Complex_hist[k] = stats.niter
      else
        if nls.sample_rate == 1.0
          n = meta_nls.nvar
          rows_qrm = vcat(rows, (meta_nls.nequ+1):(meta_nls.nequ + n))
          cols_qrm = vcat(cols, 1:n)
          vals_qrm = vcat(vals, sqrt(σk) .* ones(n))

          @assert length(rows_qrm) == length(cols_qrm)
          @assert length(rows_qrm) == length(vals_qrm)
          @assert meta_nls.nequ + n ≥ maximum(rows_qrm)
          @assert n ≥ maximum(cols_qrm)

          spmat = qrm_spmat_init(meta_nls.nequ + n, n, rows_qrm, cols_qrm, vals_qrm)
          qrm_least_squares!(spmat, vcat(-Fk, zeros(n)), s)
        else
          n = maximum(cols[sparse_sample])
          m = maximum(rows[sparse_sample])
          rows_qrm = vcat(rows[sparse_sample], (meta_nls.nequ+1):(meta_nls.nequ + n))
          cols_qrm = vcat(cols[sparse_sample], 1:n)
          vals_qrm = vcat(vals[sparse_sample], sqrt(σk) .* ones(n))

          display(nls.sample_rate)
          display(m)
          display(n)
          display(length(Fk))

          spmat = qrm_spmat_init(length(Fk) + n, n, rows_qrm, cols_qrm, vals_qrm)
          qrm_least_squares!(spmat, vcat(-Fk, zeros(n)), s)
        end
        #spmat = qrm_spmat_init(meta_nls.nequ + n, n, rows_qrm, cols_qrm, vals_qrm)
        #spmat = qrm_spmat_init(meta_nls.nequ, meta_nls.nvar, rows, cols, vals)
        #=spfct = qrm_spfct_init(spmat)
        qrm_analyse!(spmat, spfct)
        qrm_factorize!(spmat, spfct)
        z = qrm_apply(spfct, -Fk; transp = 't') #TODO include complex compatibility
        s = qrm_solve(spfct, z; transp = 'n')=#
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
        @info @sprintf "%6d %8.1e %7.4e %8.1e %7.1e %7.1e %7.1e %7.1e %1s %6.2e" k fk norm(∇fk) ρk σk μk norm(xk) norm(s) μ_stat nls.sample_rate
        #! format: off
      end
      
      #-- to compute exact quantities --#
      #=if nls.sample_rate < 1.0
        nls.sample = 1:nls.nobs
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
      elseif nls.sample_rate == 1.0
        exact_Fobj_hist[k] = fk
        exact_Metric_hist[k] = metric
      end=#
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
          #@views mobile_mean = mean(Fobj_hist[(k - Num_mean + 1):k] + Hobj_hist[(k - Num_mean + 1):k])
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
          if metric/ξ_mem ≤ 1e-1 #if the mean on the Num_mean last iterations is near the current objective value
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
  
      #changes sample with new sample rate
      nls.sample = sort(randperm(nls.nobs)[1:Int(ceil(nls.sample_rate * nls.nobs))])
      sparse_sample = sp_sample(rows, nls.sample)
      if nls.sample_rate == 1.0
        nls.sample == 1:nls.nobs || error("Sample Error : Sample should be full for 100% sampling")
      end
  
      if change_sample_rate
        # mandatory updates whenever the sample_rate chages #
        Fk = residual(nls, xk)
        Fkn = similar(Fk)
        JdFk = similar(Fk)
        fk = dot(Fk, Fk) / 2
  
        #Jk = jac_op_residual(nls, xk)
        jtprod_residual!(nls, rows, cols, vals, Fk, ∇fk)
        jac_coord_residual!(nls, nls.meta.x0, vals)
        μmax = norm(vals)
        νcpInv = (1 + θ) * (μmax^2 + μmin)
  
        #change_sample_rate = false
      end
  
      if (η1 ≤ ρk < Inf) #&& (metric ≥ η3 / μk) #successful step
        xk .= xkn
  
        if (nls.sample_rate < 1.0) && metric ≥ η3 / μk #very successful step
          μk = max(μk / λ, μmin)
        elseif (nls.sample_rate == 1.0) && (η2 ≤ ρk < Inf)
          μk = max(μk / λ, μmin)
        end
  
        if (!change_sample_rate) && (nls.sample_rate == 1.0)
          Fk .= Fkn
        else
          Fk = residual(nls, xk)
        end
        fk = dot(Fk, Fk) / 2
  
        # update gradient & Hessian
        # Jk = jac_op_residual(nls, xk)
        jac_coord_residual!(nls, xk, vals)
        jtprod_residual!(nls, rows, cols, vals, Fk, ∇fk)
  
        μmax = norm(vals)
        #μmax = opnorm(Jk)
        νcpInv = (1 + θ) * (μmax^2 + μmin)
  
        #Complex_hist[k] += nls.sample_rate
  
      else # (ρk < η1 || ρk == Inf) #|| (metric < η3 / μk) #unsuccessful step
        μk = λ * μk
      end
  
      if change_sample_rate
        change_sample_rate = false
      end
  
      tired = epoch_count ≥ maxEpoch-1 || elapsed_time > maxTime
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
    set_solver_specific!(stats, :SampleRateHist, Sample_hist[1:k])
    return stats
  end
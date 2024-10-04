#export SPLM
using Logging, Printf, LinearAlgebra
using NLPModels, ADNLPModels, QRMumps, SolverCore
using SparseArrays
using BundleAdjustmentModels
using RegularizedOptimization
using LinearOperators
using FastClosures

include("input_struct_sto.jl")

nls = BundleAdjustmentModel("problem-16-22106-pre") #dubrovnik
meta_nls_ba = nls_meta(nls)

function F!(Fx, x)
    residual!(nls, x, Fx)
end

rows = Vector{Int}(undef, nls.nls_meta.nnzj)
cols = Vector{Int}(undef, nls.nls_meta.nnzj)
vals = ones(Bool, nls.nls_meta.nnzj)
jac_structure_residual!(nls, rows, cols)
J = sparse(rows, cols, vals, nls.nls_meta.nequ, nls.meta.nvar)

jac_back = ADNLPModels.SparseADJacobian(nls.meta.nvar, F!, nls.nls_meta.nequ, nothing, J)

adnls = ADNLSModel!(F!, nls.meta.x0,  nls.nls_meta.nequ, nls.meta.lvar, nls.meta.uvar, jacobian_residual_backend = jac_back,
    jacobian_backend = ADNLPModels.EmptyADbackend,
    hessian_backend = ADNLPModels.EmptyADbackend,
    hessian_residual_backend = ADNLPModels.EmptyADbackend,
    matrix_free = true
)

sampled_options = ROSolverOptions(η3 = .4, σmax = 1e6, ϵa = 1e-8, ϵr = 1e-8, σmin = 1e-6, μmin = 1e-10, verbose = 10, maxIter = 100, maxTime = 3600.0;)
options = RegularizedOptimization.ROSolverOptions(ϵa = 1e-8, ϵr = 1e-8, σmin = 1e-6, verbose = 10, maxIter = 100, maxTime = 3600.0;)

#LM_out = levenberg_marquardt(nls; η₁ = √√eps(Float64), η₂ = 0.9, σ₁ = 3.0, σ₂ = 1/3, λ = 1.0, λmin = 1e-6, max_iter = 100, max_time = 3600,
#in_itmax = 1000)

"""
    LM_qrm(
    nls::ADNLSModel,
    options::ROSolverOptions;
    x0::AbstractVector = nls.meta.x0
  )

    LM_qrm(nls, options, version; kwargs...)

A Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖²

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖²

where F(x) and J(x) are the residual and its sparse Jacobian at x, respectively
and σ > 0 is a regularization parameter using a multicore QR factorization
(named QRMumps).

### Arguments

* `nls::AbstractADNLSModel`: a smooth nonlinear least-squares problem using AD backend
* `options::ROSolverOptions`: a structure containing algorithmic parameters

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nls.meta.x0`)

### Return values
Generic solver statistics including among others

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the residuals
"""
function LM_qrm(
    nls::ADNLSModel,
    options::ROSolverOptions;
    x0::AbstractVector = nls.meta.x0
  )
  
    # initialize time stats
    start_time = time()
    elapsed_time = 0.0
  
    # initialize passed options
    ϵ = options.ϵa
    ϵr = options.ϵr
    verbose = options.verbose
    maxIter = options.maxIter
    maxTime = options.maxTime
    η1 = options.η1
    η2 = options.η2
    λ = options.λ
    σmin = options.σmin
    σmax = options.σmax
    μmin = options.μmin
    metric = options.metric
  
    # Initialize solver constants
    m = nls.nls_meta.nequ
    n = nls.meta.nvar
  
    if verbose == 0
      ptf = Inf
    elseif verbose == 1
      ptf = round(maxIter / 10)
    elseif verbose == 2
      ptf = round(maxIter / 100)
    else
      ptf = 1
    end

    k = 0
    qrm_init()

    xk = copy(x0)
    s = zero(xk)
    xkn = similar(xk)

    # Statistics historics
    Fobj_hist = zeros(maxIter)
    Metric_hist = zeros(maxIter)
    Grad_hist = zeros(maxIter)
    Resid_hist = zeros(maxIter)
    TimeHist = []
  
    if verbose > 0
      #! format: off
      @info @sprintf "%6s %7s %7s %8s %7s %7s %7s %7s %1s" "outer" "f(x)" "  ‖∇f(x)‖" "ρ" "σ" "μ" "‖x‖" "‖s‖" "reg"
      #! format: on
    end
  
    #Generating mandatory storing objects
    Fk = residual(nls, xk)
    Fkn = similar(Fk)
    ∇fk = similar(xk)
    JdFk = similar(Fk) # temporary storage
  
    #Generating sparse structure
    rows = Vector{Int}(undef, nls.nls_meta.nnzj)
    cols = Vector{Int}(undef, nls.nls_meta.nnzj)
    vals = similar(xk, nls.nls_meta.nnzj)
    jac_structure_residual!(nls, rows, cols)
    jac_coord_residual!(nls, nls.meta.x0, vals)

    #Generating sparse structure for QRMumps solve
    rows_qrm = similar(rows, nls.nls_meta.nnzj+n)
    cols_qrm = similar(cols, nls.nls_meta.nnzj+n)
    vals_qrm = similar(xk, nls.nls_meta.nnzj+n)

    fk = dot(Fk, Fk) / 2
    jtprod_residual!(nls, rows, cols, vals, Fk, ∇fk) #Calculating gradient stored in ∇fk
    metric = norm(∇fk)

    # initialize regularization parameters
    μk = max(1e-3 / metric, μmin)
    σk = min(max(μk * metric, σmin), σmax)

    optimal = false
    tired = k ≥ maxIter || elapsed_time > maxTime
    #tired = elapsed_time > maxTime

    ϵ_increment = ϵr * metric
    ϵ += ϵ_increment  # make stopping test absolute and relative
  
    while !(optimal || tired)
      k += 1
      elapsed_time = time() - start_time
      Fobj_hist[k] = fk
      Grad_hist[k] = nls.counters.neval_jtprod_residual + nls.counters.neval_jprod_residual
      Resid_hist[k] = nls.counters.neval_residual
      metric = norm(∇fk)
      Metric_hist[k] = metric

      if k == 1
        push!(TimeHist, 0.0)
      else
        push!(TimeHist, elapsed_time)
      end
      
      if (metric < ϵ) #checks if the optimal condition is satisfied
        optimal = true
      end
  
      #update of σk
      σk = min(max(μk * metric, σmin), σmax)
  
      mk(d) = begin
        jprod_residual!(nls, rows, cols, vals, d, JdFk)
        JdFk .+= Fk
        return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2
      end
      @views rows_qrm[1:length(rows)] .= rows
      @views rows_qrm[length(rows)+1:end] .= (m+1):(m + n)
      @views cols_qrm[1:length(cols)] .= cols
      @views cols_qrm[length(cols)+1:end] .= 1:n
      @views vals_qrm[1:length(vals)] .= vals
      @views vals_qrm[length(vals)+1:end] .= sqrt(σk) .* ones(n)

      @assert length(rows_qrm) == length(cols_qrm)
      @assert length(rows_qrm) == length(vals_qrm)
      @assert m + n ≥ maximum(rows_qrm)
      @assert n ≥ maximum(cols_qrm)

      #QRMumps solving linear least squares LM subproblem
      spmat = qrm_spmat_init(m + n, n, rows_qrm, cols_qrm, vals_qrm)
      qrm_least_squares!(spmat, vcat(-Fk, zeros(n)), s)
  
      xkn .= xk .+ s
      residual!(nls, xkn, Fkn)
      fkn = dot(Fkn, Fkn) / 2

      Δobj = fk - fkn
      ρk = Δobj / (fk - mk(s))
  
      μ_stat = ρk < η1 ? "↘" : ((η2 ≤ ρk) ? "↗" : "=")
  
      if (verbose > 0) && (k % ptf == 0)
        #! format: off
        @info @sprintf "%6d %8.1e %7.4e %8.1e %7.1e %7.1e %7.1e %7.1e %1s" k fk norm(∇fk) ρk σk μk norm(xk) norm(s) μ_stat
        #! format: off
      end
      
      #step selection
      if (η1 ≤ ρk) #successful step
        xk .= xkn
  
        if (η2 ≤ ρk) #very successful step 
          μk = max(μk / λ, μmin)
        end

        Fk .= Fkn
        fk = dot(Fk, Fk) / 2
        jac_coord_residual!(nls, xk, vals)
        jtprod_residual!(nls, rows, cols, vals, Fk, ∇fk)
      else # (ρk < η1) # unsuccessful step
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
        @info "LM_qrm: terminating with ‖∇f(x)‖= $(norm(∇fk))"
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
    set_solver_specific!(stats, :NLSGradHist, Grad_hist[1:k])
    set_solver_specific!(stats, :ResidHist, Resid_hist[1:k])
    set_solver_specific!(stats, :MetricHist, Metric_hist[1:k])
    set_solver_specific!(stats, :TimeHist, TimeHist)
    return stats
end

#=io = open("log_LM_qrm.txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)=#
LM_qrm_out = LM_qrm(adnls, sampled_options)
#close(io)

sol = LM_qrm_out.solution
sol0 = nls.meta.x0
#display(norm(sol - sol0))
x0 = [sol0[3*i+1] for i in 0:(nls.npnts-1)]
y0 = [sol0[3*i+2] for i in 0:(nls.npnts-1)]
z0 = [sol0[3*i] for i in 1:nls.npnts]
plt3d0 = PlotlyJS.scatter(
            x=x0,
            y=y0,
            z=z0,
            mode="markers",
            marker=attr(
                size=1,
                opacity=0.8,
                color = "firebrick"
            ),
            type="scatter3d",
            options=Dict(:showLink => true)
)

x = [sol[3*i+1] for i in 0:(nls.npnts-1)]
y = [sol[3*i+2] for i in 0:(nls.npnts-1)]
z = [sol[3*i] for i in 1:nls.npnts]       
plt3d = PlotlyJS.scatter(
    x=x,
    y=y,
    z=z,
    mode="markers",
    marker=attr(
        size=1,
        opacity=0.8
    ),
    type="scatter3d",
    options=Dict(:showLink => true)
)

layout = Layout(scene = attr(
  xaxis = attr(
      backgroundcolor="rgb(255, 255, 255)",
      title_text = "",
      gridcolor="white",
      showbackground=false,
      zerolinecolor="white",
      tickfont=attr(size=0, color="white")),
  yaxis = attr(
      backgroundcolor="rgb(255, 255, 255)",
      title_text = "",
      gridcolor="white",
      showbackground=false,
      zerolinecolor="white",
      tickfont=attr(size=0, color="white")),
  zaxis = attr(
      backgroundcolor="rgb(255, 255, 255)",
      title_text = "",
      gridcolor="white",
      showbackground=false,
      zerolinecolor="white",
      tickfont=attr(size=0, color="white")),
      margin=attr(
          r=10, l=10,
          b=10, t=10),
      aspectmode = "manual",
      showlegend = false
      ),
      #scene_camera = camera_settings[name]
)

fig_ba = PlotlyJS.Plot(plt3d, layout)
fig_ba0 = PlotlyJS.Plot(plt3d0, layout)
display(LM_qrm_out.solver_specific[:NLSGradHist][end])
display(LM_qrm_out.elapsed_time)
#display(fig_ba)
#display(fig_ba0)
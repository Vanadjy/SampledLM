#export ROSolverOptions

mutable struct ROSolverOptions{R}
  ϵa::R  # termination criteria
  ϵr::R  # relative stopping tolerance
  neg_tol::R # tolerance when ξ < 0
  Δk::R  # trust region radius
  verbose::Int  # print every so often
  maxIter::Int  # maximum amount of inner iterations
  maxTime::Float64 #maximum time allotted to the algorithm in s
  σmin::R # minimum σk allowed for LM/R2 method
  σmax::R # maximum σk allowed for Sampled method
  μmin::R # minimum μk allowed for Sampled Methods
  η1::R  # step acceptance threshold
  η2::R  # trust-region increase threshold
  η3::R #Stochastic metric threshold
  α::R  # νk Δ^{-1} parameter
  ν::R  # initial guess for step length
  νcp::R #Initial guess for step length of Cauchy Point computation
  γ::R  # trust region buffer
  θ::R  # step length factor in relation to Hessian norm
  β::R  # TR size as factor of first PG step
  λ::R # parameter of the random stepwalk
  metric::R #parameter of the stationnarity metric of the algorithm
  spectral::Bool # for TRDH: use spectral gradient update if true, otherwise DiagonalQN
  psb::Bool # for TRDH with DiagonalQN (spectral = false): use PSB update if true, otherwise Andrei update
  reduce_TR::Bool
  M::R

  function ROSolverOptions{R}(;
    ϵa::R = √eps(R),
    ϵr::R = √eps(R),
    neg_tol::R = eps(R)^(1 / 4),
    Δk::R = one(R),
    verbose::Int = 0,
    maxIter::Int = 50,
    maxTime::Float64 = 120.0,
    σmin::R = eps(R),
    σmax::R = eps(R),
    μmin::R = eps(R),
    η1::R = √√eps(R),
    η2::R = R(0.9),
    η3::R = R(e-6),
    α::R = 1 / eps(R),
    ν::R = 1.0e-3,
    νcp::R = 1.0e-2,
    γ::R = R(3),
    θ::R = R(1e-3),
    β::R = 1 / eps(R),
    λ::R = R(3),
    metric::R = R(10),
    spectral::Bool = false,
    psb::Bool = false,
    reduce_TR::Bool = true,
    M::R = R(1e5),
  ) where {R <: Real}
    @assert ϵa ≥ 0
    @assert ϵr ≥ 0
    @assert neg_tol ≥ 0
    @assert Δk > 0
    @assert verbose ≥ 0
    @assert maxIter ≥ 0
    @assert maxTime ≥ 0
    @assert σmin ≥ 0
    @assert σmax ≥ 0
    @assert μmin ≥ 0
    @assert 0 < η1 < η2 < 1
    @assert η3 > 0
    @assert α > 0
    @assert ν > 0
    #relation between vcp and v
    @assert νcp > 0
    @assert γ > 1
    @assert θ > 0
    @assert β ≥ 1
    @assert λ > 1
    @assert metric > 0
    @assert M > 0
    return new{R}(
      ϵa,
      ϵr,
      neg_tol,
      Δk,
      verbose,
      maxIter,
      maxTime,
      σmin,
      σmax,
      μmin,
      η1,
      η2,
      η3,
      α,
      ν,
      νcp,
      γ,
      θ,
      β,
      λ,
      metric,
      spectral,
      psb,
      reduce_TR,
      M,
    )
  end
end

ROSolverOptions(args...; kwargs...) = ROSolverOptions{Float64}(args...; kwargs...)

# ------------------------------------------------------------------------------------------------ #
# ------------------------------ NEW COUNTERS STRUCTURE ------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #

mutable struct NLSGeneralCounters
  counters::Counters
  neval_residual::Float64
  neval_jac_residual::Float64
  neval_jprod_residual::Float64
  neval_jtprod_residual::Float64
  neval_hess_residual::Float64
  neval_jhess_residual::Float64
  neval_hprod_residual::Float64

  function NLSGeneralCounters()
    return new(Counters(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  end
end

function Base.getproperty(c::NLSGeneralCounters, f::Symbol)
    if f in fieldnames(Counters)
      getfield(c.counters, f)
    else
      getfield(c, f)
    end
end
  
function Base.setproperty!(c::NLSGeneralCounters, f::Symbol, x)
    if f in fieldnames(Counters)
      setfield!(c.counters, f, x)
    else
      setfield!(c, f, x)
    end
end
  
function NLPModels.sum_counters(c::NLSGeneralCounters)
    s = sum_counters(c.counters)
    for field in fieldnames(NLSGeneralCounters)
      field == :counters && continue
      s += getfield(c, field)
    end
    return s
end

mutable struct SampledNLSModel{T, S, R, J, Jt} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSGeneralCounters

  resid!::R
  jprod_resid!::J
  jtprod_resid!::Jt
  
  #stochastic parameters
  sample::AbstractVector{<:Integer}
  data_mem::AbstractVector{<:Integer}
  sample_rate::Real
  epoch_counter::AbstractVector{<:Integer}
  opt_counter::AbstractVector{<:Integer}

  function SampledNLSModel{T, S, R, J, Jt}(
    r::R,
    jv::J,
    jtv::Jt,
    nequ::Int,
    x::S,
    sample::AbstractVector{<:Integer},
    data_mem::AbstractVector{<:Integer},
    sample_rate::Real;
    kwargs...,
  ) where {T, S, R <: Function, J <: Function, Jt <: Function}
    nvar = length(x)
    meta = NLPModelMeta(nvar, x0 = x; kwargs...)
    nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x)
    epoch_counter = Int[1]
    opt_counter = Int[]
    return new{T, S, R, J, Jt}(meta, nls_meta, NLSGeneralCounters(), r, jv, jtv, sample, data_mem, sample_rate, epoch_counter, opt_counter)
  end
end

SampledNLSModel(r, jv, jtv, nequ::Int, x::S, sample::AbstractVector{<:Integer}, data_mem::AbstractVector{<:Integer}, sample_rate::Real; kwargs...) where {S} =
SampledNLSModel{eltype(S), S, typeof(r), typeof(jv), typeof(jtv)}(
    r,
    jv,
    jtv,
    nequ,
    x,
    sample,
    data_mem,
    sample_rate;
    kwargs...,
  )

  """
Represent a bundle adjustement problem in the form

    minimize   ½ ‖F(x)‖²

where `F(x)` is the vector of residuals.
"""
mutable struct SampledBAModel{T, S} <: AbstractNLSModel{T, S}
  # Meta and counters are required in every model
  meta::NLPModelMeta{T, S}
  # nls_meta
  nls_meta::NLSMeta{T, S}
  # Counters of NLPModel
  counters::NLSGeneralCounters
  # For each observation i, cams_indices[i] gives the index of thecamera used for this observation
  cams_indices::Vector{Int}
  # For each observation i, pnts_indices[i] gives the index of the 3D point observed in this observation
  pnts_indices::Vector{Int}
  # Each line contains the 2D coordinates of the observed point
  pt2d::S
  # Number of observations
  nobs::Int
  # Number of points
  npnts::Int
  # Number of cameras
  ncams::Int

  # temporary storage for residual
  k::S
  P1::S

  # temporary storage for jacobian
  JProdP321::Matrix{T}
  JProdP32::Matrix{T}
  JP1_mat::Matrix{T}
  JP2_mat::Matrix{T}
  JP3_mat::Matrix{T}
  P1_vec::S
  P1_cross::S
  P2_vec::S

  # sample features
  sample::AbstractVector{<:Integer}
  epoch_counter::AbstractVector{<:Integer}
  sample_rate::T
  opt_counter::AbstractVector{<:Integer}
end

"""
    BundleAdjustmentModel(name::AbstractString; T::Type=Float64)

Constructor of BundleAdjustmentModel, creates an NLSModel with name `name` from a BundleAdjustment archive with precision `T`.
"""
function BAmodel_sto(name::AbstractString; T::Type = Float64, sample_rate = 1.0)
  filename = get_filename(name)
  filedir = fetch_ba_name(filename)
  path_and_filename = joinpath(filedir, filename)
  problem_name = filename[1:(end - 12)]

  cams_indices, pnts_indices, pt2d, x0, ncams, npnts, nobs = BundleAdjustmentModels.readfile(path_and_filename, T = T)

  S = typeof(x0)

  # variables: 9 parameters per camera + 3 coords per 3d point
  nvar = 9 * ncams + 3 * npnts
  # number of residuals: two residuals per 2d point
  nequ = 2 * nobs

  @debug "BundleAdjustmentModel $filename" nvar nequ

  meta = NLPModelMeta{T, S}(nvar, x0 = x0, name = problem_name)
  nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x0, nnzj = 2 * nobs * 12, nnzh = 0)

  k = similar(x0)
  P1 = similar(x0)

  JProdP321 = Matrix{T}(undef, 2, 12)
  JProdP32 = Matrix{T}(undef, 2, 6)
  JP1_mat = Matrix{T}(undef, 6, 12)
  JP2_mat = Matrix{T}(undef, 5, 6)
  JP3_mat = Matrix{T}(undef, 2, 5)
  P1_vec = S(undef, 3)
  P1_cross = S(undef, 3)
  P2_vec = S(undef, 2)

  sample_nobs = sort(randperm(nobs)[1:Int(ceil(sample_rate * nobs))])
  epoch_counter = [1]
  opt_counter = Int[]

  return SampledBAModel(
    meta,
    nls_meta,
    NLSGeneralCounters(),
    cams_indices,
    pnts_indices,
    pt2d,
    nobs,
    npnts,
    ncams,
    k,
    P1,
    JProdP321,
    JProdP32,
    JP1_mat,
    JP2_mat,
    JP3_mat,
    P1_vec,
    P1_cross,
    P2_vec,
    sample_nobs,
    epoch_counter,
    sample_rate,
    opt_counter,
  )
end

mutable struct SampledADNLSModel{T, S, Si} <: AbstractNLSModel{T, S}
  adnls::ADNLSModel{T, S, Si}
  ba::SampledBAModel{T, S}
end

function SADNLSModel(adnls::ADNLSModel{T, S, Si}, ba::SampledBAModel{T, S}) where {T, S, Si}
  return SampledADNLSModel(adnls, ba)
end

# API SampledADNLSModel #

function Base.getproperty(model::SampledADNLSModel, f::Symbol)
  if f in fieldnames(ADNLSModel)
    getfield(model.adnls, f)
  elseif f in fieldnames(SampledBAModel)
    getfield(model.ba, f)
  else
    getfield(model, f)
  end
end

function Base.setproperty!(model::SampledADNLSModel, f::Symbol, x)
  if f in fieldnames(ADNLSModel)
    setfield!(model.adnls, f, x)
  elseif f in fieldnames(SampledBAModel)
    setfield!(model.ba, f, x)
  else
    setfield!(model, f, x)
  end
end

function NLPModels.residual(model::SampledADNLSModel, x)
  return residual(model.adnls, x)
end

function NLPModels.residual!(model::SampledADNLSModel, x, Fx)
  return residual!(model.adnls, x, Fx)
end

#=function NLPModels.jprod_residual!(model::SampledADNLSModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  return jprod_residual!(model.adnls, rows, cols, vals, v, Jv)
end

function NLPModels.jtprod_residual!(model::SampledADNLSModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  return jtprod_residual!(model.adnls, rows, cols, vals, v, Jtv)
end

function NLPModels.jac_structure_residual!(nls::SampledADNLSModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  return jac_structure_residual!(nls.adnls, rows, cols)
end

function NLPModels.jac_coord_residual!(nls::SampledADNLSModel, x::AbstractVector, vals::AbstractVector)
  return jac_coord_residual!(nls.adnls, x, vals)
end

function NLPModels.jac_op_residual!(nls::SampledADNLSModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector, Jv::AbstractVector, Jtv::AbstractVector)
  jac_op_residual!(nls.adnls, rows, cols, vals, Jv, Jtv)
end=#

function NLPModels.residual!(nls::SampledBAModel, x::AbstractVector, rx::AbstractVector)
  increment!(nls, :neval_residual)
  residuals!(
    x,
    rx,
    nls.cams_indices,
    nls.pnts_indices,
    nls.nobs,
    nls.npnts,
    nls.k,
    nls.P1,
    nls.pt2d,
    nls.sample,
  )
  return rx
end

function residuals!(
  xs::AbstractVector,
  rxs::AbstractVector,
  cam_indices::Vector{Int},
  pnt_indices::Vector{Int},
  nobs::Int,
  npts::Int,
  ks::AbstractVector,
  Ps::AbstractVector,
  pt2d::AbstractVector,
  sample::AbstractVector,
)
  #@info "Length of the current sample = $(length(sample))"
  @simd for i in eachindex(sample)
    cam_index = cam_indices[sample[i]]
    pnt_index = pnt_indices[sample[i]]
    pnt_range = ((pnt_index - 1) * 3 + 1):((pnt_index - 1) * 3 + 3)
    cam_range = (3 * npts + (cam_index - 1) * 9 + 1):(3 * npts + (cam_index - 1) * 9 + 9)
    x = view(xs, pnt_range)
    c = view(xs, cam_range)
    r = view(rxs, (2 * i - 1):(2 * i))
    projection!(x, c, r)
  end
  for j in eachindex(sample)
    rxs[(2 * j - 1):(2 * j)] .-= pt2d[(2 * sample[j] - 1):(2 * sample[j])]
  end
  return rxs
end

function projection!(
  p3::AbstractVector,
  r::AbstractVector,
  t::AbstractVector,
  k_1,
  k_2,
  f,
  r2::AbstractVector,
)
  θ = sqrt(dot(r, r))

  k1 = r[1] / θ
  k2 = r[2] / θ
  k3 = r[3] / θ

  #cross!(P1, k, p3)
  P1_1 = k2 * p3[3] - k3 * p3[2]
  P1_2 = k3 * p3[1] - k1 * p3[3]
  P1_3 = k1 * p3[2] - k2 * p3[1]

  #P1 .*= sin(θ)
  P1_1 *= sin(θ)
  P1_2 *= sin(θ)
  P1_3 *= sin(θ)

  #P1 .+= cos(θ) .* p3 .+ (1 - cos(θ)) .* dot(k, p3) .* k .+ t
  kp3 = p3[1] * r[1] / θ + p3[2] * r[2] / θ + p3[3] * r[3] / θ # dot(k, p3)
  P1_1 += cos(θ) * p3[1] + (1 - cos(θ)) * kp3 * k1 + t[1]
  P1_2 += cos(θ) * p3[2] + (1 - cos(θ)) * kp3 * k2 + t[2]
  P1_3 += cos(θ) * p3[3] + (1 - cos(θ)) * kp3 * k3 + t[3]

  r2[1] = -P1_1 / P1_3
  r2[2] = -P1_2 / P1_3
  s = scaling_factor(r2, k_1, k_2)
  r2 .*= f * s
  return r2
end

projection!(x, c, r2) =
  projection!(x, view(c, 1:3), view(c, 4:6), c[7], c[8], c[9], r2)

function cross!(c::AbstractVector, a::AbstractVector, b::AbstractVector)
  if !(length(a) == length(b) == length(c) == 3)
    throw(DimensionMismatch("cross product is only defined for vectors of length 3"))
  end
  a1, a2, a3 = a
  b1, b2, b3 = b
  c[1] = a2 * b3 - a3 * b2
  c[2] = a3 * b1 - a1 * b3
  c[3] = a1 * b2 - a2 * b1
  c
end

function scaling_factor(point, k1, k2)
  sq_norm_point = dot(point, point)
  return 1 + sq_norm_point * (k1 + k2 * sq_norm_point)
end

  """
      increment!(nls, s)
  
  Increment counter `s` of problem `nls`.
  """
  @inline function increment!(nls::Union{SampledNLSModel, SampledBAModel}, s::Symbol)
    increment!(nls, Val(s))
  end
  
  for fun in fieldnames(NLSGeneralCounters)
    fun == :counters && continue
    @eval increment!(nls::Union{SampledNLSModel, SampledBAModel}, ::Val{$(Meta.quot(fun))}) = nls.counters.$fun += nls.sample_rate
  end
  
  for fun in fieldnames(NLSGeneralCounters)
    @eval $NLPModels.increment!(nls::Union{SampledNLSModel, SampledBAModel}, ::Val{$(Meta.quot(fun))}) =
      nls.counters.counters.$fun += nls.sample_rate
  end

  sum_counters(nls::Union{SampledNLSModel, SampledBAModel}) = NLPModels.sum_counters(nls.counters)

for counter in fieldnames(NLSGeneralCounters)
    counter == :counters && continue
    @eval begin
      """
      $($counter)(nlp)
  
      Get the number of `$(split("$($counter)", "_")[2])` evaluations.
      """
      $counter(nls::Union{SampledNLSModel, SampledBAModel}) = nls.counters.$counter
      export $counter
    end
  end
  
  for counter in fieldnames(NLSGeneralCounters)
    @eval begin
      $counter(nls::AbstractNLSModel) = nls.counters.counters.$counter
      export $counter
    end
end
  
  function LinearOperators.reset!(nls::Union{SampledNLSModel, SampledADNLSModel})
    reset!(nls.counters)
    return nls
  end
  
  function LinearOperators.reset!(nls_counters::NLSGeneralCounters)
    for f in fieldnames(NLSGeneralCounters)
      f == :counters && continue
      setfield!(nls_counters, f, 0.0)
    end
    NLPModels.reset!(nls_counters.counters)
    return nls_counters
  end

function NLPModels.residual!(
    nls::SampledNLSModel, 
    x::AbstractVector, 
    Fx::AbstractVector,
    )
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck length(nls.sample) Fx
  # increment the relative cost for a specified sample_rate
  #nls.counters.:neval_residual += Int(floor(100 * nls.sample_rate))
  increment!(nls, :neval_residual)
  #returns the sampled function Fx whose indexes are stored in sample without computing the other lines
  nls.resid!(Fx, x; sample = nls.sample)
  Fx
end

function NLPModels.residual(nls::SampledNLSModel{T, S, R, J, Jt}, x::AbstractVector{T}) where {T, S, R, J, Jt}
  @lencheck nls.meta.nvar x
  Fx = S(undef, length(nls.sample))
  residual!(nls, x, Fx)
end

function NLPModels.residual(nls::SampledBAModel{T, S}, x::AbstractVector{T}) where {T, S}
  @lencheck nls.meta.nvar x
  Fx = S(undef, nls.nls_meta.nequ)
  residual!(nls, x, Fx)
end

include("api-sampled-Jacobian.jl")

## API for SampledBAModel ##

function NLPModels.jac_structure_residual!(
  nls::SampledBAModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)  @simd for i in eachindex(nls.sample)
    idx_obs = (i - 1) * 24
    idx_cam = 3 * nls.npnts + 9 * (nls.cams_indices[nls.sample[i]] - 1)
    idx_pnt = 3 * (nls.pnts_indices[nls.sample[i]] - 1)

    # Only the two rows corresponding to the observation i are not empty
    p = 2 * i
    @views fill!(rows[(idx_obs + 1):(idx_obs + 12)], p - 1)
    @views fill!(rows[(idx_obs + 13):(idx_obs + 24)], p)

    # 3 columns for the 3D point observed
    @inbounds cols[(idx_obs + 1):(idx_obs + 3)] .= (idx_pnt + 1):(idx_pnt + 3)
    # 9 columns for the camera
    @inbounds cols[(idx_obs + 4):(idx_obs + 12)] .= (idx_cam + 1):(idx_cam + 9)
    # 3 columns for the 3D point observed
    @inbounds cols[(idx_obs + 13):(idx_obs + 15)] .= (idx_pnt + 1):(idx_pnt + 3)
    # 9 columns for the camera
    @inbounds cols[(idx_obs + 16):(idx_obs + 24)] .= (idx_cam + 1):(idx_cam + 9)
  end
  return rows, cols
end

function NLPModels.jac_coord_residual!(
  nls::SampledBAModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  increment!(nls, :neval_jac_residual)
  T = eltype(x)

  fill!(nls.JP1_mat, zero(T))
  nls.JP1_mat[1, 7], nls.JP1_mat[2, 8], nls.JP1_mat[3, 9] = 1, 1, 1
  nls.JP1_mat[4, 10], nls.JP1_mat[5, 11], nls.JP1_mat[6, 12] = 1, 1, 1

  fill!(nls.JP2_mat, zero(T))
  nls.JP2_mat[3, 4], nls.JP2_mat[4, 5], nls.JP2_mat[5, 6] = 1, 1, 1

  @simd for i in eachindex(nls.sample)
    idx_cam = nls.cams_indices[nls.sample[i]]
    idx_pnt = nls.pnts_indices[nls.sample[i]]
    @views X = x[((idx_pnt - 1) * 3 + 1):((idx_pnt - 1) * 3 + 3)] # 3D point coordinates
    @views C = x[(3 * nls.npnts + (idx_cam - 1) * 9 + 1):(3 * nls.npnts + (idx_cam - 1) * 9 + 9)] # camera parameters
    @views r = C[1:3] # is the Rodrigues vector for the rotation
    @views t = C[4:6] # is the translation vector
    # k1, k2, f = C[7:9] is the focal length and radial distortion factors

    # JProdP321 = JP3∘P2∘P1 x JP2∘P1 x JP1
    P1!(r, t, X, nls.P1_vec, nls.P1_cross)
    P2!(nls.P1_vec, nls.P2_vec)
    JP2!(nls.JP2_mat, nls.P1_vec)
    JP1!(nls.JP1_mat, r, X, nls.P1_vec)
    JP3!(nls.JP3_mat, nls.P2_vec, C[9], C[7], C[8])
    mul!(nls.JProdP32, nls.JP3_mat, nls.JP2_mat)
    mul!(nls.JProdP321, nls.JProdP32, nls.JP1_mat)

    # Fill vals with the values of JProdP321 = [[∂P.x/∂X ∂P.x/∂C], [∂P.y/∂X ∂P.y/∂C]]
    # If a value is NaN, we put it to 0 not to take it into account
    replace!(nls.JProdP321, NaN => zero(T))
    @views vals[((i - 1) * 24 + 1):((i - 1) * 24 + 24)] = nls.JProdP321'[:]
  end
  return vals
end

function NLPModels.jac_op_residual!(
  nls::SampledBAModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck length(rows) rows cols vals
  @lencheck nls.nls_meta.nequ Jv
  @lencheck nls.meta.nvar Jtv
  prod! = @closure (res, v, α, β) -> begin
    jprod_residual!(nls, rows, cols, vals, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_residual!(nls, rows, cols, vals, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{eltype(vals)}(
    nls_meta(nls).nequ,
    nls_meta(nls).nvar,
    false,
    false,
    prod!,
    ctprod!,
    ctprod!,
  )
end

"""
    coo_prod!(rows, cols, vals, v, Av)

Compute the product of a matrix `A` given by `(rows, cols, vals)` and the vector `v`.
The result is stored in `Av`, which should have length equals to the number of rows of `A`.
"""
function NLPModels.coo_prod!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Av::AbstractVector,
)
  fill!(Av, zero(eltype(v)))
  nnz = length(rows)
  for k = 1:nnz
    i, j = rows[k], cols[k]
    Av[i] += vals[k] * v[j]
  end
  return Av
end

"""
    Jv = jprod_residual!(nls, rows, cols, vals, v, Jv)

Computes the product of the Jacobian of the residual given by `(rows, cols, vals)`
and a vector, i.e.,  ``J(x)v``, storing it in `Jv`.
"""
#=function NLPModels.jprod_residual!(
  nls::SampledBAModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck length(rows) rows cols vals
  @lencheck nls.meta.nvar v
  @lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  coo_prod!(rows, cols, vals, v, Jv)
end=#

"""
    Jtv = jtprod_residual!(nls, rows, cols, vals, v, Jtv)

Computes the product of the transpose of the Jacobian of the residual given by `(rows, cols, vals)`
and a vector, i.e.,  ``J(x)^Tv``, storing it in `Jv`.
"""
#=function NLPModels.jtprod_residual!(
  nls::SampledBAModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck length(rows) rows cols vals
  @lencheck nls.nls_meta.nequ v
  @lencheck nls.meta.nvar Jtv
  increment!(nls, :neval_jtprod_residual)
  coo_prod!(cols, rows, vals, v, Jtv)
end=#
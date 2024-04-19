export ROSolverOptions

mutable struct ROSolverOptions{R}
  ϵa::R  # termination criteria
  ϵr::R  # relative stopping tolerance
  neg_tol::R # tolerance when ξ < 0
  Δk::R  # trust region radius
  verbose::Int  # print every so often
  maxIter::Int  # maximum amount of inner iterations
  maxTime::Float64 #maximum time allotted to the algorithm in s
  σmin::R # minimum σk allowed for LM/R2 method
  μmin::R # minimum μk allowed for Sampled Methods - NEW -
  η1::R  # step acceptance threshold
  η2::R  # trust-region increase threshold
  η3::R #Stochastic metric threshold - NEW -
  α::R  # νk Δ^{-1} parameter
  ν::R  # initial guess for step length
  νcp::R #Initial guess for step length of Cauchy Point computation - NEW -
  γ::R  # trust region buffer
  θ::R  # step length factor in relation to Hessian norm
  β::R  # TR size as factor of first PG step
  λ::R # parameter of the random stepwalk - NEW -
  metric::R #parameter of the stationnarity metric of the algorithm - NEW -
  spectral::Bool # for TRDH: use spectral gradient update if true, otherwise DiagonalQN
  psb::Bool # for TRDH with DiagonalQN (spectral = false): use PSB update if true, otherwise Andrei update
  reduce_TR::Bool

  function ROSolverOptions{R}(;
    ϵa::R = √eps(R),
    ϵr::R = √eps(R),
    neg_tol::R = eps(R)^(1 / 4),
    Δk::R = one(R),
    verbose::Int = 0,
    maxIter::Int = 50,
    maxTime::Float64 = 120.0,
    σmin::R = eps(R),
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
  ) where {R <: Real}
    @assert ϵa ≥ 0
    @assert ϵr ≥ 0
    @assert neg_tol ≥ 0
    @assert Δk > 0
    @assert verbose ≥ 0
    @assert maxIter ≥ 0
    @assert maxTime ≥ 0
    @assert σmin ≥ 0
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
    return new{R}(
      ϵa,
      ϵr,
      neg_tol,
      Δk,
      verbose,
      maxIter,
      maxTime,
      σmin,
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
    )
  end
end

ROSolverOptions(args...; kwargs...) = ROSolverOptions{Float64}(args...; kwargs...)

mutable struct SampledNLSModel{T, S, R, J, Jt} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters

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
    return new{T, S, R, J, Jt}(meta, nls_meta, NLSCounters(), r, jv, jtv, sample, data_mem, sample_rate, epoch_counter, opt_counter)
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

function NLPModels.residual!(
    nls::SampledNLSModel, 
    x::AbstractVector, 
    Fx::AbstractVector,
    )
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck length(nls.sample) Fx
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

function NLPModels.jprod_residual!(
  nls::SampledNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x v
  NLPModels.@lencheck length(nls.sample) Jv
  increment!(nls, :neval_jprod_residual)
  nls.jprod_resid!(Jv, x, v; sample = nls.sample)
  #@assert Jv == Jv[sort(randperm(nls.nls_meta.nequ)[1:Int(1.0 * nls.nls_meta.nequ)])]
  Jv
end

function NLPModels.jtprod_residual!(
  nls::SampledNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x Jtv
  NLPModels.@lencheck length(nls.sample) v
  increment!(nls, :neval_jtprod_residual)
  nls.jtprod_resid!(Jtv, x, v; sample = nls.sample)
  #@assert Jtv == Jtv[sort(randperm(nls.nls_meta.nequ)[1:Int(1.0 * nls.nls_meta.nequ)])]
  Jtv
end

function NLPModels.jac_op_residual!(
  nls::SampledNLSModel,
  x::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck length(nls.sample) Jv

  prod! = @closure (res, v, α, β) -> begin
    jprod_residual!(nls, x, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_residual!(nls, x, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end

  return LinearOperator{eltype(x)}(
    length(nls.sample),
    nls_meta(nls).nvar,
    false,
    false,
    prod!,
    ctprod!,
    ctprod!,
  )
end

function NLPModels.jac_op_residual(nls::SampledNLSModel{T, S, R, J, Jt}, x::AbstractVector{T}) where {T, S, R, J, Jt}
  @lencheck nls.meta.nvar x
  Jv = S(undef, length(nls.sample))
  Jtv = S(undef, nls.meta.nvar)
  return NLPModels.jac_op_residual!(nls, x, Jv, Jtv)
end
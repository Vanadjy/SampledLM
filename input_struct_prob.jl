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
  sample_size::Int
  epoch_counter::AbstractVector{<:Integer}

  function SampledNLSModel{T, S, R, J, Jt}(
    r::R,
    jv::J,
    jtv::Jt,
    nequ::Int,
    x::S,
    sample::AbstractVector{<:Integer},
    data_mem::AbstractVector{<:Integer},
    sample_rate::Real,
    sample_size::Int,
    epoch_counter::AbstractVector{<:Integer};
    kwargs...,
  ) where {T, S, R <: Function, J <: Function, Jt <: Function}
    nvar = length(x)
    meta = NLPModelMeta(nvar, x0 = x; kwargs...)
    nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x)
    return new{T, S, R, J, Jt}(meta, nls_meta, NLSCounters(), r, jv, jtv, sample, data_mem, sample_rate, sample_size, epoch_counter)
  end
end

SampledNLSModel(r, jv, jtv, nequ::Int, x::S, sample::AbstractVector{<:Integer}, data_mem::AbstractVector{<:Integer}, sample_rate::Real, sample_size::Int, epoch_counter::AbstractVector{<:Integer}; kwargs...) where {S} =
SampledNLSModel{eltype(S), S, typeof(r), typeof(jv), typeof(jtv)}(
    r,
    jv,
    jtv,
    nequ,
    x,
    sample,
    data_mem,
    sample_rate,
    sample_size,
    epoch_counter;
    kwargs...,
  )

function NLPModels.residual!(
    nls::SampledNLSModel, 
    x::AbstractVector, 
    Fx::AbstractVector,
    )
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck nls.nls_meta.nequ Fx
  #sample = sort(randperm(nls.nls_meta.nequ)[1:Int(sample_rate * nls.nls_meta.nequ)])
  increment!(nls, :neval_residual)
  #the next function should return the sampled function Fx whose indexes are stored in sample without computing the other lines
  #TODO : faire en sorte que les indices de calcul de nls.resid! soient parcouru avec "for i in sample" au lieu de parcourir tous les indices.
  nls.resid!(Fx, x; sample = nls.sample, sample_size = nls.sample_size)
  Fx
end

function NLPModels.residual(nls::AbstractNLSModel{T, S}, x::AbstractVector{T}) where {T, S}
  @lencheck nls.meta.nvar x
  Fx = S(undef, nls.nls_meta.nequ)
  residual!(nls, x, Fx)
end

function NLPModels.jprod_residual!(
  nls::SampledNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x v
  NLPModels.@lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  nls.jprod_resid!(Jv, x, v; sample = nls.sample, sample_size = nls.sample_size)
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
  NLPModels.@lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_jtprod_residual)
  nls.jtprod_resid!(Jtv, x, v; sample = nls.sample, sample_size = nls.sample_size)
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
  @lencheck nls.nls_meta.nequ Jv

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
    nls.nls_meta.nequ,
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
  Jv = S(undef, nls.nls_meta.nequ)
  Jtv = S(undef, nls.meta.nvar)
  return NLPModels.jac_op_residual!(nls, x, Jv, Jtv)
end
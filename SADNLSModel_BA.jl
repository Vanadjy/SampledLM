"""
    SampledADNLSModel_BA

Strucure wrapping an ADNLSModel and a SampledBAModel to use Automatic Differentiation backend on sampled Bundle Adjustment problems.
"""
mutable struct SampledADNLSModel_BA{T, S, Si} <: AbstractNLSModel{T, S}
  adnls::ADNLSModel{T, S, Si}
  ba::SampledBAModel{T, S}
end

"""
    SADNLSModel_BA

Constructor of a Sampled ADNLSModel taking in arguments an ADNLSModel and a SampledBAModel.
"""
function SADNLSModel_BA(adnls::ADNLSModel{T, S, Si}, ba::SampledBAModel{T, S}) where {T, S, Si}
  return SampledADNLSModel_BA(adnls, ba)
end

# API SampledADNLSModel_BA #

function Base.getproperty(model::SampledADNLSModel_BA, f::Symbol)
  if f in fieldnames(ADNLSModel)
    getfield(model.adnls, f)
  elseif f in fieldnames(SampledBAModel)
    getfield(model.ba, f)
  else
    getfield(model, f)
  end
end

function Base.setproperty!(model::SampledADNLSModel_BA, f::Symbol, x)
  if f in fieldnames(ADNLSModel)
    setfield!(model.adnls, f, x)
  elseif f in fieldnames(SampledBAModel)
    setfield!(model.ba, f, x)
  else
    setfield!(model, f, x)
  end
end

function NLPModels.residual(model::SampledADNLSModel_BA{T, S}, x::AbstractVector{T}) where {T, S}
  return model.adnls.F!(x)
end

function NLPModels.residual!(model::SampledADNLSModel_BA{T, S}, x::AbstractVector{T}, Fx::AbstractVector{T}) where {T, S}
  return model.adnls.F!(Fx, x)
end

function NLPModels.jprod_residual!(
  nls::SampledADNLSModel_BA,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nls.meta.nvar x v
  @lencheck 2*length(nls.sample) Jv
  increment!(nls.ba, :neval_jprod_residual)
  F = nls.adnls.F!
  ADNLPModels.Jprod!(nls.adnls.adbackend.jprod_residual_backend, Jv, F, x, v, Val(:F))
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::SampledADNLSModel_BA,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nls.meta.nvar x Jtv
  @lencheck 2*length(nls.sample) v
  increment!(nls.ba, :neval_jtprod_residual)
  F = nls.adnls.F!
  b = nls.adnls.adbackend.jtprod_residual_backend
  ADNLPModels.Jtprod!(b, Jtv, F, x, v, Val(:F))
  return Jtv
end
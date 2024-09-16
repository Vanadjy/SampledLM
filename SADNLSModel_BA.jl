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
  return residual(model.adnls, x)
end

function NLPModels.residual!(model::SampledADNLSModel_BA{T, S}, x::AbstractVector{T}, Fx::AbstractVector{T}) where {T, S}
  return residual!(model.adnls, x, Fx)
end

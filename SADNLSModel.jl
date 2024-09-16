"""
    SampledADNLSModel

Strucure wrapping an ADNLSModel and a SampledNLSModel to use Automatic Differentiation backend on sampled Bundle Adjustment problems.
"""
mutable struct SampledADNLSModel{T, S, Si, R, J, Jt} <: AbstractNLSModel{T, S}
  adnls::ADNLSModel{T, S, Si}
  snls::SampledNLSModel{T, S, R, J, Jt}
end

"""
    SADNLSModel

Constructor of a Sampled ADNLSModel taking in arguments an ADNLSModel and a SampledBAModel.
"""
function SADNLSModel(adnls::ADNLSModel{T, S, Si}, snls::SampledNLSModel{T, S, R, J, Jt}) where {T, S, Si, R, J, Jt}
  return SampledADNLSModel(adnls, snls)
end

# API SampledADNLSModel #

function Base.getproperty(model::SampledADNLSModel, f::Symbol)
  if f in fieldnames(ADNLSModel)
    getfield(model.adnls, f)
  elseif f in fieldnames(SampledNLSModel)
    getfield(model.ba, f)
  else
    getfield(model, f)
  end
end

function Base.setproperty!(model::SampledADNLSModel, f::Symbol, x)
  if f in fieldnames(ADNLSModel)
    setfield!(model.adnls, f, x)
  elseif f in fieldnames(SampledNLSModel)
    setfield!(model.ba, f, x)
  else
    setfield!(model, f, x)
  end
end

function NLPModels.residual(model::SampledADNLSModel{T, S, Si, R, J, Jt}, x::AbstractVector{T}) where {T, S, Si, R, J, Jt}
  return residual(model.snls, x)
end

function NLPModels.residual!(model::SampledADNLSModel{T, S, Si, R, J, Jt}, x::AbstractVector{T}, Fx::AbstractVector{T}) where {T, S, Si, R, J, Jt}
  return residual!(model.snls, x, Fx)
end
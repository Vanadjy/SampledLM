mutable struct ProbNLSModel{T, S, R, J, Jt} <: AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters

  resid!::R
  jprod_resid!::J
  jtprod_resid!::Jt

  #probabilistic parameters
  sampler::AbstractVector{<:Integer}
  sampler_size::Int

  function ProbNLSModel{T, S, R, J, Jt}(
    r::R,
    jv::J,
    jtv::Jt,
    nequ::Int,
    x::S,
    sampler::AbstractVector{<:Integer},
    sampler_size::Int;
    kwargs...,
  ) where {T, S, R <: Function, J <: Function, Jt <: Function}
    nvar = length(x)
    meta = NLPModelMeta(nvar, x0 = x; kwargs...)
    nls_meta = NLSMeta{T, S}(nequ, nvar, x0 = x)
    return new{T, S, R, J, Jt}(meta, nls_meta, NLSCounters(), r, jv, jtv, sampler, sampler_size)
  end
end

ProbNLSModel(r, jv, jtv, nequ::Int, x::S, sampler::AbstractVector{<:Integer}, sampler_size::Int; kwargs...) where {S} =
ProbNLSModel{eltype(S), S, typeof(r), typeof(jv), typeof(jtv)}(
    r,
    jv,
    jtv,
    nequ,
    x,
    sampler,
    sampler_size;
    kwargs...,
  )

function NLPModels.residual!(
    nls::ProbNLSModel, 
    x::AbstractVector, 
    Fx::AbstractVector,
    )
  NLPModels.@lencheck nls.meta.nvar x
  NLPModels.@lencheck nls.nls_meta.nequ Fx
  increment!(nls, :neval_residual)
  #the next function should return the sampled function Fx whose indexes are stored in sampler without computing the other lines
  #TODO : faire en sorte que les indices de calcul de nls.resid! soient parcouru avec "for i in sampler" au lieu de parcourir tous les indices.
  nls.resid!(Fx, x; sampler = nls.sampler)
  Fx
end

function NLPModels.jprod_residual!(
  nls::ProbNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x v
  NLPModels.@lencheck nls.nls_meta.nequ Jv
  increment!(nls, :neval_jprod_residual)
  nls.jprod_resid!(Jv, x, v; sampler = nls.sampler)
  #@assert Jv == Jv[sort(randperm(nls.nls_meta.nequ)[1:Int(1.0 * nls.nls_meta.nequ)])]
  Jv
end

function NLPModels.jtprod_residual!(
  nls::ProbNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.@lencheck nls.meta.nvar x Jtv
  NLPModels.@lencheck nls.nls_meta.nequ v
  increment!(nls, :neval_jtprod_residual)
  nls.jtprod_resid!(Jtv, x, v; sampler = nls.sampler)
  #@assert Jtv == Jtv[sort(randperm(nls.nls_meta.nequ)[1:Int(1.0 * nls.nls_meta.nequ)])]
  Jtv
end
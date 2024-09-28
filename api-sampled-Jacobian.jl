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

  #=function NLPModels.jac_op_residual!(
    nls::SampledADNLSModel_BA,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
    vals::AbstractVector,
    Jv::AbstractVector,
    Jtv::AbstractVector,
  )
    @lencheck length(rows) rows cols vals
    @lencheck 2*length(nls.sample) Jv
    @lencheck nls.meta.nvar Jtv
    prod! = @closure (res, v, α, β) -> begin
      jprod_residual!(nls.adnls, rows, cols, vals, v, Jv)
      if β == 0
        @. res = α * Jv
      else
        @. res = α * Jv + β * res
      end
      return res
    end
    ctprod! = @closure (res, v, α, β) -> begin
      jtprod_residual!(nls.adnls, rows, cols, vals, v, Jtv)
      if β == 0
        @. res = α * Jtv
      else
        @. res = α * Jtv + β * res
      end
      return res
    end
    return LinearOperator{eltype(vals)}(
      2*length(nls.sample),
      nls_meta(nls).nvar,
      false,
      false,
      prod!,
      ctprod!,
      ctprod!,
    )
  end=#
  
  function NLPModels.jac_op_residual(nls::SampledNLSModel{T, S, R, J, Jt}, x::AbstractVector{T}) where {T, S, R, J, Jt}
    @lencheck nls.meta.nvar x
    Jv = S(undef, length(nls.sample))
    Jtv = S(undef, nls.meta.nvar)
    return NLPModels.jac_op_residual!(nls, x, Jv, Jtv)
  end
  
  #function NLPModels.jac_structure_residual! end
  
  #sp_sample must be a sample adapted to the sparse structure of the Jacobian, obtained with sp_sample(rows::AbstractVector{T}, sample::AbstractVector{<:Integer}) where {T} function
  function NLPModels.jac_structure_residual(nls::SampledNLSModel{T, S, R, J, Jt}, sp_sample::AbstractVector{<:Integer}) where {T, S, R, J, Jt}
    rows = Vector{Int}(undef, nls.nls_meta.nnzj)
    cols = Vector{Int}(undef, nls.nls_meta.nnzj)
    jac_structure_residual!(nls, rows, cols)
    rows[sp_sample], cols[sp_sample]
  end
  
  #function NLPModels.jac_coord_residual! end
  
  function jac_coord_residual(nls::SampledNLSModel{T, S, R, J, Jt}, sp_sample::AbstractVector{<:Integer}) where {T, S, R, J, Jt}
    @lencheck nls.meta.nvar x
    vals = S(undef, nls.nls_meta.nnzj)
    jac_coord_residual!(nls, x, vals)
    vals[sp_sample]
  end
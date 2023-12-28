#export bpdn_model, bpdn_nls_model

function bpdn_data_sto(m::Int, n::Int, k::Int, noise::Float64 = 0.01; bounds::Bool = false)
  m ≤ n || error("number of rows ($m) should be ≤ number of columns ($n)")
  x0 = zeros(n)
  p = randperm(n)[1:k]
  # create sparse signal
  if bounds
    x0[p[1:k]] .= 1
  else
    x0[p[1:k]] = sign.(randn(k))
  end
  Q, _ = qr(randn(n, m))
  A = Array(Array(Q)')
  b0 = A * x0
  b = b0 + noise * randn(m)

  A, b, b0, x0
end

bpdn_data_sto(compound::Int = 1, args...; bounds::Bool = false) =
bpdn_data_sto(200 * compound, 512 * compound, 10 * compound, args...; bounds = bounds)

"""
    model, nls_model, sol = bpdn_model(args...)
    model, nls_model, sol = bpdn_model(compound = 1, args...)

Return an instance of an `NLPModel` and an instance of an `NLSModel` representing
the same basis-pursuit denoise problem, i.e., the under-determined linear
least-squares objective

   ½ ‖Ax - b‖₂²,

where A has orthonormal rows and b = A * x̄ + ϵ, x̄ is sparse and ϵ is a noise
vector following a normal distribution with mean zero and standard deviation σ.

## Arguments

* `m :: Int`: the number of rows of A
* `n :: Int`: the number of columns of A (with `n` ≥ `m`)
* `k :: Int`: the number of nonzero elements in x̄
* `noise :: Float64`: noise standard deviation σ (default: 0.01).

The second form calls the first form with arguments

    m = 200 * compound
    n = 512 * compound
    k =  10 * compound

## Keyword arguments

* `bounds :: Bool`: whether or not to include nonnegativity bounds in the model (default: false).

## Return Value

An instance of a `FirstOrderModel` and of a `FirstOrderNLSModel` that represent the same
basis-pursuit denoise problem, and the exact solution x̄.

If `bounds == true`, the positive part of x̄ is returned.
"""
function bpdn_model_sto(compound::Int = 1, args... ; bounds::Bool = false, sample_rate::AbstractFloat = 1.0)
  A, b, b0, x0 = bpdn_data_sto(args...; bounds = bounds)

  #initializes sampling parameters
  sample = sort(randperm(size(A,1))[1:Int(sample_rate * size(A,1))])
  r = similar(b[1:length(sample)])

  function resid!(r, x; sample = sample)
    mul!(r, A[sample, :], x)
    r .-= b[sample]
    r
  end

  jprod_resid!(Jv, x, v; sample = sample) = mul!(Jv, A[sample, :], v) #Jv must be of length mₛ ad v of length n
  jtprod_resid!(Jtv, x, v; sample = sample) = mul!(Jtv, A[sample, :]', v) #v must be of length mₛ and Jtv of length n

  
  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end

  function grad!(g, x)
    resid!(r, x)
    #mul!(g, A', r)
    g
  end

  nlpmodel_kwargs = Dict{Symbol, Any}(:name => bounds ? "BPDNpos" : "BPDN")
  nlsmodel_kwargs = Dict{Symbol, Any}(:name => bounds ? "BPDN-LS_pos" : "BPDN-LS")
  if bounds
    nlpmodel_kwargs[:lvar] = zero(x0)
    nlpmodel_kwargs[:uvar] = fill!(similar(x0), Inf)
    nlsmodel_kwargs[:lvar] = zero(x0)
    nlsmodel_kwargs[:uvar] = fill!(similar(x0), Inf)
    x0[x0 .< 0] .= 0
  end

  FirstOrderModel(obj, grad!, zero(x0); nlpmodel_kwargs...),
  SampledNLSModel(resid!, jprod_resid!, jtprod_resid!, size(A, 1), zero(x0), sample; nlsmodel_kwargs...),
  x0
end
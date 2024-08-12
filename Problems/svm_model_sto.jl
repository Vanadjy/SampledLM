export svm_model_sto, ijcnn1_model_sto

"""
    nlp_model, sampled_nls_model, sol = svm_model_sto(args...)

Return an instance of an `NLPModel` representing the hyperbolic SVM
problem, i.e., the under-determined linear least-squares objective

   f(x) = ‖1 - tanh(b ⊙ ⟨A, x⟩)‖²,

where A is the data matrix with labels b = {-1, 1}ⁿ.

## Arguments

* `A :: Matrix{Float64}`: the data matrix
* `b :: Vector{Float64}`: the labels

With the IJCNN1 Dataset, the dimensions are:

    m = 49990
    n = 22

## Return Value

An instance of a `FirstOrderModel` that represents the complete SVM problem in NLP form, and
an instance of `SampledNLSModel` that represents the nonlinear least squares in nonlinear least squares form
with an associated sample on the number of equations.
"""

function svm_model_sto(A, b; sample_rate::AbstractFloat = 1.0)
  Ahat = Diagonal(b) * A' #dimensions : m × n

  #initializes sampling parameters
  sample = sort(randperm(size(Ahat,1))[1:Int(ceil(sample_rate * size(Ahat,1)))])
  data_mem = copy(sample)
  r = similar(b[1:length(sample)])
  tmp = similar(r)

  function resid!(r, x; sample = sample)
    mul!(r, Ahat[sample, :], x)
    r .= 1 .- tanh.(r)
    r
  end

  function jacv!(Jv, x, v; sample = sample)
    r = similar(b[1:length(sample)])
    mul!(r, Ahat[sample, :], x)
    mul!(Jv, Ahat[sample, :], v)
    Jv .= -((sech.(r)) .^ 2) .* Jv
  end

  function jactv!(Jtv, x, v; sample = sample)
    r = similar(b[1:length(sample)])
    tmp = similar(r)
    mul!(r, Ahat[sample, :], x)
    tmp .= sech.(r) .^ 2
    tmp .*= v
    tmp .*= -1
    mul!(Jtv, Ahat[sample, :]', tmp)
  end

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end
  function grad!(g, x; sample = sample)
    mul!(r, Ahat[sample, :], x)
    tmp .= (sech.(r)) .^ 2
    tmp .*= (1 .- tanh.(r))
    tmp .*= -1
    mul!(g, Ahat[sample, :]', tmp)
    g
  end

  FirstOrderModel(obj, grad!, ones(size(A, 1)), name = "Nonlinear-SVM"),
  SampledNLSModel(resid!, jacv!, jactv!, length(b), ones(size(A, 1)), sample, data_mem, sample_rate),
  b
end

function ijcnn1_model_sto(sample_rate)
  A, b = ijcnn1_generate_data_tr()
  return svm_model_sto(A', b; sample_rate = sample_rate)
end
function tan_data_train(args...)
    #load data
    A, b = MLDatasets.MNIST(split = :train)[:]
    A, b = generate_data(A, b, args...)
    return A, b
end

function tan_data_test(args...)
    A, b = MLDatasets.MNIST(split = :test)[:]
    A, b = generate_data(A, b, args...)
    return A, b
end

function generate_data(A, b, digits::Tuple{Int, Int} = (1, 7), switch::Bool = false)
    length(digits) == 2 || error("please supply two digits only")
    digits[1] != digits[2] || error("please supply two different digits")
    all(0 .≤ digits .≤ 9) || error("please supply digits from 0 to 9")
    ind = findall(x -> x ∈ digits, b)
    #reshape to matrix
    A = reshape(A, size(A, 1) * size(A, 2), size(A, 3)) ./ 255
  
    #get 0s and 1s
    b = float.(b[ind])
    b[b .== digits[2]] .= -1
    A = convert(Array{Float64, 2}, A[:, ind])
    if switch
      p = randperm(length(b))[1:Int(floor(length(b) / 3))]
      b = b[p]
      A = A[:, p]
    end
    return A, b
end

function MNIST_test_model_sto(sample_rate; digits::Tuple{Int, Int} = (1, 7), switch::Bool = false)
    A, b = tan_data_test(digits, switch)
    nlp, nls, labels = svm_model_sto(A, b; sample_rate = sample_rate)

    nlp,
    nls,
    labels
end

function MNIST_train_model_sto(sample_rate; digits::Tuple{Int, Int} = (1, 7), switch::Bool = false)
    A, b = tan_data_train(digits, switch)
    nlp, nls, labels = svm_model_sto(A, b; sample_rate = sample_rate)

    nlp,
    nls,
    labels
end

function NLPModels.jac_residual(nls::SampledNLSModel)
    increment!(nls, :neval_jac_residual)
    A, b = MLDatasets.MNIST(split = :train)[:]
    A, b = generate_data(A, b)
    Ahat = Diagonal(b) * A'
    B = tanh.(Ahat * nls.meta.x0)
    rows, cols, vals = findnz(convert(SparseMatrixCSC, Diagonal(B) * Ahat))
    return rows, cols, vals
end

#=A, b = tan_data_train((1,7), false)
#display("Percentage of zero values in A: $(count(<=(1e-16), A) / (size(A, 1) * size(A, 2)))")
#display((size(A, 1) * size(A, 2)) - (count(<=(1e-16), A)))
mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(1.0)
adnls = ADNLSModel!(mnist_nls.resid!, mnist_nls.meta.x0,  mnist_nls.nls_meta.nequ, mnist_nls.meta.lvar, mnist_nls.meta.uvar, jacobian_residual_backend = ADNLPModels.SparseADJacobian,
            jacobian_backend = ADNLPModels.EmptyADbackend,
            hessian_backend = ADNLPModels.EmptyADbackend,
            hessian_residual_backend = ADNLPModels.EmptyADbackend,
            matrix_free = true
)

rows = Vector{Int}(undef, mnist_nls.nls_meta.nnzj)
cols = Vector{Int}(undef, mnist_nls.nls_meta.nnzj)
vals = ones(Bool, mnist_nls.nls_meta.nnzj)
jac_structure_residual!(adnls, rows, cols)
J = sparse(rows, cols, vals, mnist_nls.nls_meta.nequ, mnist_nls.meta.nvar)=#
using BundleAdjustmentModels
using ADNLPModels, NLPModels

# Generating Bundle Adjustment model
name = "problem-16-22106-pre"
nls = BundleAdjustmentModel(name)

# Sampling utils
function row_sample_bam(sample::AbstractVector{<:Integer})
    aux = [vcat(2*i-1, 2*i) for i in sample]
    row_sample_ba = Int[]
    for elt in aux row_sample_ba = vcat(row_sample_ba, elt) end
    return row_sample_ba
end

sample_rate = 1.0
sample = sort(randperm(nls.nobs)[1:Int(ceil(sample_rate * nls.nobs))]) # random sample of sample_rate% of observations of nls
row_sample_ba = row_sample_bam(sample)

function F!(Fx, x)
    residual!(nls, x, Fx, sample)
end

#=adnls = ADNLSModel!(F!, nls.meta.x0,  nls.nls_meta.nequ, nls.meta.lvar, nls.meta.uvar, 
    jacobian_residual_backend = ADNLPModels.SparseADJacobian,
    jprod_residual_backend = ADNLPModels.ForwardDiffADJprod,
    jtprod_residual_backend = ADNLPModels.ReverseDiffADJtprod,
    jacobian_backend = ADNLPModels.EmptyADbackend,
    hessian_backend = ADNLPModels.EmptyADbackend,
    hessian_residual_backend = ADNLPModels.EmptyADbackend,
    matrix_free = true
)=#

function NLPModels.residual!(nls::BundleAdjustmentModel, x::AbstractVector, rx::AbstractVector, sample::Vector{<:Integer})
    #increment!(nls, :neval_residual)
    residuals!(
      x,
      rx,
      nls.cams_indices,
      nls.pnts_indices,
      nls.nobs,
      nls.npnts,
      nls.k,
      nls.P1,
      nls.pt2d,
      sample,
    )
    return rx
end

function residuals!(
    xs::AbstractVector,
    rxs::AbstractVector,
    cam_indices::Vector{Int},
    pnt_indices::Vector{Int},
    nobs::Int,
    npts::Int,
    ks::AbstractVector,
    Ps::AbstractVector,
    pt2d::AbstractVector,
    sample::AbstractVector,
  )
    @simd for i in eachindex(sample)
      cam_index = cam_indices[sample[i]]
      pnt_index = pnt_indices[sample[i]]
      pnt_range = ((pnt_index - 1) * 3 + 1):((pnt_index - 1) * 3 + 3)
      cam_range = (3 * npts + (cam_index - 1) * 9 + 1):(3 * npts + (cam_index - 1) * 9 + 9)
      x = view(xs, pnt_range)
      c = view(xs, cam_range)
      r = view(rxs, (2 * i - 1):(2 * i))
      projection!(x, c, r)
    end
    for j in eachindex(sample)
      rxs[(2 * j - 1):(2 * j)] .-= pt2d[(2 * sample[j] - 1):(2 * sample[j])]
    end
    return rxs
end

#=function projection!(
    p3::AbstractVector,
    r::AbstractVector,
    t::AbstractVector,
    k_1,
    k_2,
    f,
    r2::AbstractVector,
  )
    θ = sqrt(dot(r, r))
  
    k1 = r[1] / θ
    k2 = r[2] / θ
    k3 = r[3] / θ
  
    #cross!(P1, k, p3)
    P1_1 = k2 * p3[3] - k3 * p3[2]
    P1_2 = k3 * p3[1] - k1 * p3[3]
    P1_3 = k1 * p3[2] - k2 * p3[1]
  
    #P1 .*= sin(θ)
    P1_1 *= sin(θ)
    P1_2 *= sin(θ)
    P1_3 *= sin(θ)
  
    #P1 .+= cos(θ) .* p3 .+ (1 - cos(θ)) .* dot(k, p3) .* k .+ t
    kp3 = p3[1] * r[1] / θ + p3[2] * r[2] / θ + p3[3] * r[3] / θ # dot(k, p3)
    P1_1 += cos(θ) * p3[1] + (1 - cos(θ)) * kp3 * k1 + t[1]
    P1_2 += cos(θ) * p3[2] + (1 - cos(θ)) * kp3 * k2 + t[2]
    P1_3 += cos(θ) * p3[3] + (1 - cos(θ)) * kp3 * k3 + t[3]
  
    r2[1] = -P1_1 / P1_3
    r2[2] = -P1_2 / P1_3
    s = scaling_factor(r2, k_1, k_2)
    r2 .*= f * s
    return r2
end
  
projection!(x, c, r2) =
    projection!(x, view(c, 1:3), view(c, 4:6), c[7], c[8], c[9], r2)

function cross!(c::AbstractVector, a::AbstractVector, b::AbstractVector)
    if !(length(a) == length(b) == length(c) == 3)
      throw(DimensionMismatch("cross product is only defined for vectors of length 3"))
    end
    a1, a2, a3 = a
    b1, b2, b3 = b
    c[1] = a2 * b3 - a3 * b2
    c[2] = a3 * b1 - a1 * b3
    c[3] = a1 * b2 - a2 * b1
    c
end
  
function scaling_factor(point, k1, k2)
    sq_norm_point = dot(point, point)
    return 1 + sq_norm_point * (k1 + k2 * sq_norm_point)
end=#

sample_rate = .05
sample = sort(randperm(nls.nobs)[1:Int(ceil(sample_rate * nls.nobs))]) # random sample of sample_rate% of observations of nls
row_sample_ba = row_sample_bam(sample)

Fk = zeros(nls.nls_meta.nequ)
xk = nls.meta.x0
∇fk = similar(xk)

residual!(adnls, xk, Fk, sample)
Jk = jac_residual(adnls, xk)[row_sample_ba, :]
jtprod_residual!(adnls, xk, Fk, ∇fk)
norm(Jk'Fk - ∇fk)